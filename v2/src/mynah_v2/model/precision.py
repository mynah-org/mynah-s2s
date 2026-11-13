"""EXPERIMENTAL precision formats for the Thinker backbone linears, and a teacher-forced sensitivity probe.

Two separate questions, kept separate on purpose:
  * quality — how much does a weight format perturb the model, and WHERE (which layer group, which kind of step:
    tool-call arguments, speech text, silence)? Answered with weight *simulation* (quantize → dequantize to BF16) so
    every candidate format (INT4 g128/g64, INT8 per-channel, FP8 e4m3 per-channel, NVFP4 e2m1 + e4m3 block-16 scales)
    can be measured on the same footing, whether or not a fast kernel exists on this GPU;
  * speed — only formats with a real optimized kernel on the target device are worth serving (GB10/sm_121, torch
    2.10: int4 tinygemm yes; int8 weight-only and fp8 row-wise no, see quant.py). That is measured separately
    (quant_microbench.py) and a format is a serving candidate only if it passes both.

``SensitivityProbe`` wraps ``stt_model.forward``: the golden BF16 forward runs unchanged (the trajectory, tool calls
and audio are exactly the reference ones — teacher forcing), then for each candidate the selected linears are swapped
to the candidate's modules and the same input is run against that candidate's OWN cache (its Mamba/KV state evolves
under the candidate weights, so state-error accumulation is included). Per step and candidate: top-1 agreement and
KL(golden‖candidate) on the text, function and ASR heads. Nothing here is used by REFERENCE or by the CLIs.
"""

from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn

TARGETS = ("in_proj", "out_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ------------------------------------------------------------------------------------------------ formats (simulated)
def _int4_group(w: torch.Tensor, group: int, clip: bool = False) -> torch.Tensor:
    from mynah_v2.model.quant import _clip_range
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = _clip_range(wf) if clip else (wf.amin(-1), wf.amax(-1))
    scale = (mx - mn).clamp_min(1e-8) / 15
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15)
    return (q * scale[..., None] + mn[..., None]).reshape(n, k)


def _int8_channel(w: torch.Tensor) -> torch.Tensor:
    wf = w.float()
    s = wf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return torch.round(wf / s).clamp(-127, 127) * s


def _fp8_channel(w: torch.Tensor) -> torch.Tensor:
    wf = w.float()
    s = wf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 448
    return (wf / s).to(torch.float8_e4m3fn).float() * s


def _nvfp4(w: torch.Tensor) -> torch.Tensor:
    """NVFP4: e2m1 values, one e4m3 scale per 16 consecutive K elements, one FP32 per-tensor scale."""
    n, k = w.shape
    wf = w.float().reshape(n, k // 16, 16)
    g = wf.abs().amax().clamp_min(1e-12) / (448 * 6)
    bs = (wf.abs().amax(-1) / 6 / g).to(torch.float8_e4m3fn).float().clamp_min(2 ** -9) * g
    x = wf / bs[..., None]
    grid = E2M1.to(w.device)
    mag = grid[(x.abs()[..., None] - grid).abs().argmin(-1)]
    return (mag * x.sign() * bs[..., None]).reshape(n, k)


SIM_FORMATS = {
    "bf16": lambda w: w.float(),                      # control arm: exact copy through the same shadow machinery
    "int4g128": lambda w: _int4_group(w, 128),
    "int4g64": lambda w: _int4_group(w, 64),
    "int4g64c": lambda w: _int4_group(w, 64, clip=True),
    "int4g32": lambda w: _int4_group(w, 32),
    "int8": _int8_channel,
    "fp8": _fp8_channel,
    "nvfp4": _nvfp4,
}


def simulated_linear(lin: nn.Linear, fmt: str) -> nn.Linear:
    out = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None, device=lin.weight.device,
                    dtype=lin.weight.dtype)
    with torch.no_grad():
        w = lin.weight.data
        rows = []
        for i in range(0, w.shape[0], 4096):                      # chunked: bounded FP32 temporaries
            rows.append(SIM_FORMATS[fmt](w[i:i + 4096]).to(w.dtype))
        out.weight.copy_(torch.cat(rows))
        if lin.bias is not None:
            out.bias.copy_(lin.bias)
    out.requires_grad_(False)
    return out


def kernel_linear(lin: nn.Linear, fmt: str) -> nn.Module:
    """Real kernels (serving candidates on sm_121): weight-only Triton GEMV (w8 | f8 | w4 | nvf4, the modules the perf
    experiments swap in) or int4 tinygemm (int4g<N>[c])."""
    if fmt in ("w8", "f8", "w4", "nvf4"):
        from mynah_v2.model.wo_gemv import WOLinear
        return WOLinear(lin, fmt)
    from mynah_v2.model.quant import Int4Linear
    m = re.fullmatch(r"int4g(\d+)(c?)", fmt)
    if not m:
        raise ValueError(f"no optimized kernel for {fmt} on this device")
    return Int4Linear(lin, max_group=int(m.group(1)), clip=bool(m.group(2)))


# ------------------------------------------------------------------------------------------------ selectors
HEADS = ("lm_head", "function_head")


def thinker_linears(llm, stt=None):
    """[(layer_idx, block_type, proj_name, parent)] for every backbone linear, plus (with ``stt``) the text and
    function heads as layer -1, block type "head" (they are read every step: 2 x 1.17 GB BF16)."""
    out = []
    if stt is not None:
        for h in HEADS:
            if isinstance(getattr(stt, h, None), nn.Linear):
                out.append((-1, "head", h, stt))
    for li, layer in enumerate(llm.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        bt = getattr(layer, "block_type", "?")
        for name, mod in mixer.named_children():
            if name in TARGETS and isinstance(mod, nn.Linear):
                out.append((li, bt, name, mixer))
    return out


def select(lins, selector: str, n_layers: int):
    """Selector grammar (comma = union, leading '!' = remove):  all (backbone) | none | heads | mamba | mlp | attention
    | <block>.<proj> (head.lm_head, head.function_head) | <proj> | L<a>-<b> (layer range, inclusive) | Q<i> (depth
    quarter 0..3). Example: 'all,!mamba.in_proj', 'all,heads'."""
    def match(term, li, bt, name):
        if term == "none":
            return False
        if term == "all":
            return bt != "head"
        if term == "heads":
            return bt == "head"
        if term in ("mamba", "mlp", "attention"):
            return bt == term
        if "." in term:
            b, p = term.split(".", 1)
            return bt == b and name == p
        if term in TARGETS:
            return name == term and bt != "head"
        m = re.fullmatch(r"L(\d+)-(\d+)", term)
        if m:
            return bt != "head" and int(m.group(1)) <= li <= int(m.group(2))
        m = re.fullmatch(r"Q([0-3])", term)
        if m:
            if bt == "head":                       # heads are not in any depth quarter (was: fell through to raise)
                return False
            q = int(m.group(1))
            return q * n_layers // 4 <= li < (q + 1) * n_layers // 4
        raise ValueError(f"bad selector term {term!r}")
    chosen = set()
    for term in selector.split(","):
        neg = term.startswith("!")
        t = term[1:] if neg else term
        hit = {(li, name) for li, bt, name, _ in lins if match(t, li, bt, name)}
        chosen = chosen - hit if neg else chosen | hit
    return chosen


# ------------------------------------------------------------------------------------------------ probe
@dataclass
class Candidate:
    name: str
    fmt: str
    selector: str
    kernel: bool = False                          # real kernel modules instead of simulation
    swaps: list = field(default_factory=list)     # (mixer, proj, golden_module, candidate_module)
    caches: dict = field(default_factory=dict)    # id(golden cache) -> (golden cache, candidate cache); the golden
    #                                               object is held so its id cannot be reused while mapped
    bytes_bf16: int = 0


def _clone_cache(c):
    return copy.deepcopy(c)


def _kl(p_logits, q_logits):
    lp = torch.log_softmax(p_logits.float(), -1)
    lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def _margins(tag, logits):
    """Golden decision margin at this step: top-1 minus top-2 logit, and top-2 token (FP32, batch row 0)."""
    v, i = logits[0].float().topk(2)
    return {f"{tag}_margin": round(float(v[0] - v[1]), 4), f"{tag}_2nd": int(i[1])}


def _gap(logits, tok):
    """Candidate logit of ``tok`` minus its best other logit (positive = tok still wins)."""
    x = logits[0].float()
    t = x[tok].clone()
    x[tok] = float("-inf")
    return round(float(t - x.max()), 4)


class SensitivityProbe:
    def __init__(self, engine, specs: list[str], pad_id: int = 12):
        """specs: 'name=fmt@selector' (simulated) or 'name=fmt@selector!kernel' (real kernel)."""
        self.stt = engine.wrapper.model.stt_model
        self.llm = self.stt.llm
        self.pad_id = pad_id
        lins = thinker_linears(self.llm, self.stt)
        n_layers = len(self.llm.layers)
        built: dict = {}
        self.cands: list[Candidate] = []
        for spec in specs:
            name, rest = spec.split("=", 1)
            kernel = rest.endswith("!kernel")
            rest = rest.removesuffix("!kernel")
            fmt, selector = rest.split("@", 1)
            c = Candidate(name, fmt, selector, kernel)
            for li, proj in sorted(select(lins, selector, n_layers)):
                mixer = next(m for l2, _, p2, m in lins if l2 == li and p2 == proj)
                gold = getattr(mixer, proj)
                key = (li, proj, fmt, kernel)
                if key not in built:
                    built[key] = kernel_linear(gold, fmt) if kernel else simulated_linear(gold, fmt)
                c.swaps.append((mixer, proj, gold, built[key]))
                c.bytes_bf16 += gold.weight.numel() * gold.weight.element_size()
            self.cands.append(c)
        torch.cuda.empty_cache()
        self.lock = threading.Lock()
        self.records: list[dict] = []
        self.tag = None                       # set by the driver (item id) for per-item aggregation
        self._orig = self.stt.forward
        self.stt.forward = self._forward

    def remove(self):
        self.stt.forward = self._orig

    def reset_session(self):
        for c in self.cands:
            c.caches.clear()

    def _forward(self, input_embeds, cache=None, **kw):
        with self.lock, torch.no_grad():
            shadow_in = {}
            for c in self.cands:
                ent = c.caches.get(id(cache)) if cache is not None else None
                sc = ent[1] if ent is not None and ent[0] is cache else None
                if cache is not None and sc is None:
                    sc = _clone_cache(cache)           # first sight of this cache: candidate starts from the same state
                shadow_in[c.name] = sc
            gold = self._orig(input_embeds, cache=cache, **kw)
            g_text = gold["text_logits"][:, -1]
            g_func = gold.get("function_logits")
            g_func = g_func[:, -1] if g_func is not None else None
            g_asr = gold.get("asr_logits")
            g_asr = g_asr[:, -1] if g_asr is not None else None
            gt = int(g_text.argmax(-1)[0])
            gf = int(g_func.argmax(-1)[0]) if g_func is not None else None
            rec = {"tag": self.tag, "t": input_embeds.shape[1], "gold_text": gt, "gold_func": gf, "c": {},
                   **_margins("g_text", g_text), **(_margins("g_func", g_func) if g_func is not None else {})}
            for c in self.cands:
                for mixer, proj, _, cm in c.swaps:
                    setattr(mixer, proj, cm)
                try:
                    out = self._orig(input_embeds, cache=shadow_in[c.name], **kw)
                finally:
                    for mixer, proj, gm, _ in c.swaps:
                        setattr(mixer, proj, gm)
                if cache is not None:
                    oc = gold.get("cache")                  # explicit None checks: HF caches define __len__
                    oc = cache if oc is None else oc
                    sc_out = out.get("cache")
                    sc_out = shadow_in[c.name] if sc_out is None else sc_out
                    c.caches[id(oc)] = (oc, sc_out)
                    c.caches[id(cache)] = (cache, sc_out)
                ct = out["text_logits"][:, -1]
                r = {"text_agree": int(ct.argmax(-1)[0]) == gt, "text_kl": float(_kl(g_text, ct)[0]),
                     "text_top": int(ct.argmax(-1)[0]),
                     # candidate logit gap between the golden token and its own best other token (<0 = flipped)
                     "text_gold_gap": _gap(ct, gt)}
                if g_func is not None:
                    cf = out["function_logits"][:, -1]
                    r["func_agree"] = int(cf.argmax(-1)[0]) == gf
                    r["func_top"] = int(cf.argmax(-1)[0])
                    r["func_gold_gap"] = _gap(cf, gf)
                    r["func_kl"] = float(_kl(g_func, cf)[0])
                if g_asr is not None:
                    r["asr_agree"] = int(out["asr_logits"][:, -1].argmax(-1)[0]) == int(g_asr.argmax(-1)[0])
                rec["c"][c.name] = r
            self.records.append(rec)
            return gold
