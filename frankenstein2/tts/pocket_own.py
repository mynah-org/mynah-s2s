# SPDX-License-Identifier: MIT
# Portions ported from Kyutai's pocket-tts 4.0.1 (MIT licence, Copyright (c) Kyutai): the FlowLM / LSD head / Mimi
# (SEANet + transformer) computations, the text preparation and the voice-prompt steps. Rewritten as plain torch
# functions over the checkpoint tensors (no pocket-tts classes at runtime); the package is used only as a test
# reference (lab/pocket_check.py). Weights: Kyutai, CC-BY-4.0.
"""Our own Pocket TTS runtime (24-layer FlowLM + LSD flow head + Mimi codec), torch only.

  model = PocketOwn(weights_safetensors, tokenizer_json, lang="it", device="cuda")
  voice = model.voice_state(wav_24k)                 # zero-shot clone: Mimi encoder + speaker projection + prompt
  for pcm in model.stream(voice, "Buongiorno.", seed=0): ...   # float32 24 kHz chunks (80 ms per latent frame)

Generation (as the reference): the text is prepared (character replacements, capital, terminal punctuation) and split
into sentence chunks of <= 50 tokens; per chunk the voice-prompted KV cache is copied, the text tokens are fed, then
one latent per step: FlowLM transformer (RoPE, causal, KV cache) -> out_norm -> EOS head + LSD flow head (1 step,
noise N(0, temp) with temp 0.3) -> next latent; EOS after >= 6 frames, then a few frames more. Latents are
de-normalised (emb_std, emb_mean) and decoded by Mimi in streaming (quantizer 1x1 conv, transposed-conv upsample x16,
2-layer transformer with context 250, SEANet decoder 4-5-6 upsampling), the first 5 ms faded in.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..gpu import host

# Mimi latents of the voice files seen, keyed by (file, mtime, size, Mimi weights): Mimi is identical in every language,
# so a voice is encoded once and each language only runs its own (cheap) FlowLM prompt over the latents.
_VOICE_LATENTS: dict = {}
_VOICE_LOCK = threading.Lock()

REPLACE = {'"': '', '“': '', '”': '', '„': '', '«': '', '»': '', '’': "'", '‘': "'", '(': '', ')': '', '[': '',
           ']': '', '–': ','}
LANG_OPTS = {"it": {"remove_semicolons": False}, "fr": {"remove_semicolons": True}, "en": {"remove_semicolons": False}}
SR = 24000
FRAME_RATE = 12.5
MAX_TOKENS = 50
TEMP, EOS_THR, MIN_FRAMES_BEFORE_EOS = 0.3, -4.0, 6


# ---------------------------------------------------------------------------------------------------------- text
def prepare_text(text: str, remove_semicolons: bool) -> tuple[str, int]:
    text = text.strip()
    text = " ".join(text.translate(str.maketrans(REPLACE)).split())
    text = re.sub(r"([.!?…])\s*[,;:]", r"\1", text)
    if text == "":
        raise ValueError("empty text")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    after = 3 if len(text.split()) <= 4 else 1
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    core = text.rstrip("\"'”’)]» ")
    closers = text[len(core):].strip()
    if core and core[-1] not in ".!?…":
        if core[-1] in ",;:-–—":
            text = core.rstrip(",;:-–— ") + "." + closers
        else:
            text = text + "."
    return text, after


def _opts(lang: str) -> dict:
    """Text options of a language (langs.py: built-in or a package's language.json)."""
    from .. import langs
    if lang in LANG_OPTS:
        return LANG_OPTS[lang]
    return {"remove_semicolons": langs.get(lang).remove_semicolons}


class _Tok:
    """tokenizer.json (HF tokenizers) or tokenizer.model (SentencePiece, as some community languages ship it)."""

    def __init__(self, path):
        if str(path).endswith(".model"):
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(path))
            self.enc, self.dec = (lambda s: sp.encode(s)), sp.decode
        else:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(str(path))
            self.enc, self.dec = (lambda s: t.encode(s).ids), t.decode

    def __call__(self, s):
        return self.enc(s)

    def decode(self, ids):
        return self.dec(ids)


def _bounds(toks, bset, tok=None, skip_decimal=False):
    idx, prev = [0], False
    for i, t in enumerate(toks):
        if t in bset:
            prev = True
        else:
            if prev:
                if skip_decimal and tok is not None:
                    pre, suf = tok.decode(toks[:i]), tok.decode(toks[i:])
                    if len(pre) >= 2 and pre[-1] == "." and pre[-2].isdigit() and suf and suf[0].isdigit():
                        prev = False
                        continue
                idx.append(i)
            prev = False
    idx.append(len(toks))
    return idx


def split_sentences(tok: _Tok, text: str, remove_semicolons: bool, max_tokens: int = MAX_TOKENS) -> list:
    text, _ = prepare_text(text, remove_semicolons)
    text = text.strip()
    toks = tok(text)
    _, *eos = tok(".!...?")
    b = _bounds(toks, set(eos), tok, True)
    segs = [(b[i + 1] - b[i], tok.decode(toks[b[i]:b[i + 1]])) for i in range(len(b) - 1)]
    _, *fb = tok(",;:")
    refined = []
    for n, s in segs:
        if n <= max_tokens:
            refined.append((n, s))
            continue
        st = tok(s.strip())
        sb = _bounds(st, set(fb))
        sub = [(sb[i + 1] - sb[i], tok.decode(st[sb[i]:sb[i + 1]])) for i in range(len(sb) - 1)]
        refined.extend(sub if len(sub) > 1 else [(n, s)])
    chunks, cur, cn = [], "", 0
    for n, s in refined:
        if cur == "":
            cur, cn = s, n
        elif cn + n > max_tokens:
            chunks.append(cur.strip())
            cur, cn = s, n
        else:
            cur, cn = cur + " " + s, cn + n
    if cur:
        chunks.append(cur.strip())
    return chunks


# ---------------------------------------------------------------------------------------------------------- audio
def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resampling with a Kaiser(5.0)-windowed sinc FIR of length 2*10*max(up, down)+1 (the scipy
    defaults), zero-phase alignment as scipy.signal.resample_poly."""
    g = math.gcd(up, down)
    up, down = up // g, down // g
    if up == down == 1:
        return x.astype(np.float64)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    n = 2 * half_len + 1
    fc = 1.0 / max_rate
    m = np.arange(n) - (n - 1) / 2
    h = fc * np.sinc(fc * m) * np.kaiser(n, 5.0)
    h = h / h.sum() * up
    n_in = len(x)
    n_out = n_in * up // down + bool(n_in * up % down)
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while (n_in * up + (n + n_pre_pad + n_post_pad) - 1) // down < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate([np.zeros(n_pre_pad), h, np.zeros(n_post_pad)])
    xu = np.zeros(n_in * up)
    xu[::up] = x
    y = np.convolve(xu, h)[::down]
    return y[n_pre_remove:n_pre_remove + n_out]


def end_on_pause(wav: torch.Tensor, pause_sec=0.08, fade_sec=0.02, floor_db=35.0) -> torch.Tensor:
    frame = int(0.02 * SR)
    n = wav.shape[-1] // frame
    if n == 0:
        return wav
    rms = wav[..., :n * frame].reshape(-1, n, frame).square().mean(dim=(0, 2)).sqrt()
    db = 20 * torch.log10(rms + 1e-12)
    loud = torch.nonzero(db > db.max() - floor_db).flatten()
    end = (int(loud[-1]) + 1) * frame
    wav = wav[..., :end].clone()
    fade = min(int(fade_sec * SR), end)
    wav[..., end - fade:] *= torch.linspace(1, 0, fade, device=wav.device, dtype=wav.dtype)
    return torch.cat([wav, wav.new_zeros(*wav.shape[:-1], int(pause_sec * SR))], -1)


# ---------------------------------------------------------------------------------------------------------- model
def _pocket_swap():
    """pocket_swap's fused kernel + quantiser (the repo's pocket_swap/ package; torch + optional Triton only)."""
    try:
        from pocket_swap import fused, quant
    except ImportError:          # the repo root's pocket_swap/ dir shadows the package as a namespace: use the inner one
        for k in [k for k in sys.modules if k == "pocket_swap" or k.startswith("pocket_swap.")]:
            del sys.modules[k]
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pocket_swap"))
        from pocket_swap import fused, quant
    return fused, quant


def _rope(q, k, offset, max_period=10000.0):
    """q, k (B, T, H, D); offset: int or (B,) tensor of start positions."""
    B, T, H, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = torch.arange(T, device=q.device, dtype=torch.float32)
    if isinstance(offset, torch.Tensor) and offset.dim() == 1:
        ts = ts[None, :] + offset[:, None].float()                       # (B, T)
        ts = ts.view(B, T, 1, 1)
    else:
        ts = (ts + offset).view(1, T, 1, 1)
    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, k.shape[2], D // 2, 2)
    qr, qi, kr, ki = q[..., 0].float(), q[..., 1].float(), k[..., 0].float(), k[..., 1].float()
    rr, ri = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([(qr * rr - qi * ri).to(q.dtype), (qr * ri + qi * rr).to(q.dtype)], -1)
    ko = torch.stack([(kr * rr - ki * ri).to(k.dtype), (kr * ri + ki * rr).to(k.dtype)], -1)
    return qo.view(B, T, H, D), ko.view(B, T, k.shape[2], D)


class _KV:
    """A per-session linear KV cache for one transformer (all layers): (L, 2, Tmax, H, D) + write offset."""

    def __init__(self, L, Tmax, H, D, device, dtype):
        self.c = torch.full((L, 2, Tmax, H, D), float("nan"), device=device, dtype=dtype)
        self.off = 0

    def copy(self):
        o = _KV.__new__(_KV)
        o.c, o.off = self.c.clone(), self.off
        return o

    def grow(self, Tmax):
        if self.c.shape[2] < Tmax:
            c = torch.full((*self.c.shape[:2], Tmax, *self.c.shape[3:]), float("nan"), device=self.c.device,
                           dtype=self.c.dtype)
            c[:, :, :self.c.shape[2]] = self.c
            self.c = c


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class BaseBackbone:
    """The English 24-layer base's FlowLM transformer weights, loaded once and shared by every delta language
    (``precision``: bf16 = exact, or int8 / int4 quantised once to save memory and bandwidth)."""

    def __init__(self, checkpoint: str, device="cuda", precision: str = "bf16", group: int = 128):
        from safetensors import safe_open
        _, quant = _pocket_swap()
        from pocket_swap.package import backbone_fingerprint_file
        self.fingerprint = backbone_fingerprint_file(checkpoint)
        self.source_sha = None
        self.precision = precision
        self.terms, self.cols = {}, {}
        dev = torch.device(device)
        with safe_open(str(checkpoint), framework="pt", device=str(dev)) as f:
            for k in f.keys():
                if k.startswith("flow_lm.transformer.") and len(f.get_slice(k).get_shape()) == 2:
                    w = f.get_tensor(k).to(torch.bfloat16)
                    self.cols[k] = w.shape[1]
                    if precision == "bf16":
                        self.terms[k] = ("bf16", w, None)
                    else:
                        q, sc = quant.quantize_any(w.float(), precision, group)
                        self.terms[k] = (precision, q, sc)

    @classmethod
    def from_package(cls, pkg, device="cuda", precision: str = "bf16", group: int = 128) -> "BaseBackbone":
        """From the base language's ``exact`` package (``<root>/en/<family>/base``): its backbone stored verbatim."""
        _, quant = _pocket_swap()
        if pkg.manifest["precision"] != "exact":
            raise ValueError(f"the base backbone comes from an exact package, not {pkg.manifest['precision']!r}")
        self = cls.__new__(cls)
        self.fingerprint = pkg.manifest["base_backbone_bf16_sha256"]
        self.source_sha = (pkg.manifest.get("target") or {}).get("sha256")
        self.precision, self.terms, self.cols = precision, {}, {}
        dev = torch.device(device)
        for k in pkg.delta_keys():
            w = pkg.delta[k].to(dev, non_blocking=True).to(torch.bfloat16)
            self.cols[k] = w.shape[1]
            if precision == "bf16":
                self.terms[k] = ("bf16", w, None)
            else:
                q, sc = quant.quantize_any(w.float(), precision, group)
                self.terms[k] = (precision, q, sc)
        return self

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for term in self.terms.values() for t in term[1:] if t is not None)


class PocketOwn:
    def __init__(self, weights: str, tokenizer: str, lang: str = "it", device="cuda", W: dict | None = None,
                 dtype=torch.float32, wq: str = "", share_mimi: "PocketOwn | None" = None):
        """dtype: the FlowLM transformer's weights, activations and KV (bf16 halves its memory traffic; the flow net,
        the EOS head and Mimi stay fp32).
        wq: weight-only quantisation of the FlowLM transformer linears, pocket_swap's mode (b) format and fused
        kernel: "w4" (int4, group 128), "w8" (int8), "wfp8" (fp8 e4m3, sm_89+); "" = plain dtype weights. Implies
        bf16 activations.
        share_mimi: another language's model whose Mimi (codec + voice encoder, identical in every language)
        tensors are reused instead of loading a second copy."""
        from safetensors import safe_open
        self.dev = torch.device(device)
        if wq:
            dtype = torch.bfloat16
        self.tdtype = dtype
        self.lang = lang
        self.opts = _opts(lang)
        if W is None:
            W = {}
            with safe_open(str(weights), framework="pt", device=str(self.dev)) as f:
                for k in f.keys():
                    if share_mimi is not None and k.startswith("mimi.") and k in share_mimi.W:
                        W[k] = share_mimi.W[k]
                    else:
                        W[k] = f.get_tensor(k).float()
        if dtype != torch.float32:
            W = {k: (v.to(dtype) if k.startswith("flow_lm.transformer.") and v.is_floating_point() else v)
                 for k, v in W.items()}
        self.Q: dict = {}                     # key -> (precision, codes, scales, in_features): quantised linears
        self.QB: dict = {}                    # key -> shared base term (delta packages only)
        self.base = None
        self.wq = wq
        if wq:
            _, quant = _pocket_swap()
            prec = {"w4": "int4", "w8": "int8", "wfp8": "fp8"}[wq]
            for k in [k for k in W if k.startswith("flow_lm.transformer.") and k.endswith("_proj.weight")
                      or k.startswith("flow_lm.transformer.") and k.endswith(("linear1.weight", "linear2.weight"))]:
                w = W.pop(k)
                q, sc = quant.quantize_any(w.float(), prec, 128)
                self.Q[k] = (prec, q, sc, w.shape[1])
                del w
        self.W = W
        self.tok = _Tok(tokenizer)
        self.n_layers = sum(1 for k in W if k.startswith("flow_lm.transformer.layers.") and k.endswith("norm1.weight"))
        self.d = W["flow_lm.input_linear.weight"].shape[0]
        self.nh = 16
        self.ldim = W["flow_lm.bos_emb"].shape[0]
        self.n_res = sum(1 for k in W if k.startswith("flow_lm.flow_net.res_blocks.") and k.endswith("in_ln.weight"))
        self.mimi_layers = sum(1 for k in W if k.startswith("mimi.decoder_transformer.transformer.layers.")
                               and k.endswith("norm1.weight"))

    @classmethod
    def from_package(cls, pkg, lang: str, mimi, device="cuda", stream=None, base: "BaseBackbone | None" = None
                     ) -> "PocketOwn":
        """A language from a pocket_swap package. Weight-only (``w4`` / ``w8`` / ``wfp8``): the package's quantised
        backbone codes become the quantised linears as they are. Delta (``int4`` / ``int8`` / ``bf16``, built against
        the English 24-layer base): each linear keeps only the language's delta and computes x @ (W_base + delta)
        with the shared ``base`` backbone in pocket_swap's fused kernel (the full weight is never written); ``base``
        = the base language itself. The other FlowLM tensors and the tokenizer are used verbatim; Mimi (identical in
        every language) comes from ``mimi``: a loaded PocketOwn to share, or a checkpoint path (``mimi.*`` only).
        ``pkg``: a pocket_swap Package (host tensors, possibly pinned) or a package directory."""
        _, quant = _pocket_swap()
        if not hasattr(pkg, "manifest"):
            from pocket_swap.package import load_package
            pkg = load_package(pkg)
        prec = pkg.manifest["precision"]
        wp = {"w4": "int4", "w8": "int8", "wfp8": "fp8", "int4": "int4", "int8": "int8", "bf16": "bf16",
              "base": None, "exact": "bf16"}.get(prec, "?")
        if wp == "?":
            raise ValueError(f"{pkg.lang}: package precision {prec!r} not loadable (w4/w8/wfp8, int4/int8/bf16 "
                             "deltas, base)")
        is_delta = prec in ("int4", "int8", "bf16", "base")
        if is_delta and base is None:
            raise ValueError(f"{pkg.lang}: a {prec} package needs the base backbone (BaseBackbone)")
        if is_delta and pkg.manifest.get("base_backbone_bf16_sha256") not in (None, base.fingerprint):
            raise ValueError(f"{pkg.lang}: built against another base checkpoint than the one loaded")
        if int(pkg.manifest.get("group", 128)) != 128:
            raise ValueError(f"{pkg.lang}: group {pkg.manifest.get('group')} (the fused kernel uses 128)")
        dev = torch.device(device)
        cm = torch.cuda.stream(stream) if (stream is not None and dev.type == "cuda") else _nullctx()
        self = cls.__new__(cls)
        with cm:
            W = {}
            if isinstance(mimi, PocketOwn):
                W.update({k: v for k, v in mimi.W.items() if k.startswith("mimi.")})
            else:
                from safetensors import safe_open
                with safe_open(str(mimi), framework="pt", device=str(dev)) as f:
                    for k in f.keys():
                        if k.startswith("mimi."):
                            W[k] = f.get_tensor(k).float()
            for k, v in pkg.verbatim.items():
                v = v.to(dev, non_blocking=True)
                W[k] = (v.to(torch.bfloat16) if k.startswith("flow_lm.transformer.") else v.float()) \
                    if v.is_floating_point() else v
            Q, QB = {}, {}
            if prec == "exact" and base is not None and \
                    (pkg.manifest.get("target") or {}).get("sha256") in (base.source_sha, None) and \
                    pkg.manifest.get("base_backbone_bf16_sha256") == base.fingerprint and \
                    getattr(base, "source_sha", None) is not None:
                prec = "base"                          # the base language itself: reuse the shared backbone
            keys = list(base.terms) if prec == "base" else pkg.delta_keys()
            for k in keys:
                lin = k.endswith(("_proj.weight", "linear1.weight", "linear2.weight"))
                if prec == "base":                     # the base language: the shared base backbone as is
                    bt = base.terms[k]
                    if bt[0] == "bf16":
                        W[k] = bt[1]
                    else:
                        Q[k] = (bt[0], bt[1], bt[2], base.cols[k])
                    continue
                if prec == "exact":                    # the language's own weights, verbatim bf16 (no base needed)
                    W[k] = pkg.delta[k].to(dev, non_blocking=True).to(torch.bfloat16)
                    continue
                if wp == "bf16":
                    q, sc = pkg.delta[k].to(dev, non_blocking=True).to(torch.bfloat16), None
                else:
                    q = pkg.delta[k + ".q"].to(dev, non_blocking=True)
                    sc = pkg.delta[k + ".s"].to(dev, non_blocking=True)
                cols = q.shape[1] * (2 if wp == "int4" else 1)
                if lin:
                    Q[k] = (wp, q, sc, cols)
                    if is_delta:
                        QB[k] = base.terms[k]          # shared: x @ (W_base + delta) in the fused kernel
                else:                                  # any other 2-D backbone tensor: plain bf16
                    w = quant.dequant_any(q, sc, wp, cols, torch.float32)
                    if is_delta:
                        bt = base.terms[k]
                        w = w + quant.dequant_any(bt[1], bt[2], bt[0], cols, torch.float32)
                    W[k] = w.to(torch.bfloat16)
            if stream is not None and dev.type == "cuda":
                stream.synchronize()                   # everything resident before another stream reads it
        self.dev, self.tdtype, self.lang, self.wq = dev, torch.bfloat16, lang, prec
        self.opts = _opts(lang)
        self.Q, self.QB, self.W = Q, QB, W
        self.base = base if is_delta else None          # the shared backbone (QB empty: this is the base language)
        self.tok = _Tok(pkg.tokenizer_path)
        self.n_layers = sum(1 for k in W if k.startswith("flow_lm.transformer.layers.") and k.endswith("norm1.weight"))
        self.d = W["flow_lm.input_linear.weight"].shape[0]
        self.nh = 16
        self.ldim = W["flow_lm.bos_emb"].shape[0]
        self.n_res = sum(1 for k in W if k.startswith("flow_lm.flow_net.res_blocks.") and k.endswith("in_ln.weight"))
        self.mimi_layers = sum(1 for k in W if k.startswith("mimi.decoder_transformer.transformer.layers.")
                               and k.endswith("norm1.weight"))
        return self

    # ------------------------------------------------------------------ FlowLM
    def _lin(self, x, key):
        """F.linear on a plain weight, or on a quantised one: fused dequant GEMV for <= 16 rows (pocket_swap's
        kernel), a bf16 dequantised weight otherwise (prefills)."""
        t = self.Q.get(key)
        if t is None:
            return F.linear(x, self.W[key])
        prec, q, sc, cols = t
        b = self.QB.get(key) if self.QB else None          # shared base term of a delta package
        fused, quant = _pocket_swap()
        x = x.to(torch.bfloat16)
        if x.numel() // x.shape[-1] <= 16 and fused.available(x, (prec,) + ((b[0],) if b else ())):
            return fused.fused_linear2(x, q.shape[0], (prec, q, sc), b, 128)
        if b is None:
            return F.linear(x, quant.dequant_any(q, sc, prec, cols, torch.bfloat16))
        w = quant.dequant_any(q, sc, prec, cols, torch.float32) + quant.dequant_any(b[1], b[2], b[0], cols,
                                                                                     torch.float32)
        return F.linear(x, w.to(torch.bfloat16))

    def nbytes(self) -> dict:
        own = sum(v.numel() * v.element_size() for k, v in self.W.items() if not k.startswith("mimi."))
        q = sum(t.numel() * t.element_size() for e in self.Q.values() for t in e[1:3])
        mimi = sum(v.numel() * v.element_size() for k, v in self.W.items() if k.startswith("mimi."))
        return {"flow_plain": own, "flow_quant": q, "mimi": mimi}

    def _ln(self, x, p, eps=1e-5):
        return F.layer_norm(x, (x.shape[-1],), self.W[p + ".weight"], self.W[p + ".bias"], eps)

    def _attn(self, x, p, nh, kv: _KV | None, li: int, context=None):
        """Causal MHA with RoPE. kv=None: stateless over the whole sequence."""
        W = self.W
        B, T, d = x.shape
        hd = d // nh
        proj = self._lin(x, p + ".in_proj.weight")
        q = proj[..., :d].view(B, T, nh, hd)
        k, v = proj[..., d:].view(B, T, 2, nh, hd).unbind(2)
        off = 0 if kv is None else kv.off
        q, k = _rope(q, k, off)
        if kv is None:
            K, V = k.transpose(1, 2), v.transpose(1, 2)
            pos_k = torch.arange(T, device=x.device)
            pos_q = pos_k
        else:
            kv.c[li, 0, off:off + T] = k[0]
            kv.c[li, 1, off:off + T] = v[0]
            K = kv.c[li, 0, :off + T][None].transpose(1, 2)
            V = kv.c[li, 1, :off + T][None].transpose(1, 2)
            pos_k = torch.arange(off + T, device=x.device)
            pos_q = off + torch.arange(T, device=x.device)
        delta = pos_q[:, None] - pos_k[None, :]
        mask = (delta >= 0)
        if context is not None:
            mask = mask & (delta < context)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), K, V, mask[None, None])
        return self._lin(o.transpose(1, 2).reshape(B, T, d), p + ".out_proj.weight")

    def _flow_transformer(self, x, kv: _KV):
        W = self.W
        x = x.to(self.tdtype)
        for li in range(self.n_layers):
            p = f"flow_lm.transformer.layers.{li}"
            x = x + self._attn(self._ln(x, p + ".norm1"), p + ".self_attn", self.nh, kv, li)
            h = self._ln(x, p + ".norm2")
            x = x + self._lin(F.gelu(self._lin(h, p + ".linear1.weight"), approximate="tanh"), p + ".linear2.weight")
        return x

    def _noise(self, B, gen=None):
        noise = torch.empty(B, self.ldim, device=self.dev)
        if gen is None:
            torch.nn.init.normal_(noise, mean=0.0, std=TEMP ** 0.5)          # the reference's draw (global RNG)
        else:
            noise.normal_(0.0, TEMP ** 0.5, generator=gen)                   # a per-row generator (batching)
        return noise

    def _flow_step(self, kv: _KV, text_ids=None, latents=None, audio_cond=None, gen=None):
        """One FlowLM call (text tokens, audio conditioning and / or latents) -> (next latent (1, ldim), eos)."""
        W = self.W
        parts = []
        if text_ids is not None and len(text_ids):
            parts.append(W["flow_lm.conditioner.embed.weight"][torch.tensor([text_ids], device=self.dev)])
        if audio_cond is not None:
            parts.append(audio_cond)
        n_lat = 0
        if latents is not None:
            seq = torch.where(torch.isnan(latents), W["flow_lm.bos_emb"], latents)
            parts.append(F.linear(seq, W["flow_lm.input_linear.weight"]))
            n_lat = latents.shape[1]
        x = torch.cat(parts, 1)
        T = x.shape[1]
        kv.grow(kv.off + T)
        y = self._flow_transformer(x, kv)
        kv.off += T
        y = self._ln(y.float(), "flow_lm.out_norm")
        y = y[:, -max(n_lat, 1):] if n_lat else y[:, -1:]
        y = y.float()[:, -1]
        eos = F.linear(y, W["flow_lm.out_eos.weight"], W["flow_lm.out_eos.bias"]) > EOS_THR
        noise = self._noise(y.shape[0], gen)
        lat = noise + self._flow_net(y, noise)                            # LSD, 1 step: s = 0, t = 1
        return lat, eos

    def _flow_net(self, c, x):
        W = self.W
        p = "flow_lm.flow_net"
        x = F.linear(x, W[p + ".input_proj.weight"], W[p + ".input_proj.bias"])
        y = F.linear(c, W[p + ".cond_embed.weight"], W[p + ".cond_embed.bias"])
        temb = 0
        for i, tval in enumerate((0.0, 1.0)):
            q = f"{p}.time_embed.{i}"
            t = torch.full((c.shape[0], 1), tval, device=self.dev)
            args = t * W[q + ".freqs"]
            e = torch.cat([torch.cos(args), torch.sin(args)], -1)
            e = F.linear(F.silu(F.linear(e, W[q + ".mlp.0.weight"], W[q + ".mlp.0.bias"])),
                         W[q + ".mlp.2.weight"], W[q + ".mlp.2.bias"])
            var = 1e-5 + e.var(dim=-1, keepdim=True)
            e = e * (W[q + ".mlp.3.alpha"] * torch.rsqrt(var))
            temb = temb + e
        y = y + temb / 2
        for i in range(self.n_res):
            q = f"{p}.res_blocks.{i}"
            mod = F.linear(F.silu(y), W[q + ".adaLN_modulation.1.weight"], W[q + ".adaLN_modulation.1.bias"])
            sh, sc, g = mod.chunk(3, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            h = (x - mean) / torch.sqrt(var + 1e-6) * W[q + ".in_ln.weight"] + W[q + ".in_ln.bias"]
            h = h * (1 + sc) + sh
            h = F.linear(F.silu(F.linear(h, W[q + ".mlp.0.weight"], W[q + ".mlp.0.bias"])),
                         W[q + ".mlp.2.weight"], W[q + ".mlp.2.bias"])
            x = x + g * h
        q = f"{p}.final_layer"
        mod = F.linear(F.silu(y), W[q + ".adaLN_modulation.1.weight"], W[q + ".adaLN_modulation.1.bias"])
        sh, sc = mod.chunk(2, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        x = x * (1 + sc) + sh
        return F.linear(x, W[q + ".linear.weight"], W[q + ".linear.bias"])

    # ------------------------------------------------------------------ Mimi
    def _mimi_transformer(self, x, p, kv: _KV | None):
        """ProjectedTransformer with identity projections (d = 512), layer scale, context 250."""
        W = self.W
        x = x.transpose(1, 2)
        L = sum(1 for k in W if k.startswith(p + ".transformer.layers.") and k.endswith("norm1.weight"))
        for li in range(L):
            q = f"{p}.transformer.layers.{li}"
            a = self._attn(self._ln(x, q + ".norm1"), q + ".self_attn", 8, kv, li, context=250)
            x = x + W[q + ".layer_scale_1.scale"] * a
            h = self._ln(x, q + ".norm2")
            u = F.linear(F.gelu(F.linear(h, W[q + ".linear1.weight"]), approximate="tanh"), W[q + ".linear2.weight"])
            x = x + W[q + ".layer_scale_2.scale"] * u
        if kv is not None:
            kv.off += x.shape[1]
        return x.transpose(1, 2)

    def _conv(self, x, key, state, stride=1, dilation=1, groups=1, bias=True):
        """StreamingConv1d (constant padding): left context kept in ``state[key]`` (None: zeros = offline)."""
        w = self.W[key + ".weight"]
        k = (w.shape[-1] - 1) * dilation + 1
        tp = k - stride
        if tp:
            prev = state.get(key) if state is not None else None
            if prev is None:
                prev = x.new_zeros(x.shape[0], x.shape[1], tp)
            x = torch.cat([prev, x], -1)
            if state is not None:
                state[key] = x[..., -tp:].clone()
        return F.conv1d(x, w, self.W.get(key + ".bias") if bias else None, stride=stride, dilation=dilation,
                         groups=groups)

    def _convtr(self, x, key, state, stride, groups=1, bias=True):
        w = self.W[key + ".weight"]
        b = self.W.get(key + ".bias") if bias else None
        y = F.conv_transpose1d(x, w, b, stride=stride, groups=groups)
        pt = w.shape[-1] - stride
        if pt > 0:
            part = state.get(key)
            if part is None:
                part = y.new_zeros(y.shape[0], y.shape[1], pt)
            y[..., :pt] += part
            tail = y[..., -pt:]
            if b is not None:
                tail = tail - b[:, None]
            state[key] = tail.clone()
            y = y[..., :-pt]
        return y

    def _seanet_decoder(self, z, st):
        p = "mimi.decoder.model"
        x = self._conv(z, f"{p}.0.conv", st)
        i, mult = 1, 8
        for ratio in (6, 5, 4):
            x = self._convtr(F.elu(x), f"{p}.{i + 1}.convtr", st, stride=ratio)
            i += 2
            r = x
            h = self._conv(F.elu(r), f"{p}.{i}.block.1.conv", st, dilation=1)
            h = self._conv(F.elu(h), f"{p}.{i}.block.3.conv", st)
            x = r + h
            i += 1
        return self._conv(F.elu(x), f"{p}.{i + 1}.conv", st)

    def _seanet_encoder(self, x):
        p = "mimi.encoder.model"
        x = self._conv(x, f"{p}.0.conv", None)
        i = 1
        for ratio in (4, 5, 6):
            r = x
            h = self._conv(F.elu(r), f"{p}.{i}.block.1.conv", None)
            h = self._conv(F.elu(h), f"{p}.{i}.block.3.conv", None)
            x = r + h
            x = self._conv(F.elu(x), f"{p}.{i + 2}.conv", None, stride=ratio)
            i += 3
        return self._conv(F.elu(x), f"{p}.{i + 1}.conv", None)

    def mimi_decode(self, latent, st: dict):
        """latent (1, T, 32) de-normalised -> audio (1, 1, T*1920), streaming state in ``st``."""
        x = F.conv1d(latent.transpose(1, 2), self.W["mimi.quantizer.output_proj.weight"])
        x = self._convtr(x, "mimi.upsample.convtr.convtr", st, stride=16, groups=x.shape[1], bias=False)
        if "kv" not in st:
            st["kv"] = _KV(self.mimi_layers, 64, 8, 64, self.dev, torch.float32)
        st["kv"].grow(st["kv"].off + x.shape[-1])
        x = self._mimi_transformer(x, "mimi.decoder_transformer", st["kv"])
        return self._seanet_decoder(x, st)

    @torch.no_grad()
    def mimi_encode(self, wav):
        """wav (1, 1, N) 24 kHz -> latents (1, T, 32)."""
        frame = int(SR / FRAME_RATE)
        n = wav.shape[-1]
        extra = (math.ceil(n / frame) * frame) - n
        x = F.pad(wav, (0, extra))
        x = self._seanet_encoder(x)
        x = self._mimi_transformer(x, "mimi.encoder_transformer", None)
        x = self._downsample(x)
        return x.transpose(1, 2)

    def _downsample(self, x):
        """ConvDownsample1d (replicate padding on the first chunk, offline)."""
        w = self.W["mimi.downsample.conv.conv.weight"]
        tp = w.shape[-1] - 16
        x = torch.cat([x[..., :1].expand(-1, -1, tp), x], -1)
        return F.conv1d(x, w, stride=16)

    # ------------------------------------------------------------------ voice + generation
    @torch.no_grad()
    def voice_state(self, wav_24k: torch.Tensor, lat: torch.Tensor | None = None) -> _KV:
        """Clone a voice from a 24 kHz mono prompt (1, N): the prompted FlowLM KV cache. ``lat``: its Mimi latents,
        if already known (Mimi is the same in every language, so a voice is encoded once for all of them)."""
        if lat is None:
            wav = end_on_pause(wav_24k.to(self.dev).float())
            lat = self.mimi_encode(wav[None])
        cond = F.linear(lat.float(), self.W["flow_lm.speaker_proj_weight"])
        if "flow_lm.bos_before_voice" in self.W:
            cond = torch.cat([self.W["flow_lm.bos_before_voice"], cond], 1)
        kv = _KV(self.n_layers, cond.shape[1], self.nh, self.d // self.nh, self.dev, self.tdtype)
        self._flow_step(kv, audio_cond=cond)
        return kv

    def voice_from_file(self, path: str) -> _KV:
        """Clone a voice from a WAV file (16-bit PCM, any rate, mono or first channel)."""
        import wave
        with wave.open(str(path)) as w:
            sr, ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        pcm = pcm.reshape(-1, ch)[:, 0]
        key = (str(Path(path).resolve()), os.path.getmtime(path), os.path.getsize(path),
               self.W["mimi.encoder.model.0.conv.weight"].data_ptr())     # same Mimi tensors = same latents
        with _VOICE_LOCK:
            lat = _VOICE_LATENTS.get(key)
        if lat is not None:
            self.voice_timing = {"cached": True}
            return self.voice_state(None, lat=lat)
        t0 = time.monotonic()
        if sr != SR:
            pcm = resample_poly(pcm, SR, sr).astype(np.float32)
        wav = end_on_pause(torch.from_numpy(np.ascontiguousarray(pcm))[None].to(self.dev).float())
        lat = self.mimi_encode(wav[None])
        if self.dev.type == "cuda":
            torch.cuda.current_stream(self.dev).synchronize()
        self.voice_timing = {"cached": False, "encode_ms": round((time.monotonic() - t0) * 1000, 1),
                             "seconds": round(pcm.shape[0] / SR, 1)}
        with _VOICE_LOCK:
            while len(_VOICE_LATENTS) >= 16:
                _VOICE_LATENTS.pop(next(iter(_VOICE_LATENTS)))
            _VOICE_LATENTS[key] = lat
        return self.voice_state(None, lat=lat)

    def max_gen_len(self, n_tokens: int) -> int:
        return math.ceil((n_tokens / 3.0 + 2.0) * FRAME_RATE)

    @torch.no_grad()
    def chunks(self, text: str, frames_after_eos: int | None = None) -> list:
        """The text as the generator takes it: [(prepared chunk, frames after EOS)]."""
        out = []
        for chunk in split_sentences(self.tok, text, self.opts["remove_semicolons"]):
            prepared, guess = prepare_text(chunk, self.opts["remove_semicolons"])
            out.append((prepared, frames_after_eos if frames_after_eos is not None else guess + 2))
        return out

    def stream(self, voice: _KV, text: str, stop=None, frames_after_eos: int | None = None, gen=None):
        """Yield float32 audio chunks (24 kHz) for ``text``, one decoded latent frame at a time."""
        for prepared, after in self.chunks(text, frames_after_eos):
            if stop is not None and stop.is_set():
                return
            yield from self._stream_chunk(voice, prepared, after, stop, gen)

    def _stream_chunk(self, voice, text, after, stop, gen=None):
        ids = self.tok(text)
        kv = voice.copy()
        max_len = self.max_gen_len(len(ids))
        kv.grow(kv.off + len(ids) + max_len)
        self._flow_step(kv, text_ids=ids, gen=gen)
        x = torch.full((1, 1, self.ldim), float("nan"), device=self.dev)
        st: dict = {}
        fade = torch.linspace(0, 1, SR // 200, device=self.dev)
        eos_step = None
        mean, std = self.W["flow_lm.emb_mean"], self.W["flow_lm.emb_std"]
        for step in range(max_len):
            if stop is not None and stop.is_set():
                return
            lat, eos = self._flow_step(kv, latents=x, gen=gen)
            if bool(eos.item()) and eos_step is None and step >= MIN_FRAMES_BEFORE_EOS:
                eos_step = step
            if eos_step is not None and step >= eos_step + after:
                return
            x = lat[:, None]
            audio = self.mimi_decode(x * std + mean, st)
            if fade is not None:
                n = min(fade.numel(), audio.shape[-1])
                audio[..., :n] *= fade[:n]
                fade = None
            yield audio[0, 0]


# ---------------------------------------------------------------------------------------------------------- batching
FUSED_ATTN = True       # bf16 batched step: in-place slot attention (qwen_own's Triton decode kernel)
MIMI_BATCH = os.environ.get("POCKET_MIMI_BATCH", "1") != "0"     # one batched Mimi decode per frame for all rows
MIMI_RING = 272         # Mimi decoder KV ring per row: context 250 + one frame's 16 new positions (+ margin)


class _PoolState:
    """Streaming conv / convtr state of all rows in pooled tensors [slots, ...]: ``get`` gathers the rows of the
    current step (``idx``), assignment scatters them back. Same interface as the per-row dict PocketOwn uses."""

    def __init__(self, slots: int):
        self.slots, self.pool, self.idx = slots, {}, None

    def get(self, key):
        p = self.pool.get(key)
        return None if p is None else p.index_select(0, self.idx)

    def __setitem__(self, key, val):
        p = self.pool.get(key)
        if p is None:
            p = self.pool[key] = val.new_zeros(self.slots, *val.shape[1:])
        p.index_copy_(0, self.idx, val)

    def reset(self, slot: int):
        for p in self.pool.values():
            p[slot].zero_()


class SlotPool:
    """The per-row state of the talker: FlowLM KV slots, Mimi decoder KV rings and streaming conv state. The layout
    depends only on the architecture (24-layer FlowLM, Mimi), so every language's PocketBatch can share one pool:
    memory is paid once for ``slots`` concurrent rows, not once per resident language. The free list is shared."""

    def __init__(self, model: "PocketOwn", slots: int = 16, max_len: int = 1024):
        L, H, D = model.n_layers, model.nh, model.d // model.nh
        self.shape = (L, H, D, model.mimi_layers)
        self.kv = torch.zeros(L, 2, slots, max_len, H, D, device=model.dev, dtype=model.tdtype)
        self.max_len = max_len
        self.free = list(range(slots))
        ml, md = model.mimi_layers, 512
        self.mkv = torch.zeros(ml, 2, slots, MIMI_RING, 8, md // 8, device=model.dev)      # Mimi decoder KV rings
        self.mpos = torch.full((slots, MIMI_RING), -10 ** 9, dtype=torch.long, device=model.dev)
        self.mst = _PoolState(slots)

    def fits(self, model: "PocketOwn") -> bool:
        return self.shape == (model.n_layers, model.nh, model.d // model.nh, model.mimi_layers) and \
            self.kv.dtype == model.tdtype

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.kv, self.mkv, self.mpos))


class PocketBatch:
    """Many sessions of ONE language model stepped together (mynah-tts design: fixed KV slots, one batched step per
    latent frame for all active rows, one host sync per frame). A row = one segment being generated: its FlowLM KV
    lives in a slot of a pool (voice prefix + text + latents so far, per-row length), its Mimi decoder state in the
    row's own tensors. ``step()`` advances every active row by one latent frame and returns its audio (80 ms)."""

    def __init__(self, model: PocketOwn, slots: int = 16, max_len: int = 1024, pool: "SlotPool | None" = None):
        """pool: a SlotPool shared with the other languages' batches (one KV pool for the whole talker: the layout is
        the same in every language); None = this batch's own pool."""
        self.m = model
        self.pool = pool or SlotPool(model, slots, max_len)
        p = self.pool
        self.kv, self.max_len, self.free = p.kv, p.max_len, p.free
        self.mkv, self.mpos, self.mst = p.mkv, p.mpos, p.mst
        self.rows: dict = {}                  # key -> row dict

    def start(self, key, voice: _KV, text: str, after: int, gen=None):
        """A new row for one prepared text chunk (PocketOwn.chunks gives them with their frames-after-EOS)."""
        m = self.m
        s = self.free.pop(0)
        ids = m.tok(text)
        n0 = voice.off
        self.kv[:, :, s, :n0] = voice.c[:, :, :n0].to(self.kv.dtype)
        kv = _SlotKV(self, s, n0)
        m._flow_step(kv, text_ids=ids, gen=gen)          # text prefill on this row alone
        self.mst.reset(s)
        self.mpos[s] = -10 ** 9
        self.rows[key] = {"slot": s, "kv": kv, "x": torch.full((1, self.m.ldim), float("nan"), device=m.dev),
                          "st": {}, "moff": 0, "fade": True, "eos": None, "step": 0, "after": after, "gen": gen,
                          "max_len": m.max_gen_len(len(ids)), "done": False}
        return self.rows[key]

    def stop(self, key):
        r = self.rows.pop(key, None)
        if r is not None:
            self.free.append(r["slot"])

    @torch.no_grad()
    def step(self, keys: list):
        """One latent frame for the rows ``keys`` -> {key: audio (1920,) or None when the row finished}."""
        m, W = self.m, self.m.W
        rows = [self.rows[k] for k in keys]
        B = len(rows)
        lat_in = torch.cat([r["x"] for r in rows], 0)[:, None]                       # (B, 1, ldim)
        seq = torch.where(torch.isnan(lat_in), W["flow_lm.bos_emb"], lat_in)
        x = F.linear(seq, W["flow_lm.input_linear.weight"]).to(m.tdtype)
        slots = torch.tensor([r["slot"] for r in rows], device=m.dev)
        offs = torch.tensor([r["kv"].off for r in rows], device=m.dev)
        Lmax = int(offs.max()) + 1
        fused = m.tdtype == torch.bfloat16 and x.is_cuda and FUSED_ATTN
        if fused:
            from ..llm.qwen_own.kernels import HAVE_TRITON, decode_attention
            fused = HAVE_TRITON
        for li in range(m.n_layers):
            p = f"flow_lm.transformer.layers.{li}"
            h = m._ln(x, p + ".norm1")
            proj = m._lin(h, p + ".self_attn.in_proj.weight")
            d = m.d
            q = proj[..., :d].view(B, 1, m.nh, d // m.nh)
            k, v = proj[..., d:].view(B, 1, 2, m.nh, d // m.nh).unbind(2)
            q, k = _rope(q, k, offs)
            self.kv[li, 0, slots, offs] = k[:, 0]
            self.kv[li, 1, slots, offs] = v[:, 0]
            if fused:                     # read each row's slot in place (no gather of the pool into a batch)
                o = decode_attention(q[:, 0], self.kv[li, 0].permute(0, 2, 1, 3), self.kv[li, 1].permute(0, 2, 1, 3),
                                     slots, offs + 1)                                   # (B, H, D)
                o = o[:, None]                                                          # (B, 1, H, D)
            else:
                K = self.kv[li, 0, slots, :Lmax].transpose(1, 2)                     # (B, H, Lmax, D)
                V = self.kv[li, 1, slots, :Lmax].transpose(1, 2)
                mask = (torch.arange(Lmax, device=m.dev)[None] <= offs[:, None])[:, None, None, :]
                o = F.scaled_dot_product_attention(q.transpose(1, 2), K, V, mask).transpose(1, 2)
            x = x + m._lin(o.reshape(B, 1, d), p + ".self_attn.out_proj.weight")
            h = m._ln(x, p + ".norm2")
            x = x + m._lin(F.gelu(m._lin(h, p + ".linear1.weight"), approximate="tanh"), p + ".linear2.weight")
        for r in rows:
            r["kv"].off += 1
        y = m._ln(x.float(), "flow_lm.out_norm")[:, -1]
        eos = (F.linear(y, W["flow_lm.out_eos.weight"], W["flow_lm.out_eos.bias"]) > EOS_THR)[:, 0]
        if all(r["gen"] is not None for r in rows):
            noise = torch.cat([m._noise(1, r["gen"]) for r in rows], 0)
        else:
            noise = m._noise(B)
        lat = noise + m._flow_net(y, noise)
        mean, std = W["flow_lm.emb_mean"], W["flow_lm.emb_std"]
        eos_l = host(eos)                                                            # the one host sync per frame
        out, dec = {}, []
        for i, (k, r) in enumerate(zip(keys, rows)):
            if eos_l[i] and r["eos"] is None and r["step"] >= MIN_FRAMES_BEFORE_EOS:
                r["eos"] = r["step"]
            if (r["eos"] is not None and r["step"] >= r["eos"] + r["after"]) or r["step"] >= r["max_len"]:
                r["done"] = True
                out[k] = None
                continue
            r["step"] += 1
            r["x"] = lat[i:i + 1]
            dec.append((k, r, lat[i:i + 1]))
        if MIMI_BATCH and dec:
            lat_d = torch.cat([l_ for _, _, l_ in dec], 0) * std + mean
            idx = torch.tensor([r["slot"] for _, r, _ in dec], device=m.dev)
            moff = torch.tensor([r["moff"] for _, r, _ in dec], device=m.dev)
            wav, n = self._mimi_decode(lat_d[:, None], idx, moff)
            audios = [wav[i, 0] for i in range(len(dec))]
            for _, r, _ in dec:
                r["moff"] += n
        else:                                      # Mimi per row (its own streaming state)
            audios = [m.mimi_decode((l_ * std + mean)[:, None], r["st"])[0, 0] for _, r, l_ in dec]
        for (k, r, l_), audio in zip(dec, audios):
            if r["fade"]:
                n = min(SR // 200, audio.shape[-1])
                audio[:n] *= torch.linspace(0, 1, SR // 200, device=m.dev)[:n]
                r["fade"] = False
            out[k] = audio
        return out


    def _mimi_decode(self, latent, idx, moff):
        """Batched PocketOwn.mimi_decode: latent (B, 1, 32) de-normalised, rows' slots ``idx`` and Mimi positions
        ``moff`` -> (audio (B, 1, 1920), new positions per row)."""
        m, W = self.m, self.m.W
        st = self.mst
        st.idx = idx
        x = F.conv1d(latent.transpose(1, 2), W["mimi.quantizer.output_proj.weight"])
        x = m._convtr(x, "mimi.upsample.convtr.convtr", st, stride=16, groups=x.shape[1], bias=False)
        n = x.shape[-1]
        x = self._mimi_transformer(x, idx, moff)
        return m._seanet_decoder(x, st), n

    def _mimi_transformer(self, x, idx, moff):
        """PocketOwn._mimi_transformer for B rows at different positions: keys in per-row rings, context 250."""
        m, W = self.m, self.m.W
        p = "mimi.decoder_transformer"
        x = x.transpose(1, 2)
        B, T, d = x.shape
        nh, hd = 8, d // 8
        pos_q = moff[:, None] + torch.arange(T, device=x.device)                      # (B, T)
        ring = pos_q % MIMI_RING
        self.mpos[idx[:, None], ring] = pos_q
        pk = self.mpos[idx]                                                           # (B, R)
        delta = pos_q[:, :, None] - pk[:, None, :]
        mask = ((delta >= 0) & (delta < 250))[:, None]                               # (B, 1, T, R)
        for li in range(m.mimi_layers):
            q_ = f"{p}.transformer.layers.{li}"
            h = m._ln(x, q_ + ".norm1")
            proj = F.linear(h, W[q_ + ".self_attn.in_proj.weight"])
            q = proj[..., :d].view(B, T, nh, hd)
            k, v = proj[..., d:].view(B, T, 2, nh, hd).unbind(2)
            q, k = _rope(q, k, moff)
            self.mkv[li, 0, idx[:, None], ring] = k
            self.mkv[li, 1, idx[:, None], ring] = v
            K = self.mkv[li, 0, idx].transpose(1, 2)                                   # (B, H, R, D)
            V = self.mkv[li, 1, idx].transpose(1, 2)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), K, V, mask)
            a = F.linear(o.transpose(1, 2).reshape(B, T, d), W[q_ + ".self_attn.out_proj.weight"])
            x = x + W[q_ + ".layer_scale_1.scale"] * a
            h = m._ln(x, q_ + ".norm2")
            u = F.linear(F.gelu(F.linear(h, W[q_ + ".linear1.weight"]), approximate="tanh"), W[q_ + ".linear2.weight"])
            x = x + W[q_ + ".layer_scale_2.scale"] * u
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------------------------- several languages
CROSS_LANG = os.environ.get("POCKET_CROSS_LANG", "1") != "0"
_PLANS: dict = {}                  # (id(model), ...) -> (weakrefs, {linear key: GroupedTerms or None})


def _plan(models: list) -> dict:
    """Per backbone linear: the grouped terms of these languages (delta languages against one shared base), or None
    where they cannot share a kernel launch (the base language itself, mixed precisions, no Triton)."""
    import weakref
    key = tuple(id(m) for m in models)
    hit = _PLANS.get(key)
    if hit is not None and all(r() is m for r, m in zip(hit[0], models)):
        return hit[1]
    fused, _ = _pocket_swap()
    base = models[0].base
    terms: dict = {}
    delta = [m for m in models if m.QB]           # delta languages (the base language has none)
    base = delta[0].base if delta else None
    if base is not None and all(m.base is base for m in delta) and hasattr(fused, "GroupedTerms"):
        for k in delta[0].QB:
            if any(m.QB.get(k) is not base.terms[k] or k not in m.Q for m in delta):
                terms[k] = None
                continue
            try:                         # the base language's own rows (no delta) are computed apart: see step_many
                terms[k] = fused.GroupedTerms([m.Q[k][:3] if m.QB else None for m in models],
                                              base.terms[k], models[0].dev)
            except ValueError:
                terms[k] = None
    while len(_PLANS) >= 16:
        _PLANS.pop(next(iter(_PLANS)))
    _PLANS[key] = ([weakref.ref(m) for m in models], terms)
    return terms


@torch.no_grad()
def step_many(groups: list) -> dict:
    """One latent frame for rows of SEVERAL languages that share one SlotPool: ``groups`` = [(PocketBatch, keys)].
    The rows go through the FlowLM backbone together: attention reads the shared pool once for all rows, every
    backbone linear is one launch of pocket_swap's grouped kernel (the shared base tile is read once for all the
    languages; each language's rows get x @ (W_base + delta_lang), the same arithmetic as their own batch), and the
    Mimi decode (identical in every language) is one batched call. The small per-language parts (norms, input and
    output projections, the flow head) run per language on its rows. Per row, the latents equal those of the
    language's own ``PocketBatch.step`` (same kernels, same row counts); the audio can differ by float rounding
    (Mimi runs batched over more rows). One language, or CROSS_LANG off: each batch steps on its own."""
    groups = [(b, list(ks)) for b, ks in groups if ks]
    if not groups:
        return {}
    pool = groups[0][0].pool
    if len(groups) == 1 or not CROSS_LANG or any(b.pool is not pool for b, _ in groups):
        out = {}
        for b, ks in groups:
            out.update(b.step(ks))
        return out
    b0 = groups[0][0]
    ms = [b.m for b, _ in groups]
    m0 = ms[0]
    rows = [[b.rows[k] for k in ks] for b, ks in groups]
    cnt = [len(r) for r in rows]
    st = [sum(cnt[:i]) for i in range(len(cnt))]
    span = list(zip(ms, st, cnt))
    allrows = [r for rs in rows for r in rs]
    B = len(allrows)
    plan = _plan(ms) if max(cnt) <= 16 else {}
    fused_mod, _ = _pocket_swap()
    grouped = bool(plan) and m0.dev.type == "cuda" and fused_mod.available(
        torch.empty(1, device=m0.dev), tuple({t.prec for t in plan.values() if t is not None}))
    apart = [(m, a, n) for m, a, n in span if not m.QB]       # the base language: its own plain linears
    if grouped:
        r0 = torch.tensor(st, dtype=torch.int32, device=m0.dev)
        rc = torch.tensor([0 if not m.QB else n for m, _, n in span], dtype=torch.int32, device=m0.dev)
        md = next(m for m in ms if m.QB)

    def ln(x, key):
        return torch.cat([m._ln(x[a:a + n], key) for m, a, n in span], 0)

    def lin(x, key):
        t = plan.get(key) if grouped else None
        if t is None:
            return torch.cat([m._lin(x[a:a + n], key) for m, a, n in span], 0)
        y = fused_mod.fused_linear_grouped(x.reshape(B, -1).to(torch.bfloat16), md.Q[key][1].shape[0], t, r0, rc,
                                           128).view(B, 1, -1)
        for m, a, n in apart:
            y[a:a + n] = m._lin(x[a:a + n], key)
        return y

    xs = []
    for m, rs in zip(ms, rows):
        lat_in = torch.cat([r["x"] for r in rs], 0)[:, None]
        seq = torch.where(torch.isnan(lat_in), m.W["flow_lm.bos_emb"], lat_in)
        xs.append(F.linear(seq, m.W["flow_lm.input_linear.weight"]).to(m.tdtype))
    x = torch.cat(xs, 0)
    kv = b0.kv
    slots = torch.tensor([r["slot"] for r in allrows], device=m0.dev)
    offs = torch.tensor([r["kv"].off for r in allrows], device=m0.dev)
    Lmax = int(offs.max()) + 1
    fa = m0.tdtype == torch.bfloat16 and x.is_cuda and FUSED_ATTN
    if fa:
        from ..llm.qwen_own.kernels import HAVE_TRITON, decode_attention
        fa = HAVE_TRITON
    d, nh = m0.d, m0.nh
    for li in range(m0.n_layers):
        p = f"flow_lm.transformer.layers.{li}"
        proj = lin(ln(x, p + ".norm1"), p + ".self_attn.in_proj.weight")
        q = proj[..., :d].view(B, 1, nh, d // nh)
        k, v = proj[..., d:].view(B, 1, 2, nh, d // nh).unbind(2)
        q, k = _rope(q, k, offs)
        kv[li, 0, slots, offs] = k[:, 0]
        kv[li, 1, slots, offs] = v[:, 0]
        if fa:
            o = decode_attention(q[:, 0], kv[li, 0].permute(0, 2, 1, 3), kv[li, 1].permute(0, 2, 1, 3),
                                 slots, offs + 1)[:, None]
        else:
            K = kv[li, 0, slots, :Lmax].transpose(1, 2)
            V = kv[li, 1, slots, :Lmax].transpose(1, 2)
            mask = (torch.arange(Lmax, device=m0.dev)[None] <= offs[:, None])[:, None, None, :]
            o = F.scaled_dot_product_attention(q.transpose(1, 2), K, V, mask).transpose(1, 2)
        x = x + lin(o.reshape(B, 1, d), p + ".self_attn.out_proj.weight")
        h = ln(x, p + ".norm2")
        x = x + lin(F.gelu(lin(h, p + ".linear1.weight"), approximate="tanh"), p + ".linear2.weight")
    for r in allrows:
        r["kv"].off += 1
    lats, eoss = [], []
    for (m, a, n), rs in zip(span, rows):
        W = m.W
        y = m._ln(x[a:a + n].float(), "flow_lm.out_norm")[:, -1]
        eoss.append((F.linear(y, W["flow_lm.out_eos.weight"], W["flow_lm.out_eos.bias"]) > EOS_THR)[:, 0])
        if all(r["gen"] is not None for r in rs):
            noise = torch.cat([m._noise(1, r["gen"]) for r in rs], 0)
        else:
            noise = m._noise(n)
        lats.append(noise + m._flow_net(y, noise))
    eos_l = host(torch.cat(eoss, 0))                                              # the one host sync per frame
    out, dec = {}, []
    i = 0
    for (b, ks), rs, lat in zip(groups, rows, lats):
        mean, std = b.m.W["flow_lm.emb_mean"], b.m.W["flow_lm.emb_std"]
        for j, (k, r) in enumerate(zip(ks, rs)):
            e = eos_l[i]
            i += 1
            if e and r["eos"] is None and r["step"] >= MIN_FRAMES_BEFORE_EOS:
                r["eos"] = r["step"]
            if (r["eos"] is not None and r["step"] >= r["eos"] + r["after"]) or r["step"] >= r["max_len"]:
                r["done"] = True
                out[k] = None
                continue
            r["step"] += 1
            r["x"] = lat[j:j + 1]
            dec.append((k, r, lat[j:j + 1] * std + mean))
    if dec:
        if MIMI_BATCH:                          # Mimi is the same tensors in every language: one decode for all
            idx = torch.tensor([r["slot"] for _, r, _ in dec], device=m0.dev)
            moff = torch.tensor([r["moff"] for _, r, _ in dec], device=m0.dev)
            wav, n = b0._mimi_decode(torch.cat([l_ for _, _, l_ in dec], 0)[:, None], idx, moff)
            audios = [wav[i, 0] for i in range(len(dec))]
            for _, r, _ in dec:
                r["moff"] += n
        else:
            audios = [m0.mimi_decode(l_[:, None], r["st"])[0, 0] for _, r, l_ in dec]
        for (k, r, _), audio in zip(dec, audios):
            if r["fade"]:
                n = min(SR // 200, audio.shape[-1])
                audio[:n] *= torch.linspace(0, 1, SR // 200, device=m0.dev)[:n]
                r["fade"] = False
            out[k] = audio
    return out


class _SlotKV(_KV):
    """A row's view of the pool, usable by PocketOwn._flow_step (single-row prefill)."""

    def __init__(self, batch: PocketBatch, slot: int, off: int):
        self.b, self.s, self.off = batch, slot, off

    @property
    def c(self):
        return self.b.kv[:, :, self.s]

    def grow(self, Tmax):
        if Tmax > self.b.max_len:
            raise ValueError("segment longer than the KV slot")

    def copy(self):
        raise NotImplementedError
