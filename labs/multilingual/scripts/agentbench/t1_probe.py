"""T1 — teacher-forced function-calling DECISION probe with acoustic features and a CONTROLLED dialogue state.

What it measures (and what it does not):
  * For each front-end arm (A = VC original, C-raw, C-ridge, D-E5, D-V2.1) and clip (tenant m03 catalog, m08 missing
    info; EN/IT/FR renders of M1), the Thinker's per-frame user-audio embeddings are captured from the REAL NVIDIA
    streaming perception (pipeline driven chunk by chunk; the Thinker call is stubbed in that capture pass only).
  * They are replayed through the frozen Thinker on the LIVE numerical path (prompt prefill into the V2 hybrid cache,
    then one frame per step; FC fast-path frames from a deep copy of the cache taken after the SOTC frame), with a
    FIXED history: tenant prompt (NVIDIA FC template), agent tokens = A's recorded opener up to its EOS at frame 94,
    then PAD; function channel PAD. No RNNT forcing, no live timing.
  * Read-outs at the caller's end: function-head logP(SOTC) per frame; control tokens (text PAD/BOS/EOS); the
    teacher-forced probability of each complete tool NAME after a SOTC at the frame where logP(SOTC) peaks; the
    greedy arguments under the gold name; feature statistics vs A on the same clip.
  * A high preference for `search_catalog` is a measure CONDITIONED ON THE FIXED STATE, not proof of comprehension.

Pre-registered validity gate (A, EN m03, its own recorded history, cached path; stop if any check fails):
  G0 the tokenisation of `<TOOLCALL>[{"name": "search_catalog",` equals A's recorded call tokens;
  G1 text argmax == recorded agent token on EVERY frame 0..133 (hence every BOS/EOS); function argmax PAD on
     0..132 and SOTC at 133 (the recorded FC entry frame);
  G2 FC fast path from a clone of the cache after frame 133, with its exact inputs (agent PAD, FC silence embedding,
     previous function token): function argmax == recorded call token at all 32 FC frames (incl. EOTC);
  G2b clone isolation: the same branch run again after a different branch from the same snapshot gives bitwise
     identical function logits.
  The no-cache full forward is compared and REPORTED separately (numeric info), never gating, never measuring.
  The m08 A replay against its recorded tokens is reported (informational: m08 had no FC; its live history may
  contain forced tokens).

Tool-name comparison of different lengths: candidates are the complete strings `<TOOLCALL>[{"name": "NAME",` — the
closing `",` token (as in A's recorded call) ends the name, so the candidates are DISJOINT events and their summed
token log-probabilities (no length normalisation) are comparable; renormalised over the candidate set, with the
residual mass (other strings) reported. Tokens shared by all candidates are scored separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

PAD, BOS, EOS, SOTC, EOTC = 12, 1, 2, 20, 21
FRAME_S = 0.08
TOOLS = ["search_catalog", "find_customer", "escalate_to_human", "prepare_return", "commit_return"]
NAME_PREFIX = '<TOOLCALL>[{"name": "'
NAME_CLOSE = '",'
HIST_UPTO = 94          # A's opener ends with EOS at frame 94 (identical in m03 and m08 traces)
GATE_SOTC = 133         # A m03: SOTC predicted by the regular step at frame 133 (function.arguments.done start 133)


# ----------------------------------------------------------------------------------------------- pure helpers
def trace_session(path, k=0) -> list[dict]:
    out, cur = [], None
    for line in open(path):
        e = json.loads(line)
        if e["ev"] == "session.opened":
            cur = []
            out.append(cur)
        if cur is not None:
            cur.append(e)
    return out[k]


def frame_tokens(sess: list[dict]) -> dict[int, int]:
    """thinker.step model_step=N with k text ids covers frames N-k..N-1 (FC steps with steps=0 carry no ids)."""
    d = {}
    for e in sess:
        if e["ev"] == "thinker.step" and e.get("text_ids"):
            n = len(e["text_ids"])
            for i, t in enumerate(e["text_ids"]):
                d[e["model_step"] - n + i] = t
    return d


def fc_call(sess: list[dict]) -> tuple[int, list[int]] | None:
    for e in sess:
        if e["ev"] == "function.arguments.done":
            return int(e["model_step"]), list(e["func_ids"])
    return None


def end_frame(end_s: float) -> int:
    return int(math.ceil(end_s / FRAME_S - 1e-9))


def fixed_history(a_tokens: dict[int, int], n: int, upto: int = HIST_UPTO) -> list[int]:
    return [a_tokens.get(t, PAD) if t <= upto else PAD for t in range(n)]


def prev_shift(tokens: list[int]) -> list[int]:
    """Input token at frame t is the token of frame t-1; frame 0 gets PAD (prompt present, wrapper:2384-2396)."""
    return [PAD] + list(tokens[:-1])


def name_candidates(encode, names=TOOLS) -> tuple[list[int], dict[str, list[int]]]:
    """Tokenise `<TOOLCALL>[{"name": "NAME",` per candidate; return (shared prefix ids, {name: remaining ids})."""
    full = {n: encode(NAME_PREFIX + n + NAME_CLOSE) for n in names}
    seqs = list(full.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    common = seqs[0][:k]
    rest = {n: s[k:] for n, s in full.items()}
    for a in rest:
        for b in rest:
            if a != b and rest[b][:len(rest[a])] == rest[a]:
                raise ValueError(f"candidate {a} is a token prefix of {b}: events not disjoint")
    return common, rest


def renorm(logps: dict[str, float]) -> tuple[dict[str, float], float]:
    m = max(logps.values())
    z = sum(math.exp(v - m) for v in logps.values())
    probs = {k: math.exp(v - m) / z for k, v in logps.items()}
    covered = sum(math.exp(v) for v in logps.values())
    return probs, max(0.0, 1.0 - covered)


def gate_text(pred: list[int], rec: dict[int, int], frames) -> dict:
    mism = [(t, rec.get(t), pred[t]) for t in frames if rec.get(t) != pred[t]]
    crit = [m for m in mism if m[1] in (BOS, EOS, SOTC, EOTC) or m[2] in (BOS, EOS, SOTC, EOTC)]
    n = len(list(frames))
    return {"frames": n, "mismatches": mism[:20], "n_mismatch": len(mism), "critical": crit,
            "match_rate": 1 - len(mism) / max(1, n)}


def region_stats(arm: torch.Tensor, ref: torch.Tensor, regions: dict[str, tuple[int, int]],
                 sil_arm: torch.Tensor | None, sil_ref: torch.Tensor | None) -> dict:
    out = {}
    for name, (a, b) in regions.items():
        x, y = arm[a:b].float(), ref[a:b].float()
        if len(x) == 0:
            continue
        r = {"n": len(x), "cos_vs_A": float(torch.nn.functional.cosine_similarity(x, y, -1).mean()),
             "norm_ratio_vs_A": float((x.norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-6)).mean()),
             "norm": float(x.norm(dim=-1).mean())}
        for tag, s in (("cos_own_silence", sil_arm), ("cos_A_silence", sil_ref)):
            if s is not None:
                r[tag] = float(torch.nn.functional.cosine_similarity(x, s.reshape(1, -1).float(), -1).mean())
        out[name] = r
    return out


# ----------------------------------------------------------------------------------------------- Thinker view
class Thinker:
    """Frozen VoiceChat Thinker driven on the LIVE numerical path: prompt prefill into the V2 hybrid cache (as the
    pipeline's native-cache prefill), then ONE frame per call (the cache refuses multi-frame continuation after the
    prefill, llm_cache.position_injecting_forward), fusion of agent text, user audio and function channel exactly as
    the wrapper (wrapper:2365-2416; FC fast path wrapper:1380-1555)."""

    def __init__(self, eng, prompt: str):
        self.w = eng.wrapper
        self.stt = self.w.model.stt_model
        self.fusion = self.w.fusion_module
        self.dtype = next(self.stt.llm.parameters()).dtype
        self.dev = next(self.stt.llm.parameters()).device
        with torch.no_grad():
            pe, _ = self.w._prepare_system_prompt_embeddings(prompt)
        self.prompt_emb = pe.to(self.dtype)
        self.P = self.prompt_emb.shape[1]

    def encode(self, s: str) -> list[int]:
        ids = list(self.w.tokenizer.text_to_ids(s))
        return ids[1:] if ids and ids[0] == BOS else ids

    def decode(self, ids) -> str:
        return self.w.tokenizer.ids_to_text(list(ids))

    def silence(self) -> torch.Tensor:
        return self.w._silence_embedding_cache.reshape(1, -1).to(self.dtype)

    def frames(self, audio: torch.Tensor, agent_prev: list[int], func_prev: list[int]) -> torch.Tensor:
        e = self.stt.embed_tokens
        a = e(torch.tensor(agent_prev, device=self.dev)).to(self.dtype)[None]
        f = e(torch.tensor(func_prev, device=self.dev)).to(self.dtype)[None]
        return self.fusion(agent_text_embeds=a, user_audio_embeds=audio.to(self.dev, self.dtype)[None],
                           user_text_embeds=None, function_embeds=f)

    def new_cache(self):
        from mynah_v2.model.llm_cache import _hybrid_cache_cls
        return _hybrid_cache_cls(self.stt.llm)(self.stt.llm.config, 1, dtype=self.dtype, device=self.dev,
                                               ssm_dtype=torch.float32)

    @torch.no_grad()
    def prefill(self):
        cache = self.new_cache()
        ans = self.stt(self.prompt_emb.contiguous(), cache=cache)
        c = ans.get("cache")
        return c if c is not None else cache

    @torch.no_grad()
    def step(self, cache, audio_row: torch.Tensor, agent_prev: int, func_prev: int):
        ans = self.stt(self.frames(audio_row.reshape(1, -1), [agent_prev], [func_prev]).contiguous(), cache=cache)
        c = ans.get("cache")
        return ans["text_logits"][0, -1], ans["function_logits"][0, -1], (c if c is not None else cache)

    @staticmethod
    def clone(cache):
        return copy.deepcopy(cache)          # same isolation as the wrapper's _clone_cache for cache objects

    @torch.no_grad()
    def forward_full(self, audio, agent_prev, func_prev):
        """No-cache full forward. INFORMATION ONLY (numerical comparison); never used for a measurement."""
        x = torch.cat([self.prompt_emb, self.frames(audio, agent_prev, func_prev)], 1)
        out = self.stt(x, cache=None)
        return out["text_logits"][0, self.P:], out["function_logits"][0, self.P:]


# ----------------------------------------------------------------------------------------------- probe logic
def fc_inputs(audio_pre: torch.Tensor, sil: torch.Tensor, hist: list[int], s: int, call_tokens: list[int]):
    """Full-sequence inputs for frames 0..s (regular) + len(call_tokens) FC fast-path frames (agent PAD, FC silence
    embedding, previous function token: SOTC at s, then the call tokens). Used only for the no-cache comparison."""
    n_fc = len(call_tokens)
    audio = torch.cat([audio_pre[:s + 1], sil.expand(n_fc, -1).to(audio_pre.dtype)], 0)
    agent = list(hist[:s]) + [PAD] * (1 + n_fc)          # frame s text token was PAD in A's run (FC entry)
    func = [PAD] * s + [SOTC] + list(call_tokens)
    return audio, prev_shift(agent), prev_shift(func)


def logp_rows(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), -1)


def summ(tl, fl) -> dict:
    lt, lf = logp_rows(tl), logp_rows(fl)
    return {"text_argmax": int(lt.argmax()), "func_argmax": int(lf.argmax()),
            "text_PAD": float(lt[PAD]), "text_BOS": float(lt[BOS]), "text_EOS": float(lt[EOS]),
            "func_PAD": float(lf[PAD]), "func_SOTC": float(lf[SOTC]), "func_EOTC": float(lf[EOTC])}


def run_frames(th, cache, audio, agent_prev: list[int], func_prev: list[int], snap=()):
    """Regular frames, one per step. Returns (per-frame summaries, cache, {t: clone of the cache AFTER frame t})."""
    rows, snaps, snap = [], {}, set(snap)
    for t in range(len(agent_prev)):
        tl, fl, cache = th.step(cache, audio[t], agent_prev[t], func_prev[t])
        rows.append(summ(tl, fl))
        if t in snap:
            snaps[t] = th.clone(cache)
    return rows, cache, snaps


def branch_fc(th, snap, sil, tokens: list[int], n_free: int = 0):
    """FC fast path from a CLONE of the cache taken after the SOTC frame s (the snapshot itself is never advanced).
    FC frame k (k = 1..): agent input PAD (wrapper forces PAD at s and during FC async), audio = FC silence embedding,
    function input = SOTC (k = 1) or tokens[k-2]; the function logits of FC frame k predict tokens[k-1].
    Then n_free greedy tokens (stop at EOTC). Returns (logp of each forced token, argmax at each, free tokens,
    raw function logits of the forced frames for bitwise isolation checks)."""
    cache = th.clone(snap)
    prev, lps, am, raw, free = SOTC, [], [], [], []
    for tok in tokens:
        _, fl, cache = th.step(cache, sil, PAD, prev)
        lf = logp_rows(fl)
        lps.append(float(lf[tok]))
        am.append(int(lf.argmax()))
        raw.append(fl.detach().float().clone())
        prev = tok
    for _ in range(n_free):
        _, fl, cache = th.step(cache, sil, PAD, prev)
        nxt = int(fl.argmax())
        free.append(nxt)
        prev = nxt
        if nxt == EOTC:
            break
    return lps, am, free, raw


def measure_clip(th, audio: torch.Tensor, hist: list[int], e: int, sil: torch.Tensor, gold: str | None,
                 window=(-4, 12), arg_steps: int = 24, a_opener_bos=(19, 65)) -> dict:
    T = min(audio.shape[0], e + window[1] + 1)
    w0, w1 = max(0, e + window[0]), T
    rows, _, snaps = run_frames(th, th.prefill(), audio[:T], prev_shift(hist[:T]), [PAD] * T, snap=range(w0, w1))
    win = [dict(r, t=t, rel=t - e) for t, r in enumerate(rows) if w0 <= t < w1]
    s = max(range(w0, w1), key=lambda t: rows[t]["func_SOTC"])
    snap = snaps[s]
    snaps.clear()
    res = {"user_end_frame": e, "window": win, "sotc_peak_frame": s, "sotc_peak_rel": s - e,
           "sotc_peak_logp": rows[s]["func_SOTC"],
           "first_sotc_argmax_rel": next((r["rel"] for r in win if r["func_argmax"] == SOTC), None),
           "opener_bos_logp": {str(t): rows[t]["text_BOS"] for t in a_opener_bos if t < T},
           "text_argmax_nonpad_after_end": [(r["rel"], r["text_argmax"]) for r in win
                                            if r["rel"] >= 0 and r["text_argmax"] != PAD][:5]}
    # tool-name preference after a SOTC at the peak frame (a measure CONDITIONED on that SOTC and the fixed state)
    common, rest = name_candidates(th.encode)
    names, shared = {}, {}
    for n, suf in rest.items():
        lps, _, _, _ = branch_fc(th, snap, sil, common + suf)
        shared[n] = sum(lps[:len(common)])
        names[n] = sum(lps[len(common):])
    probs, residual = renorm(names)
    res["names"] = {"shared_prefix_logp": shared[TOOLS[0]],
                    "shared_prefix_identical": len({round(v, 6) for v in shared.values()}) == 1,
                    "name_logp": names, "name_prob_renorm": probs, "residual_mass": residual,
                    "top": max(names, key=names.get)}
    if gold:
        res["names"]["gold"] = gold
        res["names"]["gold_rank"] = sorted(names, key=names.get, reverse=True).index(gold) + 1
        _, _, free, _ = branch_fc(th, snap, sil, common + rest[gold], n_free=arg_steps)
        res["args_greedy_ids"] = free
        res["args_greedy_text"] = th.decode([i for i in free if i != EOTC])
    return res


def gate(th, audio: torch.Tensor, rec: dict[int, int], call: tuple[int, list[int]], sil, numeric_info=True) -> dict:
    """Pre-registered gate on the live (cached) path. PASS iff: every text argmax on frames 0..s equals the recorded
    agent token (so in particular every BOS/EOS); function head PAD on 0..s-1 and SOTC at s; all 32 recorded call
    tokens reproduced from a clone of the cache after s; two further branches from the same snapshot (with a different
    candidate in between) give bitwise-identical logits (no contamination). The no-cache comparison is reported
    separately and is not part of the gate."""
    s, fids = call
    assert s == GATE_SOTC and fids[0] == SOTC and fids[-1] == EOTC, (s, fids[:2], fids[-1])
    toks = fids[1:]
    hist = [rec.get(t, PAD) for t in range(s + 1)]
    rows, _, snaps = run_frames(th, th.prefill(), audio[:s + 1], prev_shift(hist), [PAD] * (s + 1), snap={s})
    pt, pf = [r["text_argmax"] for r in rows], [r["func_argmax"] for r in rows]
    g1 = gate_text(pt, rec, range(0, s + 1))
    g1["func_pad_before"] = all(pf[t] == PAD for t in range(0, s))
    g1["func_sotc_at"] = pf[s] == SOTC
    g1["func_first_nonpad"] = next((t for t in range(len(pf)) if pf[t] != PAD), None)
    g1["sotc_logp_at_s"] = rows[s]["func_SOTC"]
    lps, am, _, raw1 = branch_fc(th, snaps[s], sil, toks)
    g2_m = [(k, toks[k - 1], am[k - 1]) for k in range(1, len(toks) + 1) if am[k - 1] != toks[k - 1]]
    g2 = {"n_tokens": len(toks), "mismatches": g2_m, "call_logp": sum(lps)}
    common, rest = name_candidates(th.encode)
    branch_fc(th, snaps[s], sil, common + rest["find_customer"])        # a different branch in between
    _, _, _, raw2 = branch_fc(th, snaps[s], sil, toks)
    iso = all(torch.equal(x, y) for x, y in zip(raw1, raw2))
    ok = (not g1["critical"] and g1["match_rate"] == 1.0 and g1["func_pad_before"] and g1["func_sotc_at"]
          and not g2_m and iso)
    out = {"pass": ok, "path": "cached per-frame (live)", "G1": g1, "G2": g2, "clone_isolation_bitwise": iso}
    if numeric_info:                                                      # reported, not gating
        a, ag, fu = fc_inputs(audio, sil, hist, s, toks)
        tl, fl = th.forward_full(a, ag, fu)
        ft, ff = tl.argmax(-1).tolist(), fl.argmax(-1).tolist()
        cached_f = pf + am
        out["numeric_info_nocache"] = {
            "text_argmax_diff_frames_0_s": [t for t in range(s + 1) if ft[t] != pt[t]],
            "func_argmax_diff_frames": [t for t in range(len(cached_f)) if ff[t] != cached_f[t]],
            "nocache_text_vs_record": gate_text(ft, rec, range(0, s + 1))["mismatches"]}
    return out


# ----------------------------------------------------------------------------------------------- GPU driver
def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-root", default=str(Path.home() / "mynah-s2s/ml/data/agentbench/tts_m1"))
    ap.add_argument("--a-m03-trace", default=str(Path.home() / "mynah-s2s/ml/results/agentbench/m1-parity/vc_en_m03/trace.jsonl"))
    ap.add_argument("--a-m08-trace", default=str(Path.home() / "mynah-s2s/ml/results/agentbench/m1/vc_en_m08/trace.jsonl"))
    ap.add_argument("--budget-s", type=float, default=13.5 * 60, help="soft stop for optional units (job timeout is the hard cap)")
    ap.add_argument("--lead-s", type=float, default=7.0)
    ap.add_argument("--tail-s", type=float, default=3.0)
    ap.add_argument("--units", default="", help="override the plan: 'arm:scenario:lang,...' (all subject to --budget-s)")
    ap.add_argument("--gold-m08", default="", help="decode arguments under this tool name on m08 (e.g. find_customer)")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    if (out / "t1.json").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "multilingual"))
    import observe
    from e2e_qa import apply_variant, parse_variant
    from m0_runner import load_16k, render_prompt
    from perception_swap import FrontEnd, load_nemo_dir
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import iter_chunks

    timing, report = {}, {"units": [], "skipped": []}

    def save():
        report["timing_s"] = timing
        report["elapsed_s"] = round(time.monotonic() - T0, 1)
        (out / "t1.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    torch.manual_seed(1234)
    t = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}),
                          llm_cache=True)
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):     # decode_audio=False harness fix (decision D7)
        if audio_wave is None:
            audio_wave = torch.zeros(1, 0)
        return _orig_log(frames, audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    timing["engine_load"] = round(time.monotonic() - t, 1)
    spec = json.load(open(HERE / "tenant_furniture.json"))
    be = TenantBackend(spec)
    reg = ToolRegistry()
    for tl in spec["tools"]:
        reg.register(tl["name"], be.handlers()[tl["name"]], tl["description"], tl["parameters"])
    prompt = render_prompt(reg.schemas, spec["system_prompt"])
    report["prompt_chars"] = len(prompt)
    assert len(prompt) == 2738, len(prompt)                # identical to the live M1 sessions
    th = Thinker(eng, prompt)
    report["prompt_tokens"] = th.P
    fe = FrontEnd(eng)
    fe.snapshot()
    t = time.monotonic()
    ckpt = {"en": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-en"),
            "ml": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-ml")}
    timing["ckpt_load"] = round(time.monotonic() - t, 1)
    R = str(Path.home() / "mynah-s2s/ml/results")
    ARMS = {"A": "vc",
            "C-ridge": f"enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out.pt",
            "D-V2.1": f"enc=ml;att=56;rnnt=ml;ln={R}/v21/distill/v21/encoder_ln_best.pt;adapter={R}/v21/distill/v21/adapter_best.pt",
            "D-E5": f"enc=ml;att=56;rnnt=ml;ln={R}/distill/e5/encoder_ln_best.pt;adapter={R}/distill/e5/adapter_best.pt",
            "C-raw": "enc=ml;att=56;rnnt=ml;ln=orig"}
    man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(Path(a.audio_root) / "manifest.jsonl"))}
    clips = {}
    for sid in ("m03", "m08"):
        for lang in ("en", "it", "fr"):
            voice = "vivian" if lang == "en" else "ryan"
            x = load_16k(Path(a.audio_root) / man[(f"{sid}_t0", lang, voice)]["path"])
            end_s = a.lead_s + x.numel() / 16000
            wav = torch.cat([torch.zeros(int(a.lead_s * 16000)), x, torch.zeros(int(a.tail_s * 16000))])
            clips[(sid, lang)] = {"wav": wav, "end_frame": end_frame(end_s), "end_s": round(end_s, 2)}

    cur = {"arm": None, "sil": {}}
    feats = {}

    def use(arm):
        if cur["arm"] != arm:
            t = time.monotonic()
            apply_variant(fe, parse_variant(ARMS[arm]), ckpt)   # restores the original first; refreshes silence emb
            cur["arm"] = arm
            cur["sil"][arm] = th.silence().detach().clone()
            timing.setdefault("swap", []).append([arm, round(time.monotonic() - t, 1), fe.state.get("silence_emb")])

    def capture(arm, key):
        """Real NVIDIA streaming perception for the clip; the Thinker call is stubbed (perception does not depend on
        it). Returns the per-frame Thinker audio embeddings [T, H] and the RNNT/forcing timeline."""
        use(arm)
        t = time.monotonic()
        w = eng.wrapper
        rec, fx = [], []
        un = observe.install(w, rec, what=("forcing",))
        orig_if = w.model_llm_interface
        orig_fu = w.fusion_module.forward

        def stub(*args, **kw):
            pad = torch.full((1,), PAD, dtype=torch.long, device=th.dev)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad,
                    "cache": kw.get("cache")}

        def fuse(*args, **kw):
            ua = kw.get("user_audio_embeds")
            if ua is not None and ua.dim() == 3 and ua.shape[1] == 1:
                fx.append(ua[0, 0].detach().clone())
            return orig_fu(*args, **kw)
        w.model_llm_interface = stub
        w.fusion_module.forward = fuse
        try:
            eng.open(None)
            chunks = list(iter_chunks(clips[key]["wav"], eng.chunk_samples))
            for i, c in enumerate(chunks):
                eng.step(c, is_last=(i == len(chunks) - 1))
            eng.close()
        finally:
            w.model_llm_interface = orig_if
            w.fusion_module.forward = orig_fu
            un()
        feats[(arm, key)] = torch.stack(fx)
        forced = [r for r in rec if r.get("forced")]
        timing.setdefault("capture", []).append([arm, *key, round(time.monotonic() - t, 1), len(fx)])
        return {"frames": len(fx), "forced_in_capture": forced[:10],
                "note": "capture pass: Thinker stubbed (PAD), so RNNT-forcing rows reflect a PAD-only agent"}

    a_m03 = trace_session(a.a_m03_trace)
    rec03, call03 = frame_tokens(a_m03), fc_call(a_m03)
    a_m08 = trace_session(a.a_m08_trace)
    rec08 = frame_tokens(a_m08)
    report["history"] = {"upto": HIST_UPTO, "m03_m08_opener_identical": all(rec03.get(t) == rec08.get(t) for t in range(HIST_UPTO + 1)),
                         "special_frames_m03": [(t, v) for t, v in sorted(rec03.items()) if v in (BOS, EOS)]}

    # ---- phase 1: gate (mandatory)
    common, rest = name_candidates(th.encode)
    g0 = (common + rest["search_catalog"]) == call03[1][1:1 + len(common) + len(rest["search_catalog"])]
    report["G0_tokenisation"] = {"ok": g0, "common": common, "search_catalog": rest["search_catalog"]}
    if not g0:
        save()
        raise SystemExit("G0 failed: candidate tokenisation differs from the recorded call")
    cap = capture("A", ("m03", "en"))
    t = time.monotonic()
    g = gate(th, feats[("A", ("m03", "en"))], rec03, call03, cur["sil"]["A"])
    timing["gate"] = round(time.monotonic() - t, 1)
    report["gate"] = g
    report["gate"]["capture"] = cap
    save()
    if not g["pass"]:
        print("GATE FAILED", json.dumps(g)[:2000], flush=True)
        raise SystemExit(3)
    print("GATE PASS", flush=True)
    capture("A", ("m08", "en"))
    t = time.monotonic()
    au = feats[("A", ("m08", "en"))]
    n08 = min(len(au), max(rec08) + 1, 175)
    rows8, _, _ = run_frames(th, th.prefill(), au[:n08], prev_shift([rec08.get(i, PAD) for i in range(n08)]), [PAD] * n08)
    report["gate_m08_info"] = gate_text([r["text_argmax"] for r in rows8], rec08, range(0, n08))
    report["gate_m08_info"]["func_first_nonpad"] = next((i for i, r in enumerate(rows8) if r["func_argmax"] != PAD), None)
    timing["gate_m08"] = round(time.monotonic() - t, 1)
    save()

    hist = fixed_history(rec03, 4000)
    GOLD = {"m03": "search_catalog", "m08": a.gold_m08 or None}
    plan = [("A", "m03", "en"), ("A", "m08", "en"),
            ("C-ridge", "m03", "en"), ("C-ridge", "m08", "en"),
            ("D-V2.1", "m03", "en"), ("D-V2.1", "m08", "en"),
            ("A", "m03", "it"), ("A", "m03", "fr"), ("C-ridge", "m03", "it"), ("C-ridge", "m03", "fr"),
            ("D-V2.1", "m03", "it"), ("D-V2.1", "m03", "fr"),
            ("A", "m08", "it"), ("A", "m08", "fr"), ("C-ridge", "m08", "it"), ("C-ridge", "m08", "fr"),
            ("D-V2.1", "m08", "it"), ("D-V2.1", "m08", "fr"),
            ("D-E5", "m03", "en"), ("D-E5", "m08", "en"), ("D-E5", "m03", "it"), ("D-E5", "m03", "fr"),
            ("C-raw", "m03", "en"), ("C-raw", "m08", "en"), ("C-raw", "m03", "it"), ("C-raw", "m03", "fr")]
    n_core = 6
    if a.units:
        plan = [tuple(u.split(":")) for u in a.units.split(",")]
        n_core = 0
    unit_cost = []
    for i, (arm, sid, lang) in enumerate(plan):
        est = (max(unit_cost) * 1.2 if unit_cost else 60) + (15 if cur["arm"] != arm else 0)
        if i >= n_core and time.monotonic() - T0 + est > a.budget_s:   # default plan: the EN core always runs
            report["skipped"] = [list(p) for p in plan[i:]]
            print("BUDGET: skipping", len(plan) - i, "units", flush=True)
            break
        t = time.monotonic()
        key = (sid, lang)
        if (arm, key) not in feats:
            capture(arm, key)
        if ("A", key) not in feats:                  # reference features for the region statistics
            capture("A", key)
        au = feats[(arm, key)]
        e = clips[key]["end_frame"]
        res = measure_clip(th, au, hist, e, cur["sil"][arm], GOLD[sid])
        ref = feats[("A", key)]
        n = min(len(au), len(ref))
        res["features"] = region_stats(au[:n], ref[:n], {"lead_silence": (0, 88), "speech": (88, e), "after": (e, n)},
                                       cur["sil"][arm], cur["sil"].get("A"))
        res.update({"phase": 1 if i < 4 else 2 if i < 6 else 3 if i < 18 else 4,
                    "arm": arm, "scenario": sid, "lang": lang, "end_s": clips[key]["end_s"],
                    "unit_s": round(time.monotonic() - t, 1)})
        unit_cost.append(time.monotonic() - t)
        report["units"].append(res)
        nm = res["names"]
        print(f"{arm:8} {sid} {lang} sotc_peak {res['sotc_peak_logp']:.2f}@{res['sotc_peak_rel']:+d} "
              f"top {nm['top']} p={nm['name_prob_renorm'][nm['top']]:.2f} resid={nm['residual_mass']:.2f} "
              f"args={res.get('args_greedy_text', '')!r} ({res['unit_s']}s)", flush=True)
        save()
    fe.restore()
    save()
    print("DONE", round(time.monotonic() - T0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
