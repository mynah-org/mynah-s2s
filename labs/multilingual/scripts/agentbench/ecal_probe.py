"""E-cal: does a closed-form, variance-calibrated ridge interface (C-cal) improve slots without hurting opener and
listening? Fixed-state replay (the T1 harness: A's opener history up to frame 94, then PAD; tenant prompt; one frame
per step on the live cached path). No training, no live session. See .work/multilingual/transplant-diagnosis.md §6.

Arms: A (VC), C-ridge (ridge_ml_out.pt), C-cal (ridge_ml_out_cal.pt: per-dim std/mean matched to VC, closed form),
D-V2.1. Clips, in priority order: EN m07 (tts_m0_en), IT m02, IT m03, EN m03 (tts_m1); EN Vivian, IT Ryan; lead 7 s.

Read-outs per arm x clip (pre-registered):
  R1 opener: text logP(BOS) at A's opener BOS frames (fixed history), and the argmax there;
  R2 listening: during the caller's turn (frame 88+2 .. end-1), max text logP(BOS), number of BOS-argmax frames, and
     logP(BOS) at the clip's audio micro-pause frames (80 ms RMS < 1 % of the clip peak, inside the turn);
  R3 first call: function logP(SOTC) peak and SOTC-PAD margin in [end-4, end+12], first SOTC-argmax frame;
  R4 slots: greedy arguments under the GOLD tool name from a clone at the SOTC peak (m07, m02: find_customer;
     m03: search_catalog), parsed into slot values with provenance against the caller's words;
  R5 features (post-proj, the Thinker's audio input): norm speech / lead silence, per-dim std ratio vs A on speech,
     cos vs A per region.
Validity: A must give grounded slots on EN m07 (Mario Rossi / 20135) and EN m03 ("sofa"); otherwise R4 is void.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1_probe import (BOS, EOTC, PAD, SOTC, Thinker, branch_fc, end_frame, fixed_history, frame_tokens,  # noqa: E402
                      logp_rows, name_candidates, prev_shift, region_stats, run_frames, trace_session)

CALLER_START = 88          # 7.0 s lead at 80 ms
A_OPENER_BOS = (19, 65)    # A's recorded opener BOS frames (T1 measure_clip; 9 and 54 are EOS: job 280 assert)
CLIPS = [("m07", "en", "tts_m0_en", "find_customer"), ("m02", "it", "tts_m1", "find_customer"),
         ("m03", "it", "tts_m1", "search_catalog"), ("m03", "en", "tts_m1", "search_catalog"),
         ("m03", "fr", "tts_m1", "search_catalog"), ("m07", "it", "../v3/tts_m07_itfr", "find_customer"),
         ("m07", "fr", "../v3/tts_m07_itfr", "find_customer")]
CALLER = {("m07", "en"): "I want to return the bookcase I bought. I'm Mario Rossi, postcode two zero one three five.",
          ("m02", "it"): "Salve, sono Mario Rossi, dov'è il mio ordine?",
          ("m03", "it"): "Cerco un divano a tre posti. Cosa avete?",
          ("m03", "en"): "I'm looking for a three-seat sofa. What do you have?",
          ("m03", "fr"): "Je cherche un canapé trois places. Qu'est-ce que vous avez ?",
          ("m07", "it"): "Voglio restituire la libreria che ho comprato. Sono Mario Rossi, CAP due zero uno tre cinque.",
          ("m07", "fr"): "Je veux retourner la bibliothèque que j'ai achetée. Je suis Mario Rossi, code postal deux zéro un trois cinq."}
DIGITS = {"zéro": "0", "un": "1", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5", "sept": "7", "huit": "8",
          "neuf": "9", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
          "eight": "8", "nine": "9", "uno": "1", "due": "2", "tre": "3", "quattro": "4", "cinque": "5", "sei": "6",
          "sette": "7", "otto": "8", "nove": "9"}


# ----------------------------------------------------------------------------------------------- pure helpers
def parse_args_text(s: str) -> dict:
    """'"arguments": {"full_name": "Mario Rossi", "postcode": "20135"}}]' -> {"full_name": ..., "postcode": ...}."""
    return dict(re.findall(r'"(\w+)":\s*"([^"]*)"', s.split("arguments", 1)[-1]))


def spoken_digits(text: str) -> str:
    return "".join(DIGITS[w] for w in re.findall(r"[a-zà-ü]+", text.lower()) if w in DIGITS)


def provenance(slots: dict, caller: str) -> dict:
    """Per slot: grounded (every word or digit run is in the caller's words), partial, or ungrounded."""
    words = set(re.findall(r"[a-zà-ü0-9]+", caller.lower()))
    digits = spoken_digits(caller) + "".join(re.findall(r"\d", caller))
    out = {}
    for k, v in slots.items():
        toks = re.findall(r"[a-zà-ü0-9]+", v.lower())
        hit = [t for t in toks if t in words or (t.isdigit() and t in digits)]
        out[k] = "grounded" if toks and len(hit) == len(toks) else "partial" if hit else "ungrounded"
    return out


def pause_frames(wav: torch.Tensor, lo: int, hi: int, frame: int = 1280, thr: float = 0.01) -> list[int]:
    n = wav.numel() // frame
    rms = wav[:n * frame].reshape(n, frame).pow(2).mean(1).sqrt()
    peak = float(rms.max())
    return [t for t in range(lo, min(hi, n)) if float(rms[t]) < thr * peak]


def listening(rows: list[dict], lo: int, hi: int, pauses: list[int]) -> dict:
    span = [(t, rows[t]["text_BOS"]) for t in range(lo, min(hi, len(rows)))]
    best = max(span, key=lambda x: x[1]) if span else (None, None)
    return {"max_bos_logp": round(best[1], 3) if best[0] is not None else None, "max_bos_frame": best[0],
            "bos_argmax_frames": [t for t in range(lo, min(hi, len(rows))) if rows[t]["text_argmax"] == BOS],
            "pause_bos_logp_max": round(max((rows[t]["text_BOS"] for t in pauses if t < len(rows)), default=-99), 3),
            "n_pause_frames": len(pauses)}


# ----------------------------------------------------------------------------------------------- GPU
def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "mynah-s2s/ml/data/agentbench"))
    ap.add_argument("--a-trace", default=str(Path.home() / "mynah-s2s/ml/results/agentbench/m1-parity/vc_en_m03/trace.jsonl"))
    ap.add_argument("--arms", default="A,C-ridge,C-cal,D-V2.1")
    ap.add_argument("--budget-s", type=float, default=540.0)
    ap.add_argument("--arg-steps", type=int, default=36)
    ap.add_argument("--arm-spec", action="append", default=[], help="extra arm NAME=SPEC (front-end spec, e.g. a D-V3 "
                                                                   "checkpoint: enc=ml;att=56;rnnt=ml;ln=...;adapter=...)")
    ap.add_argument("--clips", default="", help="subset of clip keys, e.g. m07_en,m02_it,m03_it")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    if (out / "ecal.json").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "multilingual"))
    from e2e_qa import apply_variant, parse_variant
    from m0_runner import load_16k, render_prompt
    from perception_swap import FrontEnd, load_nemo_dir
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import iter_chunks
    rep = {"units": [], "skipped": [], "timing_s": {}, "features": {}}

    def save():
        rep["elapsed_s"] = round(time.monotonic() - T0, 1)
        (out / "ecal.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    torch.manual_seed(1234)
    t = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}), llm_cache=True)
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):     # decode_audio=False harness fix (decision D7)
        return _orig_log(frames, torch.zeros(1, 0) if audio_wave is None else audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    rep["timing_s"]["engine_load"] = round(time.monotonic() - t, 1)
    spec = json.load(open(HERE / "tenant_furniture.json"))
    be, reg = TenantBackend(spec), ToolRegistry()
    for tl in spec["tools"]:
        reg.register(tl["name"], be.handlers()[tl["name"]], tl["description"], tl["parameters"])
    prompt = render_prompt(reg.schemas, spec["system_prompt"])
    assert len(prompt) == 2738, len(prompt)
    th = Thinker(eng, prompt)
    w = eng.wrapper
    hist = fixed_history(frame_tokens(trace_session(a.a_trace)), 4000)
    assert all(hist[t] == BOS for t in A_OPENER_BOS), [hist[t] for t in A_OPENER_BOS]
    common, rest = name_candidates(th.encode)
    R = str(Path.home() / "mynah-s2s/ml/results")
    ARMS = {"A": "vc",
            "C-ridge": f"enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out.pt",
            "C-cal": f"enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out_cal.pt",
            "D-V2.1": f"enc=ml;att=56;rnnt=ml;ln={R}/v21/distill/v21/encoder_ln_best.pt;adapter={R}/v21/distill/v21/adapter_best.pt"}
    for x in a.arm_spec:
        k, v = x.split("=", 1)
        ARMS[k] = v
    fe = FrontEnd(eng)
    fe.snapshot()
    ckpt = {"en": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-en"),
            "ml": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-ml")}
    clips = {}
    for sid, lang, root, gold in [c for c in CLIPS if not a.clips or f"{c[0]}_{c[1]}" in a.clips.split(",")]:
        rr = Path(a.data_root) / root
        man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(rr / "manifest.jsonl"))}
        x = load_16k(rr / man[(f"{sid}_t0", lang, "vivian" if lang == "en" else "ryan")]["path"])
        wav = torch.cat([torch.zeros(7 * 16000), x, torch.zeros(3 * 16000)])
        e = end_frame(7.0 + x.numel() / 16000)
        clips[(sid, lang)] = {"wav": wav, "end": e, "gold": gold, "pauses": pause_frames(wav, CALLER_START + 2, e - 1)}
    rep["clips"] = {f"{k[0]}_{k[1]}": {"end": v["end"], "pauses": v["pauses"]} for k, v in clips.items()}
    feats, sils = {}, {}

    def capture(arm):
        t = time.monotonic()
        apply_variant(fe, parse_variant(ARMS[arm]), ckpt)       # restores the original first; refreshes silence emb
        sils[arm] = th.silence().detach().clone()
        orig_if, orig_fu = w.model_llm_interface, w.fusion_module.forward

        def stub(*args, **kw):
            pad = torch.full((1,), PAD, dtype=torch.long, device=th.dev)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad,
                    "cache": kw.get("cache")}
        for key, c in clips.items():
            fx = []

            def fuse(*args, **kw):
                ua = kw.get("user_audio_embeds")
                if ua is not None and ua.dim() == 3 and ua.shape[1] == 1:
                    fx.append(ua[0, 0].detach().clone())
                return orig_fu(*args, **kw)
            w.model_llm_interface, w.fusion_module.forward = stub, fuse
            try:
                eng.open(None)
                chunks = list(iter_chunks(c["wav"], eng.chunk_samples))
                for i, ch in enumerate(chunks):
                    eng.step(ch, is_last=(i == len(chunks) - 1))
                eng.close()
            finally:
                w.model_llm_interface, w.fusion_module.forward = orig_if, orig_fu
            feats[(arm, key)] = torch.stack(fx)
        rep["timing_s"].setdefault("capture", []).append([arm, round(time.monotonic() - t, 1)])

    arms = a.arms.split(",")
    for arm in arms:                     # captures first (cheap); one front-end swap per arm
        capture(arm)
    fe.restore()
    # R5 features: post-proj (the Thinker's audio input)
    for key, c in (clips.items() if "A" in arms else []):     # R5 needs A as the reference
        e, ref = c["end"], feats[("A", key)]
        for arm in arms:
            f = feats[(arm, key)].float()
            n = min(len(f), len(ref))
            sp, si = f[CALLER_START:e], f[10:CALLER_START - 4]
            st = region_stats(f[:n], ref[:n].float(), {"lead_silence": (10, CALLER_START - 4), "speech": (CALLER_START, e),
                                                        "after": (e + 3, n)}, sils[arm], sils["A"])
            st["speech_over_silence_norm"] = round(float(sp.norm(dim=1).mean() / si.norm(dim=1).mean()), 3)
            st["std_ratio_vs_A_speech_median"] = round(float((sp.std(0) / ref[CALLER_START:e].float().std(0).clamp_min(1e-6)).median()), 3)
            rep["features"][f"{arm}|{key[0]}_{key[1]}"] = st
    save()

    plan = [(key, arm) for key in clips for arm in arms]
    cost = []
    for i, (key, arm) in enumerate(plan):
        est = (max(cost) * 1.15) if cost else 35.0
        if time.monotonic() - T0 + est > a.budget_s:
            rep["skipped"] = [[f"{k[0]}_{k[1]}", ar] for k, ar in plan[i:]]
            print("BUDGET: skipping", rep["skipped"], flush=True)
            break
        t = time.monotonic()
        c, au = clips[key], feats[(arm, key)]
        e = c["end"]
        T = min(len(au), e + 13)
        rows, _, snaps = run_frames(th, th.prefill(), au[:T], prev_shift(hist[:T]), [PAD] * T, snap=range(e - 4, T))
        win = list(range(max(0, e - 4), T))
        s = max(win, key=lambda t_: rows[t_]["func_SOTC"])
        u = {"clip": f"{key[0]}_{key[1]}", "arm": arm, "end": e,
             "R1_opener": {str(t_): {"bos_logp": round(rows[t_]["text_BOS"], 3), "argmax": rows[t_]["text_argmax"]}
                           for t_ in A_OPENER_BOS},
             "R2_listening": listening(rows, CALLER_START + 2, e - 1, c["pauses"]),
             "R3_call": {"sotc_peak_frame": s, "sotc_peak_rel": s - e, "sotc_peak_logp": round(rows[s]["func_SOTC"], 3),
                         "sotc_minus_pad_at_peak": round(rows[s]["func_SOTC"] - rows[s]["func_PAD"], 3),
                         "first_sotc_argmax_rel": next((t_ - e for t_ in win if rows[t_]["func_argmax"] == SOTC), None),
                         "text_bos_after_end": next((t_ - e for t_ in win if t_ >= e and rows[t_]["text_argmax"] == BOS), None)}}
        _, _, free, _ = branch_fc(th, snaps[s], sils[arm], common + rest[c["gold"]], n_free=a.arg_steps)
        snaps.clear()
        txt = th.decode([i_ for i_ in free if i_ != EOTC])
        slots = parse_args_text(txt)
        u["R4_slots"] = {"gold_tool": c["gold"], "args_text": txt, "slots": slots,
                         "provenance": provenance(slots, CALLER[key])}
        u["s"] = round(time.monotonic() - t, 1)
        cost.append(u["s"])
        rep["units"].append(u)
        save()
        print(f"{u['clip']:7} {arm:7} opener={[v['bos_logp'] for v in u['R1_opener'].values()]} "
              f"listen max={u['R2_listening']['max_bos_logp']}@{u['R2_listening']['max_bos_frame']} "
              f"argmaxBOS={u['R2_listening']['bos_argmax_frames']} pause={u['R2_listening']['pause_bos_logp_max']} "
              f"SOTC-PAD={u['R3_call']['sotc_minus_pad_at_peak']:+.2f}@{u['R3_call']['sotc_peak_rel']:+d} "
              f"slots={slots} {u['R4_slots']['provenance']} ({u['s']}s)", flush=True)
    save()
    print("DONE", round(time.monotonic() - T0, 1), flush=True)


if __name__ == "__main__":
    main()
