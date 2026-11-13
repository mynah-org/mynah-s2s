"""A/B gate for one performance experiment (``mynah_v2.model.perf``) against the BF16 + cache baseline.

Same process, same engine, same input, deterministic decoding and the same RNG seed before each run, ABBA order so
monotonic drift cancels: baseline B1 → enable → experiment WARM-UP pass (graph captures, autotune: timed and reported
separately, never inside the measured window) → E1 → E2 → disable → B2. Timing = mean of (B1,B2) vs mean of (E1,E2).
Reports per run: per-chunk wall p50/p95 (steady window after --warmup), RTF = sum(wall) / audio duration of the
chunks fed, TTFA (wall time from the first chunk to the first output chunk above -50 dBFS), peak memory; and
experiment vs baseline: text/function/ASR token identity, token and audio LENGTHS, EOS frames, active-audio duration,
and the agent waveform max/mean |diff| over the common prefix. ``bit_exact`` is true only if tokens are identical AND
lengths are equal AND the waveform diff is 0 — a truncated or extended answer can never pass.
Writes an immutable run directory (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as S
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k, render_prompt  # noqa: E402


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(eng, x, prompt, seed, warmup):
    reseed(seed)
    torch.cuda.reset_peak_memory_stats()
    eng.open(prompt)
    chunks = list(iter_chunks(x, eng.chunk_samples))
    walls, audio, ttfa = [], [], None
    t_start = time.perf_counter()
    for i, c in enumerate(chunks):
        t0 = time.perf_counter()
        o = eng.step(c, is_last=(i == len(chunks) - 1))
        walls.append((time.perf_counter() - t0) * 1000)
        a = o.audio.float().cpu().reshape(-1)
        audio.append(a)
        if ttfa is None and a.numel() and float(a.pow(2).mean().sqrt()) > 0.003:
            ttfa = (time.perf_counter() - t_start) * 1000
    gt, ga, n, gf = eng.token_tensors()
    eng.close()
    st = walls[warmup:] or walls
    dur_ms = len(chunks) * eng.chunk_samples / 16
    text = gt[0, :n].tolist()
    aud = torch.cat(audio).numpy()
    fr = 1102                                                     # 50 ms at 22.05 kHz
    active = sum(float(np.sqrt(np.mean(aud[i:i + fr] ** 2))) > 0.003 for i in range(0, len(aud), fr)) * 0.05
    return {"eos_frames": [i for i, t in enumerate(text) if t == 2], "active_audio_s": round(active, 2),
            "n_tokens": int(n), "audio_samples": int(len(aud)), "wall_total_ms": round(sum(walls), 1),
            "rtf_steady_mean": round(S.mean(st) / 160.0, 3), **{"text": gt[0, :n].tolist(), "asr": ga[0, :n].tolist() if ga is not None else None,
            "func": gf[0, :n].tolist() if gf is not None else None, "audio": aud,
            "p50": round(S.median(st), 2), "p95": round(sorted(st)[int(0.95 * (len(st) - 1))], 2),
            "rtf": round(sum(walls) / dur_ms, 3), "rtf_steady": round(S.median(st) / 160.0, 3),
            "ttfa_ms": None if ttfa is None else round(ttfa, 1),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a or [], b or [])):
        if x != y:
            return i
    return None if len(a or []) == len(b or []) else min(len(a or []), len(b or []))


def compare(b, e):
    m = min(len(b["audio"]), len(e["audio"]))
    d = np.abs(b["audio"][:m] - e["audio"][:m]) if m else np.zeros(1)
    r = {"text_identical": b["text"] == e["text"], "func_identical": b["func"] == e["func"],
         "asr_identical": b["asr"] == e["asr"], "first_text_div": first_div(b["text"], e["text"]),
         "first_func_div": first_div(b["func"], e["func"]),
         "n_tokens": [b["n_tokens"], e["n_tokens"]], "audio_len": [len(b["audio"]), len(e["audio"])],
         "audio_len_equal": len(b["audio"]) == len(e["audio"]), "eos_frames": [b["eos_frames"], e["eos_frames"]],
         "eos_identical": b["eos_frames"] == e["eos_frames"],
         "active_audio_s": [b["active_audio_s"], e["active_audio_s"]],
         "audio_max_abs_diff_common": float(d.max()), "audio_mean_abs_diff_common": float(d.mean())}
    r["bit_exact"] = bool(r["text_identical"] and r["func_identical"] and r["asr_identical"] and r["audio_len_equal"]
                          and r["eos_identical"] and r["audio_max_abs_diff_common"] == 0.0)
    return r


TIMING = ("p50", "p95", "rtf", "rtf_steady", "rtf_steady_mean", "ttfa_ms", "peak_alloc_gb", "wall_total_ms")


def mean_of(runs):
    out = {}
    for k in TIMING:
        v = [r[k] for r in runs if r.get(k) is not None]
        out[k] = round(S.mean(v), 3) if v else None
    return out


def public(r):
    return {k: v for k, v in r.items() if k not in ("text", "asr", "func", "audio")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--exp", required=True, help=f"comma list, applied together: {sorted(perf.EXPERIMENTS)}")
    ap.add_argument("--base-exp", default="", help="experiments enabled on BOTH sides (the A/B then measures the "
                    "increment of --exp on top of them, e.g. thinker_w8 on top of optimized-exact)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pad", type=float, default=20.0)
    ap.add_argument("--fc-prompt", action="store_true")
    ap.add_argument("--cudnn-benchmark", action="store_true", help="for both runs (the 301 ms baseline)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="NVIDIA's default (faster) kernels: timing of the serving configuration; parity then only indicative")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": not a.nondeterministic}))
    if a.fc_prompt:
        import offline_voicechat_fc_infer as fcmod
        prompt = render_prompt(fcmod.DEFAULT_TOOLS)
    else:
        prompt = "You are a helpful assistant."
    x = torch.cat([load_16k(a.wav), torch.zeros(int(a.pad * 16000))])
    base_names = [n for n in a.base_exp.split(",") if n]
    base_infos = [perf.enable(n, eng) for n in base_names]       # both sides; warmed by the next run
    run(eng, x, prompt, a.seed, a.warmup)                      # warm everything (cudnn autotune, allocator)
    b1 = run(eng, x, prompt, a.seed, a.warmup)
    names = a.exp.split(",")
    t_en = time.perf_counter()
    infos = [perf.enable(n, eng) for n in names]
    enable_ms = (time.perf_counter() - t_en) * 1000
    warm = run(eng, x, prompt, a.seed, a.warmup)                # captures / autotune happen here, reported apart
    e1 = run(eng, x, prompt, a.seed, a.warmup)
    e2 = run(eng, x, prompt, a.seed, a.warmup)
    for n in reversed(names):
        perf.disable(n, eng)
    b2 = run(eng, x, prompt, a.seed, a.warmup)
    base, exp = b1, e1
    rep = {"deterministic": not a.nondeterministic, "base_exp": base_names, "base_exp_info": base_infos,
           "exp": names, "exp_info": infos, "wav": a.wav, "seed": a.seed,
           "cudnn_benchmark": a.cudnn_benchmark, "order": "B1 [enable, warm-up E] E1 E2 [disable] B2",
           "baseline_mean": mean_of([b1, b2]), "experiment_mean": mean_of([e1, e2]),
           "speedup_p50_ms": round(mean_of([b1, b2])["p50"] - mean_of([e1, e2])["p50"], 2),
           "experiment_setup": {"enable_ms": round(enable_ms, 1), "warmup_run": {k: warm[k] for k in TIMING},
                                "warmup_extra_wall_ms": round(warm["wall_total_ms"] - e1["wall_total_ms"], 1)},
           "runs": {"B1": public(b1), "E1": public(e1), "E2": public(e2), "B2": public(b2)},
           "vs_baseline": compare(b1, e1), "experiment_repeat": compare(e1, e2), "baseline_repeat": compare(b1, b2),
           "warmup_vs_experiment": compare(warm, e1)}
    sf.write(out / "baseline.wav", base["audio"], 22050)
    sf.write(out / "experiment.wav", exp["audio"], 22050)
    tok = eng.tokenizer
    for tag, r in (("baseline", base), ("experiment", exp)):   # internal text + turn frames for the quality gate
        (out / f"{tag}_text.json").write_text(json.dumps({
            "text_ids": r["text"], "frame_s": 0.08,
            "text": tok.ids_to_text([i for i in r["text"] if i > 2 and i != 12])}, ensure_ascii=False))
    (out / "perf_ab.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
