"""Where do the ms of a chunk go? CUDA-event spans per component, no profiler.

torch.profiler does not attribute kernels replayed inside CUDA graphs to labels (K7b: Thinker 0 ms) and inflates
host time (561 vs ~330 ms/chunk). Here each component call records a CUDA event before and after it on the CURRENT
stream (no sync) plus host perf_counter; after each chunk ONE synchronize resolves the events. Per chunk and component:
  host_ms  = host wall of the call;
  gpu_ms   = event span = time the stream spent from reaching the call's first op to finishing its last op. For a
             component that is ONE graph replay (perception with NVIDIA's graph, EarTTS full graph, Thinker graph) this
             is ~pure device work; for an eager component it also includes device idle while the host launches.
  chunk    = host wall of engine.step and the event span of the whole step.
Derived: other_ms = chunk wall - sum(component host ms) (host-only glue: embeddings, RNN-T, turn-taking, syncs);
gpu_sum = sum of component gpu spans (serial on one stream) → device-busy upper bound for graphed components.
Components: perception (perception_cache_mgr.step), thinker (stt_model.forward: backbone + heads; what thinker_graph
replaces), tts (tts_model.infer_codes_one_step), codec (audio_codec.decode). Writes an immutable run directory.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics as S
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402

REC: list = []          # (component, host_ms, ev_start, ev_end) of the current chunk


def timed(fn, name):
    @functools.wraps(fn)
    def w(*a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        try:
            return fn(*a, **k)
        finally:
            e.record()
            REC.append((name, (time.perf_counter() - t0) * 1000.0, s, e))
    return w


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--exp", default="", help="comma list of mynah_v2.model.perf experiments to enable first")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    from mynah_v2.model import perf
    exp_info = {n: perf.enable(n, eng) for n in filter(None, a.exp.split(","))}
    w, m = eng.wrapper, eng.wrapper.model
    pcm = getattr(w, "perception_cache_mgr", None)
    comps = []
    if pcm is not None:
        pcm.step = timed(pcm.step, "perception")
        comps.append("perception")
    m.stt_model.forward = timed(m.stt_model.forward, "thinker")
    m.tts_model.infer_codes_one_step = timed(m.tts_model.infer_codes_one_step, "tts")
    m.tts_model.audio_codec.decode = timed(m.tts_model.audio_codec.decode, "codec")
    comps += ["thinker", "tts", "codec"]

    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 30)])
    chunks = list(iter_chunks(x, eng.chunk_samples))[: a.warmup + a.chunks]
    eng.open("You are a helpful assistant.")
    rows = []
    for i, c in enumerate(chunks):
        torch.cuda.synchronize()
        REC.clear()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        eng.step(c)
        e.record()
        wall = (time.perf_counter() - t0) * 1000.0
        torch.cuda.synchronize()
        if i < a.warmup:
            continue
        r = {"wall_ms": wall, "chunk_gpu_span_ms": s.elapsed_time(e)}
        agg = defaultdict(lambda: [0, 0.0, 0.0])
        for name, h, es, ee in REC:
            g = agg[name]
            g[0] += 1
            g[1] += h
            g[2] += es.elapsed_time(ee)
        for name in comps:
            n, h, g = agg.get(name, [0, 0.0, 0.0])
            r[f"{name}_calls"], r[f"{name}_host_ms"], r[f"{name}_gpu_ms"] = n, h, g
        r["other_host_ms"] = wall - sum(r[f"{n}_host_ms"] for n in comps)
        r["gpu_sum_ms"] = sum(r[f"{n}_gpu_ms"] for n in comps)
        rows.append(r)
    eng.close()

    keys = [k for k in rows[0] if not k.endswith("_calls")]
    summary = {k: {"p50": pct([r[k] for r in rows], 0.5), "p95": pct([r[k] for r in rows], 0.95),
                   "mean": round(S.fmean(r[k] for r in rows), 2)} for k in keys}
    summary.update({f"{n}_calls_per_chunk": round(S.fmean(r[f"{n}_calls"] for r in rows), 2) for n in comps})
    res = {"wav": a.wav, "exp": exp_info and list(exp_info), "warmup": a.warmup, "chunks": len(rows),
           "summary": summary, "rows": rows,
           "note": "gpu_ms = CUDA-event span on the calling stream (pure device work for a single graph replay; "
                   "includes launch idle for eager code). other_host_ms = host glue outside the 4 components."}
    (out / "timeline.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"exp": list(exp_info), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
