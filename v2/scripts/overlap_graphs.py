"""Overlap upper bound with the REAL captured graphs: can EarTTS(t) hide behind Thinker(t+1) on this GPU?

Loads the engine with optimized-graph + thinker_graph, streams --warmup chunks so both graphs are captured on the live
buckets, then times graph replays only (no host logic):
  T  = Thinker step graph (backbone + heads), E = EarTTS full-step graph (one frame, CFG batch 2)
  serial     : T then E on one stream (what the loop does today, per frame)
  concurrent : T on stream 1 ‖ E on stream 2 (upper bound of a side-stream EarTTS one frame behind)
CUDA events, median of --iters. Gain upper bound per frame = serial − concurrent (×2 per chunk). Replays mutate the
graphs' static state (Mamba/KV buffers): the engine is discarded afterwards — measurement only, never served.
"""

from __future__ import annotations

import argparse
import json
import statistics as S
import sys
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def timed(fn, iters):
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(S.median(ts), 3), round(min(ts), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    for n in ("mamba_host_pos", "no_timer_sync", "eartts_graph_full", "thinker_graph"):
        perf.enable(n, eng)
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 30)])
    eng.open("You are a helpful assistant.")
    for c in list(iter_chunks(x, eng.chunk_samples))[: a.warmup]:
        eng.step(c)
    torch.cuda.synchronize()
    tg, fsg = eng._v2_thinker_graph, eng._v2_full_graph
    tgr = [v[0] for v in tg.graphs.values()]
    egr = [v[0] for v in fsg.graphs.values()]
    if not tgr or not egr:
        raise RuntimeError(f"graphs not captured: thinker={len(tgr)} eartts={len(egr)}")
    T, E = tgr[-1], egr[-1]                      # most recent buckets = the live ones
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def serial():
        T.replay()
        E.replay()

    def concurrent():
        cur = torch.cuda.current_stream()
        s1.wait_stream(cur)
        s2.wait_stream(cur)
        with torch.cuda.stream(s1):
            T.replay()
        with torch.cuda.stream(s2):
            E.replay()
        cur.wait_stream(s1)
        cur.wait_stream(s2)

    for f in (T.replay, E.replay, serial, concurrent):         # warm
        f()
    res = {"thinker_ms": timed(T.replay, a.iters), "eartts_ms": timed(E.replay, a.iters),
           "serial_ms": timed(serial, a.iters), "concurrent_ms": timed(concurrent, a.iters),
           "graphs": {"thinker": len(tgr), "eartts": len(egr)}, "gpu": torch.cuda.get_device_name()}
    res["gain_per_frame_ms_upper_bound"] = round(res["serial_ms"][0] - res["concurrent_ms"][0], 3)
    res["gain_per_chunk_ms_upper_bound"] = round(2 * res["gain_per_frame_ms_upper_bound"], 3)
    (out / "overlap.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
