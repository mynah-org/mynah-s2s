"""Thinker scaling microbench on the REAL model with the PRODUCTION batched backend (model/thinker_batch.BatchedThinker):
step time vs batch rows N (1/2/4/8 by default), CUDA graph replay and eager, aggregate throughput and memory per row.

1. One real session through the unmodified engine (prefill + a few chunks of real audio): its hybrid cache after the
   last chunk and the last real Thinker decode input ([1, 1, H] fused embedding) are captured.
2. For each N: a fresh BatchedThinker(max_rows=N), N deep copies of the cache attached as N sessions (same position,
   independent state), then `step({sid_i: x})` timed with CUDA events (warm-up excluded; graphs captured on warm-up).
   Memory: allocated/reserved before/after attach and after the timed steps, backend.memory() (bytes per slot, graph
   pool), torch peak.
This is a Thinker-only throughput/memory bound: perception, EarTTS, codec and scheduling are NOT included (the full
pipeline is measured by the multi-session gate). Writes <out>/thinker_bn.json.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics as S
import sys
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.model.thinker_batch import BatchedThinker
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402

GB = 2**30


def mem():
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    return {"alloc_GiB": round(torch.cuda.memory_allocated() / GB, 3), "reserved_GiB": round(torch.cuda.memory_reserved() / GB, 3),
            "free_GiB": round(free / GB, 3), "peak_GiB": round(torch.cuda.max_memory_allocated() / GB, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="1,2,4,8")
    ap.add_argument("--chunks", type=int, default=40, help="real audio chunks before capturing the cache")
    ap.add_argument("--lmax", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--exp", default="", help="perf experiments applied after the default profile (e.g. wor:w8:mlp)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    perf.enable("mamba_host_pos", eng)
    exp = {n: str(perf.enable(n, eng))[:200] for n in a.exp.split(",") if n}
    stt = eng.wrapper.model.stt_model
    box = {}
    orig = stt.forward

    def spy(input_embeds, cache=None, **kw):
        if input_embeds.dim() == 3 and input_embeds.shape[1] == 1 and not kw:
            box["x"] = input_embeds.detach().clone()
        return orig(input_embeds, cache=cache, **kw)
    stt.forward = spy
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 10)])
    with torch.inference_mode():
        eng.open("You are a helpful assistant.")
        for c in list(iter_chunks(x, eng.chunk_samples))[: a.chunks]:
            eng.step(c)
    stt.forward = orig
    ctx = eng._ctx()
    base_cache, frame = ctx.dynamic_cache, box["x"]
    pos0 = int(getattr(base_cache, "_v2_pos", 0))
    res = {"gpu": torch.cuda.get_device_name(), "exp": exp, "lmax": a.lmax, "pos0": pos0, "iters": a.iters,
           "mem_after_session": mem(), "runs": []}
    for n in [int(v) for v in a.ns.split(",")]:
        for graphs in (True, False):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m0 = mem()
            bt = BatchedThinker(eng, max_rows=n, lmax=a.lmax, graphs=graphs)
            sids = [f"s{i}" for i in range(n)]
            caches = [copy.deepcopy(base_cache) for _ in sids]
            m_copy = mem()
            with torch.inference_mode():
                for s, c in zip(sids, caches):
                    bt.attach(s, c)
                m_att = mem()
                frames = {s: frame for s in sids}
                for _ in range(a.warmup):
                    bt.step(frames)
                ts = []
                for _ in range(a.iters):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    bt.step(frames)
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1))
                m_end = mem()
                bm = bt.memory()
            p50 = S.median(ts)
            r = {"n": n, "graphs": graphs, "step_ms_p50": round(p50, 3), "step_ms_p95": round(sorted(ts)[int(0.95 * len(ts)) - 1], 3),
                 "frames_per_s": round(n * 1000 / p50, 1),
                 "chunk_ms_per_session_equiv": round(2 * p50, 2),          # 2 Thinker frames per 160 ms chunk
                 "mem_before": m0, "mem_cache_copies": m_copy, "mem_attached": m_att, "mem_end": m_end,
                 "backend": {k: v for k, v in vars(bm).items()} if bm is not None else None,
                 "positions_end": dict(bt.positions())}
            res["runs"].append(r)
            print("RUN " + json.dumps({k: r[k] for k in ("n", "graphs", "step_ms_p50", "step_ms_p95", "frames_per_s")}
                                      | {"alloc_end": m_end["alloc_GiB"], "peak": m_end["peak_GiB"]}), flush=True)
            with torch.inference_mode():                   # backend buffers are inference tensors
                for s in sids:
                    bt.detach(s)
                bt.close()
            del bt, caches, frames
    b1 = {r["graphs"]: r["step_ms_p50"] for r in res["runs"] if r["n"] == 1}
    for r in res["runs"]:
        r["speedup_vs_n_b1_steps"] = round(r["n"] * b1[r["graphs"]] / r["step_ms_p50"], 2) if b1.get(r["graphs"]) else None
    (out / "thinker_bn.json").write_text(json.dumps(res, indent=1))
    print("RESULT " + json.dumps([{k: r[k] for k in ("n", "graphs", "step_ms_p50", "frames_per_s", "speedup_vs_n_b1_steps")}
                                  for r in res["runs"]]), flush=True)


if __name__ == "__main__":
    main()
