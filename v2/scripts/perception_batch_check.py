"""GPU equivalence check of the batched cache-aware perception (model/perception_batch.py) on the real model.

One process, one model load, EN no-tool items (no FC: the perception cache is never reset), unpaced steps.
  R  record: each item alone through the unchanged engine (B=1 upstream graph step); every perception call's inputs
     (audio buffer, frame_idx) and outputs (encoded, raw) are kept; the first item is run twice (reproducibility).
  O  offline, perception only, caches chained from the recorded initial state:
     eager_b1  — BatchedPerception with one row (eager vs the B=1 graph: kernel effect alone);
     copies    — item 0 x N in one batch (rows must be equal to each other; batch-size effect);
     mixed     — all items in lockstep;
     late      — item 1 joins item 0 ten steps late (first-chunk and subsequent rows, different buffer fills).
     Reported: max |diff| of encoded / raw per step (max over steps) vs the recorded graph output and vs eager_b1.
     graph_*   — the same lanes through CUDA graphs per row count (pre-captured here, quiet) vs the eager batch.
  TIMING  one steady-state perception step of n sessions (host ms incl. sync, p50 of 30): upstream B=1 graph x n vs
     batched eager vs batched graph.
  T  tokens: each item re-run through the engine with its perception step replaced by a batched call together with
     the other items' recorded calls at the same step index (companion caches chained) → text / function ids vs R;
     T (eager batch) and T_graph (graph batch).
Writes <out>/perception_batch_check.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mynah_v2.model.perception_batch import BatchedPerception, step_kwargs
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k, render_prompt  # noqa: E402


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def mad(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--items", default="q12,q09,q01,s05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--tail-s", type=float, default=2.0)
    ap.add_argument("--max-chunks", type=int, default=120)
    ap.add_argument("--batch-mel", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    items = a.items.split(",")
    t_start = time.time()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    pcm = eng.wrapper.perception_cache_mgr
    assert pcm is not None and pcm.use_cudagraph, "expects the perception cache with its B=1 CUDA graphs"
    orig = pcm.step
    bp = BatchedPerception(pcm, batch_mel=bool(a.batch_mel))
    prompt = render_prompt([])
    res = {"items": items, "bdims": list(map(str, bp.bdims)), "streaming_cfg": {
        k: str(getattr(pcm.streaming_cfg, k)) for k in ("chunk_size", "shift_size", "pre_encode_cache_size",
                                                         "drop_extra_pre_encoded")},
        "preprocessor_normalize": str(getattr(getattr(pcm.preprocessor, "featurizer", None), "normalize", "?")),
        "batch_mel": bool(a.batch_mel), "gpu": torch.cuda.get_device_name()}

    def run_item(item, step_fn=None):
        wav = torch.cat([load_16k(Path(a.audio_root) / "corpus_v1" / f"{item}.wav"), torch.zeros(int(a.tail_s * 16000))])
        recs = []

        def rec_step(*args, **kw):
            d = step_kwargs(args, kw)
            init = None if recs else (d["perception_cache"].cache_last_channel.clone(),
                                      d["perception_cache"].cache_last_time.clone(),
                                      d["perception_cache"].cache_last_channel_len.clone())
            o = (step_fn or orig)(*args, **kw)
            recs.append({"audio": d["audio_input"].detach().clone(), "frame_idx": int(d["frame_idx"]),
                         "nf": int(d["num_frames_per_chunk"]), "enc": o[0].detach().clone(),
                         "raw": o[2].detach().clone(), "init": init, "k": len(recs)})
            return o
        pcm.step = rec_step
        try:
            seed_all(1234)
            eng.open(prompt)
            text, func = [], []
            for k, c in enumerate(iter_chunks(wav, eng.chunk_samples)):
                if k >= a.max_chunks:
                    break
                o = eng.step(c)
                text += list(o.new_text_ids)
                func += list(o.new_func_ids)
            eng.close()
        finally:
            del pcm.step
        return recs, text, func

    # ------------------------------------------------------------------ R
    rec, toks = {}, {}
    with torch.no_grad():
        for it in items:
            rec[it], t, f = run_item(it)
            toks[it] = (t, f)
        _, t2, f2 = run_item(items[0])
    res["R"] = {it: {"perception_calls": len(rec[it]), "text_len": len(toks[it][0]),
                     "frame_idx_first": [r["frame_idx"] for r in rec[it][:3]],
                     "audio_len_first_last": [rec[it][0]["audio"].shape[1], rec[it][-1]["audio"].shape[1]]}
                for it in items}
    res["R_repro_item0"] = {"text_identical": t2 == toks[items[0]][0], "func_identical": f2 == toks[items[0]][1]}
    print("R", json.dumps(res["R"]), res["R_repro_item0"], flush=True)

    # ------------------------------------------------------------------ O
    cls = type(pcm.get_initial_state())

    def init_state(it):
        c = rec[it][0]["init"]
        return cls(cache_last_channel=c[0].clone(), cache_last_time=c[1].clone(), cache_last_channel_len=c[2].clone())

    def chained(lanes, bp=bp):
        """lanes: list of (item, start_step). Steps all lanes in lockstep (a lane joins at its start, leaves at its
        end). Returns per lane a list of (enc, raw)."""
        st = [init_state(it) for it, _ in lanes]
        outs = [[] for _ in lanes]
        T = max(s + len(rec[it]) for it, s in lanes)
        for g in range(T):
            idx = [i for i, (it, s) in enumerate(lanes) if 0 <= g - s < len(rec[it])]
            calls = []
            for i in idx:
                it, s = lanes[i]
                r = rec[it][g - s]
                calls.append(dict(audio_input=r["audio"], frame_idx=r["frame_idx"], num_frames_per_chunk=r["nf"],
                                  perception_cache=st[i]))
            got = bp.run(calls)
            for i, (e, c, raw) in zip(idx, got):
                outs[i].append((e, raw))
                st[i] = c
        return outs

    def vs_rec(it, seq, start=0):
        return {"enc_max": max(mad(e, rec[it][k]["enc"]) for k, (e, _) in enumerate(seq)),
                "raw_max": max(mad(r, rec[it][k]["raw"]) for k, (_, r) in enumerate(seq)),
                "enc_ref_absmax": max(float(rec[it][k]["enc"].abs().max()) for k in range(len(seq)))}

    def vs_seq(s1, s2):
        return {"enc_max": max(mad(x[0], y[0]) for x, y in zip(s1, s2)),
                "raw_max": max(mad(x[1], y[1]) for x, y in zip(s1, s2)),
                "bit_identical": all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(s1, s2))}

    O = {}
    with torch.no_grad():
        b1 = {it: chained([(it, 0)])[0] for it in items}
        O["eager_b1_vs_graph"] = {it: vs_rec(it, b1[it]) for it in items}
        cp = chained([(items[0], 0)] * a.copies)
        O[f"copies{a.copies}_rows_equal"] = all(vs_seq(cp[0], cp[i])["bit_identical"] for i in range(1, a.copies))
        O[f"copies{a.copies}_vs_eager_b1"] = vs_seq(cp[0], b1[items[0]])
        O[f"copies{a.copies}_vs_graph"] = vs_rec(items[0], cp[0])
        mx = chained([(it, 0) for it in items])
        O["mixed_vs_eager_b1"] = {it: vs_seq(mx[i], b1[it]) for i, it in enumerate(items)}
        O["mixed_vs_graph"] = {it: vs_rec(it, mx[i]) for i, it in enumerate(items)}
        if len(items) > 1:
            lt = chained([(items[0], 0), (items[1], 10)])
            O["late_vs_eager_b1"] = {items[0]: vs_seq(lt[0], b1[items[0]]), items[1]: vs_seq(lt[1], b1[items[1]])}
        # graphs per row count: pre-captured here (engine closed, no FC: quiet), then the same lanes
        nmax = max(a.copies, len(items))
        bpg = BatchedPerception(pcm, batch_mel=bool(a.batch_mel), graphs=True)
        O["graph_precapture"] = bpg.precapture(nmax)
        cpg = chained([(items[0], 0)] * a.copies, bp=bpg)
        O[f"graph_copies{a.copies}_vs_eager_copies"] = vs_seq(cpg[0], cp[0])
        O[f"graph_copies{a.copies}_rows_equal"] = all(vs_seq(cpg[0], cpg[i])["bit_identical"]
                                                      for i in range(1, a.copies))
        mxg = chained([(it, 0) for it in items], bp=bpg)
        O["graph_mixed_vs_eager_mixed"] = {it: vs_seq(mxg[i], mx[i]) for i, it in enumerate(items)}
        O["graph_mixed_vs_graph_b1"] = {it: vs_rec(it, mxg[i]) for i, it in enumerate(items)}
    sj = lambda st: {k: (v if not isinstance(v, dict) else {str(x): y for x, y in v.items()}) for k, v in st.items()}
    O["bp_stats"], O["bpg_stats"] = sj(bp.stats), sj(bpg.stats)
    res["O"] = O
    print("O", json.dumps(O), flush=True)

    # ------------------------------------------------------------------ timing (per perception step of n sessions)
    # steady-state call (recorded step 20 of item 0 or its last), caches from the recorded initial state (same shapes)
    def timeit(fn, reps=30, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        xs.sort()
        return {"p50": round(xs[len(xs) // 2], 2), "min": round(xs[0], 2)}
    r0 = rec[items[0]][min(20, len(rec[items[0]]) - 1)]
    mk = lambda: dict(audio_input=r0["audio"], frame_idx=r0["frame_idx"], num_frames_per_chunk=r0["nf"],
                      perception_cache=init_state(items[0]))
    TM = {}
    with torch.no_grad():
        for n in range(1, nmax + 1):
            calls = [mk() for _ in range(n)]
            TM[n] = {"upstream_b1_graph_x_n": timeit(lambda: [orig(**c) for c in calls]),
                     "batched_eager": timeit(lambda: bp.run(calls))}
            if n >= 2:
                TM[n]["batched_graph"] = timeit(lambda: bpg.run(calls))
    res["timing_ms"] = TM
    print("TIMING", json.dumps(TM), flush=True)

    # ------------------------------------------------------------------ T
    def tokens(bp):
      T = {}
      with torch.no_grad():
        for it in items:
            comp = [x for x in items if x != it]
            cst = {x: init_state(x) for x in comp}
            k = {"n": 0}

            def batched_step(*args, **kw):
                d = step_kwargs(args, kw)
                g = k["n"]
                k["n"] += 1
                lanes = [x for x in comp if g < len(rec[x])]
                calls = [d] + [dict(audio_input=rec[x][g]["audio"], frame_idx=rec[x][g]["frame_idx"],
                                    num_frames_per_chunk=rec[x][g]["nf"], perception_cache=cst[x]) for x in lanes]
                if len(calls) == 1:
                    return orig(**d)
                got = bp.run(calls)
                for x, o in zip(lanes, got[1:]):
                    cst[x] = o[1]
                return got[0]
            _, t, f = run_item(it, step_fn=batched_step)
            rt, rf = toks[it]
            fd = next((i for i, (x, y) in enumerate(zip(t, rt)) if x != y), None if len(t) == len(rt) else min(len(t), len(rt)))
            T[it] = {"text_identical": t == rt, "func_identical": f == rf, "first_text_divergence": fd,
                     "text_len": len(t), "batched_steps": k["n"]}
      return T
    for tag, b in (("T", bp), ("T_graph", bpg)):
        T = tokens(b)
        res[tag] = T
        res[tag + "_identical"] = f"{sum(v['text_identical'] and v['func_identical'] for v in T.values())}/{len(T)}"
        print(tag, json.dumps(T), res[tag + "_identical"], flush=True)
    res["bpg_stats_end"] = sj(bpg.stats)
    res["wall_s"] = round(time.time() - t_start, 1)
    (out / "perception_batch_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
