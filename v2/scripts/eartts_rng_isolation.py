"""GB10 smoke: EarTTS per-session RNG + row isolation on the real model (no audio pipeline, no sessions' Thinker).

Part 1 (no model): a CUDA graph containing eartts_rng noise replays NEW samples when the per-row (seed, step) buffer
advances, the SAME samples for the same (seed, step), and the same as eager.
Part 2 (model, per dtype fp32 then bf16): three EarTTS "sessions" with different histories (DynamicCache prefill
through the upstream step) in pooled slots; teacher-forced inputs; --eartts-rng session on. Per-session B=1 static
path vs BatchedEarTTS in several row orders and with a neighbour swapped:
  * eager (no graphs): backbone hidden states (cond rows) max |diff| and code equality  → row isolation;
  * graphs (B=1 lazy captures + batched graphs): code equality B=1 vs batched → per-session RNG under replay.
Writes one JSON line per check to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def part1():
    from mynah_v2.model import eartts_rng as ER
    ER.STATE["enabled"], ER.STATE["orig_gumbel"] = True, None
    dev = torch.device("cuda")
    buf = torch.zeros(2, 2, dtype=torch.long, device=dev)
    x = torch.zeros(2, 1, 64, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), ER.rows(buf):
        ER.randn_like(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), ER.rows(buf):
        out = ER.randn_like(x)

    def rep(seed_step):
        buf.copy_(torch.tensor(seed_step))
        g.replay()
        return out.clone()
    r0 = rep([[5, 0], [6, 0]])
    r1 = rep([[5, 1], [6, 1]])
    r0b = rep([[5, 0], [6, 0]])
    with ER.rows(torch.tensor([[5, 1]], device=dev)):
        eager1 = ER.randn_like(x[:1])
    steps = [rep([[5, k], [6, k]]) for k in range(20)]
    distinct = len({tuple(t[0].flatten().tolist()[:8]) for t in steps})
    ER.STATE["enabled"] = False
    return {"check": "graph_replay_rng", "advance_differs": not torch.equal(r0, r1),
            "same_seed_step_same": torch.equal(r0, r0b), "replay_equals_eager": torch.equal(r1[:1], eager1),
            "rows_differ": not torch.equal(r0[0], r0[1]), "distinct_of_20_replays": distinct}


def part2(dtypes, steps):
    from mynah_v2.model import eartts_rng as ER
    from mynah_v2.model import perf
    from mynah_v2.model.capture_guard import GUARD
    from mynah_v2.model.eartts_batch import BatchedEarTTS
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    eng = ReferenceEngine(build_cfg(output_dir="/tmp/_eartts_iso_nemo", overrides={"s2s.deterministic": True}))
    perf.apply_profile("optimized-graph-s4", eng)
    ER.enable(eng)
    fsg = eng._v2_full_graph
    dt = fsg.dt
    gen = dt._get_generation_config(guidance_enabled=True)
    dev = fsg.device
    ids = [i for i in eng.tokenizer.text_to_ids("hello there, this is a short test of the speech system and its "
                                                "voice for several sessions") if i > 3]
    c = int(dt.codec_silence_tokens.numel())
    hid = []
    orig_bb = fsg.bb_forward

    def bb_rec(*a, **kw):
        out = orig_bb(*a, **kw)
        hid.append(out.last_hidden_state.detach().float())
        return out
    fsg.bb_forward = bb_rec
    res = []
    for mode in dtypes:
        info = fsg.set_mode(mode) if mode != "fp32" else None
        be = BatchedEarTTS(fsg, graphs=True)
        g = torch.Generator().manual_seed(0)

        def inp(k):
            t = ids[k % len(ids)]
            return dict(current_subword_id=torch.tensor([[t]], device=dev),
                        prev_subword_id=torch.tensor([[ids[(k + 3) % len(ids)]]], device=dev),
                        current_subword_mask=torch.tensor([[bool(k % 3)]], device=dev),
                        prev_audio_tokens=torch.randint(0, 200, (1, 1, c), generator=g).to(dev),
                        guidance_enabled=True, generation_config=gen, ignore_eos_flag_stop=True)
        sess = {}
        with torch.no_grad():
            for name, (seed, pre) in {"A": (101, 30), "B": (202, 90), "C": (303, 150)}.items():
                pkv = None
                for k in range(pre):
                    _, pkv = fsg.eager_step(**inp(k + seed), past_key_values=pkv)
                ER.CTX.seed = seed
                kv = fsg._convert(pkv, tuple(inp(0)[a] for a in ("current_subword_id", "prev_subword_id",
                                                                    "current_subword_mask", "prev_audio_tokens")),
                                  {"guidance_enabled": True, "generation_config": gen, "ignore_eos_flag_stop": True})
                ER.CTX.seed = None
                sess[name] = kv
            ins = {n: [inp(1000 + 37 * i + k) for k in range(steps)] for i, n in enumerate(sess)}
            snap = ([t.clone() for t in (fsg.pool_k, fsg.pool_v, fsg.pool_mask)],
                    {n: (kv.t, kv._v2_rng_step) for n, kv in sess.items()})

            def restore():
                for t, cc in zip((fsg.pool_k, fsg.pool_v, fsg.pool_mask), snap[0]):
                    t.copy_(cc)
                for n, kv in sess.items():
                    kv.t, kv._v2_rng_step = snap[1][n]

            def b1(n, graphs):
                saved = GUARD.may_capture
                if not graphs:
                    GUARD.may_capture = lambda *a: False
                try:
                    codes, hs = [], []
                    for k in range(steps):
                        hid.clear()
                        cd, _ = fsg.step(**ins[n][k], past_key_values=sess[n])
                        codes.append(cd.clone())
                        hs.append(hid[-1][0].clone() if hid else None)
                    return codes, hs
                finally:
                    GUARD.may_capture = saved

            def batch(order, graphs):
                be.use_graphs = graphs
                codes = {n: [] for n in order}
                hs = {n: [] for n in order}
                for k in range(steps):
                    hid.clear()
                    outs = be.run([dict(ins[n][k], past_key_values=sess[n]) for n in order])
                    for n, (cd, kv) in zip(order, outs):
                        codes[n].append(cd.clone())
                        hs[n].append(hid[-1][kv.idx].clone() if hid else None)
                return codes, hs
            keep = {}
            eq_ = lambda u, v: f"{sum(int(torch.equal(x, y)) for x, y in zip(u, v))}/{steps}"   # noqa: E731
            restore()
            again = b1("A", False)[0]
            for graphs in (False, True):                  # graphs: B=1 and batched keys captured lazily (quiet)
                ref = {}
                for n in sess:
                    restore()
                    ref[n] = b1(n, graphs)
                    keep[(graphs, "b1", n)] = ref[n][0]
                if not graphs:
                    res.append({"check": "b1_eager_repeat_identical", "dtype": mode, "eq": eq_(again, ref["A"][0])})
                else:
                    res.append({"check": "b1_graph_vs_b1_eager", "dtype": mode,
                                "eq": {n: eq_(ref[n][0], keep[(False, "b1", n)]) for n in sess}})
                print(json.dumps(res[-1]), flush=True)
                for order in (["A", "B", "C"], ["C", "A", "B"], ["B", "C"], ["A", "C"]):
                    restore()
                    t0 = time.perf_counter()
                    codes, hs = batch(order, graphs)
                    for n in order:
                        keep[(graphs, "".join(order), n)] = codes[n]
                    if graphs:
                        res.append({"check": "batched_graph_vs_batched_eager", "dtype": mode, "order": "".join(order),
                                    "eq": {n: eq_(codes[n], keep[(False, "".join(order), n)]) for n in order}})
                        print(json.dumps(res[-1]), flush=True)
                    for n in order:
                        eq = sum(int(torch.equal(x, y)) for x, y in zip(codes[n], ref[n][0]))
                        rec = {"check": "row_isolation", "dtype": mode, "graphs": graphs, "order": "".join(order),
                               "session": n, "row": sess[n].idx, "codes_equal_steps": f"{eq}/{steps}",
                               "codes_finite_range": all(int(x.min()) >= 0 for x in codes[n]),
                               "ms": round((time.perf_counter() - t0) * 1e3, 1)}
                        if not graphs:
                            d = [float((x - y).abs().max()) for x, y in zip(hs[n], ref[n][1])]
                            rec["hidden_max_abs_diff"] = round(max(d), 6)
                            rec["hidden_finite"] = all(bool(torch.isfinite(x).all()) for x in hs[n])
                        res.append(rec)
                        print(json.dumps(rec), flush=True)
            restore()
            for kv in sess.values():                       # free the slots for the next dtype
                kv._fin()
        res.append({"check": "dtype_info", "dtype": mode, "info": info, "fallback_global": ER.STATE["fallback_global"],
                    "batch_stats": {k: be.stats[k] for k in ("runs", "replays", "eager", "captures")}})
        print(json.dumps(res[-1], default=str), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dtypes", default="fp32,bf16")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        r = part1()
        print(json.dumps(r), flush=True)
        f.write(json.dumps(r) + "\n")
        for rec in part2(a.dtypes.split(","), a.steps):
            f.write(json.dumps(rec, default=str) + "\n")


if __name__ == "__main__":
    main()
