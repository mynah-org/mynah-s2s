"""Milestone 1 — Thinker with a REAL batch dimension 2: one forward over [2, 1, H] with shared weights and two
independent recurrent states (Mamba conv/SSM + attention K/V), checked row by row against two B=1 runs.

Isolated prototype; the production B=1 path is untouched (ThinkerGraph is subclassed, not modified).
1. Record: two live sessions on different audio (A, B) through the unmodified engine (BF16 + mamba_host_pos). At decode
   call K_A (resp. K_B, different → different positions) the session's hybrid cache is deep-copied and the next M real
   Thinker inputs (fused audio+text embeddings the wrapper fed) are recorded, with the live text argmax.
2. Reference B=1: from each snapshot, M teacher-forced steps through the upstream eager path → logits + final state.
   (Sanity: text argmax must equal the live run.)
3. B=2: `BatchedThinker` stacks the two states on the batch dim: Mamba conv/SSM [2, ...] (upstream decode kernels are
   batched), attention K/V in static [n_attn, 2, kvh, Lmax, hd] buffers written at a PER-ROW position (advanced-index
   scatter) with a PER-ROW additive mask (no RoPE in NemotronH attention, so positions only live in K/V + mask).
   One forward per step with input [2, 1, H]. Also B=1 through the same path (separates batching from path numerics).
4. Compare per row: text/function/ASR argmax agreement, max |Δ logits|, final conv/SSM and K/V prefix vs reference.
5. Time (CUDA events, median): eager B=1 vs B=2 step; CUDA-graph replay B=1 vs B=2; peak memory.
Writes <out>/thinker_b2.json (immutable run dir).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics as S
import sys
import time
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.model.thinker_graph import ThinkerGraph, bucket_for
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402

HEADS = ("text_logits", "function_logits", "asr_logits")


# ------------------------------------------------------------------------------------------------ batched Thinker
class BatchedThinker(ThinkerGraph):
    """ThinkerGraph buffers generalised to B rows with per-row positions. Prototype only."""

    def load_rows(self, caches):
        b = len(caches)
        ts = [int(c._v2_pos) for c in caches]
        self.K = torch.zeros(len(self.attn), b, self.kvh, self.lmax, self.hd, dtype=self.dtype, device=self.device)
        self.V = torch.zeros_like(self.K)
        self.mask = torch.full((b, 1, 1, self.lmax), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        for r, (c, t) in enumerate(zip(caches, ts)):
            for j, li in enumerate(self.attn):
                assert c.key_cache[li].shape[-2] == t, (c.key_cache[li].shape, t)
                self.K[j, r, :, :t] = c.key_cache[li][0]
                self.V[j, r, :, :t] = c.value_cache[li][0]
            self.mask[r, ..., :t] = 0.0
        cc = copy.copy(caches[0])                          # shallow: then give it its own lists
        # attention K/V live in self.K/self.V (patched update); the cache lists only keep empty placeholders
        cc.key_cache = [torch.zeros(b, 0, device=self.device) for _ in caches[0].key_cache]
        cc.value_cache = [torch.zeros(b, 0, device=self.device) for _ in caches[0].value_cache]
        cc.conv_states = [torch.cat([c.conv_states[i] for c in caches], 0) if i in self.mamba else caches[0].conv_states[i]
                          for i in range(len(caches[0].conv_states))]
        cc.ssm_states = [torch.cat([c.ssm_states[i] for c in caches], 0) if i in self.mamba else caches[0].ssm_states[i]
                         for i in range(len(caches[0].ssm_states))]
        cc.update = self._update_rows
        cc._v2_pos = max(ts)                               # only used as "> 0" (decode) by the Mamba path
        self.rows = torch.arange(b, device=self.device)
        self.posv = torch.tensor(ts, dtype=torch.long, device=self.device)
        self.host_pos = list(ts)
        self.cc = cc
        return cc

    def _update_rows(self, key_states, value_states, layer_idx, *a, **kw):
        j = self.attn.index(layer_idx)
        self.K[j][self.rows, :, self.posv, :] = key_states[:, :, 0, :].to(self.K.dtype)
        self.V[j][self.rows, :, self.posv, :] = value_states[:, :, 0, :].to(self.V.dtype)
        return self.K[j][:, :, :self.bucket], self.V[j][:, :, :self.bucket]

    def prep(self):
        """Host bookkeeping + mask before a step (outside any graph)."""
        self.mask[self.rows, 0, 0, self.posv] = 0.0
        self.bucket = bucket_for(max(self.host_pos) + 1, self.lmax)

    def advance(self):
        self.posv += 1
        self.host_pos = [t + 1 for t in self.host_pos]

    def step(self, x):
        self.prep()
        self.graph_mode = True
        try:
            out = self._orig_forward(x, cache=self.cc)
        finally:
            self.graph_mode = False
        self.advance()
        return out


# ------------------------------------------------------------------------------------------------ helpers
def snapshot(c):
    s = copy.deepcopy(c)
    s._v2_pos = int(c._v2_pos)
    return s


def heads_of(out):
    return {h: out[h].float() for h in HEADS if out.get(h) is not None}


def cuda_ms(fn, iters):
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(S.median(ts), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_a")
    ap.add_argument("wav_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-a", type=int, default=60)
    ap.add_argument("--k-b", type=int, default=100)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--lmax", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    perf.enable("mamba_host_pos", eng)
    stt = eng.wrapper.model.stt_model
    orig = stt.forward

    # ---------------- 1. record two live sessions
    rec = {}

    def record(name, wav, k):
        st = {"n": 0, "snap": None, "inputs": [], "live_text": []}

        def fwd(input_embeds, cache=None, **kw):
            dec = cache is not None and input_embeds.dim() == 3 and input_embeds.shape[1] == 1 and not kw
            if dec and st["n"] == k:
                st["snap"] = snapshot(cache)
            o = orig(input_embeds, cache=cache, **kw)
            if dec:
                if k <= st["n"] < k + a.m:
                    st["inputs"].append(input_embeds.detach().clone())
                    st["live_text"].append(int(o["text_logits"][0, -1].argmax()))
                st["n"] += 1
            return o
        stt.forward = fwd
        x = torch.cat([load_16k(wav), torch.zeros(16000 * 20)])
        eng.open("You are a helpful assistant.")
        for c in iter_chunks(x, eng.chunk_samples):
            eng.step(c)
            if len(st["inputs"]) >= a.m:
                break
        eng.close()
        stt.forward = orig
        if st["snap"] is None or len(st["inputs"]) < a.m:
            raise RuntimeError(f"{name}: not enough decode steps ({st['n']})")
        rec[name] = st

    record("A", a.wav_a, a.k_a)
    record("B", a.wav_b, a.k_b)
    torch.cuda.synchronize()

    # ---------------- 2. reference B=1 (upstream eager path)
    ref = {}
    for name, st in rec.items():
        c = snapshot(st["snap"])
        outs = []
        with torch.inference_mode():
            for x in st["inputs"]:
                outs.append(heads_of(orig(x, cache=c)))
        ref[name] = {"outs": outs, "cache": c,
                     "text_vs_live": sum(int(o["text_logits"][0, -1].argmax()) == lt
                                         for o, lt in zip(outs, st["live_text"]))}

    # ---------------- 3. batched path: B=2, and B=1 through the same path
    bt = BatchedThinker(eng, lmax=a.lmax)
    stt.forward = orig                                      # ThinkerGraph patched it: prototype drives _orig_forward

    def run_batched(names):
        caches = [snapshot(rec[n]["snap"]) for n in names]
        bt.load_rows(caches)
        outs = []
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        with torch.inference_mode():
            for i in range(a.m):
                x = torch.cat([rec[n]["inputs"][i] for n in names], 0)
                outs.append(heads_of(bt.step(x)))
        torch.cuda.synchronize()
        return outs, (torch.cuda.max_memory_allocated() - m0) / 2**20

    res = {"positions": {n: int(rec[n]["snap"]._v2_pos) for n in rec}, "m": a.m, "lmax": a.lmax,
           "sanity_ref_text_vs_live": {n: f"{ref[n]['text_vs_live']}/{a.m}" for n in ref}}
    for label, names in (("B1_A", ["A"]), ("B1_B", ["B"]), ("B2", ["A", "B"])):
        outs, peak_mb = run_batched(names)
        r = {"peak_extra_MB": round(peak_mb, 1)}
        for row, n in enumerate(names):
            agree = {h: sum(int(o[h][row, -1].argmax()) == int(rf[h][0, -1].argmax())
                            for o, rf in zip(outs, ref[n]["outs"]) if h in o and h in rf) for h in HEADS}
            maxd = max(float((o["text_logits"][row] - rf["text_logits"][0]).abs().max())
                       for o, rf in zip(outs, ref[n]["outs"]))
            # final Mamba state + K/V prefix vs the reference cache
            rc = ref[n]["cache"]
            ssm = max(float((bt.cc.ssm_states[i][row].float() - rc.ssm_states[i][0].float()).abs().max()) for i in bt.mamba)
            conv = max(float((bt.cc.conv_states[i][row].float() - rc.conv_states[i][0].float()).abs().max()) for i in bt.mamba)
            t = rc.key_cache[bt.attn[0]].shape[-2]
            kv = max(float((bt.K[j, row, :, :t].float() - rc.key_cache[li][0].float()).abs().max())
                     for j, li in enumerate(bt.attn))
            r[n] = {"argmax_agree": {h: f"{v}/{a.m}" for h, v in agree.items()}, "text_logits_max_abs_diff": round(maxd, 4),
                    "final_ssm_max_abs_diff": round(ssm, 5), "final_conv_max_abs_diff": round(conv, 5),
                    "final_K_prefix_max_abs_diff": round(kv, 5)}
        res[label] = r
        print(label, json.dumps(r), flush=True)

    # ---------------- 4. timing: eager and CUDA graph, B=1 vs B=2
    def time_case(names):
        caches = [snapshot(rec[n]["snap"]) for n in names]
        bt.load_rows(caches)
        x = torch.cat([rec[n]["inputs"][0] for n in names], 0)
        with torch.inference_mode():
            for _ in range(3):
                bt.step(x)
            eager = cuda_ms(lambda: bt.step(x), a.iters)
            # graph: capture one step at the current bucket; replay with the mask/pos bookkeeping outside
            static_x = x.clone()
            bt.prep()
            bt.graph_mode = True
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                bt._orig_forward(static_x, cache=bt.cc)
            torch.cuda.current_stream().wait_stream(s)
            bt.advance()
            bt.prep()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                bt._orig_forward(static_x, cache=bt.cc)
            bt.graph_mode = False
            bt.advance()

            def rep():
                bt.prep()
                g.replay()
                bt.advance()
            graph = cuda_ms(rep, a.iters)
        return eager, graph

    t1e, t1g = time_case(["A"])
    t2e, t2g = time_case(["A", "B"])
    res["timing_ms_per_step"] = {"B1_eager": t1e, "B2_eager": t2e, "B1_graph": t1g, "B2_graph": t2g,
                                 "throughput_gain_eager": round(2 * t1e / t2e, 2),
                                 "throughput_gain_graph": round(2 * t1g / t2g, 2)}
    res["kv_buffers_MB"] = round(2 * bt.K.numel() * bt.K.element_size() / 2**20, 1)
    res["gpu"] = torch.cuda.get_device_name()
    (out / "thinker_b2.json").write_text(json.dumps(res, indent=1))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k not in ("B1_A", "B1_B")}), flush=True)


if __name__ == "__main__":
    main()
