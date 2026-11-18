"""Batched decode of qwen_own: per-step time at batch 1..16 with the one-row GEMV kernel vs the multi-row one, and
greedy-token equality between the two (same prompts, fresh prefills).

usage: python qwen_batch_bench.py --model DIR --out out.json [--mode int8] [--steps 24]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROMPTS = [
    "Buongiorno, vorrei sapere quando arriva il mio ordine. Mi chiamo Mario Rossi e il mio cap è 10121.",
    "Hello, I am looking for a three-seat sofa, what do you have?",
    "Bonjour, je voudrais suivre ma commande, s'il vous plaît.",
    "Puoi prenotarmi un tavolo al ristorante per stasera?",
]


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="int8")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from frankenstein2.llm.qwen_own import kernels
    from frankenstein2.llm.qwen_own.engine import QwenOwn
    torch.set_grad_enabled(False)
    # decode-attention kernel vs the gather + SDPA reference on random caches (GQA 16/4 and 16/2, ragged lengths)
    torch.manual_seed(0)
    chk = {}
    for kvh in (4, 2):
        K = torch.randn(9, kvh, 700, 256, dtype=torch.bfloat16, device="cuda")
        V = torch.randn_like(K)
        q = torch.randn(6, 16, 256, dtype=torch.bfloat16, device="cuda")
        slots = torch.tensor([3, 0, 8, 5, 1, 7], device="cuda")
        lens = torch.tensor([1, 37, 700, 64, 33, 512], device="cuda")
        kernels.DECODE_ATTN = True
        a_ = kernels.decode_attention(q, K, V, slots, lens)
        kernels.DECODE_ATTN = False
        b_ = kernels.decode_attention(q, K, V, slots, lens)
        chk[f"kvh{kvh}"] = round(float((a_.float() - b_.float()).abs().max()), 4)
    kernels.DECODE_ATTN = True
    print("decode_attention max abs diff vs reference", chk, flush=True)
    eng = QwenOwn(a.model, mode=a.mode, slots=16, max_len=4096, graphs=True)
    sys_msg = {"role": "system", "content": "You are a helpful voice agent for a furniture store. Answer briefly."}

    def fresh(n):
        for s in list(eng.sess):
            eng.close(s)
        sess, first = [], []
        for i in range(n):
            s = eng.open()
            ids = eng.chat_ids([sys_msg, {"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
            logits = eng.prefill(s, ids)
            sess.append(s)
            first.append(int(logits.reshape(-1, logits.shape[-1])[-1].argmax()))
        return sess, first

    def greedy(n, steps):
        sess, toks = fresh(n)
        out = [[t] for t in toks]
        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            logits = eng.decode_batch(sess, toks)
            toks = logits.argmax(-1).tolist()
            torch.cuda.synchronize()
            times.append(time.monotonic() - t0)
            for o, t in zip(out, toks):
                o.append(t)
        times = sorted(times[2:])                                   # first steps include graph capture
        return out, round(1e3 * times[len(times) // 2], 2)

    res = {"mode": a.mode, "batches": {}, "attn_check": chk}
    for tag in ("one_row", "rows", "rows_attn"):
        kernels.ROWS_KERNEL = tag != "one_row"
        kernels.DECODE_ATTN = tag == "rows_attn"
        eng.graphs = {k: v for k, v in eng.graphs.items() if k == "pool"}   # recapture with these kernels
        for B in (1, 2, 4, 8, 10, 12, 16):
            toks, ms = greedy(B, a.steps)
            res["batches"].setdefault(str(B), {})[tag] = {"step_ms": ms, "tokens": toks}
            print(tag, B, ms, flush=True)
    # shared system-prompt prefix: copied state vs full prefill must give the same greedy tokens, and be faster
    pre = eng.shared_prefix(sys_msg)
    full = eng.chat_ids([sys_msg, {"role": "user", "content": PROMPTS[0]}])
    res["prefix"] = {"prefix_len": len(pre), "prompt_len": len(full), "is_prefix": full[:len(pre)] == list(pre)}
    if res["prefix"]["is_prefix"]:
        eng.add_prefix(pre)
        toks, ms = greedy(4, a.steps)
        res["prefix"]["tokens_equal"] = toks == res["batches"]["4"]["rows_attn"]["tokens"]
        res["prefix"]["hits"] = eng.prefix_hits
        for with_prefix in (False, True):
            saved = eng.prefixes
            eng.prefixes = saved if with_prefix else []
            s = eng.open()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            eng.prefill(s, full)
            torch.cuda.synchronize()
            res["prefix"]["prefill_ms_" + ("copy" if with_prefix else "full")] = round(1e3 * (time.monotonic() - t0), 1)
            eng.close(s)
            eng.prefixes = saved
        print("prefix", json.dumps(res["prefix"]), flush=True)
    for B, r in res["batches"].items():
        r["tokens_equal"] = r["one_row"]["tokens"] == r["rows"]["tokens"]
        r["attn_tokens_equal"] = r["rows"]["tokens"] == r["rows_attn"]["tokens"]
        r["speedup"] = round(r["one_row"]["step_ms"] / r["rows_attn"]["step_ms"], 2)
        # first divergence step per row (numerics: greedy picks can flip at near-ties)
        div = [next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), None)
               for a_, b_ in zip(r["one_row"]["tokens"], r["rows_attn"]["tokens"])]
        r["first_divergence"] = div
        print("B", B, "rows_eq", r["tokens_equal"], "attn_eq", r["attn_tokens_equal"], "speedup", r["speedup"],
              "div", div, flush=True)
    # where a batch-10 decode step goes (eager, so the profiler sees the kernels)
    from torch.profiler import ProfilerActivity, profile
    sess, toks = fresh(10)
    eng.use_graphs = False
    for _ in range(3):
        toks = eng.decode_batch(sess, toks).argmax(-1).tolist()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            toks = eng.decode_batch(sess, toks).argmax(-1).tolist()
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(table, flush=True)
    res["profile_b10"] = table
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
