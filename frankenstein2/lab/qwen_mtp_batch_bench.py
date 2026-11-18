"""Is MTP speculative decoding worth it in BATCHED decode? Upper bound from measured costs, before building it.

Per batch size B (CUDA graphs for all three forwards, same footing):
  t_dec    one plain decode step: B rows x 1 token
  t_ver    the verify forward: B rows x [token, draft] (all logits + hidden, DeltaNet rollback states)
  t_drf    the MTP head on B rows x 1 token (the next draft)
and the acceptance rate measured with the single-session greedy MTP generator on real agent prompts. A batched MTP
step emits (1 + acc) tokens per row for t_ver + t_drf, plain decode 1 token per t_dec:
  speedup_bound(B) = (1 + acc) * t_dec / (t_ver + t_drf)      (rollback copies and host logic not counted)

usage: python qwen_mtp_batch_bench.py --model DIR --mode bf16 --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.lab.qwen_batch_bench import PROMPTS  # noqa: E402

AGENT = ("You are the voice agent of Mynah Home, a furniture store. Answer in the user's language, in one or two short "
         "spoken sentences, no lists, no markdown.")
TURNS = ["Buongiorno, che divani avete a tre posti e quanto costano?", "Hello, can you tell me your opening hours?",
         "Bonjour, je voudrais savoir si vous livrez à domicile.", "Mi spieghi come funziona il reso di un tavolo?",
         "What's the difference between your oak and walnut tables?", "Avete anche letti matrimoniali in legno?"]


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from frankenstein2.llm.qwen_own.engine import QwenOwn
    torch.set_grad_enabled(False)
    eng = QwenOwn(a.model, mode=a.mode, slots=16, max_len=4096, graphs=True, mtp=True)
    res = {"mode": a.mode, "model": Path(a.model).name}
    if eng.model.mtp is None:
        res["error"] = "checkpoint has no MTP head"
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("MTP none", flush=True)
        return
    # acceptance rate on agent replies (single session, the existing greedy MTP generator)
    acc = steps = 0
    for t in TURNS:
        for s in list(eng.sess):
            eng.close(s)
        s = eng.open()
        ids = eng.chat_ids([{"role": "system", "content": AGENT}, {"role": "user", "content": t}])
        for ev in eng.generate_mtp(s, ids, max_new=64, stop=()):
            if ev.get("done"):
                acc += ev["mtp_accepted"]
                steps += ev["mtp_steps"]
    rate = acc / max(steps, 1)
    res["acceptance"] = {"steps": steps, "accepted": acc, "rate": round(rate, 3)}
    print("ACC", json.dumps(res["acceptance"]), flush=True)
    for s in list(eng.sess):
        eng.close(s)
    sess = []
    for i in range(16):
        s = eng.open()
        eng.prefill(s, eng.chat_ids([{"role": "system", "content": AGENT}, {"role": "user", "content": PROMPTS[i % 4]}]))
        sess.append(s)
    dev, H, L = eng.device, eng.cfg.hidden, 1024
    st, m = eng.state, eng.model

    def graph_of(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def timed(g, n=20):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    res["batches"] = {}
    for B in (1, 2, 4, 8, 10, 12, 16):
        slots = torch.tensor(sess[:B], device=dev)
        p0 = min(len(eng.sess[s]["ids"]) for s in sess[:B]) + 4
        ids1 = torch.ones(B, 1, dtype=torch.long, device=dev)
        pos1 = torch.full((B, 1), p0, dtype=torch.long, device=dev)
        ids2 = torch.ones(B, 2, dtype=torch.long, device=dev)
        pos2 = torch.stack([pos1[:, 0], pos1[:, 0] + 1], 1)
        hid = torch.zeros(B, 1, H, dtype=torch.bfloat16, device=dev)
        g_dec = graph_of(lambda: m.forward(ids1, st, slots, pos1, pos1[:, 0] + 1, prefill=False, kv_len=L))
        g_ver = graph_of(lambda: m.forward(ids2, st, slots, pos2, pos2[:, 1] + 1, prefill=False, kv_len=L,
                                           all_logits=True, hidden=True))
        g_drf = graph_of(lambda: m.mtp_forward(ids1, hid, st, slots, pos1, pos1[:, 0] + 1, prefill=False, kv_len=L))
        t = {"dec_ms": timed(g_dec), "ver_ms": timed(g_ver), "drf_ms": timed(g_drf)}
        t = {k: round(v, 2) for k, v in t.items()}
        t["speedup_bound"] = round((1 + rate) * t["dec_ms"] / (t["ver_ms"] + t["drf_ms"]), 3)
        res["batches"][str(B)] = t
        print("MTPB", B, json.dumps(t), flush=True)
        del g_dec, g_ver, g_drf
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
