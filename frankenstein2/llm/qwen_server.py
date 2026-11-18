"""Qwen3.5 text generation as a JSON-lines process (run with an environment whose transformers knows Qwen3.5).

  stdin   {"op": "gen", "ss", "rid", "messages", "tools"} | {"op": "cancel", "ss", "rid"} | {"op": "quit"}
  stdout  {"ev": "ready", ...} | {"ev": "delta", "ss", "rid", "text"} | {"ev": "end", "ss", "rid", "n", "first_ms"}
Greedy, thinking off (chat template ``enable_thinking=False``), bf16, one request at a time. Plain HF transformers
generate as a stand-in (no serving framework); the target is our own inference path (DESIGN.md §3). NOT VERIFIED
on a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time


def main():  # pragma: no cover - model process
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(o):
        with lock:
            out.write(json.dumps(o, ensure_ascii=False) + "\n")
            out.flush()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device).eval()
    emit({"ev": "ready", "load_s": round(time.monotonic() - t0, 1),
          "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3) if a.device != "cpu" else None})
    jobs: queue.Queue = queue.Queue()
    cancelled: set = set()

    def reader():
        for line in sys.stdin:
            c = json.loads(line)
            if c.get("op") == "cancel":
                cancelled.add((c["ss"], c["rid"]))
            else:
                jobs.put(c)
        jobs.put({"op": "quit"})
    threading.Thread(target=reader, daemon=True).start()
    while True:
        c = jobs.get()
        if c["op"] == "quit":
            return
        key = (c["ss"], c["rid"])
        if key in cancelled:
            continue
        ids = tok.apply_chat_template(c["messages"], tools=c.get("tools") or None, add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt").to(a.device)
        st = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)

        class Stop:
            def __call__(self, *_a, **_k):
                return key in cancelled
        from transformers import StoppingCriteriaList
        th = threading.Thread(target=m.generate, kwargs=dict(input_ids=ids, max_new_tokens=a.max_new, do_sample=False,
                                                              streamer=st, stopping_criteria=StoppingCriteriaList([Stop()])))
        t1 = time.monotonic()
        th.start()
        first, n = None, 0
        for piece in st:
            piece = piece.replace("<|im_end|>", "")
            if piece:
                first = first or time.monotonic() - t1
                n += 1
                emit({"ev": "delta", "ss": c["ss"], "rid": c["rid"], "text": piece})
        th.join()
        emit({"ev": "end", "ss": c["ss"], "rid": c["rid"], "n": n,
              "first_ms": round(first * 1e3, 1) if first else None})


if __name__ == "__main__":
    main()
