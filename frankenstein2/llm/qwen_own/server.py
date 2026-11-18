"""QwenOwn as a JSON-lines process with the same protocol as ../qwen_server.py, so the Qwen plugin can switch from HF
generate to our own engine by starting this module instead:

  stdin   {"op": "gen", "ss", "rid", "messages", "tools"} | {"op": "cancel", "ss", "rid"} | {"op": "quit"}
  stdout  {"ev": "ready", ...} | {"ev": "delta", "ss", "rid", "text"} | {"ev": "end", "ss", "rid", "n", "first_ms"}

One engine slot per session id `ss` (opened on first use); a session's next prompt continues from its previous prompt.
Generation stops at end of turn or right after "</tool_call>". Greedy, thinking off.

  python -m frankenstein2.llm.qwen_own.server --model <snapshot> [--mode bf16|int8|int4] [--slots 8]
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
    ap.add_argument("--mode", default="bf16")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(o):
        with lock:
            out.write(json.dumps(o, ensure_ascii=False) + "\n")
            out.flush()
    from .engine import QwenOwn
    eng = QwenOwn(a.model, device=a.device, mode=a.mode, slots=a.slots, max_len=a.max_len)
    emit({"ev": "ready", "load_s": eng.load_s, **eng.footprint()})
    jobs: queue.Queue = queue.Queue()
    cancelled: set = set()
    slots: dict = {}

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
        if c["ss"] not in slots:
            slots[c["ss"]] = eng.open()
        sid = slots[c["ss"]]
        ids = eng.chat_ids(c["messages"], c.get("tools"))
        if c.get("assistant_prefix"):
            ids = ids + eng.tok.encode(c["assistant_prefix"])
        t0 = time.monotonic()
        first, n = None, 0
        for ev in eng.generate(sid, ids, max_new=a.max_new, stop_flag=lambda: key in cancelled):
            if ev.get("done"):
                break
            first = first or time.monotonic() - t0
            n += 1
            emit({"ev": "delta", "ss": c["ss"], "rid": c["rid"], "text": ev["text"]})
        emit({"ev": "end", "ss": c["ss"], "rid": c["rid"], "n": n,
              "first_ms": round(first * 1e3, 1) if first else None})


if __name__ == "__main__":
    main()
