"""The ears as a JSON-lines process (run with the python of an environment that has transformers >= 5.13 for
Nemotron 3.5 ASR; the engine's own environment may pin another transformers).

  stdin   {"op": "open", "sid", "lang": "it", "thr": {...}}          a new session (fresh caches, GRU, rules)
          {"op": "hop", "sid", "n", "mode": "listen"|"speaking", "reset": bool, "u": b64 float32[5120], "a": ...}
          {"op": "transcribe", "id", "lang", "pcm": b64 float32}    offline transcript (content evidence)
          {"op": "mem"} | {"op": "close", "sid"} | {"op": "quit"}
  stdout  {"ev": "ready", ...} | {"ev": "opened", ...} | {"ev": "hop", "n", "frames", "events", "pieces", "ms", ...}
          | {"ev": "transcript", "id", "text", "ms"} | {"ev": "closed", "stats"} | {"ev": "error", "error"}

Several sessions, each with its own language and its own state (``EarsSession``: encoder caches, windows, RNN-T
predictor, GRU state, rules); every reply carries the session id. Hops are served in arrival order (not batched
across sessions yet). Prints nothing else on stdout (logs go to stderr).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROMPTS = {"auto": 101, "en": 0, "it": 15, "fr": 8, "de": 9, "es": 3}


def rss_gb(key: str = "VmRSS") -> float:
    """Host RSS (VmRSS) or its peak (VmHWM) in GB."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith(key + ":"):
                return round(int(line.split()[1]) / 2 ** 20, 3)
    except OSError:
        pass
    return -1.0


def b64f32(s: str):
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


def main():  # pragma: no cover - model process
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="tyger bundle dir (config.json, model.safetensors, nemotron/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--max-batch", type=int, default=16, help="hops of different sessions encoded together")
    a = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def emit(o):
        out.write(json.dumps(o, ensure_ascii=False) + "\n")
        out.flush()
    import numpy as np
    import torch
    from frankenstein2.ears.rules import Thresholds
    from frankenstein2.ears.stream import EarsSession, hop_batch, transcribe_offline
    from frankenstein2.ears.turntaker import load_bundle
    t0 = time.monotonic()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, proc, tt, cfg = load_bundle(a.bundle, check_sha=not a.no_sha, device=dev)
    pdict = dict(getattr(proc, "prompt_dictionary", None) or {}) or PROMPTS

    def mem():
        if dev == "cpu":
            return {}
        return {"cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 2 ** 30, 3),
                "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
    emit({"ev": "ready", "device": dev, "load_s": round(time.monotonic() - t0, 1), "lookahead": a.lookahead,
          "torch": torch.__version__, "tf32": torch.backends.cuda.matmul.allow_tf32,
          "capability": list(torch.cuda.get_device_capability()) if dev != "cpu" else None, **mem(),
          "rss_gb": rss_gb(), "rss_peak_gb": rss_gb("VmHWM")})
    sessions, stats = {}, {}
    import queue
    import threading
    inbox: queue.Queue = queue.Queue()

    def reader():
        for line in sys.stdin:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                continue
        inbox.put({"op": "quit"})
    threading.Thread(target=reader, daemon=True).start()
    held = []
    while True:
        c = held.pop(0) if held else inbox.get()
        op = c.get("op")
        if op == "hop" and a.max_batch > 1:      # hops of other sessions waiting now: one batched window encode
            batch, seen = [c], {c["sid"]}
            while len(batch) < a.max_batch:
                try:
                    nxt = inbox.get_nowait()
                except queue.Empty:
                    break
                if nxt.get("op") == "hop" and nxt["sid"] not in seen and nxt["sid"] in sessions:
                    batch.append(nxt)
                    seen.add(nxt["sid"])
                else:
                    held.append(nxt)
                    break
            if len(batch) > 1:
                try:
                    with torch.inference_mode():
                        rs = hop_batch([(sessions[x["sid"]], b64f32(x["u"]), b64f32(x["a"]), x["mode"],
                                         (x.get("reset") or False)) for x in batch])
                    for x, r in zip(batch, rs):
                        st = stats[x["sid"]]
                        st["hops"] += 1
                        for key in ("ms", "enc_ms", "win_ms"):
                            st[key].append(r[key])
                        emit({"ev": "hop", "sid": x["sid"], "n": x["n"], "t_ns": time.monotonic_ns(), **r})
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    for x in batch:
                        emit({"ev": "error", "op": "hop", "sid": x["sid"], "n": x["n"],
                              "error": f"{type(e).__name__}: {str(e)[:300]}"})
                continue
        try:
            if op == "quit":
                break
            if op == "open":
                lang = c.get("lang", "it")
                thr = Thresholds(**(c.get("thr") or {}))
                if lang not in PROMPTS and lang not in pdict:
                    raise ValueError(f"no ASR prompt for language {lang!r}")
                sid = c["sid"]
                sessions[sid] = EarsSession(model, proc, tt, dev, int(pdict.get(lang, PROMPTS.get(lang))), thr,
                                            k=a.lookahead, gru_features=c.get("gru_features", "window"))
                stats[sid] = {"hops": 0, "ms": [], "enc_ms": [], "win_ms": []}
                emit({"ev": "opened", "sid": sid, "lang": lang, "prompt_id": sessions[sid].rnnt.prompt_id,
                      "thr": thr.__dict__, **mem()})
            elif op == "hop":
                sid = c["sid"]
                with torch.inference_mode():
                    r = sessions[sid].hop(b64f32(c["u"]), b64f32(c["a"]), c["mode"], (c.get("reset") or False))
                st = stats[sid]
                st["hops"] += 1
                for key in ("ms", "enc_ms", "win_ms"):
                    st[key].append(r[key])
                emit({"ev": "hop", "sid": sid, "n": c["n"], "t_ns": time.monotonic_ns(), **r})
            elif op == "transcribe":
                t1 = time.monotonic()
                lang = c.get("lang", "it")
                with torch.inference_mode():
                    txt = transcribe_offline(model, proc, b64f32(c["pcm"]),
                                             int(pdict.get(lang, PROMPTS.get(lang, 101))), dev, k=a.lookahead)
                emit({"ev": "transcript", "id": c.get("id"), "text": txt, "sid": c.get("sid"),
                      "ms": round((time.monotonic() - t1) * 1e3, 1)})
            elif op == "mem":
                emit({"ev": "mem", "id": c.get("id"), **mem(), "rss_gb": rss_gb(), "rss_peak_gb": rss_gb("VmHWM")})
            elif op == "close":
                sid = c["sid"]
                sessions.pop(sid, None)
                emit({"ev": "closed", "sid": sid, "stats": stats.pop(sid, {}), **mem()})
        except Exception as e:  # noqa: BLE001 - report, keep serving
            import traceback
            traceback.print_exc()
            emit({"ev": "error", "op": op, "sid": c.get("sid"), "n": c.get("n"), "error": f"{type(e).__name__}: {str(e)[:300]}"})
    emit({"ev": "bye", **mem()})


if __name__ == "__main__":
    main()
