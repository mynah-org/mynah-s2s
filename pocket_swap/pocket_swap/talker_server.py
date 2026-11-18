"""Drop-in JSON-lines talker process backed by pocket_swap (same protocol as the engine's Pocket talker process).

  stdin   {"op": "voice", "id", "lang", "voice"}                 bind a warm-up session to (lang, voice) and report
          {"op": "say", "ss", "sid", "lang", "voice", "text", "epoch"}   one segment of session ``ss``; the session
                                                                 is (re)bound to ``lang`` / ``voice`` first
          {"op": "cancel", "ss", "epoch"} | {"op": "close", "ss"} | {"op": "mem", "id"} | {"op": "quit"}
  stdout  {"ev": "ready"} | {"ev": "voice", ...} | {"ev": "audio", "ss", "sid", "sr", "pcm16": b64}
          | {"ev": "done", "ss", "sid", "first_audio_ms", "audio_s", "gen_s"} | {"ev": "cancelled", ...}
          | {"ev": "error", ...} | {"ev": "mem", ...} | {"ev": "bye"}

    python -m pocket_swap.talker_server --base EN_24L_CLONING.safetensors --packages PACKAGES_DIR \\
        [--preset l40s-16g|gb10|l4-24g] [--mode ondemand|materialise] [--device cuda]

Default: mode "ondemand" (compressed language terms, fused on-the-fly apply), preset sized for the reference small
box. Segments are generated one at a time in arrival order (sessions interleave at segment granularity).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time

LANG = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l", "de": "german_24l",
        "es": "spanish_24l", "pt": "portuguese_24l", "nl": "dutch_24l"}


def main():  # pragma: no cover - model process
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base checkpoint (English 24-layer, with voice cloning)")
    ap.add_argument("--base-language", default="english_2026-09_24l")
    ap.add_argument("--packages", required=True)
    ap.add_argument("--preset", default="l40s-16g")
    ap.add_argument("--mode", default=None, help="override the preset's mode (ondemand | materialise)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--emit-ms", type=float, default=160.0)
    a = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(o):
        with lock:
            out.write(json.dumps(o) + "\n")
            out.flush()

    import numpy as np
    import torch

    from .config import preset

    torch.backends.mkldnn.enabled = False
    over = {"device": a.device}
    if a.mode:
        over["mode"] = a.mode
    sw = preset(a.preset, **over).build(a.base, a.base_language, a.packages)
    sr = sw.sample_rate
    emit({"ev": "ready", "device": a.device, "preset": a.preset, **sw.stats(), **sw.mem()})
    jobs: queue.Queue = queue.Queue()
    floor: dict = {}
    current = {"job": None, "stop": None}

    def reader():
        for line in sys.stdin:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if c.get("op") == "cancel":
                ss = c["ss"]
                floor[ss] = max(floor.get(ss, 0), int(c["epoch"]))
                j = current["job"]
                if j is not None and j["ss"] == ss and j["epoch"] < floor[ss]:
                    current["stop"].set()
                emit({"ev": "cancelled", "ss": ss, "epoch": c["epoch"]})
                continue
            jobs.put(c)
        jobs.put({"op": "quit"})

    threading.Thread(target=reader, daemon=True).start()
    emit_n = int(sr * a.emit_ms / 1000)
    while True:
        c = jobs.get()
        op = c.get("op")
        if op == "quit":
            break
        try:
            if op == "voice":
                t0 = time.monotonic()
                b = sw.open(f"warm:{c.get('id')}", LANG.get(c["lang"], c["lang"]), voice=c.get("voice"), timeout=30)
                emit({"ev": "voice", "id": c.get("id"), "lang": c["lang"], "slot": b.slot, "event": b.event,
                      "source": b.source, "load_s": round(time.monotonic() - t0, 3), **sw.mem()})
                sw.close(f"warm:{c.get('id')}")
            elif op == "close":
                sw.close(c["ss"])
            elif op == "mem":
                emit({"ev": "mem", "id": c.get("id"), **sw.stats(), **sw.mem()})
            elif op == "say":
                ss = c["ss"]
                if int(c["epoch"]) < floor.get(ss, 0):
                    emit({"ev": "cancelled", "ss": ss, "sid": c["sid"]})
                    continue
                sw.open(ss, LANG.get(c["lang"], c["lang"]), voice=c.get("voice"), timeout=30)
                stop = threading.Event()
                current.update(job={"ss": ss, "epoch": int(c["epoch"])}, stop=stop)
                t = time.monotonic()
                first, n, acc = None, 0, []

                def flush():
                    if acc:
                        w = np.concatenate(acc)
                        acc.clear()
                        emit({"ev": "audio", "ss": ss, "sid": c["sid"], "sr": sr,
                              "pcm16": base64.b64encode((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()).decode()})

                for x in sw.stream(ss, c["text"], stop=stop):
                    if stop.is_set():
                        break
                    w = x.cpu().numpy()
                    first = first or time.monotonic() - t
                    n += w.size
                    acc.append(w)
                    if sum(len(z) for z in acc) >= emit_n or n == w.size:
                        flush()
                if not stop.is_set():
                    flush()
                current.update(job=None, stop=None)
                emit({"ev": "cancelled" if stop.is_set() else "done", "ss": ss, "sid": c["sid"],
                      "first_audio_ms": round(first * 1e3, 1) if first else None,
                      "audio_s": round(n / sr, 2), "gen_s": round(time.monotonic() - t, 3)})
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            emit({"ev": "error", "op": op, "id": c.get("id"), "ss": c.get("ss"), "sid": c.get("sid"),
                  "error": f"{type(e).__name__}: {str(e)[:300]}"})
    try:
        emit({"ev": "bye", **sw.mem()})
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
