"""Pocket-TTS through pocket_swap as a JSON-lines talker process: one shared base + per-language packages, each
session bound to its own language slot (run with the Pocket environment's python; same protocol as pocket_server.py,
so ``PocketTalker`` drives either).

  stdin   {"op": "voice", "id": SESSION, "lang", "voice"}   bind the session to its language (+ voice); the reply
                                                            says which slot and where the language came from
          {"op": "say", "ss", "sid", "lang", "voice", "text", "epoch"}   one segment for session ss
          {"op": "cancel", "ss", "epoch"} | {"op": "close", "ss"} | {"op": "mem", "id"} | {"op": "quit"}
  stdout  {"ev": "ready", ...} | {"ev": "voice", "id", "slot", "source", "load_ms"} | {"ev": "audio", "ss", "sid",
          "pcm16"} | {"ev": "done", ...} | {"ev": "cancelled", ...} | {"ev": "error", ...} | {"ev": "mem", ...}

Mode (b) ``ondemand`` (compressed deltas resident, applied on the fly) is the default, (a) ``materialise`` an option;
the package precision is whatever the packages are (precision-agnostic). Slots and host tiers come from a pocket_swap
planner preset (``--preset``: gb10, l4-g6.xlarge, l40s-g6e.xlarge, ...), overridable with ``--slots``.
Segments are generated one at a time in arrival order.
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
from pathlib import Path

LANG = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l", "de": "german_24l"}


def plan_config(preset: str, mode: str, precision: str, languages: int, sessions: int) -> dict:
    """Slots and host-tier caps from the pocket_swap planner."""
    from pocket_swap.planner import PRESETS, PocketSizes, plan
    p = plan(PRESETS[preset], PocketSizes(), languages=languages, precision=precision, sessions=sessions)
    slots = p.slots_materialise if mode == "materialise" else p.slots_ondemand.get(precision, 0)
    return {"preset": preset, "n_slots": max(1, int(slots)), "pinned_cap": int(p.pinned_cap) or None,
            "ram_cap": int(p.pageable_cap) or None, "advice": p.advice, "vram_for_pocket_gb": round(
                p.vram_for_pocket / 2 ** 30, 2)}


def main():  # pragma: no cover - model process
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-root", required=True, help="directory that contains the pocket_swap package")
    ap.add_argument("--base", required=True, help="base checkpoint (english_2026-09_24l model.safetensors)")
    ap.add_argument("--base-language", default="english_2026-09_24l")
    ap.add_argument("--packages", required=True, help="language packages directory (any precision)")
    ap.add_argument("--mode", default="ondemand", choices=["ondemand", "materialise"])
    ap.add_argument("--preset", default="gb10")
    ap.add_argument("--languages", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--slots", type=int, default=0, help="override the planner's slot count")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--emit-ms", type=float, default=160.0)
    a = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(o):
        with lock:
            out.write(json.dumps(o) + "\n")
            out.flush()
    sys.path.insert(0, a.swap_root)
    import numpy as np
    import torch
    from pocket_swap.swapper import PocketSwapper, detect_precision
    torch.backends.mkldnn.enabled = False
    if a.threads:
        torch.set_num_threads(a.threads)
    precision = detect_precision(a.packages)
    cfg = plan_config(a.preset, a.mode, precision, a.languages, a.sessions)
    if a.slots:
        cfg["n_slots"] = a.slots
    t0 = time.monotonic()
    sw = PocketSwapper(a.base, a.base_language, a.packages, n_slots=cfg["n_slots"], device=a.device, mode=a.mode,
                       pinned_cap=cfg["pinned_cap"], ram_cap=cfg["ram_cap"],
                       ram_max_languages=None if cfg["ram_cap"] else 0)
    emit({"ev": "ready", "mode": a.mode, "precision": precision, "plan": cfg, "load_s": round(time.monotonic() - t0, 1),
          "torch": torch.__version__, **sw.mem()})
    jobs: queue.Queue = queue.Queue()
    floor: dict = {}
    current = {"ss": None, "epoch": None, "stop": None}

    def reader():
        for line in sys.stdin:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if c.get("op") == "cancel":
                ss = c["ss"]
                floor[ss] = max(floor.get(ss, 0), int(c["epoch"]))
                if current["ss"] == ss and current["epoch"] is not None and current["epoch"] < floor[ss]:
                    current["stop"].set()
                emit({"ev": "cancelled", "ss": ss, "epoch": c["epoch"]})
                continue
            jobs.put(c)
        jobs.put({"op": "quit"})
    threading.Thread(target=reader, daemon=True).start()
    emit_n = int(sw.sample_rate * a.emit_ms / 1000)
    while True:
        c = jobs.get()
        op = c.get("op")
        if op == "quit":
            break
        try:
            if op == "voice":
                b = sw.open(c["id"], LANG.get(c["lang"], c["lang"]), voice=c.get("voice") or None, timeout=30.0)
                emit({"ev": "voice", "id": c["id"], "lang": c["lang"], "slot": b.slot, "source": b.source,
                      "load_ms": round(b.load_ms, 1), "event": b.event, "stats": sw.stats(), **sw.mem()})
            elif op == "close":
                sw.close(c["ss"])
            elif op == "mem":
                emit({"ev": "mem", "id": c.get("id"), "stats": sw.stats(), **sw.mem()})
            elif op == "say":
                ss = c["ss"]
                if int(c["epoch"]) < floor.get(ss, 0):
                    emit({"ev": "cancelled", "ss": ss, "sid": c["sid"]})
                    continue
                stop = threading.Event()
                current.update(ss=ss, epoch=int(c["epoch"]), stop=stop)
                t = time.monotonic()
                first, n, acc = None, 0, []

                def flush():
                    if acc:
                        w = np.concatenate(acc)
                        acc.clear()
                        emit({"ev": "audio", "ss": ss, "sid": c["sid"], "sr": sw.sample_rate,
                              "pcm16": base64.b64encode((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()).decode()})
                for x in sw.stream(ss, c["text"], stop=stop):
                    if stop.is_set():
                        break
                    w = x.detach().float().cpu().numpy().reshape(-1)
                    first = first or time.monotonic() - t
                    n += w.size
                    acc.append(w)
                    if sum(len(z) for z in acc) >= emit_n or n == w.size:
                        flush()
                if not stop.is_set():
                    flush()
                current.update(ss=None, epoch=None, stop=None)
                emit({"ev": "cancelled" if stop.is_set() else "done", "ss": ss, "sid": c["sid"],
                      "first_audio_ms": round(first * 1e3, 1) if first else None,
                      "audio_s": round(n / sw.sample_rate, 2), "gen_s": round(time.monotonic() - t, 3)})
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
