"""Pocket-TTS as a JSON-lines talker process for several sessions (run with the Pocket environment's python).

  stdin   {"op": "voice", "id", "lang", "voice": wav path or stock name}   load (cached) the language model
                                                                           (24-layer by default) and the voice state
          {"op": "say", "ss", "sid", "lang", "voice", "text", "epoch"}     one whole text segment of session ``ss``
          {"op": "cancel", "ss", "epoch"}     drop queued / stop running segments of session ss with a lower epoch
          {"op": "mem", "id"} | {"op": "quit"}
  stdout  {"ev": "ready"} | {"ev": "voice", "id", ...} | {"ev": "audio", "ss", "sid", "pcm16": b64}
          | {"ev": "done", "ss", "sid", "first_audio_ms", "audio_s", "gen_s"} | {"ev": "cancelled", ...}
          | {"ev": "error", ...} | {"ev": "mem", ...}
Segments are generated one at a time in arrival order (sessions interleave at segment granularity). Models are kept
per language (``--max-langs`` loaded at once, least recently used dropped); pocket_swap's language slots will replace
this cache. oneDNN is disabled (aarch64 abort in its JIT, POCKET_DELTA / TTS_COMPARE notes). Logs on stderr.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import queue
import sys
import threading
import time

LANG = {"it": "italian", "fr": "french", "en": "english", "de": "german", "es": "spanish"}
STOCK = {"it": "giovanni", "fr": "estelle", "en": "alba", "de": "alba", "es": "alba"}
MODEL_24L = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l", "de": "german_24l"}


def rss_gb(key: str = "VmRSS") -> float:
    """Host RSS (VmRSS) or its peak (VmHWM) in GB."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith(key + ":"):
                return round(int(line.split()[1]) / 2 ** 20, 3)
    except OSError:
        pass
    return -1.0


def main():  # pragma: no cover - model process
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default="24l", help="24l | default")
    ap.add_argument("--max-langs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--emit-ms", type=float, default=160.0, help="batch audio into pieces of at least this length")
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
    from pocket_tts import TTSModel
    torch.backends.mkldnn.enabled = False
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    models: collections.OrderedDict = collections.OrderedDict()   # lang -> model
    voices: dict = {}                                               # (lang, voice) -> state

    def mem():
        d = {"rss_gb": rss_gb(), "rss_peak_gb": rss_gb("VmHWM")}
        if dev != "cpu":
            d.update(cuda_allocated_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 3),
                     cuda_reserved_gb=round(torch.cuda.memory_reserved() / 2 ** 30, 3),
                     cuda_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3))
        return d

    def model(lang):
        if lang in models:
            models.move_to_end(lang)
            return models[lang]
        name = MODEL_24L.get(lang) if a.variant == "24l" else LANG.get(lang, lang)
        print(f"loading {name} on {dev}", file=sys.stderr, flush=True)
        m = TTSModel.load_model(language=name or LANG.get(lang, lang))
        if dev != "cpu":
            m = m.to(dev)
        print(f"loaded {name}", file=sys.stderr, flush=True)
        models[lang] = m
        while len(models) > a.max_langs:
            old, _ = models.popitem(last=False)
            for key in [k for k in voices if k[0] == old]:
                voices.pop(key)
            if dev != "cpu":
                torch.cuda.empty_cache()
        return m

    def voice(lang, v):
        key = (lang, v or "")
        if key not in voices:
            m = model(lang)
            try:
                voices[key] = (m.get_state_for_audio_prompt(v or STOCK.get(lang, "alba")), v or "stock")
            except Exception as e:  # noqa: BLE001 - gated cloning weights: stock voice
                voices[key] = (m.get_state_for_audio_prompt(STOCK.get(lang, "alba")),
                               f"stock ({type(e).__name__})")
        return voices[key]
    emit({"ev": "ready", "device": dev, "variant": a.variant, "torch": torch.__version__, **mem()})
    jobs: queue.Queue = queue.Queue()
    floor: dict = {}                       # ss -> lowest live epoch
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
    emit_n = None
    while True:
        c = jobs.get()
        op = c.get("op")
        if op == "quit":
            break
        try:
            if op == "voice":
                t0 = time.monotonic()
                _, vname = voice(c["lang"], c.get("voice"))
                emit({"ev": "voice", "id": c.get("id"), "lang": c["lang"], "voice": vname,
                      "load_s": round(time.monotonic() - t0, 2), "langs_loaded": list(models), **mem()})
            elif op == "mem":
                emit({"ev": "mem", "id": c.get("id"), **mem()})
            elif op == "say":
                ss = c["ss"]
                if int(c["epoch"]) < floor.get(ss, 0):
                    emit({"ev": "cancelled", "ss": ss, "sid": c["sid"]})
                    continue
                m = model(c["lang"])
                vs, _ = voice(c["lang"], c.get("voice"))
                emit_n = emit_n or int(m.sample_rate * a.emit_ms / 1000)
                stop = threading.Event()
                current.update(job={"ss": ss, "epoch": int(c["epoch"])}, stop=stop)
                t = time.monotonic()
                first, n, acc = None, 0, []

                def flush():
                    if acc:
                        w = np.concatenate(acc)
                        acc.clear()
                        emit({"ev": "audio", "ss": ss, "sid": c["sid"], "sr": m.sample_rate,
                              "pcm16": base64.b64encode((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()).decode()})
                for x in m.generate_audio_stream(vs, c["text"], stop=stop):
                    if stop.is_set():
                        break
                    w = x.detach().float().cpu().numpy().reshape(-1)
                    first = first or time.monotonic() - t
                    n += w.size
                    acc.append(w)
                    if sum(len(z) for z in acc) >= emit_n or first and n == w.size:
                        flush()               # the first piece goes out at once
                if not stop.is_set():
                    flush()
                current.update(job=None, stop=None)
                emit({"ev": "cancelled" if stop.is_set() else "done", "ss": ss, "sid": c["sid"],
                      "first_audio_ms": round(first * 1e3, 1) if first else None,
                      "audio_s": round(n / m.sample_rate, 2), "gen_s": round(time.monotonic() - t, 3)})
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            emit({"ev": "error", "op": op, "id": c.get("id"), "ss": c.get("ss"), "sid": c.get("sid"),
                  "error": f"{type(e).__name__}: {str(e)[:300]}"})
    try:
        emit({"ev": "bye", **mem()})
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
