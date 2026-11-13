"""Diagnostic: GPU time vs wall time of the Thinker and EarTTS calls made by the FC background thread.

The upstream "[FC Async] Step timing breakdown" is host wall time: with CUDA graphs the Thinker replay is asynchronous
(measured 0.3 ms/step "LLM" on the Ada) and its GPU cost lands in whatever syncs next (the "TTS warmup" bucket). This
wraps ``stt_model.forward`` (Thinker) and ``tts_model.infer_codes_one_step`` (EarTTS step) — the instance attributes
current at enable time, so it composes with the graph/batching wrappers — and, for calls made OUTSIDE the main thread
(the FC thread), records CUDA events on the current stream + host wall time. Main-thread calls are counted only.
Token-neutral (no extra sync inside the step; events are resolved at ``report``). Enable AFTER the perf profile.
"""

from __future__ import annotations

import threading
import time

import torch


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def enable(engine) -> dict:
    w = engine.wrapper
    if getattr(w, "_v2_fc_profile", None) is not None:
        raise RuntimeError("fc_profile already enabled")
    main = threading.main_thread()
    recs = {"thinker": [], "eartts": []}
    counts = {"thinker_main": 0, "eartts_main": 0}
    lock = threading.Lock()

    def wrap(obj, attr, key):
        orig = getattr(obj, attr)
        had = attr in obj.__dict__

        def f(*a, **kw):
            if threading.current_thread() is main:
                counts[key + "_main"] += 1
                return orig(*a, **kw)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            s.record()
            out = orig(*a, **kw)
            e.record()
            with lock:
                recs[key].append((s, e, (time.perf_counter() - t0) * 1e3))
            return out
        setattr(obj, attr, f)
        return (obj, attr, orig, had)

    stt, tts = w.model.stt_model, w.model.tts_model
    undo = [wrap(stt, "forward", "thinker"), wrap(tts, "infer_codes_one_step", "eartts")]
    w._v2_fc_profile = {"recs": recs, "counts": counts, "undo": undo}
    return {"wrapped": ["stt_model.forward", "tts_model.infer_codes_one_step"]}


def report(engine) -> dict:
    st = getattr(engine.wrapper, "_v2_fc_profile", None)
    if st is None:
        return {}
    torch.cuda.synchronize()
    out = dict(st["counts"])
    for key, lst in st["recs"].items():
        gpu = [s.elapsed_time(e) for s, e, _ in lst]
        wall = [w for _, _, w in lst]
        out[key + "_fc"] = {"n": len(lst), "gpu_ms_p50": _pct(gpu, 0.5), "gpu_ms_p95": _pct(gpu, 0.95),
                            "gpu_ms_sum": round(sum(gpu), 1), "host_ms_p50": _pct(wall, 0.5),
                            "host_ms_sum": round(sum(wall), 1)}
    return out


def disable(engine) -> None:
    w = engine.wrapper
    st = getattr(w, "_v2_fc_profile", None)
    if st is None:
        return
    for obj, attr, orig, had in reversed(st["undo"]):
        if had:
            setattr(obj, attr, orig)
        else:
            delattr(obj, attr)
    del w._v2_fc_profile
