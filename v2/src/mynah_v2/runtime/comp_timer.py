"""Per-component time breakdown for single- and multi-session runs (diagnostic, token-neutral).

Same method as scripts/timeline_events.py: each call of a component records a CUDA event before and after it on the
CURRENT stream (no sync inside the call) plus host wall time; events are resolved once at ``report``. Components:
perception (perception_cache_mgr.step), eartts (tts_model.infer_codes_one_step), codec (audio_codec.decode), thinker
(stt_model.forward for B=1 / upstream calls; plus, when a batched backend is given, its step / step_one = the batched
Thinker). In multi-session mode every thread's calls are recorded (session workers, coordinator, FC threads); calls
from the FC threads are tagged ``fc`` (thread name) so FC work can be separated. Wrappers are instance attributes on
top of whatever is installed (graphs, quantization, rendezvous), removed by ``uninstall`` in reverse order.
gpu span = time the stream spent from the call's first op to its last op (≈ device work for one graph replay; includes
device idle during host launches for eager code; events of concurrent threads share the default stream, so spans of
overlapping calls overlap — use per-component sums as an upper bound of device-busy time).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import torch

PROFILE_LABELS = False                 # V2_TORCH_PROFILE: also open a record_function per call (comp:<c>/<tag>)


class CompTimer:
    def __init__(self):
        self.recs = defaultdict(list)          # (component, tag) -> [(ev0, ev1, host_ms)]
        self.lock = threading.Lock()
        self.undo = []
        self.enabled = True
        self.timeline = None                   # list when diagnostics ask for per-call records (start_timeline)

    def _wrap(self, obj, attr, comp):
        orig = getattr(obj, attr)
        had = attr in obj.__dict__
        timer = self

        def f(*a, **kw):
            if not timer.enabled:
                return orig(*a, **kw)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            if PROFILE_LABELS:
                tg = "fc" if "_fc_async" in threading.current_thread().name else "main"
                with torch.profiler.record_function(f"comp:{comp}/{tg}"):
                    out = orig(*a, **kw)
            else:
                out = orig(*a, **kw)
            e1.record()
            # upstream FC threads: threading.Thread(target=_fc_async_thread_fn) → default name "Thread-N (_fc_async…)"
            tag = "fc" if "_fc_async" in threading.current_thread().name else "main"
            host = (time.perf_counter() - t0) * 1e3
            with timer.lock:
                timer.recs[(comp, tag)].append((e0, e1, host))
                if timer.timeline is not None:
                    timer.timeline.append((comp, tag, threading.current_thread().name, t0, host, e0, e1))
            return out
        setattr(obj, attr, f)
        self.undo.append((obj, attr, orig, had))

    def install(self, engine, backend=None) -> "CompTimer":
        w, m = engine.wrapper, engine.wrapper.model
        pcm = getattr(w, "perception_cache_mgr", None)
        if pcm is not None and hasattr(pcm, "step"):
            self._wrap(pcm, "step", "perception")
        self._wrap(m.stt_model, "forward", "thinker_b1")
        self._wrap(m.tts_model, "infer_codes_one_step", "eartts")
        self._wrap(m.tts_model.audio_codec, "decode", "codec")
        if backend is not None:
            self._wrap(backend, "step", "thinker_batched")
            self._wrap(backend, "step_one", "thinker_step_one")
        return self

    def uninstall(self) -> None:
        for obj, attr, orig, had in reversed(self.undo):
            if had:
                setattr(obj, attr, orig)
            else:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
        self.undo.clear()

    def reset(self) -> None:
        with self.lock:
            self.recs.clear()

    def start_timeline(self) -> None:
        """Also keep one record per call (component, tag, thread, host start, host ms, gpu ms) for per-round views."""
        with self.lock:
            self.timeline = []

    def timeline_records(self, t_ref: float = 0.0) -> list[dict]:
        torch.cuda.synchronize()
        with self.lock:
            tl = list(self.timeline or [])
        return [{"comp": c, "tag": g, "thread": th, "t0_ms": round((t0 - t_ref) * 1e3, 3), "host_ms": round(h, 3),
                 "gpu_ms": round(e0.elapsed_time(e1), 3)} for c, g, th, t0, h, e0, e1 in tl]

    def report(self, rounds: int | None = None) -> dict:
        torch.cuda.synchronize()
        out = {}
        for (comp, tag), lst in sorted(self.recs.items()):
            gpu = sorted(e0.elapsed_time(e1) for e0, e1, _ in lst)
            host = sorted(h for _, _, h in lst)
            d = {"calls": len(lst), "gpu_ms_sum": round(sum(gpu), 1), "host_ms_sum": round(sum(host), 1),
                 "gpu_ms_p50": round(gpu[len(gpu) // 2], 2) if gpu else None,
                 "host_ms_p50": round(host[len(host) // 2], 2) if host else None}
            if rounds:
                d["gpu_ms_per_round"] = round(sum(gpu) / rounds, 2)
                d["host_ms_per_round"] = round(sum(host) / rounds, 2)
            out[f"{comp}/{tag}"] = d
        return out
