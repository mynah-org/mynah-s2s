"""G7: time-slicing of the default CUDA stream between the duplex frame loop and the text LLM.

Both NemotronH models (the duplex Thinker and the text LLM) must run on the default CUDA stream (their remote code
pins every block to it), so their kernels share one queue: a text-LLM prefill or a run of decode steps queued between
two frames delays the frame by its whole length (G6: frame p95 300+ ms, spikes > 1 s, the loop fell ~20 s behind).
The gate makes the GPU work of the two sides alternate instead of interleave:

  frame side   ``with gate.frame():``  one engine chunk (perception + Thinker frames + hooks), synchronized at the end
  text LLM     ``with gate.unit():``   one prefill or one decode step (+ head), synchronized at the end

The frame side has priority: a waiting frame is served before the next text-LLM unit; while the loop is behind it
keeps the gate (the text LLM waits for the next idle moment); a frame waits at most one text-LLM unit. Math is
unchanged (same kernels, same order inside each side). Model-free; tests in tests/test_gpu_gate.py.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class GpuGate:
    def __init__(self, sync=None):
        self.cv = threading.Condition()
        self.busy = None                 # "frame" | "unit" | None
        self.frames_waiting = 0
        self.owner = None
        self.sync = sync or (lambda: None)
        self.stats = {"frames": 0, "units": 0, "unit_wait_ms": 0.0, "frame_wait_ms": 0.0}

    def held_by_me(self) -> bool:
        return self.owner == threading.get_ident()

    @contextmanager
    def frame(self):
        t0 = time.monotonic()
        with self.cv:
            self.frames_waiting += 1
            while self.busy is not None:
                self.cv.wait()
            self.frames_waiting -= 1
            self.busy, self.owner = "frame", threading.get_ident()
        self.stats["frame_wait_ms"] += (time.monotonic() - t0) * 1e3
        try:
            yield
        finally:
            self.sync()
            with self.cv:
                self.busy, self.owner = None, None
                self.stats["frames"] += 1
                self.cv.notify_all()

    @contextmanager
    def unit(self):
        t0 = time.monotonic()
        with self.cv:
            while self.busy is not None or self.frames_waiting:
                self.cv.wait()
            self.busy, self.owner = "unit", threading.get_ident()
        self.stats["unit_wait_ms"] += (time.monotonic() - t0) * 1e3
        try:
            yield
        finally:
            self.sync()
            with self.cv:
                self.busy, self.owner = None, None
                self.stats["units"] += 1
                self.cv.notify_all()
