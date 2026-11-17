"""G7: time-slicing of the default CUDA stream between the duplex frame loop and the text LLM.

Both NemotronH models (the duplex Thinker and the text LLM) must run on the default CUDA stream (their remote code
pins every block to it), so their kernels share one queue: a text-LLM prefill or a run of decode steps queued between
two frames delays the frame by its whole length (G6: frame p95 300+ ms, spikes > 1 s, the loop fell ~20 s behind).
The gate makes the GPU work of the two sides alternate instead of interleave:

  frame side   ``with gate.frame():``  one engine chunk (perception + Thinker frames + hooks), synchronized at the end
  text LLM     ``with gate.unit():``   one prefill or one decode step (+ head), synchronized at the end

Policy. ``unit_budget_ms = 0``: strict frame priority (a waiting frame is served before the next text-LLM unit; G7:
the loop kept real time but the text LLM starved, replies came 10-20 s late). ``unit_budget_ms > 0``: after each
frame the text LLM may run units for that long even if a frame is waiting, then frames have priority again; a frame
waits at most the budget plus one unit. Math is unchanged (same kernels, same order inside each side). Model-free;
tests in tests/test_gpu_gate.py.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class GpuGate:
    def __init__(self, sync=None, unit_budget_ms: float = 0.0):
        self.cv = threading.Condition()
        self.busy = None                 # "frame" | "unit" | None
        self.frames_waiting = 0
        self.units_waiting = 0
        self.owner = None
        self.sync = sync or (lambda: None)
        self.budget = unit_budget_ms / 1e3
        self.frame_end = 0.0
        self.stats = {"frames": 0, "units": 0, "unit_wait_ms": 0.0, "frame_wait_ms": 0.0}

    def held_by_me(self) -> bool:
        return self.owner == threading.get_ident()

    @contextmanager
    def frame(self):
        t0 = time.monotonic()
        with self.cv:
            self.frames_waiting += 1
            while self.busy is not None or (self.units_waiting and
                                            time.monotonic() - self.frame_end < self.budget):
                self.cv.wait(timeout=0.005)
            self.frames_waiting -= 1
            self.busy, self.owner = "frame", threading.get_ident()
        self.stats["frame_wait_ms"] += (time.monotonic() - t0) * 1e3
        try:
            yield
        finally:
            self.sync()
            with self.cv:
                self.busy, self.owner = None, None
                self.frame_end = time.monotonic()
                self.stats["frames"] += 1
                self.cv.notify_all()

    @contextmanager
    def unit(self):
        t0 = time.monotonic()
        with self.cv:
            self.units_waiting += 1
            while self.busy is not None or (self.frames_waiting and
                                            time.monotonic() - self.frame_end >= self.budget):
                self.cv.wait(timeout=0.01)
            self.units_waiting -= 1
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
