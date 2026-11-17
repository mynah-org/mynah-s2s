"""G7: the frame / text-LLM GPU gate (model-free)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.gpu_gate import GpuGate  # noqa: E402


def test_frames_and_units_never_overlap_and_frames_have_priority():
    g = GpuGate()
    log, active = [], []

    def worker():
        for _ in range(30):
            with g.unit():
                active.append("u")
                assert active.count("f") == 0
                time.sleep(0.002)
                active.remove("u")
                log.append("u")

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(10):
        with g.frame():
            active.append("f")
            assert active.count("u") == 0
            time.sleep(0.005)
            active.remove("f")
            log.append("f")
        time.sleep(0.004)                       # the loop's idle time: the text LLM runs here
    t.join()
    assert log.count("f") == 10 and log.count("u") == 30 and g.stats["frames"] == 10


def test_held_by_me_and_sync_called():
    calls = []
    g = GpuGate(sync=lambda: calls.append(1))
    with g.unit():
        assert g.held_by_me()
    assert not g.held_by_me() and calls == [1]


def test_unit_budget_lets_the_text_llm_run_after_each_frame_even_when_frames_queue():
    g = GpuGate(unit_budget_ms=30)
    units = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            with g.unit():
                units.append(time.monotonic())
                time.sleep(0.005)

    t = threading.Thread(target=worker)
    t.start()
    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end:                 # back-to-back frames: the loop is "behind"
        with g.frame():
            time.sleep(0.01)
    stop.set()
    t.join()
    assert len(units) >= 10 and g.stats["frames"] >= 5          # neither side starves
    g0 = GpuGate(unit_budget_ms=0)                               # strict priority: back-to-back frames starve units
    assert g0.budget == 0
