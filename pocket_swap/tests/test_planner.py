"""Pure-python tests of the memory budget planner."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_swap.planner import GB, PRESETS, PocketSizes, Target, plan, swap_estimates  # noqa: E402


def t(vram_gb, stack_gb, unified=False, host_gb=32, vcpus=4):
    return Target("x", vram_gb * GB, unified, host_gb * GB, vcpus, 25, 1, 300, {"s": stack_gb * GB}, 6 * GB, 0.0)


def test_slots_follow_free_vram():
    p = PocketSizes()
    assert plan(t(24, 23), p).slots_materialise == 0
    big = plan(t(48, 10), p, languages=3)
    assert big.slots_materialise == 3 and big.slots_ondemand["int8"] == 3
    tight = plan(t(24, 21.8), p, languages=3)
    assert tight.slots_materialise < tight.slots_ondemand["int4"]


def test_no_room_advice():
    assert "no VRAM left" in plan(t(24, 30), PocketSizes()).advice[0]


def test_swap_estimates_monotonic():
    e = swap_estimates(t(48, 10), PocketSizes(), "int8")
    assert e["hot_vram"] < e["pinned"] < e["pageable"] < e["disk_uncompressed"]
    e4 = swap_estimates(t(48, 10), PocketSizes(), "int4")
    assert e4["pinned"] < e["pinned"]


def test_unified_has_no_separate_pinned_pool():
    p = plan(t(128, 20, unified=True, host_gb=128), PocketSizes(), languages=3)
    assert p.slots_materialise == 3 and p.pinned_cap == 0


def test_presets_run():
    for k, tg in PRESETS.items():
        assert plan(tg, PocketSizes()).target == tg.name
