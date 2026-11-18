"""Pure-python tests of deployment presets and auto-sizing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_swap.config import GiB, PRESETS, auto, preset  # noqa: E402


def test_default_mode_is_ondemand_and_presets_exist():
    for name in ("gb10", "l40s-16g", "l4-24g"):
        c = preset(name)
        assert c.mode == "ondemand" and c.disk
    assert preset("l40s-16g", mode="materialise").mode == "materialise"
    assert not PRESETS["l40s-16g"].pageable and PRESETS["l40s-16g"].pinned_cap <= 2 * GiB


def test_auto_small_host_keeps_pinned_tier_small():
    # L40S-like: 44.5 GB device, 16 GiB host, stack 26 GB VRAM / 7 GB host
    c = auto(stack_vram=26e9, stack_host=7e9, vram_total=44.5e9, host_total=16 * GiB, unified=False,
             languages_wanted=30)
    assert 10 <= c.max_slots <= 30
    assert c.pinned_cap <= 0.25 * (16 * GiB - 7e9) + 1


def test_auto_tight_device_one_slot_minimum():
    c = auto(stack_vram=21.5e9, stack_host=4e9, vram_total=22.5e9, host_total=16 * GiB, unified=False)
    assert c.max_slots == 1


def test_auto_unified():
    c = auto(stack_vram=30e9, stack_host=30e9, vram_total=128e9, host_total=128e9, languages_wanted=12)
    assert c.max_slots == 12 and not c.pageable
