"""Deployment presets: one config shape for every machine, tiers switchable, caps as fractions or bytes.

    from pocket_swap.config import preset, auto
    cfg = preset("l40s-16g")                       # or "gb10", "l4-24g"
    cfg = auto(stack_vram=9e9, stack_host=6e9)    # size from the live device / host and the resident stack
    sw = cfg.build(base_weights, "english_2026-09_24l", packages_dir)

Default mode is "ondemand" (compressed per-language terms resident, applied on the fly by the fused kernel, torch
fallback): the least memory per language. "materialise" (one bf16 backbone per resident language, no per-step
cost) stays available. The package precision is read from the manifests.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

GiB = 2 ** 30


@dataclass
class SwapConfig:
    mode: str = "ondemand"                  # "ondemand" (default) | "materialise"
    base_precision: str = "bf16"            # shared base for delta packages: bf16 | int8 | int4 (ignored for w*)
    device: str = "cuda"
    # tier switches
    vram_slots: bool = True                 # off = one slot only (every language change reloads)
    pinned: bool = True
    pageable: bool = False
    disk: bool = True                       # always the source of truth; off only documents "no catalogue"
    # caps: int = bytes, float in (0, 1] = fraction of device memory / host RAM
    max_slots: int = 10
    slot_budget: int | float | None = 0.10  # memory all language slots together may use
    vram_reserve: int = 1 * GiB             # admission control: keep this free on the device after a load
    pinned_cap: int | float | None = 0.08
    pinned_max_languages: int | None = 8
    pageable_cap: int | float | None = 0.10
    pageable_max_languages: int | None = 16
    slot_idle_s: float | None = 900.0       # free a slot nobody used for this long
    host_idle_s: float | None = 1800.0
    check_hashes: bool = False

    def build(self, base_weights: str, base_language: str, packages: str, **overrides):
        from .swapper import PocketSwapper

        c = replace(self, **overrides) if overrides else self
        return PocketSwapper(
            base_weights, base_language, packages, n_slots=c.max_slots if c.vram_slots else 1, device=c.device,
            vram_cap=c.slot_budget, vram_reserve=c.vram_reserve,
            pinned_cap=c.pinned_cap if c.pinned else 0, pinned_max_languages=c.pinned_max_languages if c.pinned else 0,
            ram_cap=c.pageable_cap if c.pageable else 0,
            ram_max_languages=c.pageable_max_languages if c.pageable else 0,
            host_idle_s=c.host_idle_s, slot_idle_s=c.slot_idle_s, check_hashes=c.check_hashes,
            mode=c.mode, base_precision=c.base_precision)

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # rich box: 128 GB unified memory shared with the whole stack. Host tiers live in the same pool, so keep them
    # small and let more languages stay resident as slots.
    "gb10": SwapConfig(max_slots=16, slot_budget=8 * GiB, vram_reserve=4 * GiB, pinned=True, pinned_cap=2 * GiB,
                       pinned_max_languages=6, pageable=False),
    # reference small box: L40S 48 GB (~44 usable) + ~16 GiB host RAM, 4 vCPU. Host RAM is the scarce side:
    # a small pinned tier, no pageable tier, catalogue on NVMe; VRAM slots carry the hot set.
    "l40s-16g": SwapConfig(max_slots=12, slot_budget=5 * GiB, vram_reserve=2 * GiB, pinned=True,
                           pinned_cap=int(1.5 * GiB), pinned_max_languages=4, pageable=False),
    # L4 24 GB (~22.5 usable) with a small LLM: few slots.
    "l4-24g": SwapConfig(max_slots=4, slot_budget=int(1.6 * GiB), vram_reserve=1 * GiB, pinned=True,
                         pinned_cap=1 * GiB, pinned_max_languages=3, pageable=False),
}


def preset(name: str, **overrides) -> SwapConfig:
    return replace(PRESETS[name], **overrides)


def _host_ram() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * GiB


def auto(stack_vram: float, stack_host: float, slot_bytes: float = 0.36e9, package_bytes: float = 0.335e9,
         vram_total: float | None = None, host_total: float | None = None, unified: bool | None = None,
         languages_wanted: int = 10, headroom: float = 0.06) -> SwapConfig:
    """Size a config from the device and host totals (read from torch / the OS when not given), the bytes the
    rest of the stack keeps resident, and the per-language sizes of the chosen packages."""
    if vram_total is None:
        import torch

        vram_total = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0
    host_total = _host_ram() if host_total is None else host_total
    if unified is None:
        unified = abs(vram_total - host_total) / max(host_total, 1) < 0.15  # same pool seen twice
    vram_left = vram_total * (1 - headroom) - stack_vram - 0.7e9  # Pocket base + Mimi
    reserve = max(1 * GiB, int(0.04 * vram_total))
    n = max(1, min(languages_wanted, int((vram_left - reserve) // slot_bytes)))
    host_left = (host_total - stack_host) if not unified else (vram_left - n * slot_bytes)
    pinned_n = max(0, min(languages_wanted - n, int(0.25 * max(host_left, 0) // package_bytes)))
    return SwapConfig(max_slots=n, slot_budget=int(n * slot_bytes) + 1, vram_reserve=reserve,
                      pinned=pinned_n > 0, pinned_cap=int(pinned_n * package_bytes) + 1 if pinned_n else 0,
                      pinned_max_languages=pinned_n or None, pageable=False)
