"""Memory budget planner for language slots (pure python, no torch).

Given the device (VRAM, unified or discrete memory, link bandwidth), the host (RAM, vCPUs) and what the rest of the
stack keeps resident, it says how much is left for Pocket language slots, picks the tier defaults (slots, pinned
host cap, pageable host cap) and estimates the swap time of each tier. Every figure passed in should say whether it
was measured or estimated; the planner only does arithmetic.

    python -m pocket_swap.planner --preset l40s-g6e.xlarge --languages 3
    python -m pocket_swap.planner --vram-gb 24 --host-gb 16 --vcpus 4 --link-gbps 25 --stack asr=3,llm=7 --languages 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

GB = 1e9


@dataclass
class PocketSizes:
    """Bytes. Defaults: Pocket 24-layer, bf16 backbone (sizes from the package/slot code; see the bench)."""

    base_backbone: float = 0.604 * GB      # shared EN backbone, bf16 (bench: base + Mimi = 0.689 GB allocated)
    mimi: float = 0.085 * GB               # shared codec (encoder + decoder), fp32
    slot_materialise: float = 0.658 * GB   # mode a: full backbone + per-language tensors (head fp32), bench-allocated
    slot_ondemand: dict = field(default_factory=lambda: {"bf16": 0.652 * GB, "int8": 0.360 * GB, "int4": 0.204 * GB,
                                                         "w8": 0.354 * GB, "w4": 0.203 * GB, "wfp8": 0.354 * GB})
    package: dict = field(default_factory=lambda: {"bf16": 0.632 * GB, "int8": 0.335 * GB, "int4": 0.184 * GB,
                                                   "w8": 0.335 * GB, "w4": 0.184 * GB, "wfp8": 0.335 * GB})
    session_state: float = 0.005 * GB      # voice/prompt KV state per (language, voice) (bench: 4.9 MB)
    gen_peak: float = 0.121 * GB           # extra allocation while one session generates (bench, mode a)


@dataclass
class Target:
    name: str
    vram: float                 # bytes visible to CUDA
    unified: bool               # host RAM and VRAM are the same pool
    host_ram: float             # bytes (for unified: the same pool as vram)
    vcpus: int
    link_gbps: float            # host->device copy bandwidth, GB/s (pinned); pageable assumed ~60 % of it
    disk_gbps: float            # sequential read, GB/s
    gpu_mem_gbps: float         # device memory bandwidth, GB/s (dequant + add is memory-bound)
    stack_vram: dict            # name -> bytes resident on the GPU besides Pocket (incl. CUDA contexts)
    stack_host: float           # bytes of host RAM used by the OS + the stack (discrete) / not counting vram
    headroom: float = 0.06      # fraction of VRAM kept free (allocator fragmentation, spikes)
    notes: str = ""


@dataclass
class Plan:
    target: str
    vram_for_pocket: float
    pocket_fixed: float
    slots_materialise: int
    slots_ondemand: dict
    pinned_cap: float
    pageable_cap: float
    swap_ms: dict
    advice: list


def swap_estimates(t: Target, p: PocketSizes, precision: str = "w8", zstd_mbps_per_core: float = 900.0) -> dict:
    """Estimated ms per tier for one language load (materialise mode); dequant+add is memory-bound on the GPU."""
    pkg = p.package[precision]
    slot = p.slot_materialise
    dequant = (pkg + p.base_backbone + slot) / (t.gpu_mem_gbps * GB) * 1e3  # read codes + base, write slot
    pinned = pkg / (t.link_gbps * GB) * 1e3 + dequant
    pageable = pkg / (0.6 * t.link_gbps * GB) * 1e3 + dequant
    disk = pkg / (t.disk_gbps * GB) * 1e3 + pageable
    zstd = pkg / (zstd_mbps_per_core * 1e6 * max(1, min(t.vcpus, 4))) * 1e3
    return {"hot_vram": 0.0, "pinned": round(pinned, 1), "pageable": round(pageable, 1),
            "disk_uncompressed": round(disk, 1), "disk_zstd_extra_cpu": round(zstd, 1)}


def plan(t: Target, p: PocketSizes, languages: int = 3, precision: str = "w8", sessions: int = 4) -> Plan:
    stack = sum(t.stack_vram.values())
    usable = t.vram * (1 - t.headroom) - stack
    fixed = p.base_backbone + p.mimi + sessions * (p.session_state + p.gen_peak)
    left = usable - fixed
    n_mat = max(0, min(languages, int(left // p.slot_materialise)))
    n_od = {k: max(0, min(languages, int(left // v))) for k, v in p.slot_ondemand.items()}
    if t.unified:
        host_free = 0.0  # same pool: host tiers only make sense for languages beyond the slots
        pageable_free = max(0.0, left - n_mat * p.slot_materialise)
    else:
        host_free = max(0.0, t.host_ram - t.stack_host)
        pageable_free = host_free
    pkg = p.package[precision]
    extra = max(0, languages - n_mat)
    pinned_cap = min(host_free * 0.5, max(extra, 1) * pkg) if not t.unified else min(pageable_free * 0.5, extra * pkg)
    pageable_cap = max(0.0, min(pageable_free - pinned_cap, max(extra, 1) * pkg))
    advice = []
    if left <= 0:
        advice.append("no VRAM left for Pocket on top of the stack: Pocket must move (CPU or another GPU)")
    elif n_mat >= languages:
        advice.append(f"mode a (materialise): all {languages} languages resident; swaps are VRAM hits")
    elif n_od.get(precision, 0) >= languages:
        advice.append(f"mode b (ondemand, {precision}): all {languages} languages fit as compressed deltas; "
                      f"per-step cost instead of slot copies")
    else:
        advice.append(f"only {n_mat} materialised slot(s): others load from host tiers (see swap_ms)")
    if t.vcpus <= 4:
        advice.append("few vCPUs: keep packages uncompressed on disk (no zstd) and dequantise on the GPU")
    if t.unified:
        advice.append("unified memory: pinned/pageable tiers share the pool with VRAM; prefer more slots over host copies")
    return Plan(t.name, usable, fixed, n_mat, n_od, pinned_cap, pageable_cap, swap_estimates(t, p, precision), advice)


# Presets. Sources for instance specs: AWS EC2 G6 / G6e pages (instance tables); GPU memory bandwidth from vendor
# datasheets. Stack figures are ESTIMATES until measured end-to-end numbers replace them.
def _stack(llm_gb: float) -> dict:
    return {"asr_0.6b+gru (est)": 3.0 * GB, "llm (est)": llm_gb * GB, "cuda_contexts (est)": 1.0 * GB}


PRESETS = {
    "gb10": Target("GB10 128 GB unified", 128 * GB, True, 128 * GB, 20, 50, 5, 273, _stack(7.0), 0.0,
                   notes="unified LPDDR5x; the whole stack shares the pool; OS/page cache not subtracted here"),
    "l4-g6.xlarge": Target("L4 24 GB, g6.xlarge (4 vCPU, 16 GiB)", 22.5 * GB, False, 16 * 1.0737 * GB, 4, 25, 1, 300,
                           _stack(7.0), 6 * GB),
    "l4-g6.2xlarge": Target("L4 24 GB, g6.2xlarge (8 vCPU, 32 GiB)", 22.5 * GB, False, 32 * 1.0737 * GB, 8, 25, 1, 300,
                            _stack(7.0), 6 * GB),
    "l40s-g6e.xlarge": Target("L40S 48 GB, g6e.xlarge (4 vCPU, 32 GiB)", 44.7 * GB, False, 32 * 1.0737 * GB, 4, 25, 1,
                              864, _stack(7.0), 6 * GB),
    "l40s-g6e.xlarge-llm-bf16": Target("L40S 48 GB, g6e.xlarge, LLM bf16", 44.7 * GB, False, 32 * 1.0737 * GB, 4, 25, 1,
                                       864, _stack(19.0), 6 * GB),
    "l40s-16g": Target("L40S 48 GB + 16 GiB host (reference small box, 4 vCPU)", 44.5 * GB, False, 16 * 1.0737 * GB,
                        4, 25, 2, 864, _stack(7.0), 8 * GB, notes="host RAM is the scarce side: tiny pinned tier"),
    "rtx6000ada": Target("RTX 6000 Ada 48 GB (host 64 GB assumed)", 47.5 * GB, False, 64 * GB, 16, 25, 3, 960,
                         _stack(7.0), 8 * GB),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pocket-swap-planner")
    ap.add_argument("--preset", choices=sorted(PRESETS), action="append")
    ap.add_argument("--vram-gb", type=float)
    ap.add_argument("--host-gb", type=float)
    ap.add_argument("--vcpus", type=int, default=4)
    ap.add_argument("--unified", action="store_true")
    ap.add_argument("--link-gbps", type=float, default=25)
    ap.add_argument("--disk-gbps", type=float, default=1)
    ap.add_argument("--gpu-mem-gbps", type=float, default=300)
    ap.add_argument("--stack", default="", help="name=GB,... resident on the GPU besides Pocket")
    ap.add_argument("--stack-host-gb", type=float, default=6)
    ap.add_argument("--languages", type=int, default=3)
    ap.add_argument("--precision", default="w8", choices=["bf16", "int8", "int4", "w8", "w4", "wfp8"])
    ap.add_argument("--sessions", type=int, default=4)
    a = ap.parse_args(argv)
    targets = [PRESETS[p] for p in a.preset or []]
    if a.vram_gb:
        stack = {k: float(v) * GB for k, v in (x.split("=") for x in a.stack.split(",") if x)}
        targets.append(Target("custom", a.vram_gb * GB, a.unified, (a.host_gb or a.vram_gb) * GB, a.vcpus,
                              a.link_gbps, a.disk_gbps, a.gpu_mem_gbps, stack, a.stack_host_gb * GB))
    if not targets:
        targets = list(PRESETS.values())
    for t in targets:
        pl = plan(t, PocketSizes(), a.languages, a.precision, a.sessions)
        d = asdict(pl)
        for k in ("vram_for_pocket", "pocket_fixed", "pinned_cap", "pageable_cap"):
            d[k] = f"{d[k] / GB:.2f} GB"
        print(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
