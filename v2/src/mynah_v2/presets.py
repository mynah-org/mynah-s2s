"""Named run configurations for the CLIs (run_stream.py, serve.py, rendezvous_pair.py).

`release` (the default) = the fastest configuration that passed every gate: CUDA graphs, batched stages, fc_merge,
fused EarTTS RMSNorm, int8 weight-only Thinker MLP, per-session EarTTS noise, graphs captured before sessions start,
cached RVQ tables in the EarTTS sampler, and the deterministic mode with flash / memory-efficient attention allowed and
without the NaN fill of uninitialized memory. Reproducible run-to-run, but not bit-identical to `release-exact`
(the attention kernels sum in a different order).
`release-exact` = the same minus the attention change: bit-identical outputs to the deterministic reference line.
`reference` = NVIDIA's reference step with none of the V2 execution optimisations (for parity checks and debugging).

The determinism switches (deterministic / fill_uninit / sdpa) are plain PyTorch settings: they never fail on another
GPU (without flash attention PyTorch picks the next available attention kernel), and their effective state is printed
(DETERMINISM ...) and recorded in the run summary.

Explicit command-line flags always win over the preset. Two policies for an optimisation that cannot be enabled on
this GPU / driver / software stack:
  * fallback (demo, serving: run_stream.py, serve.py): a WARNING is printed and the run continues without it; the
    active set is printed and recorded in the run summary, so a slower fallback is never silent;
  * strict (regression and performance gates: rendezvous_pair.py by default, or --strict): the run stops with an
    error, so a comparison is never made against a silently different configuration.
"""

from __future__ import annotations

import sys
import traceback

_SINGLE_EXP = "thinker_graph,wor:w8:mlp,codec_nosync,codec_fc_d2h,codec_fc_2codes,eartts_rvq_cache"
_MULTI_EXP = "wor:w8:mlp,codec_nosync,codec_fc_d2h,codec_fc_2codes,eartts_rvq_cache"
_MULTI_COMMON = {"graphs": 1, "profile": "optimized-graph-s4", "perc_batch": "graph", "codec_batch": "on",
                 "eartts_batch": "on", "eartts_rng": "session", "capture_policy": "precapture", "fc_merge": "on",
                 "rnnt_overlap": "on", "breakdown": True}

# single session (run_stream.py, serve.py); validated on the 40-item corpus (release vs the deterministic reference line:
# 12/12 reasoning, 10/10 small talk, tool calls and arguments unchanged, audio checked with the evaluation ASR)
SINGLE = {
    "release": {"mode": "enhanced", "perf_profile": "optimized-graph", "exp": _SINGLE_EXP,
                "deterministic": "on", "fill_uninit": "off", "sdpa": "fast"},
    "release-exact": {"mode": "enhanced", "perf_profile": "optimized-graph", "exp": _SINGLE_EXP,
                      "deterministic": "on", "fill_uninit": "off", "sdpa": "default"},
    "reference": {"mode": "reference", "perf_profile": None, "exp": "",
                  "deterministic": "off", "fill_uninit": "on", "sdpa": "default"},
}

# multi-session gate (rendezvous_pair.py); the flags of the published 4-session benchmarks
MULTI = {
    "release": {**_MULTI_COMMON, "exp": _MULTI_EXP, "deterministic": "on", "fill_uninit": "off", "sdpa": "fast"},
    "release-exact": {**_MULTI_COMMON, "exp": _MULTI_EXP, "deterministic": "on", "fill_uninit": "off",
                      "sdpa": "default"},
    "reference": {"graphs": 0, "profile": "", "exp": "", "perc_batch": "off", "codec_batch": "off", "eartts_batch": "off",
                  "eartts_rng": "global", "capture_policy": "lazy", "fc_merge": "off", "rnnt_overlap": "off",
                  "breakdown": False, "deterministic": "on", "fill_uninit": "on", "sdpa": "default"},
}


def add_determinism_args(ap) -> None:
    """--deterministic / --fill-uninit / --sdpa; default None = taken from the preset."""
    ap.add_argument("--deterministic", default=None, choices=["on", "off"],
                    help="s2s.deterministic (NeMo: torch deterministic algorithms, no flash / memory-efficient "
                         "attention). Preset default; off = NeMo's serving default, not reproducible run-to-run")
    ap.add_argument("--fill-uninit", default=None, choices=["on", "off"],
                    help="PyTorch fill_uninitialized_memory under deterministic mode (on = PyTorch default: every "
                         "torch.empty is NaN-filled, ~6 ms per multi-session round). Preset default")
    ap.add_argument("--sdpa", default=None, choices=["default", "fast"],
                    help="fast: allow flash / memory-efficient attention again after the deterministic mode disabled "
                         "them (inference only; reproducible run-to-run, not bit-identical to default). Preset default")


def determinism_overrides(args) -> dict:
    """cfg overrides to merge BEFORE the user's --override (an explicit --override s2s.deterministic=... wins)."""
    return {"s2s.deterministic": args.deterministic == "on"}


def apply_determinism(args) -> dict:
    """Call right after the engine is built (the NeMo wrapper sets the deterministic mode in its constructor)."""
    import torch
    torch.utils.deterministic.fill_uninitialized_memory = args.fill_uninit == "on"
    if args.sdpa == "fast":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    state = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),
             "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
             "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
             "flash_supported_here": _flash_supported()}
    print("DETERMINISM", state, flush=True)
    return state


def _flash_supported():
    import torch
    try:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return major >= 8                     # PyTorch's flash SDPA kernels: sm80+; otherwise memory-efficient / math
    except Exception:  # noqa: BLE001 - informational only
        return None


def fill(args, table: dict, preset: str) -> dict:
    """Set every preset key the user did not pass (argparse default None) and return the resolved values."""
    if preset not in table:
        raise SystemExit(f"unknown preset {preset!r} (choices: {', '.join(table)})")
    for k, v in table[preset].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return {k: getattr(args, k) for k in table[preset]}


def multi_profile_for(n_sessions: int, profile: str) -> str:
    """The -s4 EarTTS graph pool holds 4 sessions; use the -s8 pool above that (same experiments otherwise)."""
    return profile.replace("-s4", "-s8") if profile.endswith("-s4") and n_sessions > 4 else profile


def safe_enable(name: str, engine, failed: dict) -> object:
    from mynah_v2.model import perf
    try:
        return perf.enable(name, engine)
    except Exception as e:  # noqa: BLE001 - an optional optimisation must never stop a run
        failed[name] = f"{type(e).__name__}: {e}"
        print(f"WARNING optimisation {name!r} could not be enabled on this setup and is skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=2)
        return None


class OptimisationUnavailable(RuntimeError):
    pass


def apply(engine, profile: str | None, exp: str, strict: bool = False) -> dict:
    """Enable a perf profile, then extra experiments, one by one. Returns {"active": {...}, "skipped": {...}}.
    strict=True raises OptimisationUnavailable if any requested optimisation could not be enabled."""
    from mynah_v2.model import perf
    names = list(perf.PROFILES[profile]) if profile else []
    names += [n for n in (exp or "").split(",") if n and n not in names]
    active, failed = {}, {}
    for n in names:
        info = safe_enable(n, engine, failed)
        if n not in failed:
            active[n] = info
    print("OPTIMISATIONS active=" + ",".join(active) + (" skipped=" + ",".join(failed) if failed else ""), flush=True)
    if strict and failed:
        raise OptimisationUnavailable(
            f"strict mode: requested optimisations not available here: {failed}. A performance comparison would not "
            "match the requested configuration; fix the environment, or pass --no-strict to measure the fallback.")
    return {"profile": profile, "strict": strict, "active": {k: str(v)[:300] for k, v in active.items()},
            "skipped": failed}
