"""EXPERIMENTAL (opt-in, ``--eartts-rng session``): per-session, counter-based EarTTS sampling noise.

Why: the EarTTS samples every step (Gumbel-max over the MoG mixture components, ear_tts_model.py:588 via gumbel_like;
Gaussian noise on the chosen component, generate_step). Upstream draws both from torch's DEFAULT CUDA generator, which
is process-wide: in multi-session runs every session thread (and every FC thread) consumes the same stream, so a
session's audio depends on its neighbours and on thread timing — this is ALREADY true on the per-session (B=1)
multi-session path, not only in the batched step (one draw for all rows there).

How: noise is a pure function of (session seed, session EarTTS step, draw index within the step, element index within
the row): a 32-bit integer hash (int64 tensor ops, products < 2^63) → uniform in (0, 1) → Gumbel / Box-Muller normal.
Per-row (seed, step) live in a static device buffer [rows, 2] read inside the step (so CUDA graphs replay it), the
draw index is a host counter reset per forward (same sequence in eager, capture and replay). The same session with
the same seed therefore gets the same noise whatever its neighbours or its row (B=1 vs batched, any slot); codes are
then equal up to the floating-point differences of the batched GEMMs. Not the upstream distribution's RNG stream
(statistically equivalent: uniform / Gumbel / normal) — default off.
"""

from __future__ import annotations

import math
import sys
import threading

import torch

M32 = 0xFFFFFFFF
K1, K2, K3 = 0x2C1B3C6D, 0x297A2D39, 0x1B873593        # odd multipliers < 2^31: x (< 2^32) * k stays < 2^63


class _Ctx(threading.local):
    def __init__(self):
        self.buf = None          # [rows, 2] int64 device tensor (seed, step) of the step being run
        self.draw = 0
        self.seed = None         # pending session seed for a DynamicCache → StaticKV conversion in this thread


CTX = _Ctx()
STATE = {"enabled": False, "orig_gumbel": None, "ns": None, "draws": 0, "fallback_global": 0}


def _mix(x: torch.Tensor) -> torch.Tensor:
    x = ((x ^ (x >> 16)) * K1) & M32
    x = ((x ^ (x >> 15)) * K2) & M32
    return x ^ (x >> 16)


def uniform(shape_like: torch.Tensor, stream: int) -> torch.Tensor:
    """U(0, 1) noise shaped like ``shape_like`` ([rows, ...]); row r uses CTX.buf[r]."""
    buf = CTX.buf
    rows = shape_like.shape[0]
    per = shape_like[0].numel() if rows else 0
    e = torch.arange(per, device=shape_like.device, dtype=torch.long)
    seed, step = buf[:rows, 0:1] & M32, buf[:rows, 1:2] & M32
    key = _mix(_mix(seed ^ ((step * K3) & M32)) ^ (stream & M32))          # [rows, 1]
    h = _mix((key ^ ((e[None, :] * K1) & M32)) & M32)                     # [rows, per]
    u = ((h >> 8).to(torch.float32) + 0.5) * (1.0 / (1 << 24))
    return u.view(shape_like.shape).to(shape_like.dtype if shape_like.is_floating_point() else torch.float32)


def _next_stream() -> int:
    CTX.draw += 1
    STATE["draws"] += 1
    return CTX.draw * 2


def randn_like(t: torch.Tensor) -> torch.Tensor:
    """generate_step's Gaussian noise (perf._make_generate_step calls this)."""
    if not STATE["enabled"] or CTX.buf is None:
        if STATE["enabled"]:
            STATE["fallback_global"] += 1
        return torch.randn_like(t)
    s = _next_stream()
    u1, u2 = uniform(t, s).float(), uniform(t, s + 1).float()
    return (torch.sqrt(-2.0 * torch.log(u1)) * torch.cos((2.0 * math.pi) * u2)).to(t.dtype)


def gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Drop-in for ear_tts_model.gumbel_like (same formula, counter-based uniform)."""
    if not STATE["enabled"] or CTX.buf is None:
        if STATE["enabled"]:
            STATE["fallback_global"] += 1
        return STATE["orig_gumbel"](tensor, eps)
    u = uniform(tensor, _next_stream()).to(tensor.dtype)
    return -torch.log(-torch.log(u + eps) + eps)


class rows:
    """Context for one EarTTS forward: per-row (seed, step) buffer, draw counter reset."""

    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        self.prev = (CTX.buf, CTX.draw)
        CTX.buf, CTX.draw = (self.buf if STATE["enabled"] else None), 0
        return self

    def __exit__(self, *a):
        CTX.buf, CTX.draw = self.prev
        return False


def session_seed() -> int:
    """Seed for a session's EarTTS stream: the one the rendezvous set for this thread's request, else the process's
    torch seed (single-session runs: seed_all / manual_seed)."""
    s = CTX.seed
    return int(s if s is not None else torch.initial_seed()) & M32


def enable(engine) -> dict:
    """Patch ear_tts_model.gumbel_like (module global used by MoGHead.infer); generate_step (perf host schedule)
    already routes its noise through randn_like above. Must run before any EarTTS graph is captured."""
    from mynah_v2.model.perf import _find_eartts
    m = _find_eartts(engine)
    ns = sys.modules[type(m).__module__]
    if STATE["enabled"]:
        return {"already": True}
    if "generate_step" not in m.__dict__:
        raise RuntimeError("eartts_rng needs the host-schedule generate_step (eartts_graph_full / eartts_host_schedule)")
    STATE["orig_gumbel"], STATE["ns"] = ns.gumbel_like, ns
    ns.gumbel_like = gumbel_like
    STATE["enabled"] = True
    return {"module": ns.__name__}


def disable(engine=None) -> None:
    if STATE["enabled"] and STATE["ns"] is not None:
        STATE["ns"].gumbel_like = STATE["orig_gumbel"]
    STATE["enabled"] = False
