"""Fused dequant-add-matmul for mode b (ondemand): y = x @ (W_base + dequant(delta))^T for a few rows of x.

One Triton kernel per precision; each program owns BLOCK_N output features, walks K in BLOCK_K tiles, rebuilds
the weight tile in registers (W_base tile + codes * scales, rounded to bf16 exactly like slot materialisation) and
accumulates x @ tile^T in fp32 with tl.dot (rows of x padded to 16). The full weight is never written to memory.

Portable: `available(device)` is False without Triton, on CPU tensors, or on GPUs older than sm_80 (no bf16
tl.dot); callers then use the plain torch path in ondemand.DeltaLinear. TRITON_INTERPRET=1 runs the kernels on CPU
(used by the tests for correctness against the torch reference).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

MAX_ROWS = 16
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def available(t: torch.Tensor) -> bool:
    if not HAVE_TRITON or os.environ.get("POCKET_SWAP_FUSED", "1") == "0":
        return False
    if _INTERPRET:
        return True
    return t.is_cuda and torch.cuda.get_device_capability(t.device) >= (8, 0)


if HAVE_TRITON:

    @triton.jit
    def _kernel(X, B, Q, S, Y, M, N, K, sxm, sbn, sqn, ssn, sym,
                PREC: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                BLOCK_M: tl.constexpr, DOT_BF16: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        nmask = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            b = tl.load(B + rn[:, None] * sbn + rk[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            if PREC == 16:
                d = tl.load(Q + rn[:, None] * sqn + rk[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
                w = b + d
            else:
                s = tl.load(S + rn[:, None] * ssn + (rk // GROUP)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
                if PREC == 8:
                    q = tl.load(Q + rn[:, None] * sqn + rk[None, :], mask=nmask[:, None], other=0).to(tl.float32)
                else:
                    rk2 = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                    p = tl.load(Q + rn[:, None] * sqn + rk2[None, :], mask=nmask[:, None], other=0x88).to(tl.int32)
                    lo = ((p & 15) - 8).to(tl.float32)
                    hi = ((p >> 4) - 8).to(tl.float32)
                    q = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
                w = b + q * s
            w = w.to(tl.bfloat16)  # the weight a materialised slot would hold
            x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=(rm < M)[:, None], other=0.0).to(tl.bfloat16)
            if DOT_BF16:
                acc += tl.dot(x, tl.trans(w))
            else:  # interpreter (CPU tests): no bf16 matmul in numpy
                acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
        tl.store(Y + rm[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty), mask=(rm < M)[:, None] & nmask[None, :])


def fused_linear(x: torch.Tensor, base: torch.Tensor, precision: str, codes: torch.Tensor,
                 scales: torch.Tensor | None, group: int) -> torch.Tensor:
    """x [..., K] (at most MAX_ROWS rows after flattening) -> [..., N] in x's dtype (bf16 compute)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    N = base.shape[0]
    if M > MAX_ROWS:
        raise ValueError("fused path is for at most 16 rows")
    out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    prec = {"bf16": 16, "int8": 8, "int4": 4}[precision]
    g = K if group <= 0 else group
    block_k = 64 if g % 64 == 0 else g
    s = scales if scales is not None else codes  # unused for bf16
    grid = (triton.cdiv(N, 16),)
    _kernel[grid](x2, base, codes, s, y, M, N, K, x2.stride(0), base.stride(0), codes.stride(0), s.stride(0),
                  y.stride(0), PREC=prec, GROUP=g, BLOCK_N=16, BLOCK_K=block_k, BLOCK_M=16,
                  DOT_BF16=not _INTERPRET)
    return y.reshape(*lead, N)
