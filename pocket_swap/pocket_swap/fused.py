"""Fused dequant-add-matmul for the on-the-fly modes: y = x @ (T_base + T_lang)^T for a few rows of x.

Each term is (precision, codes, scales) with precision bf16 (dense), int8, int4 (packed) or fp8 (e4m3), or absent:
  ondemand          base bf16  + language delta int8/int4/bf16
  ondemand, base_q  base int8/int4 + language delta int8/int4  (base quantised once, shared)
  weight-only       no base    + language weights int8/int4/fp8 (no bf16 copy of the backbone anywhere)
Each program owns BLOCK_N output features, walks K in BLOCK_K tiles, rebuilds the weight tile in registers (rounded
to bf16 like a materialised slot) and accumulates x @ tile^T in fp32 with tl.dot (rows of x padded to 16). The full
weight is never written to memory.

Portable: `available(t)` is False without Triton, on CPU tensors, or below sm_80 (no bf16 tl.dot; fp8 terms need
sm_89+); callers then use the plain torch path. TRITON_INTERPRET=1 runs the kernel on CPU for the tests.
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
CODE = {None: 0, "bf16": 16, "int8": 8, "int4": 4, "fp8": 9}


def available(t: torch.Tensor, precisions=()) -> bool:
    if not HAVE_TRITON or os.environ.get("POCKET_SWAP_FUSED", "1") == "0":
        return False
    if _INTERPRET:
        return os.environ.get("POCKET_SWAP_INTERPRET_FP8") == "1" or "fp8" not in precisions
    if not t.is_cuda:
        return False
    cap = torch.cuda.get_device_capability(t.device)
    return cap >= (8, 9) if "fp8" in precisions else cap >= (8, 0)


if HAVE_TRITON:

    @triton.jit
    def _term(P, S, rn, rk, k0, sp, ss, nmask, PREC: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
        if PREC == 16:
            t = tl.load(P + rn[:, None] * sp + rk[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        else:
            s = tl.load(S + rn[:, None] * ss + (rk // GROUP)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
            if PREC == 4:
                rk2 = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                p = tl.load(P + rn[:, None] * sp + rk2[None, :], mask=nmask[:, None], other=0x88).to(tl.int32)
                lo = ((p & 15) - 8).to(tl.float32)
                hi = ((p >> 4) - 8).to(tl.float32)
                q = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
            elif PREC == 9:  # fp8 e4m3 codes passed as raw bytes: bitcast, then widen
                q = tl.load(P + rn[:, None] * sp + rk[None, :], mask=nmask[:, None], other=0)
                q = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:  # int8
                q = tl.load(P + rn[:, None] * sp + rk[None, :], mask=nmask[:, None], other=0).to(tl.float32)
            t = q * s
        return t

    @triton.jit
    def _kernel(X, B, BS, Q, S, Y, M, N, K, sxm, sbn, sbs, sqn, ssn, sym,
                BPREC: tl.constexpr, PREC: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, DOT_BF16: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        nmask = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            w = _term(Q, S, rn, rk, k0, sqn, ssn, nmask, PREC, GROUP, BLOCK_N, BLOCK_K)
            if BPREC != 0:
                w += _term(B, BS, rn, rk, k0, sbn, sbs, nmask, BPREC, GROUP, BLOCK_N, BLOCK_K)
            w = w.to(tl.bfloat16)  # the weight a materialised slot would hold
            x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=(rm < M)[:, None], other=0.0).to(tl.bfloat16)
            if DOT_BF16:
                acc += tl.dot(x, tl.trans(w))
            else:  # interpreter (CPU tests): no bf16 matmul in numpy
                acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
        tl.store(Y + rm[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty), mask=(rm < M)[:, None] & nmask[None, :])


    @triton.jit
    def _gkernel(X, B, BS, QP, SP, R0, RC, Y, N, K, sxm, sbn, sbs, sqn, ssn, sym,
                 BPREC: tl.constexpr, PREC: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, DOT_BF16: tl.constexpr):
        g = tl.program_id(0)            # the group (language) varies fastest: one base tile, every language, in L2
        pid = tl.program_id(1)
        r0 = tl.load(R0 + g)
        rc = tl.load(RC + g)
        a = tl.load(QP + g)
        if PREC == 16:
            Q = a.to(tl.pointer_type(tl.bfloat16))
        elif PREC == 8:
            Q = a.to(tl.pointer_type(tl.int8))
        else:  # int4 (packed) and fp8 codes: bytes
            Q = a.to(tl.pointer_type(tl.uint8))
        S = tl.load(SP + g).to(tl.pointer_type(tl.bfloat16))
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        nmask = rn < N
        mmask = rm < rc
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            # exactly fused_linear2's arithmetic (same order, no branch): language term, then += base term
            w = _term(Q, S, rn, rk, k0, sqn, ssn, nmask, PREC, GROUP, BLOCK_N, BLOCK_K)
            w += _term(B, BS, rn, rk, k0, sbn, sbs, nmask, BPREC, GROUP, BLOCK_N, BLOCK_K)
            w = w.to(tl.bfloat16)
            x = tl.load(X + (r0 + rm)[:, None] * sxm + rk[None, :], mask=mmask[:, None], other=0.0).to(tl.bfloat16)
            if DOT_BF16:
                acc += tl.dot(x, tl.trans(w))
            else:
                acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
        tl.store(Y + (r0 + rm)[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty),
                 mask=mmask[:, None] & nmask[None, :])


class GroupedTerms:
    """The per-language terms of ONE linear for a fixed list of languages, ready for ``fused_linear_grouped``:
    device arrays of code / scale pointers and a has-delta flag per language (False = the base language itself).
    Every language must use the same delta precision and shapes; the base term is shared."""

    def __init__(self, terms: list, base: tuple, device):
        """terms: per language (precision, codes, scales), or None for a language without a delta (the base language
        itself): its rows must be given a count of 0 (the caller computes them apart); its pointers are another
        language's, so the kernel never reads invalid memory."""
        precs = {t[0] for t in terms if t is not None}
        if len(precs) > 1:
            raise ValueError(f"mixed delta precisions {sorted(precs)}")
        self.prec = precs.pop() if precs else base[0]
        ref = next((t for t in terms if t is not None), None)
        if ref is not None and any(t is not None and (t[1].shape != ref[1].shape or t[1].stride() != ref[1].stride()
                                                    or (t[2] is not None and t[2].stride() != ref[2].stride()))
                                   for t in terms):
            raise ValueError("delta shapes differ")
        self.base = base           # the callers keep the language tensors alive (a language in use is never evicted)
        if ref is None:
            raise ValueError("no language with a delta")
        fill = [t if t is not None else ref for t in terms]
        self.qp = torch.tensor([t[1].data_ptr() for t in fill], dtype=torch.int64, device=device)
        self.sp = torch.tensor([(t[2] if t[2] is not None else t[1]).data_ptr() for t in fill], dtype=torch.int64,
                               device=device)
        self.has = torch.tensor([t is not None for t in terms], dtype=torch.int32, device=device)
        self.sqn = ref[1].stride(0) if ref is not None else 0
        self.ssn = (ref[2].stride(0) if ref[2] is not None else ref[1].stride(0)) if ref is not None else 0


def fused_linear_grouped(x: torch.Tensor, n_out: int, gt: "GroupedTerms", r0: torch.Tensor, rc: torch.Tensor,
                         group: int) -> torch.Tensor:
    """x [M, K]: rows sorted by language, language g owning rows r0[g] .. r0[g] + rc[g] (at most MAX_ROWS each).
    -> [M, n_out] = per row x @ (T_base + T_lang(row))^T, the same arithmetic as ``fused_linear2`` row by row."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
    y = torch.empty((M, n_out), device=x.device, dtype=out_dtype)
    bp, bq, bs = gt.base
    bs = bs if bs is not None else bq
    g = K if group <= 0 else group
    block_k = 64 if g % 64 == 0 else g
    G = gt.qp.shape[0]
    grid = (G, triton.cdiv(n_out, 16))
    _gkernel[grid](x2, bq, bs, gt.qp, gt.sp, r0, rc, y, n_out, K, x2.stride(0), bq.stride(0), bs.stride(0),
                   gt.sqn, gt.ssn, y.stride(0), BPREC=CODE[bp], PREC=CODE[gt.prec], GROUP=g, BLOCK_N=16, BLOCK_K=block_k, BLOCK_M=16,
                   DOT_BF16=not _INTERPRET)
    return y


def fused_linear2(x: torch.Tensor, n_out: int, lang: tuple, base: tuple | None, group: int) -> torch.Tensor:
    """x [..., K] (at most MAX_ROWS rows) -> [..., n_out]; lang/base = (precision, codes, scales or None)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    if M > MAX_ROWS:
        raise ValueError("fused path is for at most 16 rows")
    out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
    y = torch.empty((M, n_out), device=x.device, dtype=out_dtype)
    lp, lq, ls = lang
    if lp == "fp8":
        lq = lq.view(torch.uint8)
    bp, bq, bs = base if base is not None else (None, lq, ls)
    ls = ls if ls is not None else lq
    bs = bs if bs is not None else bq
    g = K if group <= 0 else group
    block_k = 64 if g % 64 == 0 else g
    grid = (triton.cdiv(n_out, 16),)
    _kernel[grid](x2, bq, bs, lq, ls, y, M, n_out, K, x2.stride(0), bq.stride(0), bs.stride(0), lq.stride(0),
                  ls.stride(0), y.stride(0), BPREC=CODE[bp], PREC=CODE[lp], GROUP=g, BLOCK_N=16, BLOCK_K=block_k,
                  BLOCK_M=16, DOT_BF16=not _INTERPRET)
    return y.reshape(*lead, n_out)


def fused_linear(x: torch.Tensor, base: torch.Tensor, precision: str, codes: torch.Tensor,
                 scales: torch.Tensor | None, group: int) -> torch.Tensor:
    """Backward-compatible form: bf16 base + delta of `precision`."""
    return fused_linear2(x, base.shape[0], (precision, codes, scales), ("bf16", base, None), group)
