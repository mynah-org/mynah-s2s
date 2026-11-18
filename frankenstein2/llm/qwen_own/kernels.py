"""Optional Triton kernels with plain-torch fallbacks (arch-agnostic: sm_80+, sm_89 Ada, sm_12x Blackwell).

awq_expert_gemv(x, qweight, qzeros, scales, eidx, group) -> y
    y[p] = x[p] @ dequant(W[eidx[p]]) for a few rows p (decode): the selected experts' AWQ 4-bit weights are read and
    dequantised inside the kernel, never materialised in bf16. x [P, K] bf16; qweight [E, K, N/8] int32 (AutoAWQ GEMM
    order); qzeros [E, K/g, N/8] int32; scales [E, K/g, N] fp16; eidx [P] int. Returns [P, N] bf16.
"""

from __future__ import annotations

import torch

from .quant import awq_dequant

try:                                                    # pragma: no cover - needs a GPU
    import triton
    import triton.language as tl

    @triton.jit
    def _awq_gemv(x_ptr, qw_ptr, qz_ptr, sc_ptr, e_ptr, y_ptr, K, N, G,
                  sxe, sxk, sqe, sqk, sze, szk, sse, ssk, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(e_ptr + p).to(tl.int64)
        cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)                        # natural output columns
        cmask = cols < N
        pcol = cols // 8                                                   # packed int32 column
        shift = ((cols % 2) * 4 + (cols % 8) // 2) * 4                     # AutoAWQ nibble order
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < K
            xv = tl.load(x_ptr + p * sxe + ks * sxk, mask=kmask, other=0.0).to(tl.float32)
            q = tl.load(qw_ptr + e * sqe + ks[:, None] * sqk + pcol[None, :], mask=kmask[:, None] & cmask[None, :], other=0)
            w = ((q >> shift[None, :]) & 0xF).to(tl.float32)
            gi = ks // G
            z = tl.load(qz_ptr + e * sze + gi[:, None] * szk + pcol[None, :], mask=kmask[:, None] & cmask[None, :], other=0)
            z = ((z >> shift[None, :]) & 0xF).to(tl.float32)
            s = tl.load(sc_ptr + e * sse + gi[:, None] * ssk + cols[None, :], mask=kmask[:, None] & cmask[None, :],
                        other=0.0).to(tl.float32)
            acc += tl.sum(xv[:, None] * ((w - z) * s), axis=0)
        tl.store(y_ptr + p * N + cols, acc.to(tl.bfloat16), mask=cmask)

    @triton.jit
    def _w8_gemv(x_ptr, q_ptr, s_ptr, y_ptr, K, N, sxp, sqn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        rows = nb * BLOCK_N + tl.arange(0, BLOCK_N)                        # output features
        rmask = rows < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < K
            xv = tl.load(x_ptr + p * sxp + ks, mask=kmask, other=0.0).to(tl.float32)
            q = tl.load(q_ptr + rows[:, None] * sqn + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0)
            acc += tl.sum(q.to(tl.float32) * xv[None, :], axis=1)
        sc = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + p * N + rows, (acc * sc).to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _awq_grouped_gemm(x_ptr, qw_ptr, qz_ptr, sc_ptr, eid_ptr, y_ptr, M, K, N, G,
                          sxe, sxm, sqe, sqk, sze, szk, sse, ssk, sye, sym,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        h = tl.program_id(0)
        mb = tl.program_id(1)
        nb = tl.program_id(2)
        e = tl.load(eid_ptr + h).to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        rmask = rows < M
        cols = nb * BN + tl.arange(0, BN)
        pcol = cols // 8
        shift = ((cols % 2) * 4 + (cols % 8) // 2) * 4
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            a = tl.load(x_ptr + h * sxe + rows[:, None] * sxm + ks[None, :], mask=rmask[:, None], other=0.0)
            q = tl.load(qw_ptr + e * sqe + ks[:, None] * sqk + pcol[None, :])
            gi = ks // G
            z = tl.load(qz_ptr + e * sze + gi[:, None] * szk + pcol[None, :])
            sc = tl.load(sc_ptr + e * sse + gi[:, None] * ssk + cols[None, :]).to(tl.float32)
            w = (((q >> shift[None, :]) & 0xF) - ((z >> shift[None, :]) & 0xF)).to(tl.float32) * sc
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        tl.store(y_ptr + h * sye + rows[:, None] * sym + cols[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])

    HAVE_TRITON = True
except Exception:                                       # pragma: no cover
    HAVE_TRITON = False


def awq_expert_gemv(x, qweight, qzeros, scales, eidx, group, use_triton=True):
    P, K = x.shape
    N = scales.shape[-1]
    if use_triton and HAVE_TRITON and x.is_cuda:
        y = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
        BLOCK_N, BLOCK_K = 64, 64
        grid = (P, triton.cdiv(N, BLOCK_N))
        _awq_gemv[grid](x, qweight, qzeros, scales, eidx, y, K, N, group,
                        x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), qzeros.stride(0),
                        qzeros.stride(1), scales.stride(0), scales.stride(1), BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        return y
    w = awq_dequant(qweight[eidx], qzeros[eidx], scales[eidx], group)       # [P, N, K]
    return (x[:, None, :] @ w.transpose(-1, -2))[:, 0]


def w8_gemv(x, q, s, use_triton=True):
    """y = x @ (q * s)^T for a few rows: x [P, K] bf16, q [N, K] int8, s [N, 1] per-output-channel scale."""
    P, K = x.shape
    N = q.shape[0]
    if use_triton and HAVE_TRITON and x.is_cuda:
        x = x.contiguous()
        y = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
        grid = (P, triton.cdiv(N, 64))
        _w8_gemv[grid](x, q, s, y, K, N, x.stride(0), q.stride(0), BLOCK_N=64, BLOCK_K=128)
        return y
    return torch.nn.functional.linear(x, q.to(torch.bfloat16) * s)


def awq_grouped_gemm(X, qweight, qzeros, scales, eids, group, use_triton=True):
    """Prefill: Y[h] = X[h] @ dequant(W[eids[h]]) for each hit expert h, X [n_hit, M, K] bf16 (rows padded per
    expert). Dequantisation happens inside the tiles (tensor-core dot), the bf16 weights are never materialised."""
    nh, M, K = X.shape
    N = scales.shape[-1]
    ok_shape = K % 64 == 0 and N % 64 == 0 and (64 % group == 0 or group % 64 == 0)
    if use_triton and HAVE_TRITON and X.is_cuda and ok_shape:
        X = X.contiguous()
        Y = torch.empty(nh, M, N, dtype=torch.bfloat16, device=X.device)
        BM = 16 if M <= 16 else 32
        grid = (nh, triton.cdiv(M, BM), N // 64)
        try:                                   # 2 pipeline stages: fits ~100 KB shared memory (sm_89, sm_12x)
            _awq_grouped_gemm[grid](X, qweight, qzeros, scales, eids, Y, M, K, N, group,
                                    X.stride(0), X.stride(1), qweight.stride(0), qweight.stride(1), qzeros.stride(0),
                                    qzeros.stride(1), scales.stride(0), scales.stride(1), Y.stride(0), Y.stride(1),
                                    BM=BM, BN=64, BK=64, num_stages=2, num_warps=4)
            return Y
        except Exception as e:                 # out of shared memory on a smaller part: the torch path below
            if "out of resource" not in str(e).lower():
                raise
    w = awq_dequant(qweight[eids], qzeros[eids], scales[eids], group)       # [n_hit, N, K]
    return X @ w.transpose(-1, -2)
