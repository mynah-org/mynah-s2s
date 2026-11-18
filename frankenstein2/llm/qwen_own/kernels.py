"""Optional Triton kernels with plain-torch fallbacks (arch-agnostic: sm_80+, sm_89 Ada, sm_12x Blackwell). Weights stay
4/8-bit in memory; they are dequantised inside the kernel tiles (no bf16 copy):

  awq_expert_gemv / nvfp4_expert_gemv   decode: y[p] = x[p] @ W[eidx[p]]^T for a few rows (dense layers: E = 1)
  int8_gemv                              decode: int8 or fp8 weights with a per-output-channel scale
  moe_align + moe_gemm                   prefill: (token, expert) pairs sorted by expert, each expert's rows padded to
                                         a multiple of BM, one program per (row block, column block); the block knows
                                         its expert. Formats: AWQ (AutoAWQ GEMM layout) and NVFP4 (ModelOpt W4A16:
                                         e2m1 values, e4m3 scale per 16 inputs, fp32 global scale).
Formats in memory:
  awq   qweight int32 [E, K, N/8], qzeros int32 [E, K/g, N/8], scales fp16 [E, K/g, N]   (K = in, N = out)
  nvfp4 q uint8 [E, N, K/2] (low nibble = even input), s e4m3 [E, N, K/16], s2 fp32 [E]
"""

from __future__ import annotations

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

try:                                                    # pragma: no cover - needs a GPU
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1(n):
        m = n & 7
        e = (m >> 1).to(tl.float32)
        f = (m & 1).to(tl.float32)
        v = tl.where(e == 0.0, f * 0.5, (1.0 + 0.5 * f) * tl.exp2(e - 1.0))
        return tl.where(n >= 8, -v, v)

    @triton.jit
    def _awq_gemv(x_ptr, qw_ptr, qz_ptr, sc_ptr, e_ptr, y_ptr, K, N, G,
                  sxe, sxk, sqe, sqk, sze, szk, sse, ssk, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(e_ptr + p).to(tl.int64)
        cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        cmask = cols < N
        pcol = cols // 8
        shift = ((cols % 2) * 4 + (cols % 8) // 2) * 4
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
    def _nvfp4_gemv(x_ptr, q_ptr, s_ptr, s2_ptr, e_ptr, y_ptr, K, N, sxp, sqe, sqn, sse, ssn,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(e_ptr + p).to(tl.int64)
        rows = nb * BLOCK_N + tl.arange(0, BLOCK_N)                         # output features
        rmask = rows < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        H: tl.constexpr = BLOCK_K // 2
        for k0 in range(0, K, BLOCK_K):
            j = tl.arange(0, H)
            kb = k0 // 2 + j                                                # byte index along K
            xe = tl.load(x_ptr + p * sxp + k0 + 2 * j).to(tl.float32)
            xo = tl.load(x_ptr + p * sxp + k0 + 2 * j + 1).to(tl.float32)
            b = tl.load(q_ptr + e * sqe + rows[:, None] * sqn + kb[None, :], mask=rmask[:, None], other=0).to(tl.int32)
            sc = tl.load(s_ptr + e * sse + rows[:, None] * ssn + (kb // 8)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            acc += tl.sum((_e2m1(b & 0xF) * xe[None, :] + _e2m1(b >> 4) * xo[None, :]) * sc, axis=1)
        s2 = tl.load(s2_ptr + e)
        tl.store(y_ptr + p * N + rows, (acc * s2).to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _int8_gemv(x_ptr, q_ptr, s_ptr, y_ptr, K, N, sxp, sqn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        nb = tl.program_id(1)
        rows = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < K
            xv = tl.load(x_ptr + p * sxp + ks, mask=kmask, other=0.0).to(tl.float32)
            m2 = rmask[:, None] & kmask[None, :]
            q = tl.load(q_ptr + rows[:, None] * sqn + ks[None, :], mask=m2)       # int8 or fp8: no typed `other`
            acc += tl.sum(tl.where(m2, q.to(tl.float32), 0.0) * xv[None, :], axis=1)
        sc = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + p * N + rows, (acc * sc).to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _awq_moe_gemm(x_ptr, qw_ptr, qz_ptr, sc_ptr, be_ptr, y_ptr, K, N, G,
                      sxm, sqe, sqk, sze, szk, sse, ssk, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        mb = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(be_ptr + mb).to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        cols = nb * BN + tl.arange(0, BN)
        pcol = cols // 8
        shift = ((cols % 2) * 4 + (cols % 8) // 2) * 4
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            a = tl.load(x_ptr + rows[:, None] * sxm + ks[None, :])
            q = tl.load(qw_ptr + e * sqe + ks[:, None] * sqk + pcol[None, :])
            gi = ks // G
            z = tl.load(qz_ptr + e * sze + gi[:, None] * szk + pcol[None, :])
            sc = tl.load(sc_ptr + e * sse + gi[:, None] * ssk + cols[None, :]).to(tl.float32)
            w = (((q >> shift[None, :]) & 0xF) - ((z >> shift[None, :]) & 0xF)).to(tl.float32) * sc
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        tl.store(y_ptr + rows[:, None] * sym + cols[None, :], acc.to(tl.bfloat16))

    @triton.jit
    def _nvfp4_moe_gemm(x_ptr, q_ptr, s_ptr, s2_ptr, be_ptr, y_ptr, K, N,
                        sxm, sqe, sqn, sse, ssn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        mb = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(be_ptr + mb).to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        cols = nb * BN + tl.arange(0, BN)                                  # output features
        H: tl.constexpr = BK // 2
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            j = tl.arange(0, H)
            kb = k0 // 2 + j
            ae = tl.load(x_ptr + rows[:, None] * sxm + (k0 + 2 * j)[None, :])
            ao = tl.load(x_ptr + rows[:, None] * sxm + (k0 + 2 * j + 1)[None, :])
            b = tl.load(q_ptr + e * sqe + cols[None, :] * sqn + kb[:, None]).to(tl.int32)        # [H, BN]
            sc = tl.load(s_ptr + e * sse + cols[None, :] * ssn + (kb // 8)[:, None]).to(tl.float32)
            wl = (_e2m1(b & 0xF) * sc).to(tl.bfloat16)
            wh = (_e2m1(b >> 4) * sc).to(tl.bfloat16)
            acc += tl.dot(ae.to(tl.bfloat16), wl) + tl.dot(ao.to(tl.bfloat16), wh)
        s2 = tl.load(s2_ptr + e)
        tl.store(y_ptr + rows[:, None] * sym + cols[None, :], (acc * s2).to(tl.bfloat16))

    @triton.jit
    def _delta_step(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, s_ptr, slot_ptr, o_ptr, H, scale,
                    DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """One decode token of the gated delta rule for one (row, head), in place on the session's fp32 state."""
        bh = tl.program_id(0)
        vb = tl.program_id(1)
        b = bh // H
        h = bh % H
        slot = tl.load(slot_ptr + b).to(tl.int64)
        dk = tl.arange(0, DK)
        dv = vb * BV + tl.arange(0, BV)
        q = tl.load(q_ptr + bh * DK + dk).to(tl.float32) * scale
        k = tl.load(k_ptr + bh * DK + dk).to(tl.float32)
        v = tl.load(v_ptr + bh * DV + dv).to(tl.float32)
        g = tl.exp(tl.load(g_ptr + bh).to(tl.float32))
        beta = tl.load(b_ptr + bh).to(tl.float32)
        base = s_ptr + (slot * H + h) * DK * DV
        S = tl.load(base + dk[:, None] * DV + dv[None, :]) * g
        kv = tl.sum(S * k[:, None], axis=0)
        S = S + k[:, None] * ((v - kv) * beta)[None, :]
        tl.store(base + dk[:, None] * DV + dv[None, :], S)
        tl.store(o_ptr + bh * DV + dv, tl.sum(S * q[:, None], axis=0))

    @triton.jit
    def _chunk_scan(vv_ptr, kc_ptr, qg_ptr, att_ptr, kd_ptr, gl_ptr, s_ptr, o_ptr, NC,
                    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """Inter-chunk recurrence of the chunked gated delta rule for one (row*head, value block): the state block
        [DK, BV] stays on chip across all chunks. Inputs per chunk (fp32, contiguous): vv [C, DV], kcum [C, DK],
        q*exp(g) [C, DK], intra-chunk attention [C, C], k*exp(g_last - g) [C, DK], g_last; state [DK, DV] in/out."""
        bh = tl.program_id(0)
        vb = tl.program_id(1)
        rc = tl.arange(0, C)
        rk = tl.arange(0, DK)
        rv = vb * BV + tl.arange(0, BV)
        S = tl.load(s_ptr + bh * DK * DV + rk[:, None] * DV + rv[None, :])
        for i in range(NC):
            base = bh * NC + i
            kc = tl.load(kc_ptr + base * C * DK + rc[:, None] * DK + rk[None, :])
            vnew = tl.load(vv_ptr + base * C * DV + rc[:, None] * DV + rv[None, :]) - \
                tl.dot(kc, S, input_precision="tf32x3")
            qg = tl.load(qg_ptr + base * C * DK + rc[:, None] * DK + rk[None, :])
            att = tl.load(att_ptr + base * C * C + rc[:, None] * C + rc[None, :])
            o = tl.dot(qg, S, input_precision="tf32x3") + tl.dot(att, vnew, input_precision="tf32x3")
            tl.store(o_ptr + base * C * DV + rc[:, None] * DV + rv[None, :], o)
            kdT = tl.load(kd_ptr + base * C * DK + rc[None, :] * DK + rk[:, None])          # [DK, C]
            gl = tl.load(gl_ptr + base)
            S = S * tl.exp(gl) + tl.dot(kdT, vnew, input_precision="tf32x3")
        tl.store(s_ptr + bh * DK * DV + rk[:, None] * DV + rv[None, :], S)

    @triton.jit
    def _wq_gemm(x_ptr, q_ptr, s_ptr, y_ptr, M, K, N, sxm, sqn, sym,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Prefill GEMM with int8 / fp8 weights dequantised inside the tile: Y = X @ (Q * s)^T, s per output row."""
        mb = tl.program_id(0)
        nb = tl.program_id(1)
        rows = mb * BM + tl.arange(0, BM)
        cols = nb * BN + tl.arange(0, BN)
        rmask = rows < M
        cmask = cols < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            kmask = ks < K
            a = tl.load(x_ptr + rows[:, None] * sxm + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
            wm = kmask[:, None] & cmask[None, :]
            w = tl.load(q_ptr + cols[None, :] * sqn + ks[:, None], mask=wm)
            w = tl.where(wm, w.to(tl.float32), 0.0).to(tl.bfloat16)
            acc += tl.dot(a.to(tl.bfloat16), w)
        sc = tl.load(s_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows[:, None] * sym + cols[None, :], (acc * sc[None, :]).to(tl.bfloat16),
                 mask=rmask[:, None] & cmask[None, :])

    HAVE_TRITON = True
except Exception:                                       # pragma: no cover
    HAVE_TRITON = False


# ---------------------------------------------------------------------------------------------------------------
# torch reference dequantisation
def nvfp4_dequant(q, s, s2) -> torch.Tensor:
    """q uint8 [..., N, K/2], s e4m3 [..., N, K/16], s2 fp32 [...] -> bf16 [..., N, K]."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=q.device, dtype=torch.float32)
    lo, hi = lut[(q & 0xF).long()], lut[(q >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).reshape(*q.shape[:-1], q.shape[-1] * 2)
    w = w.reshape(*w.shape[:-1], -1, 16) * s.float().unsqueeze(-1)
    w = w.reshape(*q.shape[:-1], -1) * s2.float().reshape(*s2.shape, *([1] * 2))
    return w.to(torch.bfloat16)


def _use(x, use_triton):
    return use_triton and HAVE_TRITON and x.is_cuda


def awq_expert_gemv(x, qweight, qzeros, scales, eidx, group, use_triton=True):
    from .quant import awq_dequant
    P, K = x.shape
    N = scales.shape[-1]
    if _use(x, use_triton):
        x = x.contiguous()
        y = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
        _awq_gemv[(P, triton.cdiv(N, 64))](x, qweight, qzeros, scales, eidx, y, K, N, group,
                                            x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
                                            qzeros.stride(0), qzeros.stride(1), scales.stride(0), scales.stride(1),
                                            BLOCK_N=64, BLOCK_K=64)
        return y
    w = awq_dequant(qweight[eidx], qzeros[eidx], scales[eidx], group)       # [P, N, K]
    return (x[:, None, :] @ w.transpose(-1, -2))[:, 0]


def nvfp4_expert_gemv(x, q, s, s2, eidx, use_triton=True):
    P, K = x.shape
    N = q.shape[-2]
    if _use(x, use_triton) and K % 64 == 0:
        x = x.contiguous()
        y = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
        bk = 256 if K % 256 == 0 else 64                                    # tuned on GB10 (sweep in labs/tyger)
        _nvfp4_gemv[(P, triton.cdiv(N, 32))](x, q, s, s2, eidx, y, K, N, x.stride(0), q.stride(0), q.stride(1),
                                              s.stride(0), s.stride(1), BLOCK_N=32, BLOCK_K=bk)
        return y
    w = nvfp4_dequant(q[eidx], s[eidx], s2[eidx])
    return (x[:, None, :] @ w.transpose(-1, -2))[:, 0]


def int8_gemv(x, q, s, use_triton=True):
    """y = x @ (q * s)^T for a few rows: x [P, K] bf16, q [N, K] int8 or float8_e4m3fn, s [N] or [N, 1]."""
    P, K = x.shape
    N = q.shape[0]
    if _use(x, use_triton):
        x = x.contiguous()
        y = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
        _int8_gemv[(P, triton.cdiv(N, 32))](x, q, s.reshape(-1), y, K, N, x.stride(0), q.stride(0), BLOCK_N=32,
                                             BLOCK_K=512)                    # tuned on GB10 (masked tail for any K)
        return y
    return torch.nn.functional.linear(x, (q.float() * s.reshape(-1, 1).float()).to(torch.bfloat16))


# ---------------------------------------------------------------------------------------------------------------
# prefill: block-aligned grouped expert GEMM
def moe_align(flat: torch.Tensor, n_experts: int, bm: int):
    """flat [P] expert id per (token, slot) pair -> (order, pos, total, block_expert): pair order[i] goes to row
    pos[i] of a buffer of `total` rows where each expert owns a run of rows padded to a multiple of bm."""
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    counts = torch.bincount(flat, minlength=n_experts)
    padded = (counts + bm - 1) // bm * bm
    offs = torch.cumsum(padded, 0) - padded
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(flat.numel(), device=flat.device) - first[e_sorted]
    pos = offs[e_sorted] + rank
    total = int(padded.sum())                                               # host sync (prefill only)
    block_expert = torch.repeat_interleave(torch.arange(n_experts, device=flat.device), padded // bm)
    return order, pos, total, block_expert


BM = 32


def moe_gemm(X, ex: dict, name: str, block_expert, awq_group=None):
    """X [total, K] (rows aligned per expert, BM-padded) -> [total, N]; weights `ex[name...]` in AWQ or NVFP4."""
    total, K = X.shape
    X = X.contiguous()
    if name + ".qweight" in ex:
        qw, qz, sc = ex[name + ".qweight"], ex[name + ".qzeros"], ex[name + ".scales"]
        N = sc.shape[-1]
        Y = torch.empty(total, N, dtype=torch.bfloat16, device=X.device)
        _awq_moe_gemm[(total // BM, N // 64)](X, qw, qz, sc, block_expert, Y, K, N, awq_group, X.stride(0),
                                               qw.stride(0), qw.stride(1), qz.stride(0), qz.stride(1), sc.stride(0),
                                               sc.stride(1), Y.stride(0), BM=BM, BN=64, BK=64, num_stages=2,
                                               num_warps=4)
        return Y
    q, s, s2 = ex[name + ".weight"], ex[name + ".weight_scale"], ex[name + ".weight_scale_2"]
    N = q.shape[-2]
    Y = torch.empty(total, N, dtype=torch.bfloat16, device=X.device)
    bn = 128 if N % 128 == 0 else 64                                        # tuned on GB10
    bk = 128 if K % 128 == 0 else 64
    _nvfp4_moe_gemm[(total // BM, N // bn)](X, q, s, s2, block_expert, Y, K, N, X.stride(0), q.stride(0), q.stride(1),
                                             s.stride(0), s.stride(1), Y.stride(0), BM=BM, BN=bn, BK=bk, num_stages=2,
                                             num_warps=4)
    return Y


def delta_step_inplace(q, k, v, g, beta, state, slots):
    """Decode step of the gated delta rule, updating `state[slots]` in place. q, k [B, H, dk] (l2-normalised), v [B, H,
    dv], g, beta [B, H]; state [S, H, dk, dv] fp32 (all session slots). Returns out [B, H, dv] fp32."""
    B, H, DK = q.shape
    DV = v.shape[-1]
    o = torch.empty(B, H, DV, dtype=torch.float32, device=q.device)
    BV = 32 if DV % 32 == 0 else DV
    _delta_step[(B * H, DV // BV)](q.contiguous(), k.contiguous(), v.contiguous(), g.float().contiguous(),
                                   beta.float().contiguous(), state, slots, o, H, DK ** -0.5, DK=DK, DV=DV, BV=BV)
    return o


def chunk_scan(vv, kcum, qg, att, kd, gl, state):
    """Triton inter-chunk scan. vv [BH, n, C, dv], kcum/qg/kd [BH, n, C, dk], att [BH, n, C, C], gl [BH, n], state
    [BH, dk, dv] fp32 (updated in place). Returns out [BH, n, C, dv] fp32."""
    BH, n, C, DV = vv.shape
    DK = kcum.shape[-1]
    o = torch.empty_like(vv)
    BV = 32
    _chunk_scan[(BH, DV // BV)](vv.contiguous(), kcum.contiguous(), qg.contiguous(), att.contiguous(), kd.contiguous(),
                                gl.contiguous(), state, o, n, C=C, DK=DK, DV=DV, BV=BV, num_warps=4,
                                num_stages=1)                  # fp32 tiles: no pipelining (~100 KB smem parts)
    return o


def wq_gemm(x, q, s):
    """Prefill: x [M, K] bf16 @ (q [N, K] int8/fp8 * s [N])^T -> [M, N] bf16, without a bf16 copy of the weight."""
    M, K = x.shape
    N = q.shape[0]
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _wq_gemm[(triton.cdiv(M, 64), triton.cdiv(N, 128))](x, q, s.reshape(-1), y, M, K, N, x.stride(0), q.stride(0),
                                                         y.stride(0), BM=64, BN=128, BK=64, num_stages=2, num_warps=4)
    return y
