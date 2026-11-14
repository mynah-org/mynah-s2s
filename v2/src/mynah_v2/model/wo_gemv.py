"""EXPERIMENTAL weight-only GEMV kernels (Triton) for the batch-1 Thinker decode: y[M, N] = x[M, K] · deq(W)[N, K]ᵀ.

Why: the decode step is weight-bandwidth-bound (O1: ~14.7 GB of BF16 backbone linears per 80 ms step); a format only
pays off on GB10/sm_121 if a kernel streams fewer bytes at near-peak bandwidth. torch 2.10 ships int4 tinygemm (fast)
but its int8 weight-only kernel is slower than BF16 and fp8 row-wise _scaled_mm is unsupported here (quant.py), so
these are our own, portable (Triton: sm_80+/sm_89 L40S/L4 as well) bandwidth-oriented kernels:
  * w8   — int8 per-output-channel symmetric, fp32 scale                      (1 B/weight)
  * f8   — fp8 e4m3 per-output-channel, fp32 scale                             (1 B/weight)
  * w4   — int4 asymmetric, group 128 along K, bf16 scale + min per group     (0.5 B/weight + 4 B/128)
  * nvf4 — NVFP4: e2m1 codes, e4m3 scale per 16 along K, fp32 tensor scale    (0.5 B/weight + 1 B/16)
Each program owns BLOCK_N output rows and streams their K extent in BLOCK_K tiles (coalesced along K), dequantizes
in registers and accumulates in fp32. Packing helpers are plain torch (CPU-testable); the kernels need CUDA + Triton.
Not used by REFERENCE; serving use only after the precision gate.
"""

from __future__ import annotations

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ------------------------------------------------------------------------------------------------ packing (torch)
ROWS = 4096          # row chunk for packing: bounded FP32 temporaries (L4-sized GPUs)


def _rowwise(w: torch.Tensor, fn):
    qs, ss = zip(*(fn(w[i:i + ROWS].float()) for i in range(0, w.shape[0], ROWS)))
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def pack_w8(w: torch.Tensor):
    def f(wf):
        s = wf.abs().amax(1).clamp_min(1e-8) / 127
        return torch.round(wf / s[:, None]).clamp(-127, 127).to(torch.int8), s
    q, s = _rowwise(w, f)
    return {"fmt": "w8", "q": q, "scale": s}


def pack_f8(w: torch.Tensor):
    def f(wf):
        s = wf.abs().amax(1).clamp_min(1e-8) / 448
        return (wf / s[:, None]).to(torch.float8_e4m3fn), s
    q, s = _rowwise(w, f)
    return {"fmt": "f8", "q": q, "scale": s}


def pack_w4(w: torch.Tensor, group: int = 128):
    n, k = w.shape
    if k % 2:
        raise ValueError(f"w4 needs an even K, got {k}")
    while k % group:                                   # e.g. K = 15680 (mlp down_proj): 128 -> 64
        group //= 2
        if group < 16:
            raise ValueError(f"no group size divides K={k}")
    wf = w.float().reshape(n, k // group, group)
    mn, mx = wf.amin(-1), wf.amax(-1)
    s = (mx - mn).clamp_min(1e-8) / 15
    q = torch.round((wf - mn[..., None]) / s[..., None]).clamp(0, 15).to(torch.uint8).reshape(n, k)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()           # low nibble = even k
    return {"fmt": "w4", "q": packed, "scale": s.to(torch.bfloat16).contiguous(),
            "min": mn.to(torch.bfloat16).contiguous(), "group": group}


def _e2m1_code(x: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(E2M1_VALUES, device=x.device)
    mag = (x.abs()[..., None] - grid).abs().argmin(-1)
    return (mag | ((x < 0).to(torch.long) << 3)).to(torch.uint8)


def pack_nvf4(w: torch.Tensor):
    n, k = w.shape
    if k % 16:
        raise ValueError(f"nvf4 needs K % 16 == 0, got {k}")
    wf = w.float().reshape(n, k // 16, 16)
    g = (wf.abs().amax().clamp_min(1e-12) / (448 * 6)).reshape(1)
    bs8 = (wf.abs().amax(-1) / 6 / g).to(torch.float8_e4m3fn)
    bs = bs8.float().clamp_min(2 ** -9) * g
    codes = _e2m1_code(wf / bs[..., None]).reshape(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return {"fmt": "nvf4", "q": packed, "bscale": bs8.contiguous(), "gscale": g.float().contiguous()}


def dequant(p) -> torch.Tensor:
    """Reference dequantization (fp32), the ground truth for the kernels."""
    f = p["fmt"]
    if f in ("w8", "f8"):
        return p["q"].float() * p["scale"][:, None]
    if f == "w4":
        lo, hi = p["q"] & 0xF, p["q"] >> 4
        q = torch.stack([lo, hi], -1).reshape(p["q"].shape[0], -1).float()
        n, k = q.shape
        g = p["group"]
        return (q.reshape(n, k // g, g) * p["scale"].float()[..., None] + p["min"].float()[..., None]).reshape(n, k)
    if f == "nvf4":
        lo, hi = p["q"] & 0xF, p["q"] >> 4
        c = torch.stack([lo, hi], -1).reshape(p["q"].shape[0], -1).long()
        grid = torch.tensor(E2M1_VALUES, device=c.device)
        v = grid[c & 7] * torch.where(c >= 8, -1.0, 1.0)
        n, k = v.shape
        bs = p["bscale"].float().clamp_min(2 ** -9) * p["gscale"]
        return (v.reshape(n, k // 16, 16) * bs[..., None]).reshape(n, k)
    raise ValueError(f)


def nbytes(p) -> int:
    return sum(v.numel() * v.element_size() for v in p.values() if isinstance(v, torch.Tensor))


# ------------------------------------------------------------------------------------------------ kernels (triton)
_KERNELS = {}


def _build():
    import triton
    import triton.language as tl

    configs = [triton.Config({"BN": bn, "BK": bk}, num_warps=nw) for bn in (4, 8, 16, 32) for bk in (128, 256, 512)
               for nw in (2, 4, 8)]

    @triton.autotune(configs=configs, key=["N", "K", "MP", "FP8"])
    @triton.jit
    def w8_kernel(x, q, s, bias, y, M, N, K, MP: tl.constexpr, HAS_BIAS: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, FP8: tl.constexpr):
        # batched GEMV: the [BN, BK] weight tile is loaded ONCE and used for all M (<= MP) activation rows
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        rm = tl.arange(0, MP)
        acc = tl.zeros((MP, BN), tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            xm = (rm[:, None] < M) & (rk[None, :] < K)
            xv = tl.load(x + rm[:, None] * K + rk[None, :], mask=xm, other=0.0).to(tl.float32)
            m = (rn[:, None] < N) & (rk[None, :] < K)
            if FP8:
                wq = tl.load(q + rn[:, None] * K + rk[None, :], mask=m, other=0.0).to(tl.float32)
            else:
                wq = tl.load(q + rn[:, None] * K + rk[None, :], mask=m, other=0).to(tl.float32)
            acc += tl.sum(wq[None, :, :] * xv[:, None, :], 2)
        sc = tl.load(s + rn, mask=rn < N, other=0.0)
        om = (rm[:, None] < M) & (rn[None, :] < N)
        out = acc * sc[None, :]
        if HAS_BIAS:
            out += tl.load(bias + rn, mask=rn < N, other=0.0)[None, :]
        tl.store(y + rm[:, None] * N + rn[None, :], out.to(y.dtype.element_ty), mask=om)

    @triton.autotune(configs=[c for c in configs if c.kwargs["BK"] >= 128], key=["N", "K"])
    @triton.jit
    def w4_kernel(x, q, s, mn, y, N, K, G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        KH = K // 2
        for k0 in range(0, K, BK):
            rh = k0 // 2 + tl.arange(0, BK // 2)                       # packed byte index
            m = (rn[:, None] < N) & (rh[None, :] < KH)
            b = tl.load(q + rn[:, None] * KH + rh[None, :], mask=m, other=0).to(tl.int32)
            xe = tl.load(x + 2 * rh, mask=2 * rh < K, other=0.0).to(tl.float32)
            xo = tl.load(x + 2 * rh + 1, mask=2 * rh + 1 < K, other=0.0).to(tl.float32)
            gi = (2 * rh) // G
            sc = tl.load(s + rn[:, None] * (K // G) + gi[None, :], mask=m, other=0.0).to(tl.float32)
            mi = tl.load(mn + rn[:, None] * (K // G) + gi[None, :], mask=m, other=0.0).to(tl.float32)
            lo = (b & 0xF).to(tl.float32)
            hi = (b >> 4).to(tl.float32)
            acc += tl.sum((lo * sc + mi) * xe[None, :] + (hi * sc + mi) * xo[None, :], 1)
        tl.store(y + rn, acc.to(y.dtype.element_ty), mask=rn < N)

    @triton.jit
    def _e2m1(c):
        e = (c >> 1) & 3
        mbit = (c & 1).to(tl.float32)
        mag = tl.where(e == 0, mbit * 0.5, (1.0 + mbit * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        return tl.where(c >= 8, -mag, mag)

    @triton.autotune(configs=[c for c in configs if c.kwargs["BK"] >= 128], key=["N", "K"])
    @triton.jit
    def nvf4_kernel(x, q, bs, gs, y, N, K, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        KH = K // 2
        g = tl.load(gs)
        for k0 in range(0, K, BK):
            rh = k0 // 2 + tl.arange(0, BK // 2)
            m = (rn[:, None] < N) & (rh[None, :] < KH)
            b = tl.load(q + rn[:, None] * KH + rh[None, :], mask=m, other=0).to(tl.int32)
            xe = tl.load(x + 2 * rh, mask=2 * rh < K, other=0.0).to(tl.float32)
            xo = tl.load(x + 2 * rh + 1, mask=2 * rh + 1 < K, other=0.0).to(tl.float32)
            bi = (2 * rh) // 16
            sc = tl.load(bs + rn[:, None] * (K // 16) + bi[None, :], mask=m, other=0.0).to(tl.float32)
            sc = tl.maximum(sc, 0.001953125) * g
            acc += tl.sum((_e2m1(b & 0xF) * xe[None, :] + _e2m1(b >> 4) * xo[None, :]) * sc, 1)
        tl.store(y + rn, acc.to(y.dtype.element_ty), mask=rn < N)

    _KERNELS.update(w8=w8_kernel, w4=w4_kernel, nvf4=nvf4_kernel, cdiv=triton.cdiv)


MAX_BATCHED_M = 16          # batched-decode rows per launch (B sessions); more rows (prompt prefill) → dequant matmul


def gemv(x: torch.Tensor, p, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x: (..., K) on CUDA; returns (..., N) in x.dtype. w8/f8: true batched kernel for up to MAX_BATCHED_M rows per
    launch (weights read once; MP = next power of two ≥ rows: 1/2/4/8/16 — round 17: with MP capped at 4, B=8 fell back
    to a full dequant per call, 130 ms/step); more rows (e.g. a prompt prefill) go through a dequantized BF16 matmul. w4/nvf4: one launch per row
    (M=1 decode kernels; batched variants not written yet)."""
    f = p["fmt"]
    n = p["q"].shape[0]
    k = x.shape[-1]
    if x.device.type != "cuda":
        w = dequant(p).to(x.dtype)
        b = None if bias is None else bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, b)
    if not _KERNELS:
        _build()
    x2 = x.reshape(-1, k)
    mrows = x2.shape[0]
    if f in ("w8", "f8"):
        if mrows > MAX_BATCHED_M:
            w = dequant(p).to(x.dtype)
            b = None if bias is None else bias.to(x.dtype)
            return torch.nn.functional.linear(x, w, b)
        x2 = x2.contiguous()
        y = torch.empty(mrows, n, device=x.device, dtype=x.dtype)
        mp = 1 << (mrows - 1).bit_length()                # 1, 2, 4, 8, 16
        grid = lambda meta: (_KERNELS["cdiv"](n, meta["BN"]),)  # noqa: E731
        bias_ptr = bias if bias is not None else p["q"]
        _KERNELS["w8"][grid](x2, p["q"], p["scale"], bias_ptr, y, mrows, n, k,
                               MP=mp, HAS_BIAS=bias is not None, FP8=(f == "f8"))
        return y.reshape(*x.shape[:-1], n)
    outs = []
    for row in x2:
        row = row.contiguous()
        y = torch.empty(n, device=x.device, dtype=x.dtype)
        grid = lambda meta: (_KERNELS["cdiv"](n, meta["BN"]),)  # noqa: E731
        if f == "w4":
            _KERNELS["w4"][grid](row, p["q"], p["scale"], p["min"], y, n, k, G=p["group"])
        elif f == "nvf4":
            _KERNELS["nvf4"][grid](row, p["q"], p["bscale"], p["gscale"], y, n, k)
        else:
            raise ValueError(f)
        outs.append(y)
    y = torch.stack(outs).reshape(*x.shape[:-1], n)
    return y if bias is None else y + bias.to(y.dtype)


class WOLinear(torch.nn.Module):
    """Drop-in for an nn.Linear with weight-only quantized storage (fmt w8 | f8 | w4 | nvf4). Keeps an empty
    ``weight`` buffer with the original dtype/device for upstream device/dtype checks (as quant.Int4Linear)."""

    def __init__(self, lin: torch.nn.Linear, fmt: str = "w8"):
        super().__init__()
        self.in_features, self.out_features, self.fmt = lin.in_features, lin.out_features, fmt
        p = PACKERS[fmt](lin.weight.data)
        self._keys = [k for k, v in p.items() if isinstance(v, torch.Tensor)]
        for k2 in self._keys:
            self.register_buffer("p_" + k2, p[k2])
        self._meta = {k2: v for k2, v in p.items() if not isinstance(v, torch.Tensor)}
        self.register_buffer("weight", torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device), persistent=False)
        self.bias = lin.bias

    def packed(self):
        return {**self._meta, **{k: getattr(self, "p_" + k) for k in self._keys}}

    def forward(self, x):
        return gemv(x, self.packed(), self.bias)

    def nbytes(self):
        return nbytes(self.packed())


PACKERS = {"w8": pack_w8, "f8": pack_f8, "w4": pack_w4, "nvf4": pack_nvf4}
