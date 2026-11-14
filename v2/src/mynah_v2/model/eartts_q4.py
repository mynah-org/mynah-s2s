"""Graph-safe Q4 linears for the EarTTS backbone.

This is intentionally separate from ``wo_gemv.WOLinear``.  EarTTS executes
classifier-free guidance rows together (normally M=2) inside a CUDA graph;
the old W4 path was a batch-1 GEMV with a Python row loop, dynamic output
allocation, and autotuning.  This module uses one fixed-shape batched Triton
launch, a persistent output buffer, and the W4 packing contract from
``wo_gemv.pack_w4``.

The first call must happen during the graph warm-up, before capture, so Triton
compilation is never attempted during replay.  The module rejects non-contiguous
activations and batches larger than its declared static capacity rather than
silently selecting another kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mynah_v2.model.wo_gemv import pack_w4

_KERNEL = None


def _build_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def q4_mm(x, q, scale, minimum, y, M, N, K, G: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        kh = K // 2
        for k0 in tl.range(0, K, BK):
            rh = k0 // 2 + tl.arange(0, BK // 2)
            xm = (rm[:, None] < M) & (2 * rh[None, :] < K)
            wm = (rn[:, None] < N) & (rh[None, :] < kh)
            b = tl.load(q + rn[:, None] * kh + rh[None, :], mask=wm, other=0).to(tl.int32)
            xe = tl.load(x + rm[:, None] * K + 2 * rh[None, :], mask=xm, other=0.0).to(tl.float32)
            xo = tl.load(x + rm[:, None] * K + 2 * rh[None, :] + 1,
                         mask=xm & (2 * rh[None, :] + 1 < K), other=0.0).to(tl.float32)
            gi = (2 * rh) // G
            gm = (rn[:, None] < N) & (gi[None, :] < K // G)
            sc = tl.load(scale + rn[:, None] * (K // G) + gi[None, :], mask=gm, other=0.0).to(tl.float32)
            mn = tl.load(minimum + rn[:, None] * (K // G) + gi[None, :], mask=gm, other=0.0).to(tl.float32)
            lo = (b & 0xF).to(tl.float32)
            hi = (b >> 4).to(tl.float32)
            acc += tl.sum((lo * sc + mn)[None, :, :] * xe[:, None, :] +
                          (hi * sc + mn)[None, :, :] * xo[:, None, :], axis=2)
        om = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(y + rm[:, None] * N + rn[None, :], acc.to(y.dtype.element_ty), mask=om)

    _KERNEL = q4_mm
    return _KERNEL


class GraphSafeQ4Linear(nn.Module):
    """Static-capacity, batched asymmetric-Q4 linear for EarTTS graph replay."""

    def __init__(self, lin: nn.Linear, group: int = 128, max_batch: int = 16):
        super().__init__()
        if lin.weight.device.type != "cuda":
            raise ValueError("GraphSafeQ4Linear requires CUDA weights")
        if lin.weight.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported EarTTS source dtype: {lin.weight.dtype}")
        p = pack_w4(lin.weight.data, group=group)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.group, self.max_batch = p["group"], max_batch
        self.register_buffer("q", p["q"].contiguous())
        self.register_buffer("scale", p["scale"].contiguous())
        self.register_buffer("minimum", p["min"].contiguous())
        self.register_buffer("weight", torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device),
                            persistent=False)
        self.register_buffer("_out", torch.empty(max_batch, self.out_features, dtype=lin.weight.dtype,
                                                  device=lin.weight.device), persistent=False)
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise RuntimeError(f"EarTTS Q4 K mismatch: got {x.shape[-1]}, expected {self.in_features}")
        x2 = x.reshape(-1, self.in_features)
        if not x2.is_contiguous():
            raise RuntimeError("EarTTS Q4 requires fully contiguous activations")
        m = x2.shape[0]
        if m > self.max_batch:
            raise RuntimeError(f"EarTTS Q4 static batch capacity {self.max_batch} exceeded by M={m}")
        y = self._out[:m]
        kernel = _build_kernel()
        grid = (triton_cdiv(m, 2), triton_cdiv(self.out_features, 64))
        kernel[grid](x2, self.q, self.scale, self.minimum, y, m, self.out_features, self.in_features,
                     G=self.group, BM=2, BN=64, BK=128)
        if self.bias is not None:
            y.add_(self.bias.to(y.dtype))
        return y.reshape(*x.shape[:-1], self.out_features)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale, self.minimum))


def triton_cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def quantize_eartts_int4(engine, group: int = 128, max_batch: int = 16) -> dict:
    """Opt-in Q4 for EarTTS's Gemma backbone only; perception and output heads stay unchanged."""
    duplex = engine.wrapper.model.tts_model
    tts = duplex.tts_model
    backbone = tts.backbone
    before = after = n = 0
    for name, mod in list(backbone.named_modules()):
        if isinstance(mod, nn.Linear):
            before += mod.weight.numel() * mod.weight.element_size()
            q = GraphSafeQ4Linear(mod, group=group, max_batch=max_batch)
            after += q.nbytes() + (mod.bias.numel() * mod.bias.element_size() if mod.bias is not None else 0)
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            setattr(parent, child_name, q)
            n += 1
    if not n:
        raise RuntimeError("no direct EarTTS backbone linears found")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"eartts_quant": "q4_triton_graph_safe", "scope": "tts_model.backbone",
            "linears": n, "group": group, "max_batch": max_batch,
            "fp_bytes": before, "q4_bytes": after}
