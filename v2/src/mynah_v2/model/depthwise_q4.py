"""Fused asymmetric INT4 depthwise Conv1d for the streaming audio codec.

The codec's causal ConvNeXt blocks use small, dense depthwise filters (usually
``kernel_size=7``).  The generic tinygemm path cannot represent these filters
efficiently because their reduction dimension is smaller than its tile.  This
module keeps one Q4 scale/minimum per channel and performs the K-tap dot
product directly from packed nibbles in one Triton launch.

It is deliberately a dynamic experiment rather than a transparent global
replacement: the codec is audio-only, so the caller can compare waveform
quality and real-time factor before enabling it in a deployment profile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def depthwise_q4(x, q, scale, minimum, bias, y, C, T, O, K: tl.constexpr,
                     BLOCK_T: tl.constexpr, HAS_BIAS: tl.constexpr):
        pc = tl.program_id(0)
        pt = tl.program_id(1)
        c = pc % C
        b = pc // C
        t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
        valid_t = t < O
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        sc = tl.load(scale + c).to(tl.float32)
        mn = tl.load(minimum + c).to(tl.float32)
        for k in tl.static_range(0, K):
            xv = tl.load(x + (b * C + c) * T + t + k, mask=valid_t, other=0.0).to(tl.float32)
            packed = tl.load(q + c * ((K + 1) // 2) + k // 2).to(tl.int32)
            nib = tl.where((k & 1) == 0, packed & 0xF, packed >> 4).to(tl.float32)
            acc += xv * (nib * sc + mn)
        if HAS_BIAS:
            acc += tl.load(bias + c).to(tl.float32)
        tl.store(y + (b * C + c) * O + t, acc.to(y.dtype.element_ty), mask=valid_t)

    _KERNEL = depthwise_q4
    return _KERNEL


def _pack_depthwise(weight: torch.Tensor):
    """Quantize ``[C, 1, K]`` to per-channel affine Q4."""
    w = weight.detach().float().reshape(weight.shape[0], weight.shape[-1])
    lo = w.amin(dim=-1)
    hi = w.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp_min(1e-8)
    q = ((w - lo[:, None]) / scale[:, None]).round().clamp(0, 15).to(torch.uint8)
    if q.shape[-1] % 2:
        q = torch.cat((q, torch.zeros(q.shape[0], 1, dtype=q.dtype, device=q.device)), dim=-1)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.contiguous(), scale.contiguous(), lo.contiguous()


class DepthwiseConv1dQ4(nn.Module):
    """Q4 replacement for a stride-1, no-padding depthwise ``Conv1d``."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if not isinstance(conv, nn.Conv1d):
            raise TypeError("DepthwiseConv1dQ4 expects nn.Conv1d")
        if not (conv.in_channels == conv.out_channels == conv.groups and conv.stride == (1,)
                and conv.dilation == (1,) and conv.padding == (0,)):
            raise ValueError("only dense stride-1 depthwise Conv1d is supported")
        q, scale, minimum = _pack_depthwise(conv.weight)
        self.channels = conv.in_channels
        self.kernel_size = conv.kernel_size[0]
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("minimum", minimum)
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise RuntimeError(f"depthwise Q4 expected [B,{self.channels},T], got {tuple(x.shape)}")
        if x.shape[-1] < self.kernel_size:
            # A few codec flush/short-chunk paths can reach the replacement
            # before ConvNeXt's causal wrapper has materialised all left
            # context.  Match the causal path here; an empty chunk stays
            # empty rather than launching a zero-size Triton grid.
            x = F.pad(x, (self.kernel_size - 1, 0))
        out_len = x.shape[-1] - self.kernel_size + 1
        if out_len == 0:
            return x[..., :0]
        if x.device.type != "cuda":
            q = self.q & 0xF
            q_hi = self.q >> 4
            taps = torch.stack((q, q_hi), dim=-1).reshape(self.channels, -1)[:, :self.kernel_size]
            weight = taps.float() * self.scale[:, None] + self.minimum[:, None]
            return F.conv1d(x, weight[:, None, :].to(dtype=x.dtype), self.bias, groups=self.channels)
        import triton
        y = torch.empty((x.shape[0], self.channels, out_len), dtype=x.dtype, device=x.device)
        bias = self.bias if self.bias is not None else self.minimum
        _kernel()[(x.shape[0] * self.channels, triton.cdiv(out_len, 128))](
            x, self.q, self.scale, self.minimum, bias, y,
            self.channels, x.shape[-1], out_len, K=self.kernel_size,
            BLOCK_T=128, HAS_BIAS=self.bias is not None,
        )
        return y

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale, self.minimum))

