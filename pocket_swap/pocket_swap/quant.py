"""Symmetric per-group quantisation of weight deltas (int8 / int4), pure torch.

A 2-D tensor [rows, cols] is split along cols into groups of `group` values (group = cols means per-row).
Each group gets one scale = max|x| / qmax, stored in bf16. int8 codes are stored as int8; int4 codes are
offset by +8 (range 1..15) and packed two per uint8 (low nibble = even column, high nibble = odd column).
"""

from __future__ import annotations

import torch

QMAX = {8: 127, 4: 7}


def _check(x: torch.Tensor, bits: int, group: int) -> int:
    if bits not in QMAX:
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    if x.dim() != 2:
        raise ValueError("only 2-D tensors are quantised")
    cols = x.shape[1]
    g = cols if group <= 0 else group
    if cols % g:
        raise ValueError(f"cols={cols} not divisible by group={g}")
    if bits == 4 and cols % 2:
        raise ValueError("int4 packing needs an even number of columns")
    return g


def quantize(x: torch.Tensor, bits: int, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (codes, scales). codes: int8 [r, c] (bits=8) or uint8 [r, c//2] (bits=4); scales: bf16 [r, c//g]."""
    g = _check(x, bits, group)
    r, c = x.shape
    qmax = QMAX[bits]
    xf = x.float().reshape(r, c // g, g)
    scale = (xf.abs().amax(-1, keepdim=True) / qmax).to(torch.bfloat16)
    s = scale.float().clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(xf / s).clamp_(-qmax, qmax).to(torch.int16).reshape(r, c)
    if bits == 8:
        codes = q.to(torch.int8)
    else:
        u = (q + 8).to(torch.uint8)
        codes = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return codes, scale.reshape(r, c // g)


def dequantize(codes: torch.Tensor, scales: torch.Tensor, bits: int, cols: int,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of `quantize`; runs on whatever device the inputs live on."""
    r = codes.shape[0]
    ng = scales.shape[1]
    g = cols // ng
    if bits == 8:
        q = codes.to(dtype)
    elif bits == 4:
        lo = (codes & 0x0F).to(torch.int16) - 8
        hi = (codes >> 4).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).reshape(r, cols).to(dtype)
    else:
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    return (q.reshape(r, ng, g) * scales.to(dtype).reshape(r, ng, 1)).reshape(r, cols)


def packed_nbytes(rows: int, cols: int, bits: int, group: int = 128) -> int:
    g = cols if group <= 0 else group
    return rows * cols * bits // 8 + rows * (cols // g) * 2
