"""INT4 for the text LLM with a separate multi-token path (frankenstein only; v2/ is not edited).

V2's ``Int4Linear`` (mynah_v2/model/quant.py) runs torch's tinygemm ``_weight_int4pack_mm``: excellent for one row
(the decode step, M == 1) but slow for long prefills (e1j: Nano first token 1.6-2.6 s per step instead of ~0.3 s).
``Int4Dispatch`` wraps one ``Int4Linear``:
  M == 1   the wrapped Int4Linear forward, unchanged (bit-identical to before);
  M  > 1   the int4 weight is dequantized into a bf16 scratch buffer shared by all layers of the same shape, and one
           cuBLAS GEMM (F.linear) runs on it. V2 has no batched int4 kernel for this weight format (its W8/W4 batched
           Triton kernel in wo_gemv.py covers <= 16 rows of its OWN packing), hence dequant + GEMM.
The dequantization uses the same quantized values and the same bf16 scales / zeros as the kernel
(w = q * scale + (zero - 8 * scale), tinygemm convention): q is recomputed once from the bf16 weight with V2's own
quantizer math (min-max per group, no clip search) and kept as 4-bit nibbles (N x K/2 bytes, as large as the packed
weight: the INT4 footprint doubles, still ~1/2 of bf16).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nibbles(w: torch.Tensor, group: int) -> torch.Tensor:
    """q in [0, 15] exactly as mynah_v2.model.quant.quantize_int4 (clip=False) computes it, packed 2 per byte."""
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = wf.amin(-1), wf.amax(-1)
    scale = (mx - mn).clamp_min(1e-8) / 15
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.uint8).reshape(n, k)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()


class Int4Dispatch(nn.Module):
    def __init__(self, q4: nn.Module, nib: torch.Tensor, scratch: dict):
        super().__init__()
        self.q4 = q4
        self.register_buffer("nib", nib, persistent=False)
        self.scratch = scratch
        self.in_features, self.out_features, self.group = q4.in_features, q4.out_features, q4.group
        self.calls = {"m1": 0, "mN": 0}

    @property
    def weight(self):                      # the remote code reads .weight.device / .dtype
        return self.q4.weight

    @property
    def bias(self):
        return self.q4.bias

    def dequant(self) -> torch.Tensor:
        n, k, g = self.out_features, self.in_features, self.group
        key = (n, k, self.nib.device)
        buf = self.scratch.get(key)
        if buf is None:
            buf = self.scratch[key] = torch.empty(n, k, dtype=torch.bfloat16, device=self.nib.device)
        q = torch.stack([self.nib & 0xF, self.nib >> 4], -1).reshape(n, k).float()
        sz = self.q4.scales_zeros.float()                     # [K/g, N, 2] (scale, zero), bf16 values
        scale, zero = sz[..., 0].t(), sz[..., 1].t()         # [N, K/g]
        w = q.view(n, k // g, g) * scale[..., None] + (zero - 8 * scale)[..., None]
        buf.copy_(w.view(n, k))
        return buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if rows == 1:
            self.calls["m1"] += 1
            return self.q4(x)
        self.calls["mN"] += 1
        w = self.dequant()
        y = F.linear(x.to(torch.bfloat16), w, None)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype)


def quantize_int4_dispatch(llm, max_group: int = 128) -> dict:
    """Replace the backbone linears of a NemotronH ``llm`` (TARGETS of V2's quantizer) by Int4Dispatch(Int4Linear)."""
    from mynah_v2.model.quant import TARGETS, Int4Linear
    scratch: dict = {}
    n = 0
    for layer in llm.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        for name, mod in list(mixer.named_children()):
            if name in TARGETS and isinstance(mod, nn.Linear):
                q4 = Int4Linear(mod, max_group=max_group, clip=False)
                setattr(mixer, name, Int4Dispatch(q4, _nibbles(mod.weight.data.to(torch.bfloat16), q4.group), scratch))
                n += 1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"int4_dispatch_linears": n, "scratch_shapes": len(scratch), "max_group": max_group}
