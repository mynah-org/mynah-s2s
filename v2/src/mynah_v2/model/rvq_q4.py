"""Fused INT4 RVQ codebook gather/sum for EarTTS depth-sum embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from mynah_v2.model.wo_gemv import pack_w4


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def rvq_embed(code, q, scale, minimum, out, P, D, V, H, G: tl.constexpr,
                  BLOCK_H: tl.constexpr):
        pid = tl.program_id(0)
        ph = tl.program_id(1)
        h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = h < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        half_h = H // 2
        groups = H // G
        for qi in tl.range(0, D):
            idx = tl.load(code + pid * D + qi).to(tl.int32)
            qbase = qi * V * half_h + idx * half_h
            sbase = qi * V * groups + idx * groups
            qb = tl.load(q + qbase + h // 2, mask=mask_h, other=0)
            nib = tl.where((h & 1) == 0, qb & 0xF, qb >> 4).to(tl.float32)
            gi = h // G
            sc = tl.load(scale + sbase + gi, mask=mask_h, other=0.0).to(tl.float32)
            mn = tl.load(minimum + sbase + gi, mask=mask_h, other=0.0).to(tl.float32)
            acc += nib * sc + mn
        tl.store(out + pid * H + h, acc, mask=mask_h)

    _KERNEL = rvq_embed
    return _KERNEL


class RVQInt4Embedding(nn.Module):
    """INT4 codebook gather with an explicit zero sentinel row.

    The NeMo masking path uses codebook index ``V`` as a padded code.  The
    packed table therefore has ``V+1`` rows; the last row is all zeros and is
    never quantized. Accumulation remains FP32 to match the upstream method.
    """

    def __init__(self, rvq_embs: torch.Tensor, group: int = 128):
        super().__init__()
        if rvq_embs.ndim != 3:
            raise ValueError(f"RVQ table must be [D,V,H], got {tuple(rvq_embs.shape)}")
        d, v, h = rvq_embs.shape
        real = pack_w4(rvq_embs.reshape(d * v, h), group=group)
        q = torch.cat([real["q"].reshape(d, v, h // 2),
                       torch.zeros(d, 1, h // 2, dtype=real["q"].dtype, device=rvq_embs.device)], dim=1)
        scale = torch.cat([real["scale"].reshape(d, v, h // real["group"]),
                           torch.zeros(d, 1, h // real["group"], dtype=real["scale"].dtype,
                                       device=rvq_embs.device)], dim=1)
        minimum = torch.cat([real["min"].reshape(d, v, h // real["group"]),
                             torch.zeros(d, 1, h // real["group"], dtype=real["min"].dtype,
                                         device=rvq_embs.device)], dim=1)
        self.depth, self.vocab, self.hidden = d, v, h
        self.group = real["group"]
        self.register_buffer("q", q.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("minimum", minimum.contiguous())

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        shape = code.shape[:-1]
        if code.shape[-1] != self.depth:
            raise ValueError(f"RVQ code depth {code.shape[-1]} != {self.depth}")
        flat = code.reshape(-1, self.depth).contiguous()
        if flat.device.type != "cuda":
            out = torch.zeros(flat.shape[0], self.hidden, dtype=torch.float32, device=flat.device)
            for i in range(self.depth):
                idx = flat[:, i]
                q = self.q[i, idx]
                lo, hi = q & 0xF, q >> 4
                val = torch.stack((lo, hi), -1).reshape(flat.shape[0], self.hidden).float()
                sc = self.scale[i, idx].float().repeat_interleave(self.group, dim=-1)
                mn = self.minimum[i, idx].float().repeat_interleave(self.group, dim=-1)
                out += val * sc + mn
            return out.reshape(*shape, self.hidden)
        import triton
        out = torch.empty(flat.shape[0], self.hidden, dtype=torch.float32, device=flat.device)
        _kernel()[(flat.shape[0], triton.cdiv(self.hidden, 128))](
            flat, self.q, self.scale, self.minimum, out,
            flat.shape[0], self.depth, self.vocab + 1, self.hidden,
            G=self.group, BLOCK_H=128,
        )
        return out.reshape(*shape, self.hidden)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale, self.minimum))
