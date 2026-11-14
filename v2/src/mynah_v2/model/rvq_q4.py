"""Fused INT4 RVQ codebook gather/sum for EarTTS depth-sum embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from mynah_v2.model.wo_gemv import pack_w4


_KERNEL = None
_ENCODE_KERNELS = None


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


def _encode_kernels():
    """Build the tiled distance/reduce/update kernels lazily."""
    global _ENCODE_KERNELS
    if _ENCODE_KERNELS is not None:
        return _ENCODE_KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def distance(r, q, scale, minimum, values, indices, P, V, H, G: tl.constexpr,
                 BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr):
        pid = tl.program_id(0)
        row = pid // tl.cdiv(V, BLOCK_V)
        vb = pid % tl.cdiv(V, BLOCK_V)
        v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
        valid_v = v < V
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        groups = H // G
        half_h = H // 2
        for h0 in tl.range(0, H, BLOCK_H):
            h = h0 + tl.arange(0, BLOCK_H)
            valid_h = h < H
            rv = tl.load(r + row * H + h, mask=valid_h, other=0.0).to(tl.float32)
            packed = tl.load(q + v[:, None] * half_h + h[None, :] // 2,
                             mask=valid_v[:, None] & valid_h[None, :], other=0)
            nib = tl.where((h[None, :] & 1) == 0, packed & 0xF, packed >> 4).to(tl.float32)
            sc = tl.load(scale + v[:, None] * groups + h[None, :] // G,
                         mask=valid_v[:, None] & valid_h[None, :], other=0.0).to(tl.float32)
            mn = tl.load(minimum + v[:, None] * groups + h[None, :] // G,
                         mask=valid_v[:, None] & valid_h[None, :], other=0.0).to(tl.float32)
            w = nib * sc + mn
            acc += tl.sum((rv[None, :] - w) * (rv[None, :] - w), axis=1)
        block = vb
        acc = tl.where(valid_v, acc, float("inf"))
        local_pos = tl.argmin(acc, axis=0)
        local_idx = tl.sum(tl.where(tl.arange(0, BLOCK_V) == local_pos, v, 0), axis=0)
        local_value = tl.sum(tl.where(tl.arange(0, BLOCK_V) == local_pos, acc, 0.0), axis=0)
        tl.store(values + row * tl.cdiv(V, BLOCK_V) + block, local_value, mask=row < P)
        tl.store(indices + row * tl.cdiv(V, BLOCK_V) + block, local_idx, mask=row < P)

    @triton.jit
    def reduce_update(values, indices, r, code, q, scale, minimum, P, D, depth, H,
                      G: tl.constexpr, NB: tl.constexpr, BLOCK_H: tl.constexpr):
        row = tl.program_id(0)
        best_value = float("inf")
        best_index = 0
        for j in tl.static_range(0, NB):
            value = tl.load(values + row * NB + j)
            index = tl.load(indices + row * NB + j).to(tl.int32)
            take = value < best_value
            best_value = tl.where(take, value, best_value)
            best_index = tl.where(take, index, best_index)
        idx = best_index.to(tl.int32)
        half_h = H // 2
        groups = H // G
        for h0 in tl.range(0, H, BLOCK_H):
            h = h0 + tl.arange(0, BLOCK_H)
            valid_h = h < H
            packed = tl.load(q + idx * half_h + h // 2, mask=valid_h, other=0)
            nib = tl.where((h & 1) == 0, packed & 0xF, packed >> 4).to(tl.float32)
            sc = tl.load(scale + idx * groups + h // G, mask=valid_h, other=0.0).to(tl.float32)
            mn = tl.load(minimum + idx * groups + h // G, mask=valid_h, other=0.0).to(tl.float32)
            pos = row * H + h
            old = tl.load(r + pos, mask=valid_h, other=0.0).to(tl.float32)
            tl.store(r + pos, (old - (nib * sc + mn)).to(r.dtype.element_ty), mask=valid_h)
        # All lanes write the same scalar index. A scalar store avoids a block
        # mask on a scalar pointer and is harmless because the value is equal.
        tl.store(code + row * D + depth, idx)

    _ENCODE_KERNELS = (distance, reduce_update)
    return _ENCODE_KERNELS


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


class RVQInt4Encoder(nn.Module):
    """Optional fused Q4 residual encoder for EarTTS depth-sum code selection.

    Each quantizer stage uses a tiled Q4 distance kernel, a small argmin
    reduction, and a fused Q4 gather/subtract update.  It intentionally does
    not replace the default path: quantization can change an argmin tie and
    therefore later RVQ codes.  The caller must validate generated codes and
    audio before enabling it for a deployment.
    """

    def __init__(self, embedding: RVQInt4Embedding, block_v: int = 32, block_h: int = 128):
        super().__init__()
        self.depth, self.vocab, self.hidden = embedding.depth, embedding.vocab, embedding.hidden
        self.group = embedding.group
        if self.hidden % self.group:
            raise ValueError("RVQ Q4 encoder requires hidden size divisible by group")
        self.block_v, self.block_h = block_v, min(block_h, self.hidden)
        self.register_buffer("q", embedding.q[:, :self.vocab].contiguous())
        self.register_buffer("scale", embedding.scale[:, :self.vocab].contiguous())
        self.register_buffer("minimum", embedding.minimum[:, :self.vocab].contiguous())

    def forward(self, residual: torch.Tensor, code: torch.Tensor, depth_str: int = 0, k: int = 72):
        if residual.shape[-1] != self.hidden or code.shape[-1] != self.depth:
            raise ValueError("RVQ Q4 encoder shape mismatch")
        if residual.device != code.device:
            raise ValueError("RVQ residual and code must share a device")
        if code.dtype not in (torch.int32, torch.int64):
            raise ValueError("RVQ code must be int32 or int64")
        if depth_str < 0 or depth_str > self.depth or k < 0:
            raise ValueError("invalid RVQ depth range")
        r = residual.reshape(-1, self.hidden).contiguous()
        flat_code = code.reshape(-1, self.depth).contiguous()
        if r.shape[0] != flat_code.shape[0]:
            raise ValueError("RVQ residual/code leading dimensions differ")
        end = min(self.depth, depth_str + k)
        if r.shape[0] == 0 or end == depth_str:
            return code
        if r.device.type != "cuda":
            for qi in range(depth_str, end):
                q = self.q[qi] & 0xF
                hi = self.q[qi] >> 4
                taps = torch.stack((q, hi), dim=-1).reshape(self.vocab, self.hidden).float()
                w = taps * self.scale[qi].float().repeat_interleave(self.group, -1) + \
                    self.minimum[qi].float().repeat_interleave(self.group, -1)
                idx = (r[:, None, :] - w[None, :, :]).square().sum(-1).argmin(-1)
                flat_code[:, qi] = idx
                r -= w[idx].to(r.dtype)
            code.reshape(-1, self.depth).copy_(flat_code)
            return code
        import triton
        distance, reduce_update = _encode_kernels()
        p = r.shape[0]
        nb = triton.cdiv(self.vocab, self.block_v)
        values = torch.empty((p, nb), dtype=torch.float32, device=r.device)
        indices = torch.empty((p, nb), dtype=torch.int32, device=r.device)
        grid_distance = (p * nb,)
        for qi in range(depth_str, end):
            distance[grid_distance](r, self.q[qi], self.scale[qi], self.minimum[qi], values, indices,
                                    p, self.vocab, self.hidden, G=self.group,
                                    BLOCK_V=self.block_v, BLOCK_H=self.block_h)
            reduce_update[(p,)](values, indices, r, flat_code, self.q[qi], self.scale[qi], self.minimum[qi],
                                p, self.depth, qi, self.hidden, G=self.group, NB=nb,
                                BLOCK_H=self.block_h)
        code.reshape(-1, self.depth).copy_(flat_code)
        return code

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale, self.minimum))
