"""Selected-mixture INT4 kernel for the EarTTS MoG mean projection.

EarTTS does not use ``proj_mus`` as a normal Linear during inference: it picks
one mixture matrix per row and performs ``x @ W[index].T``.  Quantizing that
Linear with the ordinary tinygemm module would break the direct ``weight.view``
access, so this module keeps the selected-matrix contract and dequantizes each
4-bit nibble inside one Triton launch.
"""

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
    def selected_q4(x, q, scale, minimum, ids, y, B, D, K, G: tl.constexpr,
                   BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr):
        bid = tl.program_id(0)
        pd = tl.program_id(1)
        d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = d < D
        idx = tl.load(ids + bid).to(tl.int32)
        half_k = K // 2
        groups = K // G
        q_base = idx * D * half_k + d[:, None] * half_k
        s_base = idx * D * groups + d[:, None] * groups
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            mask_k = k < K
            qb = tl.load(q + q_base + (k[None, :] // 2), mask=mask_d[:, None] & mask_k[None, :], other=0)
            nib = tl.where((k[None, :] & 1) == 0, qb & 0xF, qb >> 4).to(tl.float32)
            gi = k // G
            sc = tl.load(scale + s_base + gi[None, :], mask=mask_d[:, None] & mask_k[None, :], other=0.0)
            mn = tl.load(minimum + s_base + gi[None, :], mask=mask_d[:, None] & mask_k[None, :], other=0.0)
            xv = tl.load(x + bid * K + k, mask=mask_k, other=0.0).to(tl.float32)
            acc += tl.sum((nib * sc.to(tl.float32) + mn.to(tl.float32)) * xv[None, :], axis=1)
        tl.store(y + bid * D + d, acc.to(y.dtype.element_ty), mask=mask_d)

    _KERNEL = selected_q4
    return _KERNEL


class SelectedMoGQ4(nn.Module):
    """INT4 mixture matrices with the same selected-row semantics as NeMo."""

    def __init__(self, lin: nn.Linear, num_predictions: int, out_size: int, group: int = 128):
        super().__init__()
        if lin.out_features != num_predictions * out_size:
            raise ValueError("MoG projection shape is not num_predictions * out_size")
        n, d, k = num_predictions, out_size, lin.in_features
        self._init_packed(lin.weight.data.reshape(n, d, k), group)
        self.register_buffer("bias", lin.bias.detach().clone() if lin.bias is not None else torch.empty(0),
                             persistent=False)

    @classmethod
    def from_tensor(cls, weight: torch.Tensor, group: int = 128) -> "SelectedMoGQ4":
        if weight.ndim != 3:
            raise ValueError("selected MoG weight must be [num_predictions,out_size,in_features]")
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        out._init_packed(weight, group)
        out.register_buffer("bias", torch.empty(0, device=weight.device), persistent=False)
        return out

    def _init_packed(self, weight: torch.Tensor, group: int) -> None:
        n, d, k = weight.shape
        p = pack_w4(weight.reshape(n * d, k), group=group)
        self.num_predictions, self.out_size, self.in_features = n, d, k
        self.group = p["group"]
        self.register_buffer("q", p["q"].reshape(n, d, k // 2).contiguous())
        self.register_buffer("scale", p["scale"].reshape(n, d, k // self.group).contiguous())
        self.register_buffer("minimum", p["min"].reshape(n, d, k // self.group).contiguous())

    def selected(self, x: torch.Tensor, mixture_indices: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).contiguous()
        ids = mixture_indices.reshape(-1).contiguous()
        if x2.shape[0] != ids.shape[0]:
            raise ValueError("MoG INT4 input and mixture index row counts differ")
        if x2.device.type != "cuda":
            # CPU fallback is intentionally simple and is used by unit tests only.
            q = self.q[ids]
            lo, hi = q & 0xF, q >> 4
            vals = torch.stack((lo, hi), dim=-1).reshape(q.shape[0], self.out_size, self.in_features)
            sc = self.scale[ids].float().repeat_interleave(self.group, dim=-1)
            mn = self.minimum[ids].float().repeat_interleave(self.group, dim=-1)
            vals = vals * sc + mn
            y = torch.einsum("bk,bdk->bd", x2.float(), vals)
        else:
            import triton
            y = torch.empty((x2.shape[0], self.out_size), device=x2.device, dtype=x2.dtype)
            _kernel()[(x2.shape[0], triton.cdiv(self.out_size, 64))](
                x2, self.q, self.scale, self.minimum, ids, y,
                x2.shape[0], self.out_size, self.in_features,
                G=self.group, BLOCK_D=64, BLOCK_K=128,
            )
        if self.bias.numel():
            y = y + self.bias.view(self.num_predictions, self.out_size)[ids].to(y.dtype)
        return y.to(x.dtype).reshape(*shape, self.out_size)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale, self.minimum))
