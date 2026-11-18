"""Weight-only linears: bf16, int8 (per output channel, symmetric), int4 (per group, asymmetric, two per byte) and
AWQ int4 checkpoints (AutoAWQ GEMM layout). Plain torch: the weight is dequantised to bf16 right before the matmul
(portable to any CUDA arch and to CPU). A fused Triton path can replace `dequant` later without changing callers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)        # AutoAWQ packs 8 nibbles per int32 in this interleaved order
def awq_unpack(q: torch.Tensor) -> torch.Tensor:
    """int32 [..., rows, cols/8] (AWQ order) -> int32 [..., rows, cols] values 0..15 in natural column order (one
    shift per natural column: column 8p+j sits in nibble AWQ_REVERSE_ORDER[j] of packed word p)."""
    sh = torch.tensor([4 * r for r in AWQ_REVERSE_ORDER], device=q.device, dtype=torch.int32)
    return ((q.unsqueeze(-1) >> sh) & 0xF).reshape(*q.shape[:-1], q.shape[-1] * 8)


def awq_dequant(qweight, qzeros, scales, group: int) -> torch.Tensor:
    """AutoAWQ GEMM tensors -> bf16 weight [..., out, in] for F.linear. qweight [..., in, out/8], qzeros
    [..., in/group, out/8], scales [..., in/group, out]. Zeros and scales broadcast per group (no repeat copies)."""
    w = awq_unpack(qweight)                                                # [..., in, out]
    z = awq_unpack(qzeros)                                                 # [..., in/g, out]
    *lead, i_, o = w.shape
    w = w.reshape(*lead, i_ // group, group, o)
    w = (w - z.unsqueeze(-2)).to(scales.dtype) * scales.unsqueeze(-2)
    return w.reshape(*lead, i_, o).transpose(-1, -2).to(torch.bfloat16)


class QLinear:
    """y = x @ W^T with W stored as bf16 / int8 / int4 / awq. No bias (Qwen3.5 linears have none)."""

    def __init__(self, w: torch.Tensor | None = None, mode: str = "bf16", group: int = 128, **awq):
        self.mode, self.group = mode, group
        if mode == "awq":
            self.qweight, self.qzeros, self.scales = awq["qweight"], awq["qzeros"], awq["scales"]
            self.out_features = self.scales.shape[-1]
            return
        if mode == "nvfp4":                       # ModelOpt W4A16 NVFP4: uint8 [out, in/2], e4m3 [out, in/16], fp32
            self.q, self.s, self.s2 = awq["q"], awq["s"], awq["s2"].float().reshape(1)
            self.out_features = self.q.shape[0]
            return
        if mode == "fp8":                         # e4m3 [out, in] with a per-tensor scale (weight-only use here)
            self.q = awq["q"]
            self.s = awq["s"].float().reshape(1, 1).expand(self.q.shape[0], 1).contiguous()
            self.out_features = self.q.shape[0]
            return
        w = w.to(torch.bfloat16)
        self.out_features = w.shape[0]
        if mode == "bf16":
            self.w = w
        elif mode == "int8":
            s = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            self.q = torch.round(w.float() / s).clamp(-127, 127).to(torch.int8)
            self.s = s.to(torch.bfloat16)
        elif mode == "int4":
            o, i = w.shape
            g = group if i % group == 0 else i
            wg = w.float().reshape(o, i // g, g)
            lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
            s = ((hi - lo) / 15.0).clamp_min(1e-8)
            z = torch.round(-lo / s).clamp(0, 15)
            q = torch.round(wg / s + z).clamp(0, 15).to(torch.uint8).reshape(o, i)
            self.q = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()            # two nibbles per byte
            self.s, self.z, self.group = s.to(torch.bfloat16), z.to(torch.bfloat16), g
        else:
            raise ValueError(mode)

    def dequant(self) -> torch.Tensor:
        if self.mode == "bf16":
            return self.w
        if self.mode == "int8":
            return self.q.to(torch.bfloat16) * self.s
        if self.mode == "int4":
            lo, hi = self.q & 0xF, self.q >> 4
            q = torch.stack((lo, hi), -1).reshape(self.q.shape[0], -1)
            o, i = q.shape
            w = (q.reshape(o, i // self.group, self.group).to(torch.bfloat16) - self.z) * self.s
            return w.reshape(o, i)
        if self.mode == "nvfp4":
            from .kernels import nvfp4_dequant
            return nvfp4_dequant(self.q, self.s, self.s2[0])
        if self.mode == "fp8":
            return (self.q.float() * self.s).to(torch.bfloat16)
        return awq_dequant(self.qweight, self.qzeros, self.scales, self.group)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if self.mode in ("int8", "fp8", "nvfp4") and x.is_cuda and rows <= 16:   # decode: fused GEMV (Triton)
            from .kernels import int8_gemv, nvfp4_expert_gemv
            shp = x.shape
            x2 = x.reshape(-1, shp[-1])
            if self.mode == "nvfp4":
                z = torch.zeros(x2.shape[0], dtype=torch.long, device=x.device)
                y = nvfp4_expert_gemv(x2, self.q[None], self.s[None], self.s2, z)
            else:
                y = int8_gemv(x2, self.q, self.s)
            return y.reshape(*shp[:-1], self.out_features)
        if self.mode in ("int8", "fp8") and x.is_cuda:                     # prefill: in-tile dequant GEMM (Triton)
            from .kernels import HAVE_TRITON, wq_gemm
            if HAVE_TRITON:
                shp = x.shape
                return wq_gemm(x.reshape(-1, shp[-1]), self.q, self.s).reshape(*shp[:-1], self.out_features)
        return F.linear(x, self.dequant())

    def nbytes(self) -> int:
        ts = {"bf16": [getattr(self, "w", None)], "int8": [getattr(self, "q", None), getattr(self, "s", None)],
              "int4": [getattr(self, "q", None), getattr(self, "s", None), getattr(self, "z", None)],
              "awq": [getattr(self, "qweight", None), getattr(self, "qzeros", None), getattr(self, "scales", None)],
              "nvfp4": [getattr(self, "q", None), getattr(self, "s", None)], "fp8": [getattr(self, "q", None)]}
        return sum(t.numel() * t.element_size() for t in ts[self.mode] if t is not None)


def nvfp4_quantize(w: torch.Tensor):
    """Round-to-nearest NVFP4 (as ModelOpt's layout): w [..., N, K] float -> (uint8 [..., N, K/2], e4m3 [..., N, K/16],
    fp32 global scale [...] ). Used for checkpoints that ship bf16 experts (e.g. an MTP head) under the
    compressed-only rule. Works chunk by chunk over the leading axis to bound memory."""
    from .kernels import E2M1
    if w.dim() == 3:
        parts = [nvfp4_quantize(w[i]) for i in range(w.shape[0])]
        return tuple(torch.stack([p[j] for p in parts]) for j in range(3))
    o, i = w.shape
    g = w.float().reshape(o, i // 16, 16)
    s2 = (g.abs().amax() / (6 * 448)).clamp(min=1e-12)
    sc = (g.abs().amax(-1) / 6 / s2).clamp(min=2 ** -9).to(torch.float8_e4m3fn)
    x = g / (sc.float()[..., None] * s2)
    grid = torch.tensor(E2M1, device=w.device)
    mag = (x.abs()[..., None] - grid).abs().argmin(-1)
    nib = (mag + (x < 0).long() * 8).reshape(o, i).to(torch.uint8)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous(), sc, s2.reshape(())
