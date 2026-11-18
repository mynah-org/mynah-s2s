"""Mode (b): per-forward application of a compressed delta to the shared, read-only base weights.

Each backbone nn.Linear of a slot is replaced by a DeltaLinear that keeps a reference to the base weight
(shared by every slot, never written) and its own compressed delta (int8 / int4 codes + bf16 scales, or a bf16
delta) in preallocated buffers. Every forward computes W = W_base + dequant(delta) into a temporary and runs the
matmul, so no per-language full copy of the backbone exists. Loading a language only copies the compressed codes
into the slot's buffers (fixed addresses: captured CUDA graphs stay valid). Plain torch, any architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DeltaLinear(nn.Module):
    def __init__(self, base_weight: torch.Tensor, precision: str, group: int = 128, fused: bool | None = None):
        super().__init__()
        self.fused = fused  # None = use the Triton kernel when available, False = always plain torch
        if precision not in ("bf16", "int8", "int4"):
            raise ValueError(f"on-the-fly mode needs a delta precision, got {precision!r}")
        self.out_features, self.in_features = base_weight.shape
        self.precision, self.group = precision, group
        self.base = base_weight  # shared; plain attribute (not a parameter/buffer): never in state_dict, never written
        dev = base_weight.device
        r, c = base_weight.shape
        g = c if group <= 0 else group
        if precision == "bf16":
            self.register_buffer("d", torch.zeros(r, c, dtype=base_weight.dtype, device=dev), persistent=False)
        else:
            cols = c if precision == "int8" else c // 2
            dt = torch.int8 if precision == "int8" else torch.uint8
            fill = 0 if precision == "int8" else 0x88  # int4: nibble 8 = zero
            self.register_buffer("q", torch.full((r, cols), fill, dtype=dt, device=dev), persistent=False)
            self.register_buffer("s", torch.zeros(r, c // g, dtype=torch.bfloat16, device=dev), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        """The shared base weight, exposed for code that reads device/dtype (e.g. KV-cache init). Do not write."""
        return self.base

    def delta_nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    @torch.no_grad()
    def load(self, codes: torch.Tensor | None, scales: torch.Tensor | None = None) -> None:
        """Copy a compressed delta in place (None = zero delta, i.e. the base language)."""
        if self.precision == "bf16":
            self.d.zero_() if codes is None else self.d.copy_(codes)
        elif codes is None:
            self.q.fill_(0 if self.precision == "int8" else 0x88)
            self.s.zero_()
        else:
            self.q.copy_(codes)
            self.s.copy_(scales)

    def weight_now(self) -> torch.Tensor:
        b = self.base
        if self.precision == "bf16":
            return b + self.d
        r, c = b.shape
        ng = self.s.shape[1]
        if self.precision == "int8":
            q = self.q.to(b.dtype)
        else:
            lo = (self.q & 0x0F).to(b.dtype) - 8
            hi = (self.q >> 4).to(b.dtype) - 8
            q = torch.stack((lo, hi), dim=-1).reshape(r, c)
        # one fused multiply-add per element, computed in fp32 opmath and rounded once (as in slot materialisation)
        return torch.addcmul(b.view(r, ng, c // ng), q.view(r, ng, c // ng), self.s.view(r, ng, 1).to(b.dtype)).view(r, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not False and x.numel() // x.shape[-1] <= 16:
            from . import fused

            if fused.available(x):
                if torch.is_autocast_enabled(x.device.type):
                    x = x.to(torch.bfloat16)
                codes = self.d if self.precision == "bf16" else self.q
                scales = None if self.precision == "bf16" else self.s
                return fused.fused_linear(x, self.base, self.precision, codes, scales, self.group)
        return F.linear(x, self.weight_now())


def convert_backbone(flow_lm: nn.Module, base: dict[str, torch.Tensor], precision: str, group: int = 128,
                     prefix: str = "flow_lm.", fused: bool | None = None) -> dict[str, DeltaLinear]:
    """Replace every nn.Linear of flow_lm.transformer by a DeltaLinear on the shared base weight."""
    out: dict[str, DeltaLinear] = {}
    for name, mod in list(flow_lm.transformer.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{prefix}transformer.{name + '.' if name else ''}{child_name}.weight"
                if child.bias is not None:
                    raise ValueError(f"{full}: biased linears are not supported")
                dl = DeltaLinear(base[full], precision, group, fused)
                setattr(mod, child_name, dl)
                out[full] = dl
    return out
