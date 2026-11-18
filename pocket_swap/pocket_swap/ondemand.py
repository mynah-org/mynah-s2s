"""On-the-fly modes: every forward computes W = T_base + T_lang and runs the matmul; no bf16 copy per language.

Each backbone nn.Linear of a slot is replaced by a DeltaLinear holding
  base   a shared, read-only term (dense bf16 base weight, or the base quantised once to int8/int4), or None
  lang   the slot's own term in preallocated buffers: a delta (bf16/int8/int4) on top of the base, or - with no
         base - the language's full weights quantised (int8/int4/fp8: weight-only, "quantised end to end")
Loading a language only copies codes into the slot's buffers (fixed addresses: captured CUDA graphs stay valid).
With Triton (sm_80+, fp8 sm_89+) the dequant happens inside the GEMV tile (fused.py); otherwise plain torch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .quant import dequant_any, quantize_any


class DeltaLinear(nn.Module):
    def __init__(self, base_weight: torch.Tensor | None, precision: str, group: int = 128, fused: bool | None = None,
                 base_term: tuple | None = None, shape: tuple | None = None, device=None):
        """base_weight: dense bf16 base (shared) or None; base_term: (precision, codes, scales) of a quantised
        shared base (takes precedence); with neither, `precision` describes the full weights (weight-only)."""
        super().__init__()
        if precision not in ("bf16", "int8", "int4", "fp8"):
            raise ValueError(f"unknown precision {precision!r}")
        self.fused = fused  # None = Triton kernel when available, False = always plain torch
        if base_term is not None:
            self.base_term = base_term
        elif base_weight is not None:
            self.base_term = ("bf16", base_weight, None)
        else:
            self.base_term = None
        ref = base_weight if base_weight is not None else (base_term[1] if base_term is not None else None)
        r, c = shape if shape is not None else tuple(base_weight.shape)
        dev = device if device is not None else ref.device
        self.out_features, self.in_features = r, c
        self.precision, self.group = precision, group
        g = c if group <= 0 else group
        if precision == "bf16":
            self.register_buffer("d", torch.zeros(r, c, dtype=torch.bfloat16, device=dev), persistent=False)
        else:
            cols = c // 2 if precision == "int4" else c
            dt = {"int8": torch.int8, "int4": torch.uint8, "fp8": torch.float8_e4m3fn}[precision]
            q = torch.zeros((r, cols), dtype=torch.uint8 if precision == "int4" else torch.int8, device=dev)
            if precision == "int4":
                q.fill_(0x88)  # nibble 8 = zero
            self.register_buffer("q", q.view(dt) if precision == "fp8" else q.to(dt), persistent=False)
            self.register_buffer("s", torch.zeros(r, c // g, dtype=torch.bfloat16, device=dev), persistent=False)
        self._ref = torch.empty(0, dtype=torch.bfloat16, device=dev)

    @property
    def base(self) -> torch.Tensor | None:
        return self.base_term[1] if self.base_term is not None and self.base_term[0] == "bf16" else None

    @property
    def weight(self) -> torch.Tensor:
        """A bf16 tensor on the right device, for code that only reads device/dtype (KV-cache init). Not the weight."""
        return self._ref

    def lang_term(self) -> tuple:
        return (self.precision, self.d, None) if self.precision == "bf16" else (self.precision, self.q, self.s)

    def delta_nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for n, b in self.named_buffers() if n != "_ref")

    @torch.no_grad()
    def load(self, codes: torch.Tensor | None, scales: torch.Tensor | None = None) -> None:
        """Copy the slot's term in place (None = zero delta, i.e. the base language itself)."""
        if codes is None:
            if self.base_term is None:
                raise ValueError("weight-only slots need the language's weights")
            if self.precision == "bf16":
                self.d.zero_()
            else:
                self.q.copy_(torch.full_like(self.q.view(torch.uint8), 0x88 if self.precision == "int4" else 0)
                             .view(self.q.dtype))
                self.s.zero_()
        elif self.precision == "bf16":
            self.d.copy_(codes)
        else:
            self.q.copy_(codes)
            self.s.copy_(scales)

    def weight_now(self) -> torch.Tensor:
        """Dense bf16 weight (torch path): fp32 sum of the dequantised terms, rounded once."""
        c = self.in_features
        lp, lq, ls = self.lang_term()
        w = dequant_any(lq, ls, lp, c)
        if self.base_term is not None:
            bp, bq, bs = self.base_term
            w = w + dequant_any(bq, bs, bp, c)
        return w.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not False and x.numel() // x.shape[-1] <= 16:
            from . import fused

            precs = (self.precision,) + ((self.base_term[0],) if self.base_term is not None else ())
            if fused.available(x, precs):
                if torch.is_autocast_enabled(x.device.type):
                    x = x.to(torch.bfloat16)
                return fused.fused_linear2(x, self.out_features, self.lang_term(), self.base_term, self.group)
        w = self.weight_now()
        if torch.is_autocast_enabled(x.device.type) or x.dtype == torch.bfloat16:
            return F.linear(x.to(torch.bfloat16), w)
        return F.linear(x, w.to(x.dtype))


def quantise_base(base: dict[str, torch.Tensor], precision: str, group: int = 128) -> dict[str, tuple]:
    """Quantise the shared base backbone once (on its device): {key: (precision, codes, scales)}."""
    return {k: (precision,) + quantize_any(v, precision, group) for k, v in base.items()}


def convert_backbone(flow_lm: nn.Module, base: dict[str, torch.Tensor] | None, precision: str, group: int = 128,
                     prefix: str = "flow_lm.", fused: bool | None = None,
                     base_terms: dict[str, tuple] | None = None, device=None) -> dict[str, DeltaLinear]:
    """Replace every nn.Linear of flow_lm.transformer by a DeltaLinear.
    base: dense bf16 base weights (ondemand); base_terms: a quantised base; neither: weight-only slots."""
    out: dict[str, DeltaLinear] = {}
    for name, mod in list(flow_lm.transformer.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{prefix}transformer.{name + '.' if name else ''}{child_name}.weight"
                if child.bias is not None:
                    raise ValueError(f"{full}: biased linears are not supported")
                bt = base_terms.get(full) if base_terms else None
                bw = base.get(full) if (base and bt is None) else None
                dl = DeltaLinear(bw, precision, group, fused, base_term=bt, shape=tuple(child.weight.shape),
                                 device=device)
                setattr(mod, child_name, dl)
                out[full] = dl
    return out
