"""EXPERIMENTAL weight-only quantization of the Thinker (NemotronH backbone linears). Never used by REFERENCE.

Why int4 tinygemm: the Thinker step is bandwidth-bound (profile C2/C6: 68-74 ms/step, gemv kernels; 14.1 GB of BF16
linear weights read per step). On GB10/sm_121 with torch 2.10 native kernels (microbench D1, M=1):
bf16 64.4 ms/step, int8 weight-only ``_weight_int8pack_mm`` 86.5 ms (slower), fp8 row-wise ``_scaled_mm`` not supported
by cuBLASLt, int4 ``_weight_int4pack_mm`` 11.9 ms (5.4×). Only int4 actually pays off, hence this module.

Scope: Mamba ``in_proj``/``out_proj``, MLP ``up_proj``/``down_proj``, attention ``q/k/v/o_proj`` of the backbone layers.
Not quantized: embeddings, text/function/ASR heads, RNN-T, perception, EarTTS, codec. Asymmetric min-max per group
(torchao/tinygemm convention: w ≈ (q - 8) · scale + zero). Group 128, or 64/32 when K is not a multiple.
The replaced module keeps an empty ``weight`` buffer so the remote code's device/dtype checks still work
(``in_proj.weight.device`` for the fast-path dispatch, ``q_proj.weight.dtype``). Only the cached (incremental) path is
supported: the training-only fused kernel reads ``out_proj.weight`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TARGETS = {"in_proj", "out_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}


def _pick(k: int, max_group: int = 128) -> tuple[int, int]:
    """(group size, inner-k tiles) satisfying tinygemm: K % group == 0 and K % (16 × inner tiles) == 0."""
    group = next(g for g in (128, 64, 32) if g <= max_group and k % g == 0)
    tiles = next(t for t in (8, 4, 2) if k % (16 * t) == 0)
    return group, tiles


def _clip_range(wf: torch.Tensor, steps=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    """Per-group [min, max] shrink that minimises the int4 reconstruction MSE (HQQ/AWQ-style range search)."""
    mn0, mx0 = wf.amin(-1), wf.amax(-1)
    best_mn, best_mx = mn0.clone(), mx0.clone()
    best = torch.full_like(mn0, float("inf"))
    for f in steps:
        mid = (mx0 + mn0) / 2
        half = (mx0 - mn0) / 2 * f
        mn, mx = mid - half, mid + half
        scale = (mx - mn).clamp_min(1e-8) / 15
        q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15)
        err = ((q * scale[..., None] + mn[..., None]) - wf).pow(2).sum(-1)
        upd = err < best
        best = torch.where(upd, err, best)
        best_mn, best_mx = torch.where(upd, mn, best_mn), torch.where(upd, mx, best_mx)
    return best_mn, best_mx


def quantize_int4(w: torch.Tensor, group: int, tiles: int, clip: bool = False):
    """bf16 (N, K) -> (packed int4 weight, scales_and_zeros (K/group, N, 2) bf16)."""
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = _clip_range(wf) if clip else (wf.amin(-1), wf.amax(-1))
    scale = (mx - mn).clamp_min(1e-8) / 15
    zero = mn + scale * 8
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32).reshape(n, k)
    packed = torch._convert_weight_to_int4pack(((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8), tiles)
    sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    return packed, sz


def dequantize_int4_reference(w: torch.Tensor, group: int) -> torch.Tensor:
    """Same quantizer, dequantized in torch (for error checks without the kernel)."""
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = wf.amin(-1), wf.amax(-1)
    scale = (mx - mn).clamp_min(1e-8) / 15
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15)
    return (q * scale[..., None] + mn[..., None]).reshape(n, k)


class Int4Linear(nn.Module):
    def __init__(self, lin: nn.Linear, max_group: int = 128, clip: bool = False):
        super().__init__()
        w = lin.weight.data
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.group, self.tiles = _pick(self.in_features, max_group)
        packed, sz = quantize_int4(w.to(torch.bfloat16), self.group, self.tiles, clip=clip)
        self.register_buffer("packed", packed)
        self.register_buffer("scales_zeros", sz)
        self.register_buffer("weight", torch.empty(0, dtype=w.dtype, device=w.device), persistent=False)
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        dt = x2.dtype
        y = torch._weight_int4pack_mm(x2.to(torch.bfloat16).contiguous(), self.packed, self.group, self.scales_zeros)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(dt).reshape(*shp[:-1], self.out_features)


def quantize_thinker_int4(engine, max_group: int = 128, clip: bool = False, skip_layers: set[int] | None = None,
                          keep_original: bool = False):
    """Replace the backbone linears of ``engine`` (ReferenceEngine) in place. Returns a summary for traces, and with
    ``keep_original`` also the replaced modules so ``restore_thinker`` can undo it (A/B in one process)."""
    llm = engine.wrapper.model.stt_model.llm
    before = after = 0
    n = 0
    originals = []
    skip_layers = skip_layers or set()
    for li, layer in enumerate(llm.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None or li in skip_layers:
            continue
        for name, mod in list(mixer.named_children()):
            if name in TARGETS and isinstance(mod, nn.Linear):
                before += mod.weight.numel() * mod.weight.element_size()
                q = Int4Linear(mod, max_group=max_group, clip=clip)
                after += q.packed.numel() * q.packed.element_size() + q.scales_zeros.numel() * 2
                setattr(mixer, name, q)
                if keep_original:
                    originals.append((mixer, name, mod))
                n += 1
    torch.cuda.empty_cache()
    info = {"thinker_quant": "int4_tinygemm", "linears": n, "max_group": max_group, "clip_search": clip,
            "skip_layers": sorted(skip_layers), "bf16_GB": round(before / 2**30, 2), "int4_GB": round(after / 2**30, 2)}
    return (info, originals) if keep_original else info


def restore_thinker(originals) -> None:
    for mixer, name, mod in originals:
        setattr(mixer, name, mod)
    torch.cuda.empty_cache()
