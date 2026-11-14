"""Optional Q8 weight-only replacement for the EarTTS backbone linears."""

from __future__ import annotations

import torch
import torch.nn as nn

from mynah_v2.model.wo_gemv import WOLinear


def quantize_eartts_w8(engine) -> dict:
    """Replace every direct EarTTS backbone Linear with the batched Q8 GEMV."""
    duplex = engine.wrapper.model.tts_model
    backbone = duplex.tts_model.backbone
    before = after = n = 0
    for name, mod in list(backbone.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        before += mod.weight.numel() * mod.weight.element_size()
        q = WOLinear(mod, "w8")
        after += q.nbytes() + (mod.bias.numel() * mod.bias.element_size() if mod.bias is not None else 0)
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        setattr(parent, child_name, q)
        n += 1
    if not n:
        raise RuntimeError("no direct EarTTS backbone linears found")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"eartts_quant": "w8_batched_gemv", "scope": "tts_model.backbone", "linears": n,
            "bf16_bytes": before, "w8_bytes": after}
