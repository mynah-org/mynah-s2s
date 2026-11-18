"""Load the text-model tensors of a Qwen3.5 / 3.6 checkpoint (safetensors) straight to the device, one file at a time
(no full CPU copy: each tensor goes from the memory-mapped file to the GPU, and a layer's experts are stacked as soon as
that layer is complete, so neither host RAM nor GPU memory ever holds two copies of the experts). Vision tower and the MTP head are skipped. MoE experts are stacked over the expert axis:
bf16 -> "experts.gate_proj" [E, out, in]; AWQ -> "experts.gate_proj.qweight" [E, in, out/8] etc. bf16 MoE experts are
refused (owner rule: the MoE runs compressed only)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

PREFIXES = ("model.language_model.", "model.")
EXPERT = re.compile(r"^(layers\.\d+\.mlp\.experts)\.(\d+)\.(\w+)\.(weight|qweight|qzeros|scales)$")


def _short(name: str) -> str | None:
    if name.startswith(("model.visual.", "visual.", "mtp.")):
        return None
    if name == "lm_head.weight":
        return name
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return None


def load_weights(path, device, allow_bf16_experts=False) -> tuple[dict, dict]:
    """-> (W, info). W: short name -> tensor on `device`; info: quantisation facts from config.json."""
    from safetensors import safe_open
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    files = sorted({v for v in json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values()}) \
        if (path / "model.safetensors.index.json").exists() else ["model.safetensors"]
    tc = cfg.get("text_config", cfg)
    n_exp = tc.get("num_experts") or 0
    W, experts = {}, defaultdict(dict)
    for f in files:
        with safe_open(str(path / f), framework="pt", device=str(device)) as fh:
            for name in fh.keys():
                s = _short(name)
                if s is None:
                    continue
                m = EXPERT.match(s)
                if m:
                    base, e, proj, kind = m.groups()
                    if kind == "weight" and not allow_bf16_experts:
                        raise ValueError("bf16 MoE experts are not allowed (compressed only): use an AWQ checkpoint")
                    key = f"{base}.{proj}" + ("" if kind == "weight" else f".{kind}")
                    experts[key][int(e)] = fh.get_tensor(name)
                    if len(experts[key]) == n_exp:                       # this layer's projection is complete
                        d = experts.pop(key)
                        W[key] = torch.stack([d.pop(i) for i in range(n_exp)])
                else:
                    W[s] = fh.get_tensor(name)
    if experts:
        raise ValueError(f"incomplete expert sets: {sorted(experts)[:4]}")
    info = {"quant_method": q.get("quant_method"), "bits": q.get("bits"), "group_size": q.get("group_size"),
            "zero_point": q.get("zero_point"), "version": q.get("version")}
    return W, info
