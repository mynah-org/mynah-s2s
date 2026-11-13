"""L1: LoRA on the VoiceChat Thinker (NemotronH), attached in memory to the loaded model (no NeMo / V2 engine change).

Targets (L0 audit, .work/multilingual/l0-thinker-lora-card.md): `q_proj` and `v_proj` of the 4 attention layers
(14, 21, 30, 39) of `stt_model.llm`. Each target `nn.Linear` is replaced by `LoRALinear`:
  y = base(x) + scale * B(A(x)),  scale = alpha / r,  B = 0 at init (exact identity), A ~ Kaiming-uniform.
Precision: A and B are FP32 parameters (Adam updates are never lost to bf16 parameter rounding); the LoRA branch is
computed in FP32 from x, added to the base output in FP32 and cast once to the base output dtype (bf16 on the real model).
The cast is unavoidable (the attention runs in bf16): `branch_survival` measures which share of the output elements the
LoRA term actually changes after it. Never merged into bf16 weights (a small delta W would round away).
"""
from __future__ import annotations

import math

import torch
from torch import nn

ATTN_LAYERS = (14, 21, 30, 39)
TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, seed: int = 0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target must be nn.Linear, got {type(base).__name__}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        dev = base.weight.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        bound = 1.0 / math.sqrt(base.in_features)                 # kaiming_uniform_(a=sqrt(5)) bound, as nn.Linear/PEFT
        a.uniform_(-bound, bound, generator=g)
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        self.enabled = True

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """The LoRA term in FP32."""
        return (x.float() @ self.lora_A.t()) @ self.lora_B.t() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        return (y.float() + self.delta(x)).to(y.dtype)


def install(llm: nn.Module, layers=ATTN_LAYERS, targets=TARGETS, r: int = 8, alpha: float = 16.0, seed: int = 0) -> dict:
    """Replace llm.layers[i].mixer.<t> by LoRALinear for every (i, t). Returns {name: LoRALinear}. Every other parameter of
    `llm` is frozen."""
    for p in llm.parameters():
        p.requires_grad_(False)
    mods = {}
    for k, i in enumerate(layers):
        blk = llm.layers[i]
        if getattr(blk, "block_type", "attention") != "attention":
            raise ValueError(f"layer {i} is {blk.block_type}, not attention")
        for j, t in enumerate(targets):
            base = getattr(blk.mixer, t)
            if isinstance(base, LoRALinear):
                raise ValueError(f"layers.{i}.mixer.{t} already has LoRA")
            m = LoRALinear(base, r, alpha, seed=seed * 1000 + k * 10 + j)
            setattr(blk.mixer, t, m)
            mods[f"layers.{i}.mixer.{t}"] = m
    return mods


def lora_parameters(mods: dict) -> list[nn.Parameter]:
    return [p for m in mods.values() for p in (m.lora_A, m.lora_B)]


def n_params(mods: dict) -> int:
    return sum(p.numel() for p in lora_parameters(mods))


def state(mods: dict) -> dict:
    return {f"{n}.{w}": getattr(m, w).detach().cpu().clone() for n, m in mods.items() for w in ("lora_A", "lora_B")}


def load_state(mods: dict, st: dict, strict: bool = True):
    want = {f"{n}.{w}" for n in mods for w in ("lora_A", "lora_B")}
    if strict and set(st) != want:
        raise KeyError(f"LoRA state mismatch: missing {sorted(want - set(st))[:4]} extra {sorted(set(st) - want)[:4]}")
    with torch.no_grad():
        for n, m in mods.items():
            for w in ("lora_A", "lora_B"):
                getattr(m, w).copy_(st[f"{n}.{w}"].to(getattr(m, w).device, torch.float32))


def set_enabled(mods: dict, on: bool):
    for m in mods.values():
        m.enabled = on


def branch_survival(mod: LoRALinear, x: torch.Tensor) -> dict:
    """How much of the LoRA term survives the cast to the base output dtype: share of output elements changed, and the
    relative size of the realised change vs the FP32 delta."""
    with torch.no_grad():
        y = mod.base(x)
        d = mod.delta(x)
        out = (y.float() + d).to(y.dtype)
        real = out.float() - y.float()
        return {"changed_frac": float((out != y).float().mean()),
                "delta_abs_mean": float(d.abs().mean()),
                "realised_over_delta": float(real.abs().sum() / d.abs().sum().clamp_min(1e-30)),
                "out_abs_mean": float(y.float().abs().mean())}


def update_norms(prev: dict, mods: dict) -> dict:
    """||param_now - param_prev|| and ||param_now|| per LoRA tensor (prev = state() taken earlier)."""
    cur = state(mods)
    return {k: {"upd": float((cur[k] - prev[k]).norm()), "norm": float(cur[k].norm())} for k in cur}
