"""Interface adapters between a source speech encoder (ML) and VoiceChat's frozen `perception.proj`.

ResidualAdapter: y = x @ M + m  (affine, initialised from a ridge fit)  +  MLP(LN(x)) (zero-initialised last layer)
                 optionally + a causal depthwise temporal conv on the MLP branch (k frames of left context only, so
                 streaming stays frame-synchronous; no lookahead).
At init the adapter is exactly the ridge map, so training can only move away from a known-good English point.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, d: int = 1024, hidden: int = 2048, temporal_k: int = 0):
        super().__init__()
        self.cfg = {"d": d, "hidden": hidden, "temporal_k": temporal_k}
        self.affine = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, hidden)
        self.down = nn.Linear(hidden, d)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.temporal_k = temporal_k
        if temporal_k > 1:
            self.tconv = nn.Conv1d(hidden, hidden, temporal_k, groups=hidden)
            nn.init.zeros_(self.tconv.weight)
            nn.init.zeros_(self.tconv.bias)

    def init_affine(self, M: torch.Tensor, m: torch.Tensor):
        with torch.no_grad():
            self.affine.weight.copy_(M.T)
            self.affine.bias.copy_(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., T, d] (or [..., d] per frame when temporal_k <= 1)."""
        h = torch.nn.functional.gelu(self.up(self.ln(x)))
        if self.temporal_k > 1:
            if h.dim() == 2:
                h = h[None]
            z = torch.nn.functional.pad(h.transpose(-1, -2), (self.temporal_k - 1, 0))
            h = h + self.tconv(z).transpose(-1, -2)
            h = h.reshape(*x.shape[:-1], -1)
        return self.affine(x) + self.down(h)


def save_adapter(module: ResidualAdapter, path, extra: dict | None = None):
    torch.save({"cfg": module.cfg, "state": module.state_dict(), "extra": extra or {}}, path)


def load_adapter(path) -> ResidualAdapter:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ResidualAdapter(**ck["cfg"])
    m.load_state_dict(ck["state"])
    if m.temporal_k > 1:
        raise NotImplementedError("temporal adapters need a frame-history buffer in the streaming path")
    return m.eval()
