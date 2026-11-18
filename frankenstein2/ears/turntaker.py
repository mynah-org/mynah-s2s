# SPDX-License-Identifier: Apache-2.0
# Derived from tyger-s2s `app/plugins/ears/tyger/controller.py` and `bundle.py` @97d011d (Copyright the tyger-s2s
# authors, Apache License 2.0, see LICENSES/Apache-2.0.txt). Modified for Frankenstein v2: module definition kept
# (tensor names must match the bundle), loader reduced to the parts we use (checksums, strict load), a one-frame step
# for a single session added.
"""The turn-taking GRU (~14M parameters, fp32) that reads layer 15 of the Nemotron 3.5 ASR encoder.

Per 80 ms frame: user L15 features, agent ("2nd ear") L15 features, the user's frame energy and the previous state
(forced by the controller's mode) are projected, summed and normalised; a 3-layer GRU carries the context; five heads
give [state logits x4, hazard, yield, backchannel, endpoint] (sigmoids for the last four).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

ENC_HIDDEN = 1024
N_STATE = 4
L15 = 15


def _mlp_head(d):
    return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))


class TurnTaker(nn.Module):
    def __init__(self, width=768, layers=3):
        super().__init__()
        d = width
        self.core = nn.GRU(d, d, layers, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        self.proj_user = nn.Linear(ENC_HIDDEN, d)
        self.proj_agent = nn.Linear(ENC_HIDDEN, d)
        self.proj_energy = nn.Linear(1, d)
        self.state_emb = nn.Embedding(N_STATE + 1, d)
        self.in_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, N_STATE)
        self.hazard_head = nn.Linear(d, 1)
        self.yield_head = _mlp_head(d)
        self.bc_head = _mlp_head(d)
        self.ep_head = _mlp_head(d)

    def fuse(self, h_user, h_agent, act, prev_state):
        return self.in_norm(self.proj_user(h_user) + self.proj_agent(h_agent) + self.proj_energy(act.unsqueeze(-1))
                            + self.state_emb(prev_state))

    def outputs(self, z):
        z = self.out_norm(z)
        return torch.cat([self.state_head(z), torch.sigmoid(self.hazard_head(z)), torch.sigmoid(self.yield_head(z)),
                          torch.sigmoid(self.bc_head(z)), torch.sigmoid(self.ep_head(z))], -1)

    def forward(self, h_user, h_agent, act, prev_state, state=None):
        z, state = self.core(self.fuse(h_user, h_agent, act, prev_state), state)
        return self.outputs(z), state


class GruSession:
    """One conversation's GRU state; ``step`` runs one frame."""

    def __init__(self, model: TurnTaker, device):
        self.m, self.dev = model, device
        self.h = None

    def reset(self):
        self.h = None

    @torch.no_grad()
    def step(self, hu, ha, act_db: float, prev_state: int):
        act = torch.tensor([[(max(-3.0, min(3.0, (act_db + 50.0) / 25.0)))]], dtype=torch.float32, device=self.dev)
        prev = torch.tensor([[prev_state]], dtype=torch.long, device=self.dev)
        out, self.h = self.m(hu.view(1, 1, -1).float(), ha.view(1, 1, -1).float(), act, prev, self.h)
        return [float(x) for x in out[0, 0].tolist()]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(folder, check_sha: bool = True):
    """-> (Nemotron 3.5 ASR model fp32, its processor, TurnTaker fp32, config) on the CPU. TF32 off process-wide (the
    bundle was validated with it off)."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoProcessor
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if config.get("model_type") != "tyger_nemotron_gru":
        raise ValueError(f"not a tyger bundle: {config.get('model_type')!r}")
    if check_sha and _sha256(folder / "model.safetensors") != config["weights_sha256"]:
        raise ValueError("model.safetensors does not match its checksum")
    weights = load_file(str(folder / "model.safetensors"))
    parts = {"nemotron": {}, "turn_taker": {}}
    for k, v in weights.items():
        prefix, _, name = k.partition(".")
        parts[prefix][name] = v
    del weights
    nemotron = AutoModel.from_config(AutoConfig.from_pretrained(folder / "nemotron"), dtype=torch.float32)
    nemotron.load_state_dict(parts["nemotron"], strict=True)
    proc = AutoProcessor.from_pretrained(folder / "nemotron")
    arch = config["controller"]["architecture"]
    tt = TurnTaker(arch["width"], arch["layers"])
    tt.load_state_dict(parts["turn_taker"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return nemotron.eval(), proc, tt.eval(), config
