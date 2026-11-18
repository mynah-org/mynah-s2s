"""Model configuration read from a HF checkpoint's config.json (text part only). Qwen3.5 dense and Qwen3.5/3.6 MoE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QwenCfg:
    hidden: int
    layers: int
    layer_types: list
    vocab: int
    eps: float
    # full attention
    heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    attn_output_gate: bool
    # gated DeltaNet
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_k: int
    # MLP / MoE
    intermediate: int = 0
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    shared_intermediate: int = 0
    tie_embeddings: bool = False
    eos_ids: list = field(default_factory=list)

    @property
    def moe(self) -> bool:
        return self.num_experts > 0


def load_cfg(path) -> QwenCfg:
    c = json.loads((Path(path) / "config.json").read_text())
    t = c.get("text_config", c)
    rp = t.get("rope_parameters") or {}
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
    eos = t.get("eos_token_id", c.get("eos_token_id"))
    gen = Path(path) / "generation_config.json"
    if gen.exists():
        g = json.loads(gen.read_text()).get("eos_token_id")
        eos = g if g is not None else eos
    eos = eos if isinstance(eos, list) else [eos]
    return QwenCfg(
        hidden=t["hidden_size"], layers=t["num_hidden_layers"], layer_types=list(t["layer_types"]),
        vocab=t["vocab_size"], eps=t["rms_norm_eps"],
        heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=head_dim,
        rope_theta=rp.get("rope_theta", t.get("rope_theta", 1e7)), rotary_dim=int(head_dim * prf),
        attn_output_gate=bool(t.get("attn_output_gate", True)),
        lin_k_heads=t["linear_num_key_heads"], lin_v_heads=t["linear_num_value_heads"],
        lin_k_dim=t["linear_key_head_dim"], lin_v_dim=t["linear_value_head_dim"], conv_k=t["linear_conv_kernel_dim"],
        intermediate=t.get("intermediate_size") or 0, num_experts=t.get("num_experts") or 0,
        top_k=t.get("num_experts_per_tok") or 0, moe_intermediate=t.get("moe_intermediate_size") or 0,
        shared_intermediate=t.get("shared_expert_intermediate_size") or 0,
        tie_embeddings=bool(c.get("tie_word_embeddings", t.get("tie_word_embeddings", False))),
        eos_ids=[e for e in eos if e is not None])
