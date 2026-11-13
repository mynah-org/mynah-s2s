"""L1 preparation, CPU only (no GPU, no lock, no weights): LoRA + hybrid cache on a TINY NemotronH built from the real
Nemotron-Nano-9B-v2 remote code (snapshot 6533e8de) and config, with the V2 cache patches used by greedy_fc
(mynah_v2.model.llm_cache: _hybrid_cache_cls, position_injecting_forward, cache_passing_attention).

Checks, per scenario (no LoRA / LoRA B=0 / LoRA B random):
  full     one forward over T frames (no cache)            = the training / teacher-forced path
  cached   prefill P frames, then one frame per call        = the greedy_fc / live path
  - LoRA B=0 is bit-identical to no LoRA in both paths;
  - cached vs full differs no more with LoRA than without (the cache noise is the base model's own);
  - a non-zero LoRA moves the output well above that noise.
The CPU runs the Mamba torch path, the GPU the CUDA kernels: the GPU smoke repeats these checks on the real model.
Usage (lab box, CPU): CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python thinker_lora_cpu_check.py --snapshot <dir> --out f.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinker_lora as TL  # noqa: E402


def tiny(snapshot: str, dtype, pattern: str = "M-M*M-*-"):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    cfg.update({"hidden_size": 64, "num_hidden_layers": len(pattern), "hybrid_override_pattern": pattern,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
                "mamba_num_heads": 8, "mamba_head_dim": 16, "n_groups": 2, "mamba_num_groups": 2, "ssm_state_size": 16,
                "mamba_state_dim": 16, "chunk_size": 8, "vocab_size": 128, "use_mamba_kernels": False})
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype).eval()
    return cfg, m.backbone


def run(llm, cfg, E, P, dtype):
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    with torch.no_grad():
        full = llm(inputs_embeds=E, return_dict=True)["last_hidden_state"].float()
        cache = _hybrid_cache_cls(llm)(cfg, E.shape[0], dtype=dtype, device=E.device, ssm_dtype=torch.float32)
        outs = [llm(inputs_embeds=E[:, :P], cache_params=cache, use_cache=True, return_dict=True)["last_hidden_state"]]
        cache._v2_pos = P
        for t in range(P, E.shape[1]):
            outs.append(llm(inputs_embeds=E[:, t:t + 1], cache_params=cache, use_cache=True,
                            return_dict=True)["last_hidden_state"])
            cache._v2_pos = t + 1
        cached = torch.cat(outs, 1).float()
    return full, cached


def prepare(snapshot, dtype, pattern):
    from mynah_v2.model.llm_cache import cache_passing_attention, position_injecting_forward
    cfg, llm = tiny(snapshot, dtype, pattern)
    if not torch.cuda.is_available():               # gated RMSNorm: mamba_ssm's own torch reference instead of the triton
        from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref   # kernel (CPU; same for every scenario)
        for b in llm.layers:
            if b.block_type == "mamba":
                n = b.mixer.norm
                n.forward = (lambda n: lambda hidden_states, gate=None: rms_norm_ref(
                    hidden_states, n.weight, None, z=gate, eps=n.variance_epsilon, group_size=n.group_size,
                    norm_before_gate=False))(n)
    llm.forward = position_injecting_forward(llm.forward, holder=llm)
    attn = [i for i, b in enumerate(llm.layers) if b.block_type == "attention"]
    for i in attn:
        llm.layers[i].mixer.forward = cache_passing_attention(llm.layers[i].mixer.forward, llm)
    return cfg, llm, attn


def full(llm, E):
    with torch.no_grad():
        return llm(inputs_embeds=E, return_dict=True)["last_hidden_state"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    if not torch.cuda.is_available():               # NemotronHBlock wraps every mixer in torch.cuda.stream(...): a
        import contextlib                           # no-op on CPU (this process only; remote code untouched)
        torch.cuda.default_stream = lambda *a, **k: None
        torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
    dtype = getattr(torch, a.dtype)
    mx = lambda x, y: float((x - y).abs().max())           # noqa: E731
    torch.manual_seed(1)
    E = (torch.randn(2, 24, 64) * 0.5).to(dtype)
    P = 9
    rep = {"torch": torch.__version__, "dtype": a.dtype, "T": E.shape[1], "prefill": P,
           "note": "A: hybrid (Mamba torch path + attention), full forward only - the V2 hybrid cache supports the CUDA "
                   "Mamba path only, so the cached Mamba decode is checked in the GPU smoke. B: attention + MLP only, "
                   "full vs cached (V2 cache patches): the only cached component LoRA touches."}
    # A: hybrid model, full path: identity, effect, gradients through a frozen function-head stand-in
    cfg, llm, attn = prepare(a.snapshot, dtype, "M-M*M-*-")
    f0 = full(llm, E)
    mods = TL.install(llm, layers=tuple(attn))
    rep["A_attn_layers"], rep["A_lora_params"] = attn, TL.n_params(mods)
    rep["A_trainable_non_lora"] = [n for n, p in llm.named_parameters() if p.requires_grad and "lora_" not in n]
    f1 = full(llm, E)
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.normal_(0, 0.05)
    f2 = full(llm, E)
    rep["A_B0_equal_bitwise"] = bool(torch.equal(f0, f1))
    rep["A_lora_effect_max"] = mx(f2, f0)
    rep["A_survival_q_first_layer"] = TL.branch_survival(mods[f"layers.{attn[0]}.mixer.q_proj"], E)
    head = torch.nn.Linear(64, 16, bias=False).to(dtype)
    for p in head.parameters():
        p.requires_grad_(False)
    out = llm(inputs_embeds=E, return_dict=True)["last_hidden_state"]
    torch.nn.functional.cross_entropy(head(out).float().reshape(-1, 16), torch.randint(0, 16, (E.shape[0] * E.shape[1],))).backward()
    rep["A_grad_norm"] = {f"{n}.{w}": float(getattr(m, w).grad.norm()) for n, m in mods.items() for w in ("lora_A", "lora_B")}
    rep["A_grad_on_frozen"] = [n for n, p in llm.named_parameters() if "lora_" not in n and p.grad is not None]
    # B: attention-only model, full vs cached
    cfgb, llmb, attnb = prepare(a.snapshot, dtype, "*-*-*-")
    g0, c0 = run(llmb, cfgb, E, P, dtype)
    modsb = TL.install(llmb, layers=tuple(attnb))
    g1, c1 = run(llmb, cfgb, E, P, dtype)
    with torch.no_grad():
        for m in modsb.values():
            m.lora_B.normal_(0, 0.05)
    g2, c2 = run(llmb, cfgb, E, P, dtype)
    rep["B_B0_full_equal_bitwise"] = bool(torch.equal(g0, g1))
    rep["B_B0_cached_equal_bitwise"] = bool(torch.equal(c0, c1))
    rep["B_cache_vs_full_noLoRA_max"] = mx(c0, g0)
    rep["B_cache_vs_full_LoRA_max"] = mx(c2, g2)
    rep["B_lora_effect_full_max"] = mx(g2, g0)
    rep["B_lora_effect_cached_max"] = mx(c2, c0)
    noise = max(rep["B_cache_vs_full_noLoRA_max"], 1e-6)
    rep["checks"] = {
        "A_B0_identity_full": rep["A_B0_equal_bitwise"],
        "A_lora_changes_output": rep["A_lora_effect_max"] > 1e-2,
        "A_all_lora_grads_nonzero": all(v > 0 for v in rep["A_grad_norm"].values()),
        "A_no_grad_on_frozen": not rep["A_grad_on_frozen"] and not rep["A_trainable_non_lora"],
        "B_B0_identity_full": rep["B_B0_full_equal_bitwise"],
        "B_B0_identity_cached": rep["B_B0_cached_equal_bitwise"],
        "B_cache_noise_not_increased_by_lora": rep["B_cache_vs_full_LoRA_max"] <= 2 * noise + 1e-2,
        "B_lora_effect_above_cache_noise": min(rep["B_lora_effect_full_max"], rep["B_lora_effect_cached_max"]) > 10 * noise,
    }
    rep["pass"] = all(rep["checks"].values())
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "A_grad_norm"}, indent=1))
    print("min lora grad", min(rep["A_grad_norm"].values()))


if __name__ == "__main__":
    main()
