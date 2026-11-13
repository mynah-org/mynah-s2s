"""CPU tests for thinker_lora (L1 preparation; no model weights)."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinker_lora as TL  # noqa: E402


class Mixer(nn.Module):
    def __init__(self, h=64, q=80, kv=16):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(h, q, bias=False), nn.Linear(h, kv, bias=False), nn.Linear(h, kv, bias=False)

    def forward(self, x):
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], -1)


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.block_type = kind
        self.mixer = Mixer() if kind == "attention" else nn.Linear(64, 64)


class FakeLLM(nn.Module):
    def __init__(self, kinds=("mamba", "attention", "mlp", "attention")):
        super().__init__()
        self.layers = nn.ModuleList([Block(k) for k in kinds])

    def forward(self, x):
        return torch.cat([b.mixer(x) for b in self.layers if b.block_type == "attention"], -1)


def bf16_llm():
    torch.manual_seed(0)
    return FakeLLM().to(torch.bfloat16)


def test_identity_at_init_bitwise_bf16():
    llm = bf16_llm()
    x = torch.randn(3, 7, 64).to(torch.bfloat16)
    ref = llm(x)
    TL.install(llm, layers=(1, 3))
    assert torch.equal(llm(x), ref)


def test_install_targets_freeze_and_counts():
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1, 3), r=8)
    assert sorted(mods) == ["layers.1.mixer.q_proj", "layers.1.mixer.v_proj", "layers.3.mixer.q_proj", "layers.3.mixer.v_proj"]
    assert TL.n_params(mods) == 2 * (8 * (64 + 80) + 8 * (64 + 16))
    trainable = [n for n, p in llm.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable) and len(trainable) == 8
    assert all(p.dtype == torch.float32 for p in TL.lora_parameters(mods))
    with pytest.raises(ValueError):
        TL.install(llm, layers=(0,))                     # mamba block
    with pytest.raises(ValueError):
        TL.install(llm, layers=(1,))                     # already installed


def test_real_layer_count_formula():
    # Nemotron-Nano-9B-v2 attention: q [5120, 4480], v [1024, 4480]; 4 layers, r 8 -> 483 328 (L0 card)
    assert 4 * (8 * (4480 + 5120) + 8 * (4480 + 1024)) == 483328


def test_nonzero_lora_changes_output_and_survives_cast():
    llm = bf16_llm()
    x = torch.randn(3, 7, 64).to(torch.bfloat16)
    ref = llm(x)
    mods = TL.install(llm, layers=(1, 3))
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.normal_(0, 1e-2)
    out = llm(x)
    assert not torch.equal(out, ref)
    s = TL.branch_survival(mods["layers.1.mixer.q_proj"], x)
    assert s["changed_frac"] > 0.5 and 0.5 < s["realised_over_delta"] < 1.5
    TL.set_enabled(mods, False)
    assert torch.equal(llm(x), ref)


def test_tiny_delta_is_reported_as_lost():
    llm = bf16_llm()
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    mods = TL.install(llm, layers=(1,))
    with torch.no_grad():
        mods["layers.1.mixer.q_proj"].lora_B.fill_(1e-7)
    s = TL.branch_survival(mods["layers.1.mixer.q_proj"], x)
    assert s["changed_frac"] < 0.01                      # the diagnostic sees the rounding


def test_gradients_reach_B_at_init_and_both_after():
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1, 3))
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    llm(x).float().pow(2).mean().backward()
    for m in mods.values():
        assert m.lora_B.grad is not None and float(m.lora_B.grad.abs().sum()) > 0
        assert float(m.lora_A.grad.abs().sum()) == 0     # B = 0 -> dL/dA = 0 by construction
        assert m.lora_B.grad.dtype == torch.float32
    for n, p in llm.named_parameters():
        if "lora_" not in n:
            assert p.grad is None
    llm.zero_grad()
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.normal_(0, 1e-2)
    llm(x).float().pow(2).mean().backward()
    assert all(float(m.lora_A.grad.abs().sum()) > 0 for m in mods.values())


def test_function_head_only_loss_reaches_lora():
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1, 3))
    head = nn.Linear(2 * (80 + 16 + 16), 11).to(torch.bfloat16)
    for p in head.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    logits = head(llm(x)).float()
    nn.functional.cross_entropy(logits.reshape(-1, 11), torch.randint(0, 11, (10,))).backward()
    assert all(float(m.lora_B.grad.abs().sum()) > 0 for m in mods.values())


def test_small_adam_update_kept_in_fp32_but_lost_in_bf16():
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1,))
    prev = TL.state(mods)
    opt = torch.optim.AdamW(TL.lora_parameters(mods), lr=1e-5, weight_decay=0.0)
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    llm(x).float().pow(2).mean().backward()
    opt.step()
    u = TL.update_norms(prev, mods)
    assert u["layers.1.mixer.q_proj.lora_B"]["upd"] > 0       # FP32 parameter moved
    wb = torch.full((4,), 0.05, dtype=torch.bfloat16)          # the same step on a bf16-stored weight is swallowed
    assert torch.equal(wb + torch.tensor(1e-5, dtype=torch.bfloat16), wb)


def test_state_round_trip():
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1, 3))
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.normal_()
    st = TL.state(mods)
    llm2 = bf16_llm()
    mods2 = TL.install(llm2, layers=(1, 3))
    TL.load_state(mods2, st)
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    assert torch.equal(llm(x), llm2(x))
    with pytest.raises(KeyError):
        TL.load_state(mods2, {k: v for k, v in st.items() if "q_proj" not in k})


def test_capture_hook_does_not_replace_output():
    from l1_smoke import capture_hook
    llm = bf16_llm()
    mods = TL.install(llm, layers=(1,))
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    ref = llm(x)
    store = {}
    h = mods["layers.1.mixer.q_proj"].register_forward_hook(capture_hook(store, "q"))
    out = llm(x)
    h.remove()
    assert torch.equal(out, ref) and store["q"].shape == (2, 5, 64)
