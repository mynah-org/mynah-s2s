"""CPU tests of the own Qwen3.5 / MoE inference on tiny random weights (no checkpoint): the chunked delta rule equals
the recurrent one, prefill + decode equals one longer prefill, the session snapshot continues exactly, AWQ unpacking
round-trips, weight-only int8 / int4 stay close to bf16, and the MoE decode and prefill expert paths agree."""

import pytest

torch = pytest.importorskip("torch")

from frankenstein2.llm.qwen_own.config import QwenCfg  # noqa: E402
from frankenstein2.llm.qwen_own.model import Model, chunk_delta_rule, step_delta_rule, l2norm  # noqa: E402
from frankenstein2.llm.qwen_own.quant import AWQ_REVERSE_ORDER, QLinear, awq_dequant  # noqa: E402

torch.manual_seed(0)


def tiny_cfg(moe=False):
    return QwenCfg(hidden=64, layers=4, layer_types=["linear_attention", "linear_attention", "linear_attention",
                                                     "full_attention"], vocab=97, eps=1e-6, heads=4, kv_heads=2,
                   head_dim=32, rope_theta=1e7, rotary_dim=8, attn_output_gate=True, lin_k_heads=2, lin_v_heads=4,
                   lin_k_dim=16, lin_v_dim=16, conv_k=4, intermediate=96, num_experts=8 if moe else 0,
                   top_k=2 if moe else 0, moe_intermediate=32, shared_intermediate=32, tie_embeddings=not moe,
                   eos_ids=[0])


def tiny_weights(c: QwenCfg, awq=False):
    r = lambda *s: (torch.randn(*s) * 0.05).to(torch.bfloat16)          # noqa: E731
    W = {"embed_tokens.weight": r(c.vocab, c.hidden), "norm.weight": r(c.hidden)}
    if not c.tie_embeddings:
        W["lm_head.weight"] = r(c.vocab, c.hidden)
    cd = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim
    for i, k in enumerate(c.layer_types):
        p = f"layers.{i}."
        W[p + "input_layernorm.weight"] = r(c.hidden)
        W[p + "post_attention_layernorm.weight"] = r(c.hidden)
        if k == "linear_attention":
            a = p + "linear_attn."
            W[a + "in_proj_qkv.weight"] = r(cd, c.hidden)
            W[a + "in_proj_z.weight"] = r(c.lin_v_heads * c.lin_v_dim, c.hidden)
            W[a + "in_proj_b.weight"] = r(c.lin_v_heads, c.hidden)
            W[a + "in_proj_a.weight"] = r(c.lin_v_heads, c.hidden)
            W[a + "conv1d.weight"] = r(cd, 1, c.conv_k) * 10
            W[a + "A_log"] = torch.rand(c.lin_v_heads).log()
            W[a + "dt_bias"] = torch.ones(c.lin_v_heads).to(torch.bfloat16)
            W[a + "norm.weight"] = torch.ones(c.lin_v_dim).to(torch.bfloat16)
            W[a + "out_proj.weight"] = r(c.hidden, c.lin_v_heads * c.lin_v_dim)
        else:
            a = p + "self_attn."
            W[a + "q_proj.weight"] = r(c.heads * c.head_dim * 2, c.hidden)
            W[a + "k_proj.weight"] = r(c.kv_heads * c.head_dim, c.hidden)
            W[a + "v_proj.weight"] = r(c.kv_heads * c.head_dim, c.hidden)
            W[a + "o_proj.weight"] = r(c.hidden, c.heads * c.head_dim)
            W[a + "q_norm.weight"] = r(c.head_dim)
            W[a + "k_norm.weight"] = r(c.head_dim)
        m = p + "mlp."
        if c.moe:
            W[m + "gate.weight"] = r(c.num_experts, c.hidden)
            W[m + "shared_expert_gate.weight"] = r(1, c.hidden)
            for n, s in (("gate_proj", (c.shared_intermediate, c.hidden)), ("up_proj", (c.shared_intermediate, c.hidden)),
                         ("down_proj", (c.hidden, c.shared_intermediate))):
                W[m + f"shared_expert.{n}.weight"] = r(*s)
            for n, (o, i_) in (("gate_proj", (c.moe_intermediate, c.hidden)), ("up_proj", (c.moe_intermediate, c.hidden)),
                               ("down_proj", (c.hidden, c.moe_intermediate))):
                if awq:
                    qs = [awq_pack(r(o, i_).float() * 4, 16) for _ in range(c.num_experts)]
                    for j, part in enumerate(("qweight", "qzeros", "scales")):
                        W[m + f"experts.{n}.{part}"] = torch.stack([q[j] for q in qs])
                else:
                    W[m + f"experts.{n}"] = torch.stack([r(o, i_) for _ in range(c.num_experts)])
        else:
            W[m + "gate_proj.weight"] = r(c.intermediate, c.hidden)
            W[m + "up_proj.weight"] = r(c.intermediate, c.hidden)
            W[m + "down_proj.weight"] = r(c.hidden, c.intermediate)
    return W


def awq_pack(w: torch.Tensor, group: int):
    """Test helper: [out, in] float -> AutoAWQ GEMM (qweight [in, out/8] int32, qzeros, scales fp16)."""
    wt = w.t().contiguous()                                              # [in, out]
    i_, o = wt.shape
    g = wt.reshape(i_ // group, group, o)
    lo, hi = g.amin(1), g.amax(1)
    s = ((hi - lo) / 15).clamp_min(1e-6)
    z = torch.round(-lo / s).clamp(0, 15)
    q = torch.round(g / s[:, None] + z[:, None]).clamp(0, 15).reshape(i_, o).to(torch.int32)

    def pack(v):
        v = v.reshape(v.shape[0], -1, 8)
        order = torch.tensor(AWQ_REVERSE_ORDER).argsort()                   # inverse of the unpack permutation
        v = v[..., order]
        out = torch.zeros(v.shape[:-1], dtype=torch.int32)
        for j in range(8):
            out |= v[..., j] << (4 * j)
        return out
    return pack(q), pack(z.to(torch.int32)), s.to(torch.float16)


def test_awq_roundtrip():
    w = torch.randn(64, 32)
    qw, qz, s = awq_pack(w, 16)
    deq = awq_dequant(qw, qz, s, 16).float()
    assert deq.shape == w.shape
    assert (deq - w).abs().max() < (w.abs().max() / 7)


def test_chunk_equals_recurrent():
    B, T, H, dk, dv = 2, 150, 3, 16, 8
    q, k = l2norm(torch.randn(B, T, H, dk)), l2norm(torch.randn(B, T, H, dk))
    v = torch.randn(B, T, H, dv)
    g = -torch.rand(B, T, H) * 0.3
    beta = torch.rand(B, T, H)
    s0 = torch.randn(B, H, dk, dv) * 0.1
    out_c, s_c = chunk_delta_rule(q, k, v, g, beta, s0, chunk=64)
    s, outs = s0.clone(), []
    for t in range(T):
        o, s = step_delta_rule(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], s)
        outs.append(o)
    out_r = torch.stack(outs, 1)
    assert torch.allclose(out_c, out_r, atol=2e-4), (out_c - out_r).abs().max()
    assert torch.allclose(s_c, s, atol=2e-4)


@pytest.mark.parametrize("moe,awq", [(False, False), (True, False), (True, True)])
def test_prefill_decode_consistency(moe, awq):
    c = tiny_cfg(moe)
    m = Model(c, tiny_weights(c, awq), torch.device("cpu"), awq_group=16 if awq else None)
    ids = torch.randint(1, c.vocab, (1, 20))
    st = m.new_state(2, 64)
    slot = torch.tensor([0])
    full = m.forward(ids, st, slot, torch.arange(20)[None], torch.tensor([20]), prefill=True)
    st2 = m.new_state(2, 64)
    m.forward(ids[:, :19], st2, slot, torch.arange(19)[None], torch.tensor([19]), prefill=True)
    last = m.forward(ids[:, 19:], st2, slot, torch.tensor([[19]]), torch.tensor([20]), prefill=False)
    rel = ((full - last).norm() / full.norm()).item()             # bf16 paths (expert loop vs gathered batch)
    assert rel < 0.05, rel
    assert full.argmax() == last.argmax()


def test_batched_decode_matches_single():
    c = tiny_cfg()
    m = Model(c, tiny_weights(c), torch.device("cpu"))
    st = m.new_state(2, 64)
    a, b = torch.randint(1, c.vocab, (1, 9)), torch.randint(1, c.vocab, (1, 13))
    m.forward(a, st, torch.tensor([0]), torch.arange(9)[None], torch.tensor([9]), prefill=True)
    m.forward(b, st, torch.tensor([1]), torch.arange(13)[None], torch.tensor([13]), prefill=True)
    both = m.forward(torch.tensor([[5], [7]]), st, torch.tensor([0, 1]), torch.tensor([[9], [13]]),
                     torch.tensor([10, 14]), prefill=False)
    st1 = m.new_state(1, 64)
    m.forward(b, st1, torch.tensor([0]), torch.arange(13)[None], torch.tensor([13]), prefill=True)
    one = m.forward(torch.tensor([[7]]), st1, torch.tensor([0]), torch.tensor([[13]]), torch.tensor([14]), prefill=False)
    assert torch.allclose(both[1], one[0], atol=2e-2), (both[1] - one[0]).abs().max()


def test_continue_from_prefix():
    c = tiny_cfg()
    m = Model(c, tiny_weights(c), torch.device("cpu"))
    ids = torch.randint(1, c.vocab, (1, 30))
    st = m.new_state(1, 64)
    ref = m.forward(ids, st, torch.tensor([0]), torch.arange(30)[None], torch.tensor([30]), prefill=True)
    st2 = m.new_state(1, 64)
    m.forward(ids[:, :17], st2, torch.tensor([0]), torch.arange(17)[None], torch.tensor([17]), prefill=True)
    got = m.forward(ids[:, 17:], st2, torch.tensor([0]), torch.arange(17, 30)[None], torch.tensor([30]), prefill=True)
    assert torch.allclose(ref, got, atol=3e-2, rtol=3e-2), (ref - got).abs().max()


@pytest.mark.parametrize("mode,tol", [("int8", 0.01), ("int4", 0.1)])
def test_weight_only_quant(mode, tol):
    w = torch.randn(256, 512).to(torch.bfloat16)
    x = torch.randn(3, 512).to(torch.bfloat16)
    ref = (x.float() @ w.float().t())
    got = QLinear(w, mode=mode, group=64)(x).float()
    assert ((got - ref).norm() / ref.norm()) < tol
