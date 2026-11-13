import torch

from mynah_v2.model import perf
from mynah_v2.model.perf import fuse_rmsnorms, unfuse_rmsnorms


class Gemma3RMSNorm(torch.nn.Module):          # same code as transformers' Gemma3RMSNorm (4.56)
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def test_fused_rmsnorm_matches_gemma3_formula_and_is_reversible():
    torch.manual_seed(0)
    root = torch.nn.Sequential(Gemma3RMSNorm(64), torch.nn.Identity(), Gemma3RMSNorm(64))
    with torch.no_grad():
        for m in root:
            if isinstance(m, Gemma3RMSNorm):
                m.weight.normal_(0, 0.5)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 3, 64).to(dtype)
        ref = root(x)
        assert fuse_rmsnorms(root) == 2                # exact mode: FP32 math for every dtype
        assert fuse_rmsnorms(root) == 0                # idempotent
        out = root(x)
        assert out.dtype == ref.dtype
        d = (out.float() - ref.float()).abs()
        if dtype == torch.float32:
            assert d.max() <= 1e-5
        else:                                          # same formula as the reference: identical except rare 1-ulp ties
            assert (d <= ref.float().abs() * 2 ** -8 + 1e-6).float().mean() >= 0.99
            assert (d <= ref.float().abs() * 2 ** -7 + 1e-6).all()
        unfuse_rmsnorms(root)
        assert all("forward" not in m.__dict__ for m in root)
        assert torch.equal(root(x), ref)


def test_bf16_chain_exact_mode_equals_reference_native_mode_stays_at_bf16_noise():
    torch.manual_seed(1)
    layers = [Gemma3RMSNorm(256) for _ in range(6)]
    mix = [torch.randn(256, 256) / 16 for _ in range(6)]
    with torch.no_grad():
        for m in layers:
            m.weight.normal_(0, 0.5)
    x = torch.cat([torch.randn(8, 256) * 3, torch.randn(8, 256) * 1e-3])     # incl. near-zero rows

    def run(dt):
        h = x.to(dt)
        for m, w in zip(layers, mix):
            h = (m(h).float() @ w).to(dt) + h
        return h.float()

    def p99(a, b):
        return float(((a - b).abs() / b.abs().clamp_min(1e-2)).quantile(0.99))
    fp32, ref = run(torch.float32), run(torch.bfloat16)
    root = torch.nn.Sequential(*layers)
    fuse_rmsnorms(root)
    exact = run(torch.bfloat16)
    unfuse_rmsnorms(root)
    fuse_rmsnorms(root, bf16_native=True)
    native = run(torch.bfloat16)
    unfuse_rmsnorms(root)
    assert torch.equal(exact, ref)                                            # exact mode: bit-identical in BF16
    assert torch.isfinite(native).all()
    assert p99(native, fp32) <= 1.2 * p99(ref, fp32)                         # native: within BF16's own drift (+20 %)


def test_fusion_applies_under_profiler_labels_and_restores_them():
    from mynah_v2.runtime import op_profiler
    root = torch.nn.Sequential(Gemma3RMSNorm(16))
    op_profiler._wrap(root[0], "tts.bb.norm")
    labelled = root[0].forward
    x = torch.randn(2, 16)
    ref = root(x)
    assert fuse_rmsnorms(root) == 1
    assert root[0].forward._v2_label == "tts.bb.norm" and root[0].forward is not labelled
    assert (root(x) - ref).abs().max() <= 1e-5
    unfuse_rmsnorms(root)
    assert root[0].forward is labelled


def test_env_fallback_keeps_the_original_forward(monkeypatch):
    monkeypatch.setenv("V2_FUSED_RMSNORM", "0")
    assert perf._ernorm_enable(None)["norms"] == 0      # returns before touching the engine


def test_graph_profiles_enable_the_fused_norm():
    for name in ("optimized-graph", "optimized-graph-s4", "optimized-graph-s8"):
        assert perf.PROFILES[name][-1] == "eartts_fused_rmsnorm"
    assert "eartts_fused_rmsnorm" in perf.EXPERIMENTS
