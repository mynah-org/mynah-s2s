"""Triton weight-only GEMV kernels vs the reference dequantized product, on real Thinker shapes and awkward tails."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu

SHAPES = [(4480, 15680), (15680, 4480), (4480, 22656), (10240, 4480), (4480, 2048), (1000, 333), (512, 17)]


@pytest.fixture(scope="module", autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def rel(y, r):
    return float((y.float() - r.float()).norm() / r.float().norm().clamp_min(1e-12))


@pytest.mark.parametrize("fmt", ["w8", "f8"])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 16, 37])
def test_w8_family_matches_dequant(fmt, k, n, m):
    from mynah_v2.model import wo_gemv as W
    torch.manual_seed(k + n + m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    p = W.PACKERS[fmt](w)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    ref = F.linear(x.float(), W.dequant(p))
    y = W.gemv(x, p)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    assert rel(y, ref) < 6e-3                     # fp32 accumulation, bf16 output rounding
    # batched rows == one row at a time (weights reused, no cross-row mixing)
    for i in range(min(m, W.MAX_BATCHED_M)):
        assert m > W.MAX_BATCHED_M or rel(y[i], W.gemv(x[i:i + 1], p)[0]) < 2e-3


@pytest.mark.parametrize("fmt", ["w4", "nvf4"])
@pytest.mark.parametrize("k,n", [s for s in SHAPES if s[0] % 16 == 0])
def test_4bit_decode_matches_dequant(fmt, k, n):
    from mynah_v2.model import wo_gemv as W
    torch.manual_seed(k * 7 + n)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    p = W.PACKERS[fmt](w)
    x = torch.randn(2, k, device="cuda").to(torch.bfloat16)
    assert rel(W.gemv(x, p), F.linear(x.float(), W.dequant(p))) < 6e-3


def test_wolinear_3d_input_bias_and_fp32_activations():
    from mynah_v2.model import wo_gemv as W
    lin = torch.nn.Linear(4480, 1024, bias=True, device="cuda", dtype=torch.bfloat16)
    q = W.WOLinear(lin, "w8")
    assert q.weight.numel() == 0 and q.weight.dtype == torch.bfloat16
    x = torch.randn(1, 1, 4480, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x.float(), W.dequant(q.packed()), lin.bias.float())
    assert q(x).shape == (1, 1, 1024) and rel(q(x), ref) < 6e-3
    xf = torch.randn(1, 300, 4480, device="cuda")                        # prefill-sized, fp32
    assert rel(q(xf), F.linear(xf, W.dequant(q.packed()), lin.bias.float())) < 6e-3
