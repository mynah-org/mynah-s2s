"""INT4 dispatch: M == 1 delegates to V2's Int4Linear unchanged; M > 1 equals the reference dequantized matmul.

Needs torch and V2 on the GPU host (random small tensors; skipped elsewhere: no model or tensor test on the laptop).
Run there with FRANK_TORCH_TESTS=1 PYTHONPATH=<v2/src> python -m pytest tests/test_int4_dispatch.py
"""

from __future__ import annotations

import os

import pytest

if not os.environ.get("FRANK_TORCH_TESTS"):
    pytest.skip("torch tests run on the GPU host only (FRANK_TORCH_TESTS=1)", allow_module_level=True)

import torch  # noqa: E402

from e1.int4_dispatch import Int4Dispatch, _nibbles  # noqa: E402


def _q4(lin):
    from mynah_v2.model.quant import Int4Linear
    return Int4Linear(lin, max_group=128, clip=False)


def test_dequant_matches_v2_reference():
    from mynah_v2.model.quant import dequantize_int4_reference
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin = torch.nn.Linear(512, 384, bias=False).to(dev, torch.bfloat16)
    q4 = _q4(lin)
    d = Int4Dispatch(q4, _nibbles(lin.weight.data, q4.group), {})
    ref = dequantize_int4_reference(lin.weight.data, q4.group)          # fp32 scale / min (not bf16-rounded)
    got = d.dequant().float()
    # the kernel and the dispatch use the bf16-rounded scales / zeros: about 2^-8 relative to the group range
    assert (got - ref).abs().max().item() <= 0.02 * float(lin.weight.float().abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tinygemm int4 kernel: CUDA")
def test_m1_bit_identical_and_mN_matches_reference_matmul():
    torch.manual_seed(1)
    lin = torch.nn.Linear(1024, 768, bias=False).cuda().to(torch.bfloat16)
    q4 = _q4(lin)
    d = Int4Dispatch(q4, _nibbles(lin.weight.data, q4.group), {})
    x1 = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(d(x1), q4(x1))                                     # M == 1: unchanged path
    xn = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    y = d(xn).float()
    ref = xn.float() @ d.dequant().float().t()
    assert (y - ref).abs().max().item() <= 0.05 * ref.abs().max().item()  # bf16 GEMM vs fp32 reference
