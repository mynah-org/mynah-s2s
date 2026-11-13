"""Packing / reference dequantization of the Triton weight-only formats agree with the simulated formats."""

import torch

from mynah_v2.model import precision as P
from mynah_v2.model import wo_gemv as W


def test_pack_dequant_matches_simulation():
    torch.manual_seed(0)
    w = (torch.randn(48, 512) * 0.02)
    assert torch.allclose(W.dequant(W.pack_w8(w)), P._int8_channel(w), atol=1e-7)
    assert torch.allclose(W.dequant(W.pack_f8(w)), P._fp8_channel(w), atol=1e-7)
    assert torch.allclose(W.dequant(W.pack_nvf4(w)), P._nvfp4(w), atol=1e-7)
    # w4 keeps scale/min in bf16 (kernel storage): close to the fp32-parameter simulation
    d4, s4 = W.dequant(W.pack_w4(w)), P._int4_group(w, 128)
    assert float((d4 - s4).abs().max()) < 2e-3 * float(w.abs().max()) * 10


def test_bytes_per_weight():
    w = torch.randn(64, 1024)
    n = w.numel()
    assert W.nbytes(W.pack_w8(w)) == n + 64 * 4
    assert W.nbytes(W.pack_w4(w)) == n // 2 + 2 * (n // 128) * 2
    assert W.nbytes(W.pack_nvf4(w)) == n // 2 + n // 16 + 4


def test_nibble_order_low_is_even_k():
    w = torch.zeros(1, 128)
    w[0, 0], w[0, 1] = -1.0, 1.0
    p = W.pack_w4(w)
    assert int(p["q"][0, 0] & 0xF) == 0 and int(p["q"][0, 0] >> 4) == 15
