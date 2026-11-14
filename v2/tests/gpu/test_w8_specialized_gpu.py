"""GB10 smoke/parity coverage for Q8 counterparts of the specialized Q4 paths."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mynah_v2.model.w8_specialized import (
    ConvTranspose1dW8,
    DepthwiseConv1dW8,
    Int8LSTM,
    RVQInt8Embedding,
    SelectedMoGW8,
    W8PointwiseConv1d,
)


def _check_convtranspose_w8(conv):
    q = ConvTranspose1dW8(conv)
    x = torch.randn(2, conv.in_channels, 13, device="cuda", dtype=torch.bfloat16)
    ref = conv(x)
    got = q(x)
    assert got.shape == ref.shape
    assert torch.allclose(got.float(), ref.float(), atol=0.15, rtol=0.15)


def test_w8_convtranspose_gpu_dense_and_grouped():
    _check_convtranspose_w8(nn.ConvTranspose1d(
        4, 5, kernel_size=3, stride=2, padding=1, device="cuda", dtype=torch.bfloat16
    ))
    _check_convtranspose_w8(nn.ConvTranspose1d(
        4, 6, kernel_size=4, stride=2, padding=1, groups=2,
        device="cuda", dtype=torch.bfloat16
    ))


def test_w8_pointwise_and_depthwise_gpu():
    conv = nn.Conv1d(8, 16, 1, bias=True, device="cuda", dtype=torch.bfloat16)
    q = W8PointwiseConv1d(conv)
    x = torch.randn(2, 8, 5, device="cuda", dtype=torch.bfloat16)
    assert torch.allclose(q(x), conv(x), atol=0.10, rtol=0.10)

    dw = nn.Conv1d(8, 8, 5, groups=8, bias=True, device="cuda", dtype=torch.bfloat16)
    qdw = DepthwiseConv1dW8(dw)
    x2 = torch.randn(2, 8, 11, device="cuda", dtype=torch.bfloat16)
    assert torch.allclose(qdw(x2), dw(x2), atol=0.10, rtol=0.10)


def test_selected_mog_w8_gpu():
    n, d, k = 3, 64, 128
    lin = nn.Linear(k, n * d, bias=True, device="cuda", dtype=torch.bfloat16)
    q = SelectedMoGW8(lin, n, d)
    x = torch.randn(5, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, n, (5,), device="cuda")
    ref = torch.einsum("bk,bdk->bd", x.float(), lin.weight.view(n, d, k)[ids].float()) + lin.bias.view(n, d)[ids].float()
    assert torch.allclose(q.selected(x, ids).float(), ref, atol=0.10, rtol=0.10)


def test_rvq_w8_gpu_and_encoder():
    table = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    q = RVQInt8Embedding(table)
    code = torch.randint(0, 32, (4, 3), device="cuda")
    ref = torch.stack([table[0, code[:, 0]], table[1, code[:, 1]], table[2, code[:, 2]]]).sum(0)
    assert torch.allclose(q(code).to(table.dtype), ref, atol=0.15, rtol=0.15)
    enc = q
    residual = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out_code = torch.zeros(4, 3, device="cuda", dtype=torch.int64)
    from mynah_v2.model.w8_specialized import RVQInt8Encoder
    RVQInt8Encoder(enc, block_v=16)(residual, out_code, 0, 3)
    assert out_code.shape == (4, 3)


def test_rnnt_prediction_w8_gpu():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(32, 24, 2).cuda().bfloat16()

    base = Wrapper()
    q = Int8LSTM(base)
    fused = Int8LSTM(base, fused=True)
    x = torch.randn(3, 4, 32, device="cuda", dtype=torch.bfloat16)
    y, (h, c) = q(x)
    yf, (hf, cf) = fused(x)
    assert y.shape == (3, 4, 24)
    assert h.shape == c.shape == (2, 4, 24)
    assert torch.allclose(yf.float(), y.float(), atol=0.08, rtol=0.08)
    assert torch.allclose(hf.float(), h.float(), atol=0.08, rtol=0.08)
    assert torch.allclose(cf.float(), c.float(), atol=0.08, rtol=0.08)
