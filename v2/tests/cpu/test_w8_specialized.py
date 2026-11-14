"""CPU contracts for the Q8 counterparts of the specialized Q4 paths."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mynah_v2.model.w8_specialized import (
    DepthwiseConv1dW8,
    Int8LSTM,
    RVQInt8Embedding,
    SelectedMoGW8,
    W8PointwiseConv1d,
)


def test_w8_pointwise_and_depthwise_match_linear_convolution():
    conv = nn.Conv1d(8, 12, 1, bias=True)
    q = W8PointwiseConv1d(conv)
    x = torch.randn(2, 8, 5)
    assert torch.allclose(q(x), conv(x), atol=0.08, rtol=0.08)

    dw = nn.Conv1d(8, 8, 5, groups=8, bias=True)
    qdw = DepthwiseConv1dW8(dw)
    x2 = torch.randn(2, 8, 11)
    assert torch.allclose(qdw(x2), dw(x2), atol=0.08, rtol=0.08)


def test_selected_mog_w8_matches_selected_linear():
    n, d, k = 3, 4, 16
    lin = nn.Linear(k, n * d, bias=True)
    q = SelectedMoGW8(lin, n, d)
    x = torch.randn(7, k)
    ids = torch.randint(0, n, (7,))
    ref = torch.einsum("bk,bdk->bd", x, lin.weight.view(n, d, k)[ids]) + lin.bias.view(n, d)[ids]
    assert torch.allclose(q.selected(x, ids), ref, atol=0.08, rtol=0.08)


def test_rvq_embedding_w8_matches_table_sum():
    table = torch.randn(3, 7, 16)
    q = RVQInt8Embedding(table)
    code = torch.tensor([[0, 1, 6], [7, 2, 3]])
    safe = code.clamp_max(6)
    ref = torch.stack([table[0, safe[:, 0]], table[1, safe[:, 1]], table[2, safe[:, 2]]]).sum(0)
    ref[1] = table[1, 2] + table[2, 3]
    assert torch.allclose(q(code), ref, atol=0.08, rtol=0.08)


def test_int8_lstm_w8_contract_and_shape():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(8, 6, 2)

    q = Int8LSTM(Wrapper())
    y, state = q(torch.randn(4, 3, 8))
    assert y.shape == (4, 3, 6)
    assert state[0].shape == state[1].shape == (2, 3, 6)
