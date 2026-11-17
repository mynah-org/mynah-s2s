import torch

from mynah_v2.model.mog_q4 import SelectedMoGQ4
from mynah_v2.model.rvq_q4 import RVQInt4Embedding, RVQInt4Encoder
from mynah_v2.model.depthwise_q4 import DepthwiseConv1dQ4
from mynah_v2.model.rnnt_q4 import Int4LSTM


def test_rvq_q4_preserves_zero_sentinel_and_shape():
    torch.manual_seed(0)
    table = torch.randn(2, 8, 16)
    q = RVQInt4Embedding(table, group=8)
    code = torch.tensor([[[0, 8], [3, 2]]])
    y = q(code)
    assert y.shape == (1, 2, 16)
    expected = torch.zeros(16)
    # The sentinel is the last code in both quantizer planes.
    assert torch.allclose(q(torch.full((1, 1, 2), 8, dtype=torch.long))[0, 0], expected)


def test_rvq_q4_fused_encoder_cpu_shape_and_codes():
    torch.manual_seed(4)
    table = torch.randn(3, 8, 16)
    q = RVQInt4Embedding(table, group=8)
    enc = RVQInt4Encoder(q)
    residual = torch.randn(2, 1, 16)
    code = torch.full((2, 1, 3), 8, dtype=torch.long)
    out = enc(residual, code, 0, 3)
    assert out.shape == code.shape
    assert torch.all((out >= 0) & (out < 8))


def test_mog_selected_q4_cpu_shape_and_indices():
    torch.manual_seed(1)
    lin = torch.nn.Linear(16, 3 * 8, bias=False)
    q = SelectedMoGQ4(lin, num_predictions=3, out_size=8, group=8)
    x = torch.randn(2, 16)
    ids = torch.tensor([0, 2])
    y = q.selected(x, ids)
    ref = torch.stack((x[0] @ lin.weight[:8].t(), x[1] @ lin.weight[16:].t()))
    assert y.shape == (2, 8)
    assert torch.max((y - ref).abs()) < 0.5


def test_mog_selected_q4_accepts_parameter_tensor():
    torch.manual_seed(2)
    weight = torch.randn(3, 8, 16)
    q = SelectedMoGQ4.from_tensor(weight, group=8)
    y = q.selected(torch.randn(2, 16), torch.tensor([0, 2]))
    assert y.shape == (2, 8)


def test_depthwise_q4_cpu_shape_and_approximation():
    torch.manual_seed(3)
    conv = torch.nn.Conv1d(4, 4, 7, groups=4)
    q = DepthwiseConv1dQ4(conv)
    x = torch.randn(2, 4, 19)
    y = q(x)
    ref = conv(x)
    assert y.shape == ref.shape == (2, 4, 13)
    assert torch.max((y - ref).abs()) < 0.15


def test_rnnt_lstm_q4_cpu_contract():
    torch.manual_seed(5)
    source = torch.nn.Module()
    source.lstm = torch.nn.LSTM(16, 16, 2)
    source.dropout = None
    q = Int4LSTM(source, group=8)
    x = torch.randn(2, 3, 16)
    y, state = q(x)
    ref, ref_state = source.lstm(x)
    assert y.shape == ref.shape == (2, 3, 16)
    assert state[0].shape == ref_state[0].shape == (2, 3, 16)
    assert torch.isfinite(y).all()
    h0 = torch.zeros_like(state[0])
    c0 = torch.zeros_like(state[1])
    h0_ref, c0_ref = h0.clone(), c0.clone()
    y2, state2 = q(torch.randn(1, 3, 16), (h0, c0))
    assert y2.shape == (1, 3, 16)
    assert torch.equal(h0, h0_ref) and torch.equal(c0, c0_ref)
    empty, empty_state = q(torch.empty(0, 3, 16))
    assert empty.shape == (0, 3, 16)
    assert empty_state[0].shape == (2, 3, 16)
