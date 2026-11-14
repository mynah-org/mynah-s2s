import torch

from mynah_v2.model.mog_q4 import SelectedMoGQ4
from mynah_v2.model.rvq_q4 import RVQInt4Embedding


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
