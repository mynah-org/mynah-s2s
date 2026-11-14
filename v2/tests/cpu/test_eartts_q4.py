import pytest
import torch

from mynah_v2.model.eartts_q4 import GraphSafeQ4Linear, triton_cdiv


def test_static_grid_division():
    assert triton_cdiv(0, 2) == 0
    assert triton_cdiv(2, 2) == 1
    assert triton_cdiv(3, 2) == 2


def test_eartts_q4_requires_cuda_weights():
    with pytest.raises(ValueError, match="requires CUDA"):
        GraphSafeQ4Linear(torch.nn.Linear(128, 64), group=128)
