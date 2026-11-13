"""GPU test isolation: every test MODULE starts from the same global float32 matmul state and leaves no trace.

ReferenceEngine (like NVIDIA's inference wrapper) sets torch.set_float32_matmul_precision("medium") / TF32 globally.
Without this fixture that state leaks into later tests in the same pytest process, and FP32 exactness tests (e.g.
batched vs B=1 Thinker at atol 1e-4) then compare TF32 results (~8e-4 apart) and fail depending on test order.
Each module starts with full FP32 ("highest", TF32 off); modules that build an engine get the engine's own setting
(as in serving), and the state is restored when the module ends. Module scope, because engines are module fixtures.
"""

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def _fp32_matmul_state():
    saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(saved[0])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[1], saved[2]
