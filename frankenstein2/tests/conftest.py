"""Test tiers for Frankenstein v2, so a plain ``pytest frankenstein2/tests`` runs anywhere (CI, a laptop):

* default            pure Python + numpy: control, pipeline, LLM plugin logic, tools, tracing, text. No torch, no GPU,
                     no weights, no network. Seconds.
* ``--torch``        also the CPU tests on tiny random weights that need torch (own Qwen engine pieces, jump-forward).
                     Same as F2_TORCH_TESTS=1.
* ``--gpu``          also the GPU smoke tests (``tests/gpu``): real models on CUDA, paths from the F2_* variables
                     printed by ``frankenstein2/scripts/setup_models.py``. Same as F2_GPU_TESTS=1. Implies --torch.

Modules of an opted-out tier are not even imported (torch is never loaded by the default run).
"""

import os
from pathlib import Path

TORCH_MODULES = {"test_qwen_own.py", "test_jump_forward.py"}
HERE = Path(__file__).parent


def pytest_addoption(parser):
    parser.addoption("--torch", action="store_true", help="run the torch (CPU, tiny weights) tests")
    parser.addoption("--gpu", action="store_true", help="run the GPU smoke tests (real models, CUDA)")


def _on(config, opt, env):
    return config.getoption(opt) or os.environ.get(env) == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs torch (CPU); opt-in with --torch")
    config.addinivalue_line("markers", "gpu: needs CUDA and model weights; opt-in with --gpu")


def pytest_ignore_collect(collection_path, config):
    p = Path(collection_path)
    gpu = _on(config, "--gpu", "F2_GPU_TESTS")
    torch = gpu or _on(config, "--torch", "F2_TORCH_TESTS")
    if p.name in TORCH_MODULES and p.parent == HERE and not torch:
        return True
    if (HERE / "gpu") in (p, *p.parents) and not gpu:
        return True
    return None
