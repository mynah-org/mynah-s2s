"""Runs last (file order): the default tier must not have loaded torch (CI without torch, laptops without CUDA)."""

import os
import sys


def test_default_tier_never_imports_torch(pytestconfig):
    opted = pytestconfig.getoption("--torch") or pytestconfig.getoption("--gpu") or \
        os.environ.get("F2_TORCH_TESTS") == "1" or os.environ.get("F2_GPU_TESTS") == "1"
    if not opted:
        assert "torch" not in sys.modules
