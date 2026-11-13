"""CPU tests for opener_probe helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opener_probe import margin, opener_diff  # noqa: E402
from session_distill import PAD  # noqa: E402


def test_opener_diff_exact_and_variant():
    ref = [PAD, 2, 1, PAD, 5, 6, PAD, 7]
    assert opener_diff(ref + [PAD] * 5, ref, 10) == {"exact": True, "same_frames": True}
    shifted = [PAD, PAD, 2, 1, 5, 6, PAD, 7] + [PAD] * 4
    assert opener_diff(shifted, ref, 10) == {"exact": True, "same_frames": False}
    alt = [PAD, 2, 1, PAD, 5, 9, 6, 7] + [PAD] * 4
    d = opener_diff(alt, ref, 10)
    assert d == {"exact": False, "index": 3, "frame_own": 5, "frame_ref": 5, "tok_own": 9, "tok_ref": 6}
    longer = ref + [8, PAD]
    assert opener_diff(longer, ref, 10)["index"] == 5


def test_margin():
    assert margin([[3508, 2342, 12], [-0.4, -1.1, -3.0]], 2342, 3508) == -0.7
    assert margin([[3508, 12, 5], [-0.1, -3.0, -4.0]], 2342, 3508) is None
