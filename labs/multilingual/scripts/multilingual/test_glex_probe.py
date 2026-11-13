"""CPU tests for the G-LEX probe's synthetic targets and metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glex_probe as G  # noqa: E402
from session_distill import BOS, EOS, EOTC, PAD, SOTC, student_targets  # noqa: E402


def test_synthetic_record_and_targets():
    opener = [PAD, BOS, 5, EOS] + [PAD] * 6
    rec = G.synthetic_record(opener, cs=12, x_end=30, call_tokens=[501, 502, EOTC])
    st = student_targets(rec, 12, 30, 30, 80, False)
    assert st["text_lbl"][:4] == [PAD, BOS, 5, EOS]
    assert st["func_lbl"].index(SOTC) == 32
    assert [l for l, k in zip(st["func_lbl"], st["kind"]) if k == "fc"] == [501, 502, EOTC]
    assert st["func_in"][33] == SOTC and st["agent_in"][33] == PAD           # teacher forcing, no leakage
    assert all(x == PAD for x in st["text_lbl"][12:32])


def test_call_metrics():
    m = G.call_metrics([1, 2, 3], [1, 2, 3])
    assert m["exact_call"] and m["call_tok_acc"] == 1.0
    m = G.call_metrics([1, 9, 3], [1, 2, 3])
    assert not m["exact_call"] and m["first_error"] == 1
