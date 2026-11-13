"""CPU tests for the V0 user-text injection policy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v0_user_text as V  # noqa: E402


def test_word_events_completion_frames_and_tags():
    pieces = [(10, "▁<it-IT>"), (10, "▁ci"), (11, "ao"), (14, "▁sono"), (20, "▁Kev"), (21, "in")]
    assert V.word_events(pieces) == [(14, "ciao"), (17, "sono"), (24, "Kevin")]


def test_frame_tokens_fifo_one_per_frame_and_filter():
    enc = {" ciao": [5000, 5001], " sono": [12, 6000], " Kevin": [7000]}.get
    got = V.frame_tokens([(14, "ciao"), (15, "sono"), (30, "Kevin")], enc)
    assert got == [(14, 5000), (15, 5001), (16, 6000), (30, 7000)]       # id 12 (PAD) filtered, queue delays 'sono'


def test_no_tokens_before_completion():
    got = V.frame_tokens([(40, "x")], lambda s: [1500, 1501])
    assert got[0][0] == 40 and got[1][0] == 41


def test_bare_space_piece_is_a_boundary():
    pieces = [(5, "▁got"), (6, "▁"), (6, "a"), (7, "▁"), (8, "wood"), (8, "en")]
    assert V.word_events(pieces) == [(6, "got"), (8, "a"), (11, "wooden")]


def test_late_punctuation_does_not_delay_the_last_word():
    pieces = [(30, "▁disponibile"), (90, ".")]
    assert V.word_events(pieces) == [(33, "disponibile")]
