"""CPU tests for the A1b forced-call schedule and items."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1b_forced_call as A  # noqa: E402
from session_distill import SOTC  # noqa: E402


def test_schedule_context_sotc_prefix_then_own():
    pre = [101, 102]
    assert A.schedule(3, pre, 0, [7]) == ("text", 0)
    assert A.schedule(3, pre, 2, [7]) == ("text", 2)
    assert A.schedule(3, pre, 3, [7]) == ("fc", SOTC)
    assert A.schedule(3, pre, 4, [7]) == ("fc", 101)
    assert A.schedule(3, pre, 5, [7]) == ("fc", 102)
    assert A.schedule(3, pre, 6, [7, 8]) == ("fc", 7)
    assert A.schedule(3, pre, 7, [7, 8]) == ("fc", 8)


def test_items_gold_en_once_per_family():
    rows = [{"set": "glex_dev", "id": "g1", "lang": l, "data": "glex", "target": {}} for l in ("en", "it", "fr")]
    text = {("glex", "g1", l): l for l in ("en", "it", "fr")}
    a0 = {("glex_dev", "g1", "it"): "x", ("glex_dev", "g1", "fr"): "y"}
    got = [(i[1], i[2]["lang"]) for i in A.items_for(rows, text, a0)]
    assert got == [("gold_en", "en"), ("gold_own", "it"), ("rnnt", "it"), ("gold_own", "fr"), ("rnnt", "fr")]


def test_concept_ok():
    row = {"target": {"name": "search_catalog"}, "slots": {"concept": "picture frame"}}
    assert A.concept_ok("blue picture frame", row) and not A.concept_ok("blue couch", row)
    assert A.concept_ok(None, {"target": {"name": "find_customer"}}) is None


def test_schedule_finished_row_does_not_overrun():
    assert A.schedule(3, [101], 9, [7, 21]) == ("fc", 21)
