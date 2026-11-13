import pytest

import union_manifest as U


def test_union_refreshes_targets_drops_probes_and_excludes():
    plan = {("a", "en"): {"text": "x", "target": {"name": "escalate_to_human", "arguments": {}}, "split": "train"},
            ("a", "fr"): {"text": "y", "target": {"name": "escalate_to_human", "arguments": {}}, "split": "train"}}
    part = [{"id": "a", "lang": "en", "text": "x", "target": {"name": "escalate_to_human", "arguments": None}},
            {"id": "a", "lang": "fr", "text": "y", "target": None},
            {"id": "gt4-0001", "lang": "it", "text": "z"}]
    rows, st = U.union([part], {"a|fr"}, "grammar_defect", plan)
    assert len(rows) == 2 and st["dropped_not_in_plan"] == 1 and st["target_refreshed"] == 2
    assert all(r["target"]["arguments"] == {} for r in rows)
    assert [r.get("excluded") for r in rows] == [None, "grammar_defect"]


def test_union_rejects_text_mismatch_and_duplicates():
    plan = {("a", "en"): {"text": "x"}}
    with pytest.raises(SystemExit):
        U.union([[{"id": "a", "lang": "en", "text": "other"}]], set(), "r", plan)
    with pytest.raises(SystemExit):
        U.union([[{"id": "a", "lang": "en", "text": "x"}], [{"id": "a", "lang": "en", "text": "x"}]], set(), "r", plan)
