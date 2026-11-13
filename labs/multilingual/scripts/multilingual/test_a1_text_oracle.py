"""CPU tests for the A1 text-oracle helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_text_oracle as A  # noqa: E402


def test_step_inputs_forced_then_own():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    gen = [[9], [0]]
    assert A.step_inputs(seqs, gen, 0, 2) == [3, 3]
    assert A.step_inputs(seqs, gen, 1, 2) == [4, 4]
    assert A.step_inputs(seqs, gen, 2, 2) == [9, 5]


def test_build_items_conditions():
    rows = [{"set": "glex_dev", "id": "g1", "lang": "it", "data": "glex", "target": {}},
            {"set": "glex_dev", "id": "g1", "lang": "en", "data": "glex", "target": {}}]
    text = {("glex", "g1", "it"): "ciao", ("glex", "g1", "en"): "hi"}
    a0 = {("glex_dev", "g1", "it"): "cia"}
    conds = [(i[1], i[2]["lang"]) for i in A.build_items(rows, text, a0)]
    assert conds == [("gold_own", "it"), ("gold_en", "it"), ("rnnt", "it"), ("gold_own", "en")]


def test_query_kind():
    row = {"target": {"name": "search_catalog", "arguments": {"query": "side table"}}, "lang": "it",
           "slots": {"concept": "side table"}}
    import lexicon_v2 as L
    c = next(x for x in L.CONCEPTS if x[0] == "side table")
    assert A.query_kind("side table", row) == "canonical"
    assert A.query_kind(c[2][0], row) == "source_language"
    assert A.query_kind("large side table", row) == "concept_en_other_attrs"
    assert A.query_kind("bookshelf", row) == "other"
