"""CPU tests for the free-running G-LEX scorer (glex_free.py pure parts)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glex_free as G  # noqa: E402
from session_distill import BOS, EOTC, PAD, SOTC, rule_call_text  # noqa: E402

T_FC = {"name": "find_customer", "arguments": {"full_name": "Bruno Blanc", "postcode": "27609"}}
T_CAT = {"name": "search_catalog", "arguments": {"query": "blue picture frame"}}


def test_exact_canonical():
    s = G.score_call(rule_call_text(T_FC), T_FC)
    assert s["exact"] and s["exact_text"] and s["slots"]["postcode"]["strict"]


def test_key_order_and_spacing_still_exact_but_not_text():
    txt = '<TOOLCALL>[{"arguments": {"postcode": "27609", "full_name": "Bruno Blanc"}, "name": "find_customer"}]</TOOLCALL>'
    s = G.score_call(txt, T_FC)
    assert s["exact"] and not s["exact_text"]


def test_wrong_slot_and_normalised_match():
    txt = rule_call_text({**T_FC, "arguments": {"full_name": "bruno  blanc", "postcode": "27608"}})
    s = G.score_call(txt, T_FC)
    assert not s["exact"]
    assert not s["slots"]["full_name"]["strict"] and s["slots"]["full_name"]["norm"]
    assert not s["slots"]["postcode"]["strict"] and not s["slots"]["postcode"]["norm"]


def test_wrong_tool_extra_key_two_calls_unparsable():
    assert G.score_call(rule_call_text({"name": "search_catalog", "arguments": T_FC["arguments"]}), T_FC)["tool"] == "search_catalog"
    s = G.score_call(rule_call_text({**T_CAT, "arguments": {"query": "blue picture frame", "limit": 3}}), T_CAT)
    assert not s["exact"] and s["extra_keys"] == ["limit"]
    two = "<TOOLCALL>" + json.dumps([T_CAT, T_CAT]) + "</TOOLCALL>"
    assert G.score_call(two, T_CAT)["n_calls"] == 2 and not G.score_call(two, T_CAT)["exact"]
    bad = G.score_call('<TOOLCALL>[{"name": "search_catalog", "arguments": {"query": "blue</TOOLCALL>', T_CAT)
    assert not bad["parsed"] and not bad["exact"] and bad["slots"]["query"]["got"] is None
    assert not G.score_call("", T_CAT)["exact"]


def test_behaviour_opener_listen_timing():
    opener = [PAD, 5, 6, PAD, 7]
    cs, x_end = 6, 20
    tx = [PAD, 5, 6, 7, PAD, PAD] + [PAD] * 4 + [BOS] + [PAD] * 3 + [9] + [PAD] * 7
    fn = [PAD] * 22 + [SOTC]
    b = G.behaviour({"main_text": tx, "main_func": fn, "fc": [3, 4, EOTC]}, cs, x_end, opener)
    assert b["opener_exact"] and b["bos_in_turn"] == 1 and b["talk_in_turn"] == 1
    assert b["sotc_frame"] == 22 and b["sotc_rel_end"] == 2 and not b["sotc_early"] and b["fc_terminated"]
    b2 = G.behaviour({"main_text": tx, "main_func": [PAD] * 10 + [SOTC], "fc": [3]}, cs, x_end, opener)
    assert b2["sotc_early"] and not b2["fc_terminated"]
    b3 = G.behaviour({"main_text": [PAD, 5, 8] + tx[3:], "main_func": fn, "fc": []}, cs, x_end, opener)
    assert not b3["opener_exact"]


def test_summarise_counts():
    rows = []
    for i, (lang, ok) in enumerate([("it", True), ("it", False), ("fr", False)]):
        call = rule_call_text(T_CAT) if ok else rule_call_text({**T_CAT, "arguments": {"query": "red chair"}})
        rows.append({"set": "glex_dev", "lang": lang, "target": T_CAT, "score": G.score_call(call, T_CAT),
                     "beh": {"sotc_frame": 30, "sotc_early": False, "opener_exact": True, "bos_in_turn": 0}})
    s = G.summarise(rows)
    assert s["all|all"]["exact"] == "1/3" and s["glex_dev|it"]["exact"] == "1/2" and s["glex_dev|fr"]["slots_strict"]["query"] == "0/1"


def test_build_sets_filters_and_caps():
    gplan = [{"id": "g1", "lang": "it", "target": T_CAT}, {"id": "g2", "lang": "it", "target": T_CAT}]
    gman = {("g1", "it"): {"path": "a.wav"}}
    vplan = [{"id": f"v{i}", "lang": "fr", "target": T_FC} for i in range(5)] + [{"id": "v9", "lang": "fr", "target": None}]
    vman = {(f"v{i}", "fr"): {"path": f"{i}.wav", "split": "val" if i < 3 else "train", "asr_ok": i != 1} for i in range(5)}
    vman[("v9", "fr")] = {"path": "9.wav", "split": "val"}
    vte = {f"v{i}": {"keep": True} for i in range(10)}
    rows = G.build_sets(gplan, gman, vplan, vman, vte, per_lang_val=12, per_lang_train=1)
    by = {(r["set"], r["id"]) for r in rows}
    assert by == {("glex_dev", "g1"), ("val", "v0"), ("val", "v2"), ("trainfit", "v3")}


def _row(arm, lang, call, sotc=40, opener=True, set_="glex_dev", id_="g1", target=T_CAT):
    return {"arm": arm, "set": set_, "id": id_, "lang": lang, "target": target, "call": call,
            "score": G.score_call(call, target),
            "beh": {"sotc_frame": sotc, "sotc_before_cs": False, "opener_exact": opener, "bos_in_turn": 0,
                    "sotc_early": False}}


def test_harness_gate_and_pairing():
    import glex_free_report as R
    ok, bad = rule_call_text(T_CAT), rule_call_text({**T_CAT, "arguments": {"query": "red chair"}})
    rows = [_row("vc", "en", ok, id_=f"g{i}") for i in range(20)] + [_row("vc", "en", "", sotc=None, id_=f"n{i}") for i in range(4)]
    g = R.harness_gate(rows)
    assert g["right_tool"] == 20 and g["pass"]
    rows2 = [_row("vc", "en", "", sotc=None, id_=f"n{i}") for i in range(8)] + rows[:16]
    assert not R.harness_gate(rows2)["checks"]["right_tool"]
    a = [_row("L", "it", ok, id_="x"), _row("L", "it", bad, id_="y", sotc=50)]
    b = [_row("F", "it", bad, id_="x"), _row("F", "it", bad, id_="y", sotc=52)]
    p = R.paired(a, b)
    assert p["n"] == 2 and p["exact_a_only"] == ["glex_dev|x|it"] and p["same_call_text"] == 1
    assert p["sotc_shift"] == [0, 2] and p["exact_a"] == 1 and p["exact_b"] == 0


def test_opener_valid_accepts_only_registered_variant():
    opener = [PAD, 5, 6, PAD, 7]
    cs, x_end = 6, 20
    base = [PAD] * 15
    alt = [PAD, 5, 9, 7, PAD, PAD] + base
    other = [PAD, 5, 8, 7, PAD, PAD] + base
    fn = [PAD] * 22
    b = G.behaviour({"main_text": alt, "main_func": fn, "fc": []}, cs, x_end, opener, [[5, 9, 7]])
    assert not b["opener_exact"] and b["opener_valid"]
    b2 = G.behaviour({"main_text": other, "main_func": fn, "fc": []}, cs, x_end, opener, [[5, 9, 7]])
    assert not b2["opener_valid"]


def test_concept_preserved_diagnostic():
    import lexicon_v2 as L
    c = next(x for x in L.CONCEPTS if x[0] == "side table")
    row = {"target": {"name": "search_catalog", "arguments": {"query": "side table"}}, "slots": {"concept": "side table"}}
    assert G.concept_preserved("side table", row) and G.concept_preserved(c[2][0], row)
    assert G.concept_preserved("small " + c[4][0], row)
    assert not G.concept_preserved("item", row) and not G.concept_preserved(None, row)
    assert G.concept_preserved("x", {"target": {"name": "find_customer"}}) is None
