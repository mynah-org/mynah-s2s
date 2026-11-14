"""CPU tests for t2_factorial pure helpers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_factorial as F  # noqa: E402
from t1_probe import BOS, PAD, SOTC  # noqa: E402


def test_first_free_matches_live_traces():
    # m03: s=133, 32 call tokens, 63 response tokens -> 230; m08: 147, 32, 44 -> 225; m07: 175, 42, 74 -> 293
    assert F.first_free(133, [0] * 32, [0] * 63) == 230
    assert F.first_free(147, [0] * 32, [0] * 44) == 225
    assert F.first_free(175, [0] * 42, [0] * 74) == 293


def test_parse_pair():
    assert F.parse_pair("12/22s") == (12, 22, True)
    assert F.parse_pair("0/0") == (0, 0, False)


def test_trace_lengths_first_fc_only():
    sess = [{"ev": "function.arguments.done", "steps": 32}, {"ev": "tool.result.injected", "steps": 45, "chars": 145},
            {"ev": "tool.result.consumed", "model_step": 224}, {"ev": "function.arguments.done", "steps": 41},
            {"ev": "tool.result.injected", "steps": 9, "chars": 1}]
    assert F.trace_lengths(sess) == {"call_steps": 32, "result_steps": 45, "result_chars": 145, "consumed": 224}


def test_summarise_and_margin():
    rows = [{"pos": 225, "text": PAD, "func": PAD, "sotc_margin": -1.0},
            {"pos": 226, "text": PAD, "func": SOTC, "sotc_margin": 0.5}]
    o = F.summarise(rows)
    assert o["first"] == "new_call" and o["new_call_sotc"] == 226 and o["max_sotc_margin"] == 0.5
    rows = [{"pos": 230, "text": BOS, "func": PAD, "sotc_margin": -3.0}, {"pos": 231, "text": 7, "func": PAD,
                                                                           "sotc_margin": -9.0}]
    o = F.summarise(rows)
    assert o["first"] == "reply" and o["reply_ids"] == [7]
    lp = torch.log_softmax(torch.tensor([0.0, 1.0, 3.0]), 0)
    assert F.margin(lp, 2) == 2.0


def test_scenarios_declared():
    assert set(F.SCEN) == {"m03_empty", "m08_vr", "m07_chain"}
    assert F.SCEN["m08_vr"]["h5"][1] == "12/22" and F.SCEN["m07_chain"]["h5"][1] == "0/0"
    assert len(F.CONTRACTS.split(",")) == 6


# ---------------------------------------------------------------- P1 / P3 variants (owner 2026-11-14)
def _tenant_results():
    import json
    from tenant_backend import TenantBackend
    spec = json.load(open(Path(__file__).resolve().parent / "tenant_furniture.json"))
    be = TenantBackend(spec)
    h = be.handlers()
    return {"m03_empty": h["search_catalog"]({"query": "zzz qqq"}),
            "m08_vr": TenantBackend(spec).handlers()["find_customer"]({"full_name": "Lindvik"}),
            "m07_chain": TenantBackend(spec).handlers()["find_customer"]({"full_name": "Mario Rossi", "postcode": "20135"}),
            "m02_nf": '{"order_id": "QGQ999", "found": false}'}


def test_structure_result_minimal_status_strings_only():
    import json
    res = _tenant_results()
    for sc, st in (("m03_empty", "no_results"), ("m02_nf", "not_found")):
        orig, new = json.loads(res[sc]), json.loads(F.structure_result(res[sc]))
        assert {k: new[k] for k in orig} == orig and new["status"] == st          # every original field kept
        added = {k: v for k, v in new.items() if k not in orig}
        assert all(isinstance(v, str) and not any(ch.isdigit() for ch in v) for v in added.values())
    assert F.structure_result(res["m08_vr"]) == res["m08_vr"]            # already has a string status
    assert F.structure_result(res["m07_chain"]) == res["m07_chain"]      # productive: byte-identical
    assert F.structure_result("not json") == "not json"


def test_variant_prompt_and_plan():
    base = "You are the voice assistant."
    assert F.variant_prompt(base, "base") == base and F.variant_prompt(base, "p3") == base
    assert F.variant_prompt(base, "p1").endswith(F.P1_RULE) and F.variant_prompt(base, "p1p3") == F.variant_prompt(base, "p1")
    assert [F.prompt_variant(v) for v in F.VARIANTS] == ["orig", "p1", "orig", "p1", "orig"]
    assert [F.result_variant(v) for v in F.VARIANTS] == ["orig", "orig", "p3", "p3", "orig"]
    assert "do not retry the tool call for the same request" in F.P1_RULE and F.SOTC_BIAS < 0
    assert F.parse_plan("m08_vr:A:p1,m07_chain:A", [], [], []) == [("m08_vr", "A", "p1"), ("m07_chain", "A", "base")]
    assert F.parse_plan("", ["m08_vr", "m07_chain"], ["A"], ["base", "p3"]) == [
        ("m08_vr", "A", "base"), ("m08_vr", "A", "p3"), ("m07_chain", "A", "base"), ("m07_chain", "A", "p3")]
