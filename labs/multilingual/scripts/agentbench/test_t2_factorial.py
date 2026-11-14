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
    assert set(F.SCEN) == {"m03_empty", "m08_vr", "m07_chain", "m02_nf", "t14_nf", "t13_found"}
    assert {k: v.get("live_outcome") for k, v in F.SCEN.items()} == {
        "m03_empty": None, "m08_vr": None, "m07_chain": "chain", "m02_nf": "identical_recall",
        "t14_nf": "identical_recall", "t13_found": "reply"}
    assert F.SCEN["t14_nf"]["post_fc_zeros"] and F.SCEN["t13_found"]["post_fc_zeros"]
    assert not F.SCEN["m02_nf"].get("post_fc_zeros")      # job 111 capture kept for the typed comparison
    m02 = F.SCEN["m02_nf"]
    assert m02["family"] == "engine" and m02["lead_s"] == 0.0 and m02["h5"][1] == "12/22"
    assert m02["expect_result"] == '{"order_id": "QGQ999", "found": false}' and m02["prompt_chars"] == 3366
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
    assert [F.prompt_variant(v) for v in F.VARIANTS] == ["orig", "p1", "orig", "p1", "orig", "orig", "app", "appnt",
                                                         "orig"]
    assert [F.result_variant(v) for v in F.VARIANTS] == ["orig", "orig", "p3", "p3", "orig", "typed", "orig", "orig",
                                                         "orig"]
    assert F.variant_prompt(base, "app").endswith(F.APP_CONTRACT)
    assert F.variant_prompt(base, "appnt").endswith(F.APP_CONTRACT + " /no_think")
    assert "at most one tool call at a time" in F.APP_CONTRACT and "Never invent" in F.APP_CONTRACT
    assert "do not retry the tool call for the same request" in F.P1_RULE and F.SOTC_BIAS < 0
    assert F.parse_plan("m08_vr:A:p1,m07_chain:A", [], [], []) == [("m08_vr", "A", "p1"), ("m07_chain", "A", "base")]
    assert F.parse_plan("", ["m08_vr", "m07_chain"], ["A"], ["base", "p3"]) == [
        ("m08_vr", "A", "base"), ("m08_vr", "A", "p3"), ("m07_chain", "A", "base"), ("m07_chain", "A", "p3")]


def _wrapper_like_conv(obj):
    """Mimics the wrapper's _convert_nums_in_json leaf rule: bool is an int, inflect fails on it -> str()."""
    if isinstance(obj, dict):
        return {k: _wrapper_like_conv(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wrapper_like_conv(v) for v in obj]
    if isinstance(obj, bool):
        return str(obj)
    if isinstance(obj, int):
        return {7319: "seven thousand three hundred and nineteen"}.get(obj, f"n{obj}")
    return obj


def test_typed_convert_keeps_bools_and_converts_numbers():
    import json
    import re
    w = '<TOOL_RESPONSE>[{"order_id": "QGQ999", "found": false}]</TOOL_RESPONSE>'
    m = re.match(r"(<TOOL_RESPONSE>)(.*?)(</TOOL_RESPONSE>)", w)
    bug = m.group(1) + json.dumps(_wrapper_like_conv(json.loads(m.group(2)))) + m.group(3)
    assert '"found": "False"' in bug and len(bug) == 73                          # = the live FX1 m02 response
    typed = F.typed_convert(w, _wrapper_like_conv)
    assert typed == '<TOOL_RESPONSE>[{"order_id": "QGQ999", "found": false}]</TOOL_RESPONSE>'
    assert F.typed_convert('<TOOL_RESPONSE>[{"code": 7319, "ok": true}]</TOOL_RESPONSE>', _wrapper_like_conv) == \
        '<TOOL_RESPONSE>[{"code": "seven thousand three hundred and nineteen", "ok": true}]</TOOL_RESPONSE>'


def _cell(first, text=None, at=None):
    return {"contracts": {"12/22": {"outcome": {"first": first, "new_call_text": text, "new_call_sotc": at}}}}


def test_gate_rules():
    call = '<TOOLCALL>[{"name": "get_order_status", "arguments": {"order_id": "qgq999"}}]'
    g = F.gate("m02_nf", _cell("new_call", call.replace(" ", "  "), 192), call)
    assert g["identical_recall"] and not g["variant_pass"] and g["recall_at"] == 192
    assert F.gate("m02_nf", _cell("reply"), call)["variant_pass"]
    other = F.gate("m02_nf", _cell("new_call", call.replace("qgq999", "qqq999"), 192), call)
    assert other["other_call"] and not other["variant_pass"]                   # pass needs a reply
    pr = '<TOOLCALL>[{"name": "prepare_return", "arguments": {"item": "Lysa bookcase", "customer_id": "C1002"}}]'
    assert F.gate("m07_chain", _cell("new_call", pr, 293), "x")["chain_kept"]
    assert not F.gate("m07_chain", _cell("reply"), "x")["chain_kept"]
    assert not F.gate("m07_chain", _cell("new_call", pr.replace("C1002", "Mario"), 293), "x")["chain_kept"]


def test_gate_reply_state():
    call = '<TOOLCALL>[{"name": "get_order_status", "arguments": {"order_id": "XYZ123"}}]'
    g = F.gate("t13_found", _cell("reply"), call)
    assert g["reply"] and g["variant_pass"]
    assert F.gate("t14_nf", _cell("new_call", call, 176), call)["identical_recall"]


def test_gpu_only_main_has_no_undefined_names():
    """main() runs only on the GPU: a static check catches a broken edit before a GPU job does (job 129: NameError)."""
    import pytest
    api = pytest.importorskip("pyflakes.api")
    from pyflakes import reporter
    import io
    out, err = io.StringIO(), io.StringIO()
    api.checkPath(str(Path(__file__).resolve().parent / "t2_factorial.py"), reporter.Reporter(out, err))
    bad = [l for l in out.getvalue().splitlines() if "undefined name" in l or "referenced before assignment" in l]
    assert not bad, bad
