"""capability_v1 scoring: instruction format, forbidden tool, missing args (ask, not invent), change of intent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from score_corpus import score_item  # noqa: E402


def S(text, calls=()):
    return {"text": text, "tool_calls": list(calls)}


def test_instruction_one_word_ok_and_too_long_fails():
    it = {"id": "c01", "category": "instruction", "expect": {"answer_regex": r"\bblue\b", "max_words": 3, "no_tool_call": True}}
    assert score_item(it, S("<s>Hi! How can I help?</s><s>Blue.</s>"), None)["ok_model"]
    r = score_item(it, S("<s>Hi!</s><s>The sky is blue because of Rayleigh scattering.</s>"), None)
    assert r["answer_model"] and not r["format_ok"] and not r["ok_model"]


def test_forbidden_tool_called_fails():
    it = {"id": "c04", "category": "tool_forbidden", "expect": {"no_tool_call": True}}
    assert score_item(it, S("<s>Hi</s><s>Usually three to five business days.</s>"), None)["ok_model"]
    bad = S("<s>Hi</s><s>Let me check that.</s>", [{"name": "get_order_status", "args": {"order_id": "X"}}])
    assert not score_item(it, bad, None)["ok_model"]


def test_missing_args_must_ask_not_invent():
    it = {"id": "c06", "category": "tool_missing_args",
          "expect": {"no_tool_call": True, "ask_regex": r"order (id|number|code)|which order"}}
    assert score_item(it, S("<s>Hi</s><s>Sure, what is your order number?</s>"), None)["ok_model"]
    assert not score_item(it, S("<s>Hi</s><s>Sure, one moment.</s>"), None)["ok_model"]
    inv = S("<s>Hi</s><s>What is your order number?</s>", [{"name": "get_order_status", "args": {"order_id": "123"}}])
    assert not score_item(it, inv, None)["ok_model"]


def test_change_of_intent_forbids_the_abandoned_tool():
    it = {"id": "c09", "category": "tool_random",
          "expect": {"function": "generate_random_number", "min": 1, "max": 5, "forbid_function": "get_order_status"}}
    good = S("<s>Hi</s><s>Three.</s>", [{"name": "generate_random_number", "args": {"min": 1, "max": 5}}])
    assert score_item(it, good, None)["ok_model"]
    both = S("<s>Hi</s><s>Three.</s>", [{"name": "get_order_status", "args": {"order_id": "A"}},
                                       {"name": "generate_random_number", "args": {"min": 1, "max": 5}}])
    r = score_item(it, both, None)
    assert r["forbidden_called"] and not r["ok_model"]


def test_must_not_regex_catches_invented_data():
    it = {"id": "c07", "category": "tool_orders",
          "expect": {"function": "get_order_status", "order_id": "QQQ999", "found": False,
                     "must_not_regex": r"\b(shipped|delivered)\b|november"}}
    call = [{"name": "get_order_status", "args": {"order_id": "QQQ999"}}]
    assert score_item(it, S("<s>Hi</s><s>I could not find that order.</s>", call), None)["ok_model"]
    r = score_item(it, S("<s>Hi</s><s>I could not find it, but it shipped on November 3.</s>", call), None)
    assert r["must_not_hit"] and not r["ok_model"]


def test_outcome_classes():
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from screen_compare import outcome
    assert outcome(True, True) == "pass" and outcome(False, True) == "fixed"
    assert outcome(True, False) == "regression" and outcome(False, False) == "intrinsic_fail"
