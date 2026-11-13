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
