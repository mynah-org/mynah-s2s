"""Session measurements on a synthetic run record."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.lab.metrics import analyse, cer, check_tools  # noqa: E402


def test_analyse_latencies_barge_and_false_barges():
    tl = [{"label": "U1", "start_k": 10, "end_k": 40}, {"label": "U2", "start_k": 100, "end_k": 130}]
    ev = [{"ev": "ears.event", "kind": "ENDPOINT", "model_step": 48},
          {"ev": "ctl.action", "kind": "OPEN_TURN", "model_step": 49, "turn_id": 1},
          {"ev": "ears.event", "kind": "BARGE", "model_step": 110, "onset_s": 0.8},
          {"ev": "audio.flush", "model_step": 111, "reason": "barge_in", "dropped_samples": 24000},
          {"ev": "ears.event", "kind": "BARGE", "model_step": 300},
          {"ev": "tool.completed", "name": "find_customer", "args": {"name": "Mario Rossi", "postcode": "10121"}}]
    segs = [{"epoch": 0, "turn": 1, "seg": 0, "lane": "agent", "holding": True, "first_step": 52, "released": 19200},
            {"epoch": 0, "turn": 1, "seg": 1, "lane": "agent", "holding": False, "first_step": 70, "released": 1920}]
    wall = [i * 0.08 for i in range(400)]
    m = analyse(tl, ev, segs, wall, 0.0, client_delay_steps=1,
                tools_expected=[{"name": "find_customer", "args": {"postcode": "10121"}}])
    u1 = m["utterances"][0]
    assert u1["endpoint_s"] == round(8 * 0.08, 3) and u1["first_holding_s"] == round(12 * 0.08, 3)
    assert u1["first_reply_s"] == round(30 * 0.08, 3)
    assert m["barge_ins"][0]["onset_to_stop_s"] == round(11 * 0.08, 3) and m["false_barges"] == [300]
    assert m["tools_check"]["ok"] == 1


def test_cer_and_tools():
    assert cer("Buongiorno, come sta?", "buongiorno come sta") == 0.0
    assert check_tools([{"name": "a", "args": {}}], [{"name": "a"}, {"name": "a"}])["ok"] == 1
