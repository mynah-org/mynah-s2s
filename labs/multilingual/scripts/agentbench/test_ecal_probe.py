"""CPU tests for ecal_probe pure helpers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecal_probe as E  # noqa: E402


def test_parse_and_provenance():
    s = ' "arguments": {"full_name": "Mario Rossi", "postcode": "20135"}}]'
    slots = E.parse_args_text(s)
    assert slots == {"full_name": "Mario Rossi", "postcode": "20135"}
    p = E.provenance(slots, E.CALLER[("m07", "en")])
    assert p == {"full_name": "grounded", "postcode": "grounded"}
    p = E.provenance({"full_name": "Michael Moore", "postcode": "2135"}, E.CALLER[("m07", "en")])
    assert p == {"full_name": "ungrounded", "postcode": "ungrounded"}   # 2135 is not a digit run of 20135
    p = E.provenance({"query": "divano a tre posti"}, E.CALLER[("m03", "it")])
    assert p["query"] == "grounded"
    assert E.provenance({"query": "dining table"}, E.CALLER[("m03", "it")])["query"] == "ungrounded"


def test_pause_frames_and_listening():
    w = torch.ones(1280 * 20) * 0.5
    w[1280 * 10:1280 * 12] = 0.0
    assert E.pause_frames(w, 2, 19) == [10, 11]
    rows = [{"text_BOS": -5.0, "text_argmax": 12} for _ in range(20)]
    rows[11] = {"text_BOS": -0.4, "text_argmax": 1}
    r = E.listening(rows, 2, 19, [10, 11])
    assert r["max_bos_frame"] == 11 and r["bos_argmax_frames"] == [11] and r["pause_bos_logp_max"] == -0.4
