"""Reconstruction of the FC sub-timeline from reference fast-path records, with the geometry observed on the NVIDIA
FC sample (live path, GB10 2026-11-10, results/gpu-tests/fc_sample_records.json): SOTC predicted at frame 96 by the
regular step, fast path entered with current_frame_idx=97; call tokens 97-132, EOTC 133 (phase 1: 36 steps), 36 forced
response tokens 134-169, EOTR predicted at 170 (phase 2: 37 steps, current_frame_idx=133)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from mynah_v2.model import tokens as T  # noqa: E402
from mynah_v2.model.reference import FCRecord  # noqa: E402


def _records():
    P = T.PAD_ID
    gt = [P] * 200
    gf = [P] * 200
    gf[96] = T.SOTC_ID
    for f in range(97, 133):
        gf[f] = 1000 + f
    gf[133] = T.EOTC_ID
    for f in range(134, 170):
        gf[f] = 2000 + f
    gf[170] = T.EOTR_ID
    p1 = FCRecord(1, 97, 36, gt[97:97 + 37], gf[97:97 + 37], None, 0, 1,       # recorder slice [start, start+steps]
                  pre_text=gt[96], pre_func=gf[96])
    p2 = FCRecord(2, 133, 37, gt[133:133 + 38], gf[133:133 + 38], "<TOOL_RESPONSE>[]</TOOL_RESPONSE>", 1, 2)
    return [p1, p2]


def test_reconstruction_matches_observed_geometry():
    from run_stream import fc_timeline_from_records
    (c,) = fc_timeline_from_records(_records())["calls"]
    assert c["violations"] == []
    (span,) = c["spans"]
    assert len(span["call_tokens"]) == 36
    # relative steps inside the reconstructed window: SOTC 0 (frame 96), EOTC 37, response 38..73, EOTR 74
    assert (span["sotc_step"], span["eotc_step"], span["resp_start"], span["resp_end"], span["eotr_step"]) == (0, 37, 38, 73, 74)


def test_record_without_pre_token_is_flagged():
    """Regression: the first recorder dropped the SOTC frame -> checker must report it, not pass silently."""
    from run_stream import fc_timeline_from_records
    p1, p2 = _records()
    p1.pre_func = p1.pre_text = None
    (c,) = fc_timeline_from_records([p1, p2])["calls"]
    assert any("EOTC outside a call" in v for v in c["violations"])


def test_free_text_after_eotc_would_be_flagged():
    from run_stream import fc_timeline_from_records
    p1, p2 = _records()
    p2.text[1] = 777          # agent text on the first response frame
    (c,) = fc_timeline_from_records([p1, p2])["calls"]
    assert any("agent text 777" in v for v in c["violations"])


def test_reconstruction_when_sotc_is_the_first_fast_path_frame():
    """Second measured geometry (en_s2/en_s4, GB10 2026-11-10): SOTC on start_frame itself, PAD before it."""
    from run_stream import fc_timeline_from_records
    P = T.PAD_ID
    gt, gf = [P] * 200, [P] * 200
    gf[101] = T.SOTC_ID
    for f in range(102, 137):
        gf[f] = 1000 + f
    gf[137] = T.EOTC_ID
    for f in range(138, 170):
        gf[f] = 2000 + f
    gf[170] = T.EOTR_ID
    p1 = FCRecord(1, 101, 36, gt[101:138], gf[101:138], None, 0, 1, pre_text=gt[100], pre_func=gf[100])
    p2 = FCRecord(2, 137, 33, gt[137:171], gf[137:171], "<TOOL_RESPONSE>[]</TOOL_RESPONSE>", 1, 2)
    (c,) = fc_timeline_from_records([p1, p2])["calls"]
    assert c["violations"] == []
    (span,) = c["spans"]
    assert (span["sotc_step"], span["eotc_step"], span["eotr_step"]) == (0, 36, 69)
