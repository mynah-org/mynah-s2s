"""Turn rules (Tyger port) on synthetic head-output streams."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.ears.rules import RulesState, Thresholds, energy_db  # noqa: E402

SPEECH, SIL = -20.0, -70.0


def out(p_end=0.0, yld=0.0, bc=0.0, haz=0.0):
    return [1.0, 0.0, 0.0, 0.0, haz, yld, bc, p_end]


def run(r, rows):
    evs = []
    for o, db in rows:
        e, reset = r.on_frame(o, db)
        evs += [x for x in e if x["kind"] != "FRAME"]
        if reset:
            evs.append({"kind": "RESET", "k": r.frame - 1})
    return evs


def test_endpoint_after_speech_two_frames():
    r = RulesState()
    rows = [(out(), SPEECH)] * 10 + [(out(0.95), SIL)] * 3
    evs = run(r, rows)
    eps = [e for e in evs if e["kind"] == "ENDPOINT"]
    assert len(eps) == 1 and eps[0]["k"] == 11          # second frame over the threshold
    assert any(e["kind"] == "RESET" for e in evs)


def test_no_endpoint_without_enough_speech_or_in_warmup():
    r = RulesState()
    assert not run(r, [(out(0.99), SIL)] * 20)            # silence only: nobody spoke
    r = RulesState()
    rows = [(out(0.99), SPEECH)] * 4 + [(out(0.99), SIL)] * 2   # 320 ms of speech < 350 ms, inside warm-up
    assert not [e for e in run(r, rows) if e["kind"] == "ENDPOINT"]


def test_threshold_is_configurable():
    rows = [(out(), SPEECH)] * 10 + [(out(0.87), SIL)] * 4
    assert not [e for e in run(RulesState(), rows) if e["kind"] == "ENDPOINT"]
    assert [e for e in run(RulesState(thr=Thresholds(ep_thr=0.85)), rows) if e["kind"] == "ENDPOINT"]


def test_barge_in_speaking_with_onset():
    r = RulesState()
    assert r.switch_mode("speaking")
    rows = [(out(), SIL)] * 6 + [(out(yld=0.1), SPEECH)] * 5 + [(out(yld=0.6), SPEECH)] * 2
    evs = run(r, rows)
    b = [e for e in evs if e["kind"] == "BARGE"]
    assert len(b) == 1 and b[0]["k"] == 12
    assert abs(b[0]["onset_s"] - 7 * 0.08) < 1e-6         # 7 voiced frames by then
    assert not [e for e in evs if e["kind"] == "ENDPOINT"]  # the endpoint head is not read while speaking


def test_barge_needs_warmup_after_mode_switch():
    r = RulesState()
    r.switch_mode("speaking")
    evs = run(r, [(out(yld=0.9), SPEECH)] * 6)
    assert not evs


def test_backchannel_refractory():
    r = RulesState()
    rows = [(out(), SPEECH)] * 8 + [(out(bc=0.5), SPEECH)] * 30
    bcs = [e for e in run(r, rows) if e["kind"] == "BACKCHANNEL"]
    assert len(bcs) >= 2
    assert bcs[1]["k"] - bcs[0]["k"] >= 15 + 2


def test_gru_prev_state_forced_by_mode():
    r = RulesState()
    assert r.gru_prev_state() == 0
    r.switch_mode("speaking")
    assert r.gru_prev_state() == 2
    assert not r.switch_mode("speaking")


def test_energy_db():
    assert energy_db([0.0] * 1280) < -100
    assert -21 < energy_db([0.1] * 1280) < -19


def test_silence_fallback_ends_a_turn_when_p_end_plateaus():
    r = RulesState()
    rows = [(out(), SPEECH)] * 10 + [(out(0.89), SIL)] * 20
    eps = [e for e in run(r, rows) if e["kind"] == "ENDPOINT"]
    assert len(eps) == 1 and eps[0]["by"] == "silence" and eps[0]["k"] == 10 + 15 - 1
    r = RulesState(thr=Thresholds(fallback_sil_frames=0))
    assert not [e for e in run(r, rows) if e["kind"] == "ENDPOINT"]


def test_noise_floor_raises_the_voiced_threshold_on_a_noisy_line():
    r = RulesState()
    for _ in range(800):
        r.on_frame(out(), -40.0)            # a steady -40 dB line noise for 64 s
    assert r.floor_db > -50 and not r.voiced(-40.0) and r.voiced(-20.0)
