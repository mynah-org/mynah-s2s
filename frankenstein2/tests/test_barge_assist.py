"""Barge-in assist: echo-aware onset + ASR confirmation, backchannels and echo do not barge."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.control.barge import BargeAssist  # noqa: E402
from frankenstein2.tests.fakes import pieces  # noqa: E402


def lvl(db):
    return np.full(1280, 10 ** (db / 20), np.float32)


def drive(b, mic_db, agent_db, words=None, n=20):
    fired = None
    for k in range(n):
        c = b.step(k, lvl(mic_db(k)), lvl(agent_db(k)), True)
        ps = [p for p in (words or []) if p["f"] <= k - 4]       # the ASR lags ~4 frames
        why = b.decide(c, [p for p in ps if c and p["f"] >= c["onset"] - 1]) if c else None
        if why and fired is None:
            fired = (k, why, c["onset"])
    return fired


def test_clean_mic_cut_in_fires_on_first_real_word():
    b = BargeAssist()
    f = drive(b, lambda k: -20 if k >= 5 else -80, lambda k: -20, pieces(6, "aspetti un attimo"))
    assert f is not None and f[1] == "asr_word" and f[2] == 5 and f[0] <= 11


def test_backchannel_does_not_fire():
    b = BargeAssist()
    assert drive(b, lambda k: -20 if 5 <= k < 10 else -80, lambda k: -20, pieces(6, "sì mh")) is None


def test_echo_alone_does_not_fire_once_coupling_is_learned():
    b = BargeAssist()
    # the mic carries the agent at -15 dB coupling (no AEC): louder than the prior allows until it is learned
    assert drive(b, lambda k: -35, lambda k: -20, None, n=60) is None
    assert -16 <= b.coupling <= -14


def test_long_speech_without_words_fires():
    b = BargeAssist()
    f = drive(b, lambda k: -20 if k >= 25 else -80, lambda k: -20, None, n=45)
    assert f is not None and f[1] == "long_speech"


def test_echo_words_of_the_agent_text_do_not_confirm():
    b = BargeAssist()
    c = {"onset": 0, "frames": 4}
    assert b.decide(c, pieces(0, "il suo ordine"), agent_text="Il suo ordine è partito.") is None
    assert b.decide(c, pieces(0, "aspetti"), agent_text="Il suo ordine è partito.") == "asr_word"
