"""The endpoint bench's pure parts: where speech is in a clip, and the per-language summary."""

import numpy as np

from frankenstein2.lab.endpoint_bench import hold_effect, mid_gap, speech_span, summarize


def test_speech_span_finds_the_voiced_part():
    x = np.zeros(16000, np.float32)
    x[4000:12000] = 0.3 * np.sin(np.arange(8000) / 5.0)
    a, b = speech_span(x)
    assert abs(a - 4000) <= 320 and abs(b - 12000) <= 320
    assert speech_span(np.zeros(16000, np.float32)) is None


def test_summary_counts_latency_misses_and_premature_endpoints():
    rows = [{"lang": "de", "kind": "single", "latency_ms": 400}, {"lang": "de", "kind": "single", "latency_ms": 800},
            {"lang": "de", "kind": "single", "latency_ms": None},
            {"lang": "de", "kind": "pause_0.6", "premature": True, "voiced_before": False, "rescued": True},
            {"lang": "de", "kind": "pause_0.6", "premature": False, "voiced_before": False}]
    d = summarize(rows)["de"]
    assert d["lines"] == 3 and d["missed"] == 1 and d["latency_ms_p50"] == 600 and d["latency_ms_max"] == 800
    assert d["pause_0.6"] == {"n": 2, "premature": 1, "veto_would_hold": 0, "premature_after_hold": 0}


def test_mid_gap_cuts_at_the_quiet_point_inside_the_speech():
    x = np.zeros(32000, np.float32)
    t = np.arange(8000) / 5.0
    x[4000:12000] = 0.3 * np.sin(t)                 # word 1
    x[12800:20800] = 0.3 * np.sin(t)                # word 2, a 50 ms gap before it
    c = mid_gap(x)
    assert 12000 <= c <= 12800


def test_hold_effect_rescues_a_cut_after_a_conjunction():
    pcs = [(4, "▁vorrei"), (5, "▁un"), (6, "▁divano"), (7, "▁e"), (16, "▁una"), (17, "▁poltrona")]
    assert hold_effect([9], pcs, "it", hold=10) == [(True, True)]        # "...e": held, words at 16 <= 19
    assert hold_effect([9], pcs, "it", hold=5) == [(True, False)]        # held, but the words come too late
    assert hold_effect([18], pcs, "it", hold=10) == [(False, False)]     # "...poltrona": a finished turn
