"""G0: the playback model of the real talker (e1/flash_client.Player) and the live-ASR post-processing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.flash_client import Player  # noqa: E402
from e1.live_asr import ASRPost  # noqa: E402

SR = 100          # 100 samples per second keeps the arithmetic readable


def test_prebuffer_back_to_back_and_underrun():
    p = Player(sr=SR, prebuffer_s=0.3)
    a = p.on_audio(1.0, 0, 1, 0, 0, 2, 16)           # 0.16 s
    assert a.t_start == 1.3
    b = p.on_audio(1.1, 0, 1, 0, 2, 6, 32)           # arrives while a is scheduled: back to back
    assert abs(b.t_start - 1.46) < 1e-9
    c = p.on_audio(2.0, 0, 1, 0, 6, 14, 64)          # scheduled audio ended at 1.78: underrun
    assert c.t_start == 2.0 and len(p.gaps) == 1 and abs(p.gaps[0]["gap_s"] - 0.22) < 1e-6
    d = p.on_audio(5.0, 0, 2, 0, 0, 2, 16)           # a new turn while idle: prebuffer again, not a gap
    assert d.t_start == 5.3 and len(p.gaps) == 1


def test_cancel_cuts_playback_and_drops_stale_audio():
    p = Player(sr=SR, prebuffer_s=0.0)
    a = p.on_audio(1.0, 0, 1, 0, 0, 25, 200)         # plays 1.0 .. 3.0
    b = p.on_audio(1.5, 0, 1, 0, 25, 50, 200)        # scheduled 3.0 .. 5.0
    p.on_cancel(2.0, 1)
    assert a.keep == 100 and b.dropped and p.end == 2.0 and p.cuts[0]["was_playing"]
    assert p.on_audio(2.1, 0, 1, 0, 50, 75, 200) is None and len(p.stale) == 1
    c = p.on_audio(2.5, 1, 2, 0, 0, 2, 16)
    assert c is not None and c.t_start == 2.5
    assert not p.audible(2.7) and p.audible(2.6)


def test_render_places_audio_on_the_session_clock():
    import numpy as np
    p = Player(sr=SR, prebuffer_s=0.0)
    p.on_audio(1.0, 0, 1, 0, 0, 1, 10, pcm=np.ones(10, np.float32))
    out = p.render(t0=0.5, total_s=2.0)
    assert out[49] == 0 and out[50] == 1 and out[59] == 1 and out[60] == 0


def test_asr_post_words_activity_and_eou():
    post = ASRPost(close_after=3, eou_frames=10, bou_frames=3)
    evs = [post.step(k, p) for k, p in enumerate([["▁cia"], ["o"], [], [], [], ["<it-IT>"]] + [[]] * 12)]
    words = [(e["k"], w) for e in evs for w in e["words"]]
    assert words and words[0][1] == "ciao"
    assert evs[0]["active"] and evs[2]["active"] and not evs[4]["active"]
    assert any(e["eou"] for e in evs)
