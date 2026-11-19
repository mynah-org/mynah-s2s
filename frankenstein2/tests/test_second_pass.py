"""The second ASR pass: an uncertain turn (voice before the first transcribed piece, or voice with no text) is
transcribed again offline before it goes to the LLM; a clear turn is not; a missing answer never blocks the turn."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.control.second_pass import SecondPassConfig, Suspicion, TurnAudio, choose  # noqa: E402
from frankenstein2.core import Pipeline, PipelineConfig  # noqa: E402
from frankenstein2.ears.client import EarsLink  # noqa: E402
from frankenstein2.llm.base import ScriptedLLM  # noqa: E402
from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.tracing import Tracer  # noqa: E402
from frankenstein2.tts.base import FakeTalker  # noqa: E402

Z = np.zeros(1280, np.float32)


def voiced(f0, f1, db=-20.0):
    return [{"k": f, "db": db, "mode": "listen"} for f in range(f0, f1)]


def test_suspicion_unheard_onset_and_empty_with_voice():
    c = SecondPassConfig()
    s = Suspicion(c)
    for f in voiced(10, 30):
        s.frame(f["k"], f["db"])
    s.pieces([22, 23, 24])
    assert s.verdict("vielen Dank") == "unheard_onset"           # 10..18 voiced before 22 - lag
    s2 = Suspicion(c)
    for f in voiced(10, 20):
        s2.frame(f["k"], f["db"])
    s2.pieces([11, 12])
    assert s2.verdict("perfekt vielen dank") is None             # the first piece right after the onset
    s3 = Suspicion(c)
    for f in voiced(10, 20):
        s3.frame(f["k"], f["db"])
    assert s3.verdict("") == "empty_with_voice"
    s4 = Suspicion(c)
    for k in range(10, 30):
        s4.frame(k, -20.0, "speaking")                           # the agent's time: not the user's voice
        s4.frame(k, -60.0)                                       # silence while listening
    s4.pieces([25])
    assert s4.verdict("ok") is None
    s.reset(40)
    assert s.voiced == [] and s.first_piece is None


def test_turn_audio_cut_and_choose():
    a = TurnAudio(keep_s=0.8)                                    # 10 steps kept
    for k in range(20):
        a.push(k, np.full(1280, k, np.float32))
    x = a.cut(12, 15)
    assert len(x) == 3 * 1280 and x[0] == 12 and x[-1] == 14
    assert len(a.cut(0, 5)) == 0                                 # dropped
    assert choose("und auf Wiedersehen", "Perfekt, vielen Dank und auf Wiedersehen") == \
        ("Perfekt, vielen Dank und auf Wiedersehen", "offline")
    assert choose("perfekt vielen dank", "Perfekt, vielen Dank.") == ("perfekt vielen dank", "streaming")
    assert choose("ok", None) == ("ok", "streaming") and choose("ok", "") == ("ok", "streaming")


def make(script, transcribe, cfg=None):
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: script.get(n), transcribe=transcribe)
    llm = ScriptedLLM(lambda m: [("text", "Gern geschehen.")], lang="de")
    tr = Tracer(session_id="s")
    p = Pipeline("s", EarsLink(be, "s", "de"), llm, FakeTalker(lang="de"),
                 PipelineConfig(lang="de", **({"second_pass": cfg} if cfg else {})), tr)
    p.open()
    return p, be, llm, tr


def _run(p, n):
    for _ in range(n):
        p.step(Z)


def test_uncertain_turn_is_heard_again_offline():
    # voice from frame 4, the stream's first piece only at frame 14 ("und auf Wiedersehen"), endpoint at 18
    script = {1: {"frames": voiced(4, 8)}, 2: {"frames": voiced(8, 12)},
              3: {"frames": voiced(12, 16), "pieces": pieces(14, "und auf Wiedersehen")},
              4: {"frames": voiced(16, 18), "events": [{"kind": "ENDPOINT", "k": 18}]}}
    p, be, llm, tr = make(script, lambda sid, pcm, lang: "Perfekt, vielen Dank und auf Wiedersehen")
    _run(p, 40)
    assert be.transcribed and be.transcribed[0]["lang"] == "de" and be.transcribed[0]["samples"] > 10 * 1280
    assert llm.submitted[0][1][-1] == {"role": "user", "content": "Perfekt, vielen Dank und auf Wiedersehen"}
    sp = [e for e in tr.events if e["ev"] == "asr.second_pass"]
    assert sp and sp[0]["used"] == "offline" and sp[0]["reason"] == "unheard_onset" and not sp[0]["timed_out"]


def test_clear_turn_is_not_transcribed_again():
    script = {1: {"frames": voiced(4, 8), "pieces": pieces(5, "perfekt vielen")},
              2: {"frames": voiced(8, 10), "pieces": pieces(8, "dank"), "events": [{"kind": "ENDPOINT", "k": 11}]}}
    p, be, llm, tr = make(script, lambda sid, pcm, lang: "should not be asked")
    _run(p, 30)
    assert be.transcribed == []
    assert llm.submitted[0][1][-1]["content"] == "perfekt vielen dank"


def test_no_offline_answer_falls_back_to_the_stream_after_the_deadline():
    script = {1: {"frames": voiced(4, 8)}, 2: {"frames": voiced(8, 12)},
              3: {"frames": voiced(12, 16), "pieces": pieces(14, "und auf Wiedersehen")},
              4: {"events": [{"kind": "ENDPOINT", "k": 16}]}}
    p, be, llm, tr = make(script, lambda sid, pcm, lang: None)
    _run(p, 40)
    assert llm.submitted[0][1][-1]["content"] == "und auf Wiedersehen"
    sp = [e for e in tr.events if e["ev"] == "asr.second_pass"][0]
    assert sp["timed_out"] and sp["used"] == "streaming" and sp["wait_ms"] >= 480


def test_second_pass_can_be_switched_off():
    from frankenstein2.control.second_pass import SecondPassConfig as C
    script = {1: {"frames": voiced(4, 8)}, 2: {"frames": voiced(8, 12)},
              3: {"frames": voiced(12, 16), "pieces": pieces(14, "und auf Wiedersehen")},
              4: {"events": [{"kind": "ENDPOINT", "k": 16}]}}
    p, be, llm, tr = make(script, lambda sid, pcm, lang: "Perfekt", cfg=C(enabled=False))
    _run(p, 30)
    assert be.transcribed == [] and llm.submitted[0][1][-1]["content"] == "und auf Wiedersehen"
