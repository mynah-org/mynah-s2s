"""The continuation check: an endpoint on a text that cannot end there waits for more words; more words keep the turn
open (the ears end it again later), none and the turn ends as it was."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.control.continuation import wants_more  # noqa: E402
from frankenstein2.core import Pipeline, PipelineConfig  # noqa: E402
from frankenstein2.ears.client import EarsLink  # noqa: E402
from frankenstein2.llm.base import ScriptedLLM  # noqa: E402
from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.tracing import Tracer  # noqa: E402
from frankenstein2.tts.base import FakeTalker  # noqa: E402

Z = np.zeros(1280, np.float32)


def test_wants_more():
    assert wants_more("vorrei un divano e", "it") and wants_more("ich möchte ein Sofa und", "de")
    assert wants_more("how much is the", "en") and wants_more("je voudrais une table pour", "fr")
    assert wants_more("Guten Tag,", "de") and wants_more("नमस्ते, क्या आपके पास सोफ़ा है और", "hi")
    assert not wants_more("Perfekt, vielen Dank.", "de") and not wants_more("how much is it?", "en")
    assert not wants_more("grazie mille", "it") and not wants_more("", "it")
    assert not wants_more("ありがとうございました と", "ja")           # Japanese / Korean: not judged


def make(script, cfg=None):
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: script.get(n))
    llm = ScriptedLLM(lambda m: [("text", "Certo.")], lang="it")
    tr = Tracer(session_id="s")
    p = Pipeline("s", EarsLink(be, "s", "it"), llm, FakeTalker(lang="it"), cfg or PipelineConfig(lang="it"), tr)
    p.open()
    return p, llm, tr


def _run(p, n):
    for _ in range(n):
        p.step(Z)


def test_a_pause_after_a_conjunction_keeps_the_turn_open():
    script = {2: {"pieces": pieces(4, "vorrei un divano e"), "events": [{"kind": "ENDPOINT", "k": 9}]},
              4: {"pieces": pieces(16, "una poltrona")}, 5: {"events": [{"kind": "ENDPOINT", "k": 22}]}}
    p, llm, tr = make(script)
    _run(p, 40)
    assert len(llm.submitted) == 1
    assert llm.submitted[0][1][-1]["content"] == "vorrei un divano e una poltrona"
    done = [e for e in tr.events if e["ev"] == "asr.hold_done"]
    assert done and done[0]["resumed"] is True


def test_no_more_words_ends_the_turn_after_the_wait():
    script = {2: {"pieces": pieces(4, "vorrei un divano e"), "events": [{"kind": "ENDPOINT", "k": 9}]}}
    p, llm, tr = make(script)
    _run(p, 40)
    assert llm.submitted[0][1][-1]["content"] == "vorrei un divano e"
    done = [e for e in tr.events if e["ev"] == "asr.hold_done"][0]
    assert done["resumed"] is False and done["waited_ms"] == 800


def test_a_finished_sentence_is_not_held_and_the_hold_can_be_off():
    script = {2: {"pieces": pieces(4, "vorrei un divano"), "events": [{"kind": "ENDPOINT", "k": 9}]}}
    p, llm, tr = make(script)
    _run(p, 16)
    assert len(llm.submitted) == 1 and not [e for e in tr.events if e["ev"] == "asr.hold"]
    script2 = {2: {"pieces": pieces(4, "vorrei un divano e"), "events": [{"kind": "ENDPOINT", "k": 9}]}}
    p2, llm2, _ = make(script2, PipelineConfig(lang="it", continuation_hold_steps=0))
    _run(p2, 16)
    assert len(llm2.submitted) == 1
