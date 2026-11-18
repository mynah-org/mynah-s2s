"""The session pipeline end to end with fakes: ears script, scripted LLM with real tool handlers, fake talker."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.core import HOLDING, Pipeline, PipelineConfig  # noqa: E402
from frankenstein2.ears.client import EarsLink  # noqa: E402
from frankenstein2.llm.base import ScriptedLLM  # noqa: E402
from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.tracing import Tracer  # noqa: E402
from frankenstein2.tts.base import FakeTalker  # noqa: E402

Z = np.zeros(1280, np.float32)


def orders_handler(db):
    def get_order_status(args):
        row = db.get(args.get("order_id"))
        return json.dumps({"found": row is not None, **(row or {})})
    return get_order_status


def make(sid, lang, ears_events, llm_script, tools=None, be=None, talker=None):
    """ears_events: {hop n: {"events": [...], "pieces": [...]}} (frames absolute)."""
    be = be or FakeEarsBackend(lambda s, n, mode, reset, u, a: ears_events.get(s, {}).get(n))
    llm = ScriptedLLM(llm_script, tools=tools, lang=lang)
    talker = talker or FakeTalker(lang=lang)
    tr = Tracer(session_id=sid)
    p = Pipeline(sid, EarsLink(be, sid, lang), llm, talker, PipelineConfig(lang=lang), tr)
    p.open()
    return p, be, llm, talker, tr


def run(p, steps):
    outs = []
    for _ in range(steps):
        outs.append(p.step(Z))
    return outs


def ev(kind, frame, **kw):
    return dict(kind=kind, k=frame, **kw)


def test_simple_turn_is_spoken_and_closed():
    ears = {"s": {2: {"pieces": pieces(4, "buongiorno vorrei informazioni"), "events": [ev("ENDPOINT", 10)]}}}
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: [("text", "Buongiorno! "), ("text", "Mi dica.")])
    outs = run(p, 60)
    assert llm.submitted[0][1] == [{"role": "user", "content": "buongiorno vorrei informazioni"}]
    assert [s["text"] for s in talker.said] == ["Buongiorno!", "Mi dica."]
    assert sum(o["agent"].any() for o in outs) > 0
    assert p.ctl.state.value == "listening"
    assert p.conv.messages[-1] == {"role": "assistant", "content": "Buongiorno! Mi dica."}
    modes = [h["mode"] for h in be.hops]
    assert "speaking" in modes and modes[-1] == "listen"
    assert any(h["reset"] for h in be.hops[4:])                 # the turn end reset the ears


def test_tool_turn_holding_phrase_digits_and_history():
    db = {"XYZ123": {"status": "shipped"}}
    ears = {"s": {2: {"pieces": pieces(4, "dov'è il mio ordine X Y Z uno due tre"), "events": [ev("ENDPOINT", 11)]}}}
    script = lambda m: [("decided",), ("wait", 3), ("tool", "get_order_status", {"order_id": "X Y Z uno due tre"}),
                        ("text", "Il suo ordine è stato spedito.")]
    p, be, llm, talker, tr = make("s", "it", ears, script, tools={"get_order_status": orders_handler(db)})
    run(p, 60)
    assert llm.calls[0]["args"] == {"order_id": "XYZ123"} and '"found": true' in llm.calls[0]["result"]
    assert talker.said[0]["text"] == HOLDING["it"]
    roles = [m["role"] for m in p.conv.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert HOLDING["it"] not in json.dumps(p.conv.messages)     # the holding phrase never enters the history


def test_barge_in_keeps_only_heard_text_and_flushes():
    long = "Il suo ordine è stato spedito ieri dal magazzino. Arriverà entro tre giorni lavorativi. Grazie."
    ears = {"s": {2: {"pieces": pieces(4, "dov'è il mio ordine"), "events": [ev("ENDPOINT", 11)]},
                  7: {"pieces": pieces(28, "aspetti un attimo"), "events": [ev("BARGE", 30, onset_s=0.24)]},
                  10: {"pieces": pieces(36, "ho cambiato indirizzo"), "events": [ev("ENDPOINT", 43)]}}}
    replies = iter([[("text", long)], [("text", "Va bene.")]])
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: next(replies))
    outs = run(p, 80)
    fl = tr.of("audio.flush")
    assert fl and fl[0]["reason"] == "barge_in" and fl[0]["dropped_samples"] > 0
    agent_msgs = [m["content"] for m in p.conv.messages if m["role"] == "assistant"]
    assert agent_msgs[0].endswith(" —") and agent_msgs[0].startswith("Il suo ordine")
    assert len(agent_msgs[0]) < len(long)
    second = llm.submitted[1][1]
    assert second[-1] == {"role": "user", "content": "aspetti un attimo ho cambiato indirizzo"}
    assert talker.cancels == [1]
    stale_audio = [i for i, o in enumerate(outs) if o["agent"].any() and i > fl[0]["model_step"]]
    first_new = min(stale_audio) if stale_audio else None
    # after the flush only the new turn's audio plays (epoch 1)
    assert first_new is None or all(s["epoch"] == 1 for s in talker.said if s["text"] == "Va bene.")


def test_barge_before_audio_merges_turns():
    ears = {"s": {2: {"pieces": pieces(4, "cerco un divano"), "events": [ev("ENDPOINT", 9)]},
                  3: {"pieces": pieces(12, "a tre posti"), "events": [ev("BARGE", 14, onset_s=0.2)]},
                  5: {"events": [ev("ENDPOINT", 21)]}}}
    replies = iter([[("wait", 30), ("text", "x")], [("text", "Abbiamo il modello Alba.")]])
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: next(replies))
    run(p, 60)
    assert llm.submitted[1][1] == [{"role": "user", "content": "cerco un divano a tre posti"}]
    assert tr.of("history.drop_user")


def test_backchannel_is_played_but_not_in_second_ear_or_history():
    ears = {"s": {1: {"pieces": pieces(3, "allora io volevo"), "events": [ev("BACKCHANNEL", 6)]}}}
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: [])
    outs = run(p, 30)
    assert [s["text"] for s in talker.said] == ["Mh-mh."]
    assert any(o["audio"].any() for o in outs)
    assert not any(o["agent"].any() for o in outs)
    assert all(h["a_rms"] == 0.0 for h in be.hops)
    assert p.conv.messages == []


def test_words_after_an_early_endpoint_cancel_and_merge():
    ears = {"s": {2: {"pieces": pieces(4, "cerco un divano"), "events": [ev("ENDPOINT", 9)]},
                  3: {"pieces": pieces(12, "a tre posti")},
                  6: {"events": [ev("ENDPOINT", 24)]}}}
    replies = iter([[("decided",), ("wait", 20), ("text", "x")], [("text", "Abbiamo il modello Alba.")]])
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: next(replies))
    run(p, 60)
    assert [a["reason"] for a in tr.of("ctl.action") if a["kind"] == "CANCEL"] == ["user_continues"]
    assert llm.submitted[1][1] == [{"role": "user", "content": "cerco un divano a tre posti"}]


def test_punctuation_or_tag_after_endpoint_is_not_a_continuation():
    ears = {"s": {2: {"pieces": pieces(4, "vorrei sapere dov'è il mio ordine"), "events": [ev("ENDPOINT", 10)]},
                  3: {"pieces": [{"f": 14, "p": ".", "id": 0}, {"f": 14, "p": "▁", "id": 0},
                                 {"f": 14, "p": "<it-IT>", "id": 0}]}}}
    p, be, llm, talker, tr = make("s", "it", ears, lambda m: [("wait", 10), ("text", "Certo.")])
    run(p, 40)
    assert not [a for a in tr.of("ctl.action") if a["kind"] == "CANCEL"]
    assert [s["text"] for s in talker.said] == ["Certo."]
