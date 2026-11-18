"""Isolation: sessions in different languages (IT, EN, FR) stepped interleaved on SHARED backends (one ears
backend, one tool database), one of them with a barge-in, must give each session exactly what it gives alone."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.tests.test_pipeline import ev, make, orders_handler  # noqa: E402

Z = np.zeros(1280, np.float32)
DB = {"XYZ123": {"status": "shipped"}, "QQQ999": {"status": "processing"}}

EARS = {
    "it": {2: {"pieces": pieces(4, "ordine X Y Z uno due tre"), "events": [ev("ENDPOINT", 11)]},
           10: {"pieces": pieces(40, "no aspetti"), "events": [ev("BARGE", 42, onset_s=0.16)]},
           13: {"pieces": pieces(48, "un altro ordine"), "events": [ev("ENDPOINT", 55)]}},
    "en": {3: {"pieces": pieces(8, "order Q Q Q nine nine nine"), "events": [ev("ENDPOINT", 15)]},
           12: {"pieces": pieces(44, "thank you"), "events": [ev("ENDPOINT", 50)]}},
    "fr": {1: {"pieces": pieces(1, "bonjour je voudrais"), "events": [ev("BACKCHANNEL", 6)]},
           4: {"pieces": pieces(14, "une information"), "events": [ev("ENDPOINT", 19)]}},
}
SCRIPTS = {
    "it": lambda m: ([("decided",), ("tool", "get_order_status", {"order_id": m[-1]["content"].split("ordine ")[-1]}),
                      ("text", "Il suo ordine è stato spedito ieri dal magazzino centrale. Arriverà presto.")]
                     if "X Y Z" in m[-1]["content"] else [("text", "Certo, mi dica il numero.")]),
    "en": lambda m: ([("decided",), ("tool", "get_order_status", {"order_id": "Q Q Q nine nine nine"}),
                      ("text", "Your order is being processed.")] if "nine" in m[-1]["content"]
                     else [("text", "You're welcome.")]),
    "fr": lambda m: [("text", "Bien sûr, je vous écoute.")],
}


def build(langs, be):
    return {lang: make(lang, lang, EARS, SCRIPTS[lang], tools={"get_order_status": orders_handler(DB)}, be=be)
            for lang in langs}


def snapshot(sess, outs):
    p, be, llm, talker, tr = sess
    return {"history": json.dumps(p.conv.messages, ensure_ascii=False),
            "submitted": json.dumps(llm.submitted, ensure_ascii=False), "calls": json.dumps(llm.calls),
            "said": json.dumps([(s["text"], s["epoch"], s["lang"]) for s in talker.said], ensure_ascii=False),
            "audio": b"".join(o["audio"].tobytes() for o in outs),
            "hops": json.dumps([(h["n"], h["mode"], h["reset"], round(h["a_rms"], 6))
                                for h in be.hops if h["sid"] == p.sid]),
            "asr_turns": json.dumps([e["text"] for e in tr.of("asr.turn")], ensure_ascii=False)}


def run_alone(lang, steps=90):
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: EARS[s].get(n))
    s = build([lang], be)[lang]
    outs = [s[0].step(Z) for _ in range(steps)]
    return snapshot(s, outs)


def test_three_languages_interleaved_equal_alone():
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: EARS[s].get(n))
    sess = build(["it", "en", "fr"], be)
    outs = {lang: [] for lang in sess}
    for _ in range(90):
        for lang in ("fr", "it", "en"):
            outs[lang].append(sess[lang][0].step(Z))
    for lang in sess:
        assert snapshot(sess[lang], outs[lang]) == run_alone(lang), lang


def test_languages_reach_every_component():
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: EARS[s].get(n))
    sess = build(["it", "en", "fr"], be)
    for _ in range(90):
        for lang in sess:
            sess[lang][0].step(Z)
    assert {s: m["lang"] for s, m in be.opened.items()} == {"it": "it", "en": "en", "fr": "fr"}
    it, en, fr = sess["it"], sess["en"], sess["fr"]
    assert it[2].calls[0]["args"] == {"order_id": "XYZ123"} and en[2].calls[0]["args"] == {"order_id": "QQQ999"}
    assert it[3].said[0]["text"] == "Un momento, controllo." and en[3].said[0]["text"] == "One moment, let me check."
    assert fr[3].said[0]["text"] == "Mh-mh." and all(s["lang"] == "fr" for s in fr[3].said)
    assert it[3].cancels == [1] and en[3].cancels == [] and fr[3].cancels == []
    # the normaliser runs in each session's language
    seg = [e for e in en[4].of("tts.segment")]
    assert all(e["speech"] == e["text"] for e in seg)
    assert any(m["content"].endswith(" —") for m in it[0].conv.messages if m["role"] == "assistant")
    assert not any(m["content"].endswith(" —") for m in en[0].conv.messages + fr[0].conv.messages)
