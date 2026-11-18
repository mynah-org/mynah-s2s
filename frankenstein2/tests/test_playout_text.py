"""Playout (real-time release, flush, heard text, 2nd ear), chunker, spoken numbers, ears hop clock + transcript."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.audio.playout import Playout, Resampler, SecondEar  # noqa: E402
from frankenstein2.ears.client import EarsLink, UserText  # noqa: E402
from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.text.chunker import Chunker  # noqa: E402
from frankenstein2.text.spoken_numbers import normalize_args  # noqa: E402

S = 1920


def tone(n, amp=0.1):
    return (amp * np.ones(n)).astype(np.float32)


def test_release_is_paced_one_step_at_a_time():
    p = Playout()
    p.register(0, 1, 0, "agent", "Buongiorno.")
    p.enqueue(0, 1, 0, tone(5 * S))
    out = [p.release()[2]["agent"] for _ in range(7)]
    assert out == [S, S, S, S, S, 0, 0]
    assert p.turn_first_step(0, 1) == 0


def test_flush_drops_unreleased_and_later_audio_is_stale():
    p = Playout()
    p.register(0, 1, 0, "agent", "x")
    p.enqueue(0, 1, 0, tone(10 * S))
    p.release()
    assert p.flush(1) == 9 * S
    assert not p.enqueue(0, 1, 0, tone(S)) and p.stale == 1
    assert p.release()[2]["agent"] == 0


def test_heard_text_cuts_in_proportion_at_a_word():
    p = Playout(client_delay_steps=0)
    p.register(0, 1, 0, "agent", "Il suo ordine è partito ieri.")
    p.register(0, 1, 1, "agent", "Arriva domani.")
    p.enqueue(0, 1, 0, tone(6 * S))
    p.seg_done(0, 1, 0)
    p.enqueue(0, 1, 1, tone(4 * S))
    p.seg_done(0, 1, 1)
    for _ in range(3):
        p.release()
    assert p.heard_text(0, 1) == ("Il suo ordine", False)
    for _ in range(4):
        p.release()
    assert p.heard_text(0, 1) == ("Il suo ordine è partito ieri.", False)
    for _ in range(4):
        p.release()
    assert p.heard_text(0, 1) == ("Il suo ordine è partito ieri. Arriva domani.", True)


def test_holding_phrase_not_in_heard_text():
    p = Playout(client_delay_steps=0)
    p.register(0, 1, 0, "agent", "Un momento, controllo.", holding=True)
    p.register(0, 1, 1, "agent", "Trovato.")
    p.enqueue(0, 1, 0, tone(2 * S))
    p.seg_done(0, 1, 0)
    p.enqueue(0, 1, 1, tone(2 * S))
    p.seg_done(0, 1, 1)
    for _ in range(5):
        p.release()
    assert p.heard_text(0, 1) == ("Trovato.", True)


def test_second_ear_gets_agent_delayed_and_not_backchannel():
    p, ear = Playout(), SecondEar(client_delay_steps=1)
    p.register(0, 1, 0, "agent", "x")
    p.register(0, 1, 9, "bc", "mh")
    p.enqueue(0, 1, 0, tone(S, 0.2))
    p.enqueue(0, 1, 9, tone(3 * S, 0.5))
    heard = []
    for _ in range(3):
        heard.append(float(np.abs(ear.take()).max()))
        mix, agent, _ = p.release()
        ear.push(agent)
    assert heard[0] == 0.0 and heard[1] > 0.15 and heard[2] < 0.05   # agent one step late; bc never
    assert float(mix.max()) < 0.05 + 0.7 * 0.5 + 1e-6


def test_underrun_counted_once_per_gap():
    p = Playout()
    p.register(0, 1, 0, "agent", "x")
    p.enqueue(0, 1, 0, tone(S))
    p.release()
    p.release()
    p.release()
    assert len(p.underruns) == 1
    p.seg_done(0, 1, 0)


def test_resampler_lengths():
    r = Resampler(24000, 16000)
    r.s = None
    n = sum(len(r(np.zeros(1920, np.float32))) for _ in range(10))
    assert abs(n - 12800) <= 2


def test_chunker_sentences_numbers_and_flush():
    c = Chunker("it", "sentence")
    segs = []
    for piece in ["Il divano costa 49,", "90 €. Arriva", " il 14/11 alle 10:30. Grazie"]:
        segs += c.feed(piece)
    segs += c.flush()
    raw = [r for r, _ in segs]
    assert raw == ["Il divano costa 49,90 €.", "Arriva il 14/11 alle 10:30.", "Grazie"]
    assert segs[0][1] == "Il divano costa quarantanove euro e novanta."
    assert not any(ch.isdigit() for _, s in segs for ch in s)


def test_chunker_whole_and_progressive():
    c = Chunker("en", "whole")
    assert c.feed("Hello. How are you?") == []
    assert c.flush()[0][0] == "Hello. How are you?"
    p = Chunker("en", "progressive", min_piece=5)
    segs = p.feed("Your total is 49") + p.feed(",90 euros today ") + p.flush()
    assert "".join(r + " " for r, _ in segs).split() == "Your total is 49,90 euros today".split()
    assert all("49" not in r or "49,90" in r for r, _ in segs)


def test_spoken_numbers_in_tool_args():
    a = normalize_args({"name": "Mario Rossi", "postcode": "uno zero uno due uno", "query": "divano a tre posti"},
                       "it")
    assert a == {"name": "Mario Rossi", "postcode": "10121", "query": "divano a tre posti"}
    assert normalize_args({"order_id": "X Y Z one two three"}, "en") == {"order_id": "XYZ123"}
    assert normalize_args({"order_id": "Q Q Q nove nove nove"}, "it") == {"order_id": "QQQ999"}
    assert normalize_args({"code": "sept cinq zéro un un"}, "fr") == {"code": "75011"}
    assert normalize_args({"cap": "ventitré"}, "it") == {"cap": "23"}
    assert normalize_args({"note": "un tavolo per due"}, "it") == {"note": "un tavolo per due"}
    assert normalize_args({"postcode": "10121"}, "it") == {"postcode": "10121"}


def test_user_text_take_and_drop():
    u = UserText()
    u.add(pieces(10, "vorrei un divano"))
    assert u.pending() == "vorrei un divano"
    assert u.take() == "vorrei un divano" and u.pending() == ""
    u.add(pieces(40, "eco del agente"))
    u.add(pieces(60, "aspetti un attimo"))
    assert u.drop_before(55) == 3
    assert u.take() == "aspetti un attimo"


def test_ears_link_hops_one_in_flight_and_reset_order():
    be = FakeEarsBackend(lambda *a: {}, delay=6)
    link = EarsLink(be, "s1", "it")
    link.open()
    z = np.zeros(1280, np.float32)
    for i in range(8):
        if i == 5:
            link.request_reset()
        link.push(z, z, "listen" if i < 4 else "speaking")
        link.poll()
    assert [h["n"] for h in be.hops] == [0]                 # hop 1 waits for hop 0's result
    for _ in range(12):
        link.poll()
    assert [(h["n"], h["mode"], h["reset"]) for h in be.hops] == [(0, "listen", False), (1, "speaking", True)]
