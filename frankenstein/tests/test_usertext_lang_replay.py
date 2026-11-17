"""User-text scheduling (causal, one id per frame), turn-taking counters, language plumbing, replay driving."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from frankenstein_s2s import tokens as T
from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter
from frankenstein_s2s.driver import FakeTalkerDriver
from frankenstein_s2s.lang import language_instruction, session_language, with_language
from frankenstein_s2s.replay import drive, load
from frankenstein_s2s.usertext import (InjectionMode, TranscriptInserter, TurnTaker, UserTextScheduler,
                                       frame_inputs)


def fake_thinker_encode(word: str) -> list:
    """' word' -> one id per 3 characters (a stand-in for BPE splitting), ids >= 1000; 'SPECIAL' -> id 12."""
    w = word.strip()
    if w == "SPECIAL":
        return [12]
    return [1000 + (hash(w[i:i + 3]) % 50000) for i in range(0, len(w), 3)]


def test_one_token_per_frame_and_causal_release():
    s = UserTextScheduler(fake_thinker_encode, close_after=3)
    out = {}
    asr = {0: ["▁buon", "gior"], 1: ["no"], 4: ["▁vorrei"], 5: ["▁una"], 6: [","]}
    for k in range(14):
        s.on_asr(k, asr.get(k, []))
        out[k] = s.step(k)
    # "buongiorno" closes when "vorrei" starts (k=4): its 4 ids leave one per frame from k=4
    assert [k for k, v in out.items() if v is not None] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert s.words == ["buongiorno", "vorrei", "una,"]
    # causality: nothing is released before the word was complete
    assert all(v is None for k, v in out.items() if k < 4)


def test_close_after_idle_frames_and_flush():
    s = UserTextScheduler(fake_thinker_encode, close_after=3)
    s.on_asr(0, ["▁ciao"])
    assert s.step(0) is None and s.step(1) is None and s.step(2) is None
    assert s.step(3) is not None                               # closed 3 frames after its last piece
    s.on_asr(10, ["▁grazie"])
    s.flush(10)
    assert s.step(10) is not None


def test_language_tags_late_punctuation_and_specials_dropped():
    s = UserTextScheduler(fake_thinker_encode)
    s.on_asr(0, ["."])                                         # punctuation with no open word
    s.on_asr(1, ["▁SPECIAL", "<it-IT>"])
    s.flush(2)
    reasons = [d[2] for d in s.dropped]
    assert reasons == ["late_punctuation", "lang_tag", "special_id"]
    assert s.backlog == 0


def test_turn_taker_bou_and_eou():
    tt = TurnTaker(eou_frames=4, bou_frames=2)
    ev = []
    for k, nb in enumerate([1, 0, 1, 1, 1, 0, 0, 0, 0, 0]):
        ev += [(k, e) for e in tt.step(k, bool(nb))]
    assert ev == [(3, "bou"), (8, "eou")]
    assert not tt.user_speaking


def test_frame_inputs_modes():
    m0 = frame_inputs(InjectionMode.M0_BASELINE, None)
    assert m0 == {"user_audio": "perception", "audio_weight": 1.0, "user_text": None, "user_text_weight": 0.0,
                  "agent": "feedback"}
    # M1_CTX: real frames unchanged; the transcript lives only in inserted steps (prompt layout, agent PAD)
    assert frame_inputs(InjectionMode.M1_CTX, None) == m0
    ins = frame_inputs(InjectionMode.M1_CTX, 1234, inserted=True)
    assert ins["user_audio"] == ("token", 1234) and ins["agent"] == "pad" and ins["user_text"] is None
    with pytest.raises(ValueError):
        frame_inputs(InjectionMode.M1_CTX, 1234)                   # never onto a real audio frame
    with pytest.raises(ValueError):
        frame_inputs(InjectionMode.M0_BASELINE, 1234, inserted=True)
    assert frame_inputs(InjectionMode.M1_FORCE, None)["agent"] == "forced"
    with pytest.raises(ValueError):
        frame_inputs(InjectionMode.M1_FORCE, 1234)                 # the transcript goes to the brain, not the frame
    chan = frame_inputs(InjectionMode.M1_CHAN, 1234)
    assert chan["user_text"] == ("token", 1234) and chan["user_text_weight"] == 1.0
    assert frame_inputs(InjectionMode.M2_DAMPED, None, damped_audio_weight=0.3)["audio_weight"] == 0.3
    assert frame_inputs(InjectionMode.M3_BYPASS, None)["user_audio"] == "silence"
    with pytest.raises(ValueError):
        frame_inputs(InjectionMode.M1_CHAN, T.EOS_ID)


def test_inserter_is_causal_and_holds_bos():
    s = UserTextScheduler(fake_thinker_encode, close_after=3)
    ins = TranscriptInserter(policy="eou", prefix_ids=(1,), suffix_ids=(2,))
    s.on_close = lambda k, w, ids: ins.on_word(k, w, ids)
    s.on_asr(0, ["\u2581vorrei"])
    s.on_asr(2, ["\u2581due"])
    assert ins.hold_bos and not ins.spans                        # "vorrei" closed at k=2, waits for EOU
    for k in range(3, 6):
        s.step(k)                                                 # "due" closes at k=5
    span = ins.on_eou(12)
    assert span.after_k == 12 and span.words == ("vorrei", "due") and span.ids[0] == 1 and span.ids[-1] == 2
    assert not ins.hold_bos
    with pytest.raises(ValueError):
        ins.on_word(13, "x", [5])                                 # special ids are never inserted
    word = TranscriptInserter(policy="word")
    a = word.on_word(4, "ciao", [1500, 1501])
    b = word.on_word(7, "grazie", [1600])
    assert (a.after_k, a.ids, b.after_k, b.ids) == (4, (1500, 1501), 7, (1600,))
    assert word.on_eou(9) is None
    with pytest.raises(ValueError):
        TranscriptInserter(policy="late")
    late = TranscriptInserter()
    late.on_word(10, "x", [1500])
    with pytest.raises(ValueError):
        late.on_eou(9)                                            # no EOU before the last heard word


def test_session_language_end_to_end():
    it = session_language("IT")
    assert (it.asr_locale, it.talker_lang) == ("it-IT", "italian")
    prompt = with_language("You are the Mynah Home store assistant.", it)
    assert prompt.endswith(language_instruction(it)) and "Always answer in Italian" in prompt
    with pytest.raises(ValueError):
        session_language("xx")
    a = TalkerAdapter([FakeTalkerDriver()], decode=lambda ids: "".join(" w" for _ in ids),
                      config=AdapterConfig(talker_lang=it.talker_lang))
    a.begin_turn(0)
    a.add_text(1, "Va bene.")
    a.end_turn(2)
    assert a.commands[0].lang == "italian"


def test_tokens_match_v2_engine_constants():
    v2 = Path(__file__).resolve().parents[2] / "v2/src/mynah_v2/model/tokens.py"
    if not v2.exists():
        pytest.skip("V2 sources not present")
    src = v2.read_text()
    for name in ("BOS_ID", "EOS_ID", "PAD_ID", "SOTC_ID", "EOTC_ID", "EOTR_ID"):
        m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.M)
        assert m and int(m.group(1)) == getattr(T, name), name


def test_replay_drives_adapter(tmp_path):
    vocab = {1000: " Your", 1001: " order", 1002: " shipped", 1003: "."}
    rows = [{"format": "vc_replay_v1", "frame_s": 0.08}]
    ids = [12, 1, 1000, 1001, 1002, 1003, 12, 12, 12, 2]
    rows += [{"k": k, "id": i, "kind": "x"} for k, i in enumerate(ids)]
    rows += [{"event": "turn.started", "k": 1}]
    p = tmp_path / "r.replay.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    hdr, frames, events = load(p)
    drv = FakeTalkerDriver()
    a = TalkerAdapter([drv], decode=lambda xs: "".join(vocab[x] for x in xs))
    drive(a, frames, events)
    texts = "".join(c.text for c in a.commands if hasattr(c, "text"))
    assert texts == "Your order shipped."
    assert any(e.what == "eos_after_commit" for e in a.log)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        load(bad)
