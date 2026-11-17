"""Adapter state machine driven by synthetic VoiceChat streams: one test per duplex case in DESIGN.md."""

from __future__ import annotations

import pytest

from conftest import qwen_encode
from frankenstein_s2s.adapter import AdapterConfig, State, TalkerAdapter
from frankenstein_s2s.events import Cancel, Frame, StartSegment, ToolEvent

REPLY = ["BOS", " Your", " order", " is", " ready", "."]


def test_burst_word_commit_burst_commit_and_eos_drain(rig):
    k = rig.run(REPLY)
    a = rig.adapter
    # word-level commits: "Your" only when " order" arrives; the last word waits for a separator or the commit
    starts = rig.cmds("StartSegment")
    assert len(starts) == 1 and starts[0].text == "Your" and starts[0].k == 2 and starts[0].segment == 0
    assert [c.text for c in rig.cmds("AppendText")] == [" order", " is", " ready."]
    assert a.state is State.SPEAKING
    k = rig.run(["PAD"], k)
    assert a.state is State.SPEAKING and not rig.cmds("FinalizeSegment")
    k = rig.run(["PAD"], k)                                   # 2nd PAD after text: burst commit
    fin = rig.cmds("FinalizeSegment")
    assert len(fin) == 1 and fin[0].reason == "burst_commit" and a.state is State.FINAL
    k = rig.run(["PAD*20", "EOS"], k)                         # VoiceChat EOS after its long PAD tail
    drain = rig.logs("eos_after_commit")
    assert drain and drain[0].detail["pad_tail_frames"] == 21
    assert a.state is State.DRAINING
    rig.pump()
    assert a.state is State.IDLE
    assert rig.gate.played_frames == 4 * rig.driver.frames_per_word
    assert not rig.cmds("Cancel")


def test_partial_words_are_never_committed(rig):
    rig.run(["BOS", " Accom", "mod", "ating", " guests"])
    assert rig.spoken_text() == "Accommodating"               # " guests" still open
    rig.run(["PAD", "PAD"], 5)
    assert rig.spoken_text() == "Accommodating guests"


def test_qwen_ids_equal_whole_segment_encoding(rig):
    rig.run(REPLY + ["PAD", "PAD"])
    ids = []
    for c in rig.cmds():
        if type(c).__name__ in ("StartSegment", "AppendText"):
            ids += list(c.qwen_ids)
    assert ids == qwen_encode("Your order is ready.")


def test_single_pad_pause_inside_burst_does_not_commit(rig):
    rig.run(["BOS", " One", " moment", "PAD", " please", "."])
    assert not rig.cmds("FinalizeSegment")
    assert rig.adapter.state is State.SPEAKING


def test_long_pause_then_more_text_opens_new_segment(rig):
    k = rig.run(["BOS", " Sure", ".", "PAD*3", " Anything", " else", "?", "PAD*2"])
    starts = rig.cmds("StartSegment")
    assert [(s.turn, s.segment, s.text) for s in starts] == [(1, 0, "Sure."), (1, 1, "Anything")]
    assert rig.logs("text_after_commit")
    assert [f.segment for f in rig.cmds("FinalizeSegment")] == [0, 1]
    assert not rig.cmds("Cancel")


def test_eos_before_commit_without_speech_is_natural_end(rig):
    rig.run(["BOS", " Bye", " now", "EOS"])
    fin = rig.cmds("FinalizeSegment")
    assert len(fin) == 1 and fin[0].reason == "eos"
    assert rig.spoken_text() == "Bye now"
    assert rig.logs("eos_before_commit")[0].action.startswith("AMBIGUOUS -> natural")
    assert not rig.cmds("Cancel")


def test_eos_before_commit_with_recent_speech_is_interruption(make_rig):
    rig = make_rig(barge_in="thinker")
    rig.run(["BOS", " Let", " me", " explain", " the", "EOS"], speech=frozenset({3, 4}), pump=True)
    cancels = rig.cmds("Cancel")
    assert len(cancels) == 1 and cancels[0].reason == "eos_with_user_speech"
    assert rig.adapter.epoch == 1 and rig.gate.epoch == 1
    assert rig.adapter.state is State.IDLE


def test_barge_in_cancels_and_flushes_stale_audio(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.driver.produce()                                      # audio generated, in flight (not yet delivered)
    in_flight = rig.driver.take()
    assert in_flight and all(c.epoch == 0 for c in in_flight)
    k = rig.run(["PAD", "PAD", "PAD"], k, speech=frozenset({k, k + 1, k + 2}))
    cancels = rig.cmds("Cancel")
    assert len(cancels) == 1 and cancels[0].reason == "barge_in" and cancels[0].new_epoch == 1
    for c in in_flight:                                       # delivered after the cancel: must be dropped
        assert rig.gate.accept(c) is False
    assert rig.gate.dropped_stale == len(in_flight) and rig.gate.played_frames == 0
    # the Thinker keeps writing its old turn for a while: nothing of it may reach the talker
    n = len(rig.adapter.commands)
    rig.run([" and", " more", "PAD*3", "EOS"], k)
    assert len(rig.adapter.commands) == n
    assert rig.logs("tok_outside_turn") and rig.logs("eos_outside_turn")


def test_barge_in_flushes_queued_but_unplayed_audio(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.driver.produce()
    for c in rig.driver.take():
        rig.gate.accept(c)                                    # queued for playback, not played yet
    queued = sum(c.n_frames for c in rig.gate.queue)
    rig.run(["PAD*3"], k, speech=frozenset({k, k + 1, k + 2}))
    assert rig.gate.flushed_frames == queued and not rig.gate.queue


def test_short_backchannel_does_not_cancel(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.run(["PAD*4"], k, speech=frozenset({k, k + 1}))      # 2 frames < barge_in_frames=3
    assert not rig.cmds("Cancel")


def test_thinker_mode_ignores_speech_until_eos(make_rig):
    rig = make_rig(barge_in="thinker")
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.run(["PAD*6"], k, speech=frozenset(range(k, k + 6)))
    assert not rig.cmds("Cancel")


def test_second_turn_after_cancel_is_clean(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    k = rig.run(["PAD*3"], k, speech=frozenset({k, k + 1, k + 2}))
    k = rig.run(["EOS", "PAD*4"], k)
    k = rig.run(["BOS", " Of", " course", ".", "PAD", "PAD", "PAD*3", "EOS"], k)
    starts = rig.cmds("StartSegment")
    assert [(s.epoch, s.turn, s.segment) for s in starts] == [(0, 1, 0), (1, 2, 0)]
    assert starts[1].text == "Of"                             # fresh prefill with the new turn's first word
    rig.pump()
    assert rig.adapter.state is State.IDLE
    assert all(c.epoch == 1 for c in rig.gate.played)
    assert rig.spoken_text(epoch=1) == "Of course."


def test_tool_call_preamble_wait_result_and_answer(rig):
    k = rig.run(["BOS", " Let", " me", " check", ".", "f:SOTC"])
    fin = rig.cmds("FinalizeSegment")
    assert fin[-1].reason == "tool_call" and rig.spoken_text() == "Let me check."
    assert rig.adapter.state is State.TOOL_WAIT
    k = rig.run(["PAD*5", "f:EOTC", "PAD*10"], k)
    rig.adapter.tool(ToolEvent(k, "pending", "c1"))
    rig.adapter.tool(ToolEvent(k + 1, "result", "c1"))
    assert rig.adapter.state is State.TOOL_WAIT
    k = rig.run(["f:EOTR"], k + 2)
    assert rig.adapter.state is State.DRAINING                # tool done; the preamble audio is still pending
    rig.pump()
    assert rig.adapter.state is State.IDLE
    k = rig.run(["BOS", " It", " ships", " today", ".", "PAD", "PAD", "EOS"], k)
    starts = rig.cmds("StartSegment")
    assert [(s.epoch, s.turn, s.text) for s in starts] == [(0, 1, "Let"), (0, 2, "It")]
    assert not rig.cmds("Cancel")


def test_text_without_bos_after_tool_opens_implicit_turn(rig):
    k = rig.run(["BOS", " Checking", ".", "f:SOTC", "PAD*3"])
    rig.adapter.tool(ToolEvent(k, "resumed", "c1"))
    rig.run([" Found", " it", ".", "PAD", "PAD"], k)
    assert rig.logs("text_without_bos")
    assert rig.cmds("StartSegment")[-1].text == "Found" and rig.cmds("StartSegment")[-1].turn == 2


def test_inject_on_hold_utterance_only_when_silent(rig):
    k = rig.run(["BOS", "f:SOTC", "PAD*2"])                    # call without preamble
    assert rig.adapter.state is State.TOOL_WAIT
    assert rig.adapter.inject_utterance(k, "One moment please.", "on_hold")
    assert rig.adapter.state is State.TOOL_WAIT and rig.adapter.busy   # TOOL_WAIT wins; the ack is audible
    assert rig.cmds("StartSegment")[-1].text == "One moment please." and rig.cmds("FinalizeSegment")[-1].reason == "inject:on_hold"
    assert not rig.adapter.inject_utterance(k + 1, "Still checking.", "reminder")   # previous one still audible
    rig.pump()
    assert rig.adapter.state is State.TOOL_WAIT


def test_long_reply_is_cut_into_segments_at_sentence_end(make_rig):
    rig = make_rig(segment_max_chars=20)
    spec = ["BOS"] + [" First", " sentence", " is", " long", ".", " Second", " one", ".", "PAD", "PAD"]
    rig.run(spec)
    segs = [(s.segment, s.text) for s in rig.cmds("StartSegment")]
    assert segs == [(0, "First"), (1, "Second")]
    assert [f.reason for f in rig.cmds("FinalizeSegment")] == ["segment_length", "burst_commit"]


def test_missing_frames_are_logged_and_do_not_count_as_pad(rig):
    rig.run(["BOS", " Hello", "MISS*3", " there", "PAD"])
    assert len(rig.logs("missing_text_id")) == 3
    assert not rig.cmds("FinalizeSegment")
    assert rig.spoken_text() == "Hello"


def test_bos_inside_open_turn(rig):
    k = rig.run(["BOS", " I", " was", " say"])
    k = rig.run(["BOS", " Sorry", ".", "PAD", "PAD"], k)
    assert rig.cmds("Cancel")[0].reason == "bos_restart"
    assert rig.cmds("StartSegment")[-1].epoch == 1


def test_bos_after_commit_without_eos_lets_audio_drain(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.run(["BOS", " Also", ",", "PAD", "PAD"], k)
    assert not rig.cmds("Cancel")
    assert [(s.turn, s.text) for s in rig.cmds("StartSegment")] == [(1, "Your"), (2, "Also,")]


def test_specials_on_text_channel_and_tokens_outside_turn_are_dropped(rig):
    rig.adapter.feed(Frame(0, 20))
    rig.adapter.feed(Frame(1, 999))
    rig.run([" stray"], 2)
    assert {e.what for e in rig.adapter.log} == {"sotc_on_text_channel", "special_on_text_channel", "tok_outside_turn"}
    assert not rig.adapter.commands


def test_empty_turn(rig):
    rig.run(["BOS", "PAD*3", "EOS"])
    assert rig.logs("empty_turn") and not rig.adapter.commands


def test_cascade_text_source_uses_same_rules(make_rig):
    rig = make_rig(talker_lang="italian")
    a = rig.adapter
    a.begin_turn(0)
    a.add_text(1, "Cert")
    a.add_text(2, "o, il suo ord")
    a.add_text(3, "ine è pronto.")
    a.end_turn(4)
    assert rig.spoken_text() == "Certo, il suo ordine è pronto."
    assert all(s.lang == "italian" for s in rig.cmds("StartSegment"))
    assert rig.cmds("FinalizeSegment")[-1].reason == "end_turn"


def test_cascade_interrupted_turn_cancels(rig):
    a = rig.adapter
    a.begin_turn(0)
    a.add_text(1, "Well, the thing is ")
    a.end_turn(2, interrupted=True)
    assert rig.cmds("Cancel")[0].reason == "end_turn_interrupted" and a.epoch == 1


def test_config_validation():
    with pytest.raises(ValueError):
        AdapterConfig(barge_in="never")
    with pytest.raises(ValueError):
        AdapterConfig(commit_pads=0)


def test_no_command_ever_targets_an_old_epoch(rig):
    k = rig.run(REPLY + ["PAD", "PAD"])
    k = rig.run(["PAD*3"], k, speech=frozenset({k, k + 1, k + 2}))
    k = rig.run(["BOS", " Yes", ".", "PAD", "PAD", "EOS"], k)
    epoch = 0
    for c in rig.adapter.commands:
        if isinstance(c, Cancel):
            assert c.epoch == epoch
            epoch = c.new_epoch
        else:
            assert c.epoch == epoch


def test_decode_required_for_token_input():
    a = TalkerAdapter([], decode=None)
    a.feed(Frame(0, 1))
    with pytest.raises(RuntimeError):
        a.feed(Frame(1, 1234))
