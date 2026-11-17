"""M1-force per-frame decisions, and the forced stream driving the talker adapter end to end."""

from __future__ import annotations

from frankenstein_s2s import tokens as T
from frankenstein_s2s.events import Frame
from frankenstein_s2s.forcing import AgentChannelForcer, ForceState

PAD, BOS, EOS = T.PAD_ID, T.BOS_ID, T.EOS_ID


def run(f, predicted, start=0, hook=None):
    out = []
    for i, p in enumerate(predicted):
        if hook:
            hook(start + i, f)
        out.append(f.step(start + i, p))
    return out


def test_model_bos_opens_turn_and_reply_is_forced_one_id_per_frame():
    f = AgentChannelForcer()
    steps = run(f, [PAD, BOS])
    assert steps[1].write is None and steps[1].to_adapter == BOS and f.need_reply
    f.add_reply([2001, 2002, 2003], done=True)
    # the model would say its own (English) words: overridden; the model's PAD tail and EOS are kept
    steps = run(f, [1500, 2002, 1501, PAD, PAD, EOS], start=2)
    assert [s.write for s in steps] == [2001, None, 2003, None, None, None]
    assert [s.to_adapter for s in steps] == [2001, 2002, 2003, PAD, PAD, EOS]
    assert f.state is ForceState.LISTEN and f.overrides == 2 and f.agreements == 1


def test_reply_not_ready_forces_pad_and_gives_adapter_no_text():
    f = AgentChannelForcer(wait_max=3)
    run(f, [BOS])
    steps = run(f, [1500, PAD, 1501], start=1)
    assert [s.write for s in steps] == [PAD, None, PAD] and all(s.to_adapter is None for s in steps)
    steps = run(f, [PAD], start=4)
    assert steps[0].write == EOS and f.log[-1][2] == "no_reply_text"


def test_generation_underrun_mid_reply_is_not_a_commit():
    f = AgentChannelForcer()
    run(f, [BOS])
    f.add_reply([2001])
    steps = run(f, [PAD, PAD, PAD], start=1)
    assert steps[0].to_adapter == 2001 and steps[1].to_adapter is None and steps[2].to_adapter is None
    f.add_reply([2002], done=True)
    assert f.step(4, PAD).to_adapter == 2002


def test_barge_in_cut_forces_eos_and_drops_rest():
    f = AgentChannelForcer()
    run(f, [BOS])
    f.add_reply([2001, 2002, 2003, 2004], done=True)
    run(f, [PAD, PAD], start=1)
    f.cut()
    s = f.step(3, PAD)
    assert s.write == EOS and s.to_adapter == EOS and f.state is ForceState.LISTEN
    assert f.turn_ids == [2001, 2002]                         # what was actually forced: the history keeps this
    assert f.log[-1][3]["dropped_reply_ids"] == 2


def test_eou_fallback_forces_bos_when_duplex_stays_silent():
    f = AgentChannelForcer(bos_fallback_frames=3)
    f.on_eou(10)
    steps = run(f, [PAD, PAD, PAD, PAD], start=10)
    assert [s.write for s in steps[:3]] == [None, None, None] and steps[3].write == BOS


def test_model_text_while_listening_is_never_spoken():
    f = AgentChannelForcer()
    s = f.step(0, 1500)
    assert s.write == PAD and s.to_adapter == PAD


def test_tail_cap_and_text_after_reply():
    f = AgentChannelForcer(tail_max=2)
    run(f, [BOS])
    f.add_reply([2001], done=True)
    steps = run(f, [PAD, 1500, PAD, PAD], start=1)
    assert steps[1].write == PAD and steps[3].write == EOS


def test_forced_stream_drives_adapter(make_rig):
    rig = make_rig(barge_in="thinker")
    v = rig.vocab
    reply = [v.id(" Certo"), v.id(","), v.id(" arriva"), v.id(" domani"), v.id(".")]
    f = AgentChannelForcer()
    model = [PAD, BOS, 1500, 1501, 1502, 1503, 1504, PAD, PAD, PAD, EOS]
    for k, p in enumerate(model):
        if k == 2:
            f.add_reply(reply, done=True)
        s = f.step(k, p)
        rig.adapter.feed(Frame(k, s.to_adapter))
    assert rig.spoken_text() == "Certo, arriva domani."
    assert [c.reason for c in rig.cmds("FinalizeSegment")] == ["burst_commit"]
    assert rig.logs("eos_after_commit")


def test_cut_in_tail_closes_turn():
    f = AgentChannelForcer()
    run(f, [BOS])
    f.add_reply([2001], done=True)
    run(f, [PAD, PAD], start=1)                               # 2001 forced, then TAIL
    assert f.state is ForceState.TAIL
    f.cut()
    assert f.step(3, PAD).write == EOS and f.state is ForceState.LISTEN


def test_controller_owned_eos_right_after_the_forced_text():
    f = AgentChannelForcer(eos_hold=0)
    run(f, [BOS])
    f.add_reply([2001, 2002], done=True)
    steps = run(f, [PAD, PAD, PAD, PAD], start=1)
    assert [s.write for s in steps[:2]] == [2001, 2002]
    assert steps[2].write == EOS                                 # EOS on the first frame after the forced text
    assert f.log[-1][2] == "controller_eos"
