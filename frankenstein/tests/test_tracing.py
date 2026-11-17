"""Event traces: expected event sequences per scenario, JSONL format, and the timeline viewer."""

from __future__ import annotations

import json

import pytest

from conftest import Rig
from frankenstein_s2s import tokens as T
from frankenstein_s2s.events import Frame, ToolEvent
from frankenstein_s2s.forcing import AgentChannelForcer
from frankenstein_s2s.trace_view import main as view_main
from frankenstein_s2s.trace_view import render
from frankenstein_s2s.tracing import NullTracer, Tracer, thinker_frame
from frankenstein_s2s.usertext import TranscriptInserter, TurnTaker, UserTextScheduler

REPLY = ["BOS", " Your", " order", " is", " ready", "."]


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        self.t += 1_000_000
        return self.t


def traced_rig(**cfg):
    tr = Tracer(clock=Clock(), session_id="s1")
    rig = Rig(**cfg)
    rig.adapter.tracer = tr
    rig.driver.tracer = tr
    rig.gate.tracer = tr
    return rig, tr


def key_events(tr, keep=("adapter.", "talker.", "audio.", "turn.")):
    skip = {"adapter.commit_word", "talker.append", "audio.delivered", "audio.played"}
    return [e for e in tr.names() if e.startswith(keep) and e not in skip]


def test_unknown_event_rejected():
    with pytest.raises(ValueError):
        NullTracer().emit("adapter.typo")
    with pytest.raises(ValueError):
        Tracer().emit("nope")


def test_burst_drain_sequence_and_both_clocks():
    rig, tr = traced_rig()
    rig.run(REPLY + ["PAD", "PAD", "PAD*3", "EOS"])
    rig.pump()
    assert key_events(tr) == [
        "turn.started", "adapter.turn_open", "adapter.segment_start", "talker.prefill",
        "adapter.finalize", "talker.finalize", "adapter.burst_commit", "adapter.eos_drain", "adapter.turn_close",
        "talker.chunk", "talker.codec_eos", "adapter.talker_done", "audio.playback.started"]
    commits = tr.of("adapter.commit_word")
    assert [c["text"] for c in commits] == ["Your", " order", " is", " ready."]
    assert [c["model_step"] for c in commits] == [2, 3, 4, 5]
    assert all("t_ns" in e and e["session_id"] == "s1" for e in tr.events)
    drain = tr.of("adapter.eos_drain")[0]
    assert drain["turn_id"] == 1 and drain["epoch"] == 0 and drain["pad_tail_frames"] == 4


def test_barge_in_sequence_with_stale_drop():
    rig, tr = traced_rig()
    k = rig.run(REPLY + ["PAD", "PAD"])
    rig.driver.produce()
    in_flight = rig.driver.take()
    rig.run(["PAD*3"], k, speech=frozenset({k, k + 1, k + 2}))
    for c in in_flight:
        rig.gate.accept(c)
    tail = key_events(tr)[key_events(tr).index("adapter.barge_in"):]
    assert tail == ["adapter.barge_in", "adapter.cancel", "talker.cancel", "audio.flush", "adapter.epoch",
                    "adapter.turn_close", "adapter.stale_drop"]
    assert tr.of("adapter.epoch")[0]["epoch"] == 1 and tr.of("adapter.stale_drop")[0]["current_epoch"] == 1


def test_tool_wait_and_eos_ambiguous_events():
    rig, tr = traced_rig(barge_in="thinker")
    k = rig.run(["BOS", " Let", " me", " check", ".", "f:SOTC", "PAD*2", "f:EOTC"])
    rig.adapter.tool(ToolEvent(k, "pending", "c1"))
    k = rig.run(["f:EOTR", "BOS", " Done", " now", "EOS"], k + 1)
    waits = [(e["what"]) for e in tr.of("adapter.tool_wait")]
    assert waits == ["tool_call_start", "eotc", "eotr"]
    amb = tr.of("adapter.eos_ambiguous")
    assert len(amb) == 1 and "natural" in amb[0]["action"]


def test_forcer_and_user_side_events():
    tr = Tracer(clock=Clock())
    f = AgentChannelForcer(tracer=tr)
    f.step(0, T.BOS_ID)
    f.add_reply([2001, 2002], done=True)
    for k, p in enumerate([1500, 2002, 1501, T.EOS_ID], start=1):
        s = f.step(k, p)
        thinker_frame(tr, k, p, s.write if s.write is not None else p, func_id=T.PAD_ID)
    assert tr.names("force.") == ["force.turn_open", "force.override", "force.override", "force.turn_close"]
    assert [e["forced"] for e in tr.of("thinker.frame")] == [True, False, True, False]
    s = UserTextScheduler(lambda w: [1500 + len(w)], tracer=tr)
    tt = TurnTaker(eou_frames=2, bou_frames=1, tracer=tr)
    ins = TranscriptInserter(tracer=tr)
    s.on_close = lambda k, w, ids: ins.on_word(k, w, ids)
    s.on_asr(0, ["▁ciao", "<it-IT>"])
    tt.step(0, True)
    s.flush(1)
    tt.step(1, False)
    tt.step(2, False)
    ins.on_eou(2)
    assert tr.names("asr.") + tr.names("turn.") + tr.names("insert.") == [
        "asr.drop", "asr.word", "turn.bou", "turn.eou", "insert.span"]


def test_jsonl_writer_and_viewer(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    tr = Tracer(path=path, clock=Clock(), session_id="s1", store_text=False)
    rig = Rig()
    rig.adapter.tracer = rig.driver.tracer = rig.gate.tracer = tr
    for f_ in [Frame(0, T.BOS_ID)]:
        thinker_frame(tr, 0, T.BOS_ID, T.BOS_ID)
        rig.adapter.feed(f_)
    rig.run([" Hi", " there", ".", "PAD", "PAD", "EOS"], 1)
    rig.pump()
    tr.close()
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["ev"] == "thinker.frame" and lines[0]["model_step"] == 0
    assert all("text" not in e for e in lines)                      # store_text=False
    out = render(lines)
    assert "=== agent turn 1 (epoch 0) ===" in out and "burst_commit" in out and "codec_eos" in out
    assert view_main([str(path), "--turn", "1"]) == 0
    assert "THINKER" in capsys.readouterr().out
