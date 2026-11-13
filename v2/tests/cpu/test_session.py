"""Session runtime against a deterministic fake engine (no model). These tests check OUR scheduling logic
(buffering, replay, tracing, chunking); they say nothing about model parity."""

import json

import torch

from mynah_v2.model import tokens as T
from mynah_v2.model.reference import FCRecord, StepOutput
from mynah_v2.runtime.session import EnhancedOptions, Mode, Session, iter_chunks
from mynah_v2.tracing.events import Tracer

CS = 2560  # 160 ms @ 16 kHz, as the reference config (chunk_size_in_secs 0.16)


class FakeEngine:
    """2 Thinker steps per chunk. FC 'background' is active for chunks [fc_from, fc_to)."""

    def __init__(self, fc_from=None, fc_to=None, eos_at=2):
        self.chunk_samples = CS
        self.eos_at = eos_at     # chunk on which the fake agent ends its turn (None = keeps speaking)
        self.fc_from, self.fc_to = fc_from, fc_to
        self.k = 0
        self.frame = 0
        self.seen = []           # (k, chunk mean) of every chunk the engine received
        self.opened = None
        self.closed = False

    def open(self, prompt):
        self.opened = prompt

    def fc_active(self):
        return self.fc_from is not None and self.fc_from <= self.k < self.fc_to

    def step(self, pcm, is_last=False):
        self.seen.append(float(pcm.mean()))
        recs = []
        if self.fc_from is not None and self.k == self.fc_from - 1:
            recs.append(FCRecord(1, self.frame, 5, [T.PAD_ID] * 6, [T.SOTC_ID, 1, 2, 3, T.EOTC_ID, T.PAD_ID], None, 0, 1))
        if self.fc_to is not None and self.k == self.fc_to - 1:
            recs.append(FCRecord(2, self.frame + 5, 4, [T.PAD_ID] * 5, [9, 9, 9, T.EOTR_ID, T.PAD_ID], "<TOOL_RESPONSE>[]</TOOL_RESPONSE>", 0, 1))
        self.k += 1
        if not self.fc_active():
            self.frame += 2
        # agent turn: BOS on chunk 1, EOS on chunk eos_at (idle afterwards)
        text = [T.BOS_ID, 30] if self.k == 1 else [T.EOS_ID, T.PAD_ID] if self.k == self.eos_at else [T.PAD_ID, T.PAD_ID]
        return StepOutput(audio=torch.zeros(1764 * 2), text_delta="hi" if self.k == 1 else "", asr_delta="",
                          func_delta="", frame_idx=self.frame, new_text_ids=text, new_func_ids=[T.PAD_ID] * 2,
                          fc_active=self.fc_active(), fc_records=recs, latency_ms=1.0)

    def close(self):
        self.closed = True


def _chunks(n):
    # chunk i has constant value (i+1)/1000 so the engine input order is observable
    return [torch.full((CS,), (i + 1) / 1000) for i in range(n)]


def test_reference_mode_feeds_everything_in_order_and_traces():
    eng, tr = FakeEngine(fc_from=3, fc_to=6), Tracer(None, "s")
    s = Session(eng, tr, Mode.REFERENCE)
    s.open("prompt")
    for c in _chunks(10):
        s.feed_chunk(c)
    s.close()
    assert [round(v, 6) for v in eng.seen] == [round((i + 1) / 1000, 6) for i in range(10)]
    assert s.stats.chunks_buffered == 0 and s.stats.silence_chunks_injected == 0
    names = [e.name for e in tr.events]
    assert names[0] == "session.opened" and names[-1] == "session.closed"
    assert "function.started" in names and "function.arguments.done" in names
    assert "tool.result.injected" in names and "tool.result.consumed" in names
    assert names.count("turn.started") == 1


def test_enhanced_buffers_during_fc_and_replays_in_order():
    eng, tr = FakeEngine(fc_from=3, fc_to=6), Tracer(None, "s")
    s = Session(eng, tr, Mode.ENHANCED, EnhancedOptions(buffer_during_fc=True, replay_after_answer=False))
    s.open(None)
    data = _chunks(10)
    for c in data:
        s.feed_chunk(c)
    s.flush()
    vals = [round(v, 6) for v in eng.seen]
    real = [round(float(c.mean()), 6) for c in data]
    # every real chunk reaches the engine exactly once, in order; FC window got silence instead
    assert [v for v in vals if v != 0.0] == real
    assert vals.count(0.0) == s.stats.silence_chunks_injected == s.stats.chunks_buffered == 3
    assert s.stats.chunks_replayed == 3
    assert [e.name for e in tr.events].count("audio.replayed") == 1


def test_enhanced_buffer_cap_drops_oldest():
    eng = FakeEngine(fc_from=1, fc_to=100)
    s = Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(max_buffer_s=0.32, speech_rms=0.0))  # 2 chunks, all speech
    for c in _chunks(6):
        s.feed_chunk(c)
    assert s.stats.chunks_dropped == 3 and len(s._fc_buffer) == 2


def test_feed_pcm16_chunking_handles_partial_chunks():
    eng = FakeEngine()
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    pcm = (torch.ones(CS * 2 + 100) * 0.5 * 32767).to(torch.int16).numpy().tobytes()
    s.feed_pcm16(pcm[:1000])
    assert eng.k == 0
    s.feed_pcm16(pcm[1000:])
    assert eng.k == 2 and len(s._pending) == 200


def test_iter_chunks_pads_last():
    cs = list(iter_chunks(torch.ones(CS + 10), CS))
    assert len(cs) == 2 and cs[1].numel() == CS and float(cs[1][10:].abs().sum()) == 0.0


def test_trace_file_is_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    tr = Tracer(p, "s9")
    s = Session(FakeEngine(), tr, Mode.REFERENCE)
    s.open(None)
    s.feed_chunk(torch.zeros(CS))
    s.close()
    tr.close()
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert all("t_ns" in r and r["session_id"] == "s9" for r in rows)
    ts = [r["t_ns"] for r in rows]
    assert ts == sorted(ts)


def test_unknown_tool_policy_set_per_mode():
    """The adapter only routes; the answer to an unknown tool is controller policy (ENHANCED only)."""
    import json as _j
    from mynah_v2.agent.controller import Controller
    from mynah_v2.agent.tools import CallStatus, ToolLedger
    eng = FakeEngine()
    eng.unknown_tool_handler = "unset"
    Session(eng, Tracer(None), Mode.REFERENCE, controller=Controller(None)).open(None)
    assert eng.unknown_tool_handler is None                     # reference: nothing injected
    Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(error_on_unknown_tool=False)).open(None)
    assert eng.unknown_tool_handler is None
    Session(eng, Tracer(None), Mode.ENHANCED).open(None)       # no controller: stateless error
    assert _j.loads(eng.unknown_tool_handler('[{"name": "nope", "arguments": {}}]'))["error"] == "unknown_tool"
    led = ToolLedger("s")
    Session(eng, Tracer(None), Mode.ENHANCED, controller=Controller(None, ledger=led)).open(None)
    out = _j.loads(eng.unknown_tool_handler("not json"))
    assert out["error"] == "invalid_call"
    (c,) = led.calls.values()
    assert c.status == CallStatus.FAILED and c.error == "invalid_call"


def test_run_paced_keeps_line_open_after_late_fc():
    from mynah_v2.runtime.session import run_paced
    eng = FakeEngine(fc_from=4, fc_to=30)     # FC starts during input and outlasts input (5) + tail (2 chunks)
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    s._agent_speaking = False
    eng_step = eng.step

    def step(pcm, is_last=False):          # fake agent goes idle (EOS) after the first chunk
        o = eng_step(pcm, is_last)
        o.new_text_ids = [T.PAD_ID, T.PAD_ID] if eng.k > 1 else [T.BOS_ID, T.EOS_ID]
        return o
    eng.step = step
    run_paced(s, torch.zeros(CS * 5), pace="fast", tail_silence_s=0.32, post_fc_s=0.8)
    # FC ends after chunk 30 -> at least 0.8 s (5 chunks) more are fed after it
    assert 30 + 4 <= eng.k <= 30 + 6        # >= 0.8 s after the last FC-active chunk (k=29)


def test_drain_fc_waits_for_a_call_running_past_the_input():
    eng = FakeEngine(fc_from=4, fc_to=12)     # input ends at chunk 5 while the call runs until chunk 12
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    for c in _chunks(5):
        s.feed_chunk(c)
    n = s.drain_fc(post_fc_s=0.48, max_extra_s=30, realtime=False)
    assert not eng.fc_active()
    assert 12 + 2 <= eng.k <= 12 + 4          # call finished, then >= 0.48 s (3 chunks) of line time
    assert n == eng.k - 5
    names = [e.name for e in s.tracer.events]
    assert "tool.result.consumed" in names


def test_drain_fc_is_a_noop_without_tool_calls():
    eng = FakeEngine()
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    for c in _chunks(4):
        s.feed_chunk(c)
    assert s.drain_fc(realtime=False) == 0 and eng.k == 4


def test_drain_fc_is_capped():
    eng = FakeEngine(fc_from=1, fc_to=10_000)  # a call that never ends
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    s.feed_chunk(torch.zeros(CS))
    assert s.drain_fc(max_extra_s=1.6, realtime=False) == 10


def test_release_fc_stops_and_forgets_leftover_background_state():
    """Reference pipeline keeps _fc_async_bg[stream_id] across close/open_session; our close must clear it."""
    import threading
    from mynah_v2.model.reference import ReferenceEngine
    abort, quit_ = threading.Event(), threading.Event()
    th = threading.Thread(target=abort.wait, args=(5,))
    th.start()

    class P:
        _fc_async_bg = {0: {"thread": th, "abort_event": abort, "quit_async_event": quit_}}
        closed = False

        def close_session(self):
            self.closed = True

    eng = object.__new__(ReferenceEngine)
    eng.pipeline, eng.stream_id = P(), 0
    assert eng.fc_active()
    eng.close()
    assert abort.is_set() and quit_.is_set() and not th.is_alive()
    assert not eng.fc_active() and eng.pipeline.closed and eng.fc_released == 1
    eng.close()                               # idempotent: nothing left to release
    assert eng.fc_released == 1


def test_drain_fc_realtime_paces_like_a_live_caller():
    """The reference FC fast path makes generate_step return at once; unpaced silence closed a live call in ~7 s."""
    import time
    eng = FakeEngine(fc_from=1, fc_to=10_000)
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    s.feed_chunk(torch.zeros(CS))
    t0 = time.monotonic()
    assert s.drain_fc(max_extra_s=0.48) == 3
    assert time.monotonic() - t0 >= 2 * 0.16 - 0.01      # 3 chunks -> 2 full chunk intervals of wall time


def test_drain_fc_ends_on_agent_silence_even_without_eos():
    """Measured on the real engine: answer after the call, then PAD forever with no EOS."""
    eng = FakeEngine(fc_from=2, fc_to=6, eos_at=None)          # BOS at chunk 1, never EOS
    s = Session(eng, Tracer(None), Mode.REFERENCE)
    s.open(None)
    for _ in range(3):                                         # call running from chunk 2
        s.feed_chunk(torch.zeros(CS))
    assert s._agent_speaking and eng.fc_active()
    n = s.drain_fc(post_fc_s=0.48, max_extra_s=30, realtime=False)
    assert 0 < n < 30 / 0.16 and 6 + 2 <= eng.k <= 6 + 4


def test_enhanced_buffer_keeps_speech_and_short_pauses_only():
    """Regression (GB10 en_s3_interrupt): a long call + silence filled the 30 s cap and the user's words were dropped."""
    eng = FakeEngine(fc_from=1, fc_to=10_000)
    s = Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(max_buffer_s=3.2, keep_silence_s=0.32))
    s.open(None)
    s.feed_chunk(torch.zeros(CS))                             # before the call
    seq = [0.0] * 5 + [0.1] * 4 + [0.0] * 50 + [0.2] * 3 + [0.0] * 50
    for v in seq:
        s.feed_chunk(torch.full((CS,), v))
    kept = [round(float(c.mean()), 3) for c in s._fc_buffer]
    # leading silence skipped; speech 0.1 x4, 2 pause chunks, speech 0.2 x3, 2 trailing pause chunks
    assert kept == [0.1] * 4 + [0.0] * 2 + [0.2] * 3 + [0.0] * 2
    assert s.stats.chunks_dropped == 0
    assert s.stats.chunks_silence_skipped == 5 + 48 + 48


def test_enhanced_output_gate_mutes_long_pad_only_stretches():
    """Reference EarTTS babbles after long PAD-only text (NVIDIA run(), en_s5 + 60 s); ENHANCED mutes it."""
    eng = FakeEngine()                    # text on chunks 1-2 (BOS/EOS), PAD afterwards; fake audio is zeros
    eng_step = eng.step

    def step(pcm, is_last=False):
        o = eng_step(pcm, is_last)
        o.audio = torch.ones_like(o.audio)          # "babble"
        return o
    eng.step = step
    got = []
    s = Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(mute_idle_s=0.48), on_audio=lambda a: got.append(a))
    s.open(None)
    for _ in range(10):
        s.feed_chunk(torch.zeros(CS))
    loud = [bool(a.abs().sum() > 0) for a in got]
    # last text on chunk 2 -> chunks 3,4,5 within 0.48 s, muted from chunk 6 on
    assert loud == [True] * 5 + [False] * 5
    assert s.stats.audio_muted_samples == 5 * got[0].numel()
    ref = []
    r = Session(eng, Tracer(None), Mode.REFERENCE, on_audio=lambda a: ref.append(a))
    eng.k = 0
    r.open(None)
    for _ in range(10):
        r.feed_chunk(torch.zeros(CS))
    assert all(a.abs().sum() > 0 for a in ref)      # REFERENCE never gates


def test_tts_pad_silence_knob_is_per_session_and_reference_keeps_default():
    calls = []
    eng = FakeEngine()
    eng.set_tts_pad_silence = lambda on: calls.append(on)
    Session(eng, Tracer(None), Mode.REFERENCE, EnhancedOptions(tts_pad_silence=True)).open(None)
    Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(tts_pad_silence=True)).open(None)
    Session(eng, Tracer(None), Mode.ENHANCED).open(None)
    assert calls == [False, True, False]


class AnsweringEngine(FakeEngine):
    """After the call (chunks fc_from..fc_to-1) the agent answers: BOS at chunk fc_to+1, EOS at fc_to+answer."""

    def __init__(self, answer=3, **kw):
        super().__init__(eos_at=None, **kw)
        self.answer = answer

    def step(self, pcm, is_last=False):
        o = super().step(pcm, is_last)
        if self.k == self.fc_to + 1:
            o.new_text_ids = [T.BOS_ID, 30]
        elif self.k == self.fc_to + self.answer:
            o.new_text_ids = [T.EOS_ID, T.PAD_ID]
        elif self.k > 1:
            o.new_text_ids = [T.PAD_ID, T.PAD_ID]
        return o


def test_replay_after_answer_holds_user_audio_until_agent_eos():
    eng, tr = AnsweringEngine(fc_from=3, fc_to=6, answer=3), Tracer(None, "s")
    s = Session(eng, tr, Mode.ENHANCED, EnhancedOptions(speech_rms=0.0, replay_after_answer=True))
    s.open(None)
    data = _chunks(14)
    for c in data:
        s.feed_chunk(c)
    s.flush()
    vals = [round(v, 6) for v in eng.seen]
    real = [round(float(c.mean()), 6) for c in data]
    assert [v for v in vals if v != 0.0] == real                 # nothing lost, order kept
    # FC window (3 chunks) + hold while the agent answers (BOS at k=7 .. EOS at k=9) got silence
    first_replay = vals.index(real[3])                           # chunk 3 = first one buffered (FC from k=3)
    # engine inputs 3-5: FC (silence); 6-8: hold while the agent answers (BOS k=7 .. EOS k=9); replay from 9
    assert all(v == 0.0 for v in vals[3:first_replay]) and first_replay == 9
    reasons = [e.fields["reason"] for e in tr.events if e.name == "audio.buffered"]
    assert "fc_active" in reasons and "hold_answer" in reasons


def test_replay_after_answer_gives_up_when_no_answer_starts():
    eng = AnsweringEngine(fc_from=3, fc_to=6, answer=3)
    eng.step_orig = eng.step
    eng.step = lambda pcm, is_last=False: _no_bos(eng.step_orig(pcm, is_last))
    s = Session(eng, Tracer(None), Mode.ENHANCED,
                EnhancedOptions(speech_rms=0.0, replay_after_answer=True, replay_no_answer_s=0.32))
    s.open(None)
    for c in _chunks(12):
        s.feed_chunk(c)
    assert s.stats.chunks_replayed > 0 and not s._fc_buffer


def _no_bos(o):
    o.new_text_ids = [T.PAD_ID if t == T.BOS_ID else t for t in o.new_text_ids]
    return o


def test_drain_keeps_going_while_user_audio_is_held_for_replay():
    eng = AnsweringEngine(fc_from=3, fc_to=6, answer=30)            # long answer: hold lasts until the cap
    s = Session(eng, Tracer(None), Mode.ENHANCED,
                EnhancedOptions(speech_rms=0.0, replay_after_answer=True, replay_hold_max_s=0.8))
    s.open(None)
    for c in _chunks(8):
        s.feed_chunk(c)
    assert s._fc_buffer                                              # still held
    s.drain_fc(post_fc_s=0.32, max_extra_s=10, realtime=False)
    assert not s._fc_buffer and s.stats.chunks_replayed > 0          # replayed inside the drain, before close


def test_enhanced_outside_fc_feeds_the_engine_exactly_like_reference():
    """#22: outside a tool call ENHANCED changes nothing the model receives (the idle gate only mutes OUTPUT audio),
    so barge-in / turn-taking outside FC is the reference behaviour in both modes."""
    data = _chunks(12)
    ref_eng, enh_eng = FakeEngine(eos_at=None), FakeEngine(eos_at=None)
    for eng, mode, opt in ((ref_eng, Mode.REFERENCE, None), (enh_eng, Mode.ENHANCED, EnhancedOptions(mute_idle_s=0.16))):
        s = Session(eng, Tracer(None), mode, opt)
        s.open("p")
        for c in data:
            s.feed_chunk(c)
        s.flush()
    assert ref_eng.seen == enh_eng.seen
