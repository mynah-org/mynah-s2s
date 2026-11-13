"""Session runtime: PCM in -> engine steps -> PCM/text/events out.

Modes
  REFERENCE  every input chunk goes to the engine as-is, exactly when it arrives. During a reference FC fast path
             the NVIDIA pipeline shows the user's audio to RNNT only, never to the Thinker (P0 §11); we keep that.
  ENHANCED   isolated, individually switchable extensions on top of the same engine:
             * buffer_during_fc: while the FC background thread runs, the engine receives silence chunks (which
               the reference would feed the Thinker as silence anyway, fc_async_use_real_audio=false) and the
               real user audio is buffered; when the FC completes the buffered audio is replayed, so no user
               speech is lost (the reference drops it).
Layering (the project notes §5/§12): this Session is the realtime model scheduler/adapter — frame
timeline, FC barrier, buffering/replay, drain, close of the model stream. Application state (tool calls, ids,
timeouts, facts, history, grounding audit) belongs to the conversation controller (``mynah_v2.agent.controller``),
which the Session only notifies. In REFERENCE the controller is an observer: nothing it does changes what the model
receives; ENHANCED-only policies (rehydration) are gated here by mode.

The engine is anything with ``chunk_samples``, ``open(prompt)``, ``step(chunk, is_last) -> StepOutput``,
``fc_active()`` and ``close()`` (ReferenceEngine, or a fake in CPU tests).
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Protocol

import torch

from mynah_v2.model import tokens as T
from mynah_v2.tracing import events as E


class Mode(str, Enum):
    REFERENCE = "reference"
    ENHANCED = "enhanced"


@dataclass
class EnhancedOptions:
    buffer_during_fc: bool = True
    max_buffer_s: float = 30.0          # hard cap on buffered audio; older audio is dropped (and traced) beyond it
    # Only speech is worth replaying: chunks below speech_rms are skipped, except up to keep_silence_s of silence after
    # speech (keeps pauses between words). Measured on GB10 en_s3_interrupt: the call kept the fast path active for 443
    # chunks; buffering everything let the 30 s cap drop the user's interruption (chunks 82-103) and replay only
    # silence. RMS is a crude VAD (fine for clean/synthetic input; a real VAD belongs here for noisy lines).
    speech_rms: float = 0.003           # ~ -50 dBFS
    # Output gate: outside a tool call, mute agent audio once the text channel has been PAD-only for mute_idle_s.
    # Reference EarTTS can babble after long PAD-only stretches: measured with NVIDIA's own run() on en_s5 + 60 s
    # silence (tokens identical to V2 step(); ASR "you not beat zero integrating indication…" at 27-64 s, stochastic).
    # None disables. FC ack/reminder audio is TTS-only (no text) and is never gated (fc_active).
    # 3 s was too short: speech lags the text stream at sentence ends (GB10 cache-fc en_s2: last text token "six." at
    # step 381, still being spoken 3 s later; the gate cut it — ASR "november 15th 202"). Babble onset measured
    # ~17 s after the last text in NVIDIA run() → 10 s.
    mute_idle_s: float | None = 10.0
    keep_silence_s: float = 0.48
    # Output arbitration for the replay: hold the buffered user audio (and keep buffering new audio) until the agent
    # has finished the answer that follows the tool result (EOS), it has not started one within replay_no_answer_s,
    # or replay_hold_max_s elapsed. Measured on GB10 (B6 rehydrate): replaying at consumption overlapped the start of
    # the answer, the model yielded to the replayed speech and the order facts were never spoken. GPU A/B (C7/C8,
    # 3 scenarios, ASR): with the hold both answers complete, the s3 interruption is still answered, 0 grounding
    # violations → default ON in ENHANCED.
    replay_after_answer: bool = True
    replay_hold_max_s: float = 10.0
    replay_no_answer_s: float = 2.0
    # NVIDIA's own idle-silence substitution in the TTS loop (reference knob, off in the default config; see
    # ReferenceEngine.set_tts_pad_silence). Prevents the babble at the source instead of muting it afterwards.
    tts_pad_silence: bool = False
    error_on_unknown_tool: bool = True  # inject an error TOOL_RESPONSE instead of nothing (reference)
    rehydrate_after_s: float | None = None  # restart the model stream with a facts summary (needs a controller)


class Engine(Protocol):
    chunk_samples: int

    def open(self, system_prompt: str | None) -> None: ...
    def step(self, pcm16k: torch.Tensor, is_last: bool = False): ...
    def fc_active(self) -> bool: ...
    def close(self) -> None: ...


@dataclass
class SessionStats:
    chunks_in: int = 0
    chunks_to_engine: int = 0
    chunks_buffered: int = 0
    chunks_replayed: int = 0
    chunks_dropped: int = 0
    silence_chunks_injected: int = 0
    chunks_silence_skipped: int = 0     # ENHANCED: silent chunks during a call not kept for replay
    audio_muted_samples: int = 0        # ENHANCED: agent audio zeroed by the idle output gate
    calls_cancelled_on_close: int = 0
    step_ms: list[float] = field(default_factory=list)
    # steps that did model work: during the reference FC fast path generate_step returns at once (~0 ms), which
    # made run-level averages look faster than the model is; RTF must be computed on these
    step_ms_model: list[float] = field(default_factory=list)
    audio_out_samples: int = 0


class Session:
    def __init__(self, engine: Engine, tracer: E.Tracer, mode: Mode = Mode.REFERENCE,
                 enhanced: EnhancedOptions | None = None,
                 on_audio: Callable[[torch.Tensor], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None,
                 controller=None):
        self.engine = engine
        self.tracer = tracer
        self.mode = Mode(mode)
        self.enh = enhanced or EnhancedOptions() if self.mode == Mode.ENHANCED else None
        self.on_audio = on_audio
        self.on_event = on_event
        self.stats = SessionStats()
        self.turn_id = 0
        self._agent_speaking = False
        self._pending = bytearray()             # partial chunk (int16 LE) not yet a full engine chunk
        self._fc_buffer: collections.deque[torch.Tensor] = collections.deque()
        self._buf_silence_run = 0               # consecutive silent chunks kept after the last buffered speech
        self._in_fc = False
        self._fc_end_chunk: int | None = None   # chunks_to_engine when the last FC fast path ended
        self._answer_started = False            # agent BOS seen after that
        self._answer_done = False               # agent EOS seen after that BOS
        self._fc_seen = 0
        self.model_step = 0
        self.text_ids: list[int] = []           # per-frame text/function token ids as delivered by the engine
        self.func_ids: list[int] = []
        self.audio_frame = 0
        self.closed = False
        self.text = []
        self.func_text = []
        # application-state authority (observer in REFERENCE; policies such as rehydration only in ENHANCED)
        self.controller = controller
        self.context_chunks = 0                 # chunks fed to the current model stream (since open/rehydrate)
        self.last_fc_chunk = -1                 # chunks_to_engine index of the last step with an FC fast path active
        self.last_text_chunk = -1               # chunks_to_engine index of the last step with non-PAD agent text

    # ------------------------------------------------------------------------------------------- lifecycle
    def open(self, system_prompt: str | None) -> None:
        t0 = time.monotonic_ns()
        if hasattr(self.engine, "set_tts_pad_silence"):   # per session: REFERENCE always the default config
            self.engine.set_tts_pad_silence(bool(self.enh is not None and self.enh.tts_pad_silence))
        if hasattr(self.engine, "unknown_tool_handler"):
            # REFERENCE: None (nothing injected, as NVIDIA). ENHANCED: the controller's policy answers the model.
            handler = None
            if self.enh is not None and self.enh.error_on_unknown_tool:
                from mynah_v2.agent.controller import unknown_tool_error
                handler = self.controller.unknown_tool_response if self.controller is not None else unknown_tool_error
            self.engine.unknown_tool_handler = handler
        if hasattr(self.engine, "multi_call_handler"):
            # REFERENCE: None (first call only, as NVIDIA). ENHANCED with a controller that has the tools: all calls.
            self.engine.multi_call_handler = (self.controller.execute_calls
                                              if self.enh is not None and self.controller is not None
                                              and getattr(self.controller, "tools", None) else None)
        self.engine.open(system_prompt)
        self.tracer.emit(E.SESSION_OPENED, mode=self.mode.value, prompt_chars=len(system_prompt or ""),
                         enhanced=vars(self.enh) if self.enh else None,
                         latency_ms=round((time.monotonic_ns() - t0) / 1e6, 1))

    def rehydrate(self) -> None:
        """ENHANCED: restart the model stream with base prompt + verified facts + recent history."""
        prompt = self.controller.rehydration_prompt()
        t0 = time.monotonic_ns()
        ctx_s = self.context_chunks * self.engine.chunk_samples / T.INPUT_SR
        self.engine.close()
        self.engine.open(prompt)
        self.context_chunks = 0
        self.tracer.emit(E.SESSION_REHYDRATED, context_s=round(ctx_s, 2), prompt_chars=len(prompt),
                         summary=self.controller.summary(), model_step_before=self.model_step,
                         n=self.controller.rehydrations, latency_ms=round((time.monotonic_ns() - t0) / 1e6, 1))

    CLOSE_KINDS = ("normal", "cancel", "force")

    def close(self, kind: str = "normal", join_timeout_s: float | None = None) -> None:
        """End the session. Callers that want the in-flight call answered first run ``drain_fc`` before this.

        kind = "normal" (after a drain), "cancel" (caller hung up, no drain) or "force" (shutdown: short join).
        In every case open tool calls are CANCELLED in the ledger (a later backend result is recorded as late, never
        applied), the reference's FC background state is aborted and released (it would otherwise leak into the
        next session on this stream), and the model stream is closed."""
        if self.closed:
            return
        if kind not in self.CLOSE_KINDS:
            raise ValueError(f"close kind must be one of {self.CLOSE_KINDS}")
        self.closed = True
        t0 = time.monotonic_ns()
        cancelled = self.controller.on_session_closing(kind) if self.controller is not None else []
        self.stats.calls_cancelled_on_close = len(cancelled)
        fc_was_active = bool(self.engine.fc_active())
        released = False
        if hasattr(self.engine, "release_fc"):
            jt = join_timeout_s if join_timeout_s is not None else (2.0 if kind == "force" else 60.0)
            released = bool(self.engine.release_fc(join_timeout_s=jt))
        self.engine.close()
        s = self.stats
        self.tracer.emit(E.SESSION_CLOSED, kind=kind, chunks_in=s.chunks_in, chunks_to_engine=s.chunks_to_engine,
                         buffered=s.chunks_buffered, replayed=s.chunks_replayed, dropped=s.chunks_dropped,
                         silence_injected=s.silence_chunks_injected, model_step=self.model_step,
                         audio_out_s=round(s.audio_out_samples / T.OUTPUT_SR, 3),
                         fc_active_at_close=fc_was_active or None, fc_released=released or None,
                         cancelled_calls=[c.call_id for c in cancelled] or None,
                         grounding_violations=(self.controller.grounding_violations
                                               if self.controller is not None else 0) or None,
                         unanswered_requests=(len(self.controller.requests_outstanding)
                                              if self.controller is not None else 0) or None,
                         close_ms=round((time.monotonic_ns() - t0) / 1e6, 1))

    # ------------------------------------------------------------------------------------------- input
    def feed_pcm16(self, data: bytes, is_last: bool = False) -> None:
        """Append little-endian int16 mono PCM at 16 kHz; runs the engine for every complete chunk."""
        self._pending.extend(data)
        nb = self.engine.chunk_samples * 2
        while len(self._pending) >= nb:
            raw = bytes(self._pending[:nb])
            del self._pending[:nb]
            chunk = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
            self.feed_chunk(chunk, is_last=is_last and len(self._pending) < nb)

    def feed_chunk(self, chunk: torch.Tensor, is_last: bool = False) -> None:
        self.stats.chunks_in += 1
        rms = float(chunk.pow(2).mean().sqrt())
        self.tracer.emit(E.AUDIO_RECEIVED, audio_frame=self.audio_frame, samples=int(chunk.numel()), rms=round(rms, 5))
        self.audio_frame += 1
        if self.mode == Mode.ENHANCED and self.enh.buffer_during_fc and self.engine.fc_active():
            self._buffer(chunk, rms, self.audio_frame - 1)
            self._run(torch.zeros_like(chunk), is_last=False, injected_silence=True)
            return
        if self.mode == Mode.ENHANCED and self._fc_buffer:
            if self._holding_replay():
                self._buffer(chunk, rms, self.audio_frame - 1, reason="hold_answer")
                self._run(torch.zeros_like(chunk), is_last=False, injected_silence=True)
                return
            self._replay()
        self._run(chunk, is_last=is_last)

    def _holding_replay(self) -> bool:
        if not self.enh.replay_after_answer or self._fc_end_chunk is None:
            return False
        dur = self.engine.chunk_samples / T.INPUT_SR
        waited = (self.stats.chunks_to_engine - self._fc_end_chunk) * dur
        if self._answer_done or waited >= self.enh.replay_hold_max_s:
            return False
        if not self._answer_started and waited >= self.enh.replay_no_answer_s:
            return False
        return True

    def _buffer(self, chunk: torch.Tensor, rms: float, audio_frame: int, reason: str = "fc_active") -> None:
        dur = self.engine.chunk_samples / T.INPUT_SR
        if rms >= self.enh.speech_rms:
            self._buf_silence_run = 0
        elif not self._fc_buffer or self._buf_silence_run >= int(round(self.enh.keep_silence_s / dur)):
            self.stats.chunks_silence_skipped += 1          # leading silence, or a long pause: nothing to replay
            return
        else:
            self._buf_silence_run += 1
        self._fc_buffer.append(chunk)
        self.stats.chunks_buffered += 1
        cap = int(self.enh.max_buffer_s / dur)
        dropped = 0
        while len(self._fc_buffer) > cap:
            self._fc_buffer.popleft()
            self.stats.chunks_dropped += 1
            dropped += 1
        self.tracer.emit(E.AUDIO_BUFFERED, buffered=len(self._fc_buffer), reason=reason, audio_frame=audio_frame,
                         rms=round(rms, 5), dropped=dropped or None)

    def _replay(self) -> None:
        n = len(self._fc_buffer)
        self.tracer.emit(E.AUDIO_REPLAYED, chunks=n, seconds=round(n * self.engine.chunk_samples / T.INPUT_SR, 3))
        self._buf_silence_run = 0
        while self._fc_buffer:
            c = self._fc_buffer.popleft()
            self.stats.chunks_replayed += 1
            self._run(c, is_last=False)

    def flush(self, tail_silence_s: float = 0.0) -> None:
        """End of input: replay what is buffered, then optional trailing silence (lets the agent answer)."""
        if self._fc_buffer and not self.engine.fc_active():
            self._replay()
        n = int(round(tail_silence_s * T.INPUT_SR / self.engine.chunk_samples))
        for _ in range(n):
            self.feed_chunk(torch.zeros(self.engine.chunk_samples))
        if self._fc_buffer:
            self._replay()

    def drain_fc(self, post_fc_s: float = 4.0, max_extra_s: float = 90.0, realtime: bool = True) -> int:
        """Keep the line open after a tool call: while an FC fast path is running, or ended less than ``post_fc_s``
        ago, or the agent produced text less than ``post_fc_s`` ago, feed silence (capped by ``max_extra_s``). Agent
        silence is measured on the text channel (PAD only), not by EOS: on GB10 the model answered after a call and
        then emitted PAD for 700+ steps without EOS, so an EOS-based wait ran to the cap. A live caller waits on
        the line; closing earlier would cut the answer and, in the reference pipeline, leave the FC background state
        behind. Returns the number of silence chunks fed.
        ``realtime``: at most one chunk per chunk duration of wall time, like a live caller. Required for the reference
        engine: while its non-blocking FC fast path runs, ``generate_step`` returns at once (the background thread
        advances at its own pace), so unpaced silence would exhaust the cap in seconds and close mid-call (measured:
        562 chunks in ~7 s on the GB10 WebSocket demo)."""
        cs = self.engine.chunk_samples
        dur = cs / T.INPUT_SR
        n, cap, last = 0, int(max_extra_s / dur), time.monotonic()
        while n < cap:
            if self.engine.fc_active() or self._fc_buffer:
                pass                        # call running, or user audio still held/buffered for replay
            elif self.last_fc_chunk < 0:
                break
            elif (self.stats.chunks_to_engine - max(self.last_fc_chunk, self.last_text_chunk)) * dur >= post_fc_s:
                break                       # call over and the agent has been silent (PAD only) for post_fc_s
            if realtime:
                wait = last + dur - time.monotonic() if n else 0.0
                if wait > 0:
                    time.sleep(wait)
                last = time.monotonic()
            self.feed_chunk(torch.zeros(cs))
            n += 1
        return n

    # ------------------------------------------------------------------------------------------- engine
    def _maybe_rehydrate(self) -> None:
        if self.mode != Mode.ENHANCED or self.controller is None or not self.enh or not self.enh.rehydrate_after_s:
            return
        self.controller.max_context_s = self.enh.rehydrate_after_s
        ctx_s = self.context_chunks * self.engine.chunk_samples / T.INPUT_SR
        if self.controller.needs_rehydration(ctx_s, agent_idle=not self._agent_speaking,
                                             fc_active=self.engine.fc_active()):
            self.rehydrate()

    def _run(self, chunk: torch.Tensor, is_last: bool, injected_silence: bool = False) -> None:
        self._maybe_rehydrate()
        self.stats.chunks_to_engine += 1
        self.context_chunks += 1
        if injected_silence:
            self.stats.silence_chunks_injected += 1
        out = self.engine.step(chunk, is_last=is_last)
        self.stats.step_ms.append(out.latency_ms)
        if not (out.fc_active or self._in_fc):
            self.stats.step_ms_model.append(out.latency_ms)
        if out.fc_active or self.engine.fc_active():
            self.last_fc_chunk = self.stats.chunks_to_engine
        prev = self.model_step
        self.model_step = out.frame_idx
        self.text_ids += out.new_text_ids
        self.func_ids += out.new_func_ids
        self.tracer.emit(E.THINKER_STEP, model_step=out.frame_idx, steps=out.frame_idx - prev,
                         latency_ms=round(out.latency_ms, 2), fc_active=out.fc_active,
                         text_ids=out.new_text_ids or None, func_ids=[i for i in out.new_func_ids if i != T.PAD_ID] or None,
                         silence_in=injected_silence or None)
        if any(t != T.PAD_ID for t in out.new_text_ids):
            self.last_text_chunk = self.stats.chunks_to_engine
        for tid in out.new_text_ids:
            if tid == T.BOS_ID and self._fc_end_chunk is not None and not self._in_fc:
                self._answer_started = True
            elif tid == T.EOS_ID and self._answer_started:
                self._answer_done = True
            if tid == T.BOS_ID and not self._agent_speaking:
                self._agent_speaking = True
                self.turn_id += 1
                self.tracer.emit(E.TURN_STARTED, turn_id=self.turn_id, model_step=out.frame_idx, who="agent")
            elif tid == T.EOS_ID:
                self._agent_speaking = False
        # FC records first: a response consumed (EOTR) earlier in this chunk must be known before this chunk's text
        for rec in out.fc_records:
            self._trace_fc(rec)
        if out.text_delta:
            self.text.append(out.text_delta)
            if self.controller:
                self.controller.on_agent_text(out.text_delta, model_step=out.frame_idx, turn_id=self.turn_id)
            self.tracer.emit(E.TEXT_GENERATED, model_step=out.frame_idx, turn_id=self.turn_id, text=out.text_delta)
        if out.asr_delta and out.asr_delta.strip():
            self.tracer.emit(E.ASR_GENERATED, model_step=out.frame_idx, text=out.asr_delta)
            if self.controller:
                self.controller.on_user_text(out.asr_delta)
        if out.func_delta:
            self.func_text.append(out.func_delta)
        if out.fc_active and not self._in_fc:
            self._in_fc = True
            self.tracer.emit(E.FUNCTION_STARTED, model_step=out.frame_idx, turn_id=self.turn_id)
        elif not out.fc_active and self._in_fc:
            self._in_fc = False
            self._fc_end_chunk = self.stats.chunks_to_engine
            self._answer_started = self._answer_done = False
        if out.audio.numel():
            audio = out.audio
            self.stats.audio_out_samples += int(audio.numel())
            muted = self._gate_idle(out)
            if muted:
                self.stats.audio_muted_samples += int(audio.numel())
                audio = torch.zeros_like(audio)
            self.tracer.emit(E.AUDIO_GENERATED, model_step=out.frame_idx, samples=int(audio.numel()),
                             muted=True if muted else None)
            if self.on_audio:
                self.on_audio(audio)

    def _gate_idle(self, out) -> bool:
        if self.enh is None or self.enh.mute_idle_s is None or self.last_text_chunk < 0:
            return False
        if out.fc_active or self.engine.fc_active():
            return False
        dur = self.engine.chunk_samples / T.INPUT_SR
        return (self.stats.chunks_to_engine - self.last_text_chunk) * dur > self.enh.mute_idle_s

    def _trace_fc(self, rec) -> None:
        f = [i for i in rec.func if i != T.PAD_ID]
        if rec.phase == 1:
            if self.controller is not None and T.EOTC_ID in rec.func:
                self.controller.on_call_requested(rec.start_frame + rec.func.index(T.EOTC_ID))
            self.tracer.emit(E.FUNCTION_ARGS_DONE, model_step=rec.start_frame, steps=rec.steps, func_ids=f,
                             turn_id=self.turn_id, latency_ms=round((rec.t_end_ns - rec.t_start_ns) / 1e6, 1))
        else:
            self.tracer.emit(E.TOOL_RESULT_INJECTED, model_step=rec.start_frame, steps=rec.steps,
                             chars=len(rec.tool_response_text or ""), turn_id=self.turn_id,
                             latency_ms=round((rec.t_end_ns - rec.t_start_ns) / 1e6, 1))
            if T.EOTR_ID in rec.func:
                step = rec.start_frame + rec.func.index(T.EOTR_ID)
                call_id = (self.controller.on_result_consumed(rec.tool_response_text, step)
                           if self.controller is not None else None)
                self.tracer.emit(E.TOOL_RESULT_CONSUMED, model_step=step, turn_id=self.turn_id, call_id=call_id)


def iter_chunks(wav: torch.Tensor, chunk_samples: int) -> Iterable[torch.Tensor]:
    n = wav.numel()
    for i in range(0, n, chunk_samples):
        c = wav[i:i + chunk_samples]
        if c.numel() < chunk_samples:
            c = torch.nn.functional.pad(c, (0, chunk_samples - c.numel()))
        yield c


def run_paced(session: Session, wav: torch.Tensor, pace: str = "realtime", tail_silence_s: float = 4.0,
              post_fc_s: float | None = None, max_extra_s: float = 90.0, drain: bool = True) -> None:
    """Drive a session from a 16 kHz mono waveform, then trailing silence.
    pace='realtime': chunk k is sent at t0 + k*chunk_dur, or immediately if the engine is behind (live client with
    backlog); pace='fast': no sleeping.
    A live user keeps the line open while the agent answers. If an FC fast path is still running (or finished)
    near the end of the input, we keep sending silence until it completes and then at least ``post_fc_s``
    (default = tail_silence_s) more, extended while the agent is still speaking, capped by ``max_extra_s``."""
    cs = session.engine.chunk_samples
    dur = cs / T.INPUT_SR
    post_fc_s = tail_silence_s if post_fc_s is None else post_fc_s
    chunks = list(iter_chunks(wav, cs)) + [torch.zeros(cs)] * int(round(tail_silence_s / dur))
    t0 = time.monotonic()
    k = 0

    def send(c):
        nonlocal k
        if pace == "realtime":
            wait = t0 + k * dur - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        session.feed_chunk(c)
        k += 1

    for c in chunks:
        send(c)
    if drain:   # drain=False: the caller hangs up right after its input (close kind cancel/force)
        session.drain_fc(post_fc_s=post_fc_s, max_extra_s=max_extra_s, realtime=(pace == "realtime"))
    session.flush(0.0)
