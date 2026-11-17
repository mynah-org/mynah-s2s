"""Talker side: the driver protocol, a fake driver for tests, the playback gate, and the Flash driver stub.

The real driver (``FlashTalkerDriver``) is NOT implemented here. Its contract follows the transplant harness design
(frame-by-frame Qwen3-TTS talker on the official ``qwen_tts`` package with the Sophon Flash fine-tuned talker
weights):

* ``StartSegment``  -> new request, fresh prefill: role tokens + [tts_pad x5, tts_bos] + codec[think, think_bos,
  language id, think_eos, speaker x-vector, pad] + first text id + codec_bos. No acoustic state carried over.
* ``AppendText``    -> Qwen ids appended to the request's trailing text; one text embedding consumed per 80 ms codec
  frame.
* text underrun     -> BLOCK: no talker step until the next text id or ``tts_eos`` (state kept bit-exactly). Never
  feed ``tts_pad`` mid-sentence (it destroys content); ``tts_pad`` only after ``tts_eos`` (Qwen's own layout).
* ``FinalizeSegment`` -> append ``tts_eos``, unmask codec EOS, let the model end with its learned codec EOS
  (runaway guard: 12 frames per text token + 60).
* ``Cancel``        -> stop stepping every request of that epoch, drop undelivered chunks; audio already handed to
  the transport is dropped by the ``PlaybackGate`` (epoch check).
* codec chunk ramp 2, 4, 8, 12, 16, 25 frames with 25 frames of left context; 24 kHz output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .events import AppendText, AudioChunk, Cancel, Command, FinalizeSegment, StartSegment, TalkerDone
from .tracing import NullTracer


@runtime_checkable
class TalkerDriver(Protocol):
    """Receives adapter commands in order. Must be non-blocking: heavy work happens on the driver's own thread."""

    def submit(self, cmd: Command) -> None: ...


class FlashTalkerDriver:
    """Interface stub for the Sophon Flash / Qwen3-TTS-12Hz-1.7B talker. See the module docstring for the contract.

    Not implemented on purpose: model loading and GPU stepping belong to a later stage (DESIGN.md, E2/E3)."""

    def __init__(self, talker_dir: str, voice: str, device: str = "cuda", mode: str = "block"):
        if mode not in ("block", "pad"):
            raise ValueError("mode must be 'block' (default, state-preserving) or 'pad' (diagnostic only)")
        self.talker_dir, self.voice, self.device, self.mode = talker_dir, voice, device, mode

    def submit(self, cmd: Command) -> None:  # pragma: no cover - stub
        raise NotImplementedError("FlashTalkerDriver is a design stub; see DESIGN.md 'Staged plan'")


@dataclass
class _Seg:
    epoch: int
    turn: int
    segment: int
    words: int = 0
    emitted_words: int = 0
    final: bool = False
    done: bool = False


@dataclass
class FakeTalkerDriver:
    """Deterministic stand-in: every committed word becomes ``frames_per_word`` codec frames.

    ``produce()`` turns pending words into AudioChunks (what a real talker would have generated so far). Chunks that
    were produced but not yet consumed by the caller stay in ``outbox`` even after a Cancel: this simulates audio
    already in flight, which the PlaybackGate must drop.
    """

    frames_per_word: int = 3
    commands: list = field(default_factory=list)
    outbox: deque = field(default_factory=deque)
    done: list = field(default_factory=list)
    segs: dict = field(default_factory=dict)
    cancelled_epochs: set = field(default_factory=set)
    tracer: object = field(default_factory=NullTracer)

    def submit(self, cmd: Command) -> None:
        self.commands.append(cmd)
        tr = self.tracer
        if isinstance(cmd, StartSegment):
            key = (cmd.epoch, cmd.turn, cmd.segment)
            if key in self.segs:
                raise AssertionError(f"segment opened twice: {key}")
            self.segs[key] = _Seg(cmd.epoch, cmd.turn, cmd.segment, words=len(cmd.text.split()))
            tr.emit("talker.prefill", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch, lang=cmd.lang,
                    voice=cmd.voice, n_text_ids=len(cmd.qwen_ids))
        elif isinstance(cmd, AppendText):
            seg = self._open(cmd.epoch, cmd.turn, cmd.segment)
            seg.words += len(cmd.text.split())
            tr.emit("talker.append", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch,
                    n_text_ids=len(cmd.qwen_ids))
        elif isinstance(cmd, FinalizeSegment):
            seg = self._open(cmd.epoch, cmd.turn, cmd.segment)
            seg.final = True
            tr.emit("talker.finalize", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch)
        elif isinstance(cmd, Cancel):
            self.cancelled_epochs.add(cmd.epoch)
            tr.emit("talker.cancel", cmd.k, epoch=cmd.epoch, reason=cmd.reason)
            for s in self.segs.values():
                if s.epoch == cmd.epoch:
                    s.done = True

    def _open(self, epoch, turn, segment) -> _Seg:
        seg = self.segs.get((epoch, turn, segment))
        if seg is None:
            raise AssertionError(f"command for a segment that was never started: {(epoch, turn, segment)}")
        if seg.final:
            raise AssertionError(f"command after FinalizeSegment: {(epoch, turn, segment)}")
        if seg.done:
            raise AssertionError(f"command for a cancelled segment: {(epoch, turn, segment)}")
        return seg

    def produce(self) -> list:
        """Generate audio for every pending word of live segments, in (epoch, turn, segment) order."""
        made = []
        for key in sorted(self.segs):
            s = self.segs[key]
            if s.done:
                continue
            n = s.words - s.emitted_words
            last = s.final
            if n > 0 or last:
                s.emitted_words = s.words
                chunk = AudioChunk(s.epoch, s.turn, s.segment, n * self.frames_per_word, last=last)
                made.append(chunk)
                self.tracer.emit("talker.chunk", None, turn_id=s.turn, segment=s.segment, epoch=s.epoch,
                                 frames=chunk.n_frames, last=last)
                if last:
                    s.done = True
                    self.done.append(TalkerDone(s.epoch, s.turn, s.segment))
                    self.tracer.emit("talker.codec_eos", None, turn_id=s.turn, segment=s.segment, epoch=s.epoch)
        self.outbox.extend(made)
        return made

    def take(self) -> list:
        out = list(self.outbox)
        self.outbox.clear()
        return out


@dataclass
class PlaybackGate:
    """Last line of defence against stale audio: only chunks of the current epoch are queued / played.

    ``observe`` must see every adapter command (the adapter does this for its sinks)."""

    epoch: int = 0
    queue: deque = field(default_factory=deque)
    played_frames: int = 0
    dropped_stale: int = 0
    flushed_frames: int = 0
    played: list = field(default_factory=list)
    tracer: object = field(default_factory=NullTracer)

    def submit(self, cmd: Command) -> None:  # TalkerDriver-compatible sink
        if isinstance(cmd, Cancel):
            self.epoch = cmd.new_epoch
            n = sum(c.n_frames for c in self.queue)
            self.flushed_frames += n
            self.queue.clear()
            self.tracer.emit("audio.flush", cmd.k, epoch=cmd.epoch, new_epoch=cmd.new_epoch, frames=n)

    def accept(self, chunk: AudioChunk) -> bool:
        if chunk.epoch != self.epoch:
            self.dropped_stale += 1
            self.tracer.emit("adapter.stale_drop", None, turn_id=chunk.turn, segment=chunk.segment,
                             epoch=chunk.epoch, current_epoch=self.epoch, frames=chunk.n_frames)
            return False
        self.queue.append(chunk)
        self.tracer.emit("audio.delivered", None, turn_id=chunk.turn, segment=chunk.segment, epoch=chunk.epoch,
                         frames=chunk.n_frames)
        return True

    def play_all(self) -> int:
        n = 0
        while self.queue:
            c = self.queue.popleft()
            if not self.played:
                self.tracer.emit("audio.playback.started", None, turn_id=c.turn, epoch=c.epoch)
            self.played.append(c)
            n += c.n_frames
            self.tracer.emit("audio.played", None, turn_id=c.turn, segment=c.segment, epoch=c.epoch,
                             frames=c.n_frames)
        self.played_frames += n
        return n
