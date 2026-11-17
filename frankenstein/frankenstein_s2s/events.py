"""Typed inputs and outputs of the talker adapter.

Inputs arrive on the Thinker's 80 ms model clock (frame index ``k``). Outputs are commands for a ``TalkerDriver``
plus audio-side bookkeeping. Every command carries the ``epoch`` it belongs to: a cancel bumps the epoch, and any
audio tagged with an older epoch is stale and must never reach playback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


# ----------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Frame:
    """One Thinker step as seen by the adapter.

    text_id     the id emitted on the agent text channel at this step (None = missing from the trace)
    func_id     the id emitted on the function channel (None or PAD when idle)
    user_speech external evidence that the user is speaking at this frame (VAD / streaming-ASR non-blank)
    """

    k: int
    text_id: int | None
    func_id: int | None = None
    user_speech: bool = False


TOOL_EVENT_KINDS = ("pending", "result", "resumed", "timeout", "cancelled")


@dataclass(frozen=True)
class ToolEvent:
    """Out-of-band tool lifecycle event from the runtime (V2 trace: tool.pending, tool.result.injected, ...)."""

    k: int
    kind: str
    call_id: str = ""

    def __post_init__(self):
        if self.kind not in TOOL_EVENT_KINDS:
            raise ValueError(f"unknown tool event kind {self.kind!r}")


@dataclass(frozen=True)
class TalkerDone:
    """The talker finished a segment (codec EOS or runaway cap) and all its audio was delivered."""

    epoch: int
    turn: int
    segment: int


# ----------------------------------------------------------------------------------------------- commands
@dataclass(frozen=True)
class StartSegment:
    """Open a new talker request: fresh prefill (role + codec prefix + language id + voice + first text).

    One reply turn can span several segments (multi-burst text, long replies). Segments of the same epoch must be
    played in (turn, segment) order; a driver may prefill the next one while the previous is still playing.
    """

    k: int
    epoch: int
    turn: int
    segment: int
    lang: str
    voice: str
    text: str
    qwen_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AppendText:
    """Append committed text (complete words) to the open segment's trailing text."""

    k: int
    epoch: int
    turn: int
    segment: int
    text: str
    qwen_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FinalizeSegment:
    """The segment's text is final: feed ``tts_eos`` after the last text embedding and let the talker drain."""

    k: int
    epoch: int
    turn: int
    segment: int
    reason: str


@dataclass(frozen=True)
class Cancel:
    """Stop stepping every request of ``epoch`` and flush its undelivered / unplayed audio."""

    k: int
    epoch: int
    new_epoch: int
    reason: str


Command = Union[StartSegment, AppendText, FinalizeSegment, Cancel]


# ----------------------------------------------------------------------------------------------- audio side
@dataclass(frozen=True)
class AudioChunk:
    """A piece of talker audio tagged with its origin. ``n_frames`` codec frames of 80 ms each."""

    epoch: int
    turn: int
    segment: int
    n_frames: int
    last: bool = False
    payload: bytes = b""


@dataclass
class LogEntry:
    """Every translation decision that is not a plain pass-through: ambiguous, dropped or synthesized."""

    k: int
    what: str
    action: str
    detail: dict = field(default_factory=dict)
