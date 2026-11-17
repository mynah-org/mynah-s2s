"""Event tracing for the Frankenstein engine: one JSONL line per event, both clocks on every line.

Line format = the V2 trace (``v2/src/mynah_v2/tracing/events.py``): {"ev", "t_ns", "session_id", ...fields}, with
``model_step`` = the 80 ms model frame index k (the V2 field name), plus ``turn_id`` and ``epoch`` where they apply.
V2 event names are reused where the meaning is the same (thinker.step, turn.started, function.*, tool.*, asr.generated,
audio.*); Frankenstein adds the families below. Unknown names are rejected, as in V2, so traces stay greppable.

No audio is ever stored. User text (ASR words) is stored because reading the flow is the point of these traces; a
deployment that must not keep it passes ``store_text=False``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

V2_REUSED = {
    "session.opened", "session.closed", "turn.started", "thinker.step", "text.generated", "asr.generated",
    "function.started", "function.arguments.done", "tool.pending", "tool.running", "tool.completed", "tool.failed",
    "tool.result.injected", "tool.result.consumed", "audio.generated", "audio.playback.started",
}

FRANKENSTEIN = {
    # duplex Thinker, per real frame: predicted id, id fed back, function id
    "thinker.frame",
    # M1-force: content forcing on the agent channel
    "force.turn_open", "force.override", "force.underrun", "force.turn_close", "force.bos_suppressed",
    # M1-ctx: transcript inserted as extra steps
    "insert.span",
    # user side: multilingual streaming ASR and external turn-taking
    "asr.partial", "asr.word", "asr.revised", "asr.final", "asr.drop", "turn.bou", "turn.eou",
    # text-mode brain (same Thinker weights, text path)
    "brain.request", "brain.first_token", "brain.last_token", "brain.text", "brain.tool_decision", "brain.skip",
    "brain.tool_call", "brain.tool_result",
    # adapter (state machine decisions)
    "adapter.turn_open", "adapter.turn_close", "adapter.turn_restart", "adapter.turn_implicit",
    "adapter.commit_word", "adapter.segment_start", "adapter.segment_restart", "adapter.burst_commit",
    "adapter.finalize", "adapter.eos_drain", "adapter.eos_ambiguous", "adapter.barge_in", "adapter.cancel",
    "adapter.epoch", "adapter.tool_wait", "adapter.drop", "adapter.inject", "adapter.retokenize",
    "adapter.stale_drop", "adapter.talker_done",
    # controller (CONTROL.md): every intent and every action
    "ctl.intent", "ctl.action",
    # talker and audio delivery
    "talker.prefill", "talker.append", "talker.step", "talker.block", "talker.finalize", "talker.codec_eos",
    "talker.cancel", "talker.chunk", "audio.delivered", "audio.flush", "audio.underrun", "audio.played",
}

ALL_EVENTS = frozenset(V2_REUSED | FRANKENSTEIN)


class TraceSink(Protocol):
    def emit(self, ev: str, k: Optional[int] = None, **fields: Any) -> None: ...


class NullTracer:
    """Default: does nothing (but still validates names, so a typo fails in tests, not in a run)."""

    def emit(self, ev: str, k: Optional[int] = None, **fields: Any) -> None:
        if ev not in ALL_EVENTS:
            raise ValueError(f"unknown trace event {ev!r}")


class Tracer(NullTracer):
    """In-memory and/or JSONL tracer. ``path=None`` keeps events in memory only (tests)."""

    def __init__(self, path: str | Path | None = None, session_id: str | None = None,
                 clock: Callable[[], int] = time.monotonic_ns, store_text: bool = True, keep: bool = True):
        self.session_id = session_id
        self.clock = clock
        self.store_text = store_text
        self.events: list = [] if keep else None
        self._fh = open(path, "a", buffering=1) if path else None
        self._lock = threading.Lock()

    def emit(self, ev: str, k: Optional[int] = None, **fields: Any) -> None:
        super().emit(ev)
        d = {"ev": ev, "t_ns": self.clock()}
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if k is not None:
            d["model_step"] = k
        for key, v in fields.items():
            if v is None:
                continue
            if not self.store_text and key in ("text", "word", "words"):
                continue
            d[key] = list(v) if isinstance(v, tuple) else v
        with self._lock:
            if self.events is not None:
                self.events.append(d)
            if self._fh:
                self._fh.write(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n")

    def names(self, prefix: str = "") -> list:
        return [e["ev"] for e in (self.events or []) if e["ev"].startswith(prefix)]

    def of(self, ev: str) -> list:
        return [e for e in (self.events or []) if e["ev"] == ev]

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


def thinker_frame(tracer: TraceSink, k: int, predicted: Optional[int], fed: Optional[int],
                  func_id: Optional[int] = None, piece: Optional[str] = None, source: str = "model") -> None:
    """Runner helper: one line per real Thinker frame. ``fed`` = the id actually written back (forced or not)."""
    tracer.emit("thinker.frame", k, predicted=predicted, fed=fed, forced=fed is not None and fed != predicted,
                func_id=func_id, piece=piece, source=source)
