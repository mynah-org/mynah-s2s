"""Structured event trace (JSONL). The trace, not free-text logs, is the source of truth for V2 runs.

Every event carries a monotonic timestamp (ns) plus whatever identifiers apply: session_id, turn_id, call_id,
model step (Thinker frame index), audio frame index, token ids, stage latency. No audio and no user text is
stored unless the caller passes it explicitly (``store_text``).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# Canonical event names (the steer's minimum set + a few runtime-specific ones).
AUDIO_RECEIVED = "audio.received"
AUDIO_ENCODED = "audio.encoded"
THINKER_STEP = "thinker.step"
TEXT_GENERATED = "text.generated"
ASR_GENERATED = "asr.generated"
FUNCTION_STARTED = "function.started"
FUNCTION_ARGS_DONE = "function.arguments.done"
TOOL_PENDING = "tool.pending"
TOOL_RUNNING = "tool.running"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"
TOOL_MULTI_CALL = "tool.multi_call"               # ENHANCED: several calls in one call text
TOOL_RESULT_LATE = "tool.result.late"            # result after timeout/cancel/stale, or a 2nd delivery
TOOL_RESULT_INJECTED = "tool.result.injected"
TOOL_RESULT_CONSUMED = "tool.result.consumed"
AUDIO_GENERATED = "audio.generated"
AUDIO_PLAYBACK_STARTED = "audio.playback.started"
AUDIO_BUFFERED = "audio.buffered"
AUDIO_REPLAYED = "audio.replayed"
SESSION_OPENED = "session.opened"
SESSION_CLOSED = "session.closed"
SESSION_REHYDRATED = "session.rehydrated"
TURN_STARTED = "turn.started"
GROUNDING_VIOLATION = "grounding.violation"      # agent text while a tool call is still open

ALL_EVENTS = frozenset(v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, str) and "." in v)


@dataclass
class Event:
    name: str
    t_ns: int
    session_id: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"ev": self.name, "t_ns": self.t_ns}
        if self.session_id is not None:
            d["session_id"] = self.session_id
        d.update(self.fields)
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


class Tracer:
    """Thread-safe JSONL tracer. ``path=None`` keeps events in memory only (tests)."""

    def __init__(self, path: str | Path | None = None, session_id: str | None = None,
                 clock: Callable[[], int] = time.monotonic_ns, keep: bool = True):
        self.session_id = session_id
        self.clock = clock
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1) if path else None
        self.events: list[Event] = [] if keep else None  # type: ignore[assignment]
        self.t0 = clock()
        self._subs: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        """Live listener (e.g. the WebSocket transport). Called synchronously, outside the lock."""
        self._subs.append(fn)

    def emit(self, name: str, **fields: Any) -> Event:
        if name not in ALL_EVENTS:
            raise ValueError(f"unknown event name {name!r}")
        ev = Event(name=name, t_ns=self.clock(), session_id=self.session_id,
                   fields={k: v for k, v in fields.items() if v is not None})
        with self._lock:
            if self.events is not None:
                self.events.append(ev)
            if self._fh:
                self._fh.write(ev.to_json() + "\n")
        for fn in self._subs:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001 - a broken listener must not break the session
                pass
        return ev

    def of(self, name: str) -> list[Event]:
        return [e for e in (self.events or []) if e.name == name]

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


def read_trace(path: str | Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
