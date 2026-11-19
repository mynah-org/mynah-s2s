"""Event tracing for Frankenstein v2: one JSONL line per event, {"ev", "t_ns", "session_id", "model_step", ...}
(the V2 / v1 line format; ``model_step`` = the 80 ms step). Names are validated so traces stay greppable.

Families:
  ears.*     ears.hop (per hop: compute ms, in-flight lag), ears.frame (per 80 ms frame: p_end / yield / bc / hazard /
             energy / mode / state), ears.event (ENDPOINT / BARGE / BACKCHANNEL), ears.reset, ears.error
  asr.*      asr.piece (streaming transcript pieces with their frame), asr.turn (the text sent to the LLM),
             asr.drop (speech heard while the agent spoke, not part of a barge-in), asr.second_pass.request /
             asr.second_pass (an uncertain turn transcribed again offline: reason, both texts, which one was used),
             asr.hold / asr.hold_done (an endpoint on a text that wants a continuation waits for more words)
  ctl.*      every controller intent and action (with epoch, turn_id, request_id)
  llm.*      llm.request, llm.text, llm.decided, llm.tool, llm.done, llm.stale
  tool.*     tool.started (name, raw args, call_id), tool.waiting (s, names, silent_steps),
             tool.completed (name, normalised args, raw args, result, call_id, status, ms)
  tts.*      tts.segment (text, speech text, lane, sid), tts.first_audio, tts.done, tts.cancel, tts.error
  audio.*    audio.release (per step with audio: samples, lane), audio.flush, audio.underrun, audio.stale
  history.*  history.agent (heard text, cut flag), history.drop_user
  engine.*   engine.session, engine.mem, engine.teardown
No audio is stored in traces. ``store_text=False`` drops user and agent text fields.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

EVENTS = frozenset({
    "ears.hop", "ears.frame", "ears.event", "ears.reset", "ears.error",
    "asr.piece", "asr.turn", "asr.drop", "asr.second_pass.request", "asr.second_pass", "asr.hold", "asr.hold_done",
    "ctl.intent", "ctl.action",
    "llm.request", "llm.text", "llm.decided", "llm.tool", "llm.done", "llm.stale",
    "tool.started", "tool.completed", "tool.waiting",
    "tts.segment", "tts.first_audio", "tts.done", "tts.cancel", "tts.error",
    "audio.release", "audio.flush", "audio.underrun", "audio.stale",
    "history.agent", "history.drop_user",
    "engine.session", "engine.mem", "engine.teardown",
})
TEXT_FIELDS = ("text", "user_text", "speech", "heard", "pieces", "streaming", "offline")


class Tracer:
    def __init__(self, path=None, session_id=None, clock=time.monotonic_ns, store_text=True, keep=True, sink=None):
        self.session_id, self.clock, self.store_text = session_id, clock, store_text
        self.events = [] if keep else None
        self._fh = open(path, "a", buffering=1) if path else None
        self._lock = threading.Lock()
        self.sink = sink                     # optional callable(name, dict) (e.g. a V2 tracer bridge)

    def emit(self, ev: str, k=None, **fields) -> None:
        if ev not in EVENTS:
            raise ValueError(f"unknown trace event {ev!r}")
        d = {"ev": ev, "t_ns": self.clock()}
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if k is not None:
            d["model_step"] = k
        for key, v in fields.items():
            if v is None or (not self.store_text and key in TEXT_FIELDS):
                continue
            d[key] = list(v) if isinstance(v, tuple) else v
        with self._lock:
            if self.events is not None:
                self.events.append(d)
            if self._fh:
                self._fh.write(json.dumps(d, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        if self.sink is not None:
            try:
                self.sink(ev, d)
            except Exception:  # noqa: BLE001 - a broken listener never breaks a session
                pass

    def of(self, ev: str) -> list:
        return [e for e in (self.events or []) if e["ev"] == ev]

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


class NullTracer(Tracer):
    def __init__(self):
        super().__init__(keep=False)
