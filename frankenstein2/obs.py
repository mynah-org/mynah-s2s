"""Observability for Frankenstein v2: short log lines, a small per-server SQLite trace store, and a viewer.

The session Tracer (``tracing.py``) emits every event once; this module only listens (``Tracer.sink``), so nothing
here can change what a session does. Three consumers, all optional:

* ``summarize(ev, d)`` -> one short line, or None for events that are not worth a line (per-frame / per-hop noise):

      [3f2a turn=2 +12.48s] asr: "vorrei sapere dove è il mio ordine" (it, endpoint)
      [3f2a turn=2 +12.61s] llm: request r2 (5 messages)
      [3f2a turn=2 call=call1 +14.70s] tool: calling find_customer(full_name="Mario Rossi", postcode="10121")
      [3f2a turn=2 call=call1 +15.02s] tool: find_customer -> completed in 318 ms {"found": true, "order": ...}

  Texts are cut to ``max_chars``; the full texts, arguments and results stay in the store.
* ``TraceStore(path)``: one SQLite file per server (WAL, a writer thread, so the scheduler only enqueues).
  Tables: ``sessions`` (one row per session), ``events`` (every stored event: wall time, monotonic ns, 80 ms step,
  turn / request / call ids, the short line, the full JSON), ``spans`` (derived intervals with a duration: a turn
  from transcript to close, an LLM request from submit to done, a tool call from launch to result, the reply's
  first audio). Per-frame / per-hop events are skipped unless ``level="all"``.
  The span model follows OpenTelemetry's (session = trace, ids, start / end, status, attributes) without its
  dependency; exporting to an OTLP collector is NOT implemented.
* ``python -m frankenstein2.obs DB [--session ID] [--spans] [--last N]``: the timeline of a session (its log
  lines), or its spans with durations.

Privacy: the store keeps what the session said and what the tools returned, for debugging. ``store_text=False`` on
the Tracer (or ``--trace-no-text`` on the server) drops user and agent text before it reaches any sink.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

NOISY = frozenset({"ears.frame", "ears.hop", "audio.release", "asr.piece", "ctl.intent"})
CTL_SHOWN = frozenset({"CANCEL", "CLOSE_TURN", "HOLDING_PHRASE", "BACKCHANNEL"})
log = logging.getLogger("frankenstein2.trace")


def _cut(s, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _args(a: dict | None, n: int) -> str:
    if not isinstance(a, dict):
        return _cut(a, n)
    return _cut(", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items()), n)


def summarize(ev: str, d: dict, max_chars: int = 90) -> str | None:
    """The short line for one event (no prefix), or None."""
    t = lambda k="text": _cut(d.get(k, ""), max_chars)  # noqa: E731
    if ev == "engine.session":
        return f"session: open lang={d.get('language')} voice={d.get('voice') or '-'}"
    if ev == "engine.teardown":
        return f"session: closed after {d.get('steps', 0) * 0.08:.1f} s, {d.get('endpoints', 0)} endpoints, " \
               f"{d.get('barges', 0)} barge-ins"
    if ev == "asr.turn":
        return f"asr: \"{t()}\" ({d.get('reason', '')})"
    if ev == "ears.event":
        kind = d.get("kind")
        return None if kind == "BACKCHANNEL" else f"ears: {kind}" + (f" (by {d['by']})" if d.get("by") else "")
    if ev == "llm.request":
        return f"llm: request r{d.get('request_id')} ({d.get('n_messages', '?')} messages)"
    if ev == "llm.decided":
        return f"llm: r{d.get('request_id')} decided to call a tool"
    if ev == "llm.done":
        info = d.get("info") or {}
        extra = " [skipped]" if info.get("skipped") else ""
        return f"llm: r{d.get('request_id')} done \"{t()}\"{extra}"
    if ev == "llm.stale":
        return f"llm: r{d.get('request_id')} stale"
    if ev == "tool.started":
        return f"tool: calling {d.get('name')}({_args(d.get('args'), max_chars)})"
    if ev == "tool.waiting":
        return f"tool: still running after {d.get('s')} s: {', '.join(d.get('names') or [])}"
    if ev == "tool.completed":
        return f"tool: {d.get('name')} -> {d.get('status', 'completed')} in {d.get('ms', '?')} ms " \
               f"{_cut(d.get('result', ''), max(20, max_chars // 2))}"
    if ev == "tts.segment":
        lane = d.get("lane")
        tag = " (holding)" if d.get("holding") else (" (backchannel)" if lane == "bc" else "")
        return f"tts: \"{t()}\"{tag}"
    if ev == "tts.first_audio":
        return None
    if ev in ("tts.error", "ears.error"):
        return f"{ev.split('.')[0]}: ERROR {_cut(d.get('error', ''), max_chars)}"
    if ev == "audio.flush":
        return f"audio: flushed ({d.get('reason', '')})"
    if ev == "audio.underrun":
        return "audio: underrun"
    if ev == "ctl.action" and d.get("kind") in CTL_SHOWN:
        return f"ctl: {d.get('kind')} ({d.get('reason', '')})"
    if ev == "history.agent" and d.get("cut"):
        return f"history: agent cut at \"{t('heard')}\""
    return None


class LineFormatter:
    """Adds the ``[session turn= call= +t]`` prefix; keeps the per-session start time and current turn."""

    def __init__(self, max_chars: int = 90):
        self.max_chars = max_chars
        self.t0: dict = {}
        self.turn: dict = {}

    def __call__(self, ev: str, d: dict) -> str | None:
        sid = str(d.get("session_id", "-"))
        t0 = self.t0.setdefault(sid, d.get("t_ns", 0))
        if d.get("turn_id") is not None:
            self.turn[sid] = d["turn_id"]
        if ev == "engine.teardown":
            self.t0.pop(sid, None)
        body = summarize(ev, d, self.max_chars)
        if body is None:
            return None
        parts = [sid[-6:]]
        if self.turn.get(sid) is not None:
            parts.append(f"turn={self.turn[sid]}")
        if d.get("call_id"):
            parts.append(f"call={d['call_id']}")
        parts.append(f"+{(d.get('t_ns', t0) - t0) / 1e9:.2f}s")
        return f"[{' '.join(parts)}] {body}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, started_wall REAL, ended_wall REAL, language TEXT,
    voice TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, session_id TEXT, wall REAL, t_ns INTEGER, step INTEGER,
    ev TEXT, turn_id INTEGER, request_id INTEGER, call_id TEXT, line TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS spans (id INTEGER PRIMARY KEY, session_id TEXT, kind TEXT, name TEXT, turn_id INTEGER,
    request_id INTEGER, call_id TEXT, start_wall REAL, end_wall REAL, dur_ms REAL, status TEXT, attrs TEXT);
CREATE INDEX IF NOT EXISTS ev_session ON events(session_id, t_ns);
CREATE INDEX IF NOT EXISTS span_session ON spans(session_id, start_wall);
"""


class TraceStore:
    """SQLite trace store, shared by every session of a server. ``sink(ev, d)`` only enqueues; a writer thread
    commits in batches. ``level``: "normal" (default, no per-frame / per-hop events) or "all"."""

    def __init__(self, path, level: str = "normal", max_chars: int = 90, flush_s: float = 0.5):
        self.path = str(path)
        self.level = level
        self.fmt = LineFormatter(max_chars)
        self.q: queue.Queue = queue.Queue(maxsize=100_000)
        self.dropped = 0
        self.flush_s = flush_s
        self._open: dict = {}                  # (session, kind, key) -> span start
        self._stop = threading.Event()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;" + SCHEMA)
        self._db.commit()
        self._t = threading.Thread(target=self._run, name="f2-trace-store", daemon=True)
        self._t.start()

    # called on the session's thread: never blocks, never raises
    def sink(self, ev: str, d: dict) -> None:
        if self.level != "all" and ev in NOISY:
            return
        try:
            self.q.put_nowait((ev, dict(d), time.time()))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self._stop.is_set() or not self.q.empty():
            batch = []
            try:
                batch.append(self.q.get(timeout=self.flush_s))
                while len(batch) < 2000:
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if batch:
                try:
                    self._write(batch)
                except Exception as e:  # noqa: BLE001 - the store must never take a session down
                    log.warning("trace store write failed: %s", e)

    def _write(self, batch):
        db = self._db
        for ev, d, wall in batch:
            sid = d.get("session_id")
            line = self.fmt(ev, d)
            db.execute("INSERT INTO events (session_id, wall, t_ns, step, ev, turn_id, request_id, call_id, line, "
                       "data) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (sid, wall, d.get("t_ns"), d.get("model_step"), ev, d.get("turn_id"), d.get("request_id"),
                        d.get("call_id"), line, json.dumps(d, ensure_ascii=False, default=str)))
            self._spans(db, ev, d, sid, wall)
        db.commit()

    def _begin(self, sid, kind, key, wall, **attrs):
        self._open[(sid, kind, key)] = (wall, attrs)

    def _end(self, db, sid, kind, key, wall, name=None, status="ok", **attrs):
        st = self._open.pop((sid, kind, key), None)
        if st is None:
            return
        t0, a0 = st
        a = {**a0, **attrs}
        db.execute("INSERT INTO spans (session_id, kind, name, turn_id, request_id, call_id, start_wall, end_wall, "
                   "dur_ms, status, attrs) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, kind, name or a.get("name") or kind, a.get("turn_id"), a.get("request_id"),
                    a.get("call_id"), t0, wall, round((wall - t0) * 1000, 1), status,
                    json.dumps(a, ensure_ascii=False, default=str)))

    def _spans(self, db, ev, d, sid, wall):
        if ev == "engine.session":
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?)",
                       (sid, wall, None, d.get("language"), d.get("voice"), json.dumps(d, default=str)))
        elif ev == "engine.teardown":
            db.execute("UPDATE sessions SET ended_wall=? WHERE session_id=?", (wall, sid))
            for (s, kind, key) in [k for k in self._open if k[0] == sid]:
                self._end(db, s, kind, key, wall, status="unfinished")
        elif ev == "asr.turn":
            self._begin(sid, "turn", d.get("turn_id"), wall, turn_id=d.get("turn_id"), text=d.get("text"))
            self._begin(sid, "first_audio", d.get("turn_id"), wall, turn_id=d.get("turn_id"))
        elif ev == "tts.first_audio":
            self._end(db, sid, "first_audio", self.fmt.turn.get(str(sid)), wall, name="first_audio")
        elif ev == "ctl.action" and d.get("kind") in ("CLOSE_TURN", "CANCEL"):
            self._end(db, sid, "turn", d.get("turn_id"), wall, name="turn",
                      status="ok" if d.get("kind") == "CLOSE_TURN" else f"cancelled:{d.get('reason')}")
            self._end(db, sid, "first_audio", d.get("turn_id"), wall, status="no_audio")
        elif ev == "llm.request":
            self._begin(sid, "llm", d.get("request_id"), wall, request_id=d.get("request_id"),
                        turn_id=d.get("turn_id"), name="llm")
        elif ev in ("llm.done", "llm.stale"):
            self._end(db, sid, "llm", d.get("request_id"), wall, status="ok" if ev == "llm.done" else "stale",
                      text=d.get("text"))
        elif ev == "tool.started":
            self._begin(sid, "tool", d.get("call_id"), wall, call_id=d.get("call_id"), name=d.get("name"),
                        args=d.get("args"), turn_id=d.get("turn_id"), request_id=d.get("request_id"))
        elif ev == "tool.completed":
            self._end(db, sid, "tool", d.get("call_id"), wall, status=d.get("status", "completed"),
                      result=d.get("result"), tool_ms=d.get("ms"))

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._t.join(timeout)
        self._db.close()


class LogSink:
    """Writes the short lines to a logger (``frankenstein2.trace``, INFO)."""

    def __init__(self, logger: logging.Logger | None = None, max_chars: int = 90):
        self.log = logger or log
        self.fmt = LineFormatter(max_chars)

    def sink(self, ev: str, d: dict) -> None:
        if ev in NOISY:
            return
        line = self.fmt(ev, d)
        if line is not None:
            self.log.info(line)


def fanout(*sinks):
    """One Tracer sink calling several; a failing sink is skipped, never propagated."""
    live = [s for s in sinks if s is not None]

    def sink(ev, d):
        for s in live:
            try:
                s(ev, d)
            except Exception:  # noqa: BLE001
                pass
    return sink


# ------------------------------------------------------------------------------------------------- viewer
def show(db_path: str, session: str | None = None, spans: bool = False, last: int = 0, out=print) -> None:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    if session is None:
        rows = db.execute("SELECT session_id, started_wall, ended_wall, language, voice FROM sessions "
                          "ORDER BY started_wall").fetchall()
        rows = rows[-last:] if last else rows
        for sid, t0, t1, lang, voice in rows:
            dur = f"{t1 - t0:.1f} s" if t1 else "open"
            n = db.execute("SELECT COUNT(*) FROM spans WHERE session_id=? AND kind='tool'", (sid,)).fetchone()[0]
            out(f"{sid}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}  {lang}  {voice or '-'}  "
                f"{dur}  tools={n}")
        return
    if spans:
        for kind, name, t0, ms, status, call_id, turn in db.execute(
                "SELECT kind, name, start_wall, dur_ms, status, call_id, turn_id FROM spans WHERE session_id=? "
                "ORDER BY start_wall", (session,)):
            out(f"{time.strftime('%H:%M:%S', time.localtime(t0))}  {kind:<11} {name or '':<20} {ms:>9.1f} ms  "
                f"{status:<12} turn={turn} {call_id or ''}")
        return
    rows = db.execute("SELECT line FROM events WHERE session_id=? AND line IS NOT NULL ORDER BY t_ns",
                      (session,)).fetchall()
    for (line,) in (rows[-last:] if last else rows):
        out(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frankenstein v2 trace store viewer")
    ap.add_argument("db")
    ap.add_argument("--session", help="session id (a unique suffix is enough); without it: list the sessions")
    ap.add_argument("--spans", action="store_true", help="spans with durations instead of the timeline")
    ap.add_argument("--last", type=int, default=0, help="only the last N lines / sessions")
    a = ap.parse_args(argv)
    sid = a.session
    if sid:
        db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        hits = [r[0] for r in db.execute("SELECT session_id FROM sessions WHERE session_id LIKE ?", (f"%{sid}",))]
        sid = hits[0] if len(hits) == 1 else sid
    show(a.db, sid, a.spans, a.last)


if __name__ == "__main__":
    main()
