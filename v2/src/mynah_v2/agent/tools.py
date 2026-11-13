"""Tool ledger + tool registry adapter + a deterministic SQLite test tool.

The NVIDIA reference executes a tool as ``handler(arguments) -> str`` with no call id
(streaming_s2s_pipeline.py:767-822). We keep that exact call signature towards the pipeline (REFERENCE mode is
unchanged) and wrap each handler so that every execution gets a call_id, a ledger entry with a lifecycle
(pending -> running -> completed | failed | timed_out | cancelled | stale; late or duplicate results are
recorded, never applied) and trace events. The ledger is application state;
it never touches model state.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from mynah_v2.tracing import events as E


class CallStatus(str, Enum):
    PENDING = "pending"          # call opened (arguments parsed), backend not started yet
    RUNNING = "running"          # backend executing
    COMPLETED = "completed"
    FAILED = "failed"            # backend raised / returned an error
    TIMED_OUT = "timed_out"      # no result within the timeout: an explicit timeout error went to the model
    CANCELLED = "cancelled"      # session closed / call aborted
    STALE = "stale"              # superseded by a newer intent (epoch fence)


OPEN = {CallStatus.PENDING, CallStatus.RUNNING}
TERMINAL = {CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.TIMED_OUT, CallStatus.CANCELLED, CallStatus.STALE}


@dataclass(eq=False)
class ToolCall:
    call_id: str
    name: str
    arguments: dict
    turn_id: int
    intent_id: str | None
    epoch: int                      # execution-version fence: results from an older epoch are stale
    status: CallStatus = CallStatus.PENDING
    result: str | None = None
    error: str | None = None
    t_start_ns: int = 0
    t_end_ns: int | None = None
    model_step: int | None = None
    history: list[tuple[int, str]] = field(default_factory=list)
    late: list[tuple[int, str]] = field(default_factory=list)   # (t_ns, "late"|"duplicate") deliveries after close
    done: threading.Event = field(default_factory=threading.Event, repr=False)


class InvalidTransition(RuntimeError):
    pass


def error_payload(kind: str, message: str) -> str:
    """Tool-error JSON in the reference's shape (streaming_s2s_pipeline.py: timeout -> {"error","message"})."""
    return json.dumps({"error": kind, "message": message})


TIMEOUT_MESSAGE = "I'm sorry, that's taking longer than expected."     # reference text (pipeline timeout branch)


class ToolLedger:
    """Thread-safe record of every tool call in a session: pending -> running -> one terminal state.

    Results reach the ledger through ``deliver``/``deliver_error``, which never raise: a result for a call that is
    already closed (timed out, cancelled, stale) or a second delivery is recorded as ``tool.result.late`` and
    ignored. ``complete``/``fail`` are the strict versions (raise ``InvalidTransition`` on a closed call)."""

    def __init__(self, session_id: str, tracer: E.Tracer | None = None, clock=time.monotonic_ns):
        self.session_id = session_id
        self.tracer = tracer
        self.clock = clock
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self.calls: dict[str, ToolCall] = {}
        self.epoch = 0
        self.on_close: list[Callable[[ToolCall], None]] = []

    def bump_epoch(self) -> int:
        """Invalidate every open call (e.g. the user corrected the request). Returns the new epoch."""
        with self._lock:
            self.epoch += 1
            for c in self.calls.values():
                if c.status in OPEN:
                    self._set(c, CallStatus.STALE, error="superseded")
            return self.epoch

    def open(self, name: str, arguments: dict, turn_id: int, intent_id: str | None = None,
             model_step: int | None = None) -> ToolCall:
        with self._lock:
            cid = f"{self.session_id}-c{next(self._seq)}"
            c = ToolCall(cid, name, arguments, turn_id, intent_id, self.epoch, t_start_ns=self.clock(),
                         model_step=model_step)
            c.history.append((c.t_start_ns, CallStatus.PENDING.value))
            self.calls[cid] = c
        if self.tracer:
            self.tracer.emit(E.TOOL_PENDING, call_id=cid, tool=name, args=arguments, turn_id=turn_id,
                             intent_id=intent_id, epoch=c.epoch, model_step=model_step)
        return c

    def start(self, call_id: str) -> ToolCall:
        with self._lock:
            c = self.calls[call_id]
            if c.status != CallStatus.PENDING:
                raise InvalidTransition(f"{c.call_id}: {c.status.value} -> running")
            c.status = CallStatus.RUNNING
            c.history.append((self.clock(), CallStatus.RUNNING.value))
        if self.tracer:
            self.tracer.emit(E.TOOL_RUNNING, call_id=call_id, tool=c.name, epoch=c.epoch)
        return c

    def _set(self, c: ToolCall, st: CallStatus, result: str | None = None, error: str | None = None) -> None:
        if c.status in TERMINAL:
            raise InvalidTransition(f"{c.call_id}: {c.status.value} -> {st.value}")
        c.status, c.result, c.error = st, result, error
        c.t_end_ns = self.clock()
        c.history.append((c.t_end_ns, st.value))
        c.done.set()
        for fn in self.on_close:
            try:
                fn(c)
            except Exception:  # noqa: BLE001 - listeners never break the ledger
                pass
        if self.tracer:
            name = E.TOOL_COMPLETED if st == CallStatus.COMPLETED else E.TOOL_FAILED
            self.tracer.emit(name, call_id=c.call_id, tool=c.name, status=st.value, error=error,
                             latency_ms=round((c.t_end_ns - c.t_start_ns) / 1e6, 2), epoch=c.epoch)

    def complete(self, call_id: str, result: str) -> ToolCall:
        with self._lock:
            c = self.calls[call_id]
            if c.epoch != self.epoch and c.status in OPEN:
                self._set(c, CallStatus.STALE, result=result, error="epoch fence")
                return c
            self._set(c, CallStatus.COMPLETED, result=result)
            return c

    def fail(self, call_id: str, error: str, status: CallStatus = CallStatus.FAILED) -> ToolCall:
        with self._lock:
            c = self.calls[call_id]
            self._set(c, status, error=error)
            return c

    def _late(self, c: ToolCall, result: str | None, error: str | None) -> None:
        kind = "duplicate" if c.status in (CallStatus.COMPLETED, CallStatus.FAILED) else "late"
        t = self.clock()
        c.late.append((t, kind))
        if self.tracer:
            self.tracer.emit(E.TOOL_RESULT_LATE, call_id=c.call_id, tool=c.name, status=c.status.value, kind=kind,
                             has_result=result is not None, error=error,
                             after_close_ms=round((t - (c.t_end_ns or t)) / 1e6, 2))

    def deliver(self, call_id: str, result: str) -> bool:
        """Backend result. True if it closed the call; False if the call was already closed (late/duplicate)."""
        with self._lock:
            c = self.calls.get(call_id)
            if c is None:
                raise KeyError(call_id)
            if c.status in TERMINAL:
                self._late(c, result, None)
                return False
            if c.epoch != self.epoch:
                self._set(c, CallStatus.STALE, result=result, error="epoch fence")
            else:
                self._set(c, CallStatus.COMPLETED, result=result)
            return True

    def deliver_error(self, call_id: str, error: str) -> bool:
        with self._lock:
            c = self.calls[call_id]
            if c.status in TERMINAL:
                self._late(c, None, error)
                return False
            self._set(c, CallStatus.FAILED, error=error)
            return True

    def time_out(self, call_id: str) -> bool:
        with self._lock:
            c = self.calls[call_id]
            if c.status in TERMINAL:
                return False
            self._set(c, CallStatus.TIMED_OUT, error="timeout")
            return True

    def cancel(self, call_id: str, reason: str = "cancelled") -> bool:
        with self._lock:
            c = self.calls[call_id]
            if c.status in TERMINAL:
                return False
            self._set(c, CallStatus.CANCELLED, error=reason)
            return True

    def cancel_open(self, reason: str) -> list[ToolCall]:
        """Cancel every open call (session close / force-close). Late results are then recorded, never applied."""
        with self._lock:
            out = [c for c in self.calls.values() if c.status in OPEN]
            for c in out:
                self._set(c, CallStatus.CANCELLED, error=reason)
            return out

    def pending(self) -> list[ToolCall]:
        """Open calls (pending or running)."""
        with self._lock:
            return [c for c in self.calls.values() if c.status in OPEN]


Handler = Callable[[dict], str]


def model_response_for(c: ToolCall) -> str:
    """What the model receives as TOOL_RESPONSE for a closed call (reference error shape for every failure)."""
    if c.status == CallStatus.COMPLETED:
        return c.result if c.result is not None else ""
    if c.status == CallStatus.TIMED_OUT:
        return error_payload("timeout", TIMEOUT_MESSAGE)
    if c.status == CallStatus.CANCELLED:
        return error_payload("cancelled", "The request was cancelled.")
    if c.status == CallStatus.STALE:
        return error_payload("superseded", "The request changed before this result arrived; confirm with the user.")
    err = c.error or "error"
    return json.dumps({"error": err.split(": ", 1)[-1]}) if c.status == CallStatus.FAILED else error_payload(
        "error", err)


MAX_RECOMMENDED_TOOLS = 5     # paper §7: "at most 5 tools per session" recommended


class ToolRegistry:
    """Named tools exposed to the model (schemas for the system prompt) and to the reference pipeline."""

    def __init__(self):
        self.handlers: dict[str, Handler] = {}
        self.schemas: list[dict] = []
        self.ack_messages: dict[str, str] = {}

    def register(self, name: str, handler: Handler, description: str, parameters: dict,
                 ack_message: str | None = None) -> None:
        self.handlers[name] = handler
        if len(self.schemas) + 1 > MAX_RECOMMENDED_TOOLS:
            import warnings
            warnings.warn(f"{len(self.schemas) + 1} tools registered; the paper (§7) recommends at most "
                          f"{MAX_RECOMMENDED_TOOLS} per session", stacklevel=2)
        self.schemas.append({"function": {"name": name, "description": description, "parameters": parameters}})
        if ack_message:
            self.ack_messages[name] = ack_message

    def on_hold_messages(self) -> dict[str, list[str]]:
        """Per-tool on-hold messages in the reference format (``s2s.fc_on_hold_messages_path`` JSON:
        {tool_name: phrase | [phrases]}, Layer 1 of the pipeline's on-hold state machine)."""
        return {k: [v] for k, v in self.ack_messages.items()}

    def bind(self, ledger: ToolLedger, turn_id_fn: Callable[[], int], step_fn: Callable[[], int | None],
             timeout_s: float | None = None, intent_fn: Callable[[], str | None] | None = None
             ) -> dict[str, Handler]:
        """Wrap handlers with ledger bookkeeping. Signature towards the pipeline stays ``handler(args) -> str``.

        ``timeout_s=None`` (REFERENCE): the backend runs inline, exceptions become ``{"error": ...}`` (reference
        behaviour, streaming_s2s_pipeline.py:819-822); the pipeline's own timeout applies.
        ``timeout_s`` set (ENHANCED): the backend runs in its own thread and the handler returns after at most
        ``timeout_s`` — with the result, or an explicit timeout error, or a cancellation error if the session closed.
        A result arriving later is recorded as late and never reaches the model or the facts memory. (The reference
        pipeline's own timeout cannot unblock: its ``with ThreadPoolExecutor`` joins the tool thread on exit.)"""
        out: dict[str, Handler] = {}
        for name, fn in self.handlers.items():
            def wrapped(args: dict, _name=name, _fn=fn) -> str:
                a = args if isinstance(args, dict) else {}
                c = ledger.open(_name, a, turn_id_fn(), intent_id=intent_fn() if intent_fn else None,
                                model_step=step_fn())
                if timeout_s is None:
                    ledger.start(c.call_id)
                    try:
                        res = _fn(a)
                    except Exception as e:  # noqa: BLE001 - tool errors are data for the model
                        ledger.deliver_error(c.call_id, f"{type(e).__name__}: {e}")
                        return json.dumps({"error": str(e)})
                    ledger.deliver(c.call_id, res)
                    return res if c.status == CallStatus.COMPLETED else model_response_for(c)

                def run():
                    try:
                        r = _fn(a)
                    except Exception as e:  # noqa: BLE001
                        ledger.deliver_error(c.call_id, f"{type(e).__name__}: {e}")
                    else:
                        ledger.deliver(c.call_id, r)

                ledger.start(c.call_id)
                threading.Thread(target=run, name=f"tool-{c.call_id}", daemon=True).start()
                if not c.done.wait(timeout_s):
                    ledger.time_out(c.call_id)
                return model_response_for(c)
            out[name] = wrapped
        return out


# ---------------------------------------------------------------------------------------------------------------
# Deterministic SQLite order tool (same fixture family as the earlier VoiceChat/MoshiRAG cycle: orders by id).
# ---------------------------------------------------------------------------------------------------------------
ORDERS_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "The order id the user spoke, e.g. ABC456"}},
    "required": ["order_id"],
}

# Same facts as the earlier test fixture so scenario WAVs stay comparable across project cycles;
# dates spelled out (TTS-friendly, model-card advice: "ASCII-only and TTS-friendly" responses).
DEFAULT_ORDERS = [
    ("XYZ123", "shipped", "November 12, 2026", "7319"),
    ("ABC456", "processing", "November 15, 2026", "4402"),
    ("JKL789", "cancelled", "", ""),
]


def make_orders_db(path: str | Path, rows=DEFAULT_ORDERS) -> Path:
    path = Path(path)
    con = sqlite3.connect(path)
    con.execute("create table if not exists orders(order_id text primary key, status text, delivery_date text,"
                " tracking_code text)")
    con.executemany("insert or replace into orders values (?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def normalize_order_id(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def orders_tool(db_path: str | Path, delay_s: float = 0.0) -> Handler:
    def get_order_status(args: dict) -> str:
        if delay_s:
            time.sleep(delay_s)
        oid = normalize_order_id(args.get("order_id", ""))
        con = sqlite3.connect(db_path)
        try:
            row = con.execute("select order_id,status,delivery_date,tracking_code from orders where order_id=?",
                              (oid,)).fetchone()
        finally:
            con.close()
        if row is None:
            return json.dumps({"order_id": oid, "found": False})
        return json.dumps({"order_id": row[0], "found": True, "status": row[1], "delivery_date": row[2],
                           "tracking_code": row[3]})
    return get_order_status
