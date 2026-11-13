"""Conversation controller: the authority on application state (never model state).

Owns the tool ledger (session/turn/intent/call ids, call lifecycle, timeouts, cancellation, late results), the
verified-facts memory, the application history and the grounding audit. The realtime scheduler (``runtime.session``)
notifies it (agent text, user text, result consumed, session closing); the model never sees controller state except
through mechanisms the model supports (a TOOL_RESPONSE for an open call, or a new session prompt).
In REFERENCE mode it is an observer (bookkeeping + audit only); ENHANCED policies (rehydration) are enabled by the
session according to the mode.

* FactsMemory: verified facts = completed tool results with provenance (call_id, tool, args, time). SQLite file or
  in-memory. Only ledger-COMPLETED, non-stale results become facts (stale / timed-out / failed never do).
* History: application-level conversation log (agent text from the model's text channel, user text from RNN-T);
  it is NOT the model's KV/Mamba state and is never fed back except through an explicit rehydration prompt.
* Rehydration: the paper trains on ~2 min of audio context (§7); runtimes allow far more but beyond 2 min is out of
  distribution. ``needs_rehydration`` fires once the session has consumed ``max_context_s`` of audio and the agent
  is idle with no FC running; ``rehydration_prompt`` = base system prompt + a compact facts/history summary. The
  session then restarts the model stream with that prompt (see Session.rehydrate). This is our extension, not a
  reference mechanism.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from mynah_v2.agent.tools import CallStatus, ToolCall, ToolLedger
from mynah_v2.tracing import events as E


@dataclass
class Fact:
    call_id: str
    tool: str
    args: dict
    result: dict | str
    t_ns: int


class FactsMemory:
    def __init__(self, path: str | Path | None = None):
        self._lock = threading.Lock()
        self.con = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self.con.execute("create table if not exists facts(call_id text primary key, tool text, args text,"
                         " result text, t_ns integer)")

    def add_from_call(self, c: ToolCall) -> bool:
        if c.status != CallStatus.COMPLETED or c.result is None:
            return False
        try:
            res = json.loads(c.result)
        except (json.JSONDecodeError, TypeError):
            res = c.result
        if isinstance(res, dict) and "error" in res:
            return False
        with self._lock:
            self.con.execute("insert or replace into facts values (?,?,?,?,?)",
                             (c.call_id, c.name, json.dumps(c.arguments, sort_keys=True), json.dumps(res),
                              c.t_end_ns or time.monotonic_ns()))
            self.con.commit()
        return True

    def all(self) -> list[Fact]:
        with self._lock:
            rows = self.con.execute("select call_id,tool,args,result,t_ns from facts order by t_ns").fetchall()
        return [Fact(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4]) for r in rows]

    def latest_by_key(self) -> list[Fact]:
        """One fact per (tool, args): the most recent result wins (a later lookup supersedes an older one)."""
        seen: dict[tuple, Fact] = {}
        for f in self.all():
            seen[(f.tool, json.dumps(f.args, sort_keys=True))] = f
        return sorted(seen.values(), key=lambda f: f.t_ns)


@dataclass
class Utterance:
    role: str        # "agent" | "user"
    text: str
    t_ns: int


def unknown_tool_error(call_text: str) -> str:
    """TOOL_RESPONSE for a call the registry cannot execute (same error shape as the reference timeout)."""
    from mynah_v2.model import tokens as T
    parsed = T.parse_call_text(call_text)
    ok = bool(parsed) and parsed[0].ok
    name = parsed[0].name if ok else None
    return json.dumps({"error": "unknown_tool" if ok else "invalid_call",
                       "message": f"The tool {name or 'call'} is not available."})


class Controller:
    def __init__(self, base_prompt: str | None, memory: FactsMemory | None = None, max_context_s: float = 100.0,
                 history_chars: int = 600, prompt_fn=None, ledger: ToolLedger | None = None,
                 tracer: E.Tracer | None = None, tools: dict | None = None, call_timeout_s: float = 8.0):
        """prompt_fn(extra_text) -> full system prompt; default appends extra_text to base_prompt. Use it to put
        the summary inside the system message of the NVIDIA FC template rather than after the tool list."""
        self.ledger = ledger
        self.tools = tools or {}            # bound handlers (ledger-wrapped) by name, for multi-call execution
        self.call_timeout_s = call_timeout_s
        self.tracer = tracer if tracer is not None else (ledger.tracer if ledger is not None else None)
        self.grounding_violations = 0
        self.consumed: dict[str, int] = {}          # call_id -> model step of EOTR (result actually consumed)
        # model-side view: calls the model REQUESTED (EOTC seen) whose response it has not consumed yet. Independent
        # of the ledger: a call with no handler (reference unknown tool) never reaches the ledger but is still open
        # for the model, which may then answer from nothing (measured on GPU: invented delivery date).
        self.requests_outstanding: list[int] = []   # model steps of EOTC
        self.base_prompt = base_prompt or ""
        self.prompt_fn = prompt_fn
        self.memory = memory or FactsMemory()
        self.max_context_s = max_context_s
        self.history_chars = history_chars
        self.history: list[Utterance] = []
        # full duplex: user ASR and agent text deltas arrive interleaved; aggregate them per turn (user text is closed
        # when the agent opens a turn "<s>", agent text when it closes it "</s>")
        self._pending = {"user": "", "agent": ""}
        self.rehydrations = 0
        if ledger is not None:
            ledger.on_close.append(self.on_tool_call_closed)

    # ---- inputs from the session
    def on_agent_text(self, text: str, model_step: int | None = None, turn_id: int | None = None) -> None:
        """Agent words from the model's text channel. Grounding audit: words while a tool call is still open
        (backend not answered) are recorded as violations — facts may only follow a consumed result."""
        if text.strip():
            open_calls = self.ledger.pending() if self.ledger is not None else []
            pending_req = list(self.requests_outstanding)
            if open_calls or pending_req:
                self.grounding_violations += 1
                if self.tracer:
                    self.tracer.emit(E.GROUNDING_VIOLATION, model_step=model_step, turn_id=turn_id,
                                     call_ids=[c.call_id for c in open_calls] or None,
                                     unanswered_requests=pending_req or None, text=text)
        self._turn_text("agent", text)

    def on_user_text(self, text: str) -> None:
        self._turn_text("user", text)

    def _turn_text(self, role: str, text: str) -> None:
        if role == "user":
            self._pending["user"] += text
            return
        if "<s>" in text:                       # agent opens a turn: what the user said so far is one utterance
            self._flush("user")
        self._pending["agent"] += text
        if "</s>" in text:
            self._flush("agent")

    def _flush(self, role: str) -> None:
        t = self._pending[role]
        self._pending[role] = ""
        self._append(role, t)

    def turns(self) -> list[Utterance]:
        """Closed utterances plus what is still open (for summaries)."""
        out = list(self.history)
        for role in ("user", "agent"):
            if self._pending[role].strip():
                out.append(Utterance(role, self._pending[role], time.monotonic_ns()))
        return out

    def _append(self, role: str, text: str) -> None:
        if not text or not text.strip():
            return
        if self.history and self.history[-1].role == role:
            self.history[-1].text += text
        else:
            self.history.append(Utterance(role, text, time.monotonic_ns()))

    def on_tool_call_closed(self, c: ToolCall) -> None:
        self.memory.add_from_call(c)

    def on_call_requested(self, model_step: int) -> None:
        """The model finished a call (EOTC): it is now waiting for a TOOL_RESPONSE."""
        self.requests_outstanding.append(model_step)

    def on_result_consumed(self, tool_response_text: str | None, model_step: int) -> str | None:
        """The model consumed a TOOL_RESPONSE (EOTR emitted). Link it to the newest closed, not yet consumed call
        whose model response it contains. A queued/returned result is NOT consumed until this happens."""
        if self.requests_outstanding:
            self.requests_outstanding.pop(0)
        if self.ledger is None:
            return None
        from mynah_v2.agent.tools import TERMINAL, model_response_for
        cands = [c for c in sorted(self.ledger.calls.values(), key=lambda c: c.t_start_ns, reverse=True)
                 if c.call_id not in self.consumed and c.status in TERMINAL]
        if not cands:
            return None
        # exact body match first; the pipeline may rewrite numbers to words (fc_convert_num_to_text), so fall back
        # to the newest closed call (the model handles one call at a time: SOTC is ignored while a call is open)
        hit = next((c for c in cands if tool_response_text and model_response_for(c) in tool_response_text), cands[0])
        self.consumed[hit.call_id] = model_step
        return hit.call_id

    def unknown_tool_response(self, call_text: str) -> str:
        """ENHANCED policy for a call the registry cannot execute: record it as a FAILED call and answer the model
        with an explicit error (the reference injects nothing and the model is left waiting for a response)."""
        body = unknown_tool_error(call_text)
        if self.ledger is not None:
            from mynah_v2.model import tokens as T
            parsed = T.parse_call_text(call_text)
            ok = bool(parsed) and parsed[0].ok
            c = self.ledger.open(parsed[0].name if ok else "<invalid>", parsed[0].arguments if ok else {},
                                 turn_id=-1)
            self.ledger.start(c.call_id)
            self.ledger.deliver_error(c.call_id, json.loads(body)["error"])
        return body

    def execute_calls(self, call_text: str) -> str:
        """ENHANCED policy for a call list (paper §2.2 payload: one or more calls; template: responses as
        ``[r1, r2]``). Every call runs (in parallel, one ledger entry each through the bound handlers); responses are
        returned in the order of the calls, comma-joined so the pipeline's ``<TOOL_RESPONSE>[...]`` wrapper yields
        the template's list. Unknown names get the unknown-tool error in their slot; a call still running after
        ``call_timeout_s`` gets the reference timeout error in its slot."""
        import threading
        from mynah_v2.agent.tools import TIMEOUT_MESSAGE, error_payload
        from mynah_v2.model import tokens as T
        calls = T.parse_call_text(call_text)
        out: list[str | None] = [None] * len(calls)

        def run(i, c):
            if not c.ok:
                out[i] = json.dumps({"error": "invalid_call"})
                return
            h = self.tools.get(c.name) or self.tools.get(c.name.replace("-", "_"))
            if h is None:
                out[i] = self.unknown_tool_response(json.dumps([{"name": c.name, "arguments": c.arguments}]))
                return
            try:
                out[i] = h(c.arguments)
            except Exception as e:  # noqa: BLE001
                out[i] = json.dumps({"error": str(e)})
        ts = [threading.Thread(target=run, args=(i, c), daemon=True) for i, c in enumerate(calls)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(self.call_timeout_s)
        res = [r if r is not None else error_payload("timeout", TIMEOUT_MESSAGE) for r in out]
        if self.tracer:
            self.tracer.emit(E.TOOL_MULTI_CALL, n=len(calls), names=[c.name for c in calls])
        return ", ".join(res)

    def on_session_closing(self, kind: str) -> list[ToolCall]:
        """Session end: every open call is cancelled; a backend result arriving later is recorded as late."""
        return self.ledger.cancel_open(f"session_closed:{kind}") if self.ledger is not None else []

    # ---- rehydration
    def needs_rehydration(self, context_audio_s: float, agent_idle: bool, fc_active: bool) -> bool:
        return context_audio_s >= self.max_context_s and agent_idle and not fc_active

    def summary(self) -> str:
        lines = []
        facts = self.memory.latest_by_key()
        if facts:
            lines.append("Verified tool results so far (use these, do not call the tool again for them):")
            for f in facts:
                lines.append(f"- {f.tool}({json.dumps(f.args, sort_keys=True)}) returned {json.dumps(f.result)}")
        hist = []
        n = 0
        for u in reversed(self.turns()):
            # the text channel carries the turn markers as literal "<s>"/"</s>": never put them into a new prompt
            t = " ".join(re.sub(r"</?s>", " ", u.text).split())
            if not t:
                continue
            if n + len(t) > self.history_chars:
                break
            hist.append(f"{u.role}: {t}")
            n += len(t)
        if hist:
            lines.append("Recent conversation (most recent last):")
            lines += list(reversed(hist))
        return "\n".join(lines)

    def rehydration_prompt(self) -> str:
        s = self.summary()
        self.rehydrations += 1
        extra = f"This conversation is continuing.\n{s}" if s else ""
        if self.prompt_fn is not None:
            return self.prompt_fn(extra)
        return f"{self.base_prompt}\n\n{extra}" if extra else self.base_prompt
