"""Per-session call registry + admission control (EXPERIMENT, CPU prototype; not wired, nothing uses it by default).

Why: on the official checkpoint the FC resume contract only chooses which error the model makes. (PAD, PAD) loses
legitimate chained calls (EN m07: verified ``find_customer`` -> ``prepare_return``); (PAD, EOTR) keeps them but
re-opens the same call after a negative result (engine FX1: m02 11x ``get_order_status(qgq999)`` -> found=false,
t14 9x, t13 7x after a corrected id; no caller speech between the repeats). The model takes both decisions at the
same point, so no resume contract separates them. The application has to decide which requested calls run.

Where it sits: the engine's single pre-execution hook, the executor the NeMo FC loop calls with the decoded call text
(``pipeline._execute_tool_call``; per session already: ``ReferenceEngine._wrap_executor`` for B=1, the routed
per-session executor of ``model.session_state`` for B>1). ``guarded_executor`` wraps that executor. The Thinker and
the resume tokens are never touched. A call that is not executed still gets a TOOL_RESPONSE (the pipeline injects
whatever the executor returns), so phase 2, the EOTR and the hand-back run as for an executed call; the duplex state
machine is unchanged. The response says it was not executed, why, and what to do next.

What already exists and is reused, not duplicated:
- ``ToolLedger`` (per session): call ids, lifecycle, timeouts, cancellation, late results. The registry links each
  executed entry to its ledger ``call_id`` and asks the ledger whether a timed-out backend is still running.
- The pipeline already ignores SOTC while a call is open, so the model cannot open two calls at once in a session;
  ``max_active`` guards the remaining paths (a call list in one opening, a backend still running after a timeout).
- Session close already cancels open ledger calls (``Controller.on_session_closing``); ``CallGuard.close`` clears the
  registry.

Three mechanisms (one ``CallGuard`` per session; no lock or state shared between sessions, so B sessions run B calls
at once):
1. single-flight: at most ``max_active`` executions in flight per session (default 1). A list of calls in one opening
   runs sequentially at 1.
2. registry: a bounded LRU (``max_entries``) of recent requests, with TTL (``ttl_s``), keyed by
   (session_id, input_epoch, tool, canonical args). ``call_id`` is NOT in the key (each attempt has a new one).
   Entry states: PENDING (executing, or timed out with the backend still running), COMPLETED, TIMED_OUT, REJECTED.
   Entries outlive the call: the loops come AFTER the result, so a completed entry must still be there.
   ``input_epoch`` = caller speech events seen so far (non-empty RNN-T user text).
3. admission (in this order):
   - busy:        ``max_active`` executions in flight -> not executed;
   - in flight:   same tool + args still PENDING -> not executed (never a second backend run);
   - budget:      2 negative results, or 2 refusals, in the current input epoch -> "stop and ask the caller";
   - side effect: tools in ``side_effects`` never re-run identical arguments within the TTL, whatever the epoch,
                  unless the call carries the tool's idempotency key (then the backend deduplicates); a timed-out
                  side-effect call is never retried automatically (outcome unknown);
   - repeat:      read-only tool, same args, COMPLETED in the same input epoch -> not executed; the response reuses
                  the earlier result. After new caller speech the same call runs (a legitimate retry);
   - provenance:  a parameter with ``"x-source": "result"`` must equal a value an earlier PRODUCTIVE result returned
                  in this session (no invented ids; r3: ``prepare_return(customer_id="Mario Rossi")``).
                  ``"x-source": "caller"`` slots are only audited (spelled ids in ASR text are too noisy to block on
                  without a measurement).
   A call with NEW arguments after a negative result is a real retry and is not blocked by similarity (budget only).

What this cannot do: stop the model from OPENING a call. A refused call still costs one FC cycle (the model writes the
call, reads the response, hands back). The registry bounds backend runs and retries, not FC time."""

from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

NEGATIVE_STATUS = {"not_found", "no_results", "verification_required", "denied", "error", "not_executed", "failed"}
PRODUCTIVE, UNPRODUCTIVE, TIMEOUT = "productive", "unproductive", "timeout"
PENDING, COMPLETED, TIMED_OUT, REJECTED = "pending", "completed", "timed_out", "rejected"


def default_outcome(result: str | None) -> str:
    try:
        r = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        return PRODUCTIVE if result else UNPRODUCTIVE
    if not isinstance(r, dict):
        return PRODUCTIVE if r else UNPRODUCTIVE
    if r.get("error") == "timeout":
        return TIMEOUT
    if "error" in r or r.get("found") is False or str(r.get("status", "")).lower() in NEGATIVE_STATUS:
        return UNPRODUCTIVE
    return PRODUCTIVE


def canon_args(args: dict) -> str:
    def n(v):
        if isinstance(v, str):
            return "".join(ch for ch in v.lower() if ch.isalnum())
        if isinstance(v, dict):
            return {k: n(x) for k, x in v.items()}
        if isinstance(v, list):
            return [n(x) for x in v]
        return v
    return json.dumps(n(args or {}), sort_keys=True)


def _scalars(x) -> set[str]:
    if isinstance(x, dict):
        return set().union(*(_scalars(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_scalars(v) for v in x)) if x else set()
    return {str(x).strip().lower()} if x is not None and str(x).strip() else set()


@dataclass
class Entry:
    key: tuple                    # (session_id, input_epoch, tool, canonical args)
    tool: str
    args: dict
    epoch: int
    t_ns: int
    state: str = PENDING
    call_id: str | None = None    # ledger id of the execution (tracing / completion); never part of the key
    result: str | None = None
    outcome: str | None = None
    rule: str | None = None       # REJECTED: which rule


class CallRegistry:
    """Bounded per-session LRU of recent requests with TTL. Not thread-safe by itself: CallGuard holds the lock."""

    def __init__(self, session_id: str = "", ttl_s: float = 120.0, max_entries: int = 64, clock=time.monotonic_ns):
        self.session_id, self.ttl_ns, self.max_entries, self.clock = session_id, int(ttl_s * 1e9), max_entries, clock
        self._d: collections.OrderedDict[int, Entry] = collections.OrderedDict()
        self._seq = 0
        self.expired = 0

    def sweep(self) -> None:
        now = self.clock()
        for k in [k for k, e in self._d.items() if e.state != PENDING and now - e.t_ns > self.ttl_ns]:
            del self._d[k]
            self.expired += 1
        while len(self._d) > self.max_entries:            # LRU bound; an in-flight entry is never evicted
            k = next((k for k, e in self._d.items() if e.state != PENDING), None)
            if k is None:
                break
            del self._d[k]
            self.expired += 1

    def add(self, e: Entry) -> Entry:
        self._seq += 1
        self._d[self._seq] = e
        self.sweep()
        return e

    def find(self, tool: str, canon: str, epoch: int | None, states: set[str]) -> Entry | None:
        """Newest live entry for (tool, args) in ``states``; ``epoch=None`` = any input epoch."""
        for e in reversed(self._d.values()):
            if e.tool == tool and e.key[3] == canon and e.state in states and (epoch is None or e.epoch == epoch):
                return e
        return None

    def entries(self) -> list[Entry]:
        return list(self._d.values())

    def active(self) -> int:
        return sum(e.state == PENDING for e in self._d.values())

    def clear(self) -> None:
        self._d.clear()


@dataclass
class Decision:
    allow: bool
    rule: str | None = None
    reason: str = ""
    response: str | None = None        # the TOOL_RESPONSE to inject when not executed
    entry: Entry | None = None
    audit: dict = field(default_factory=dict)


NEXT_STEP = {
    "repeat": "Tell the caller this result. Do not call the tool again unless the caller gives new details.",
    "in_flight": "The same request is still running. Tell the caller you are still checking. Do not call it again.",
    "busy": "Another request is still running. Wait for it before calling a tool.",
    "side_effect_done": "This was already done. Tell the caller. Do not repeat it unless the caller asks again.",
    "side_effect_unknown": "The outcome of this request is unknown. Do not repeat it. Tell the caller and offer a "
                           "person to check.",
    "ungrounded": "Do not guess identifiers. Use the earlier lookup, or ask the caller.",
    "budget": "Stop calling tools and ask the caller.",
}


class CallGuard:
    def __init__(self, schemas: list[dict] | dict | None = None, session_id: str = "", ledger=None,
                 classify: Callable[[str | None], str] = default_outcome, max_unproductive: int = 2,
                 max_active: int = 1, ttl_s: float = 120.0, max_entries: int = 64,
                 side_effects: dict[str, str | None] | None = None, max_issued: int = 512, clock=time.monotonic_ns):
        """schemas: the registry's tool schemas (``ToolRegistry.schemas`` list, or {name: parameters}).
        side_effects: {tool: idempotency-key parameter or None} for tools that change state (never re-run blindly).
        ledger: the session's ToolLedger (optional) to link call ids and see timed-out backends still running."""
        if isinstance(schemas, list):
            schemas = {s["function"]["name"]: s["function"].get("parameters") or {} for s in schemas}
        self.params = schemas or {}
        self.session_id, self.ledger, self.classify = session_id, ledger, classify
        self.max_unproductive, self.max_active = max_unproductive, max_active
        self.side_effects = side_effects or {}
        self.reg = CallRegistry(session_id, ttl_s, max_entries, clock)
        self.clock = clock
        self.lock = threading.Lock()
        self.input_epoch = 0
        # values returned by productive results (provenance), bounded LRU; NOT tied to the registry TTL: an id
        # issued early (a verified customer, a pending operation) stays usable for the rest of the session
        self.issued: collections.OrderedDict[str, None] = collections.OrderedDict()
        self.max_issued = max_issued
        self._neg, self._ref, self._budget_epoch = 0, 0, 0
        self.log: collections.deque[dict] = collections.deque(maxlen=max_entries * 4)
        self.closed = False

    # ---- inputs
    def on_user_text(self, text: str) -> None:
        if text and any(ch.isalnum() for ch in text):
            with self.lock:
                self.input_epoch += 1

    def close(self) -> None:
        """Session close / teardown: free every entry. Open ledger calls are cancelled by the controller."""
        with self.lock:
            self.reg.clear()
            self.issued.clear()
            self.closed = True

    # ---- helpers (lock held)
    def _budget_spent(self) -> bool:
        if self._budget_epoch != self.input_epoch:          # the caller spoke: a new turn, a new budget
            self._neg, self._ref, self._budget_epoch = 0, 0, self.input_epoch
        return max(self._neg, self._ref) >= self.max_unproductive

    def _active(self) -> int:
        """Executions in flight: admitted and not returned, or timed out with the backend still running."""
        return sum(e.state == PENDING or (e.state == TIMED_OUT and self._backend_running(e))
                   for e in self.reg.entries())

    def _backend_running(self, e: Entry) -> bool:
        """A timed-out call whose backend has not returned yet (the ledger records a late result when it does)."""
        if self.ledger is None or e.call_id is None:
            return False
        c = self.ledger.calls.get(e.call_id)
        return c is not None and c.status.value == "timed_out" and not c.late

    # ---- admission
    def admit(self, name: str, args: dict) -> Decision:
        args = args if isinstance(args, dict) else {}
        canon = canon_args(args)
        props = (self.params.get(name) or {}).get("properties") or {}
        audit = {}
        with self.lock:
            self.reg.sweep()
            ep = self.input_epoch
            side = name in self.side_effects
            idem = self.side_effects.get(name)
            caller_slots = [k for k, s in props.items() if isinstance(s, dict) and s.get("x-source") == "caller"
                            and k in args]
            if caller_slots:
                audit["caller_slots_unchecked"] = caller_slots
            same_any = self.reg.find(name, canon, None, {PENDING, COMPLETED, TIMED_OUT})
            in_flight = same_any is not None and (same_any.state == PENDING or
                                                  (same_any.state == TIMED_OUT and self._backend_running(same_any)))
            ungrounded = [k for k, s in props.items() if isinstance(s, dict) and s.get("x-source") == "result"
                          and k in args and str(args[k]).strip().lower() not in self.issued]
            if self.closed:
                d = Decision(False, "closed", "the session is closed")
            elif in_flight:
                d = Decision(False, "in_flight", "the same request is still running")
            elif (n := self._active()) >= self.max_active:
                d = Decision(False, "busy", f"{n} request(s) still running in this session")
            elif self._budget_spent():
                d = Decision(False, "budget", f"{self._neg} negative results and {self._ref} refused calls without "
                                              "new input from the caller")
            elif side and not (idem and str(args.get(idem, "")).strip()) and same_any is not None:
                rule = "side_effect_unknown" if same_any.state == TIMED_OUT else "side_effect_done"
                d = Decision(False, rule, "a request that changes state already ran with these details",
                             audit={"previous_result": same_any.result})
            elif not side and (prev := self.reg.find(name, canon, ep, {COMPLETED})) is not None:
                d = Decision(False, "repeat", "the same call already ran and the caller has not spoken since",
                             audit={"previous_result": prev.result, "previous_outcome": prev.outcome})
            elif ungrounded:
                d = Decision(False, "ungrounded", f"{', '.join(ungrounded)} must come from an earlier tool result",
                             audit={"ungrounded": ungrounded})
            else:
                d = Decision(True)
            d.audit.update(audit)
            e = Entry((self.session_id, ep, name, canon), name, args, ep, self.clock())
            if d.allow:
                d.entry = self.reg.add(e)
            else:
                e.state, e.rule = REJECTED, d.rule
                self.reg.add(e)
                d.response = self._response(d)
                if d.rule != "closed":
                    self._budget_spent()
                    self._ref += 1
            self.log.append({"tool": name, "args": args, "allow": d.allow, "rule": d.rule, "input_epoch": ep,
                             **{k: v for k, v in d.audit.items() if k != "previous_result"}})
            return d

    def complete(self, d: Decision, result: str | None) -> str:
        """After an admitted call ran: store the result (the entry stays for repeats), the ids it issued and its
        outcome (budget). Returns the outcome."""
        out = self.classify(result)
        with self.lock:
            e = d.entry
            e.result, e.outcome, e.t_ns = result or "", out, self.clock()
            e.state = TIMED_OUT if out == TIMEOUT else COMPLETED
            if self.ledger is not None:            # link the ledger entry this execution opened (newest match)
                cands = [c for c in list(self.ledger.calls.values())
                         if c.name == e.tool and canon_args(c.arguments) == e.key[3]]
                if cands:
                    e.call_id = max(cands, key=lambda c: c.t_start_ns).call_id
            if out == PRODUCTIVE:
                try:
                    vals = _scalars(json.loads(result))
                except (json.JSONDecodeError, TypeError):
                    vals = set()
                for v in vals:
                    self.issued[v] = None
                    self.issued.move_to_end(v)
                while len(self.issued) > self.max_issued:
                    self.issued.popitem(last=False)
            elif not self.closed:
                self._budget_spent()
                self._neg += 1
        return out

    @staticmethod
    def _response(d: Decision) -> str:
        body = {"status": "not_executed", "reason": d.reason}
        if "previous_result" in d.audit:
            prev = d.audit["previous_result"]
            try:
                body["previous_result"] = json.loads(prev)
            except (json.JSONDecodeError, TypeError):
                body["previous_result"] = prev
        body["next_step"] = NEXT_STEP.get(d.rule, NEXT_STEP["budget"])
        return json.dumps(body)


def guarded_executor(execute: Callable[[str], str | None], guard: CallGuard, tracer=None) -> Callable[[str], str | None]:
    """Wrap an executor ``call_text -> TOOL_RESPONSE | None`` (the session's ``pipeline._execute_tool_call``).
    Every parsed call is admitted or refused; admitted calls run through ``execute`` one at a time (single-flight per
    session; a list keeps its order, responses comma-joined as the controller's multi-call path does); refused ones
    get the guard's response in their slot. Unparsable text goes to ``execute`` untouched (unknown-tool policy)."""
    from mynah_v2.model import tokens as T

    def run(call_text: str) -> str | None:
        calls = T.parse_call_text(call_text)
        if not calls or not all(c.ok for c in calls):
            return execute(call_text)
        out = []
        for c in calls:
            d = guard.admit(c.name, c.arguments)
            if tracer is not None:
                tracer.emit("tool.guard", tool=c.name, args=c.arguments, allow=d.allow, rule=d.rule)
            if not d.allow:
                out.append(d.response)
                continue
            one = call_text if len(calls) == 1 else json.dumps([{"name": c.name, "arguments": c.arguments}])
            try:
                res = execute(one)
            except Exception as e:  # noqa: BLE001 - a crashing executor must not leave the entry in flight
                res = json.dumps({"error": f"{type(e).__name__}: {e}"})
            guard.complete(d, res)
            out.append(res)
        if all(r is None for r in out):
            return None
        return ", ".join(r if r is not None else json.dumps({"error": "invalid_call"}) for r in out)

    return run
