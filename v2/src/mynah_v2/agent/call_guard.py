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
import re
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


# ------------------------------------------------------------------ ASR shadow (observe only, never acts)
_DIG = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9"}
_STOP = {"a", "i", "is", "my", "the", "and", "it", "to", "of", "me", "you", "do", "be", "on", "at", "in", "an", "or",
         "so", "we", "he", "she", "its", "can", "hi", "ok", "um", "uh", "for", "get", "see", "yes", "but", "not", "now",
         "how", "who", "why", "was", "are", "has", "had", "did", "our", "her", "his", "him", "out", "all", "any",
         "one's", "am", "as", "by", "if", "no", "up", "us", "hey"}
_NEG = re.compile(r"\b(don'?t|do not|not|no|never|without|never mind)\b")
_TRANSFER = re.compile(r"\b(operator|human|real person|a person|someone|an agent|representative)\b")
_TRANSFER_VERB = re.compile(r"\b(speak|talk|transfer|connect|put me through)\b")


def spoken_canon(s: str) -> str:
    """Spoken id / number -> canonical form: digit words -> digits, single letters and short letter runs kept
    ("Wxy six one five" -> WXY615); filler words dropped."""
    out = []
    for w in re.findall(r"[a-z]+|\d", str(s).lower()):
        if w in _DIG:
            out.append(_DIG[w])
        elif w.isdigit():
            out.append(w)
        elif len(w) <= 3 and w not in _STOP:
            out.append(w)
    return "".join(out).upper()


def spoken_spans(s: str) -> set[str]:
    """Maximal runs of spoken id pieces (digit words, digits, single letters, short non-stop letter runs), each run
    canonicalised ("order X Y Z one two three" -> {"XYZ123"}; "eight nines" -> {"99999999"}). A slot matches only a
    WHOLE run: a substring of a longer run (9999 inside eight nines) is a mismatch."""
    spans, cur = set(), []
    for w in re.findall(r"[a-z]+|\d", str(s).lower()):
        if w in _DIG:
            cur.append(_DIG[w])
        elif w.isdigit():
            cur.append(w)
        elif len(w) <= 3 and w not in _STOP:
            cur.append(w)
        else:
            if cur:
                spans.add("".join(cur).upper())
            cur = []
    if cur:
        spans.add("".join(cur).upper())
    return spans


def id_like(v) -> bool:
    """A value worth checking against the caller's speech: has a digit, or is a short code (no spaces)."""
    v = str(v)
    return bool(re.search(r"\d", v)) or (" " not in v.strip() and 3 <= len(v.strip()) <= 10 and v.strip().isalnum()
                                          and not v.strip().isalpha())


class AsrShadow:
    """Caller ASR (the Thinker's ASR channel as the session traces it) kept per session, bounded, to compare with the
    function head's decisions. OBSERVE ONLY: it annotates the guard's decisions and records intents; it never
    refuses, executes or changes a call."""

    def __init__(self, intent_tools: dict[str, str] | None = None, max_chars: int = 4000):
        self.intent_tools = intent_tools or {}           # e.g. {"transfer": "escalate_to_human"}
        self.max_chars = max_chars
        self.segments: collections.deque[tuple[int | None, str]] = collections.deque()
        self.n_chars = 0
        self.intents: list[dict] = []

    def add(self, text: str, model_step: int | None = None) -> None:
        self.segments.append((model_step, text))
        self.n_chars += len(text)
        while self.n_chars > self.max_chars and self.segments:
            self.n_chars -= len(self.segments.popleft()[1])
        heard = self.text()
        for sentence in re.split(r"[.?!]", heard)[-3:]:
            t = sentence.lower()
            m = _TRANSFER.search(t)
            if m and _TRANSFER_VERB.search(t) and not any(i["intent"] == "transfer" and i["evidence"] == t.strip()
                                                          for i in self.intents):
                neg = bool(_NEG.search(t[:m.end()]))
                self.intents.append({"intent": "transfer", "evidence": t.strip(), "negated": neg,
                                     "model_step": model_step, "expected": self.intent_tools.get("transfer"),
                                     "fulfilled": False if not neg else None})

    def text(self) -> str:
        return "".join(t for _, t in self.segments)

    def slot_check(self, args: dict) -> dict:
        spans = spoken_spans(self.text())
        out = {}
        for k, v in (args or {}).items():
            if isinstance(v, str) and id_like(v):
                cv = "".join(ch for ch in v if ch.isalnum()).upper()     # the call's value is a code, not speech
                out[k] = "match" if cv and cv in spans else "mismatch" if spans else "absent"
        return out

    def on_call(self, name: str) -> None:
        for i in self.intents:
            if i.get("expected") == name and i["fulfilled"] is False:
                i["fulfilled"] = True

    def report(self) -> dict:
        return {"asr_chars": self.n_chars, "intents": list(self.intents),
                "unfulfilled": [i for i in self.intents if i["fulfilled"] is False]}


# Compact refusals (opt-in, response_style="compact"): every refusal token is one FC phase-2 step (~165 ms on GB10);
# the full repeat refusal is 86 tokens (87 steps, 14.1 s), these are ~30. The earlier result is NOT restated: the model
# already read it in its own context. Admission rules are identical in both styles.
COMPACT = {
    "repeat": {"status": "already_done", "next_step": "Tell the caller."},
    "in_flight": {"status": "in_progress", "next_step": "Tell the caller you are still checking."},
    "busy": {"status": "busy", "next_step": "Wait for the current request."},
    "side_effect_done": {"status": "already_done", "next_step": "Tell the caller it is done."},
    "side_effect_unknown": {"status": "unknown", "next_step": "Do not repeat. Offer a person."},
    "ungrounded": {"status": "not_executed", "next_step": "Ask the caller."},
    "budget": {"status": "not_executed", "next_step": "Ask the caller."},
    "closed": {"status": "not_executed", "next_step": "Ask the caller."},
}
RESPONSE_STYLES = ("full", "compact", "echo")
# "compact" FAILED live (job 122): {"status": "already_done"} was read as "the order is done" (m02 said "Your order
# ... is already done"; t14 lost the not-found outcome and re-looked it up after the caller's next question).
# "echo" keeps the outcome: a repeat refusal restates the EARLIER result the model already read, plus a short note
# (t14 / m02: 45 phase-2 steps instead of 87); the other rules use the compact texts.
ECHO_NOTE = "Already checked. Tell the caller."


class CallGuard:
    def __init__(self, schemas: list[dict] | dict | None = None, session_id: str = "", ledger=None,
                 classify: Callable[[str | None], str] = default_outcome, max_unproductive: int = 2,
                 max_active: int = 1, ttl_s: float = 120.0, max_entries: int = 64,
                 side_effects: dict[str, str | None] | None = None, max_issued: int = 512, clock=time.monotonic_ns,
                 asr_shadow: AsrShadow | None = None, response_style: str = "full"):
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
        self.resume_log: collections.deque[dict] = collections.deque(maxlen=max_entries * 4)
        self._last: dict | None = None
        self.closed = False
        self.asr = asr_shadow                    # optional ASR shadow (observe only)
        if response_style not in RESPONSE_STYLES:
            raise ValueError(f"response_style must be one of {RESPONSE_STYLES}")
        self.response_style = response_style

    # ---- inputs
    def on_user_text(self, text: str, model_step: int | None = None) -> None:
        if text and any(ch.isalnum() for ch in text):
            with self.lock:
                self.input_epoch += 1
                if self.asr is not None:
                    self.asr.add(text, model_step)

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
            if self.asr is not None:              # shadow: annotate only, the decision below ignores it
                audit["asr_slots"] = self.asr.slot_check(args)
                self.asr.on_call(name)
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
            self._last = {"outcome": None if d.allow else "refused", "epoch": ep}
            return d

    def complete(self, d: Decision, result: str | None) -> str:
        """After an admitted call ran: store the result (the entry stays for repeats), the ids it issued and its
        outcome (budget). Returns the outcome."""
        out = self.classify(result)
        with self.lock:
            e = d.entry
            e.result, e.outcome, e.t_ns = result or "", out, self.clock()
            e.state = TIMED_OUT if out == TIMEOUT else COMPLETED
            self._last = {"outcome": out, "epoch": e.epoch}
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

    def resume_contract(self) -> str:
        """Resume "by_outcome" (the guarded default): the FC resume contract for the hand-back that is about to happen, from the
        last call of this session. (PAD, EOTR) lets the model chain a further call (a positive result whose data feeds
        the next tool); (PAD, PAD) after a negative / timed-out / refused result, or when the caller spoke after the
        call was issued (the new request leads). Observed with (PAD, EOTR): identical re-calls after 10/11 negative
        results and 0/13 positive ones. A call still in flight (the phase-1 hand-back) keeps (PAD, EOTR)."""
        with self.lock:
            last = getattr(self, "_last", None)
            if self.reg.active() or last is None or last["outcome"] is None:
                c = "pad_eotr"
            elif last["outcome"] != PRODUCTIVE or self.input_epoch != last["epoch"]:
                c = "pad_pad"
            else:
                c = "pad_eotr"
            self.resume_log.append({"contract": c, "last": dict(last) if last else None, "input_epoch": self.input_epoch})
            return c

    def _response(self, d: Decision) -> str:
        if self.response_style == "echo" and d.rule in ("repeat", "side_effect_done") and "previous_result" in d.audit:
            try:
                prev = json.loads(d.audit["previous_result"])
            except (json.JSONDecodeError, TypeError):
                prev = None
            if isinstance(prev, dict):
                return json.dumps(dict(prev, note=ECHO_NOTE))
            return json.dumps({"previous_result": d.audit["previous_result"], "note": ECHO_NOTE})
        if self.response_style in ("compact", "echo"):
            return json.dumps(COMPACT.get(d.rule, COMPACT["budget"]))
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
                from mynah_v2.tracing import events as E
                tracer.emit(E.TOOL_GUARD, tool=c.name, args=c.arguments, allow=d.allow, rule=d.rule,
                            input_epoch=guard.input_epoch)
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


def install(engine, guard: CallGuard, tracer=None) -> Callable[[], None]:
    """Put ``guard`` in front of a single-session engine's executor (``engine.pipeline._execute_tool_call``, the
    function NeMo's FC loop calls; set before the session's FC threads start). Returns ``uninstall``: restores the
    previous executor and frees the guard. For B>1 the routed per-session executor would be wrapped instead."""
    p = engine.pipeline
    prev = p._execute_tool_call
    p._execute_tool_call = guarded_executor(prev, guard, tracer)

    def uninstall() -> None:
        p._execute_tool_call = prev
        guard.close()
    return uninstall


def summary(guard: CallGuard) -> dict:
    """What a run records: every decision, the refusals by rule, the registry size at the end."""
    log = list(guard.log)
    by_rule: dict[str, int] = {}
    for r in log:
        if not r["allow"]:
            by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1
    out = {"decisions": log, "admitted": sum(r["allow"] for r in log), "refused": by_rule,
           "input_epoch": guard.input_epoch, "entries_left": len(guard.reg.entries())}
    if guard.resume_log:
        out["resume"] = list(guard.resume_log)
    if guard.asr is not None:
        out["asr_shadow"] = guard.asr.report()
        out["asr_slot_mismatches"] = [{"tool": r["tool"], "args": r["args"], "asr_slots": r.get("asr_slots")}
                                      for r in log if any(v == "mismatch" for v in (r.get("asr_slots") or {}).values())]
    return out


# ------------------------------------------------------------------ session attachment (the default path)
def guard_handler(name: str, fn, guard: CallGuard, tracer=None):
    """A bound tool handler (``handler(args) -> str``, what NeMo's executor calls for one parsed call) behind the
    session's guard: refused calls answer with the guard's response and never reach ``fn`` (nor the ledger inside it).
    Every path that executes session tools goes through these handlers: one session, several sessions (the routed
    per-session registry), and the controller's multi-call path."""
    def handler(args, *a, **kw):
        d = guard.admit(name, args if isinstance(args, dict) else {})
        if tracer is not None:
            from mynah_v2.tracing import events as E
            tracer.emit(E.TOOL_GUARD, tool=name, args=args, allow=d.allow, rule=d.rule, input_epoch=guard.input_epoch)
        if not d.allow:
            return d.response
        try:
            res = fn(args, *a, **kw)
        except Exception as e:  # noqa: BLE001 - a crashing handler must not leave the entry in flight
            res = json.dumps({"error": f"{type(e).__name__}: {e}"})
            guard.complete(d, res)
            raise
        guard.complete(d, res)
        return res
    handler.__wrapped__ = fn
    handler.__v2_call_guard__ = guard
    return handler


def attach_session_guard(engine, guard: CallGuard, resume: str | None = "by_outcome", tracer=None) -> None:
    """Put ``guard`` on the tools the CURRENT session bound (``engine.session_tools``) and make the engine feed it the
    caller's ASR (read by ``ReferenceEngine.step``). ``resume`` = the FC resume contract for this
    session's stream while guarded (fc_ctx_fix reads ``engine._v2_resume_by_stream``; the guard itself is kept in
    ``engine._v2_guard_by_stream``, both keyed by stream id): "by_outcome" (default) asks the guard at each hand-back
    (``CallGuard.resume_contract``: (PAD, EOTR) after a positive result so a dependent call can follow, (PAD, PAD)
    after a negative / refused one or new caller speech); "pad_eotr" / "pad_pad" fix one contract; None keeps the
    engine's contract. Undone by
    ``detach_session_guard`` (the engine's close calls it) and by the tool unbind."""
    binding = engine.session_tools
    reg = binding.registry
    for name in list(getattr(binding, "names", [])):
        if name in reg and not hasattr(reg[name], "__v2_call_guard__"):
            reg[name] = guard_handler(name, reg[name], guard, tracer)
    _by(engine, "_v2_guard_by_stream")[engine.stream_id] = guard
    if resume:
        _by(engine, "_v2_resume_by_stream")[engine.stream_id] = resume


def _by(engine, name: str) -> dict:
    d = getattr(engine, name, None)
    if d is None:
        d = {}
        setattr(engine, name, d)
    return d


def session_guard(engine):
    """The guard of the engine's CURRENT stream (keyed by stream id: one engine serves sessions one after the other or,
    through MultiSessionEngine, several at once with one stream id each)."""
    return (getattr(engine, "_v2_guard_by_stream", None) or {}).get(getattr(engine, "stream_id", None))


def detach_session_guard(engine) -> dict | None:
    """End of the session: the summary of the guard's decisions (also kept in ``engine._v2_guard_summaries[stream]``);
    the guard and the resume override of this stream are removed."""
    sid = getattr(engine, "stream_id", None)
    g = _by(engine, "_v2_guard_by_stream").pop(sid, None)
    _by(engine, "_v2_resume_by_stream").pop(sid, None)
    s = summary(g) if g is not None else None
    if s is not None:
        _by(engine, "_v2_guard_summaries")[sid] = s
    return s
