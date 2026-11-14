"""Pre-execution call guard (EXPERIMENT, CPU prototype; not wired into the Session, nothing uses it by default).

Why: on the official checkpoint the FC resume contract only chooses which error the model makes. (PAD, PAD) loses
legitimate chained calls (EN m07: verified ``find_customer`` -> ``prepare_return``); (PAD, EOTR) keeps them but
re-opens the same call after a negative result (engine FX1: m02 11x ``get_order_status(qgq999)`` -> found=false,
t14 9x, t13 7x after a corrected id). The model takes both decisions at the same point, so the resume contract cannot
separate them. This guard is the application-side half: it sees every call the model requests BEFORE the backend
runs, with the session state the model does not have, and decides whether to execute it.

It sits at the engine's single pre-execution hook (``ReferenceEngine.pipeline._execute_tool_call``, the executor the
NeMo FC loop calls with the decoded call text) via ``guarded_executor``. It never touches the Thinker or the resume
tokens. A rejected call is NOT dropped: the model always receives a TOOL_RESPONSE (the pipeline injects whatever the
executor returns), so the FC loop, its phase 2, the EOTR and the hand-back run exactly as for an executed call, and the
duplex state machine is unchanged. The response says the call was not executed, why, and what to do next.

Rules (all generic; tenant-specific business rules stay in the tenant backend):
- R1 repeat: same tool and same normalised arguments as an earlier EXECUTED call, with no new caller speech since that
  call completed -> not executed; the response carries the earlier result. A repeat after the caller spoke again is a
  legitimate retry (the caller repeated or confirmed the id) and runs.
- R2 provenance: a parameter whose schema says ``"x-source": "result"`` (an id the backend issues, e.g. a
  ``customer_id``) must equal a value that an earlier PRODUCTIVE result returned in this session. Otherwise the slot was
  invented (r3: ``prepare_return(customer_id="Mario Rossi")``) -> not executed. Parameters without ``x-source`` are not
  checked; caller-provided slots (``"x-source": "caller"``) are only AUDITED here (spelled ids in ASR text are too
  noisy to block on without a measurement).
- R3 budget: at most ``max_unproductive`` negative results, or ``max_unproductive`` rejections, without new caller
  speech (counted apart, so one negative result and its rejected repeat do not block a retry with new arguments);
  past either, every call is answered "stop and ask the caller" without running.
A call with new arguments after a negative result (a real retry with a different id) is not blocked by similarity:
it runs, within the R3 budget.

The outcome of a result is classified by ``classify`` (default ``default_outcome``: JSON with an ``error`` key, a
``found: false`` or a negative ``status`` -> unproductive). Every decision is kept in ``log`` for the audit/trace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

NEGATIVE_STATUS = {"not_found", "no_results", "verification_required", "denied", "error", "not_executed", "failed"}
PRODUCTIVE, UNPRODUCTIVE = "productive", "unproductive"


def default_outcome(result: str | None) -> str:
    try:
        r = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        return PRODUCTIVE if result else UNPRODUCTIVE
    if not isinstance(r, dict):
        return PRODUCTIVE if r else UNPRODUCTIVE
    if "error" in r or r.get("found") is False or str(r.get("status", "")).lower() in NEGATIVE_STATUS:
        return UNPRODUCTIVE
    return PRODUCTIVE


def norm_args(args: dict) -> str:
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
class Decision:
    allow: bool
    rule: str | None = None            # None | "repeat" | "ungrounded" | "budget"
    reason: str = ""
    response: str | None = None        # the TOOL_RESPONSE to inject when not executed
    audit: dict = field(default_factory=dict)


@dataclass
class _Done:
    name: str
    key: str
    result: str
    outcome: str
    user_epoch: int


class CallGuard:
    def __init__(self, schemas: list[dict] | dict | None = None, classify: Callable[[str | None], str] = default_outcome,
                 max_unproductive: int = 2):
        """schemas: the registry's tool schemas (``ToolRegistry.schemas`` list, or {name: parameters})."""
        if isinstance(schemas, list):
            schemas = {s["function"]["name"]: s["function"].get("parameters") or {} for s in schemas}
        self.params = schemas or {}
        self.classify = classify
        self.max_unproductive = max_unproductive
        self.user_epoch = 0                      # caller speech events seen so far (RNN-T user text, non-empty)
        self.done: list[_Done] = []
        self.issued: set[str] = set()            # scalar values returned by productive results (R2)
        self._unprod, self._rej, self._unprod_epoch = 0, 0, 0
        self.log: list[dict] = []

    # ---- inputs
    def on_user_text(self, text: str) -> None:
        if text and text.strip():
            self.user_epoch += 1

    def _over_budget(self) -> bool:
        if self._unprod_epoch != self.user_epoch:          # the caller spoke: a new turn, a new budget
            self._unprod, self._rej, self._unprod_epoch = 0, 0, self.user_epoch
        return max(self._unprod, self._rej) >= self.max_unproductive

    def check(self, name: str, args: dict) -> Decision:
        key = norm_args(args)
        audit = {}
        props = (self.params.get(name) or {}).get("properties") or {}
        if self._over_budget():
            d = Decision(False, "budget", f"{self._unprod} negative results and {self._rej} rejected calls without "
                                          "new input from the caller")
        else:
            prev = next((p for p in reversed(self.done) if p.name == name and p.key == key), None)
            ungrounded = [k for k, sch in props.items() if isinstance(sch, dict) and sch.get("x-source") == "result"
                          and k in (args or {}) and str(args[k]).strip().lower() not in self.issued]
            caller_slots = [k for k, sch in props.items() if isinstance(sch, dict) and sch.get("x-source") == "caller"
                            and k in (args or {})]
            if caller_slots:
                audit["caller_slots_unchecked"] = caller_slots
            if prev is not None and prev.user_epoch == self.user_epoch:
                d = Decision(False, "repeat", "the same call already ran and the caller has not spoken since",
                             audit={"previous_result": prev.result, "previous_outcome": prev.outcome})
            elif ungrounded:
                d = Decision(False, "ungrounded", f"{', '.join(ungrounded)} must come from an earlier tool result",
                             audit={"ungrounded": ungrounded})
            else:
                d = Decision(True)
        d.audit.update(audit)
        if not d.allow:
            d.response = self._reject_response(d)
            self._rej += 1
        self.log.append({"tool": name, "args": args, "allow": d.allow, "rule": d.rule, "user_epoch": self.user_epoch,
                         **d.audit})
        return d

    def record(self, name: str, args: dict, result: str | None) -> str:
        """After an executed call: remember it (R1), the ids it issued (R2) and its outcome (R3)."""
        out = self.classify(result)
        self.done.append(_Done(name, norm_args(args), result or "", out, self.user_epoch))
        if out == PRODUCTIVE:
            try:
                self.issued |= _scalars(json.loads(result))
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            self._over_budget()
            self._unprod += 1
        return out

    @staticmethod
    def _reject_response(d: Decision) -> str:
        body = {"status": "not_executed", "reason": d.reason}
        if d.rule == "repeat":
            prev = d.audit.get("previous_result")
            try:
                body["previous_result"] = json.loads(prev)
            except (json.JSONDecodeError, TypeError):
                body["previous_result"] = prev
            body["next_step"] = "Tell the caller this result. Do not call the tool again unless the caller gives new details."
        elif d.rule == "ungrounded":
            body["next_step"] = "Do not guess identifiers. Use the earlier lookup, or ask the caller."
        else:
            body["next_step"] = "Stop calling tools and ask the caller."
        return json.dumps(body)


def guarded_executor(execute: Callable[[str], str | None], guard: CallGuard, tracer=None) -> Callable[[str], str | None]:
    """Wrap an executor ``call_text -> TOOL_RESPONSE | None`` (the engine's ``pipeline._execute_tool_call``).
    Every parsed call is checked; allowed calls run through ``execute`` one by one (a list keeps its order, responses
    comma-joined as the controller's multi-call path does); rejected ones get the guard's response in their slot.
    Unparsable text goes to ``execute`` untouched (the existing unknown-tool policy answers it)."""
    from mynah_v2.model import tokens as T

    def run(call_text: str) -> str | None:
        calls = T.parse_call_text(call_text)
        if not calls or not all(c.ok for c in calls):
            return execute(call_text)
        out = []
        for c in calls:
            d = guard.check(c.name, c.arguments)
            if tracer is not None:
                tracer.emit("tool.guard", tool=c.name, args=c.arguments, allow=d.allow, rule=d.rule)
            if not d.allow:
                out.append(d.response)
                continue
            one = call_text if len(calls) == 1 else json.dumps([{"name": c.name, "arguments": c.arguments}])
            res = execute(one)
            guard.record(c.name, c.arguments, res)
            out.append(res)
        if all(r is None for r in out):
            return None
        return ", ".join(r if r is not None else json.dumps({"error": "invalid_call"}) for r in out)

    return run
