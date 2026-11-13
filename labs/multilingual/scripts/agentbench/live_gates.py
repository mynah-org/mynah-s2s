"""Live-session gates and the resume-contract label (CPU, pure functions; owner 2026-11-13, M4).

resume_contract(runtime_fix) names the FC hand-back contract a live run actually used, from what the run recorded:
- "PAD,PAD (engine fc_ctx_fix)": this repository's engine fix is enabled (its default on the official checkpoint);
- "PAD,EOTR (lab NeMo patch)":   the frozen lab NeMo patch with its flags on (historical lab results);
- "upstream 0,0":                neither (the NVIDIA defect, tokens id 0 at the resume step);
- "INVALID double fix":          both at once: never a valid run.
Results with different contracts are not comparable. The label goes with every result.

m07_chain(rec) is the mandatory live regression for the resume contract. In EN m07 ("I want to return the bookcase
… Mario Rossi, postcode 20135"), the legitimate chain is a verified `find_customer` (customer C1002) followed by a
`prepare_return` call. It checks that the (PAD, PAD) resume does not lose that second call; the lab measured that loss
on the official checkpoint (private note fc-resume-contract §11). Pass needs:
- the run used the engine contract (PAD,PAD); any other contract is NOT VALID for this gate, never a pass;
- `find_customer` verified for C1002;
- a later `prepare_return`;
- no `commit_return` / `request_return` submitted without the caller's confirmation.
Extra calls after the chain (re-opened calls) are reported, not failed.
"""
from __future__ import annotations

ENGINE = "PAD,PAD (engine fc_ctx_fix)"
LAB = "PAD,EOTR (lab NeMo patch)"
UPSTREAM = "upstream 0,0"
DOUBLE = "INVALID double fix"


def resume_contract(rf: dict) -> str:
    eng = bool((rf.get("engine_fc_ctx_fix") or {}).get("enabled"))
    lab = bool(rf.get("patched")) and any(str(rf.get(k, "0")) != "0" for k in ("S2S_FIX_CONTEXT_PAD", "S2S_FIX_FC_TOKEN_SYNC"))
    if eng and lab:
        return DOUBLE
    return ENGINE if eng else LAB if lab else UPSTREAM


def m07_chain(rec: dict) -> dict:
    out = {"gate": "m07_chain", "scenario": rec.get("scenario"), "lang": rec.get("lang"), "valid": True, "pass": False,
           "reasons": []}
    contract = resume_contract(rec.get("runtime_fix") or {})
    out["resume_contract"] = contract
    if rec.get("scenario") != "m07" or rec.get("lang") != "en":
        out["valid"] = False
        out["reasons"].append("not the EN m07 session")
    if contract != ENGINE:
        out["valid"] = False
        out["reasons"].append(f"resume contract is {contract!r}, the gate needs {ENGINE!r}")
    calls, audit = rec.get("calls", []), rec.get("audit", [])
    names = [c.get("name") for c in calls]
    out["calls"] = names
    i_find = next((i for i, c in enumerate(calls) if c.get("name") == "find_customer"), None)
    verified = any(a.get("tool") == "find_customer" and a.get("decision") == "verified" and a.get("customer") == "C1002"
                   for a in audit)
    i_prep = next((i for i, c in enumerate(calls) if c.get("name") == "prepare_return" and i_find is not None and i > i_find),
                  None)
    unconfirmed = [a for a in audit if a.get("tool") in ("commit_return", "request_return")
                   and (a.get("tool") == "request_return" or a.get("unconfirmed"))]
    out["checks"] = {"find_customer_verified_C1002": verified, "prepare_return_after_find": i_prep is not None,
                     "no_unconfirmed_submission": not unconfirmed}
    out["extra_calls_after_chain"] = names[i_prep + 1:] if i_prep is not None else []
    if not verified:
        out["reasons"].append("find_customer not verified for C1002")
    if i_prep is None:
        out["reasons"].append("no prepare_return after find_customer (the legitimate second call was lost)")
    if unconfirmed:
        out["reasons"].append("a return was submitted without the caller's confirmation")
    out["fc_trace"] = rec.get("fc_events")
    out["resume_inputs"] = rec.get("fc_resume_inputs")
    out["pass"] = out["valid"] and all(out["checks"].values())
    return out


# ------------------------------------------------------------------ FC trace and FCX (no duplicate / re-opened call)
FC_EVENTS = ("function.started", "function.arguments.done", "tool.pending", "tool.running", "tool.completed",
             "tool.failed", "tool.multi_call", "tool.result.injected", "tool.result.consumed", "tool.result.late")
KEEP = ("model_step", "turn_id", "call_id", "tool", "args", "status", "error", "steps", "func_ids")


def fc_events(trace_path) -> list[dict]:
    """The session's function-call events, in order, from the engine's trace (trace.jsonl)."""
    import json
    out = []
    for ln in open(trace_path):
        d = json.loads(ln)
        if d.get("ev") in FC_EVENTS:
            out.append({"ev": d["ev"], **{k: d[k] for k in KEEP if k in d and d[k] is not None}})
    return out


def _sig(e: dict) -> str:
    import json
    return e.get("tool", "") + " " + json.dumps(e.get("args") or {}, sort_keys=True, ensure_ascii=False)


def fcx_check(rec: dict) -> dict:
    """No duplicated or re-opened call, read from the trace events (engine semantics untouched):
    - every function opening yields at least one tool call (an opening that executes nothing = an empty re-open;
      several calls from one opening are legitimate in ENHANCED mode, `tool.multi_call`);
    - every call id terminates once (completed / failed);
    - no call repeats an earlier completed call with the same tool and arguments (a re-open of a finished call).
    The resume inputs the engine fix recorded (what the first step after each hand-back consumed) are reported."""
    ev = rec.get("fc_events")
    out = {"gate": "fcx", "valid": ev is not None, "pass": False, "reasons": []}
    if ev is None:
        out["reasons"].append("no fc_events in the record (trace not captured)")
        return out
    opens = sum(e["ev"] == "function.started" for e in ev)
    empty, cur = 0, None                                   # per opening: did a call follow before the next opening?
    for e in ev:
        if e["ev"] == "function.started":
            empty += cur is False
            cur = False
        elif e["ev"] == "tool.pending" and cur is not None:
            cur = True
    empty += cur is False
    pend = [e for e in ev if e["ev"] == "tool.pending"]
    term = [e for e in ev if e["ev"] in ("tool.completed", "tool.failed")]
    per_id = {}
    for e in term:
        per_id[e.get("call_id")] = per_id.get(e.get("call_id"), 0) + 1
    done_sigs, dups = {}, []
    sig_of = {e.get("call_id"): _sig(e) for e in pend}
    for e in ev:
        if e["ev"] == "tool.pending":
            s = _sig(e)
            if s in done_sigs:
                dups.append({"call_id": e.get("call_id"), "repeats": done_sigs[s], "call": s})
        elif e["ev"] == "tool.completed":
            done_sigs.setdefault(sig_of.get(e.get("call_id"), ""), e.get("call_id"))
    out["counts"] = {"openings": opens, "empty_openings": empty, "calls": len(pend), "terminated": len(term),
                     "consumed": sum(e["ev"] == "tool.result.consumed" for e in ev)}
    out["checks"] = {"no_empty_opening": empty == 0, "each_call_terminates_once": all(v == 1 for v in per_id.values())
                     and len(per_id) == len(pend), "no_repeat_of_completed_call": not dups}
    out["duplicates"] = dups
    out["resume_inputs"] = rec.get("fc_resume_inputs")
    if empty:
        out["reasons"].append(f"{empty} of {opens} function openings executed no call (empty re-open)")
    if not out["checks"]["each_call_terminates_once"]:
        out["reasons"].append(f"call terminations per id: {per_id}")
    if dups:
        out["reasons"].append(f"{len(dups)} call(s) repeat an earlier completed call")
    out["pass"] = all(out["checks"].values())
    return out
