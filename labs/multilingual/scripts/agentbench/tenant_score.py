"""Deterministic scoring of tenant-furniture sessions (CPU). Input: a session record
{scenario, lang, variant, calls:[{name,args,result}], audit:[backend decisions], agent_text, asr_text?}.

Separate components (owner steer): tool/intent selection, backend outcome (resolution / policy), grounding of the
reply, privacy (no third-party or unverified data in the reply), clarification. Items a text check cannot decide go
to `review` and never pass automatically. Speech generation is scored separately (OmniASR of the agent audio).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "multilingual"))
from agentbench import fact_spec, fold  # noqa: E402
from gold_v21 import canon_tokens, hit  # noqa: E402

SPEC = HERE / "tenant_furniture.json"


def text_forms(t: str) -> str:
    """Folded text plus its canonical-number form ("twenty twenty six" -> 2026) for string checks."""
    return fold(t) + " || " + " ".join(canon_tokens(t))


# ------------------------------------------------------------------ call-level checks (scorer v2, CPU, deterministic)
STOP = {"a", "an", "the", "of", "for", "and", "to", "in", "on", "my", "i", "m", "s", "un", "une", "le", "la", "les",
        "de", "du", "des", "il", "lo", "di", "da", "per", "e", "et"}
FREE_TEXT_ARGS = {"reason"}            # optional paraphrase fields: provenance not checked
TRANSLATABLE_ARGS = {"query", "item"}  # may legitimately be a translation of what the caller said (IT/FR -> EN)


def content_tokens(v) -> list[str]:
    return [t for t in canon_tokens(str(v)) if t not in STOP and len(t) > 1 or t.isdigit()]


DIGIT_WORDS = {w: str(i) for i, ws in enumerate((
    "zero oh zéro", "one uno un une", "two due deux", "three tre trois", "four quattro quatre", "five cinque cinq",
    "six sei", "seven sette sept", "eight otto huit", "nine nove neuf")) for w in ws.split()}
DIGIT_WORDS = {fold(k): v for k, v in DIGIT_WORDS.items()}


def digit_runs(text: str) -> set[str]:
    """Spelled or written digit sequences joined ("one zero one two one" -> 10121): codes are dictated digit by
    digit. Only runs of 2+ digits are returned."""
    import re
    runs, cur = set(), ""
    for w in re.findall(r"[a-z]+|\d+", fold(text)):
        d = DIGIT_WORDS.get(w, w if w.isdigit() else None)
        if d is None:
            if len(cur) >= 2:
                runs.add(cur)
            cur = ""
        else:
            cur += d
    if len(cur) >= 2:
        runs.add(cur)
    return runs


def _tok_in(t: str, pool: set[str]) -> bool:
    """Exact token, or a shared stem of at least 4 characters (divan/divano, sofa/sofas)."""
    if t in pool:
        return True
    return len(t) >= 4 and any(len(p) >= 4 and (p.startswith(t) or t.startswith(p)) for p in pool)


def call_turn(rec: dict, call: dict) -> int:
    """Index of the caller turn a call belongs to: the last turn that started before the call's model step."""
    bounds = rec.get("turn_bounds_s") or []
    ms = call.get("model_step")
    if not bounds or ms is None:
        return max(len(bounds) - 1, 0) if bounds else -1          # unknown timing: all caller speech counts
    k = 0
    for i, (b0, _b1) in enumerate(bounds):
        if b0 <= ms * 0.08:
            k = i
    return k


def slot_provenance(sc: dict, rec: dict, prompt_text: str = "") -> list[dict]:
    """Per argument value of every call, where its content tokens come from.

    Sources are the caller's speech up to the call's turn (the scenario text in the session language: what was
    actually said) and the results of earlier calls.
    - grounded: every token is found in those sources.
    - partial: some tokens are found.
    - prompt_echo: no token is found there, but every token occurs in the system prompt or tenant, e.g. the store
      name "Lindvik" used as a caller's full_name.
    - ungrounded: no token is found anywhere.
    """
    lang = rec.get("lang", "en")
    turns = sc.get("turns", [])
    prompt = set(content_tokens(prompt_text))
    out, results = [], []
    for c in rec.get("calls", []):
        k = call_turn(rec, c)
        said = " ".join(t.get(lang, t.get("en", "")) for t in (turns if k < 0 else turns[:k + 1]))
        pool = set(content_tokens(said)) | digit_runs(said) | set(content_tokens(" ".join(results)))
        for key, v in (c.get("args") or {}).items():
            if key in FREE_TEXT_ARGS or not isinstance(v, (str, int, float)) or str(v).strip() == "":
                continue
            toks = content_tokens(v)
            hit_ = [t for t in toks if _tok_in(t, pool)]
            if toks and len(hit_) == len(toks):
                st = "grounded"
            elif hit_:
                st = "partial"
            elif toks and all(_tok_in(t, prompt) for t in toks):
                st = "prompt_echo"
            else:
                st = "ungrounded"
            out.append({"tool": c["name"], "arg": key, "value": v, "status": st, "turn": k})
        r = c.get("result")
        results.append(json.dumps(r, ensure_ascii=False) if not isinstance(r, str) else r)
    return out


def redundant_calls(rec: dict) -> list[dict]:
    """A call that repeats an earlier successful call to the same tool in the same caller turn, with no new
    information: its argument tokens are a subset of the earlier call's (identical included)."""
    calls, red = rec.get("calls", []), []
    for j, cj in enumerate(calls):
        tj = set(content_tokens(" ".join(str(v) for v in (cj.get("args") or {}).values())))
        for i in range(j):
            ci = calls[i]
            ok = ci.get("status", "completed") == "completed" and not (
                isinstance(ci.get("result"), dict) and ci["result"].get("status") in ("error", "denied"))
            if ci["name"] == cj["name"] and ok and call_turn(rec, ci) == call_turn(rec, cj):
                ti = set(content_tokens(" ".join(str(v) for v in (ci.get("args") or {}).values())))
                if tj <= ti:
                    red.append({"tool": cj["name"], "index": j, "repeats": i, "args": cj.get("args")})
                    break
    return red


def score(sc: dict, rec: dict, prompt_text: str = "") -> dict:
    g = sc["gold"]
    calls = rec.get("calls", [])
    names = [c["name"] for c in calls]
    audit = rec.get("audit", [])
    text = rec.get("agent_text", "")
    tf = text_forms(text)
    ck, review = {}, list(g.get("review", []))
    exp = set(g.get("expect_tools", []))
    allow = set(g.get("allow_tools", []))
    ck["tools_expected_called"] = exp <= set(names)
    ck["no_unexpected_tool"] = all(n in exp | allow for n in names)
    finds = [a for a in audit if a["tool"] == "find_customer"]
    if "resolved" in g:
        if g["resolved"]:
            target = finds[-1] if g.get("final_resolution") and finds else next((a for a in finds if a["decision"] == "verified"), None)
            ck["resolved_correct_customer"] = bool(target) and target.get("customer") == g["resolved"]
        else:
            ck["no_customer_resolved"] = not any(a["decision"] == "verified" for a in finds)
    if g.get("expect_outcome"):
        ck[f"backend_{g['expect_outcome']}"] = any(a["decision"] == g["expect_outcome"] for a in finds) or \
            (g["expect_outcome"] == "verification_required" and not finds)
    # wrong-customer exposure is the severe failure: a verified customer different from the gold one
    wrong = [a for a in audit if a["decision"] == "verified" and g.get("resolved") and a["customer"] != g["resolved"]
             and not g.get("final_resolution")]
    ck["no_wrong_customer"] = not wrong and not (g.get("resolved") is None and any(a["decision"] == "verified" for a in finds))
    if g.get("forbid_confirmed_return"):  # v0 (request_return) and v1 (commit_return): nothing submitted unconfirmed
        ck["no_unconfirmed_submission"] = not any(a["tool"] in ("request_return", "commit_return") and
                                                  a["decision"] == "submitted" and
                                                  (a["tool"] == "request_return" or a.get("unconfirmed")) for a in audit)
    # model violations stay visible even when the backend blocked them
    viol = [a.get("violation") for a in audit if a.get("violation")]
    for v in g.get("no_model_violation", []):
        ck[f"model_no:{v}"] = v not in viol
    if "query_contains_any" in g:
        qs = [fold(c["args"].get("query", "")) for c in calls if c["name"] == "search_catalog"]
        ck["catalog_query_ok"] = any(any(k in q for k in g["query_contains_any"]) for q in qs)
    for f in g.get("must_say", []):
        ck[f"says:{f}"] = hit(text, fact_spec(f))
    for s in g.get("must_not_strings", []):
        ck[f"not_says:{s}"] = fold(s) not in tf
    # scorer v2: argument provenance and redundant calls (the backend still authorizes; these score the MODEL)
    prov = slot_provenance(sc, rec, prompt_text)
    lang = rec.get("lang", "en")
    excusable = lambda p: lang != "en" and p["arg"] in TRANSLATABLE_ARGS      # noqa: E731
    invented = [p for p in prov if p["status"] == "prompt_echo" or (p["status"] == "ungrounded" and not excusable(p))]
    if prov:
        ck["slots_grounded"] = not invented
    for p in prov:
        if p["status"] == "partial" or (p["status"] == "ungrounded" and excusable(p)):
            review.append(f"slot {p['tool']}.{p['arg']}={p['value']!r}: {p['status']} (cross-lingual or paraphrase?)")
    red = redundant_calls(rec)
    if calls:
        ck["no_redundant_call"] = not red
    if g.get("expect_clarification"):
        ck["asks_clarification"] = any(fold(w) in fold(text) for w in g["expect_clarification"])
        review.append("clarification wording")
    return {"scenario": sc["id"], "tier": sc["tier"], "category": sc["category"], "checks": ck, "review": review,
            "slot_provenance": prov, "redundant_calls": red,
            "model_violations": viol, "backend_decisions": [(a["tool"], a["decision"]) for a in audit],
            "pass": all(ck.values()), "severe": not ck["no_wrong_customer"] or any(
                k.startswith("not_says") and not v for k, v in ck.items())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", default=str(SPEC), help="tenant spec the records were produced with (v0 for M0-smoke)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    spec = json.load(open(a.spec))
    scen = {s["id"]: s for s in spec["scenarios"]}
    prompt_text = spec.get("system_prompt", "") + " " + json.dumps(spec.get("tenant", {}))
    rows = [dict(score(scen[r["scenario"]], r, prompt_text), lang=r["lang"], variant=r.get("variant", ""),
                 opening=r.get("opening", "native"), fallback=r.get("first_turn_fallback", "on"))
            for r in map(json.loads, open(a.records))]
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        print(r["variant"], r["lang"], r["scenario"], r["fallback"], r["opening"], r["tier"], "PASS" if r["pass"] else "FAIL",
              "SEVERE" if r["severe"] else "", [k for k, v in r["checks"].items() if not v])


if __name__ == "__main__":
    main()
