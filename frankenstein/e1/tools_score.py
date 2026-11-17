"""Score E1c part 1 rows (CPU): tool DECISION / PARAMETERS / EXECUTION / RESUME, reply language, invented facts.

  python e1/tools_score.py RUN_DIR LAB_SCRIPTS > RUN_DIR/tools_score.json

Tenant scenarios use the lab scorer unchanged (agentbench/tenant_score.score: expected tools, backend outcome, slot
provenance, grounding / must-say, must-not-say, clarification). The order-tool scenarios use the gold in the row.
Language ID = function-word count (e1/score.py lang_id), a crude check.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e1.score import lang_id  # noqa: E402

FACT_WORDS = re.compile(r"shipped|spedit|consegn|deliver|novemb|\b20\d\d\b|tracking|in lavorazione|processing", re.I)


def norm_id(s) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def score_orders(r: dict) -> dict:
    g = r["gold"]
    calls = r["calls"]
    names = [c["name"] for c in calls]
    want = g["tool"]
    dec = want in names and (g.get("forbid") not in names)
    params = None
    if want in names:
        c = next(c for c in calls if c["name"] == want)
        if want == "get_order_status":
            params = norm_id(c["args"].get("order_id", "")) == g["args"]["order_id"]
        else:
            params = all(int(c["args"].get(k, -1)) == v for k, v in g["args"].items()
                         if str(c["args"].get(k, "")).lstrip("-").isdigit()) and all(k in c["args"] for k in g["args"])
    executed = bool(calls) and all(c["executed"] for c in calls) and all(l["status"] == "completed" for l in r["ledger"])
    text = r["agent_text"]
    resume = None
    invented = False
    if calls:
        resume = bool(r["replies"]) and not r["log"][-1].get("toolcall")
        if want == "get_order_status" and g.get("found") is False:
            invented = bool(FACT_WORDS.search(text))
            resume = resume and not invented
        elif want == "get_order_status":
            resume = resume and any(w in text.lower() for w in g.get("must_say_any", []))
        elif want == "generate_random_number":
            res = next((json.loads(c["result"]).get("result") for c in calls if c["name"] == want
                        and c["result"].startswith("{")), None)
            resume = resume and res is not None and str(res) in text
    else:
        invented = want == "get_order_status" and bool(FACT_WORDS.search(text))
    return {"decision": dec, "params": params, "execution": executed if calls else None, "resume": resume,
            "invented": invented}


def score_tenant(r: dict, scen: dict, ts) -> dict:
    sc = scen[r["scenario"]]
    rec = {"scenario": r["scenario"], "lang": r["lang"], "variant": r["arm"], "calls": r["calls"],
           "audit": r["audit"], "agent_text": r["agent_text"]}
    s = ts.score(sc, rec)
    ck = s["checks"]
    g = sc["gold"]
    exp = set(g.get("expect_tools", []))
    names = [c["name"] for c in r["calls"]]
    dec = ck["tools_expected_called"] and ck["no_unexpected_tool"]
    params = None
    if r["calls"]:
        params = ck.get("slots_grounded", True) and ck.get("resolved_correct_customer", True) and \
            ck.get("catalog_query_ok", True) and ck.get("no_wrong_customer", True)
    executed = (all(c["executed"] for c in r["calls"]) and all(l["status"] == "completed" for l in r["ledger"])) \
        if r["calls"] else None
    says = [v for k, v in ck.items() if k.startswith("says:")]
    resume = None
    if r["calls"]:
        resume = bool(r["replies"]) and not r["log"][-1].get("toolcall") and all(says)
    invented = (not ck.get("slots_grounded", True)) or any(not v for k, v in ck.items() if k.startswith("not_says"))
    return {"decision": dec, "params": params, "execution": executed, "resume": resume, "invented": invented,
            "clarification": ck.get("asks_clarification"), "lab_pass": s["pass"], "lab_checks": ck,
            "expected": sorted(exp), "called": names, "review": s["review"]}


def main():
    run, lab = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(lab / "agentbench"))
    sys.path.insert(0, str(lab / "multilingual"))
    import tenant_score as ts
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    scen = {s["id"]: s for s in spec["scenarios"]}
    out = {"rows": [], "summary": {}}
    for p in sorted(run.glob("rows_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            s = score_orders(r) if r["domain"] == "orders" else score_tenant(r, scen, ts)
            s.update({"arm": r["arm"], "lang": r["lang"], "domain": r["domain"], "scenario": r["scenario"],
                      "reply_lang": lang_id(r["agent_text"])[0], "calls": [(c["name"], c["args"]) for c in r["calls"]],
                      "reply": r["agent_text"][:300], "turns": r["turns"]})
            out["rows"].append(s)
    for s in out["rows"]:
        k = f"{s['arm']}|{s['lang']}"
        a = out["summary"].setdefault(k, {"n": 0})
        a["n"] += 1
        for f in ("decision", "params", "execution", "resume", "invented"):
            if s[f] is not None:
                a.setdefault(f, [0, 0])
                a[f][0] += bool(s[f])
                a[f][1] += 1
        a.setdefault("reply_lang_ok", [0, 0])
        a["reply_lang_ok"][0] += s["reply_lang"] == s["lang"]
        a["reply_lang_ok"][1] += 1
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
