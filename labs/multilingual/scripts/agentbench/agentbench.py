"""agentbench v0 — deterministic gold checks for the voice-agent product benchmark (CPU only).

  validate                 schema + fixture consistency of domains.json (codes in gold exist in the fixture unless
                           the category is not_found; ISO dates; enum values; <= 5 tools; EN/IT/FR on every turn)
  plan --out FILE          TTS plan rows (scenario x turn x lang x voice), the input of render_qa.py --plan
  score --obs FILE         score observed sessions (JSONL: {scenario, lang, variant, calls:[{name,args}], agent_text})
Gold decides; nothing here asks an LLM. Checks that cannot be decided from text (must_not, refusal quality,
clarification wording) are reported as `review` items for a human, never auto-passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "multilingual"))
from gold_v21 import hit as gold_hit  # noqa: E402
from gold_v21 import parse_date  # noqa: E402

SPEC = HERE / "domains.json"
LANGS = ("en", "it", "fr")
SLOT_WORDS = {"tracking_code": ["tracking", "code"], "policy_number": ["policy", "number"], "order_id": ["order", "number"]}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c)).strip()


def norm_arg(k: str, v):
    v = fold(v)
    if k in ("tracking_code", "policy_number", "claim_id", "order_id"):
        return re.sub(r"[^a-z0-9]", "", v)
    if k.endswith("date"):
        d = parse_date(v)
        return "%04d-%02d-%02d" % d if d and d[1] and d[2] else v
    return re.sub(r"\s+", " ", v)


def args_match(gold: dict, got: dict) -> bool:
    for k, gv in gold.items():
        if gv == "*":
            if k not in got or not str(got[k]).strip():
                return False
            continue
        if k not in got or norm_arg(k, got[k]) != norm_arg(k, gv):
            return False
    return True


def fact_spec(f: str) -> dict:
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", f):
        return {"kind": "date", "ymd": parse_date(f)}
    if re.fullmatch(r"\d+", f):
        return {"kind": "number", "value": float(f), "unit": []}
    return {"kind": "phrase", "phrases": [" ".join(fold(f).split())]}


def score_one(sc: dict, obs: dict) -> dict:
    g = sc["gold"]
    calls = obs.get("calls", [])
    names = [c["name"] for c in calls]
    allow = set(g.get("allow_extra_calls", []))
    req = g["tool_calls"]
    res = {"scenario": sc["id"], "category": sc["category"], "checks": {}, "review": []}
    ck = res["checks"]
    ck["no_forbidden_tool"] = not any(n in g.get("forbidden_tools", []) for n in names)
    if not req:
        ck["no_call"] = all(n in allow for n in names)
    else:
        ok_sel, ok_args = True, True
        for r in req:
            mine = [c for c in calls if c["name"] == r["name"]]
            if not mine:
                ok_sel = ok_args = False
                continue
            target = mine[-1] if g.get("final_call_only") else mine[0]
            ok_args &= args_match(r["args"], target.get("args", {}))
        extra = [n for n in names if n not in {r["name"] for r in req} | allow]
        ck["tool_selected"] = ok_sel and not extra
        ck["args_correct"] = ok_args
        if "max_calls" in g:
            ck["max_calls"] = len(calls) <= g["max_calls"]
    for f in g.get("must_say", []):
        ck[f"says:{f}"] = gold_hit(obs.get("agent_text", ""), fact_spec(f))
    if g.get("expect_clarification"):
        words = SLOT_WORDS.get(g["expect_clarification"], [g["expect_clarification"]])
        t = fold(obs.get("agent_text", ""))
        ck["asks_for_slot"] = not calls and all(w in t for w in words)
        res["review"].append(f"clarification wording ({g['expect_clarification']})")
    if g.get("expect_refusal"):
        res["review"].append("refusal / redirect quality")
    for m in g.get("must_not", []):
        res["review"].append(f"must not: {m}")
    res["pass"] = all(ck.values())
    return res


def cmd_validate(spec):
    errs, n = [], Counter()
    for dn, d in spec["domains"].items():
        if len(d["tools"]) > 5:
            errs.append(f"{dn}: {len(d['tools'])} tools > 5")
        tools = {t["name"]: t for t in d["tools"]}
        fx = json.dumps(d["fixture"])
        for sc in d["scenarios"]:
            n[(dn, sc["category"])] += 1
            for t in sc["turns"]:
                if set(t) != set(LANGS):
                    errs.append(f"{sc['id']}: turn langs {sorted(t)}")
            for c in sc["gold"]["tool_calls"] + [{"name": x, "args": {}} for x in sc["gold"].get("forbidden_tools", [])]:
                if c["name"] not in tools:
                    errs.append(f"{sc['id']}: unknown tool {c['name']}")
                    continue
                props = tools[c["name"]]["parameters"]["properties"]
                for k, v in c["args"].items():
                    if k not in props:
                        errs.append(f"{sc['id']}: arg {k} not in {c['name']}")
                    if "enum" in props.get(k, {}) and v not in props[k]["enum"]:
                        errs.append(f"{sc['id']}: {k}={v} not in enum")
                    if k.endswith("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                        errs.append(f"{sc['id']}: date {v}")
                    if k in ("tracking_code", "policy_number", "claim_id", "order_id") and sc["category"] != "not_found" \
                            and v not in fx:
                        errs.append(f"{sc['id']}: {v} not in fixture")
                if sc["gold"]["tool_calls"] and c in sc["gold"]["tool_calls"]:
                    miss = set(tools[c["name"]]["parameters"]["required"]) - set(c["args"])
                    if miss:
                        errs.append(f"{sc['id']}: missing required {miss}")
    print(json.dumps({"scenarios": sum(n.values()), "by_domain_category": {f"{a}|{b}": v for (a, b), v in sorted(n.items())},
                      "errors": errs}, indent=1))
    return not errs


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("validate")
    p = sp.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--voices", default="ryan")
    p = sp.add_parser("score")
    p.add_argument("--obs", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(SPEC))
    if a.cmd == "validate":
        sys.exit(0 if cmd_validate(spec) else 1)
    if a.cmd == "plan":
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"{out} exists")
        rows = []
        for dn, d in spec["domains"].items():
            for sc in d["scenarios"]:
                for k, t in enumerate(sc["turns"]):
                    for l in LANGS:
                        for v in a.voices.split(","):
                            rows.append({"id": f"{sc['id']}_t{k}", "split": "agentbench", "cat": f"{dn}:{sc['category']}",
                                         "lang": l, "voice": v, "text": t[l]})
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(len(rows), "rows")
        return
    scen = {sc["id"]: sc for d in spec["domains"].values() for sc in d["scenarios"]}
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    res = [dict(score_one(scen[o["scenario"]], o), lang=o["lang"], variant=o.get("variant", "")) for o in map(json.loads, open(a.obs))]
    agg = defaultdict(Counter)
    for r in res:
        agg[(r["variant"], r["lang"])]["pass"] += r["pass"]
        agg[(r["variant"], r["lang"])]["n"] += 1
        for k, v in r["checks"].items():
            agg[(r["variant"], r["lang"])][k.split(":")[0] + ("_ok" if v else "_fail")] += 1
    out.write_text(json.dumps({"rows": res, "summary": {f"{v}|{l}": dict(c) for (v, l), c in sorted(agg.items())}},
                              indent=1, ensure_ascii=False))
    print(json.dumps({f"{v}|{l}": f"{c['pass']}/{c['n']}" for (v, l), c in sorted(agg.items())}))


if __name__ == "__main__":
    main()
