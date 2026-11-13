"""A1 report (CPU): text-oracle results per arm x head x condition x language on the held-out items (G-LEX-dev + val):
call made, right tool, exact, strict slots, query canonicalisation (canonical EN / source language kept / other), and the
paired gold_own vs rnnt differences per item.
Usage: a1_report.py --jsonl a1_rows.jsonl --out a1_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [d for d in map(json.loads, open(a.jsonl)) if "score" in d]
    stops = [d for d in map(json.loads, open(a.jsonl)) if "stop" in d]
    rep = {"stops": stops, "groups": {}, "paired_gold_vs_rnnt": {}}
    keys = sorted({(r["arm"], r["head"], r["cond"]) for r in rows})
    for arm, head, cond in keys:
        for lang in ("en", "it", "fr"):
            sel = [r for r in rows if (r["arm"], r["head"], r["cond"], r["lang"]) == (arm, head, cond, lang)]
            if not sel:
                continue
            sl = defaultdict(lambda: [0, 0])
            for r in sel:
                for k, v in r["score"]["slots"].items():
                    sl[k][0] += v["strict"]
                    sl[k][1] += 1
            n = len(sel)
            rep["groups"][f"{arm}|{head}|{cond}|{lang}"] = {
                "n": n, "called": f"{sum(r['score']['parsed'] for r in sel)}/{n}",
                "right_tool": f"{sum(r['score']['tool'] == r['target']['name'] for r in sel)}/{n}",
                "exact": f"{sum(r['score']['exact'] for r in sel)}/{n}",
                "slots": {k: f"{v[0]}/{v[1]}" for k, v in sorted(sl.items())},
                "query_kind": dict(Counter(r["query_kind"] for r in sel if r["query_kind"]))}
    for arm, head in sorted({(r["arm"], r["head"]) for r in rows}):
        g = {(r["set"], r["id"], r["lang"]): r for r in rows if (r["arm"], r["head"], r["cond"]) == (arm, head, "gold_own")}
        t = {(r["set"], r["id"], r["lang"]): r for r in rows if (r["arm"], r["head"], r["cond"]) == (arm, head, "rnnt")}
        both = sorted(set(g) & set(t))
        if both:
            rep["paired_gold_vs_rnnt"][f"{arm}|{head}"] = {
                "n": len(both), "gold_only_exact": [f"{k[1]}|{k[2]}" for k in both if g[k]["score"]["exact"] and not t[k]["score"]["exact"]],
                "rnnt_only_exact": [f"{k[1]}|{k[2]}" for k in both if t[k]["score"]["exact"] and not g[k]["score"]["exact"]]}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("stops", stops)
    for k, v in rep["groups"].items():
        print(f"{k:32s} called {v['called']:6s} tool {v['right_tool']:6s} exact {v['exact']:6s} slots {v['slots']} q {v['query_kind']}")
    for k, v in rep["paired_gold_vs_rnnt"].items():
        print("PAIRED", k, v["n"], "gold-only exact", len(v["gold_only_exact"]), "rnnt-only exact", len(v["rnnt_only_exact"]))


if __name__ == "__main__":
    main()
