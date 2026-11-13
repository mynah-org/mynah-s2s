"""Per-entity paired table (CPU): for every held-out clip (G-LEX-dev + val), each slot's gold value and the value each arm
generated free-running, strict match marked; per-arm counts by slot x language; paired gains / losses vs a reference arm.
Concept preservation (diagnostic) for queries. Usage:
  entity_table.py --rows a.jsonl b.jsonl ... --arms s103_last,l1_s120,b0_s120 --ref l1_s120 --out t.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glex_free import concept_preserved  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--sets", required=True, help="free_sets.json (slots for concept preservation)")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    sets = {(r["set"], r["id"], r["lang"]): r for r in json.load(open(a.sets))["rows"]}
    for r in sets.values():
        if isinstance(r["slots"], str):
            r["slots"] = eval(r["slots"])  # noqa: S307 - our own plan files
    got = {}
    for p in a.rows:
        for ln in open(p):
            d = json.loads(ln)
            if "score" in d and d["arm"] in arms and d["set"] in ("glex_dev", "val"):
                got.setdefault((d["arm"], d["set"], d["id"], d["lang"]), d)
    keys = sorted({k[1:] for k in got if all((x,) + k[1:] in got for x in arms)})
    table, counts, paired = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: {"gain": [], "loss": []})
    for k in keys:
        row = {"clip": "|".join(k), "gold": sets[k]["target"]["arguments"]}
        for arm in arms:
            d = got[(arm,) + k]
            sl = d["score"]["slots"]
            row[arm] = {s: [v["got"], v["strict"]] for s, v in sl.items()}
            for s, v in sl.items():
                counts[(arm, s, k[2])][0] += v["strict"]
                counts[(arm, s, k[2])][1] += 1
            if "query" in sl:
                cp = concept_preserved(sl["query"]["got"] if d["score"]["tool"] == "search_catalog" else None, sets[k])
                row[arm]["concept_preserved"] = cp
                counts[(arm, "concept_preserved", k[2])][0] += bool(cp)
                counts[(arm, "concept_preserved", k[2])][1] += 1
        for arm in arms:
            if arm == a.ref:
                continue
            for s in row["gold"]:
                r0, r1 = row[a.ref].get(s, [None, False])[1], row[arm].get(s, [None, False])[1]
                if r1 and not r0:
                    paired[f"{arm}|{s}"]["gain"].append(row["clip"])
                if r0 and not r1:
                    paired[f"{arm}|{s}"]["loss"].append(row["clip"])
        table.append(row)
    summ = {f"{arm}|{s}|{lg}": f"{v[0]}/{v[1]}" for (arm, s, lg), v in sorted(counts.items())}
    Path(a.out).write_text(json.dumps({"summary": summ, "paired_vs_ref": paired, "table": table}, indent=1, ensure_ascii=False))
    for lg in ("en", "it", "fr"):
        for s in ("full_name", "postcode", "query", "concept_preserved"):
            print(f"{lg} {s:18s}", "  ".join(f"{arm}: {summ.get(f'{arm}|{s}|{lg}', '-'):6s}" for arm in arms))
    for k, v in paired.items():
        print("PAIRED vs", a.ref, k, "gain", len(v["gain"]), "loss", len(v["loss"]))


if __name__ == "__main__":
    main()
