"""Corpus v2 (CPU, read-only): distribution of the clips that can actually enter training, and the FR gaps.

A clip is TRAINABLE when strict ASR is ok, it is not excluded (session_distill.usable_keys) and its split is train/val.
The teacher admission of the full corpus is not known yet (no full teacher); the §17 sample admitted 98 %.
Per language: intent, concept, name, postcode class (digits / grouped clear / grouped vigesimal 70-99 / grouped
ambiguous), duration bucket. A cell is a FR GAP when FR keeps < 75 % of what was planned AND FR keeps at least 10
points less than IT for the same cell (so generic TTS loss is not reported as a FR gap).
Also checks that every excluded clip is out of the training keys built by session_distill.kept_keys.
Usage: coverage_audit.py --manifest m.jsonl [--manifest s4x.jsonl] --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_distill as S  # noqa: E402


def pc_class(slots: dict, lang: str) -> str:
    f = slots.get("postcode_form")
    if not f:
        return "none"
    if f == "digits":
        return "digits"
    pc = slots["postcode"]
    a, b, c = int(pc[:2]), int(pc[2]), int(pc[3:])
    if b and ((a >= 20 and a % 10 == 0) or (lang == "fr" and a in (10, 70, 90) and b >= 7)):
        return "grouped_ambiguous"
    if lang == "fr" and (a >= 70 or c >= 70):
        return "grouped_vigesimal"
    return "grouped_clear"


def dur_bucket(d: float) -> str:
    return "<4s" if d < 4 else "4-8s" if d < 8 else "8-12s" if d < 12 else ">=12s"


def audit(man: dict) -> dict:
    use = S.usable_keys(man)
    tab = defaultdict(lambda: defaultdict(Counter))   # dim -> value -> (lang, planned|trainable)
    for (i, l), r in man.items():
        if r["split"] not in ("train", "val"):
            continue
        s = r.get("slots") or {}
        src = "s4x" if i.startswith("s4x-") else "s4"
        dims = {"intent": f"{src}:{r['cat']}", "concept": s.get("concept"), "name": s.get("name"),
                "postcode": pc_class(s, l), "duration": dur_bucket(r["dur"]),
                "intent_x_postcode": f"{r['cat']}|{pc_class(s, l)}"}
        for d, v in dims.items():
            if v is None:
                continue
            tab[d][v][(l, "planned")] += 1
            tab[d][v][(l, "trainable")] += (i, l) in use
    out, gaps = {}, []
    for d, vals in tab.items():
        out[d] = {}
        for v, c in sorted(vals.items(), key=lambda x: str(x[0])):
            row = {}
            for l in ("en", "it", "fr"):
                p, t = c[(l, "planned")], c[(l, "trainable")]
                row[l] = {"trainable": t, "planned": p, "pct": round(100 * t / p, 1) if p else None}
            out[d][str(v)] = row
            fr, it = row["fr"]["pct"], row["it"]["pct"]
            if fr is not None and it is not None and fr < 75 and it - fr >= 10 and row["fr"]["planned"] >= 3:
                gaps.append({"dim": d, "value": str(v), "fr": row["fr"], "it": row["it"]})
    # exclusion leak check: the trainer's own key builder, every family admitted
    items = {i: {"keep": True} for i, _ in man}
    split = {i: r["split"] for (i, _), r in man.items()}
    tr, va = S.kept_keys(items, use, split, ["en", "it", "fr"])
    keys = set(tr) | set(va)
    excl = [k for k, r in man.items() if r.get("excluded")]
    leak = [k for k in excl if k in keys]
    per_reason = Counter(r["excluded"] for r in man.values() if r.get("excluded"))
    totals = {l: {"trainable": sum(1 for k in keys if k[1] == l),
                  "planned": sum(1 for (i, ll), r in man.items() if ll == l and r["split"] in ("train", "val"))}
              for l in ("en", "it", "fr")}
    return {"totals_train_val": totals, "tables": out, "fr_gaps": sorted(gaps, key=lambda g: g["fr"]["pct"]),
            "exclusion_check": {"excluded": len(excl), "by_reason": dict(per_reason), "in_training_keys": len(leak),
                                "leaked": [f"{i}|{l}" for i, l in leak][:10]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    man = {}
    for p in a.manifest:
        for r in map(json.loads, open(p)):
            man[(r["id"], r["lang"])] = r
    rep = audit(man)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("totals_train_val", "exclusion_check")}, indent=1))
    for g in rep["fr_gaps"]:
        print("FR GAP", g["dim"], g["value"], g["fr"], "IT", g["it"]["pct"])


if __name__ == "__main__":
    main()
