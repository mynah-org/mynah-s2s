"""Corpus v2 (CPU): merge several teacher.json files into one (owner GO 2026-11-13, full teacher + training prep).

The teachers must share opener_end and tail_s (same prompt, same session layout). Families present in several inputs
are taken from the FIRST input that has them (priority = order of --teacher). Every merged item records its origin in
`teacher_src`. Admission / quarantine fields are kept as decided by the run that decoded the family; `end` is
recomputed later by `prep` / at train time from the manifest.
Usage: merge_teachers.py --teacher a.json --teacher b.json ... --out merged.json [--ids-only list.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def merge(teachers: list[tuple[str, dict]]) -> dict:
    oe = {d["opener_end"] for _, d in teachers}
    tail = {d["tail_s"] for _, d in teachers}
    if len(oe) != 1 or len(tail) != 1:
        raise SystemExit(f"incompatible teachers: opener_end {oe}, tail_s {tail}")
    items, src = {}, Counter()
    for name, d in teachers:
        for i, v in d["items"].items():
            if i in items:
                src[f"shadowed:{name}"] += 1
                continue
            items[i] = dict(v, teacher_src=name)
            src[name] += 1
    first = teachers[0][1]
    return {"prompt": first.get("prompt"), "opener_end": oe.pop(), "tail_s": tail.pop(), "items": items,
            "merge": dict(src)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    ts = [(p, json.load(open(p))) for p in a.teacher]
    m = merge(ts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, ensure_ascii=False))
    print(json.dumps({"items": len(m["items"]), "merge": m["merge"],
                      "reasons": dict(Counter(v["reason"] for v in m["items"].values()))}))


if __name__ == "__main__":
    main()
