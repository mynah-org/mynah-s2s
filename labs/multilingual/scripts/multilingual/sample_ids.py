"""Corpus v2 decision checkpoint (owner 2026-11-12): the representative teacher sample (CPU, deterministic).
  forced : the 4 pilot transfer families (escalate fix) + every NEW out-of-scope family
  rest   : NEW families (not in the pilot plan), train/val, stratified by intent proportionally to the full plan,
           drawn with a fixed seed; EN clip verification is NOT used to pick (no selection bias toward clean clips).
Usage: sample_ids.py --full plan_v4_full.jsonl --pilot plan_v4_pilot.jsonl --n 100 --out sample_ids.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def pick(full: list[dict], pilot_ids: set, n: int, seed: int) -> list[str]:
    fam = {r["id"]: r for r in full if r["lang"] == "en" and r["split"] in ("train", "val")}
    forced = sorted(i for i, r in fam.items() if r["cat"] == "s4:transfer" and i in pilot_ids)
    forced += sorted(i for i, r in fam.items() if r["cat"] == "s4:out_of_scope" and i not in pilot_ids)
    new = defaultdict(list)
    for i, r in sorted(fam.items()):
        if i not in pilot_ids and i not in forced:
            new[r["cat"]].append(i)
    left = n - len(forced)
    tot = sum(len(v) for v in new.values())
    quota = {c: max(1, round(left * len(v) / tot)) for c, v in new.items()}
    while sum(quota.values()) > left:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < left:
        quota[max(quota, key=lambda c: len(new[c]) - quota[c])] += 1
    rng = random.Random(seed)
    out = list(forced)
    for c in sorted(new):
        out += sorted(rng.sample(new[c], min(quota[c], len(new[c]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    full = [json.loads(x) for x in open(a.full)]
    pilot = {json.loads(x)["id"] for x in open(a.pilot)}
    ids = pick(full, pilot, a.n, a.seed)
    cat = {r["id"]: r["cat"] for r in full if r["lang"] == "en"}
    json.dump(ids, open(a.out, "w"))
    print(len(ids), dict(Counter(cat[i] for i in ids)))


if __name__ == "__main__":
    main()
