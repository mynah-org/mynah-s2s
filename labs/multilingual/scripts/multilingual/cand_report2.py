"""Report of the forced-candidate test, run 2 (CPU). Reads cand_run2.jsonl (cand_test.py, batch 1, paired arms).

Per (arm, clip, distractor), with g / d the per-token function log-probs of the gold / distractor call and fd the first
differing token:  total = sum(g) - sum(d)   norm = mean(g[fd:-1]) - mean(d[fd:-1])   (EOTC excluded)
Only COMPLETE pairs enter: the clip has a record for every arm with every condition. Reported separately: pairwise
accuracy (share total > 0) and margins per arm and condition, audio - neutral, groups by slot x language, and the
paired flips between the two arms (arm A prefers gold and arm B does not, and vice versa).
A positive margin is a teacher-forced preference, NOT evidence that the model would generate the value on its own.
Usage: cand_report2.py --jsonl cand_run2.jsonl --out report.json [--arms s103,s200]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(xs):
    xs = np.asarray(xs, float)
    if not len(xs):
        return None
    return {"n": int(len(xs)), "acc_pos": f"{int((xs > 0).sum())}/{len(xs)}", "median": round(float(np.median(xs)), 2),
            "q25": round(float(np.percentile(xs, 25)), 2), "q75": round(float(np.percentile(xs, 75)), 2)}


def margins(rec, c):
    g_t, d_t = rec["toks"]["0"], rec["toks"][str(c)]
    fd = next((k for k, (p, q) in enumerate(zip(g_t, d_t)) if p != q), min(len(g_t), len(d_t)))
    out = {}
    for cond, lp in rec["logp"].items():
        g, d = lp["0"], lp[str(c)]
        out[cond] = {"total": sum(g) - sum(d), "norm": float(np.mean(g[fd:-1])) - float(np.mean(d[fd:-1]))}
    return out, len(g_t), len(d_t), fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="s103,s200")
    a = ap.parse_args()
    arms = a.arms.split(",")
    meta, smoke, stops, recs = None, None, [], defaultdict(dict)
    for ln in open(a.jsonl):
        d = json.loads(ln)
        if "meta" in d:
            meta = d["meta"]
        elif "smoke" in d:
            smoke = d["smoke"]
        elif "stop" in d:
            stops.append(d)
        else:
            recs[d["clip"]][d["arm"]] = d
    conds = meta["conds"]
    complete = [c for c, v in recs.items() if all(x in v and set(conds) <= set(v[x]["logp"]) for x in arms)]
    groups = defaultdict(lambda: defaultdict(list))
    flips, rows = [], []
    for ci in sorted(complete, key=meta["order"].index):
        base = recs[ci][arms[0]]
        for c in range(1, len(base["cands"])):
            cd = base["cands"][c]
            m = {x: margins(recs[ci][x], c) for x in arms}
            lg, ld, fd = m[arms[0]][1:]
            slot = cd["slot"]
            row = {"clip": f"{base['id']}|{base['lang']}", "slot": slot, "kind": cd["kind"], "distractor": cd["value"],
                   "len_gold": lg, "len_dist": ld, "first_diff": fd}
            for x in arms:
                for cond in conds:
                    row[f"{x}_{cond}_total"] = round(m[x][0][cond]["total"], 3)
                    row[f"{x}_{cond}_norm"] = round(m[x][0][cond]["norm"], 3)
            rows.append(row)
            for k in ("all", f"slot:{slot}", f"lang:{base['lang']}", f"slot_lang:{slot}|{base['lang']}",
                      f"slot_kind:{slot}|{cd['kind']}"):
                for x in arms:
                    for mm in ("total", "norm"):
                        for cond in conds:
                            groups[k][f"{x}|{cond}|{mm}"].append(m[x][0][cond][mm])
                        if "audio" in conds and "neutral" in conds:
                            groups[k][f"{x}|audio-neutral|{mm}"].append(m[x][0]["audio"][mm] - m[x][0]["neutral"][mm])
                groups[k][f"{arms[1]}-{arms[0]}|audio|total"].append(m[arms[1]][0]["audio"]["total"]
                                                                     - m[arms[0]][0]["audio"]["total"])
            a0, a1 = m[arms[0]][0]["audio"]["total"] > 0, m[arms[1]][0]["audio"]["total"] > 0
            if a0 != a1:
                flips.append({**row, "direction": f"{arms[0]}_only" if a0 else f"{arms[1]}_only"})
    flip_counts = defaultdict(lambda: defaultdict(int))
    for f in flips:
        flip_counts[f"slot:{f['slot']}"][f["direction"]] += 1
        flip_counts["all"][f["direction"]] += 1
    rep = {"meta": meta, "smoke": smoke, "stops": stops, "clips_recorded": len(recs), "clips_complete": len(complete),
           "pairs_complete": len(rows), "groups": {k: {mm: stats(x) for mm, x in g.items()} for k, g in sorted(groups.items())},
           "flip_counts": {k: dict(v) for k, v in flip_counts.items()}, "flips": flips, "rows": rows}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("smoke", smoke, "stops", stops)
    print("clips recorded", len(recs), "complete", len(complete), "pairs", len(rows))
    for k in sorted(groups, key=lambda k: (k != "all", k)):
        g = groups[k]
        f = lambda mm: (g[mm] and stats(g[mm])["acc_pos"], g[mm] and stats(g[mm])["median"])  # noqa: E731
        print(f"{k:24s}", " | ".join(f"{x}: aud {f(f'{x}|audio|total')} neu {f(f'{x}|neutral|total')} "
                                     f"a-n {f(f'{x}|audio-neutral|total')}" for x in arms))
    print("flips", {k: dict(v) for k, v in flip_counts.items()})


if __name__ == "__main__":
    main()
