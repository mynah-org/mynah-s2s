"""Report of the forced-candidate test (CPU). Reads cand_results.json and candidates.json (cand_test.py).

Per (arm, clip, distractor): with g / d the per-token function log-probs of the gold / distractor call and fd the first
differing token,
  total   = sum(g) - sum(d)                                  (shared prefix cancels)
  norm    = mean(g[fd:-1]) - mean(d[fd:-1])                  (EOTC excluded; per-token, length-robust)
for the conditions audio / neutral / swap; audio_contrib = audio - neutral. Pairwise accuracy = share of total > 0.
Grouped by source (glex_dev / val), slot, distractor kind, language; inversion examples kept.
Usage: cand_report.py --dir results/v4/cand_test --out report.json
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
    return {"n": int(len(xs)), "acc_pos": round(float((xs > 0).mean()), 3), "median": round(float(np.median(xs)), 2),
            "q25": round(float(np.percentile(xs, 25)), 2), "q75": round(float(np.percentile(xs, 75)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    res = json.load(open(D / "cand_results.json"))
    rows = json.load(open(D / "candidates.json"))
    rep = {"smoke": res.get("smoke"), "arms": {}}
    for arm, v in res["arms"].items():
        if not isinstance(v, dict):
            rep["arms"][arm] = v
            continue
        groups = defaultdict(lambda: defaultdict(list))
        inv = []
        for i, r in enumerate(rows):
            lp = v["logp"].get(str(i), {})
            if "audio" not in lp:
                continue
            g_t = v["toks"][f"{i}|0"]
            for c in range(1, len(r["cands"])):
                d_t = v["toks"][f"{i}|{c}"]
                fd = next((k for k, (p, q) in enumerate(zip(g_t, d_t)) if p != q), min(len(g_t), len(d_t)))
                cd = r["cands"][c]
                vals = {}
                for cond in ("audio", "neutral", "swap"):
                    if cond in lp and str(0) in lp[cond] and str(c) in lp[cond]:
                        g, d = lp[cond]["0"], lp[cond][str(c)]
                        vals[cond] = {"total": sum(g) - sum(d),
                                      "norm": float(np.mean(g[fd:-1])) - float(np.mean(d[fd:-1]))}
                if "audio" not in vals:
                    continue
                keys = ["all", f"src:{r['src']}", f"slot:{cd['slot']}", f"slot_kind:{cd['slot']}|{cd['kind']}",
                        f"src_slot:{r['src']}|{cd['slot']}", f"lang:{r['lang']}", f"src_lang:{r['src']}|{r['lang']}"]
                for k in keys:
                    for m in ("total", "norm"):
                        groups[k][f"audio_{m}"].append(vals["audio"][m])
                        if "neutral" in vals:
                            groups[k][f"neutral_{m}"].append(vals["neutral"][m])
                            groups[k][f"audio_contrib_{m}"].append(vals["audio"][m] - vals["neutral"][m])
                        if "swap" in vals:
                            groups[k][f"swap_{m}"].append(vals["swap"][m])
                            groups[k][f"audio_paired_with_swap_{m}"].append(vals["audio"][m])
                if vals["audio"]["total"] < 0:
                    inv.append({"clip": f"{r['id']}|{r['lang']}", "src": r["src"], "slot": cd["slot"],
                                "kind": cd["kind"], "gold": r["target"]["arguments"].get(cd["slot"]),
                                "distractor": cd["value"], "audio_total": round(vals["audio"]["total"], 2),
                                "neutral_total": round(vals.get("neutral", {}).get("total", float("nan")), 2),
                                "len_gold": len(g_t), "len_dist": len(d_t)})
        rep["arms"][arm] = {"groups": {k: {m: stats(x) for m, x in g.items()} for k, g in sorted(groups.items())},
                            "inversions": inv}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("smoke", rep["smoke"])
    for arm, v in rep["arms"].items():
        if not isinstance(v, dict):
            print(arm, v)
            continue
        print("==", arm)
        for k in ("all", "src:glex_dev", "src:val", "src_slot:glex_dev|full_name", "src_slot:glex_dev|postcode",
                  "src_slot:glex_dev|query", "src_slot:val|full_name", "src_slot:val|postcode", "src_slot:val|query"):
            g = v["groups"].get(k)
            if not g:
                continue
            f = lambda m: (g[m]["acc_pos"], g[m]["median"]) if g.get(m) else None      # noqa: E731
            print(f"{k:28s} audio {f('audio_total')} neutral {f('neutral_total')} contrib {f('audio_contrib_total')} "
                  f"swap {f('swap_total')} | norm audio {f('audio_norm')} neutral {f('neutral_norm')}")


if __name__ == "__main__":
    main()
