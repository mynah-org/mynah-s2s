"""Report of a glex_free run (CPU): per-arm summaries, the pre-registered harness gate on the official VC arm, and the
paired case-by-case comparison of two arms (e.g. step 103 with sil=last vs sil=first) on the clips both scored.

Harness gate v2 (owner, 2026-11-13, new L1/L2 cycle): identical checks with opener_valid (the two official opener token
sequences) in place of opener_exact; v1 (opener_exact) stays reported and historical results are not rewritten.
Harness gate v1 (fixed 2026-11-13 before the run, owner: "if the official EN check fails, the baseline is not validated"):
on the VC arm's EN rows: right tool >= 17/24 (teacher agreement on the train corpus 1101/1300 = 85 %, margin for n = 24
and the unconstrained decoder), opener exact >= 23/24, no SOTC before the caller starts, >= 90 % of produced calls parse.
Exact is reported, not gated. The gate scales to the actual number of EN rows (17/24, 23/24 proportionally, ceil).
Usage: glex_free_report.py --jsonl free_rows.jsonl --out report.json [--gate-arm vc] [--pair s103_last,s103_first]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from glex_free import summarise


def harness_gate(rows: list[dict], opener_key: str = "opener_exact") -> dict:
    en = [r for r in rows if r["lang"] == "en"]
    n = len(en)
    if not n:
        return {"n": 0, "pass": False, "why": "no EN rows"}
    called = [r for r in en if r["beh"]["sotc_frame"] is not None]
    g = {"n": n,
         "right_tool": sum(r["score"]["tool"] == r["target"]["name"] for r in en),
         "need_right_tool": math.ceil(17 * n / 24),
         "opener_key": opener_key,
         "opener_exact": sum(r["beh"].get(opener_key, r["beh"]["opener_exact"]) for r in en),
         "need_opener": math.ceil(23 * n / 24),
         "sotc_before_cs": sum(r["beh"]["sotc_before_cs"] for r in en),
         "parsed": sum(r["score"]["parsed"] for r in called), "called": len(called),
         "exact": sum(r["score"]["exact"] for r in en)}
    g["checks"] = {"right_tool": g["right_tool"] >= g["need_right_tool"], "opener": g["opener_exact"] >= g["need_opener"],
                   "no_early_sotc": g["sotc_before_cs"] == 0,
                   "parse": g["called"] == 0 or g["parsed"] >= 0.9 * g["called"]}
    g["pass"] = all(g["checks"].values())
    return g


def paired(ra: list[dict], rb: list[dict]) -> dict:
    """Case-by-case comparison on the clips both arms scored (same set, id, language)."""
    kb = {(r["set"], r["id"], r["lang"]): r for r in rb}
    both = [(r, kb[(r["set"], r["id"], r["lang"])]) for r in ra if (r["set"], r["id"], r["lang"]) in kb]
    out = {"n": len(both), "same_call_text": 0, "exact_a_only": [], "exact_b_only": [], "tool_changed": [],
           "call_presence_changed": [], "sotc_shift": [], "opener_changed": 0, "bos_changed": 0, "differ": []}
    for a, b in both:
        key = f"{a['set']}|{a['id']}|{a['lang']}"
        out["same_call_text"] += a["call"] == b["call"]
        if a["score"]["exact"] and not b["score"]["exact"]:
            out["exact_a_only"].append(key)
        if b["score"]["exact"] and not a["score"]["exact"]:
            out["exact_b_only"].append(key)
        if a["score"]["tool"] != b["score"]["tool"]:
            out["tool_changed"].append(key)
        if (a["beh"]["sotc_frame"] is None) != (b["beh"]["sotc_frame"] is None):
            out["call_presence_changed"].append(key)
        if a["beh"]["sotc_frame"] is not None and b["beh"]["sotc_frame"] is not None:
            out["sotc_shift"].append(b["beh"]["sotc_frame"] - a["beh"]["sotc_frame"])
        out["opener_changed"] += a["beh"]["opener_exact"] != b["beh"]["opener_exact"]
        out["bos_changed"] += (a["beh"]["bos_in_turn"] > 0) != (b["beh"]["bos_in_turn"] > 0)
        if a["call"] != b["call"]:
            out["differ"].append({"clip": key, "a": a["call"][10:120], "b": b["call"][10:120],
                                  "a_exact": a["score"]["exact"], "b_exact": b["score"]["exact"]})
    s = out["sotc_shift"]
    out["sotc_shift_summary"] = {"n": len(s), "nonzero": sum(x != 0 for x in s),
                                 "max_abs": max((abs(x) for x in s), default=0)}
    out["exact_a"] = sum(a["score"]["exact"] for a, _ in both)
    out["exact_b"] = sum(b["score"]["exact"] for _, b in both)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-arm", default="vc")
    ap.add_argument("--pair", action="append", default=[], help="A,B (repeatable)")
    a = ap.parse_args()
    meta, checks, batches, stops, rows = None, [], [], [], []
    for ln in open(a.jsonl):
        d = json.loads(ln)
        if "meta" in d:
            meta = d["meta"]
        elif "batch" in d:
            batches.append(d["batch"])
        elif "stop" in d:
            stops.append(d)
        elif "sil_mode" in d:
            checks.append(d)
        elif "score" in d:
            rows.append(d)
    arms = sorted({r["arm"] for r in rows})
    rep = {"meta": meta, "silence_checks": checks, "stops": stops, "batches_last": {x: [b for b in batches if b["arm"] == x][-1:]
                                                                                    for x in arms},
           "summaries": {x: summarise([r for r in rows if r["arm"] == x]) for x in arms}}
    rep["harness_gate"] = harness_gate([r for r in rows if r["arm"] == a.gate_arm])               # v1, historical
    rep["harness_gate_v2_opener_valid"] = harness_gate([r for r in rows if r["arm"] == a.gate_arm], "opener_valid")
    rep["paired"] = {}
    for pr in a.pair:
        pa, pb = pr.split(",")
        rep["paired"][f"{pa}_vs_{pb}"] = paired([r for r in rows if r["arm"] == pa], [r for r in rows if r["arm"] == pb])
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("HARNESS GATE v1 (opener_exact)", json.dumps(rep["harness_gate"]))
    print("HARNESS GATE v2 (opener_valid)", json.dumps(rep["harness_gate_v2_opener_valid"]))
    for x in arms:
        print("==", x)
        for k, v in rep["summaries"][x].items():
            print(f"  {k:20s} exact {v['exact']:7s} tool {v['right_tool']:7s} nocall {v['no_call']:6s} early {v['sotc_early']:6s} "
                  f"opener {v['opener_exact']:7s} valid {v.get('opener_valid', '-'):7s} bos {v['bos_in_turn_clips']:6s} slots {v['slots_strict']}")
    for k, p in rep["paired"].items():
        print("PAIRED", k, {kk: v for kk, v in p.items() if kk not in ("differ", "sotc_shift")})


if __name__ == "__main__":
    main()
