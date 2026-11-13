"""CPU diff of a live session's observe `logits` rows against a T2 replay's per-position dump (same positions).

For every position: |delta| of logP for text PAD/BOS and function PAD/SOTC (and EOTC where present), argmax
agreement, and the replay's consumed inputs. Reports the first position whose max |delta| exceeds each threshold, the
growth profile per segment (main 0..133, fc1 134..165, fc2 166..229, free 230..), and the decisive margins.
Usage: live_vs_replay.py <observe.jsonl> <t2.json> [--cell D-V2.1:own] [--out diff.json]
"""
from __future__ import annotations

import argparse
import json

KEYS = (("text", "PAD"), ("text", "BOS"), ("func", "PAD"), ("func", "SOTC"), ("func", "EOTC"))


def seg(p):
    return "main" if p <= 133 else "fc1" if p <= 165 else "fc2" if p <= 229 else "free"


def diff(live_rows, positions, upto=231):
    live = {}
    for r in live_rows:
        if r.get("kind") == "logits" and r.get("t") is not None:
            live.setdefault(r["t"], r)
    out = []
    for d in positions:
        p = d["pos"]
        if p > upto or p not in live:
            continue
        L = live[p]
        deltas = {f"{h}_{k}": round(d[h][k] - L[h][k], 4) for h, k in KEYS if k in d.get(h, {}) and k in L.get(h, {})}
        out.append({"pos": p, "seg": seg(p), "max_abs": max(abs(v) for v in deltas.values()) if deltas else None,
                    "deltas": deltas, "argmax_text_same": d["pred_text"] == L["pred_text"],
                    "argmax_func_same": d["pred_func"] == L["pred_func"],
                    "live_pred": [L["pred_text"], L["pred_func"]], "replay_pred": [d["pred_text"], d["pred_func"]],
                    "inputs": {k: d.get(k) for k in ("agent_in", "func_in", "audio", "v2_pos_before")}})
    return out


def summarise(rows):
    res = {"positions_compared": len(rows)}
    for thr in (0.05, 0.25, 0.5, 1.0):
        f = next((r for r in rows if r["max_abs"] is not None and r["max_abs"] > thr), None)
        res[f"first_over_{thr}"] = None if f is None else {k: f[k] for k in ("pos", "seg", "max_abs", "deltas")}
    res["first_argmax_diff"] = next(({k: r[k] for k in ("pos", "seg", "live_pred", "replay_pred")} for r in rows
                                     if not (r["argmax_text_same"] and r["argmax_func_same"])), None)
    prof = {}
    for r in rows:
        if r["max_abs"] is None:
            continue
        s = prof.setdefault(r["seg"], {"n": 0, "mean": 0.0, "max": 0.0, "argmax_diffs": 0})
        s["n"] += 1
        s["mean"] += r["max_abs"]
        s["max"] = max(s["max"], r["max_abs"])
        s["argmax_diffs"] += int(not (r["argmax_text_same"] and r["argmax_func_same"]))
    for s in prof.values():
        s["mean"] = round(s["mean"] / max(s["n"], 1), 4)
    res["profile"] = prof
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("observe")
    ap.add_argument("t2")
    ap.add_argument("--cell", default="D-V2.1:own")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arm, kind = a.cell.split(":")
    t2 = json.load(open(a.t2))
    cell = next(c for c in t2["cells"] if c["arm"] == arm and c.get("kind") == kind)
    rows = diff([json.loads(x) for x in open(a.observe)], cell["positions"])
    res = {"summary": summarise(rows), "rows": rows}
    print(json.dumps(res["summary"], indent=1))
    for r in rows:
        if r["pos"] in (0, 1, 19, 94, 132, 133, 134, 135, 150, 165, 166, 167, 200, 228, 229, 230, 231):
            print(r["pos"], r["seg"], r["max_abs"], r["deltas"], r["live_pred"], r["replay_pred"], r["inputs"])
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
