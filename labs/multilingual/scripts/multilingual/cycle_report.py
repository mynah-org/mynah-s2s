"""L1/L2 cycle report (CPU): paired free-running comparison of LoRA checkpoints with the step-103 baseline (sil=last,
same sets / order / batch 8 / prompt / decoder), and the pre-registered stop rules and signal criterion of
l0-thinker-lora-card.md section 9 (opener check = opener_valid, owner 2026-11-13; opener_exact always reported).

Rules per checkpoint (vs baseline, counts on the same clips):
  STOP  opener: opener_valid count below baseline, or more clips with BOS argmax inside the caller turn than baseline
  STOP  EN: G-LEX-dev EN exact more than 1 below baseline
  STOP  memorisation: IT+FR held-out exact (G-LEX-dev + val) below baseline while train-fit exact >= baseline + 5
  SIGNAL (operational, not statistical): IT+FR G-LEX-dev exact >= baseline + 3 AND val IT+FR exact not below baseline
Usage: cycle_report.py --rows a.jsonl b.jsonl ... --baseline s103_last --arms l1_s30,l1_s60,l1_s120 --out report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glex_free import summarise
from glex_free_report import paired


def count(rows, pred, **flt):
    sel = [r for r in rows if all((r[k] in v if isinstance(v, (list, tuple, set)) else r[k] == v) for k, v in flt.items())]
    return sum(pred(r) for r in sel), len(sel)


def exact(r):
    return r["score"]["exact"]


def rules(base: list[dict], arm: list[dict]) -> dict:
    keys = {(r["set"], r["id"], r["lang"]) for r in base} & {(r["set"], r["id"], r["lang"]) for r in arm}
    b = [r for r in base if (r["set"], r["id"], r["lang"]) in keys]
    a = [r for r in arm if (r["set"], r["id"], r["lang"]) in keys]
    m = {}
    for name, rows in (("base", b), ("arm", a)):
        m[name] = {
            "opener_valid": count(rows, lambda r: r["beh"].get("opener_valid", r["beh"]["opener_exact"])),
            "opener_exact": count(rows, lambda r: r["beh"]["opener_exact"]),
            "bos_in_turn_clips": count(rows, lambda r: r["beh"]["bos_in_turn"] > 0),
            "en_glex_exact": count(rows, exact, set="glex_dev", lang="en"),
            "itfr_glex_exact": count(rows, exact, set="glex_dev", lang=("it", "fr")),
            "itfr_val_exact": count(rows, exact, set="val", lang=("it", "fr")),
            "itfr_heldout_exact": count(rows, exact, set=("glex_dev", "val"), lang=("it", "fr")),
            "trainfit_exact": count(rows, exact, set="trainfit"),
            "en_val_exact": count(rows, exact, set="val", lang="en"),
            "right_tool": count(rows, lambda r: r["score"]["tool"] == r["target"]["name"]),
            "no_call": count(rows, lambda r: r["beh"]["sotc_frame"] is None),
            "sotc_early": count(rows, lambda r: r["beh"]["sotc_early"]),
        }
    B, A = m["base"], m["arm"]
    stop = {
        "opener": A["opener_valid"][0] < B["opener_valid"][0] or A["bos_in_turn_clips"][0] > B["bos_in_turn_clips"][0],
        "en_regression": A["en_glex_exact"][0] < B["en_glex_exact"][0] - 1,
        "memorisation": A["itfr_heldout_exact"][0] < B["itfr_heldout_exact"][0]
        and A["trainfit_exact"][0] >= B["trainfit_exact"][0] + 5,
    }
    signal = A["itfr_glex_exact"][0] >= B["itfr_glex_exact"][0] + 3 and A["itfr_val_exact"][0] >= B["itfr_val_exact"][0]
    return {"n_paired": len(keys), "counts": {k: {"base": f"{B[k][0]}/{B[k][1]}", "arm": f"{A[k][0]}/{A[k][1]}"} for k in B},
            "stop": stop, "any_stop": any(stop.values()), "signal": signal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--baseline", default="s103_last")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in a.rows:
        for ln in open(p):
            d = json.loads(ln)
            if "score" in d:
                rows.append(d)
    base = [r for r in rows if r["arm"] == a.baseline]
    seen, b2 = set(), []
    for r in base:                                        # one row per clip (jobs 349 + 351 together)
        k = (r["set"], r["id"], r["lang"])
        if k not in seen:
            seen.add(k)
            b2.append(r)
    base = b2
    rep = {"baseline": a.baseline, "n_baseline": len(base), "arms": {}}
    rep["summary_baseline"] = summarise(base)
    for arm in a.arms.split(","):
        ar = [r for r in rows if r["arm"] == arm]
        if not ar:
            rep["arms"][arm] = "no rows"
            continue
        p = paired(base, ar)
        rep["arms"][arm] = {"rules": rules(base, ar), "summary": summarise(ar),
                            "paired": {k: v for k, v in p.items() if k != "sotc_shift"}}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("baseline", a.baseline, len(base))
    for arm, v in rep["arms"].items():
        if isinstance(v, str):
            print(arm, v)
            continue
        r = v["rules"]
        print(f"== {arm} paired {r['n_paired']} stop {r['stop']} signal {r['signal']}")
        for k, c in r["counts"].items():
            print(f"   {k:20s} base {c['base']:7s} arm {c['arm']}")
        p = v["paired"]
        print("   exact gained", len(p["exact_b_only"]), "lost", len(p["exact_a_only"]), "same call", p["same_call_text"],
              "tool changed", len(p["tool_changed"]), "call presence changed", len(p["call_presence_changed"]))


if __name__ == "__main__":
    main()
