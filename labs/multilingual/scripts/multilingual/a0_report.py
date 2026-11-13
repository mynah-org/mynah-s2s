"""A0 report (CPU): per encoder/decoder arm x language (held-out G-LEX-dev + val together and per set), the three scoring
layers kept apart: entity (strict), meaning (lenient), canonical EN string; plus mean CER.
Usage: a0_report.py --jsonl a0_rows.jsonl --out a0_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [d for d in map(json.loads, open(a.jsonl)) if "res" in d]
    arms = list(rows[0]["res"])
    rep = {}
    for arm in arms:
        for grp in ("all", "glex_dev", "val"):
            for lang in ("en", "it", "fr"):
                sel = [r for r in rows if r["lang"] == lang and (grp == "all" or r["set"] == grp)]
                if not sel:
                    continue
                c = defaultdict(lambda: [0, 0])
                cer = []
                for r in sel:
                    x = r["res"][arm]
                    cer.append(x["cer"])
                    for layer in ("entity", "meaning"):
                        for k, v in x[layer].items():
                            c[f"{layer}:{k}"][0] += bool(v)
                            c[f"{layer}:{k}"][1] += 1
                    if x["canonical_en"] is not None:
                        c["canonical_en:query"][0] += bool(x["canonical_en"])
                        c["canonical_en:query"][1] += 1
                rep[f"{arm}|{grp}|{lang}"] = {"n": len(sel), "cer_mean": round(sum(cer) / len(cer), 3),
                                              **{k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())}}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    keys = ["entity:name", "meaning:name", "entity:postcode", "meaning:postcode", "entity:concept", "meaning:concept",
            "entity:attrs", "canonical_en:query"]
    print(f"{'arm|set|lang':24s} {'n':>3s} {'cer':>6s} " + " ".join(f"{k.split(':')[0][:3]}:{k.split(':')[1][:5]:5s}" for k in keys))
    for k, v in rep.items():
        if "|all|" in k:
            print(f"{k:24s} {v['n']:3d} {v['cer_mean']:6.3f} " + " ".join(f"{v.get(x, '-'):>9s}" for x in keys))


if __name__ == "__main__":
    main()
