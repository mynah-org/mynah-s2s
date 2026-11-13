"""Re-score the existing V2.1 predictions (E5 vs V2.1) with gold matcher v2 (gold_v21). CPU only, read-only inputs.

Per split (dev, test, uncond) x language x variant: CORRECT / WRONG / REFUSAL (refusal takes precedence, as in the
strict score), echo items excluded; paired 3x3 transition matrices E5 -> V2.1; sign test on CORRECT; examples of
the most significant transitions; agreement with the blinded manual labels of the old automatic hits.
Usage: score_v21.py --out ml/results/v21/rescore_v2.json   (never overwrites)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from labpaths import MYNAH_HOME  # noqa: E402  (MYNAH_HOME / MYNAH_LAB_HOME, no hard-coded home layout)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus_v2 import is_refusal  # noqa: E402
from gold_v21 import echo, hit, spec  # noqa: E402

HOME = MYNAH_HOME
D, R = HOME / "ml/data/v21", HOME / "ml/results/v21"
SPLITS = {"dev": "e2e-dev", "test": "e2e-test", "uncond": "e2e-uncond"}
# blinded manual labels of the OLD automatic hits (manual_test.json / manual_devuncond.json): indices judged false
MANUAL = {"manual_test_key.json": {2, 6, 13, 14, 15, 17, 19, 27, 28},
          "manual_devuncond_key.json": {2, 5, 11, 13, 22, 24, 25, 33, 39}}


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def short(v):
    return "e5" if "/distill/e5/" in v else ("v21" if "/v21/" in v else v[:20])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    qa = {it["id"]: it for it in json.load(open(D / "qa_v21.json"))["items"]}
    fam = {json.loads(l)["id"]: json.loads(l) for l in open(D / "families.jsonl")}
    sp = {i: spec(it["cat"], fam[i]["gold_text"] or "", fam[i]["gold_aliases"]) for i, it in qa.items()}
    ech = {i for i, it in qa.items() if echo(it["text"]["en"], sp[i])}
    rows = {}
    for s, d in SPLITS.items():
        for l in open(R / d / "rows.jsonl"):
            r = json.loads(l)
            t = r["agent_text"]
            cls = "REFUSAL" if is_refusal(t) else ("CORRECT" if hit(t, sp[r["id"]]) else "WRONG")
            rows[(s, short(r["variant"]), r["id"], r["lang"])] = {"cls": cls, "text": t}
    rep = {"matcher": "gold_v21 (unit-tested, test_gold_v21.py)", "echo_excluded": sorted(ech), "splits": {}}
    for s in SPLITS:
        S = rep["splits"][s] = {"by_variant": {}, "paired": {}, "examples": {}}
        langs = sorted({k[3] for k in rows if k[0] == s})
        ids = sorted({k[2] for k in rows if k[0] == s} - ech)
        for v in ("e5", "v21"):
            S["by_variant"][v] = {l: dict(Counter(rows[(s, v, i, l)]["cls"] for i in ids if (s, v, i, l) in rows))
                                  for l in langs}
            for l in langs:
                S["by_variant"][v][l]["n"] = sum(1 for i in ids if (s, v, i, l) in rows)
        pool_b = pool_c = 0
        for l in langs:
            m = Counter()
            ex = defaultdict(list)
            for i in ids:
                if (s, "e5", i, l) in rows and (s, "v21", i, l) in rows:
                    x, y = rows[(s, "e5", i, l)], rows[(s, "v21", i, l)]
                    m[f"{x['cls']}->{y['cls']}"] += 1
                    if x["cls"] != y["cls"] and len(ex[f"{x['cls']}->{y['cls']}"]) < 3:
                        ex[f"{x['cls']}->{y['cls']}"].append({"id": i, "q": qa[i]["text"]["en"][:120],
                                                              "gold": qa[i]["answer_text"],
                                                              "e5": x["text"].replace("<s>", "")[:200],
                                                              "v21": y["text"].replace("<s>", "")[:200]})
            b = sum(n for k, n in m.items() if k.startswith("CORRECT->") and not k.endswith("CORRECT"))
            c = sum(n for k, n in m.items() if k.endswith("->CORRECT") and not k.startswith("CORRECT"))
            pool_b += b if l != "en" else 0
            pool_c += c if l != "en" else 0
            S["paired"][l] = {"matrix": dict(sorted(m.items())), "e5_only_correct": b, "v21_only_correct": c,
                              "sign_p": round(sign_p(b, c), 4)}
            S["examples"][l] = dict(ex)
        S["paired"]["non_en_pooled"] = {"e5_only_correct": pool_b, "v21_only_correct": pool_c,
                                        "sign_p": round(sign_p(pool_b, pool_c), 4)}
    # agreement with the manual labels (only the old automatic hits were labelled)
    agree = Counter()
    for kf, F in MANUAL.items():
        for k in json.load(open(R / kf)):
            s = k.get("split", "test")
            r = rows.get((s, k["v"], k["id"], k["lang"]))
            if r is None or k["id"] in ech:
                agree["echo_or_missing"] += 1
                continue
            manual = k["n"] not in F
            agree[f"manual_{'true' if manual else 'false'}->new_{r['cls']}"] += 1
    rep["manual_agreement_on_old_hits"] = dict(sorted(agree.items()))
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    for s, S in rep["splits"].items():
        print(s, {v: {l: (x.get("CORRECT", 0), x.get("REFUSAL", 0), x["n"]) for l, x in d.items()}
                  for v, d in S["by_variant"].items()}, S["paired"]["non_en_pooled"])
    print("manual agreement", rep["manual_agreement_on_old_hits"], "echo", len(ech))


if __name__ == "__main__":
    main()
