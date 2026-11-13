"""Nano-9B answer judge (GPU) — SECONDARY metric only, calibrated first.

calibrate: judge the E5/a1r Ryan dev+held-out rows (QA v1) and compare with the manual strict labels of
  experiments.md (regex hit, minus d01, minus the item-level exclusions listed there). Reports agreement, Cohen's
  kappa, confusion and every disagreement.
apply: judge the rows of a V2 evaluation (QA v2 answers) -> per variant x language judge accuracy.
Prompt: question (English reference), expected answer, assistant reply -> "YES"/"NO" (greedy, /no_think).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = Path.home() / "mynah-s2s/ml/results"

# manual strict exclusions (experiments.md, E5 vs a1r sections): (run, variant, lang, id)
MANUAL_FP = {
    ("dev", "a1r"): {("it", "d06"), ("it", "d15"), ("fr", "d03"), ("fr", "d06"), ("fr", "d16"), ("es", "d06"),
                     ("es", "d15")},
    ("dev", "e5"): {("it", "d15"), ("it", "d16"), ("fr", "d03"), ("fr", "d04"), ("fr", "d16"), ("de", "d08"),
                    ("de", "d16"), ("es", "d06")},
    ("ho", "a1r"): {("it", "h15"), ("de", "h02"), ("de", "h04"), ("es", "h04")},
    ("ho", "e5"): {("fr", "h04"), ("fr", "h07"), ("de", "h04"), ("de", "h16"), ("es", "h04"), ("es", "h09")},
}

PROMPT = ("You are grading a voice assistant. The user asked (translated to English): \"{q}\"\n"
          "A correct answer is: \"{a}\"\nThe assistant replied: \"{r}\"\n"
          "Does the assistant's reply correctly answer the user's question (a refusal, an unrelated reply, or a reply "
          "that only repeats the question is NOT correct)? Answer with exactly one word: YES or NO.")


def judge(nano, triples):
    outs = nano.run([PROMPT.format(q=q, a=a, r=re.sub(r"</?s>", " ", r).strip()[:400]) for q, a, r in triples], 4)
    return [o.strip().upper().startswith("YES") for o in outs], outs


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "apply"])
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    from corpus_v2 import Nano
    from e2e_qa import score
    nano = Nano(a.batch)
    if a.mode == "calibrate":
        qa = {it["id"]: it for it in json.load(open(HERE / "qa_corpus.json"))["items"]}
        recs = []
        for run in ("dev", "ho"):
            for l in open(R / f"e2e-rv-a1r-e5-{run}/rows.jsonl"):
                r = json.loads(l)
                if r["id"] == "d01":
                    continue
                v = "e5" if "ln=" in r["variant"] else "a1r"
                hit = score(r["agent_text"], qa[r["id"]]["answers"])
                manual = hit and (r["lang"], r["id"]) not in MANUAL_FP[(run, v)]
                recs.append({"run": run, "v": v, "lang": r["lang"], "id": r["id"], "manual": manual,
                             "q": qa[r["id"]]["text"]["en"], "a": re.sub(r"\\b|[\^$?*+()\[\]|]", "", qa[r["id"]]["answers"][0]).replace(".", " "),
                             "reply": r["agent_text"]})
        j, raw = judge(nano, [(x["q"], x["a"], x["reply"]) for x in recs])
        for x, y, o in zip(recs, j, raw):
            x["judge"], x["judge_raw"] = y, o
        pairs = [(x["manual"], x["judge"]) for x in recs]
        rep = {"n": len(recs), "agreement": round(sum(m == g for m, g in pairs) / len(pairs), 4),
               "kappa": round(kappa(pairs), 4), "confusion(manual,judge)": {str(k): v for k, v in Counter(pairs).items()},
               "by_lang": {l: round(sum(x["manual"] == x["judge"] for x in recs if x["lang"] == l)
                                    / sum(1 for x in recs if x["lang"] == l), 3) for l in sorted({x["lang"] for x in recs})},
               "disagreements": [{k: x[k] for k in ("run", "v", "lang", "id", "manual", "judge", "reply")}
                                 for x in recs if x["manual"] != x["judge"]]}
    else:
        corpus = {it["id"]: it for it in json.load(open(a.corpus))["items"]}
        recs = []
        for f in a.rows:
            for l in open(f):
                r = json.loads(l)
                it = corpus.get(r["id"])
                if not it:
                    continue
                recs.append({"v": r["variant"], "lang": r["lang"], "id": r["id"], "q": it["text"]["en"],
                             "a": it.get("answer_text") or ", ".join(it.get("kw", [])), "reply": r["agent_text"]})
        j, raw = judge(nano, [(x["q"], x["a"], x["reply"]) for x in recs])
        acc = defaultdict(list)
        for x, y in zip(recs, j):
            x["judge"] = y
            acc[(x["v"], x["lang"])].append(y)
        rep = {"n": len(recs), "acc": {f"{v}|{l}": f"{sum(x)}/{len(x)}" for (v, l), x in sorted(acc.items())},
               "rows": [{k: x[k] for k in ("v", "lang", "id", "judge")} for x in recs]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("disagreements", "rows")}, indent=1))


if __name__ == "__main__":
    main()
