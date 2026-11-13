"""Deterministic scoring of e2e_qa rows for corpus V2 (CPU). Never overwrites.

strict(row) = regex answer hit on the normalised agent text AND not a refusal AND non-empty (rules from E4/E5;
the manual exclusions used there were refusals, enumerations and echoes; the first two are automated here, echoes
are reported by phenomenon `recall` separately).
Reports per variant x language: strict accuracy, raw regex accuracy, refusal rate, empty rate; per phenomenon;
paired sign tests between variants on the same (id, lang); RNNT transcript WER/CER vs the reference question
text (ML RNNT, language tags removed; identical across dual-encoder variants by construction, checked);
per-frame step latency (ms_p50 median) per variant.
Usage: score_v2.py --rows a.jsonl [b.jsonl ...] --corpus qa_v2.json --out report.json [--ref-variant SUBSTR]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus_v2 import is_refusal  # noqa: E402


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def words(s):
    return re.sub(r"[^\w\s']", " ", norm(re.sub(r"<[a-z]{2,3}-[A-Z]{2}>", " ", s))).split()


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sign_p(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def short(v):
    m = re.search(r"distill/([^/]+)/", v)
    return "vc" if v == "vc" else (m.group(1) if m else v[:40])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-ids", default="", help="comma list of ids to drop (e.g. contaminated)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    cj = json.load(open(a.corpus))
    corpus = {it["id"]: it for it in cj["items"]}
    v21 = cj.get("version") == "qa_v21"  # gold forms are matched with corpus_v21.gold_hit (punctuation-free norm)
    if v21:
        from corpus_v21 import gold_hit
    excl = set(filter(None, a.exclude_ids.split(",")))
    rows = []
    for f in a.rows:
        for l in open(f):
            r = json.loads(l)
            if r["id"] in excl or r["id"] not in corpus:
                continue
            t = r["agent_text"]
            hit = (gold_hit(re.sub(r"</?s>", " ", t), corpus[r["id"]]["answers"]) if v21
                   else any(re.search(x, norm(t)) for x in corpus[r["id"]]["answers"]))
            r["v"] = short(r["variant"])
            r["regex"] = hit
            r["refusal"] = is_refusal(t)
            r["empty"] = len(words(re.sub(r"</?s>", " ", t))) == 0
            r["strict"] = hit and not r["refusal"] and not r["empty"]
            rows.append(r)
    rep = {"corpus": a.corpus, "rows": a.rows, "n_rows": len(rows), "variants": {}, "paired": {}, "rnnt": {},
           "latency_ms_p50": {}}
    by = defaultdict(list)
    for r in rows:
        by[(r["v"], r["lang"])].append(r)
    for (v, lang), rs in sorted(by.items()):
        d = rep["variants"].setdefault(v, {})
        d[lang] = {"n": len(rs), "strict": sum(r["strict"] for r in rs), "regex": sum(r["regex"] for r in rs),
                   "refusal": sum(r["refusal"] for r in rs), "empty": sum(r["empty"] for r in rs),
                   "strict_acc": round(sum(r["strict"] for r in rs) / len(rs), 3)}
        ph = defaultdict(list)
        for r in rs:
            ph[corpus[r["id"]].get("cat", r.get("cat", ""))].append(r["strict"])
        d[lang]["by_phenomenon"] = {k: f"{sum(x)}/{len(x)}" for k, x in sorted(ph.items())}
    vs = sorted({r["v"] for r in rows})
    key = {(r["v"], r["id"], r["lang"]): r for r in rows}
    for i, v1 in enumerate(vs):
        for v2 in vs[i + 1:]:
            res = {}
            for lang in sorted({r["lang"] for r in rows}):
                ks = [(rid, lg) for (v, rid, lg) in key if v == v1 and lg == lang and (v2, rid, lg) in key]
                b = sum(key[(v1, *k)]["strict"] and not key[(v2, *k)]["strict"] for k in ks)
                c = sum(key[(v2, *k)]["strict"] and not key[(v1, *k)]["strict"] for k in ks)
                if ks:
                    res[lang] = {"n": len(ks), f"{v1}_only": b, f"{v2}_only": c, "sign_p": round(sign_p(b, c), 4)}
            ne = [x for l, x in res.items() if l != "en"]
            if ne:
                b = sum(x[f"{v1}_only"] for x in ne)
                c = sum(x[f"{v2}_only"] for x in ne)
                res["non_en_pooled"] = {f"{v1}_only": b, f"{v2}_only": c, "sign_p": round(sign_p(b, c), 4)}
            rep["paired"][f"{v1} vs {v2}"] = res
    # RNNT transcript quality (reference = question text)
    for (v, lang), rs in sorted(by.items()):
        if v == "vc":
            continue
        we = sum(ed(words(r["question"]), words(r["user_asr"])) for r in rs)
        wn = sum(max(1, len(words(r["question"]))) for r in rs)
        ce = sum(ed(list("".join(words(r["question"]))), list("".join(words(r["user_asr"])))) for r in rs)
        cn = sum(max(1, len("".join(words(r["question"])))) for r in rs)
        rep["rnnt"].setdefault(v, {})[lang] = {"wer": round(we / wn, 4), "cer": round(ce / cn, 4), "n": len(rs)}
    same = Counter()
    for (v, rid, lg), r in key.items():
        for v2 in vs:
            if v2 != v and v2 != "vc" and v != "vc" and (v2, rid, lg) in key:
                same[r["user_asr"] == key[(v2, rid, lg)]["user_asr"]] += 1
    rep["rnnt_identical_across_dual_variants"] = dict(same)
    for v in vs:
        ms = sorted(r["ms_p50"] for r in rows if r["v"] == v)
        rep["latency_ms_p50"][v] = {"median_of_clip_p50": ms[len(ms) // 2] if ms else None, "n": len(ms)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    with open(out.with_suffix(".rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("v", "id", "lang", "cat", "strict", "regex", "refusal", "empty",
                                                  "agent_text", "user_asr")}, ensure_ascii=False) + "\n")
    for v, d in rep["variants"].items():
        print(v, {l: f"{x['strict']}/{x['n']}" for l, x in d.items()})
    for k, v in rep["paired"].items():
        print(k, v.get("non_en_pooled"))


if __name__ == "__main__":
    main()
