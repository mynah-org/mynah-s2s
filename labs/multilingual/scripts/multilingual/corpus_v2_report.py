"""Corpus v2 decision-checkpoint report (CPU only). Three measures kept apart (owner, 2026-11-12):

  coverage  : clips / families per intent x language, strict-ASR ok, usable (= asr_ok and not excluded);
  quality   : teacher admission on the decoded families (agree / teacher_no_call / quarantine), per intent and per
              pre-registered bucket (language x duration, phrases, attrs, synonym, postcode form);
  exposure  : what the trainer would actually see. `session_distill.train` draws every batch uniformly over the
              (family, lang) train keys (`rng.sample(tr, batch)`), so the per-intent exposure share equals the share of
              usable admitted train keys. For families not decoded yet, the projection assumes the sample's per-intent
              admission rate (labelled as a projection).

Usage: corpus_v2_report.py --manifest m.jsonl --teacher teacher.json [--source s4] --out report.json
Gates (thresholds unchanged, card §6/§12): IT/FR strict ASR >= 90 % over all IT/FR clips; family admission >= 75 %;
every bucket >= 60 %.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def dur_bucket(d: float) -> str:
    return "<4s" if d < 4 else "4-8s" if d < 8 else "8-12s" if d < 12 else ">=12s"


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def report(man: list[dict], items: dict) -> dict:
    by_key = {(r["id"], r["lang"]): r for r in man}
    fam = {}
    for r in man:
        fam.setdefault(r["id"], r)
    usable = {k for k, r in by_key.items() if r.get("asr_ok", True) and not r.get("excluded")}

    # coverage
    cov = defaultdict(Counter)
    for (i, l), r in by_key.items():
        c = cov[f"{r['cat']}|{l}"]
        c["n"] += 1
        c["asr_ok"] += bool(r.get("asr_ok", True))
        c["usable"] += (i, l) in usable
        if r["split"] in ("train", "val"):
            c[f"usable_{r['split']}"] += (i, l) in usable
    itfr = [r for r in man if r["lang"] in ("it", "fr")]
    asr_gate = {"ok": sum(bool(r.get("asr_ok", True)) for r in itfr), "n": len(itfr)}
    asr_gate["pct"] = pct(asr_gate["ok"], asr_gate["n"])
    asr_gate["per_lang"] = {l: pct(sum(bool(r.get("asr_ok", True)) for r in itfr if r["lang"] == l),
                                   sum(r["lang"] == l for r in itfr)) for l in ("it", "fr")}
    fail_cause = Counter(f"{r['lang']}:{(r.get('cause_history') or ['?'])[-1]}" for r in man if not r.get("asr_ok", True))
    cov_fam = Counter()
    for i, r in fam.items():
        if any((i, l) in usable for l in ("en", "it", "fr")):
            cov_fam[r["cat"]] += 1

    # quality on the decoded families
    q_int = defaultdict(Counter)
    buckets = defaultdict(Counter)
    for i, v in items.items():
        r = fam[i]
        q_int[r["cat"]][v["reason"] if not v["keep"] else ("kept:" + v["reason"])] += 1
        s = r.get("slots") or {}
        fb = {"intent": r["cat"], "phrases": str(r.get("n_phrases")),
              "attrs": str(len(s.get("attrs") or [])) if s.get("concept") else None,
              "synonym": ("yes" if any((s.get("synonym") or {}).values()) else "no") if s.get("concept") else None,
              "postcode_form": s.get("postcode_form")}
        for b, val in fb.items():
            if val is not None:
                buckets[f"{b}={val}"]["n"] += 1
                buckets[f"{b}={val}"]["kept"] += bool(v["keep"])
        for l in ("en", "it", "fr"):
            m = by_key.get((i, l))
            if m is None:
                continue
            k = f"lang_dur={l}|{dur_bucket(m['dur'])}"
            buckets[k]["n"] += 1
            buckets[k]["kept"] += bool(v["keep"]) and (i, l) in usable
    bk = {k: {"kept": c["kept"], "n": c["n"], "pct": pct(c["kept"], c["n"])} for k, c in sorted(buckets.items())}
    adm = sum(bool(v["keep"]) for v in items.values())

    # exposure: decoded (actual) and projected over every train family
    rate = {c: sum(n for k, n in q.items() if k.startswith("kept:")) / sum(q.values()) for c, q in q_int.items()}
    actual, proj = Counter(), Counter()
    for (i, l) in usable:
        r = by_key[(i, l)]
        if r["split"] != "train":
            continue
        if i in items:
            actual[r["cat"]] += bool(items[i]["keep"])
        proj[r["cat"]] += rate.get(r["cat"], 1.0) if i not in items else bool(items[i]["keep"])
    tot_a, tot_p = sum(actual.values()), sum(proj.values())
    expo = {"sampler": "uniform over (family, lang) train keys",
            "decoded_train_keys": {c: {"keys": n, "share_pct": pct(n, tot_a)} for c, n in actual.most_common()},
            "projected_full_train_keys": {c: {"keys": round(n, 1), "share_pct": pct(n, tot_p)}
                                          for c, n in proj.most_common()}}
    return {
        "coverage": {"clips_per_intent_lang": {k: dict(v) for k, v in sorted(cov.items())},
                     "families_with_any_usable_clip": dict(cov_fam),
                     "itfr_strict_asr": asr_gate, "failed_clips_final_cause": dict(fail_cause)},
        "quality": {"families_decoded": len(items), "admitted": adm, "admission_pct": pct(adm, len(items)),
                    "per_intent": {k: dict(v) for k, v in sorted(q_int.items())}, "buckets": bk,
                    "buckets_below_60": {k: v for k, v in bk.items() if v["pct"] is not None and v["pct"] < 60}},
        "exposure": expo,
        "gates": {"itfr_asr_ge_90": asr_gate["pct"] >= 90, "admission_ge_75": pct(adm, len(items)) >= 75,
                  "every_bucket_ge_60": not any(v["pct"] < 60 for v in bk.values() if v["pct"] is not None)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--source", default=None, help="keep only families whose id starts with this prefix + '-'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    man = [json.loads(x) for x in open(a.manifest)]
    if a.source:
        man = [r for r in man if r["id"].startswith(a.source + "-")]
    items = json.load(open(a.teacher))["items"]
    rep = report(man, items)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
