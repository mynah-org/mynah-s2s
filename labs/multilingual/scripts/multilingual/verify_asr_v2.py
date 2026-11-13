"""Corpus v2 clip verification (CPU): does the RENDERED AUDIO, as heard by OmniASR, contain every target slot?

Input: the plan rows (text, target, slots) and asr_qa.py rows.jsonl (hyp per clip). Per clip:
  concept   : the concept noun or the synonym actually used (accent-insensitive, token overlap >= 0.8);
  attributes: every said attribute surface form;
  name      : every name token with character similarity >= 0.75;
  postcode  : the exact 5 digits, parsed from the hypothesis (spoken digits, grouped numbers or numerals).
A clip failing any check is INVALID for its (family, language); the family's other languages stay. Reasons are kept.
Usage: verify_asr_v2.py --plan plan.jsonl --asr rows.jsonl --out verify.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402
import postcode_v2b as P  # noqa: E402


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", s))


def contains(hyp: str, phrase: str, thr: float = 0.8) -> bool:
    h, p = fold(hyp).split(), fold(phrase).split()
    if not p:
        return True
    if " ".join(p) in " ".join(h):
        return True
    n = len(p)                                          # fuzzy: best window of the same length
    best = 0.0
    for i in range(max(1, len(h) - n + 1)):
        best = max(best, difflib.SequenceMatcher(None, " ".join(h[i:i + n]), " ".join(p)).ratio())
    return best >= thr


def name_ok(hyp: str, name: str, thr: float = 0.75) -> bool:
    h = fold(hyp).split()
    return all(any(difflib.SequenceMatcher(None, t, w).ratio() >= thr for w in h) for t in fold(name).split())


def check(row: dict, hyp: str, max_k: int = 3) -> list[str]:
    s, t, lang = row["slots"], row.get("target"), row["lang"]
    bad = []
    if s.get("concept") and t and t.get("name") == "search_catalog":
        c = next(x for x in L.CONCEPTS if x[0] == s["concept"])
        syn = (s.get("synonym") or {}).get(lang)
        surface = syn or (c[0] if lang == "en" else (c[2][0] if lang == "it" else c[4][0]))
        if not contains(hyp, surface):
            bad.append("concept")
        for an in s.get("attrs", []):
            a = next(x for x in L.ATTRS if x[0] == an)
            forms = [a[0]] if lang == "en" else list(a[2] if lang == "it" else a[3])
            if not any(contains(hyp, f) for f in forms):
                bad.append(f"attr:{an}")
    if t and t.get("name") == "find_customer":
        if s.get("name") and not name_ok(hyp, s["name"]):
            bad.append("name")
        pc = (t.get("arguments") or {}).get("postcode")
        if pc and s.get("postcode_reading"):        # corpus v2b: unique 5-digit acoustic parse (postcode_v2b)
            if not P.hyp_has_postcode(hyp, lang, pc):
                bad.append("postcode")
        elif pc:                                    # unhyphenated ASR numbers are ambiguous: accept any reading
            reads = {L.parse_spoken_digits(hyp, lang, max_k=max_k), L.parse_spoken_digits(hyp, lang, True, max_k)}
            spoken = [L.spoken_postcode(pc, lang, g) for g in (False, True)]
            if not any(pc in r for r in reads) and not any(fold(sp) in fold(hyp) for sp in spoken):
                bad.append("postcode")
    return bad


def classify(bad: list[str], hyp: str, cer, man_row: dict | None, slots: dict, lang: str) -> str:
    """Cause of a failing clip, kept separately (owner): an ASR failure is NOT assumed to be a TTS failure.
      tts_verified : objective TTS evidence: runaway length (> 39 s), speaking rate outside 8-25 chars/s, or the whole
                     utterance unreadable (CER > 0.35)
      asr_probable : only slot checks fail, the rest of the utterance is clean (CER <= 0.15), and the slot shows an ASR
                     artefact: mixed letter/digit tokens or a <= 1-digit miss for postcodes, a phonetically close
                     spelling (token similarity >= 0.6) for names
      uncertain    : anything else. Every class stays EXCLUDED from training (no lenient acceptance without evidence)."""
    if not bad:
        return "ok"
    cps = (man_row or {}).get("chars_per_s")
    if "tts_overlong" in bad or (cps is not None and not 8 <= cps <= 25) or (cer is not None and cer > 0.35):
        return "tts_verified"
    if cer is not None and cer <= 0.15 and all(b in ("name", "postcode") for b in bad):
        ok = True
        if "postcode" in bad:
            pc = slots.get("postcode") or ""
            mixed = bool(re.search(r"\d[a-zà-ü]|[a-zà-ü]\d", hyp.lower()))
            reads = [L.parse_spoken_digits(hyp, lang), L.parse_spoken_digits(hyp, lang, True), "".join(re.findall(r"\d", hyp))]
            near = any(sum(x != y for x, y in zip(r_[i:i + 5], pc)) <= 1 and len(r_[i:i + 5]) == 5
                       for r_ in reads for i in range(max(1, len(r_) - 4)))
            ok &= mixed or near
        if "name" in bad:
            h = fold(hyp).split()
            ok &= all(any(difflib.SequenceMatcher(None, t, w).ratio() >= 0.6 for w in h)
                      for t in fold(slots.get("name") or "").split())
        if ok:
            return "asr_probable"
    return "uncertain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="", help="TTS manifest (chars_per_s) for the TTS-cause evidence")
    ap.add_argument("--max-k", type=int, default=3, help="longest unhyphenated number form parsed (3 = frozen v2 "
                    "behaviour; 4 = also 'quatre vingt dix sept'; re-checking every v2 round with 4 changes 0 rows)")
    a = ap.parse_args()
    plan = {}
    for p in a.plan:
        for r in map(json.loads, open(p)):
            plan[(r["id"], r["lang"])] = r
    ap_man = {}
    if a.manifest:
        ap_man = {(r["id"], r["lang"]): r for r in map(json.loads, open(a.manifest))}
    rows = [json.loads(x) for x in open(a.asr)]
    skp = Path(a.asr).parent / "skipped.jsonl"
    if skp.exists():                                     # clips ASR could not take (> 40 s): invalid by definition
        rows += [dict(r, hyp="", ref="", cer=None, skipped=r.get("reason")) for r in map(json.loads, open(skp))]
    res, agg = {}, defaultdict(Counter)
    for r in rows:
        key = (r["id"], r["lang"])
        if key not in plan:
            continue
        bad = ["tts_overlong"] if r.get("skipped") else check(plan[key], r["hyp"], a.max_k)
        cause = classify(bad, r["hyp"], r.get("cer"), ap_man.get(key), plan[key]["slots"], key[1])
        res[f"{key[0]}|{key[1]}"] = {"ok": not bad, "bad": bad, "cause": cause, "hyp": r["hyp"], "ref": r["ref"],
                                     "cer": r.get("cer")}
        agg[f"cause:{key[1]}"][cause] += 1
        cat = plan[key]["cat"]
        for k in (f"lang:{key[1]}", f"cat:{cat}", f"split:{plan[key]['split']}"):
            agg[k]["n"] += 1
            agg[k]["ok"] += int(not bad)
        agg[f"cause:{key[1]}"]["n"] += 0
        for b in bad:
            agg[f"lang:{key[1]}"][f"bad:{b.split(':')[0]}"] += 1
    rep = {"clips": len(res), "ok": sum(v["ok"] for v in res.values()),
           "by": {k: (dict(v) if k.startswith("cause:") else dict(v, ok_frac=round(v["ok"] / v["n"], 3)))
                  for k, v in sorted(agg.items())}, "rows": res}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
