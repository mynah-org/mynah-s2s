"""D1 pilot, step 1 (CPU): deterministic clip sample and slot WORD spans (owner GO 2026-11-13, card d1-pilot-card.md).

From the train_v1 inputs (strict-ASR ok, not excluded, split train, family admitted), 80 clips per language:
  pc        30  find_customer with a DIGIT-BY-DIGIT postcode (v2 'digits', v2b 'digits'): 5 digit words + first name
  name      10  status_name_only (first name, no postcode)
  concept   25  catalog with a concept spoken in its base surface (no synonym)
  capacity  15  s4x catalog_capacity (two-seat / three-seat)
Families are drawn so that IT and FR share them whenever both clips are usable (cross-language invariance pairs).
Words = the reference text lower-cased and split on anything that is not a letter or digit (hyphens and apostrophes
split words; they are not spoken). Slot spans are word-index ranges; the aligner turns them into times.
Usage: d1_sample.py --manifest m.jsonl --teacher t.json --out d1_sample.json [--seed 20261113]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon_v2 as L  # noqa: E402
import postcode_v2b as P  # noqa: E402

QUOTA = {"pc": 30, "name": 10, "concept": 25, "capacity": 15}


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[^0-9a-zà-ÿœæç]+", text.lower()) if w]


def find_sub(ws: list[str], sub: list[str], start: int = 0) -> int | None:
    for i in range(start, len(ws) - len(sub) + 1):
        if ws[i:i + len(sub)] == sub:
            return i
    return None


def concept_surface(concept: str, lang: str) -> str:
    c = next(x for x in L.CONCEPTS if x[0] == concept)
    return c[0] if lang == "en" else (c[2][0] if lang == "it" else c[4][0])


def capacity_forms(lang: str) -> dict[str, list[str]]:
    out = {}
    for a in L.ATTRS:
        if a[0] in ("two-seat", "three-seat"):
            out[a[0]] = [a[0]] if lang == "en" else list(a[2] if lang == "it" else a[3])
    return out


def slots_of(r: dict, kind: str) -> list[dict] | None:
    ws, s, lang = words(r["text"]), r["slots"] or {}, r["lang"]
    spans = []
    if kind in ("pc", "name"):
        first = words(s["name"])[0]
        i = find_sub(ws, [first])
        if i is None:
            return None
        spans.append({"task": "first_name", "label": s["name"].split()[0], "w0": i, "w1": i + 1})
    if kind == "pc":
        sp = words(P.spoken(s["postcode"], lang, "digits"))
        i = find_sub(ws, sp)
        if i is None or len(sp) != 5:
            return None
        for k, d in enumerate(s["postcode"]):
            spans.append({"task": "digit", "label": d, "pos": k, "w0": i + k, "w1": i + k + 1})
    if kind == "concept":
        sw = words(concept_surface(s["concept"], lang))
        i = find_sub(ws, sw)
        if i is None:
            return None
        spans.append({"task": "concept", "label": s["concept"], "category": s.get("category"), "w0": i,
                      "w1": i + len(sw)})
    if kind == "capacity":
        hit = None
        for lab, forms in capacity_forms(lang).items():
            if lab in s.get("attrs", []):
                for f in forms:
                    i = find_sub(ws, words(f))
                    if i is not None:
                        hit = {"task": "capacity", "label": lab, "w0": i, "w1": i + len(words(f))}
                        break
        if hit is None:
            return None
        spans.append(hit)
    return spans


def kind_of(r: dict) -> str | None:
    s, cat = r["slots"] or {}, r["cat"]
    if cat in ("s4:status_id", "s4:return_id") and s.get("postcode") and \
            (s.get("postcode_reading") or s.get("postcode_form")) == "digits":
        return "pc"
    if cat == "s4:status_name_only" and s.get("name"):
        return "name"
    if cat == "s4:catalog" and s.get("concept") and not any((s.get("synonym") or {}).values()):
        return "concept"
    if cat == "s4:catalog_capacity":
        return "capacity"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(a.manifest))}
    items = json.load(open(a.teacher))["items"]
    ok = {k for k, r in man.items() if r.get("asr_ok", True) and not r.get("excluded") and r["split"] == "train"
          and items.get(k[0], {}).get("keep")}
    rng = random.Random(a.seed)
    fams = sorted({i for i, _ in ok})
    rng.shuffle(fams)
    fams.sort(key=lambda i: not ((i, "it") in ok and (i, "fr") in ok))   # IT+FR pairs first (stable)
    got = {l: Counter() for l in ("en", "it", "fr")}
    out = []
    for i in fams:
        for l in ("en", "it", "fr"):
            k = (i, l)
            if k not in ok:
                continue
            r = man[k]
            kd = kind_of(r)
            if kd is None or got[l][kd] >= QUOTA[kd]:
                continue
            sp = slots_of(r, kd)
            if not sp:
                continue
            got[l][kd] += 1
            out.append({"id": i, "lang": l, "kind": kd, "path": r["path"], "text": r["text"], "words": words(r["text"]),
                        "voice": r.get("voice"), "cat": r["cat"], "target": r.get("target"), "slots_meta": r["slots"],
                        "spans": sp, "dur": r["dur"]})
    Path(a.out).write_text(json.dumps({"seed": a.seed, "quota": QUOTA, "clips": out}, ensure_ascii=False, indent=1))
    pairs = len({c["id"] for c in out if c["lang"] == "it"} & {c["id"] for c in out if c["lang"] == "fr"})
    print(json.dumps({"clips": len(out), "per_lang": {l: dict(c) for l, c in got.items()}, "it_fr_same_family": pairs,
                      "spans": dict(Counter(s["task"] for c in out for s in c["spans"]))}))


if __name__ == "__main__":
    main()
