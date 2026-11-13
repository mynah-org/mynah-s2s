"""Corpus v2b (CPU, PROPOSAL; owner GO for design + CPU audit only, 2026-11-13): a separate, versioned supplement that
fills the FR gaps of corpus v2 (card §20) with postcodes a listener can decode without punctuation.

Nothing of corpus v2 is modified: `session_corpus_v2` (plans, G-LEX-dev/test), the `s4x` supplement and `idev` are only
IMPORTED (templates, noun phrases, splits, target rules). v2b has its own ids (`s4b-NNNN`), seed, plan file and version.
  intents   return_id 70, status_id 50, status_name_only 30 (the slot intents where FR loses most in v2);
  concepts  TRAIN concepts only, no held-out attribute x concept pair, never agentbench / G-LEX concepts;
  names     TRAIN names only (v2 name_split), minus the homophone diagnostic names (Margaux, Faure: their spelling
            cannot be recovered from the sound; kept in the v2 diagnostic, never generated here);
  postcodes 50 % grouped with the `d3s` reading (department as a number + 3 single digits, digit by digit when that is
            ambiguous; postcode_v2b: 0 ambiguous of 100 000 in EN/IT/FR), 50 % digit by digit. Grouped postcodes draw
            the department from 70-99 with probability 0.45 (natural share 0.30) so the FR vigesimal numbers that
            strict ASR keeps only ~40 % of the time stay represented after filtering.
Targets use the v2 rules unchanged (find_customer {full_name[, postcode]}), with the FIXED generator (synonym gender,
pronoun "elle").
Usage: session_corpus_v2b.py --out plan_v2b.jsonl --stats v2b_stats.json [--audit v2b_audit.json --frozen a.jsonl ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402
import postcode_v2b as P  # noqa: E402
import session_corpus_v2 as C  # noqa: E402

VERSION = "v2b.1"
SEED = 20261116
LANGS = C.LANGS
MIX = {"return_id": 70, "status_id": 50, "status_name_only": 30}
HOMOPHONE_DIAGNOSTIC = {"first": {"Margaux"}, "last": {"Faure"}}
P_GROUPED, P_VIGESIMAL_DEPT = 0.5, 0.45


def make_postcode(rng: random.Random, grouped: bool) -> str:
    while True:
        if grouped:                                   # d3s: department 70-99 with probability 0.45 (natural 0.30)
            dept = rng.randrange(70, 100) if rng.random() < P_VIGESIMAL_DEPT else rng.randrange(0, 70)
        else:                                         # digit by digit: natural, uniform 00-99
            dept = rng.randrange(0, 100)
        pc = f"{dept:02d}" + "".join(rng.choice("0123456789") for _ in range(3))
        if pc not in L.EXCL_PC:
            return pc


def pick_name(rng: random.Random) -> str:
    while True:
        first, last = rng.choice(L.FIRST), rng.choice(L.LAST)
        if first in HOMOPHONE_DIAGNOSTIC["first"] or last in HOMOPHONE_DIAGNOSTIC["last"]:
            continue
        if f"{first} {last}".lower() not in L.EXCL_NAMES and C.name_split(first, last) == "train":
            return f"{first} {last}"


def build(seed: int = SEED, mix: dict | None = None) -> list[dict]:
    rng = random.Random(seed)
    mix = mix or MIX
    cs = C.split_concepts()
    concepts = [c for c in L.CONCEPTS if cs[c[0]] == "train"]
    plan = [k for k, n in mix.items() for _ in range(n)]
    rng.shuffle(plan)
    out, seen = [], set()
    for intent in plan:
        for _ in range(1000):
            seqs, tool, _w = C.INTENTS[intent]
            seq = rng.choice(seqs)
            c = rng.choice(concepts)
            pool = [a for a in L.ATTRS if C.applicable(a, c) and not C.heldout_combo(a, c)]
            attrs = sorted(rng.sample(pool, min(rng.choice([0, 1]), len(pool))), key=lambda a: a[1])
            name = pick_name(rng)
            grouped = rng.random() < P_GROUPED
            pc = make_postcode(rng, grouped)
            reading = "d3s" if grouped else "digits"
            phrases = []
            for pool_ in seq:
                phrases.append(rng.choice([x for x in pool_ if x not in phrases]))
            if rng.random() < 0.4:
                phrases = [rng.choice(C.GREET)] + phrases
            syn_pick = {l: None for l in LANGS}
            for l, syn in (("it", c[3]), ("fr", c[5])):
                if syn and rng.random() < 0.3:
                    syn_pick[l] = rng.choice(syn)
            uses_np = any(("{ind}" in p[0] or "{def}" in p[0] or "{poss}" in p[0] or "{Def}" in p[0]) for p in phrases)
            text = {}
            for i, l in enumerate(LANGS):
                np_ = C.noun_phrase(c, attrs, l, syn_pick[l]) if uses_np else None
                text[l] = " ".join(C.fill(p[i], l, np_, name, P.spoken(pc, l, reading)) for p in phrases)
            if text["en"] in seen:
                continue
            seen.add(text["en"])
            says_pc = any("{pc}" in p[0] for p in phrases)
            target = {"name": "find_customer", "arguments": {"full_name": name, **({"postcode": pc} if says_pc else {})}}
            out.append({"id": f"s4b-{len(out):04d}", "intent": intent, "text": text, "n_phrases": len(phrases),
                        "target": target,
                        "slots": {"concept": c[0] if uses_np else None, "category": c[1] if uses_np else None,
                                  "attrs": [a[0] for a in attrs] if uses_np else [], "synonym": syn_pick,
                                  "name": name, "postcode": pc if says_pc else None,
                                  "postcode_form": ("grouped" if grouped else "digits") if says_pc else None,
                                  "postcode_reading": reading if says_pc else None,
                                  "heldout_combo": False, "concept_split": cs[c[0]] if uses_np else None,
                                  "probe_type": None},
                        "voice": rng.choice(["ryan", "vivian"]),
                        "split": "val" if rng.random() < 0.1 else "train"})
            break
        else:
            raise SystemExit(f"could not build a {intent} family")
    return out


def derivable(item: dict) -> list[str]:
    """Every target slot recoverable from the caller's text in every language: the name string, and the postcode as
    the UNIQUE 5-digit acoustic parse of its spoken form (postcode_v2b, no punctuation)."""
    bad, t, s = [], item["target"], item["slots"]
    for l in LANGS:
        tx = item["text"][l]
        if s["name"].lower() not in tx.lower():
            bad.append(f"{l}:name")
        pc = t["arguments"].get("postcode")
        if pc:
            sp = P.spoken(pc, l, s["postcode_reading"])
            if sp not in tx:
                bad.append(f"{l}:postcode_text")
            if P.parses(P.text_tokens(sp, l), l) != {pc}:
                bad.append(f"{l}:postcode_ambiguous")
    return bad


def audit(items: list[dict], frozen: list[dict]) -> dict:
    """Leakage and separation: no G-LEX / held-out name, concept or combination; no caller text already in a frozen
    plan (v2 full, s4x, G-LEX dev/test, idev); ids disjoint; derivability."""
    cs = C.split_concepts()
    frozen_text = {(r["lang"], r["text"]) for r in frozen}
    frozen_ids = {r["id"] for r in frozen}
    glex_names = {(r["slots"] if isinstance(r["slots"], dict) else {}).get("name") for r in frozen
                  if r["id"].startswith(("gd4-", "gt4-"))} - {None}
    rep = Counter()
    for it in items:
        s = it["slots"]
        rep["not_derivable"] += bool(derivable(it))
        rep["id_clash"] += it["id"] in frozen_ids
        rep["text_in_frozen_plan"] += sum((l, it["text"][l]) in frozen_text for l in LANGS)
        rep["glex_name"] += s["name"] in glex_names
        rep["homophone_name"] += any(w in HOMOPHONE_DIAGNOSTIC["first"] | HOMOPHONE_DIAGNOSTIC["last"]
                                     for w in s["name"].split())
        first, last = s["name"].split(" ", 1)
        rep["heldout_name"] += C.name_split(first, last) != "train"
        if s["concept"]:
            rep["non_train_concept"] += cs[s["concept"]] != "train"
            rep["heldout_combo"] += any(C.heldout_combo(next(a for a in L.ATTRS if a[0] == an),
                                                        next(c for c in L.CONCEPTS if c[0] == s["concept"]))
                                        for an in s["attrs"])
        rep["ille"] += bool(re.search(r"\bille\b", it["text"]["fr"]))
    return {k: rep[k] for k in ("not_derivable", "id_clash", "text_in_frozen_plan", "glex_name", "homophone_name",
                                "heldout_name", "non_train_concept", "heldout_combo", "ille")}


def stats(items: list[dict]) -> dict:
    pcs = [i for i in items if i["slots"]["postcode"]]
    vig = lambda i: int(i["slots"]["postcode"][:2]) >= 70            # noqa: E731
    est = {l: sorted(len(i["text"][l]) / 15.0 for i in items) for l in LANGS}
    q = lambda xs, p: round(xs[int(p * (len(xs) - 1))], 1)          # noqa: E731
    return {"version": VERSION, "families": len(items), "intents": dict(Counter(i["intent"] for i in items)),
            "split": dict(Counter(i["split"] for i in items)),
            "postcode_reading": dict(Counter(i["slots"]["postcode_reading"] for i in pcs)),
            "d3s_fell_back_to_digits": sum(1 for i in pcs if i["slots"]["postcode_reading"] == "d3s"
                                           and P.spoken(i["slots"]["postcode"], "fr", "d3s")
                                           == P.spoken(i["slots"]["postcode"], "fr", "digits")),
            "grouped_vigesimal_dept_70_99": sum(1 for i in pcs if i["slots"]["postcode_form"] == "grouped" and vig(i)),
            "grouped_total": sum(1 for i in pcs if i["slots"]["postcode_form"] == "grouped"),
            "names": len({i["slots"]["name"] for i in items}),
            "concepts": len({i["slots"]["concept"] for i in items if i["slots"]["concept"]}),
            "synonym_used": {l: sum(bool(i["slots"]["synonym"][l]) for i in items) for l in ("it", "fr")},
            "est_speech_s_p10_p50_p90": {l: [q(v, .1), q(v, .5), q(v, .9)] for l, v in est.items()},
            "clips": 3 * len(items),
            "sha256_text": hashlib.sha256(json.dumps([i["text"] for i in items], ensure_ascii=False).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--audit", default="")
    ap.add_argument("--frozen", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    for p in (a.out, a.stats, a.audit):
        if p and Path(p).exists():
            raise SystemExit(f"{p} exists")
    items = build(a.seed)
    with open(a.out, "w") as f:
        for it in items:
            for l in LANGS:
                f.write(json.dumps({"id": it["id"], "split": it["split"], "cat": f"s4:{it['intent']}", "lang": l,
                                    "voice": it["voice"], "text": it["text"][l], "target": it["target"],
                                    "slots": it["slots"], "n_phrases": it["n_phrases"], "supplement": "v2b",
                                    "generator_version": VERSION}, ensure_ascii=False) + "\n")
    st = stats(items)
    Path(a.stats).write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(json.dumps(st, indent=1, ensure_ascii=False))
    if a.audit:
        frozen = [json.loads(x) for p in a.frozen for x in open(p)]
        rep = audit(items, frozen)
        Path(a.audit).write_text(json.dumps(rep, indent=1))
        print("AUDIT", json.dumps(rep))


if __name__ == "__main__":
    main()
