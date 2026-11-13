"""Corpus V2.1 (gold-first): MKQA (human IT/FR queries, gold answers) + SVAMP (short arithmetic, gold numbers).

The LLMs never invent facts here. Qwen2.5 only translates SVAMP (numbers checked deterministically); Nano-9B-v2
only answers the EN text to record text capability (the knowledge filter). Teacher replies are later validated
against the gold (teacher_gold), not against an LLM answer.

Subcommands (data dir V21 = $MYNAH_LAB_HOME/data/v21):
  select       CPU  MKQA + SVAMP candidates (form filters only) -> cand.jsonl
  textfilter   GPU  Nano answers each EN question from text -> text_filter.jsonl (pass/fail kept, never dropped)
  translate    GPU  SVAMP EN -> IT/FR with Qwen2.5; numbers must be identical -> svamp_tr.jsonl
  build        CPU  splits + plans + qa_v21.json (gold regexes) + build_report.json
  assemble     CPU  training manifest (Qwen3-TTS clips only; no Common Voice in V2.1)
  teacher_gold CPU  keep training ids whose VC teacher reply matches the gold -> keep_ids.json
Gold matching (controlled normalisation, no fuzzy matching): lower case, accents folded, punctuation removed;
  numbers: digits (with or without thousands separators) or English number words (VC speaks English);
  dates: the year (digits or spoken form) is required and sufficient (rule stated in the report);
  entities / short phrases: the gold text or one of its MKQA aliases as a whole phrase, a leading article dropped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from labpaths import LAB_HOME  # noqa: E402  (MYNAH_HOME / MYNAH_LAB_HOME, no hard-coded home layout)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus_v2 import Nano, QWEN_TR, content, is_refusal, jacc, norm, num_words, sha, write_new  # noqa: E402

V21 = LAB_HOME / "data/v21"
LANGS = ("en", "it", "fr")
SEED = 20261112
TIME_SENSITIVE = re.compile(r"\b(current|currently|latest|now|this year|last|newest|recent|next|new season|"
                            r"20(1[5-9]|2\d))\b")
QWORD = re.compile(r"^(who|what|when|where|which|how|in what|in which|what's|whats|whos|who's)\b")


# ------------------------------------------------------------------------------------------------ gold matching
def year_forms(y: int) -> list[str]:
    f = {str(y), num_words(y)}
    if 2000 <= y < 2100:
        f.add("two thousand" + ("" if y % 100 == 0 else " and " + num_words(y % 100)))
        f.add("two thousand" + ("" if y % 100 == 0 else " " + num_words(y % 100)))
    return sorted(f)


def big_words(n: int) -> str:
    """English words up to the millions (VC replies are text-normalised to English words)."""
    if n < 1000:
        return num_words(n)
    if n < 1_000_000:
        return big_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + big_words(n % 1000))
    return big_words(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else " " + big_words(n % 1_000_000))


def int_forms(n: int) -> list[str]:
    f = {str(n), big_words(n)}
    if 1000 <= n < 10000:
        f.add(num_words(n))  # year-style reading ("nineteen eighty nine")
    return sorted(f)


def phrase_regex(p: str) -> str | None:
    p = norm(p)
    p = re.sub(r"^(the|a|an) ", "", p)
    if len(p) < 3:
        return None
    return r"\b" + re.escape(p).replace(r"\ ", r" ") + r"\b"


def gold_forms(typ: str, text: str, aliases: list[str]) -> list[str]:
    """Regexes on corpus_v2.norm(reply) (score_v2 normalises differently: it keeps apostrophes; the forms here
    contain only [a-z0-9 ,] so both normalisations agree)."""
    out = set()
    if typ in ("number", "number_with_unit"):
        for t in [text] + aliases:
            m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", t.replace(",", ""))
            if m:
                v = float(m.group(1))
                if v == int(v):
                    for s in int_forms(int(v)):
                        if "," in s:
                            continue  # thousands separators are removed from the reply before matching
                        sfx = "(?:st|nd|rd|th)?" if s.isdigit() else ""  # "the 35th state" for gold 35
                        out.add(r"\b" + s.replace(" and ", " (?:and )?").replace(" ", "[ -]") + sfx + r"\b")
        return sorted(out)
    if typ == "date":
        d = parse_date(text)
        return [date_regex(*d)] if d else []
    for t in [text] + aliases:
        r = phrase_regex(t)
        if r:
            out.add(r)
    return sorted(out)


MONTHS = "january february march april may june july august september october november december".split()
ORD = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
       "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()


def ordinal_words(d: int) -> str:
    if d <= 20:
        return ORD[d - 1]
    if d < 30:
        return "twenty " + ORD[d - 21]
    return "thirtieth" if d == 30 else "thirty first"


def parse_date(text: str):
    """MKQA date gold -> (year, month|None, day|None). Formats seen: YYYY-MM-DD, YYYY-MM, YYYY, 'D month YYYY',
    'month YYYY', 'YYYY month D'."""
    t = text.strip().lower()
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", t)
    if m:
        return int(m.group(1)), int(m.group(2)) if m.group(2) else None, int(m.group(3)) if m.group(3) else None
    y = re.search(r"\b(\d{4})\b", t)
    mo = next((i + 1 for i, n in enumerate(MONTHS) if re.search(rf"\b{n}\b", t)), None)
    if not y:
        return None
    rest = re.sub(r"\b\d{4}\b", " ", t)
    dd = re.search(r"\b(\d{1,2})\b", rest)
    return int(y.group(1)), mo, int(dd.group(1)) if (dd and mo) else None


def date_regex(y: int, mo, d) -> str:
    """ONE regex with look-aheads: every component given by the gold must appear in the reply (any order).
    Year-only gold -> year; month+year -> both; full date -> day, month and year (owner review, 2026-11-11)."""
    parts = [r"(?=.*\b(?:" + "|".join(f.replace(" ", "[ -]") for f in year_forms(y)) + r")\b)"]
    if mo:
        parts.append(rf"(?=.*\b{MONTHS[mo - 1]}\b)")
    if d:
        alts = {str(d), f"{d}(?:st|nd|rd|th)", ordinal_words(d).replace(" ", "[ -]"), num_words(d).replace(" ", "[ -]")}
        parts.append(r"(?=.*\b(?:" + "|".join(sorted(alts)) + r")\b)")
    return "^" + "".join(parts)


def gold_hit(reply: str, forms: list[str]) -> bool:
    n = norm(re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", reply))  # 384,400 / 384.400 -> 384400
    return any(re.search(f, n) for f in forms)


# ------------------------------------------------------------------------------------------------ select
def cmd_select(a):
    rng = random.Random(SEED)
    mk = [json.loads(l) for l in gzip.open(V21 / "src/mkqa/mkqa.jsonl.gz")]
    cands, drop = [], Counter()
    for r in mk:
        ans = r["answers"]["en"][0]
        typ = ans["type"]
        q = r["query"].strip()
        if typ not in ("entity", "date", "number", "number_with_unit", "short_phrase"):
            drop["answer_type"] += 1
            continue
        if not 4 <= len(q.split()) <= 14:
            drop["length"] += 1
            continue
        if TIME_SENSITIVE.search(q.lower()):
            drop["time_sensitive"] += 1
            continue
        if not QWORD.match(q.lower()):
            drop["no_question_word"] += 1
            continue
        if any(not (r["queries"].get(l) or "").strip() for l in ("it", "fr")):
            drop["missing_it_fr"] += 1
            continue
        forms = gold_forms(typ, ans.get("text") or "", ans.get("aliases") or [])
        if not forms:
            drop["no_checkable_gold"] += 1
            continue
        cands.append({"id": f"mk{r['example_id']}", "source": "mkqa", "cat": typ, "en": q,
                      "it": r["queries"]["it"].strip(), "fr": r["queries"]["fr"].strip(),
                      "gold_text": ans.get("text"), "gold_aliases": ans.get("aliases") or [], "gold": forms})
    # stratified subsample by type (proportional, capped) -> knowledge filter is the expensive part
    by = defaultdict(list)
    for c in cands:
        by[c["cat"]].append(c)
    quota = {"entity": 1100, "date": 350, "number": 200, "number_with_unit": 150, "short_phrase": 100}
    mk_sel = []
    for t, q in quota.items():
        pool = sorted(by[t], key=lambda c: c["id"])
        mk_sel += rng.sample(pool, min(q, len(pool)))
    sv = []
    for f in ("train.json", "test.json"):
        for r in json.load(open(V21 / "src/svamp" / f)):
            body, qq = r["Body"].strip(), r["Question"].strip()
            text = (body if body.endswith(".") else body + ".") + " " + qq
            ans = float(r["Answer"])
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            if ans != int(ans) or ans < 0 or len(text) > 200 or len(nums) > 3 or any("." in n for n in nums):
                drop["svamp_form"] += 1
                continue
            sv.append({"id": f"sv{r['ID'].replace('chal-', '')}", "source": "svamp", "cat": "arithmetic",
                       "en": text, "gold_text": str(int(ans)), "gold_aliases": [], "svamp_type": r["Type"],
                       "gold": gold_forms("number", str(int(ans)), [])})
    sv = rng.sample(sorted(sv, key=lambda c: c["id"]), min(400, len(sv)))
    out = mk_sel + sv
    write_new(V21 / "cand.jsonl", out)
    info = {"seed": SEED, "mkqa_sha256": sha(V21 / "src/mkqa/mkqa.jsonl.gz"),
            "svamp_sha256": {f: sha(V21 / "src/svamp" / f) for f in ("train.json", "test.json")},
            "mkqa_passing_form_filters": len(cands), "mkqa_selected": len(mk_sel), "svamp_selected": len(sv),
            "dropped": dict(drop), "selected_by_cat": dict(Counter(c["cat"] for c in out))}
    (V21 / "select_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------------ text filter
def cmd_textfilter(a):
    cands = [json.loads(l) for l in open(V21 / "cand.jsonl")]
    nano = Nano(a.batch)
    t0 = time.monotonic()
    outs = nano.run([f"Answer the question briefly, in one short English sentence: {c['en']}" for c in cands], 48)
    rows = []
    for c, o in zip(cands, outs):
        rows.append({"id": c["id"], "source": c["source"], "cat": c["cat"], "answer": o, "pass": gold_hit(o, c["gold"])})
    write_new(V21 / "text_filter.jsonl", rows)
    c = Counter((r["source"], r["cat"], r["pass"]) for r in rows)
    info = {"model": nano.rev, "n": len(rows), "pass": sum(r["pass"] for r in rows), "seconds": round(time.monotonic() - t0),
            "by": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(c.items(), key=str)}}
    (V21 / "text_filter_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------------ translate SVAMP
def cmd_translate(a):
    cands = [json.loads(l) for l in open(V21 / "cand.jsonl") if '"svamp"' in l]
    tr = Nano(a.batch, path=QWEN_TR, system="You are a professional translator for spoken voice-assistant requests.")
    names = {"it": "Italian", "fr": "French"}
    rows, bad = {c["id"]: {"id": c["id"]} for c in cands}, Counter()
    for l, name in names.items():
        outs = tr.run([f"Translate this short math word problem into natural spoken {name}. Keep every number "
                       f"exactly as digits. Output only the translation.\n\n{c['en']}" for c in cands], 160)
        for c, o in zip(cands, outs):
            o = re.sub(r"\s+", " ", o).strip().strip('"')
            src = sorted(re.findall(r"\d+", c["en"]))
            got = sorted(re.findall(r"\d+", o))
            ok = src == got and len(o) > 10 and not re.search(r"[!]{2,}", o)
            rows[c["id"]][l] = o
            rows[c["id"]][f"{l}_ok"] = ok
            if not ok:
                bad[l] += 1
    write_new(V21 / "svamp_tr.jsonl", list(rows.values()))
    info = {"model": QWEN_TR, "n": len(cands), "numbers_mismatch_or_bad": dict(bad)}
    (V21 / "translate_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ------------------------------------------------------------------------------------------------ build
def speakable(q: str) -> str:
    q = q.strip()
    q = q[0].upper() + q[1:] if q else q
    return q if q[-1:] in "?.!" else q + "?"


def cmd_build(a):
    rng = random.Random(SEED + 1)
    cands = {json.loads(l)["id"]: json.loads(l) for l in open(V21 / "cand.jsonl")}
    tf = {json.loads(l)["id"]: json.loads(l) for l in open(V21 / "text_filter.jsonl")}
    svt = {json.loads(l)["id"]: json.loads(l) for l in open(V21 / "svamp_tr.jsonl")}
    drop = Counter()
    fams = []
    for i, c in cands.items():
        if c["source"] == "svamp":
            t = svt.get(i, {})
            if not (t.get("it_ok") and t.get("fr_ok")):
                drop["svamp_translation_check"] += 1
                continue
            c = {**c, "it": t["it"], "fr": t["fr"]}
        typ = "number" if c["source"] == "svamp" else c["cat"]
        c["gold"] = gold_forms(typ, c["gold_text"] or "", c["gold_aliases"])  # current matcher (date rule revised)
        if not c["gold"]:
            drop["no_checkable_gold"] += 1
            continue
        c["text_pass"] = gold_hit(tf[i]["answer"], c["gold"])
        fams.append(c)
    # leakage vs earlier eval sets (QA v1, QA v2 incl. excluded, MINDS-14 is a different domain): content Jaccard
    prev = []
    for p in (HERE / "qa_corpus.json", V21.parent / "v2/qa_v2.json"):
        if p.exists():
            prev += [content(it["text"]["en"]) for it in json.load(open(p))["items"]]
    for p in (V21.parent / "v2/qa_v2_excluded.json",):
        if p.exists():
            prev += [content(it["text"]["en"]) for it in json.load(open(p))]
    leak = [f["id"] for f in fams if any(jacc(content(f["en"]), q) >= 0.5 for q in prev)]
    fams = [f for f in fams if f["id"] not in set(leak)]
    # within-corpus near duplicates: keep the first by id
    seen, dedup = [], []
    for f in sorted(fams, key=lambda f: f["id"]):
        cs = content(f["en"])
        if any(jacc(cs, s) >= 0.7 for s in seen):
            drop["near_duplicate"] += 1
            continue
        seen.append(cs)
        dedup.append(f)
    fams = dedup
    passed = [f for f in fams if f["text_pass"]]
    failed = [f for f in fams if not f["text_pass"]]
    rng.shuffle(passed)
    rng.shuffle(failed)

    def take(pool, src, n):
        got = [f for f in pool if f["source"] == src][:n]
        for f in got:
            pool.remove(f)
        return got
    split = {}
    for f in take(passed, "mkqa", a.test_mk) + take(passed, "svamp", a.test_sv):
        split[f["id"]] = "heldout"
    for f in take(passed, "mkqa", a.dev_mk) + take(passed, "svamp", a.dev_sv):
        split[f["id"]] = "dev"
    for f in take(failed, "mkqa", a.uncond_mk) + take(failed, "svamp", a.uncond_sv):
        split[f["id"]] = "heldout_uncond"
    for f in take(passed, "mkqa", a.val_mk) + take(passed, "svamp", a.val_sv):
        split[f["id"]] = "val"
    for f in take(passed, "mkqa", a.train_mk) + take(passed, "svamp", a.train_sv):
        split[f["id"]] = "train"
    for f in fams:
        f["split"] = split.get(f["id"], "unused")
    write_new(V21 / "families.jsonl", fams)
    qa = {"version": "qa_v21", "note": "gold answers from MKQA (EN gold + aliases) and SVAMP; IT/FR queries: MKQA human "
          "translations, SVAMP Qwen2.5 translations with identical numbers; heldout_uncond = items Nano failed from text",
          "items": [{"id": f["id"], "cat": f"{f['source']}:{f['cat']}", "split": f["split"],
                     "text": {l: speakable(f[l]) for l in LANGS}, "answers": f["gold"], "answer_text": f["gold_text"],
                     "text_pass": f["text_pass"], "source": f["source"]}
                    for f in fams if f["split"] in ("dev", "heldout", "heldout_uncond")]}
    write_new(V21 / "qa_v21.json", obj=qa)
    plan_eval, plan_train = [], []
    for it in qa["items"]:
        for l in LANGS:
            plan_eval.append({"id": it["id"], "split": it["split"], "cat": it["cat"], "lang": l, "voice": "ryan",
                              "text": it["text"][l]})
    voices = ("ryan", "vivian")
    k = 0
    for f in fams:
        if f["split"] in ("train", "val"):
            v = voices[k % 2]
            k += 1
            for l in LANGS:
                plan_train.append({"id": f["id"], "split": f["split"], "cat": f"{f['source']}:{f['cat']}", "lang": l,
                                   "voice": v, "text": speakable(f[l])})
    write_new(V21 / "plan_eval.jsonl", plan_eval)
    write_new(V21 / "plan_train.jsonl", plan_train)
    rep = {"families": len(fams), "leakage_removed": len(leak), "dropped": dict(drop),
           "text_pass_rate": {s: f"{sum(f['text_pass'] for f in fams if f['source'] == s)}/"
                                 f"{sum(1 for f in fams if f['source'] == s)}" for s in ("mkqa", "svamp")},
           "splits": {f"{s}|{src}": n for (s, src), n in sorted(Counter((f["split"], f["source"]) for f in fams).items())},
           "plan_eval_rows": len(plan_eval), "plan_train_rows": len(plan_train)}
    (V21 / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_fix_eval(a):
    """Apply the manual review of the eval set (exclusions + IT/FR replacements). Originals kept as *.orig. Then
    freeze: writes ml/results/v21/eval_freeze.json with the hashes (no edits after this point)."""
    rev = json.load(open(a.review))
    qa_p, plan_p = V21 / "qa_v21.json", V21 / "plan_eval.jsonl"
    for p in (qa_p, plan_p):
        o = p.with_name(p.name + ".orig")
        if o.exists():
            raise SystemExit(f"{o} exists: fix_eval already applied")
        o.write_bytes(p.read_bytes())
    qa = json.load(open(qa_p))
    ids = {it["id"] for it in qa["items"]}
    for k in ("exclude", "it", "fr"):
        miss = set(rev[k]) - ids
        if miss:
            raise SystemExit(f"{k}: ids not in the eval set {sorted(miss)}")
    n = Counter()
    kept = []
    for it in qa["items"]:
        if it["id"] in rev["exclude"]:
            n["excluded"] += 1
            continue
        for l in ("it", "fr"):
            if it["id"] in rev[l]:
                it.setdefault("text_mt", {})[l] = it["text"][l]
                it["text"][l] = rev[l][it["id"]]
                n[f"replaced_{l}"] += 1
        kept.append(it)
    qa["items"] = kept
    qa["eval_review"] = {"review": a.review, "sha256": sha(Path(a.review)), "log": dict(n),
                         "rater": "assistant, text only; MKQA texts verbatim (human); SVAMP IT/FR reviewed"}
    qa_p.write_text(json.dumps(qa, indent=1, ensure_ascii=False))
    keep_ids = {it["id"] for it in kept}
    rows = [json.loads(l) for l in open(plan_p.with_name(plan_p.name + ".orig"))]
    rows = [r for r in rows if r["id"] in keep_ids]
    for r in rows:
        if r["lang"] in ("it", "fr") and r["id"] in rev[r["lang"]]:
            r["text"] = rev[r["lang"]][r["id"]]
    plan_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    fz = LAB_HOME / "results/v21/eval_freeze.json"
    write_new(fz, obj={"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "rule": "no edits driven by model outputs; a later change is a new eval version",
                       "sha256": {str(p): sha(p) for p in (qa_p, plan_p, Path(a.review))},
                       "by_split": dict(Counter(it["split"] for it in kept)), "n_items": len(kept)})
    print(json.dumps({"log": dict(n), "items": len(kept), "plan_rows": len(rows),
                      "by_split": dict(Counter(it["split"] for it in kept))}))


def cmd_assemble(a):
    rend = [json.loads(l) for l in open(V21 / "tts_train/manifest.jsonl")]
    # SVAMP IT/FR train rows are machine translations: a text audit of 20 (seed 20261113) found clear errors in 11 IT
    # and 3 FR (Chinese text, English leftovers, changed quantities), so only their EN rows are kept (EN rehearsal,
    # the teacher answers the same EN clip). MKQA IT/FR are human translations and are all kept.
    svamp = {json.loads(l)["id"] for l in open(V21 / "families.jsonl") if '"svamp"' in l}
    rows, dropped = [], Counter()
    for r in rend:
        if r["id"] in svamp and r["lang"] != "en":
            dropped[r["lang"]] += 1
            continue
        rows.append({**r, "path": str(V21 / "tts_train" / r["path"]), "source": "qwen3tts"})
    print("dropped SVAMP MT rows:", dict(dropped))
    write_new(V21 / "train/manifest.jsonl", rows)
    print(json.dumps([[list(k), v] for k, v in Counter((r["lang"], r["split"]) for r in rows).most_common()]))


def cmd_teacher_gold(a):
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())["items"]
    fams = {json.loads(l)["id"]: json.loads(l) for l in open(V21 / "families.jsonl")}
    keep, drop = [], []
    for i, it in prep.items():
        t = it["teacher_text"]
        f = fams[i]
        reason = None
        if len(norm(t).split()) < 2:
            reason = "empty"
        elif is_refusal(t):
            reason = "refusal"
        elif not gold_hit(t, f["gold"]):
            reason = "teacher_not_gold"
        (drop if reason else keep).append(i if not reason else {"id": i, "reason": reason, "teacher": t[:160],
                                                                 "gold": f["gold_text"]})
    by = Counter((fams[i]["source"], fams[i]["split"]) for i in keep)
    # survival per answer type and cause (owner: check the filter does not keep only the easy questions)
    tot = Counter(f"{fams[i]['source']}:{fams[i]['cat']}" for i in prep)
    kept_t = Counter(f"{fams[i]['source']}:{fams[i]['cat']}" for i in keep)
    cause_t = Counter((f"{fams[d['id']]['source']}:{fams[d['id']]['cat']}", d["reason"]) for d in drop)
    surv = {t: {"kept": kept_t[t], "of": n, **{r: c for (tt, r), c in cause_t.items() if tt == t}} for t, n in sorted(tot.items())}
    write_new(Path(a.out), obj={"keep": sorted(keep), "drop": drop,
                                "drop_reasons": dict(Counter(d["reason"] for d in drop)),
                                "kept_by": {f"{k[0]}|{k[1]}": v for k, v in sorted(by.items())},
                                "survival_by_type": surv,
                                "note": "the teacher runs on EN clips only, so survival is per family (identical for "
                                        "en/it/fr rows of a family)"})
    print(len(keep), "kept;", dict(Counter(d["reason"] for d in drop)), dict(by))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select")
    p = sp.add_parser("textfilter")
    p.add_argument("--batch", type=int, default=16)
    p = sp.add_parser("translate")
    p.add_argument("--batch", type=int, default=16)
    p = sp.add_parser("build")
    for k, v in (("test_mk", 45), ("test_sv", 15), ("dev_mk", 22), ("dev_sv", 8), ("uncond_mk", 12),
                 ("uncond_sv", 4), ("val_mk", 45), ("val_sv", 15), ("train_mk", 750), ("train_sv", 200)):
        p.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
    p = sp.add_parser("fix_eval")
    p.add_argument("--review", required=True)
    sp.add_parser("assemble")
    p = sp.add_parser("teacher_gold")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "textfilter": cmd_textfilter, "translate": cmd_translate, "build": cmd_build,
     "assemble": cmd_assemble, "fix_eval": cmd_fix_eval, "teacher_gold": cmd_teacher_gold}[a.cmd](a)


if __name__ == "__main__":
    main()
