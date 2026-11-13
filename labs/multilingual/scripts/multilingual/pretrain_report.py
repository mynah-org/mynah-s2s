"""Pre-training report (CPU + tokenizer only; owner GO 2026-11-13, phase 1 -> request for the training GO).

On the exact inputs `session_distill.py train` will read (one teacher.json, one manifest):
  1. keys       final train / val keys (trainer's own usable_keys + kept_keys) per language x source x intent;
  2. teacher    admission and quarantine causes per source x intent;
  3. sampling   effective share of each source / intent / language under the uniform sampler (rng.sample over the
                train keys), and expected presentations for a given number of optimizer steps x effective batch;
  4. leakage    no key from G-LEX dev/test (ids, caller texts, held-out names), idev, agentbench concepts or names,
                non-train concepts, held-out attribute x concept pairs;
  5. targets    for every kept family: the call tokens decode to the canonical rule call of the CURRENT plan target (or
                no call for rule-no-call intents), and for every train/val key the real student targets
                (`student_targets`) contain the SOTC inside the audio, no agent text over the caller, the onset EOS;
  6. FR 70-99   the diagnostic set candidates: usable FR VAL clips whose spoken postcode has a 70-99 number (val keys
                are never in the gradient); reported, not used for checkpoint selection.
Usage: pretrain_report.py --teacher t.json --manifest m.jsonl --glex dev.jsonl --glex test.jsonl --out r.json
       [--tokenizer NAME] [--steps 390 --eff-batch 8]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon_v2 as L  # noqa: E402
import session_corpus_v2 as C  # noqa: E402
import session_distill as S  # noqa: E402
from audit_transfer_timing import audit_one  # noqa: E402


def source(i: str) -> str:
    return "s4x" if i.startswith("s4x-") else "v2b" if i.startswith("s4b-") else "v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glex", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("--steps", type=int, default=390)
    ap.add_argument("--eff-batch", type=int, default=8)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    d = json.load(open(a.teacher))
    items = d["items"]
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(a.manifest))}
    use = S.usable_keys(man)
    split = {i: r["split"] for (i, _), r in man.items()}
    tr, va = S.kept_keys(items, use, split, ["en", "it", "fr"])
    rep = {}

    # 1. keys
    keys = defaultdict(Counter)
    for part, ks in (("train", tr), ("val", va)):
        for i, l in ks:
            r = man[(i, l)]
            keys[part][f"{l}|{source(i)}|{r['cat']}"] += 1
            keys[part][f"lang:{l}"] += 1
            keys[part][f"source:{source(i)}"] += 1
            keys[part][f"intent:{r['cat']}"] += 1
    rep["keys"] = {p: dict(sorted(c.items())) for p, c in keys.items()}
    rep["keys_total"] = {"train": len(tr), "val": len(va)}

    # 2. teacher
    tq = defaultdict(Counter)
    for i, v in items.items():
        tq[f"{source(i)}|{man[(i, 'en')]['cat']}"][v["reason"]] += 1
    rep["teacher"] = {k: dict(c) for k, c in sorted(tq.items())}
    rep["teacher_total"] = dict(Counter(v["reason"] for v in items.values()))
    rep["families_not_decoded"] = dict(Counter(f"{source(i)}|{split[i]}" for i in {i for i, _ in man} if i not in items))

    # 3. sampling
    n = len(tr)
    pres = a.steps * a.eff_batch
    share = defaultdict(Counter)
    for i, l in tr:
        share["source"][source(i)] += 1
        share["intent"][man[(i, l)]["cat"]] += 1
        share["lang"][l] += 1
    rep["sampling"] = {"sampler": "uniform over train keys", "train_keys": n, "planned_presentations": pres,
                       "epochs": round(pres / n, 2),
                       **{dim: {k: {"keys": c, "share_pct": round(100 * c / n, 1),
                                    "expected_presentations": round(pres * c / n)} for k, c in cnt.most_common()}
                          for dim, cnt in share.items()}}

    # 4. leakage
    glex = [json.loads(x) for p in a.glex for x in open(p)]
    glex_ids = {r["id"] for r in glex}
    glex_text = {(r["lang"], r["text"]) for r in glex}
    glex_names = {(r["slots"] if isinstance(r["slots"], dict) else eval(r["slots"])).get("name") for r in glex} - {None}
    cs = C.split_concepts()
    lk = Counter()
    for i, l in tr + va:
        r = man[(i, l)]
        s = r["slots"] or {}
        lk["glex_id"] += i in glex_ids
        lk["glex_text"] += (l, r["text"]) in glex_text
        lk["glex_name"] += (s.get("name") in glex_names)
        lk["idev_or_other_split"] += r["split"] not in ("train", "val")
        if s.get("concept"):
            lk["agentbench_concept"] += s["concept"] in L.EVAL_CONCEPTS
            lk["non_train_concept"] += cs[s["concept"]] != "train"
            c = next(x for x in L.CONCEPTS if x[0] == s["concept"])
            lk["heldout_combo"] += any(C.heldout_combo(next(x for x in L.ATTRS if x[0] == an), c) for an in s["attrs"])
        if s.get("name"):
            first, last = s["name"].split(" ", 1)
            lk["heldout_or_excluded_name"] += (C.name_split(first, last) != "train"
                                               or s["name"].lower() in L.EXCL_NAMES)
        lk["excluded_clip"] += bool(r.get("excluded"))
    rep["leakage"] = {k: lk[k] for k in ("glex_id", "glex_text", "glex_name", "idev_or_other_split",
                                         "agentbench_concept", "non_train_concept", "heldout_combo",
                                         "heldout_or_excluded_name", "excluded_clip")}

    # 5. targets
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    tgt = Counter()
    bad = []
    for i, v in items.items():
        if not v["keep"]:
            continue
        rule = man[(i, "en")].get("target")
        func_after = [t for t in range(v["lead"], len(v["main_func"])) if v["main_func"][t] == S.SOTC]
        if rule is None:
            ok = not func_after and not v.get("fc")
            tgt["nocall_ok" if ok else "nocall_bad"] += 1
        else:
            got = tok.decode([x for x in v["fc"] if x != S.EOTC])
            ok = got == S.rule_call_text(rule) and v["fc"][-1] == S.EOTC and bool(func_after)
            tgt["call_canonical" if ok else "call_mismatch"] += 1
        if not ok and len(bad) < 10:
            bad.append({"id": i, "reason": v["reason"], "rule": rule, "fc": tok.decode(v.get("fc") or [])})
    tail = int(math.ceil(d.get("tail_s", 5.0) * 16000 / S.FRAME))
    tim = Counter()
    for i, l in tr + va:
        v = items[i]
        if man[(i, "en")].get("target") is None:
            continue
        r = audit_one(v, l, v["end"][l] + tail)
        tim["keys"] += 1
        tim["sotc_present"] += r["sotc_present"]
        tim["talk_over"] += r["talk_over"] > 0
        tim["pre_call_text"] += r["pre_call_text"] > 0
        tim["onset_eos"] += r["onset_eos"]
        tim[f"sotc_rel_end:{r['sotc_rel_end']}"] += 1
    rep["targets"] = {**dict(tgt), "examples_bad": bad, "timing_call_keys": dict(sorted(tim.items()))}

    # 6. FR 70-99 diagnostic candidates (val only)
    diag = []
    for i, l in va:
        s = man[(i, l)]["slots"] or {}
        pc = s.get("postcode")
        spoken_7099 = s.get("postcode_form") == "grouped" and (int(pc[:2]) >= 70 or (not s.get("postcode_reading")
                                                                                      and int(pc[3:]) >= 70)) if pc else False
        if l == "fr" and spoken_7099:                 # a 70-99 NUMBER is spoken (digit-by-digit readings excluded)
            diag.append(f"{i}|{l}")
    rep["fr_70_99_val_diagnostic"] = {"n": len(diag), "keys": diag}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("keys_total", "teacher_total", "families_not_decoded", "leakage")}, indent=1))
    print(json.dumps({k: v for k, v in rep["targets"].items() if k != "examples_bad"}, indent=1))
    print("bad examples", rep["targets"]["examples_bad"][:3])
    print("sampling", json.dumps({k: rep["sampling"][k] for k in ("train_keys", "epochs", "source", "lang")}))
    print("intent", json.dumps({k: v["share_pct"] for k, v in rep["sampling"]["intent"].items()}))
    print("FR 70-99 val diagnostic", rep["fr_70_99_val_diagnostic"]["n"])


if __name__ == "__main__":
    main()
