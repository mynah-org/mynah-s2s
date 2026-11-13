"""Corpus v2: merge the original render and the re-render rounds into one manifest (CPU).

Per clip, the FIRST attempt (original, rr1, rr2, ...) whose strict verification is ok is used (symlinked into <out>);
if none passes, the clip is kept in the manifest with asr_ok = false (excluded from training) and the cause history
(tts_verified / asr_probable / uncertain per attempt) is recorded. Also writes the list of families whose teacher must
be re-decoded: the EN clip changed, or the rule has no call (post-turn SOTC suppression), or the family was
quarantined before.
Usage: merge_rerenders.py --plan plan_all.jsonl --round tts verify_r0.json --round tts_rr1 verify_r1.json ...
       --teacher teacher.json --out tts_merged --redo redo_ids.json --report report.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--round", nargs=2, action="append", required=True, metavar=("TTS_DIR", "VERIFY_JSON"))
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--redo", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "manifest.jsonl").exists():
        raise SystemExit(f"{out} exists")
    plan = [json.loads(x) for x in open(a.plan)]
    rounds = []
    for tts, ver in a.round:
        man = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(tts) / "manifest.jsonl"))}
        rows = json.load(open(ver))["rows"]
        rounds.append((Path(tts), man, rows))
    rows_out, hist, attempts = [], Counter(), Counter()
    changed_en = set()
    for p in plan:
        key = (p["id"], p["lang"])
        chosen, causes = None, []
        for k, (tts, man, ver) in enumerate(rounds):
            v = ver.get(f"{key[0]}|{key[1]}")
            if key not in man or v is None:
                continue
            attempts[key] = k
            causes.append(v.get("cause", "ok" if v["ok"] else "uncertain"))
            if v["ok"]:
                chosen = (k, tts, man[key])
                break
        if chosen is None:                             # never verified: keep the original render, excluded
            k, tts, m = 0, rounds[0][0], rounds[0][1][key]
            ok = False
        else:
            k, tts, m = chosen
            ok = True
        dst = out / m["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink((tts / m["path"]).resolve(), dst)
        rows_out.append({**m, "asr_ok": ok, "render_round": k, "cause_history": causes})
        hist[(p["lang"], "ok" if ok else f"excluded:{causes[-1] if causes else 'missing'}")] += 1
        if p["lang"] == "en" and k > 0:
            changed_en.add(p["id"])
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prev = json.load(open(a.teacher))["items"]
    rule_none = {p["id"] for p in plan if p["lang"] == "en" and not p.get("target") and p["split"] in ("train", "val")}
    quarantined = {i for i, v in prev.items() if not v["keep"]}
    redo = sorted((changed_en & set(prev)) | (rule_none & set(prev)) | quarantined)
    Path(a.redo).write_text(json.dumps(redo))
    regen = Counter(r["render_round"] for r in rows_out if r["render_round"] > 0)
    rep = {"clips": len(rows_out), "asr_ok": sum(r["asr_ok"] for r in rows_out),
           "per_lang": {l: {"ok": sum(1 for r in rows_out if r["lang"] == l and r["asr_ok"]),
                            "n": sum(1 for r in rows_out if r["lang"] == l)} for l in ("en", "it", "fr")},
           "excluded_by_final_cause": dict(Counter(f"{r['lang']}:{r['cause_history'][-1] if r['cause_history'] else 'missing'}"
                                                   for r in rows_out if not r["asr_ok"])),
           "recovered_by_round": {f"round{k}": n for k, n in sorted(regen.items())},
           "regenerations_attempted": sum(max(0, len(r["cause_history"]) - 1) for r in rows_out),
           "teacher_redo": {"n": len(redo), "changed_en": len(changed_en & set(prev)), "rule_none": len(rule_none & set(prev)),
                            "quarantined_before": len(quarantined)}}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
