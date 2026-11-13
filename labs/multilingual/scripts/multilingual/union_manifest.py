"""Corpus v2: one training manifest from several merged parts (pilot merged, full-new merged, supplement merged) (CPU).
Only clips whose (id, lang) is in --plan are kept (G-LEX / probe clips of a part are dropped and counted); their
target / slots / cat / split are REFRESHED from the current plan (e.g. the escalate fix: the pilot manifest still carries
`arguments: null`), and the spoken text must be identical (the audio is of that text). Every clip is symlinked into <out>; clips listed in --exclude get `excluded: <reason>` (kept in the manifest, out of
training via session_distill.usable_keys), never deleted. Duplicate (id, lang) keys across parts are an error.
Usage: union_manifest.py --part ml/data/v4/tts_merged --part ml/data/v4/tts_new_merged --exclude grammar_defect.json
       --reason grammar_defect --out ml/data/v4/tts_full_merged --report report.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


REFRESH = ("target", "slots", "cat", "split", "n_phrases")


def union(parts: list[list[dict]], exclude: set, reason: str, plan: dict) -> tuple[list[dict], dict]:
    seen, out, stat = set(), [], Counter()
    for rows in parts:
        for r in rows:
            k = (r["id"], r["lang"])
            if k not in plan:
                stat["dropped_not_in_plan"] += 1
                continue
            if k in seen:
                raise SystemExit(f"duplicate clip {k}")
            seen.add(k)
            p = plan[k]
            if p["text"] != r["text"]:
                raise SystemExit(f"text differs from the plan for {k}: {r['text'][:60]!r} vs {p['text'][:60]!r}")
            stat["target_refreshed"] += int(p.get("target") != r.get("target"))
            out.append({**r, **{f: p.get(f) for f in REFRESH},
                        **({"excluded": reason} if f"{k[0]}|{k[1]}" in exclude else {})})
    missing = exclude - {f"{r['id']}|{r['lang']}" for r in out}
    if missing:
        raise SystemExit(f"exclusions not found in the manifest: {sorted(missing)[:5]}")
    stat["plan_keys_without_clip"] = len(set(plan) - seen)
    return out, dict(stat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--plan", action="append", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--exclude-key", default="train_exclude")
    ap.add_argument("--reason", default="grammar_defect")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "manifest.jsonl").exists() or Path(a.report).exists():
        raise SystemExit("output exists")
    exc = set(json.load(open(a.exclude))[a.exclude_key]) if a.exclude else set()
    parts = []
    for p in a.part:
        rows = [json.loads(x) for x in open(Path(p) / "manifest.jsonl")]
        for r in rows:
            dst = out / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink((Path(p) / r["path"]).resolve(), dst)
        parts.append(rows)
    plan = {(r["id"], r["lang"]): r for p in a.plan for r in map(json.loads, open(p))}
    rows, stat = union(parts, exc, a.reason, plan)
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {**stat, "clips": len(rows), "families": len({r["id"] for r in rows}),
           "per_lang": {l: {"n": sum(r["lang"] == l for r in rows),
                            "asr_ok": sum(r["lang"] == l and r.get("asr_ok", True) for r in rows),
                            "usable": sum(r["lang"] == l and r.get("asr_ok", True) and not r.get("excluded") for r in rows)}
                        for l in ("en", "it", "fr")},
           "excluded": dict(Counter(f"{r['lang']}:{r['excluded']}" for r in rows if r.get("excluded"))),
           "excluded_and_asr_failed": sum(1 for r in rows if r.get("excluded") and not r.get("asr_ok", True))}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
