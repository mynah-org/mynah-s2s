"""Corpus v2 (CPU): apply one more re-render round on top of an already merged manifest.

  plan  : retry_from_manifest(...) -> plan rows (text frozen, from --plan) of every clip of --langs whose asr_ok is false
          in the merged manifest (excluded clips included: the ASR gate denominator is all clips).
  apply : for each key whose strict verification in --verify is ok, the row points to the new render
          (render_round = --round-index, asr_ok true); otherwise the base row is kept and the new cause is appended to
          `cause_history`. Every other field of the base row (refreshed target, `excluded`) is kept. Clips are symlinked
          into --out; nothing is deleted.
Usage: apply_round.py plan --base DIR --plan plan.jsonl --langs it,fr --out retry.jsonl
       apply_round.py apply --base DIR --round DIR --verify verify.json --round-index 3 --out DIR --report r.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def retry_rows(base: list[dict], plan: dict, langs: set) -> list[dict]:
    out = []
    for r in base:
        k = (r["id"], r["lang"])
        if r["lang"] in langs and not r.get("asr_ok", True):
            p = plan[k]
            if p["text"] != r["text"]:
                raise SystemExit(f"text differs from the plan for {k}")
            out.append(p)
    return out


def apply(base: list[dict], new: dict, verify: dict, idx: int) -> tuple[list[dict], dict]:
    rows, stat = [], Counter()
    for r in base:
        key = f"{r['id']}|{r['lang']}"
        v = verify.get(key)
        if v is None:
            rows.append(r)
            continue
        stat[f"tried:{r['lang']}"] += 1
        hist = list(r.get("cause_history") or []) + ["ok" if v["ok"] else v.get("cause", "?")]
        if v["ok"] and not r.get("asr_ok", True):
            m = new[(r["id"], r["lang"])]
            if m["text"] != r["text"]:
                raise SystemExit(f"round text differs for {key}")
            rows.append({**r, "path": m["path"], "sha256": m.get("sha256"), "dur": m["dur"], "seed": m.get("seed"),
                         "chars_per_s": m.get("chars_per_s"), "asr_ok": True, "render_round": idx,
                         "cause_history": hist, "_src": "round"})
            stat[f"recovered:{r['lang']}"] += 1
        else:
            rows.append({**r, "cause_history": hist})
    return rows, dict(stat)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--langs", default="it,fr")
    p.add_argument("--out", required=True)
    q = sub.add_parser("apply")
    q.add_argument("--base", required=True)
    q.add_argument("--round", required=True)
    q.add_argument("--verify", required=True)
    q.add_argument("--round-index", type=int, required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--report", required=True)
    a = ap.parse_args()
    base = [json.loads(x) for x in open(Path(a.base) / "manifest.jsonl")]
    if a.cmd == "plan":
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists")
        plan = {(r["id"], r["lang"]): r for r in map(json.loads, open(a.plan))}
        rows = retry_rows(base, plan, set(a.langs.split(",")))
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("retry rows", len(rows), dict(Counter(r["lang"] for r in rows)))
        return
    out = Path(a.out)
    if (out / "manifest.jsonl").exists() or Path(a.report).exists():
        raise SystemExit("output exists")
    new = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.round) / "manifest.jsonl"))}
    rows, stat = apply(base, new, json.load(open(a.verify))["rows"], a.round_index)
    for r in rows:
        src = Path(a.round if r.pop("_src", None) == "round" else a.base) / r["path"]
        dst = out / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src.resolve(), dst)
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {**stat, "clips": len(rows),
           "per_lang": {l: {"n": sum(r["lang"] == l for r in rows),
                            "asr_ok": sum(r["lang"] == l and r.get("asr_ok", True) for r in rows)}
                        for l in ("en", "it", "fr")}}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
