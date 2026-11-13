"""Category table for the post-fix re-baseline (CPU) from fix_eval.py output rows.

Columns, never merged: opener / turn-taking | tool choice | argument grounding | post-tool response | audio.
Usage: rebaseline_report.py fix_eval.json [...] [--md out.md]
"""
from __future__ import annotations

import argparse
import json


def row(r: dict) -> dict:
    s1 = r.get("S1") or {}
    talk = s1.get("overlap_frames_new_gt94") or []
    forced = s1.get("forced_bos_before_call") or []
    opener = "ok" if r.get("S1_pass") else f"talk-over {len(talk)} fr" + (f", forced BOS {forced[:2]}" if forced else "")
    calls = r.get("calls") or []
    tool = f"{len(calls)}: " + ", ".join(f"{c[0]}({json.dumps(c[1], ensure_ascii=False)})" for c in calls)[:200]
    bad = [p for p in r.get("slot_provenance") or [] if p[2] != "grounded"]
    grounding = "ok" if not bad else "; ".join(f"{a}={v!r}:{st}" for a, v, st in bad)[:160]
    post = []
    for x in r.get("resume_inputs") or []:
        na = x.get("next_action") or {}
        nxt = "call" if na.get("pred_func") == 20 else "reply/PAD"
        post.append(f"@{x['resume_frame']} in={x['consumed']} next={nxt} m={na.get('sotc_minus_pad')}")
    lat = (r.get("latency") or {}).get("first_reply_bos_frame")
    failed = [k for k in r.get("score_failed") or [] if k.startswith(("says", "not_says", "asks", "no_redundant",
                                                                       "resolved", "backend", "model_no"))]
    return {"session": r["session"], "opening": r.get("opening"), "opener/turn": opener, "tool": tool,
            "grounding": grounding, "post-tool": " | ".join(post) or "no FC",
            "reply BOS": lat, "content fails": ", ".join(failed) or "-",
            "audio (ASR)": (r.get("asr_text") or "")[-140:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = [row(r) for f in a.files for r in json.load(open(f))]
    cols = list(rows[0].keys()) if rows else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(x[c]).replace("|", "/") for c in cols) + " |" for x in rows]
    print("\n".join(md))
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
