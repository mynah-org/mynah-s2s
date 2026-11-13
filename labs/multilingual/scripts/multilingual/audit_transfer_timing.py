"""Corpus v2 (CPU, read-only): timing / turn-boundary audit of the student targets of rule-call families, split by
admission status (teacher_no_call = synthetic SOTC vs agree = the teacher's own SOTC). Owner request 2026-11-13.

For every admitted family of the chosen intents and every usable language clip, the REAL student targets are built
with `session_distill.student_targets` (same call as `train`) and checked:
  sotc_rel_end   frames from the caller's speech end in that language (x_end) to the supervised SOTC;
  sotc_present   the SOTC is inside the session audio (lead + clip + tail), so the call is supervised at all;
  talk_over      agent text tokens (not PAD/EOS) labelled between the caller onset and x_end (talking over the caller);
  pre_call_text  agent text tokens between x_end and the SOTC (an offer / question before the call);
  onset_eos      the turn-yield EOS is labelled after the caller onset and before the SOTC.
Usage: audit_transfer_timing.py --teacher t.json --manifest m.jsonl [--cats s4:transfer] --out report.json
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
import session_distill as S  # noqa: E402


def audit_one(v: dict, lang: str, n_audio: int) -> dict:
    x_end = v["end"][lang]
    st = S.student_targets(v, v["lead"], v["en_end"], x_end, n_audio, lang == "en")
    frames = [(p[1], t, f) for p, t, f in zip(st["pos"], st["text_lbl"], st["func_lbl"]) if p[0] == "a"]
    sotc = next((t for t, _, f in frames if f == S.SOTC), None)
    cs = v["lead"]
    talk = sum(1 for t, x, _ in frames if cs <= t < x_end and x not in (S.PAD, S.EOS))
    pre = sum(1 for t, x, _ in frames if sotc is not None and x_end <= t < sotc and x not in (S.PAD, S.EOS, S.BOS))
    eos = any(x == S.EOS and cs <= t and (sotc is None or t < sotc) for t, x, _ in frames)
    return {"sotc_present": sotc is not None, "sotc_rel_end": None if sotc is None else sotc - x_end,
            "talk_over": talk, "pre_call_text": pre, "onset_eos": eos,
            "fc_tokens": sum(1 for k in st["kind"] if k == "fc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cats", default="s4:transfer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    d = json.load(open(a.teacher))
    tail = int(math.ceil(d.get("tail_s", 4.0) * 16000 / S.FRAME))
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(a.manifest))}
    use = S.usable_keys(man)
    cats = set(a.cats.split(","))
    agg = defaultdict(lambda: defaultdict(Counter))
    rows = []
    for i, v in sorted(d["items"].items()):
        cat = man[(i, "en")]["cat"]
        if cat not in cats or not v["keep"]:
            continue
        for lang in ("en", "it", "fr"):
            if (i, lang) not in use:
                continue
            r = audit_one(v, lang, v["end"][lang] + tail)
            g = agg[f"{cat}|{v['reason']}"]
            g["n"][lang] += 1
            g["sotc_present"][r["sotc_present"]] += 1
            g["sotc_rel_end"][r["sotc_rel_end"]] += 1
            g["talk_over>0"][r["talk_over"] > 0] += 1
            g["pre_call_text>0"][r["pre_call_text"] > 0] += 1
            g["onset_eos"][r["onset_eos"]] += 1
            rows.append({"id": i, "lang": lang, "reason": v["reason"], **r})
    rep = {"tail_frames": tail, "groups": {k: {m: dict(sorted(c.items(), key=lambda x: str(x[0])))
                                               for m, c in g.items()} for k, g in agg.items()}, "rows": rows}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["groups"], indent=1, default=str))


if __name__ == "__main__":
    main()
