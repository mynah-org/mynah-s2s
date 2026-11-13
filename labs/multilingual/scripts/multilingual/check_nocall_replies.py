"""Corpus v2 (CPU): for rule-no-call intents, does the constrained teacher's reply ask for the missing data without
inventing it (owner, 2026-11-12)?
  status_no_id / return_no_id : the reply must ASK for the name and/or postcode (or order details) and must not contain a
                                full name, a postcode or any digit run the caller did not say;
  out_of_scope                : the reply must decline / state the scope, with no tool call;
Only the reply AFTER the caller's turn is checked (the opener is identical for every family).
Usage: check_nocall_replies.py --teacher teacher.json --plan plan.jsonl --out f.json [--tokenizer NAME]
The reply is what the student is SUPERVISED on: the agent-channel tokens at frames >= the caller onset (`lead`), decoded
with --tokenizer. Fix 2026-11-13: the old string method (strip the canonical opener from `teacher_text`) returned the
whole text when a decode batch produced a variant opener, and that opener ("I can only help with ...") satisfied the
out-of-scope check by itself. The string method is kept only as a fallback when no tokenizer is given.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ASK = re.compile(r"\b(name|postcode|post code|zip|order number|details|full name)\b", re.I)
SCOPE = re.compile(r"\b(only help|can only|furniture orders|not able to|cannot help|can't help)\b", re.I)


def reply_after_opener(text: str, opener: str) -> str:
    t = text.replace("</s>", " ").replace("<s>", " ")
    o = opener.replace("</s>", " ").replace("<s>", " ")
    t, o = " ".join(t.split()), " ".join(o.split())
    return t[len(o):].strip() if t.startswith(o) else t


def reply_from_frames(v: dict, tok) -> str:
    """Supervised reply: agent text tokens at frames >= lead (PAD 12 dropped, BOS/EOS rendered as spaces)."""
    ids = [x for x in v["main_text"][v["lead"]:] if x != 12]
    return " ".join(tok.decode(ids).replace("</s>", " ").replace("<s>", " ").split())


def judge(intent: str, reply: str, said: str) -> tuple[bool, str]:
    digits_said = set(re.findall(r"\d+", said))
    invented_digits = [d for d in re.findall(r"\d{3,}", reply) if d not in digits_said]
    if intent in ("s4:status_no_id", "s4:return_no_id"):
        if invented_digits:
            return False, f"invented digits {invented_digits}"
        if not ASK.search(reply):
            return False, "does not ask for the missing data"
        return True, "asks"
    if intent == "s4:out_of_scope":
        return (True, "declines") if SCOPE.search(reply) else (False, "no scope statement")
    return True, "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="")
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    d = json.load(open(a.teacher))
    plan = {r["id"]: r for r in map(json.loads, open(a.plan)) if r["lang"] == "en"}
    opener = "Hi! How can I help you today? I can help you with furniture orders, product questions, and returns. " \
             "Just let me know what you need."
    rows, c = [], Counter()
    for i, v in d["items"].items():
        p = plan.get(i)
        if not p or p.get("target"):
            continue
        rep = reply_from_frames(v, tok) if tok else reply_after_opener(v.get("teacher_text", ""), opener)
        ok, why = judge(p["cat"], rep, p["text"])
        rows.append({"id": i, "intent": p["cat"], "keep": v["keep"], "reason": v["reason"], "ok": ok, "why": why,
                     "caller": p["text"], "reply": rep[:300], "suppressed": v.get("suppressed")})
        c[(p["cat"], ok)] += 1
    Path(a.out).write_text(json.dumps({"summary": {f"{k[0]}|{'ok' if k[1] else 'bad'}": n for k, n in c.items()},
                                       "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps({f"{k[0]}|{'ok' if k[1] else 'bad'}": n for k, n in c.items()}))
    for r in rows[:8]:
        print(r["intent"], r["ok"], r["why"], "|", r["caller"][:80], "->", r["reply"][:160])


if __name__ == "__main__":
    main()
