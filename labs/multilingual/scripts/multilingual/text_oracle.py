"""Diagnostic upper bounds (NOT a proposed product path): the VoiceChat Thinker weights, text-only, answering the same
QA items from (a) the multilingual RNNT transcript produced inside the streaming pipeline (`user_asr` of an e2e run
with `rnnt=ml`; only the `<xx-XX>` language tags are removed) and (b) the reference question text (perfect ASR).

Same text path as B2 `vcsup` (Nano chat template, `/no_think`, id 12 banned, greedy) and the same answer regexes as
e2e_qa.py → separates ASR/representation failures from LLM comprehension failures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e2e_qa import score  # noqa: E402
from llm_text_bench import load, reply_lang  # noqa: E402

SYS = "/no_think You are a helpful voice assistant. Answer briefly."


@torch.no_grad()
def ask(tok, m, text):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    g = m.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=0, bad_words_ids=[[12]], eos_token_id=[11, 2])
    out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
    return re.split(r"<SPECIAL_1[12]>|</s>|<extra_id_1>", out)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="rows.jsonl of an e2e run with the ML RNNT (user_asr = transcript)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    out.mkdir(parents=True)
    answers = {it["id"]: it["answers"] for it in json.load(open(a.corpus))["items"]}
    rows = [json.loads(l) for l in open(a.rows)]
    tok, m, info = load("vcsup")
    res, summ = [], {}
    for r in rows:
        asr = re.sub(r"<[a-z]{2,3}-[A-Z]{2}>", " ", r["user_asr"]).strip()
        for src, text in (("rnnt", asr), ("reference", r["question"])):
            o = ask(tok, m, text)
            ok = score(o, answers[r["id"]])
            x = {"src": src, "lang": r["lang"], "id": r["id"], "voice": r["voice"], "input": text, "out": o, "ok": ok,
                 "reply_lang": reply_lang(o)}
            res.append(x)
            summ.setdefault(src, {}).setdefault(r["lang"], []).append(ok)
            print(f"[{src}] {r['lang']} {r['id']} {'OK' if ok else '--'} | {text[:50]!r} -> {o[:90]!r}", flush=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res) + "\n")
    s = {k: {l: f"{sum(v)}/{len(v)}" for l, v in d.items()} for k, d in summ.items()}
    (out / "summary.json").write_text(json.dumps({"source_rows": a.rows, "model": info, "summary": s}, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
