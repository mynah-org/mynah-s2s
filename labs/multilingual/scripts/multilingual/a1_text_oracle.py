"""A1 (owner GO 2026-11-13; GPU only after A0 + the CPU harness check; cap 6 GPU-min): VoiceChat Thinker TEXT path on
the same G-LEX-dev / val items as the free-running speech evaluation. Outside the duplex loop; no text in the audio slot.

Text path actually available in the checkpoint (lexical-loss-workstream.md section 2):
  - VoiceChat-11B has no duplex user-text channel (predict_user_text False, user_text_embeds None in the wrapper);
  - the Thinker LLM was co-trained on plain text (text_to_text_loss_weight 0.5: llm(embed_tokens(ids)) -> lm_head);
    the text-data prompt format is NOT in the released config (input_cfg ''), so the Nemotron-Nano-v2 chat template
    is used. The duplex system prompt string IS that template's system+tools block (same text), so the system message
    = the exact duplex prompt; '/no_think' is appended (the template strips it from the content).
Conditions per item: gold_own (TTS text in the clip's language), gold_en (the parallel EN family text; IT/FR rows only),
rnnt (the A0 transcript, ML encoder original LNs att [56,0], forced language prompt).
Arms: base Thinker; L1 s120 LoRA (thinker_lora, unmerged). Heads: lm_head (primary); function_head on a small subset.
Decoding: greedy; the shared system prefix is prefilled once per batch, then every row advances one token per step
(forced user / header tokens, then its own argmax), so rows never need padding (V2 hybrid cache, as greedy_fc).
PAD (12) is banned as in E3 (text_oracle.py); stop at EOS 2 / 11, '</TOOLCALL>' or --max-new tokens.
Usage: a1_text_oracle.py --sets free_sets.json --a0 a0_rows.jsonl --glex-data D --val-data D --teacher T --lora L --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon_v2 as L  # noqa: E402
from glex_free import score_call  # noqa: E402

STOP_IDS = (2, 11)
BAN_ID = 12
CALL_RE = re.compile(r"<TOOLCALL>.*?</TOOLCALL>", re.S)


def build_items(rows, text, a0):
    """[(item_id, condition, row, user_text)]; rnnt only where A0 produced a transcript."""
    items = []
    for r in rows:
        root = "glex" if r["data"] == "glex" else "val"
        own = text.get((root, r["id"], r["lang"]))
        if own:
            items.append((f"{r['set']}|{r['id']}|{r['lang']}|gold_own", "gold_own", r, own))
        if r["lang"] != "en" and text.get((root, r["id"], "en")):
            items.append((f"{r['set']}|{r['id']}|{r['lang']}|gold_en", "gold_en", r, text[(root, r["id"], "en")]))
        hyp = a0.get((r["set"], r["id"], r["lang"]))
        if hyp is not None:
            items.append((f"{r['set']}|{r['id']}|{r['lang']}|rnnt", "rnnt", r, hyp))
    return items


def step_inputs(seqs: list[list[int]], gen: list[list[int]], t: int, prefix: int) -> list[int]:
    """Input token of every row at decode step t (t = 0 is the first token after the shared prefix): its own forced
    token while inside its prompt, else its last generated token."""
    out = []
    for s, g in zip(seqs, gen):
        k = prefix + t
        out.append(s[k] if k < len(s) else g[-1])
    return out


def query_kind(gen_query: str | None, row: dict) -> str | None:
    """canonical = the EN canonical query; source = the IT/FR surface form of the concept (not translated);
    other = anything else; None when not a catalog item or no query."""
    if row["target"]["name"] != "search_catalog" or gen_query is None:
        return None
    gold = row["target"]["arguments"]["query"].lower().strip()
    q = gen_query.lower().strip()
    if q == gold:
        return "canonical"
    c = next((x for x in L.CONCEPTS if x[0] == (row.get("slots") or {}).get("concept")), None)
    if c and row["lang"] in ("it", "fr"):
        forms = [c[2][0]] + list(c[3]) if row["lang"] == "it" else [c[4][0]] + list(c[5])
        if any(f.lower() in q for f in forms):
            return "source_language"
    if c and c[0] in q:
        return "concept_en_other_attrs"
    return "other"


def main():  # pragma: no cover - GPU
    import torch
    from transformers import AutoTokenizer
    from distill_adapter import Duplex, load_engine
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    import thinker_lora as TL
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--sets-keep", default="glex_dev,val")
    ap.add_argument("--a0", required=True)
    ap.add_argument("--a0-arm", default="E0|lang")
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--tokenizer", required=True, help="Nemotron-Nano-9B-v2 snapshot dir (chat template)")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--budget-s", type=float, default=300)
    ap.add_argument("--plan-only", action="store_true", help="CPU: render, tokenize and report; no model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows = [r for r in json.load(open(a.sets))["rows"] if r["set"] in a.sets_keep.split(",")]
    for r in rows:
        if isinstance(r["slots"], str):
            r["slots"] = eval(r["slots"])  # noqa: S307 - our own plan files
    text = {}
    for tag, root in (("glex", a.glex_data), ("val", a.val_data)):
        for m in map(json.loads, open(Path(root) / "manifest.jsonl")):
            text[(tag, m["id"], m["lang"])] = m.get("text", "")
    a0 = {}
    for ln in open(a.a0):
        d = json.loads(ln)
        if "res" in d:
            a0[(d["set"], d["id"], d["lang"])] = d["res"][a.a0_arm]["hyp"]
    items = build_items(rows, text, a0)
    prompt = json.loads(Path(a.teacher).read_text())["prompt"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    def ids_for(user_text):
        msgs = [{"role": "system", "content": prompt + " /no_think"}, {"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    seqs = {i[0]: ids_for(i[3]) for i in items}
    pre = min(len(s) for s in seqs.values())
    first = next(iter(seqs.values()))
    common = next((k for k in range(pre) if any(s[k] != first[k] for s in seqs.values())), pre)
    plan = {"n_items": len(items), "by_condition": {c: sum(1 for i in items if i[1] == c) for c in ("gold_own", "gold_en", "rnnt")},
            "common_prefix_tokens": common, "max_len": max(len(s) for s in seqs.values()),
            "rendered_example": tok.decode(first)[-400:], "template_removed_no_think": "/no_think" not in tok.decode(first)}
    (out / "a1_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(json.dumps(plan, indent=1, ensure_ascii=False), flush=True)
    if a.plan_only:
        return
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prompt)                          # the duplex prompt embedding is not used on the text path
    mods = None
    ck = torch.load(a.lora, map_location="cpu", weights_only=False)
    jl = out / "a1_rows.jsonl"

    def emit(d):
        with open(jl, "a") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    emit({"meta": {"plan": plan, "lora": a.lora, "t_load": round(time.monotonic() - t0, 1)}})
    e = D.stt.embed_tokens

    def decode(batch_items, head):
        B = len(batch_items)
        ss = [seqs[i[0]] for i in batch_items]
        cache = _hybrid_cache_cls(D.stt.llm)(D.stt.llm.config, B, dtype=D.dtype, device=D.dev, ssm_dtype=torch.float32)
        with torch.no_grad():
            px = e(torch.tensor([ss[0][:common]], device=D.dev)).to(D.dtype).expand(B, -1, -1).contiguous()
            D.stt(px, cache=cache)
            gen = [[] for _ in range(B)]
            done = [False] * B
            t = 0
            while not all(done) and t < max(len(s) for s in ss) - common + a.max_new:
                inp = step_inputs(ss, [g or [0] for g in gen], t, common)
                ans = D.stt(e(torch.tensor(inp, device=D.dev)[:, None]).to(D.dtype), cache=cache)
                cache = ans.get("cache", cache)
                lg = (ans["text_logits"] if head == "lm" else ans["function_logits"])[:, -1].float()
                lg[:, BAN_ID] = -float("inf")
                nxt = lg.argmax(-1).tolist()
                for b in range(B):
                    if done[b] or common + t + 1 < len(ss[b]):
                        continue
                    gen[b].append(int(nxt[b]))
                    txt = tok.decode(gen[b])
                    if gen[b][-1] in STOP_IDS or "</TOOLCALL>" in txt or len(gen[b]) >= a.max_new:
                        done[b] = True
                t += 1
        return [tok.decode([x for x in g if x not in STOP_IDS]) for g in gen]

    runs = [("base", "lm", [i for i in items]),
            ("l1_s120", "lm", [i for i in items if i[1] in ("gold_own", "rnnt")]),
            ("base", "func", [i for i in items if i[1] == "gold_own" and i[2]["lang"] in ("it", "fr")])]
    for arm, head, its in runs:
        if arm == "l1_s120" and mods is None:
            mods = TL.install(D.stt.llm, layers=tuple(ck["layers"]), targets=tuple(ck["targets"]), r=ck["r"], alpha=ck["alpha"])
            TL.load_state(mods, ck["state"])
        if mods is not None:
            TL.set_enabled(mods, arm == "l1_s120")
        its = sorted(its, key=lambda i: len(seqs[i[0]]))
        for b in range(0, len(its), a.batch):
            if time.monotonic() - t0 > a.budget_s:
                emit({"stop": "budget", "arm": arm, "head": head, "done": b})
                break
            chunk = its[b:b + a.batch]
            outs = decode(chunk, head)
            for (iid, cond, r, utext), o in zip(chunk, outs):
                m = CALL_RE.search(o)
                sc = score_call(m.group(0) if m else "", r["target"])
                q = None
                if m and sc["parsed"]:
                    try:
                        q = (json.loads(m.group(0)[10:-11])[0].get("arguments") or {}).get("query")
                    except (json.JSONDecodeError, IndexError, AttributeError):
                        q = None
                emit({"arm": arm, "head": head, "item": iid, "cond": cond, "set": r["set"], "id": r["id"], "lang": r["lang"],
                      "target": r["target"], "user_text": utext, "out": o[:400], "score": sc,
                      "query_kind": query_kind(q, r), "gen_query": q})
            print(arm, head, b + len(chunk), "/", len(its), round(time.monotonic() - t0, 1), flush=True)
    print("done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
