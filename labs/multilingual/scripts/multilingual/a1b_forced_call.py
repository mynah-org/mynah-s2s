"""A1b (owner GO 2026-11-13, hard cap 5 GPU-min incl. loading): forced-call text oracle with the REAL function head.

Separates text understanding, canonicalisation to the English targets, and argument generation. It is NOT a duplex
compatibility test: the call DECISION is forced, and the user turn is text, not audio.
Per item:
  context  the Nano-v2 chat template with the exact duplex prompt as system message (as A1), user = the condition text,
           then 'Assistant\\n<think></think>' (plain token embeddings, as the text_to_text path);
  call     the function-channel FC fast path of the duplex runtime (wrapper FC async / greedy_fc): every step is
           fusion(agent = PAD, audio = the wrapper's silence embedding, function = previous function token), read by
           function_head. Step 1 consumes SOTC (20). The format-only prefix common to every target,
           '<TOOLCALL>[{"name": "' (10 tokens, CPU-verified), is forced; tool name, arguments and EOTC (21) are free.
           Stop at EOTC or --max-call tokens.
  lm diag  (diagnostic only, after the function-head runs): the same forced prefix as plain tokens, read by lm_head.
Conditions: gold_en (the parallel EN text, once per family), gold_own (IT/FR), rnnt (A0 E0|lang, IT/FR).
Order (fixed so the budget cuts the least essential): fh base gold_en, fh base gold_own, fh base rnnt,
fh L1 s120 gold_own, fh L1 s120 gold_en, lm base gold_own.
Scoring (strict, as glex_free): tool, names, postcodes, concept semantic (the EN concept noun present in the query),
EN canonical query exact, full call exact.
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
from glex_free import score_call  # noqa: E402
from session_distill import EOTC, PAD, SOTC, rule_call_text  # noqa: E402

CALL_RE = re.compile(r"<TOOLCALL>.*?</TOOLCALL>", re.S)


def items_for(rows, text, a0):
    """gold_en once per (set, family); gold_own and rnnt for IT/FR rows."""
    out, seen = [], set()
    for r in rows:
        root = "glex" if r["data"] == "glex" else "val"
        fam = (r["set"], r["id"])
        if fam not in seen and text.get((root, r["id"], "en")):
            seen.add(fam)
            out.append((f"{r['set']}|{r['id']}|en|gold_en", "gold_en", {**r, "lang": "en"}, text[(root, r["id"], "en")]))
        if r["lang"] in ("it", "fr"):
            if text.get((root, r["id"], r["lang"])):
                out.append((f"{r['set']}|{r['id']}|{r['lang']}|gold_own", "gold_own", r, text[(root, r["id"], r["lang"])]))
            if a0.get((r["set"], r["id"], r["lang"])) is not None:
                out.append((f"{r['set']}|{r['id']}|{r['lang']}|rnnt", "rnnt", r, a0[(r["set"], r["id"], r["lang"])]))
    return out


def schedule(ctx_len: int, prefix: list[int], t: int, gen: list[int]) -> tuple[str, int]:
    """Row input at step t counted from the start of its context: ('text', ctx token index) while in the context, then
    ('fc', function token): SOTC, the forced prefix, then the row's own generated function tokens."""
    if t < ctx_len:
        return "text", t
    k = t - ctx_len
    if k == 0:
        return "fc", SOTC
    if k <= len(prefix):
        return "fc", prefix[k - 1]
    return "fc", gen[min(k - len(prefix) - 1, len(gen) - 1)]     # a finished row repeats its last token (ignored)


def concept_ok(query: str | None, row: dict) -> bool | None:
    if row["target"]["name"] != "search_catalog":
        return None
    c = (row.get("slots") or {}).get("concept")
    return bool(query) and bool(c) and c.lower() in query.lower()


def main():  # pragma: no cover - GPU
    import torch
    from transformers import AutoTokenizer
    from distill_adapter import Duplex, load_engine
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    import thinker_lora as TL
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--a0", required=True)
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-call", type=int, default=56)
    ap.add_argument("--budget-s", type=float, default=240)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows = [r for r in json.load(open(a.sets))["rows"] if r["set"] in ("glex_dev", "val")]
    for r in rows:
        if isinstance(r["slots"], str):
            r["slots"] = eval(r["slots"])  # noqa: S307 - our own plan files
    text = {}
    for tag, root in (("glex", a.glex_data), ("val", a.val_data)):
        for m in map(json.loads, open(Path(root) / "manifest.jsonl")):
            text[(tag, m["id"], m["lang"])] = m.get("text", "")
    a0 = {(d["set"], d["id"], d["lang"]): d["res"]["E0|lang"]["hyp"] for d in map(json.loads, open(a.a0)) if "res" in d}
    items = items_for(rows, text, a0)
    prompt = json.loads(Path(a.teacher).read_text())["prompt"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ctx = {i[0]: tok.apply_chat_template([{"role": "system", "content": prompt + " /no_think"},
                                          {"role": "user", "content": i[3]}], add_generation_prompt=True, tokenize=True)
           for i in items}
    encs = [tok.encode(rule_call_text(r["target"]), add_special_tokens=False) for r in rows]
    k = next(j for j in range(min(map(len, encs))) if len({e[j] for e in encs}) > 1)
    prefix = encs[0][:k]
    first = next(iter(ctx.values()))
    common = next((j for j in range(min(map(len, ctx.values()))) if any(c[j] != first[j] for c in ctx.values())),
                  min(map(len, ctx.values())))
    plan = {"n_items": len(items), "by_cond": {c: sum(1 for i in items if i[1] == c) for c in ("gold_en", "gold_own", "rnnt")},
            "forced_prefix": tok.decode(prefix), "forced_prefix_ids": prefix, "sotc": SOTC, "eotc": EOTC,
            "common_ctx_tokens": common, "max_ctx": max(map(len, ctx.values()))}
    (out / "a1b_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if a.plan_only:
        return
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prompt)
    e = D.stt.embed_tokens
    sil = D.w._silence_embedding_cache.reshape(-1).to(D.dev, D.dtype)
    pad_e = e(torch.tensor([PAD], device=D.dev))[0].to(D.dtype)
    ck = torch.load(a.lora, map_location="cpu", weights_only=False)
    mods = None
    jl = out / "a1b_rows.jsonl"

    def emit(d):
        with open(jl, "a") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    emit({"meta": {"plan": plan, "t_load": round(time.monotonic() - t0, 1)}})

    @torch.no_grad()
    def run_fh(chunk):
        B = len(chunk)
        cs = [ctx[i[0]] for i in chunk]
        cache = _hybrid_cache_cls(D.stt.llm)(D.stt.llm.config, B, dtype=D.dtype, device=D.dev, ssm_dtype=torch.float32)
        D.stt(e(torch.tensor([cs[0][:common]], device=D.dev)).to(D.dtype).expand(B, -1, -1).contiguous(), cache=cache)
        gen = [[] for _ in range(B)]
        done = [False] * B
        t = common
        while not all(done):
            xs = []
            for b in range(B):
                kind, v = schedule(len(cs[b]), prefix, t, gen[b] if gen[b] else [PAD])
                if kind == "text":
                    xs.append(e(torch.tensor([cs[b][v]], device=D.dev))[0].to(D.dtype))
                else:
                    xs.append(D.fusion(agent_text_embeds=pad_e[None, None], user_audio_embeds=sil[None, None],
                                       user_text_embeds=None,
                                       function_embeds=e(torch.tensor([[v]], device=D.dev)).to(D.dtype))[0, 0])
            ans = D.stt(torch.stack(xs)[:, None], cache=cache)
            cache = ans.get("cache", cache)
            nxt = ans["function_logits"][:, -1].float().argmax(-1).tolist()
            for b in range(B):
                if done[b] or t < len(cs[b]) + len(prefix):        # output after the last forced input only
                    continue
                gen[b].append(int(nxt[b]))
                if gen[b][-1] == EOTC or len(gen[b]) >= a.max_call:
                    done[b] = True
            t += 1
        return [tok.decode(prefix + [x for x in g if x != EOTC]) + "|EOTC" * (bool(g) and g[-1] == EOTC) for g in gen]

    @torch.no_grad()
    def run_lm(chunk):
        B = len(chunk)
        cs = [ctx[i[0]] + prefix for i in chunk]
        cache = _hybrid_cache_cls(D.stt.llm)(D.stt.llm.config, B, dtype=D.dtype, device=D.dev, ssm_dtype=torch.float32)
        D.stt(e(torch.tensor([cs[0][:common]], device=D.dev)).to(D.dtype).expand(B, -1, -1).contiguous(), cache=cache)
        gen = [[] for _ in range(B)]
        done = [False] * B
        t = common
        while not all(done):
            inp = [cs[b][t] if t < len(cs[b]) else gen[b][-1] for b in range(B)]
            ans = D.stt(e(torch.tensor(inp, device=D.dev)[:, None]).to(D.dtype), cache=cache)
            cache = ans.get("cache", cache)
            lg = ans["text_logits"][:, -1].float()
            lg[:, PAD] = -float("inf")
            nxt = lg.argmax(-1).tolist()
            for b in range(B):
                if done[b] or t < len(cs[b]) - 1:
                    continue
                gen[b].append(int(nxt[b]))
                if gen[b][-1] in (2, 11) or "</TOOLCALL>" in tok.decode(gen[b]) or len(gen[b]) >= a.max_call:
                    done[b] = True
            t += 1
        return [tok.decode(prefix + [x for x in g if x not in (2, 11)]) for g in gen]

    runs = [("base", "fh", "gold_en"), ("base", "fh", "gold_own"), ("base", "fh", "rnnt"),
            ("l1_s120", "fh", "gold_own"), ("l1_s120", "fh", "gold_en"), ("base", "lm", "gold_own")]
    stop = False
    for arm, head, cond in runs:
        if arm == "l1_s120" and mods is None:
            mods = TL.install(D.stt.llm, layers=tuple(ck["layers"]), targets=tuple(ck["targets"]), r=ck["r"], alpha=ck["alpha"])
            TL.load_state(mods, ck["state"])
        if mods is not None:
            TL.set_enabled(mods, arm == "l1_s120")
        its = sorted([i for i in items if i[1] == cond], key=lambda i: len(ctx[i[0]]))
        for b in range(0, len(its), a.batch):
            if time.monotonic() - t0 > a.budget_s:
                emit({"stop": "budget", "arm": arm, "head": head, "cond": cond, "done": b})
                stop = True
                break
            chunk = its[b:b + a.batch]
            outs = run_fh(chunk) if head == "fh" else run_lm(chunk)
            for (iid, c, r, utext), o in zip(chunk, outs):
                body = o.replace("|EOTC", "")
                m = CALL_RE.search(body)
                sc = score_call(m.group(0) if m else "", r["target"])     # strict: no repair of unterminated calls
                q = sc["slots"].get("query", {}).get("got") if sc["parsed"] else None
                emit({"arm": arm, "head": head, "cond": c, "item": iid, "set": r["set"], "id": r["id"], "lang": r["lang"],
                      "target": r["target"], "user_text": utext, "out": o[:400], "score": sc, "eotc": o.endswith("|EOTC"),
                      "concept_ok": concept_ok(q, r),
                      "canonical_exact": (q == r["target"]["arguments"].get("query")) if r["target"]["name"] == "search_catalog" else None})
            print(arm, head, cond, b + len(chunk), "/", len(its), round(time.monotonic() - t0, 1), flush=True)
        if stop:
            break
    print("done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
