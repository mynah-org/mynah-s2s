"""B1 (owner GO 2026-11-13): text-teacher distributions on the canonical call tokens, for the masked KL of
`session_distill train --kl-teacher`.

Teacher = the VoiceChat Thinker with the REAL function head, A1b format: Nano-v2 chat template with the exact duplex
prompt as system message and the clip's GOLD transcript (its own language) as the user turn, then the FC fast-path
steps (fusion of agent PAD, the wrapper's silence embedding, the previous function token). Inputs SOTC, c0, c1, ...
teacher-forced with the CANONICAL call tokens (items[i]['fc'], the training target), so position j predicts c_j from
exactly the same call prefix the student sees at its j-th 'fc' position (student_targets: func_in = SOTC, c0, ...).
Only TRAIN keys with a call. Stored per key, for the masked positions only: top-32 token ids and log-probs (full-vocab
log-softmax), plus whether the teacher's argmax reproduces each span exactly.
Mask: token positions overlapping the characters of the tool name, `full_name` and `postcode` values in the canonical
call text. `query` is never distilled (A1b: the teacher does not canonicalise IT/FR concepts).
Output: torch .pt {key "id|lang": {"pos": [j...], "ids": [[32]...], "lp": [[32]...], "spans": {slot: [j...]},
"exact": {slot: bool}}} + a report json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from session_distill import EOTC, PAD, SOTC, kept_keys, rule_call_text, usable_keys  # noqa: E402

DISTILL_SLOTS = ("tool", "full_name", "postcode")


def char_spans(call: str, target: dict) -> dict[str, tuple[int, int]]:
    """Character spans in the canonical call text: the tool name and the distilled argument values."""
    out = {}
    i = call.find('"' + target["name"] + '"')
    if i >= 0:
        out["tool"] = (i + 1, i + 1 + len(target["name"]))
    for k in ("full_name", "postcode"):
        v = (target.get("arguments") or {}).get(k)
        if isinstance(v, str):
            s = json.dumps(v, ensure_ascii=False)[1:-1]
            j = call.find('"' + s + '"')
            if j >= 0:
                out[k] = (j + 1, j + 1 + len(s))
    return out


def token_spans(call_tokens: list[int], decode, spans: dict) -> dict[str, list[int]]:
    """Token positions (index into call_tokens) whose characters overlap each span; EOTC never included."""
    out = {k: [] for k in spans}
    for q, t in enumerate(call_tokens):
        if t == EOTC:
            continue
        lo, hi = len(decode(call_tokens[:q])), len(decode(call_tokens[:q + 1]))
        for k, (s0, s1) in spans.items():
            if lo < s1 and hi > s0:
                out[k].append(q)
    return out


def main():  # pragma: no cover - GPU
    import torch
    from transformers import AutoTokenizer
    from distill_adapter import Duplex, load_engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher-prompt", required=True, help="teacher.json with the duplex prompt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--keys-from-sequence", default="", help="batch_sequence.json: only the keys the training samples")
    ap.add_argument("--save-every", type=int, default=8, help="batches between incremental saves (resume-safe)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.data) / "manifest.jsonl"))}
    split = {r["id"]: r["split"] for r in man.values()}
    tr, _ = kept_keys(prep["items"], usable_keys(man), split, ["en", "it", "fr"])
    items = {i: v for i, v in prep["items"].items() if v["keep"]}
    prompt = json.loads(Path(a.teacher_prompt).read_text())["prompt"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    dec = tok.decode
    keys, rep = [], {"n_train_keys": len(tr), "skip": {}}
    plan = {}
    wanted = None
    if a.keys_from_sequence:                         # the training draws a known, fixed batch sequence (Random(1))
        wanted = {k for b in json.load(open(a.keys_from_sequence))["seq"] for k in b}
        rep["keys_from_sequence"] = len(wanted)
    for i, l in tr:
        if wanted is not None and f"{i}|{l}" not in wanted:
            continue
        it = items[i]
        tgt = man[(i, "en")].get("target")
        if not it.get("fc") or not tgt or not tgt.get("arguments"):
            rep["skip"]["no_rule_call"] = rep["skip"].get("no_rule_call", 0) + 1
            continue
        fc = it["fc"]
        call = rule_call_text(tgt)
        if tok.encode(call, add_special_tokens=False) + [EOTC] != fc:
            rep["skip"]["fc_mismatch"] = rep["skip"].get("fc_mismatch", 0) + 1
            continue
        text = man[(i, l)].get("text")
        if not text:
            rep["skip"]["no_text"] = rep["skip"].get("no_text", 0) + 1
            continue
        spans = token_spans(fc, dec, char_spans(call, tgt))
        ctx = tok.apply_chat_template([{"role": "system", "content": prompt + " /no_think"},
                                       {"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        plan[f"{i}|{l}"] = {"fc": fc, "spans": spans, "ctx": ctx}
        keys.append(f"{i}|{l}")
    rep["n_keys"] = len(keys)
    rep["spans_per_slot"] = {s: sum(1 for k in keys if plan[k]["spans"].get(s)) for s in DISTILL_SLOTS}
    ex = keys[0]
    rep["example"] = {"key": ex, "call": dec(plan[ex]["fc"][:-1]),
                      "span_text": {s: dec([plan[ex]["fc"][j] for j in v]) for s, v in plan[ex]["spans"].items()}}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (out / "teacher_kl_plan.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.plan_only:
        return
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prompt)
    e = D.stt.embed_tokens
    sil = D.w._silence_embedding_cache.reshape(-1).to(D.dev, D.dtype)
    pad_e = e(torch.tensor([PAD], device=D.dev))[0].to(D.dtype)
    part = out / "teacher_kl.partial.pt"
    store = torch.load(part, map_location="cpu", weights_only=False) if part.exists() else {}
    todo = [k for k in keys if k not in store]
    print("resume:", len(store), "done,", len(todo), "to do", flush=True)
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        seqs, fcpos = [], []
        with torch.no_grad():
            for k in chunk:
                p = plan[k]
                ctx_e = e(torch.tensor(p["ctx"], device=D.dev)).to(D.dtype)
                fin = torch.tensor([SOTC] + p["fc"][:-1], device=D.dev)
                fx = D.fusion(agent_text_embeds=pad_e.expand(len(fin), -1)[None], user_audio_embeds=sil.expand(len(fin), -1)[None],
                              user_text_embeds=None, function_embeds=e(fin)[None].to(D.dtype))[0]
                seqs.append(torch.cat([ctx_e, fx], 0))
                fcpos.append(len(p["ctx"]))
            X = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)       # right padding: causal, never read
            o = D.stt(X, cache=None)
            for r, k in enumerate(chunk):
                p = plan[k]
                n = len(p["fc"])
                lf = torch.log_softmax(o["function_logits"][r, fcpos[r]:fcpos[r] + n].float(), -1)    # [n, V]
                arg = lf.argmax(-1).tolist()
                exact = {s: bool(js) and all(arg[j] == p["fc"][j] for j in js) for s, js in p["spans"].items()}
                pos = sorted({j for s, js in p["spans"].items() if exact[s] for j in js})
                v, ids = lf[pos].topk(a.topk, -1) if pos else (torch.empty(0), torch.empty(0))
                store[k] = {"pos": pos, "ids": ids.int().cpu(), "lp": v.half().cpu(), "spans": p["spans"], "exact": exact,
                            "gold_in_topk": [int(p["fc"][j]) in ids[q].tolist() for q, j in enumerate(pos)] if pos else []}
        if (b // a.batch + 1) % a.save_every == 0:
            torch.save(store, part)
        print("teacher", len(store), "/", len(keys), round(time.monotonic() - t0, 1), flush=True)
    torch.save(store, part)
    assert set(store) >= set(keys), "incomplete teacher"
    torch.save(store, out / "teacher_kl.pt")
    rep["exact_spans"] = {s: sum(bool(v["exact"].get(s)) for v in store.values()) for s in DISTILL_SLOTS}
    rep["keys_with_any_distilled_position"] = sum(1 for v in store.values() if v["pos"])
    rep["gold_in_top32_all"] = all(all(v["gold_in_topk"]) for v in store.values())
    rep["t"] = round(time.monotonic() - t0, 1)
    (out / "teacher_kl_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("n_keys", "spans_per_slot", "exact_spans", "keys_with_any_distilled_position",
                                          "gold_in_top32_all", "t")}), flush=True)


if __name__ == "__main__":
    main()
