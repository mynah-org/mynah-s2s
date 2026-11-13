"""Forced-candidate contrastive test (owner GO 2026-11-13, hard cap 6 GPU-min incl. smoke and loads).

Question: when the model hears a slot value it has not been trained on, does it give MORE probability to the value
actually spoken than to a wrong one, and does that preference come from the audio or from the decoder's prior?

For each clip (held-out only: G-LEX-dev clips with strict-ASR ok, plus train_v1 VAL clips; never G-LEX-test, never idev)
and each call slot, candidates differ from the canonical call in ONE slot value only (same tool, same keys, same
prefix):
  full_name  close   first name with the smallest edit distance in the whole name lexicon (ties: alphabetical)
             train   the closest TRAIN-split first name different from gold and from `close`
  postcode   last    last digit + 1 (mod 10)
             mid     third digit + 5 (mod 10)
  query      close   the concept with the smallest edit distance on the EN name (any split), same attributes
             train   the closest TRAIN concept of the same category (else any train concept), same attributes
These rules are fixed here, before any result. Scores are teacher-forced function-head log-probs of every call token
(student_targets layout, as glex_probe): total = sum over all call tokens (the shared prefix cancels exactly), and
norm = mean log-prob over the tokens from the first differing token to the end of the call (before EOTC); token counts
are recorded. Delta = gold - distractor.
Conditions: audio (real), neutral (caller frames replaced by the encoded silence frame, same length), swap (audio of
another clip of the same language and intent, on a pre-registered subset: every third clip).
Modes: --cpu-check (no model): candidates, tokenisation, prefix equality, positions, no target token in the context.
       default (GPU): smoke (determinism, prefix-logp equality, positions) then the arms in order; abort on smoke fail.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon_v2 as L  # noqa: E402
import session_corpus_v2 as C  # noqa: E402
from glex_probe import parse_spec, synthetic_record  # noqa: E402
from session_distill import BOS, EOTC, FRAME, PAD, rule_call_text, session_wav, student_targets  # noqa: E402


def edit(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def train_first_names() -> list[str]:
    return [f for i, f in enumerate(L.FIRST) if i < int(0.8 * len(L.FIRST))]


def closest(gold: str, pool: list[str], exclude: set) -> str | None:
    cand = sorted((edit(gold.lower(), p.lower()), p) for p in pool if p not in exclude)
    return cand[0][1] if cand else None


def distractors(target: dict, slots: dict) -> list[dict]:
    """[{slot, kind, target}] - each target differs from the gold one in exactly one argument value."""
    out, args = [], dict(target.get("arguments") or {})
    if target["name"] == "find_customer" and "full_name" in args:
        first, last = args["full_name"].split(" ", 1)
        close = closest(first, L.FIRST, {first})
        train = closest(first, train_first_names(), {first, close})
        for kind, f in (("close", close), ("train", train)):
            if f:
                out.append({"slot": "full_name", "kind": kind, "value": f"{f} {last}",
                            "target": {**target, "arguments": {**args, "full_name": f"{f} {last}"}}})
    if target["name"] == "find_customer" and "postcode" in args:
        pc = args["postcode"]
        for kind, pos, inc in (("last", 4, 1), ("mid", 2, 5)):
            v = pc[:pos] + str((int(pc[pos]) + inc) % 10) + pc[pos + 1:]
            out.append({"slot": "postcode", "kind": kind, "value": v,
                        "target": {**target, "arguments": {**args, "postcode": v}}})
    if target["name"] == "search_catalog" and slots.get("concept"):
        c = next(x for x in L.CONCEPTS if x[0] == slots["concept"])
        attrs = sorted([next(a for a in L.ATTRS if a[0] == n) for n in slots.get("attrs", [])], key=lambda a: a[1])
        cs = C.split_concepts()
        close = closest(c[0], [x[0] for x in L.CONCEPTS], {c[0]})
        same_cat = [x[0] for x in L.CONCEPTS if cs[x[0]] == "train" and x[1] == c[1]]
        train = closest(c[0], same_cat or [x[0] for x in L.CONCEPTS if cs[x[0]] == "train"], {c[0], close})
        for kind, name in (("close", close), ("train", train)):
            if name:
                cc = next(x for x in L.CONCEPTS if x[0] == name)
                q = C.canonical_query(cc, attrs)
                out.append({"slot": "query", "kind": kind, "value": q,
                            "target": {**target, "arguments": {**args, "query": q}}})
    return out


def select_clips(glex_plan, glex_man, glex_verify, val_man, val_teacher, val_plan_rows, per_lang_val=4):
    """G-LEX-dev clips with strict-ASR ok + VAL clips (2 find_customer + 2 catalog per language, deterministic)."""
    rows = []
    for r in glex_plan:
        k = (r["id"], r["lang"])
        if k in glex_man and glex_verify.get(f"{r['id']}|{r['lang']}", {}).get("ok"):
            rows.append({**r, "src": "glex_dev", "data": "glex", "path": glex_man[k]["path"]})
    for lang in ("en", "it", "fr"):
        got = {"find_customer": 0, "search_catalog": 0}
        for r in sorted(val_plan_rows, key=lambda x: x["id"]):
            k = (r["id"], r["lang"])
            m = val_man.get(k)
            if r["lang"] != lang or not m or m["split"] != "val" or not m.get("asr_ok", True) or m.get("excluded"):
                continue
            if not val_teacher.get(r["id"], {}).get("keep") or not r.get("target"):
                continue
            t = r["target"]["name"]
            if t in got and got[t] < per_lang_val // 2 and (t != "find_customer" or "postcode" in r["target"]["arguments"]):
                got[t] += 1
                rows.append({**r, "src": "val", "data": "val", "path": m["path"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glex-plan", required=True)
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--glex-verify", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--val-teacher", required=True)
    ap.add_argument("--val-plans", nargs="+", required=True)
    ap.add_argument("--teacher", required=True, help="teacher.json with prompt / opener (as glex_probe)")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conds", default="audio,neutral")
    ap.add_argument("--src", default=None, help="keep only this source (e.g. glex_dev)")
    ap.add_argument("--soft-mem-mib", type=int, default=12000)
    ap.add_argument("--budget-s", type=float, default=300)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    gplan = [r if isinstance(r["slots"], dict) else {**r, "slots": eval(r["slots"])}
             for r in map(json.loads, open(a.glex_plan))]
    gman = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.glex_data) / "manifest.jsonl"))}
    gver = json.load(open(a.glex_verify))["rows"]
    vman = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.val_data) / "manifest.jsonl"))}
    vte = json.load(open(a.val_teacher))["items"]
    vplan = [r for p in a.val_plans for r in map(json.loads, open(p))]
    rows = select_clips(gplan, gman, gver, vman, vte, vplan)
    for r in rows:
        r["cands"] = [{"slot": None, "kind": "gold", "value": None, "target": r["target"]}] + distractors(r["target"], r["slots"])
    for i, r in enumerate(rows):                              # swap partner: next clip, same language and tool
        r["swap_with"] = None
        if i % 3 == 0:
            j = next((j for j in range(len(rows)) if j != i and rows[j]["lang"] == r["lang"]
                      and rows[j]["target"]["name"] == r["target"]["name"] and rows[j]["id"] != r["id"]), None)
            r["swap_with"] = j

    if a.cpu_check:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        enc = lambda t: [x for x in tok.encode(t, add_special_tokens=False)]          # noqa: E731
        prep = json.load(open(a.teacher))
        opener = next(iter(prep["items"].values()))["main_text"][:prep["opener_end"] + 1]
        cs = prep["opener_end"] + 1 + 12
        chk = {"clips": len(rows), "by_src_lang": {}, "cands": 0, "one_slot_changed": 0, "same_tool_keys": 0,
               "prefix_equal": 0, "first_diff_inside_slot": 0, "leak_in_context": 0, "fc_positions_ok": 0,
               "examples": []}
        for r in rows:
            chk["by_src_lang"][f"{r['src']}|{r['lang']}"] = chk["by_src_lang"].get(f"{r['src']}|{r['lang']}", 0) + 1
            g = r["target"]
            gt = enc(rule_call_text(g))
            x_end = cs + int(math.ceil(r.get("dur", 6.0) * 16000 / FRAME)) if "dur" in r else cs + 80
            for c in r["cands"][1:]:
                chk["cands"] += 1
                t = c["target"]
                diff = [k for k in set(g["arguments"]) | set(t["arguments"]) if g["arguments"].get(k) != t["arguments"].get(k)]
                chk["one_slot_changed"] += len(diff) == 1 and diff[0] == c["slot"]
                chk["same_tool_keys"] += t["name"] == g["name"] and set(t["arguments"]) == set(g["arguments"])
                ct = enc(rule_call_text(t))
                fd = next((k for k, (p, q) in enumerate(zip(gt, ct)) if p != q), min(len(gt), len(ct)))
                pre_text = tok.decode(gt[:fd])
                slot_start = rule_call_text(g).find(json.dumps(g["arguments"][c["slot"]], ensure_ascii=False)[1:-1])
                chk["prefix_equal"] += gt[:fd] == ct[:fd]
                chk["first_diff_inside_slot"] += len(pre_text) >= slot_start
                rec = synthetic_record(opener, cs, x_end, ct + [EOTC])
                st = student_targets(rec, cs, x_end, x_end, x_end + 40, False)
                fc_idx = [j for j, k in enumerate(st["kind"]) if k == "fc"]
                chk["fc_positions_ok"] += [st["func_lbl"][j] for j in fc_idx] == ct + [EOTC] and \
                    all(st["func_in"][j] == st["func_lbl"][j - 1] for j in fc_idx)
                ctx = [x for j, x in enumerate(st["agent_in"]) if st["kind"][j] != "fc"]
                slot_ids = set(enc(json.dumps(t["arguments"][c["slot"]], ensure_ascii=False)[1:-1])) - set(opener)
                chk["leak_in_context"] += any(x in slot_ids for x in ctx if x not in (PAD, BOS))
                if len(chk["examples"]) < 12:
                    chk["examples"].append({"clip": f"{r['id']}|{r['lang']}", "src": r["src"], "slot": c["slot"],
                                            "kind": c["kind"], "gold": rule_call_text(g), "cand": rule_call_text(t),
                                            "len_gold": len(gt), "len_cand": len(ct), "first_diff_token": fd})
        (out / "candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        (out / "cpu_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in chk.items() if k != "examples"}, indent=1))
        for e in chk["examples"][:8]:
            print(e["clip"], e["src"], e["slot"], e["kind"], "|", e["cand"][38:110], "| len", e["len_gold"], e["len_cand"], "fd", e["first_diff_token"])
        return
    if a.src:
        rows = [r for r in rows if r["src"] == a.src]
    run_gpu(a, rows, out, t0)


def clip_order(rows: list[dict]) -> list[int]:
    """Pre-registered processing order (run 2): find_customer clips first (they are the only name / postcode cases),
    then round-robin over languages, so a budget stop truncates every language symmetrically."""
    fc = [i for i, r in enumerate(rows) if r["target"]["name"] == "find_customer"]
    rest = {lg: [i for i, r in enumerate(rows) if r["target"]["name"] != "find_customer" and r["lang"] == lg]
            for lg in ("en", "it", "fr")}
    out = sorted(fc, key=lambda i: ("en", "it", "fr").index(rows[i]["lang"]))
    for k in range(max(map(len, rest.values()), default=0)):
        out += [rest[lg][k] for lg in ("en", "it", "fr") if k < len(rest[lg])]
    return out


def mem_avail_mib() -> int:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) // 1024
    return -1


def run_gpu(a, rows, out, t0):  # pragma: no cover - GPU
    """Run 2 (owner GO 2026-11-13): batch 1 everywhere, both arms on the same clip before moving on (paired by
    construction), one JSON line appended + fsync'd per (clip, arm), soft memory stop before the job-level mem_guard."""
    import os
    import soundfile as sf
    import torch
    from adapters import load_adapter
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    assert a.batch == 1, "run 2 is batch 1 only"
    conds = a.conds.split(",")
    prep = json.loads(Path(a.teacher).read_text())
    opener = next(iter(prep["items"].values()))["main_text"][:prep["opener_end"] + 1]
    cs = prep["opener_end"] + 1 + 12
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prep["prompt"])
    tok = D.w.tokenizer
    print("engine", round(time.monotonic() - t0, 1), "mem_avail", mem_avail_mib(), flush=True)

    def enc(t):
        ids = list(tok.text_to_ids(t))
        return ids[1:] if ids and ids[0] == BOS else ids
    se = SpeechEncoder("ml")
    prm = dict(se.enc.named_parameters())
    orig = {k: v.detach().clone() for k, v in prm.items()}
    arms = []
    for spec in a.arm:
        name, sp = spec.split("=", 1)
        cfg = parse_spec(sp)
        arms.append((name, sp, torch.load(cfg["ln"], map_location="cpu", weights_only=False),
                     load_adapter(cfg["adapter"]).to(D.dev).float()))
    toks = {(i, c): enc(rule_call_text(cd["target"])) + [EOTC] for i, r in enumerate(rows) for c, cd in enumerate(r["cands"])}

    def set_arm(st):
        with torch.no_grad():
            for k, v in orig.items():
                prm[k].copy_(v)
            for k, v in st.items():
                prm[k].copy_(v.to(prm[k].device, prm[k].dtype))

    def score_one(ad, f, x_end, cond, tk):
        """Batch 1: one sequence, one forward; log-softmax only at the call rows."""
        rec = synthetic_record(opener, cs, x_end, tk)
        stt = student_targets(rec, cs, x_end, x_end, f.shape[0], False)
        sil = f[2]
        src = torch.stack([(sil if (cond == "neutral" and cs <= p[1] < x_end) else f[p[1]]) if p[0] == "a" else sil
                           for p in stt["pos"]])
        fc_idx = [j for j, kd in enumerate(stt["kind"]) if kd == "fc"]
        with torch.no_grad():
            audio = D.proj(ad(src[None]).to(D.proj.weight.dtype))
            AI = torch.tensor([stt["agent_in"]], device=D.dev)
            FI = torch.tensor([stt["func_in"]], device=D.dev)
            e = D.stt.embed_tokens
            fr = D.fusion(agent_text_embeds=e(AI).to(D.dtype), user_audio_embeds=audio.to(D.dtype),
                          user_text_embeds=None, function_embeds=e(FI).to(D.dtype))
            o = D.stt(torch.cat([D.prompt_emb, fr], 1), cache=None)
            P = D.prompt_emb.shape[1]
            lf = torch.log_softmax(o["function_logits"][0, [P + j for j in fc_idx]].float(), -1)
            v = [float(lf[q, t]) for q, t in enumerate(tk)]
            del o, lf, fr, audio
        return v, len(stt["pos"])

    jl = out / "cand_run2.jsonl"
    done = set()
    if jl.exists():                                  # resume-safe: never redo or overwrite a saved (clip, arm)
        for ln in open(jl):
            d = json.loads(ln)
            if "clip" in d:
                done.add((d["clip"], d["arm"]))

    def emit(d):
        with open(jl, "a") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    order = clip_order(rows)
    emit({"meta": {"cs": cs, "conds": conds, "arms": [(n, sp) for n, sp, _, _ in arms], "order": order,
                   "batch": 1, "t": round(time.monotonic() - t0, 1), "mem_avail_mib": mem_avail_mib()}})
    torch.cuda.reset_peak_memory_stats()
    # SMOKE (only what the new batching needs): first clip, first arm, gold/audio scored twice at batch 1.
    i0 = order[0]
    n0, _, st0, ad0 = arms[0]
    set_arm(st0)
    x0 = sf.read(Path(a.glex_data if rows[i0]["data"] == "glex" else a.val_data) / rows[i0]["path"],
                                    dtype="float32")[0]
    with torch.no_grad():
        f0 = se(torch.from_numpy(session_wav(x0, cs, 3.0)))["out"].to(D.dev).float()
    xe0 = cs + int(math.ceil(len(x0) / FRAME))
    l1, ln1 = score_one(ad0, f0, xe0, "audio", toks[(i0, 0)])
    l2, _ = score_one(ad0, f0, xe0, "audio", toks[(i0, 0)])
    smoke = {"clip": i0, "determinism_max_abs": round(max(abs(x - y) for x, y in zip(l1, l2)), 5),
             "n_fc_scored": len(l1), "n_fc_tokens": len(toks[(i0, 0)]), "seq_len": ln1,
             "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
             "mem_avail_mib": mem_avail_mib(), "t": round(time.monotonic() - t0, 1)}
    smoke["pass"] = smoke["determinism_max_abs"] < 0.05 and len(l1) == len(toks[(i0, 0)]) and \
        smoke["mem_avail_mib"] > a.soft_mem_mib
    emit({"smoke": smoke})
    print("SMOKE", json.dumps(smoke), flush=True)
    if not smoke["pass"]:
        raise SystemExit("smoke failed: stop")
    rate = None                                       # worst seconds per candidate (both arms, all conds)
    for n_done, i in enumerate(order):
        r = rows[i]
        el = time.monotonic() - t0
        if rate is not None and el + 1.15 * rate * len(r["cands"]) > a.budget_s:
            emit({"stop": "budget", "t": round(el, 1), "clips_done": n_done})
            print("stop: budget", round(el, 1), flush=True)
            break
        x = sf.read(Path(a.glex_data if r["data"] == "glex" else a.val_data) / r["path"], dtype="float32")[0]
        x_end = cs + int(math.ceil(len(x) / FRAME))
        tp = time.monotonic()
        for name, sp, st, ad in arms:
            if (i, name) in done:
                continue
            set_arm(st)
            with torch.no_grad():
                f = se(torch.from_numpy(session_wav(x, cs, 3.0)))["out"].to(D.dev).float()
            lp = {}
            for cond in conds:
                for c in range(len(r["cands"])):
                    m = mem_avail_mib()
                    if m < a.soft_mem_mib:
                        emit({"stop": "soft_mem", "mem_avail_mib": m, "clip": i, "arm": name})
                        raise SystemExit(f"soft memory stop: MemAvailable {m} MiB")
                    lp.setdefault(cond, {})[str(c)] = score_one(ad, f, x_end, cond, toks[(i, c)])[0]
            emit({"clip": i, "arm": name, "id": r["id"], "lang": r["lang"], "src": r["src"], "logp": lp,
                  "toks": {str(c): toks[(i, c)] for c in range(len(r["cands"]))},
                  "cands": [{k: cd[k] for k in ("slot", "kind", "value")} for cd in r["cands"]],
                  "t": round(time.monotonic() - t0, 1), "mem_avail_mib": mem_avail_mib(),
                  "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20)})
            del f
        dt = time.monotonic() - tp
        rate = max(rate or 0.0, dt / len(r["cands"]))
        print("clip", n_done, i, r["lang"], len(r["cands"]), round(dt, 1), round(time.monotonic() - t0, 1),
              mem_avail_mib(), flush=True)
    print("done", round(time.monotonic() - t0, 1), flush=True)


AD = None

if __name__ == "__main__":
    main()
