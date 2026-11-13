"""Thinker precision sensitivity map over the fixed corpus (teacher-forced, mynah_v2.model.precision.SensitivityProbe).

The BF16 golden runs exactly as the reference (deterministic, cache ON, real tools from the corpus SQLite rows, the
NVIDIA FC fast path); every candidate (format × layer-group selection) shadows it on the same inputs with its own
Mamba/KV state. Per candidate, aggregated by corpus category × step kind:
  fc   = steps where the golden function head emits a non-PAD token (tool-call name/arguments region);
  text = golden text head emits a non-PAD token (speech content);
  pad  = both PAD (listening / silence).
Metrics: top-1 agreement (text / function / ASR heads) and KL(golden‖candidate) mean + p95. The first-order
question is causal: which layer groups flip tool-argument and speech tokens, and does restoring them to BF16 fix it.

Writes <out>/sensitivity.json (aggregates + candidate sizes) and <out>/records.jsonl.gz (per step). Refuses an
existing --out. Audio decoding is off (the Thinker trajectory does not depend on it under greedy text decoding).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

PAD = 12


def kind(rec):
    if rec["gold_func"] is not None and rec["gold_func"] != PAD:
        return "fc"
    return "text" if rec["gold_text"] != PAD else "pad"


TURN = {1, 2}          # <s> (turn start) / </s> (EOS) on the text channel


def flip_class(g, c):
    """Decision type of a text-channel disagreement: turn-taking (BOS/EOS involved), speak/silence timing (PAD vs
    a word) or content (two different words)."""
    if g in TURN or c in TURN:
        return "turn"
    if g == PAD or c == PAD:
        return "timing"
    return "content"


def decision_flips(dec, name):
    """Every text-channel step (PAD included) where the teacher-forced candidate's argmax differs from the golden one,
    with the golden margin (top1 - top2 logit) and the candidate's gap for the golden token. Also the golden margin
    distribution at turn-taking steps, to tell 'near-tie decisions flipped' from 'large errors'."""
    flips = []
    for j, r in enumerate(dec):
        x = r["c"][name]
        if not x["text_agree"]:
            flips.append({"step": j, "class": flip_class(r["gold_text"], x["text_top"]), "gold": r["gold_text"],
                          "cand": x["text_top"], "gold_margin": r.get("g_text_margin"),
                          "cand_gold_gap": x.get("text_gold_gap"), "kl": round(x["text_kl"], 4)})
    by = {}
    for f in flips:
        by[f["class"]] = by.get(f["class"], 0) + 1
    turn_m = sorted(r.get("g_text_margin") for r in dec if r["gold_text"] in TURN and r.get("g_text_margin") is not None)
    return {"all_text_flips": len(flips), "flips_by_class": by, "flip_list": flips[:40],
            "turn_steps": len(turn_m), "turn_gold_margin_min": turn_m[0] if turn_m else None,
            "turn_gold_margins": turn_m[:20]}


def aggregate(records, cands, cat_of):
    out = {}
    for c in cands:
        groups = {}
        for r in records:
            if r["t"] != 1:            # prompt prefill: not a decode step
                continue
            for key in (f'{cat_of[r["tag"]]}/{kind(r)}', f"ALL/{kind(r)}"):
                groups.setdefault(key, []).append(r["c"][c])
        agg = {}
        for key, rs in sorted(groups.items()):
            d = {"n": len(rs), "text_agree": round(float(np.mean([x["text_agree"] for x in rs])), 4),
                 "text_kl_mean": round(float(np.mean([x["text_kl"] for x in rs])), 5),
                 "text_kl_p95": round(float(np.percentile([x["text_kl"] for x in rs], 95)), 5)}
            if "func_agree" in rs[0]:
                d["func_agree"] = round(float(np.mean([x["func_agree"] for x in rs])), 4)
                d["func_kl_mean"] = round(float(np.mean([x["func_kl"] for x in rs])), 5)
            if "asr_agree" in rs[0]:
                d["asr_agree"] = round(float(np.mean([x["asr_agree"] for x in rs])), 4)
            agg[key] = d
        out[c] = agg
    return out


def main():
    # engine imports here: the analysis helpers above stay importable without the NeMo stack (CPU tests)
    from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolRegistry, make_orders_db, orders_tool
    from mynah_v2.model.precision import SensitivityProbe
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import iter_chunks
    from run_stream import load_16k, render_prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--cand", action="append", required=True, help="name=fmt@selector[!kernel] (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=None, help="comma list of item ids (default: whole corpus)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cdir = Path(a.corpus_dir)
    db = make_orders_db(out / "orders.sqlite", rows=[tuple(r) for r in json.loads((cdir / "orders.json").read_text())])
    reg = ToolRegistry()
    reg.register("get_order_status", orders_tool(db), "Get the status, delivery date and tracking code of a customer "
                 "order by its order id.", ORDERS_SCHEMA)
    import offline_voicechat_fc_infer as fcmod
    reg.schemas += fcmod.DEFAULT_TOOLS
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"),
                                    overrides={"s2s.deterministic": True, "s2s.decode_audio": False}),
                          tools={"get_order_status": orders_tool(db)})
    prompt = render_prompt(reg.schemas)
    t0 = time.monotonic()
    probe = SensitivityProbe(eng, a.cand)
    build_s = time.monotonic() - t0
    sizes = {c.name: {"fmt": c.fmt, "selector": c.selector, "kernel": c.kernel, "linears": len(c.swaps),
                      "bf16_GB": round(c.bytes_bf16 / 2**30, 2)} for c in probe.cands}
    print(json.dumps({"candidates": sizes, "build_s": round(build_s, 1),
                      "alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 1)}), flush=True)
    items = sorted(p for p in cdir.glob("*.json") if p.name not in ("orders.json", "manifest.json"))
    if a.items:
        keep = set(a.items.split(","))
        items = [p for p in items if p.stem in keep]
    cat_of, per_item = {}, {}
    torch.cuda.reset_peak_memory_stats()
    for side in items:
        meta = json.loads(side.read_text())
        cat_of[meta["id"]] = meta["category"]
        x = load_16k(str(side.with_suffix(".wav")))
        torch.manual_seed(a.seed)
        probe.reset_session()
        probe.tag = meta["id"]
        n0 = len(probe.records)
        ts = time.monotonic()
        eng.open(prompt)
        chunks = list(iter_chunks(x, eng.chunk_samples))
        dur = eng.chunk_samples / 16000
        fc_chunks = 0
        for c in chunks:
            # the reference FC fast path runs in a background thread on WALL time while generate_step returns at
            # once: pace like a live caller while it is active, never feed it unpaced (session.drain_fc docstring)
            if eng.fc_active():
                time.sleep(dur)
                fc_chunks += 1
            eng.step(c)
        extra = 0
        while eng.fc_active() and extra < int(60 / dur):      # call still running at the end: wait it out, paced
            time.sleep(dur)
            eng.step(torch.zeros(eng.chunk_samples))
            extra += 1
        eng.close()
        eng.release_fc()
        recs = probe.records[n0:]
        dec = [r for r in recs if r["t"] == 1]
        first = {}
        for c in probe.cands:
            idx = [j for j, r in enumerate(dec) if kind(r) != "pad" and not r["c"][c.name]["text_agree"]]
            fidx = [j for j, r in enumerate(dec) if kind(r) == "fc" and not r["c"][c.name].get("func_agree", True)]
            first[c.name] = {"text_flips": len(idx), "first_text_flip": idx[0] if idx else None,
                             "func_flips": len(fidx), "first_func_flip": fidx[0] if fidx else None,
                             **decision_flips(dec, c.name)}
        per_item[meta["id"]] = {"category": meta["category"], "steps": len(dec), "fc_paced_chunks": fc_chunks,
                                "drain_chunks": extra, "fc_calls": len(eng.fc_records), "fc_steps": sum(kind(r) == "fc" for r in dec),
                                "text_steps": sum(kind(r) == "text" for r in dec), "s": round(time.monotonic() - ts, 1),
                                "flips": first}
        print(json.dumps({"item": meta["id"], **{k: v for k, v in per_item[meta["id"]].items() if k != "flips"},
                          "func_flips": {k: v["func_flips"] for k, v in first.items()},
                          "text_flips": {k: v["text_flips"] for k, v in first.items()},
                          "decision_flips": {k: v["flips_by_class"] for k, v in first.items()}}), flush=True)
    probe.remove()
    names = [c.name for c in probe.cands]
    rep = {"candidates": sizes, "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "aggregate": aggregate(probe.records, names, cat_of), "items": per_item,
           "doc": "teacher-forced: golden BF16 trajectory; candidates shadow it with their own cache; see precision.py"}
    (out / "sensitivity.json").write_text(json.dumps(rep, indent=1))
    with gzip.open(out / "records.jsonl.gz", "wt") as f:
        for r in probe.records:
            f.write(json.dumps(r) + "\n")
    for c in names:
        al = rep["aggregate"][c]
        print(c, json.dumps({k: al[k] for k in al if k.startswith("ALL/")}), flush=True)


if __name__ == "__main__":
    main()
