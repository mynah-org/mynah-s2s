"""Opener probe (official VoiceChat VC arm; inference only). Owner steer 2026-11-13: locate the cause of the 2 opener
failures of job 349 (val s4-0124, s4-0162), not obtain a gate PASS.

Known (CPU audit, l0-thinker-lora-card.md section 16): both failures are the official model's alternative opener
("I can only help with ..." instead of "I can help you with ..."), which diverges at opener token 15 (frame 68),
about 2.5 s before the caller starts (cs = 100). The VC encoder is causal and the lead features of different clips are
identical to 3e-7 on CPU. The question left: is the variant chosen by (a) GPU numerics of the inputs (length-dependent
encoder output on GPU), (b) batch composition, or (c) run-to-run non-determinism (engine built with
s2s.deterministic False) at a near-tie?
Design: only the first --frames frames are decoded (the opener region; the caller never starts before cs).
  step 0  GPU input identity: max |diff| of the encoder output and of the bf16 Thinker input over frames < cs, all clips
  run A   job 349 batch 3 composition (8 val EN clips, same order), twice            -> composition + determinism
  run B   batch 1, the 2 failed + 2 passed clips                                     -> batch-size effect
  run C   batch 1, the 2 failed clips again                                          -> run-to-run at batch 1
Per run and clip: opener variant, frame of the first differing token vs the reference opener, and the top-3 text
log-probs at every frame (margin between the two variants' tokens at the divergence frame). One fsync'd JSON line per run.
Usage: opener_probe.py --sets free_sets.json --val-data <root> --teacher teacher.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labpaths import resolve  # noqa: E402  (archived clips: absolute symlinks into the historical layout)
from session_distill import PAD, session_wav  # noqa: E402

BATCH3 = ["s4-0045", "s4-0107", "s4-0108", "s4-0124", "s4-0130", "s4-0158", "s4-0162", "s4-0166"]
FAILED, PASSED = ["s4-0124", "s4-0162"], ["s4-0108", "s4-0130"]
DIV_INDEX, ALT_TOKEN = 15, 2342      # CPU audit: the variant replaces opener token 15 (3508) by 2342 (150/150 teacher cases)


def opener_diff(text_tokens: list[int], ref_tokens: list[int], cs: int) -> dict:
    """text_tokens: per-frame agent tokens; ref_tokens: per-frame reference opener. First non-PAD token that differs,
    with its index in the non-PAD sequence and its frame in each stream."""
    own = [(t, x) for t, x in enumerate(text_tokens[:cs]) if x != PAD]
    ref = [(t, x) for t, x in enumerate(ref_tokens) if x != PAD]
    for k, (a, b) in enumerate(zip(own, ref)):
        if a[1] != b[1]:
            return {"exact": False, "index": k, "frame_own": a[0], "frame_ref": b[0], "tok_own": a[1], "tok_ref": b[1]}
    if len(own) != len(ref):
        k = min(len(own), len(ref))
        return {"exact": False, "index": k, "frame_own": own[k][0] if k < len(own) else None,
                "frame_ref": ref[k][0] if k < len(ref) else None, "tok_own": None, "tok_ref": None}
    timing = [t for t, _ in own] == [t for t, _ in ref]
    return {"exact": True, "same_frames": timing}


def margin(topk_row, tok_a: int, tok_b: int):
    """log p(a) - log p(b) from a greedy_fc top-k row ([ids], [logps]); None when one is outside the top-k."""
    ids, lps = topk_row
    if tok_a in ids and tok_b in ids:
        return round(lps[ids.index(tok_a)] - lps[ids.index(tok_b)], 4)
    return None


def main():  # pragma: no cover - GPU
    import os

    import soundfile as sf
    import torch
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    from session_distill import greedy_fc
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--budget-s", type=float, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "opener_probe.jsonl"
    if jl.exists():
        raise SystemExit(f"{jl} exists")
    t0 = time.monotonic()

    def emit(d):
        with open(jl, "a") as fh:
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    rows = {r["id"]: r for r in json.load(open(a.sets))["rows"] if r["set"] == "val" and r["lang"] == "en"}
    prep = json.loads(Path(a.teacher).read_text())
    ref = next(iter(prep["items"].values()))["main_text"][:prep["opener_end"] + 1]
    cs = prep["opener_end"] + 1 + 12
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prep["prompt"])
    se = SpeechEncoder("vc")
    sil = D.w._silence_embedding_cache.reshape(-1).to(D.dev)
    emit({"meta": {"cs": cs, "frames": a.frames, "t_load": round(time.monotonic() - t0, 1),
                   "precision": {"s2s.deterministic (load_engine override)": False,
                                 "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                 "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cuda.matmul.allow_bf16_reduced_precision_reduction":
                                     torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                 "cudnn.deterministic": torch.backends.cudnn.deterministic,
                                 "cudnn.benchmark": torch.backends.cudnn.benchmark,
                                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                 "thinker_dtype": str(D.dtype), "ssm_cache_dtype": "float32"}}})
    # step S: decoder state after the prompt prefill (fresh cache, as greedy_fc), row by row
    from mynah_v2.model.llm_cache import _hybrid_cache_cls

    def prefill(B):
        c = _hybrid_cache_cls(D.stt.llm)(D.stt.llm.config, B, dtype=D.dtype, device=D.dev, ssm_dtype=torch.float32)
        with torch.no_grad():
            D.stt(D.prompt_emb.expand(B, -1, -1).contiguous(), cache=c)
        return c

    def state_diff(c1, r1, c2, r2):
        m = {}
        for name in ("ssm_states", "conv_states", "key_cache", "value_cache"):
            vals = [float((x[r1].float() - y[r2].float()).abs().max()) for x, y in zip(getattr(c1, name), getattr(c2, name))
                    if x.numel() and y.numel()]
            m[name] = max(vals) if vals else None
        return m
    c8a, c8b, c1 = prefill(8), prefill(8), prefill(1)
    emit({"stepS_prefill_state": {"within_batch8_row_k_vs_row0": {k: state_diff(c8a, k, c8a, 0) for k in range(1, 8)},
                                  "batch8_rows_vs_batch1": {k: state_diff(c8a, k, c1, 0) for k in (0, 3, 6)},
                                  "batch8_run_a_vs_run_b": {k: state_diff(c8a, k, c8b, k) for k in (0, 3, 6)},
                                  "note": "initial cache = zeros by construction (fresh per greedy_fc call)"},
          "t": round(time.monotonic() - t0, 1)})
    del c8a, c8b, c1
    ref_nz = [(t, x) for t, x in enumerate(ref) if x != PAD]
    div_frame, ref_tok = ref_nz[DIV_INDEX]
    enc, emb = {}, {}
    for i in BATCH3:
        x = sf.read(resolve(Path(a.val_data) / rows[i]["path"]), dtype="float32")[0]
        with torch.no_grad():
            f = se(torch.from_numpy(session_wav(x, cs, 5.0)))["out"]
            enc[i] = f[:cs].float()
            emb[i] = D.proj(f.to(D.dev).to(D.proj.weight.dtype))[: a.frames].to(D.dtype)
    r0 = BATCH3[2]
    emit({"step0": {i: {"enc_max_abs_vs_" + r0: float((enc[i] - enc[r0]).abs().max()),
                        "thinker_in_bf16_elems_diff_vs_" + r0: int((emb[i][:cs] != emb[r0][:cs]).sum()),
                        "thinker_in_first_diff_frame": next((t for t in range(min(cs, a.frames))
                                                             if bool((emb[i][t] != emb[r0][t]).any())), None)}
                    for i in BATCH3}, "t": round(time.monotonic() - t0, 1)})

    def run(tag, ids):
        if time.monotonic() - t0 > a.budget_s:
            emit({"stop": "budget", "run": tag})
            return
        A = torch.stack([emb[i] for i in ids])
        recs = greedy_fc(D, A, sil, [a.frames] * len(ids), topk=3)
        res = {}
        for i, rec in zip(ids, recs):
            d = opener_diff(rec["main_text"], ref, cs)
            top = rec["topk"][div_frame]
            row = {"variant": "ref" if d["exact"] else "alt", "diff": d, "batch_pos": ids.index(i), "batch_size": len(ids),
                   "div_frame": div_frame, "margin_ref_minus_alt": margin(top, ref_tok, ALT_TOKEN),
                   "top1_minus_top2": round(top[1][0] - top[1][1], 4), "top3_at_div": top,
                   "topk": {t: rec["topk"][t] for t in range(55, min(80, len(rec["topk"])))}}
            if not d["exact"] and d.get("tok_own") is not None:
                f = d["frame_own"]
                row["margin_own_minus_ref_at_div"] = margin(rec["topk"][f], d["tok_own"], d["tok_ref"])
            res[i] = row
        emit({"run": tag, "ids": ids, "res": res, "t": round(time.monotonic() - t0, 1)})
        print(tag, {i: v["variant"] for i, v in res.items()}, round(time.monotonic() - t0, 1), flush=True)
    run("A1_batch3", BATCH3)
    run("A2_batch3", BATCH3)
    for i in FAILED + PASSED:
        run(f"B_b1_{i}", [i])
    for i in FAILED:
        run(f"C_b1_{i}", [i])
    print("done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
