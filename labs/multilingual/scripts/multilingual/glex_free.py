"""Free-running G-LEX evaluation (L1 primary metric; owner steer 2026-11-13). Inference only, no gradient.

Unlike glex_probe (teacher-forced, gold prefix, full-vocab argmax per token), the model here generates everything itself:
the opener on the lead-in, listening during the caller turn, the SOTC decision and the call tokens. Decoder:
session_distill.greedy_fc, the same batched duplex greedy decode that produced the D-V3 teachers.

Context, checked against the production VoiceChat path (l0-thinker-lora-card.md section 12):
  - system prompt: teacher.json prompt = agentbench render_prompt (official render_fc_system_prompt + DEFAULT_TEMPLATE),
    embedded by the wrapper's own _prepare_system_prompt_embeddings;
  - greedy argmax (s2s_streaming.yaml temperature 0); stt in eval mode, so the config inference boosts apply as live;
  - FC fast path: after SOTC, agent PAD, the silence embedding and the previous function token until EOTC
    (fc_async_use_real_audio false); fc_sotc_boost 0 (unset);
  - silence embedding: the FIRST frame of the perception output on 1 s of zeros, as the wrapper's
    _create_silence_embedding_cache; for ML arms through the arm's encoder + adapter + VC proj;
  - not modelled: the tool response and resume after EOTC (G-LEX ends at the call), threading and real-time pacing.
Lead-in: cs = opener_end + 1 + lead_extra frames of digital silence (as glex_probe / D-V3); tail --tail-s seconds.
Arms: SPEC as glex_probe (enc=vc | enc=ml;ln=..;adapter=..) plus optional lora=<lora state .pt> (thinker_lora) and
sil=first|last: the FC silence embedding = FIRST frame of the encoded 1 s of zeros (production, wrapper:1049; for enc=vc
the wrapper's own cache tensor) or the LAST frame (training rule, SessionML.silence). --arm-sets / --arm-langs restrict
an arm to some sets / languages; rows keep the sets-file order (glex_dev, val, trainfit).
Results: one JSON line per clip (fsync'd), so a stop keeps every finished batch.
Usage: glex_free.py --sets sets.json --teacher teacher.json --arm NAME=SPEC ... --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from session_distill import BOS, EOTC, FRAME, PAD, SOTC, TOL, rule_call_text, session_wav  # noqa: E402

CALL_RE = re.compile(r"<TOOLCALL>(.*)</TOOLCALL>", re.S)


def parse_call(text: str):
    """[calls] from a generated call text, or None if it does not parse."""
    m = CALL_RE.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else [v]


def norm(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def score_call(text: str, target: dict) -> dict:
    """exact = one call, same tool, same argument keys and values (strict string equality); per-slot strict and
    case/space-normalised matches; exact_text = byte-identical canonical text."""
    calls = parse_call(text)
    out = {"parsed": calls is not None, "exact_text": text == rule_call_text(target)}
    if not calls:
        return {**out, "tool": None, "exact": False, "n_calls": 0,
                "slots": {k: {"strict": False, "norm": False, "got": None} for k in (target.get("arguments") or {})}}
    c = calls[0] if isinstance(calls[0], dict) else {}
    args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
    gold = target.get("arguments") or {}
    slots = {k: {"strict": args.get(k) == v, "norm": norm(args.get(k)) == norm(v), "got": args.get(k)} for k, v in gold.items()}
    exact = len(calls) == 1 and c.get("name") == target["name"] and set(args) == set(gold) and all(s["strict"] for s in slots.values())
    return {**out, "tool": c.get("name"), "exact": exact, "n_calls": len(calls), "extra_keys": sorted(set(args) - set(gold)),
            "slots": slots}


def concept_preserved(query: str | None, row: dict) -> bool | None:
    """Diagnostic only (owner 2026-11-13; not a gate, G-LEX exact and the EN canonical query are unchanged): the
    generated query keeps the spoken concept, either translated (the EN concept noun) or in the source language (the
    IT/FR surface form or a listed synonym). A generic query ('item', 'furniture') or another product does not count."""
    if row["target"]["name"] != "search_catalog":
        return None
    if not query:
        return False
    import lexicon_v2 as L
    c = next((x for x in L.CONCEPTS if x[0] == (row.get("slots") or {}).get("concept")), None)
    if c is None:
        return None
    q = norm(query)
    forms = [c[0], c[2][0], c[4][0], *c[3], *c[5]]
    syn = (row.get("slots") or {}).get("synonym") or {}
    forms += [v for v in syn.values() if v]
    return any(norm(f) in q for f in forms if f)


def behaviour(rec: dict, cs: int, x_end: int, opener: list[int], valid_openers: list[list[int]] | None = None) -> dict:
    """Opener, listening and timing from a greedy_fc record (frame-indexed main_text / main_func)."""
    tx, fn = rec["main_text"], rec["main_func"]
    own = [t for t in tx[:cs] if t != PAD]
    ref = [t for t in opener if t != PAD]
    turn = range(cs + 2, max(cs + 2, x_end - TOL))
    sotc = next((t for t, f in enumerate(fn) if f == SOTC), None)
    diff = next((k for k, (x, y) in enumerate(zip(own, ref)) if x != y), None if len(own) == len(ref) else min(len(own), len(ref)))
    nz = [(t, x) for t, x in enumerate(tx[:cs]) if x != PAD]
    return {"opener_exact": own == ref, "opener_prefix_ok": own[:len(ref)] == ref,
            "opener_valid": own == ref or any(own == v for v in (valid_openers or [])), "opener_first_diff": diff,
            "opener_len": [len(own), len(ref)], "opener_tail_frames": nz[-4:],
            "bos_in_turn": sum(1 for t in turn if t < len(tx) and tx[t] == BOS),
            "talk_in_turn": sum(1 for t in turn if t < len(tx) and tx[t] not in (PAD, BOS)),
            "sotc_frame": sotc, "sotc_rel_end": None if sotc is None else sotc - x_end,
            "sotc_early": sotc is not None and sotc < x_end - TOL, "sotc_before_cs": sotc is not None and sotc < cs,
            "fc_len": len(rec["fc"]), "fc_terminated": bool(rec["fc"]) and rec["fc"][-1] == EOTC}


def summarise(rows: list[dict]) -> dict:
    out = {}
    keys = sorted({(r["set"], r["lang"]) for r in rows})
    for k in [("all", "all")] + [(s, "all") for s in sorted({r["set"] for r in rows})] + keys:
        sel = [r for r in rows if (k[0] == "all" or r["set"] == k[0]) and (k[1] == "all" or r["lang"] == k[1])]
        if not sel:
            continue
        n = len(sel)
        slots = {}
        for r in sel:
            for s, v in r["score"]["slots"].items():
                slots.setdefault(s, [0, 0, 0])
                slots[s][0] += v["strict"]
                slots[s][1] += v["norm"]
                slots[s][2] += 1
        out[f"{k[0]}|{k[1]}"] = {
            "exact": f"{sum(r['score']['exact'] for r in sel)}/{n}",
            "right_tool": f"{sum(r['score']['tool'] == r['target']['name'] for r in sel)}/{n}",
            "no_call": f"{sum(r['beh']['sotc_frame'] is None for r in sel)}/{n}",
            "sotc_early": f"{sum(r['beh']['sotc_early'] for r in sel)}/{n}",
            "opener_exact": f"{sum(r['beh']['opener_exact'] for r in sel)}/{n}",
            "opener_valid": f"{sum(r['beh'].get('opener_valid', r['beh']['opener_exact']) for r in sel)}/{n}",
            "bos_in_turn_clips": f"{sum(r['beh']['bos_in_turn'] > 0 for r in sel)}/{n}",
            "slots_strict": {s: f"{v[0]}/{v[2]}" for s, v in slots.items()},
            "slots_norm": {s: f"{v[1]}/{v[2]}" for s, v in slots.items()}}
    return out


def build_sets(glex_plan, glex_man, val_plan_rows, val_man, val_teacher, per_lang_val=12, per_lang_train=8):
    """glex_dev: every G-LEX-dev clip with audio (as glex_probe P1/P2). val / trainfit: call clips of the train_v1 corpus
    (split val / train), ASR ok, not excluded, teacher-kept, sorted by id, the first N per language."""
    rows = []
    for r in glex_plan:
        k = (r["id"], r["lang"])
        if k in glex_man and r.get("target"):
            rows.append({"set": "glex_dev", "id": r["id"], "lang": r["lang"], "target": r["target"], "data": "glex",
                         "path": glex_man[k]["path"], "slots": r.get("slots")})
    for split, name, cap in (("val", "val", per_lang_val), ("train", "trainfit", per_lang_train)):
        for lang in ("en", "it", "fr"):
            got = 0
            for r in sorted(val_plan_rows, key=lambda x: x["id"]):
                m = val_man.get((r["id"], r["lang"]))
                if got >= cap or r["lang"] != lang or not m or m["split"] != split or not m.get("asr_ok", True) \
                        or m.get("excluded") or not r.get("target") or not r["target"].get("arguments") \
                        or not val_teacher.get(r["id"], {}).get("keep"):
                    continue
                got += 1
                rows.append({"set": name, "id": r["id"], "lang": lang, "target": r["target"], "data": "val",
                             "path": m["path"], "slots": r.get("slots")})
    return rows


def main():  # pragma: no cover - GPU
    import os

    import soundfile as sf
    import torch
    from adapters import load_adapter
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    from glex_probe import parse_spec
    from session_distill import greedy_fc
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="json from --build-sets")
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--arm-sets", action="append", default=[], help="NAME=set1,set2 (default: every set)")
    ap.add_argument("--arm-langs", action="append", default=[], help="NAME=en,it (default: every language)")
    ap.add_argument("--lead-extra", type=int, default=12)
    ap.add_argument("--opener-valid", default="", help="json {ref, alt}: the two official opener token sequences")
    ap.add_argument("--user-text", default="", help="V0 json: per-row [[frame, nano_id]] user-text stream")
    ap.add_argument("--arm-rows", action="append", default=[], help="NAME=N: only the first N selected rows (layout)")
    ap.add_argument("--ut-weight", type=float, default=None,
                    help="V1: use OUR OWN AddFusion instance with this user_text_weight (the official wrapper builds 0.0 "
                         "without an asr_head, wrapper:765-771); agent / audio / function weights copied from the wrapper's")
    ap.add_argument("--delta-probe", action="store_true", help="V1: Thinker logit / hidden deltas text vs no text, 1st batch")
    ap.add_argument("--tail-s", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=600)
    ap.add_argument("--soft-mem-mib", type=int, default=12000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "free_rows.jsonl"
    if jl.exists():
        raise SystemExit(f"{jl} exists")
    t0 = time.monotonic()
    rows = json.load(open(a.sets))["rows"]
    prep = json.loads(Path(a.teacher).read_text())
    opener = next(iter(prep["items"].values()))["main_text"][:prep["opener_end"] + 1]
    cs = prep["opener_end"] + 1 + a.lead_extra
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prep["prompt"])
    tok = D.w.tokenizer
    fusion_info = {"wrapper_fusion": type(D.fusion).__name__, "wrapper_user_text_weight": D.fusion.user_text_weight}
    if a.ut_weight is not None:            # our instance; the wrapper's fusion_module object and code stay untouched
        from nemo.collections.speechlm2.parts.fusion import create_fusion_module
        wf = D.fusion
        D.fusion = create_fusion_module(None, D.stt.embed_tokens.weight.shape[1], wf.agent_text_weight,
                                        wf.user_audio_weight, a.ut_weight, wf.function_weight).to(D.dev, D.dtype).eval()
        assert D.w.fusion_module is wf and wf.user_text_weight == fusion_info["wrapper_user_text_weight"]
        fusion_info.update(ours=type(D.fusion).__name__, ours_weights=[D.fusion.agent_text_weight, D.fusion.user_audio_weight,
                                                                        D.fusion.user_text_weight, D.fusion.function_weight])
    valid = []
    if a.opener_valid:
        ov = json.load(open(a.opener_valid))
        assert [x for x in opener if x != PAD] == ov["ref"], "opener_valid ref differs from the teacher opener"
        valid = [ov["alt"]]
    arm_sets = dict(s.split("=", 1) for s in a.arm_sets)
    arm_langs = dict(s.split("=", 1) for s in a.arm_langs)
    arm_rows = {k: int(v) for k, v in (s.split("=", 1) for s in a.arm_rows)}
    ut_stream = json.load(open(a.user_text))["stream"] if a.user_text else {}
    import hashlib
    config = {"prompt_sha256": hashlib.sha256(prep["prompt"].encode()).hexdigest(), "prompt_chars": len(prep["prompt"]),
              "prompt_frames": int(D.prompt_emb.shape[1]), "thinker_dtype": str(D.dtype), "ssm_cache_dtype": "float32",
              "cache": "mynah_v2 _hybrid_cache_cls (V2 patches)", "decoder": "session_distill.greedy_fc argmax, windows None",
              "function_head": list(D.stt.function_head.weight.shape), "stt_training_mode": bool(D.stt.training),
              "boosts": {k: D.stt.cfg.get(k, None) for k in ("inference_pad_boost", "inference_bos_boost",
                                                            "inference_eos_boost", "inference_user_pad_boost")},
              "opener_end": prep["opener_end"], "lead_extra": a.lead_extra, "tail_s": a.tail_s, "batch": a.batch}

    def mem():
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) // 1024
        return -1

    def emit(d):
        with open(jl, "a") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    config["fusion"] = fusion_info
    emit({"meta": {"cs": cs, "config": config, "arms": a.arm, "arm_sets": arm_sets, "arm_langs": arm_langs, "n_rows": len(rows),
                   "mem_avail_mib": mem(), "t": round(time.monotonic() - t0, 1)}})
    print("engine", round(time.monotonic() - t0, 1), mem(), flush=True)
    encs, done_rows, mods = {}, [], None
    for spec in a.arm:
        name, sp = spec.split("=", 1)
        cfg = parse_spec(sp)
        if cfg["enc"] not in encs:
            encs[cfg["enc"]] = SpeechEncoder(cfg["enc"])
        se = encs[cfg["enc"]]
        saved = None
        if cfg.get("ln"):
            st = torch.load(cfg["ln"], map_location="cpu", weights_only=False)
            prm = dict(se.enc.named_parameters())
            saved = {k: prm[k].detach().clone() for k in st}
            with torch.no_grad():
                for k, v in st.items():
                    prm[k].copy_(v.to(prm[k].device, prm[k].dtype))
        ad = load_adapter(cfg["adapter"]).to(D.dev).float() if cfg.get("adapter") else None
        if cfg.get("lora"):                     # one LoRA install per process; later LoRA arms only load their state
            import thinker_lora as TL
            ck = torch.load(cfg["lora"], map_location="cpu", weights_only=False)
            if mods is None:
                mods = TL.install(D.stt.llm, layers=tuple(ck["layers"]), targets=tuple(ck["targets"]), r=ck["r"],
                                  alpha=ck["alpha"])
            TL.load_state(mods, ck["state"])
            TL.set_enabled(mods, True)
            emit({"arm": name, "lora": cfg["lora"], "lora_step": ck.get("step"),
                  "lora_B_norm": round(float(sum(v.float().norm() ** 2 for k, v in ck["state"].items()
                                                 if k.endswith("lora_B")) ** 0.5), 6)})
        elif mods is not None:
            import thinker_lora as TL
            TL.set_enabled(mods, False)

        def to_thinker(f):
            with torch.no_grad():
                y = ad(f.to(D.dev).float()) if ad is not None else f.to(D.dev)
                return D.proj(y.to(D.proj.weight.dtype))
        sil_mode = cfg.get("sil", "first")
        assert sil_mode in ("first", "last"), sil_mode
        with torch.no_grad():
            fs = se(torch.zeros(16000))["out"]
            sil_first = to_thinker(fs[0:1])[0]                               # production rule: FIRST frame
            sil_last = to_thinker(fs[-1:])[0]                                # training rule (SessionML.silence)
            wc = getattr(D.w, "_silence_embedding_cache", None)
            if cfg["enc"] == "vc" and sil_mode == "first" and wc is not None:
                sil_first = wc.reshape(-1).to(D.dev)                         # the production tensor itself
            sil = (sil_first if sil_mode == "first" else sil_last).to(D.dtype)
        cos = lambda u, v: round(float(torch.nn.functional.cosine_similarity(u.float().reshape(1, -1),   # noqa: E731
                                                                              v.float().reshape(1, -1))[0]), 4)
        chk = {"arm": name, "sil_mode": sil_mode, "silence_first_vs_last_cos": cos(sil_first, sil_last),
               "silence_first_vs_last_l2": round(float((sil_first.float() - sil_last.float()).norm()), 4),
               "silence_norm": round(float(sil.float().norm()), 4)}
        if cfg["enc"] == "vc" and wc is not None:
            chk["vc_own_first_frame_vs_wrapper_cache_cos"] = cos(to_thinker(fs[0:1])[0], wc)   # harness check
        emit(chk)
        sel = [r for r in rows if (name not in arm_sets or r["set"] in arm_sets[name].split(","))
               and (name not in arm_langs or r["lang"] in arm_langs[name].split(","))]
        sel = sel[:arm_rows[name]] if name in arm_rows else sel
        ut_mode, audio_mode = cfg.get("ut"), cfg.get("audio", "real")         # V1: ut = zero | tok; audio = real | zero
        emit({"arm": name, "user_text_mode": ut_mode, "audio_mode": audio_mode, "n_rows": len(sel),
              "user_text_embedding": "Thinker input table embed_tokens (NeMo's documented init of embed_asr_tokens is a "
                                     "copy of it, duplex_stt_model.py:283); zero vector on frames without a token"})
        torch.cuda.reset_peak_memory_stats()
        for b in range(0, len(sel), a.batch):
            if time.monotonic() - t0 > a.budget_s:
                emit({"stop": "budget", "arm": name, "done": b})
                break
            if mem() < a.soft_mem_mib:
                emit({"stop": "soft_mem", "arm": name, "mem": mem()})
                raise SystemExit("soft memory stop")
            chunk = sel[b:b + a.batch]
            embs, ends, nfr, uts = [], [], [], []
            for r in chunk:
                x = sf.read(Path(a.glex_data if r["data"] == "glex" else a.val_data) / r["path"], dtype="float32")[0]
                w = session_wav(x, cs, a.tail_s)
                if audio_mode == "zero":
                    w = w * 0.0
                with torch.no_grad():
                    f = se(torch.from_numpy(w))["out"]
                embs.append(to_thinker(f))
                if ut_mode:
                    u = torch.zeros(embs[-1].shape[0], embs[-1].shape[-1], device=D.dev, dtype=D.dtype)
                    if ut_mode == "tok":
                        for fr, tid in ut_stream.get(f"{r['set']}|{r['id']}|{r['lang']}", []):
                            if fr < u.shape[0]:
                                u[fr] = D.stt.embed_tokens(torch.tensor([tid], device=D.dev))[0].to(D.dtype)
                    uts.append(u)
                ends.append(cs + int(math.ceil(len(x) / FRAME)))
                nfr.append(embs[-1].shape[0])
            A = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
            UT = torch.nn.utils.rnn.pad_sequence(uts, batch_first=True) if ut_mode else None
            if a.delta_probe and ut_mode == "tok" and b == 0:          # Thinker-level effect of the channel, no decoding
                with torch.no_grad():
                    T = A.shape[1]
                    pad = torch.full((A.shape[0], T), PAD, dtype=torch.long, device=D.dev)
                    ep = D.stt.embed_tokens(pad).to(D.dtype)
                    outs = []
                    for u in (None, UT[:, :T].to(D.dtype)):
                        x = D.fusion(agent_text_embeds=ep, user_audio_embeds=A.to(D.dtype), user_text_embeds=u, function_embeds=ep)
                        o = D.stt(torch.cat([D.prompt_emb.expand(A.shape[0], -1, -1), x], 1), cache=None)
                        P = D.prompt_emb.shape[1]
                        outs.append((x, o["function_logits"][:, P:].float(), o["text_logits"][:, P:].float()))
                    for k, r in enumerate(chunk):
                        dx = (outs[1][0][k] - outs[0][0][k]).float().abs().amax(-1)
                        df = (outs[1][1][k] - outs[0][1][k]).abs().amax(-1)
                        dt = (outs[1][2][k] - outs[0][2][k]).abs().amax(-1)
                        tf = [int(f) for f, _ in ut_stream.get(f"{r['set']}|{r['id']}|{r['lang']}", []) if f < T]
                        off = [t for t in range(T) if t not in set(tf)]
                        emit({"delta_probe": {"row": f"{r['set']}|{r['id']}|{r['lang']}", "token_frames": tf[:3] + tf[-2:],
                              "fusion_dmax_on_token_frames": round(float(dx[tf].max()), 4) if tf else None,
                              "fusion_dmax_off_token_frames": round(float(dx[off].max()), 6) if off else None,
                              "func_logit_dmax": round(float(df.max()), 3), "text_logit_dmax": round(float(dt.max()), 3),
                              "first_frame_func_d_gt_0p5": next((t for t in range(T) if float(df[t]) > 0.5), None)}})
            recs = greedy_fc(D, A, sil, nfr, user_text=UT)
            for r, rec, xe in zip(chunk, recs, ends):
                call = tok.ids_to_text([t for t in rec["fc"] if t != EOTC])
                row = {"arm": name, "set": r["set"], "id": r["id"], "lang": r["lang"], "target": r["target"],
                       "call": call, "score": score_call(call, r["target"]), "beh": behaviour(rec, cs, xe, opener, valid),
                       "agent_text": tok.ids_to_text([t for t in rec["main_text"] if t != PAD])[-200:], "x_end": xe,
                       "tokens": {"text_nonpad": [[t, x] for t, x in enumerate(rec["main_text"]) if x != PAD],
                                  "func_nonpad": [[t, x] for t, x in enumerate(rec["main_func"]) if x != PAD],
                                  "fc": rec["fc"]}}
                emit(row)
                done_rows.append(row)
            emit({"batch": {"arm": name, "done": b + len(chunk), "of": len(sel), "t": round(time.monotonic() - t0, 1),
                            "mem_avail_mib": mem(), "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)}})
            print(name, b + len(chunk), "/", len(sel), round(time.monotonic() - t0, 1), mem(), flush=True)
        if saved:
            prm = dict(se.enc.named_parameters())
            with torch.no_grad():
                for k_, v in saved.items():
                    prm[k_].copy_(v)
    summ = {name: summarise([r for r in done_rows if r["arm"] == name]) for name in {r["arm"] for r in done_rows}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    print("done", round(time.monotonic() - t0, 1), flush=True)


def cli_build_sets():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-sets", action="store_true")
    ap.add_argument("--glex-plan", required=True)
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--val-teacher", required=True)
    ap.add_argument("--val-plans", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gplan = list(map(json.loads, open(a.glex_plan)))
    gman = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.glex_data) / "manifest.jsonl"))}
    vman = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.val_data) / "manifest.jsonl"))}
    vte = json.load(open(a.val_teacher))["items"]
    vplan = [r for p in a.val_plans for r in map(json.loads, open(p))]
    rows = build_sets(gplan, gman, vplan, vman, vte)
    from collections import Counter
    ids_by = {s: {r["id"] for r in rows if r["set"] == s} for s in ("glex_dev", "val", "trainfit")}
    rep = {"counts": dict(Counter(f"{r['set']}|{r['lang']}|{r['target']['name']}" for r in rows)),
           "overlap_val_trainfit": len(ids_by["val"] & ids_by["trainfit"]),
           "overlap_glex_trainfit": len(ids_by["glex_dev"] & ids_by["trainfit"])}
    Path(a.out).write_text(json.dumps({"report": rep, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if "--build-sets" in sys.argv:
        cli_build_sets()
    else:
        main()
