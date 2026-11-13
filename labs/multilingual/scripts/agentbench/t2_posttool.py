"""T2 — fixed-state probe of the POST-TOOL decision: after the tool result is injected, does the Thinker start the spoken
reply (text BOS, as A does at frame 232 in EN m03) or open a new call (function SOTC, as D-V2.1 did live at frame 231
in job 252 fallback-off and job 253 guided)?

State (identical for every arm; only the perception front-end differs):
- positions 0..133: regular frames; agent input = A's recorded tokens (shifted), user audio = the arm's own features
  of the EN m03 clip (NVIDIA streaming perception, Thinker stubbed in the capture pass), function input PAD. The
  function output at 133 is SOTC (A).
- 134..165, FC phase 1 (wrapper:1380-1555): agent PAD, audio = the FC silence embedding, function input SOTC then A's
  call tokens (`search_catalog("3-seat sofa")`, 32 tokens incl. EOTC).
- 166..229, FC phase 2 (two-phase re-entry, wrapper:1279-1296 and :1584-1592): the 63 TOOL_RESPONSE tokens (the
  backend result, numbers converted to words exactly as wrapper:2005-2023, tokenised by the wrapper's tokenizer) are
  FORCED on the function channel at 166..228; position 229 is the model's own function token (A: EOTR, "result
  consumed at 229"). Agent PAD, FC silence embedding. 64 steps = A's recorded `tool.result.injected steps=64`.
- 230..: regular frames, FREE greedy run: agent input = own previous text token, function input = own previous
  function token, user audio = the arm's features of steady caller silence (clip tail >= 5.6 s after the caller,
  beyond the encoder's 70-frame left context, like the live wall-clock frames after the FC). Stops at a function
  SOTC (a new call) or after N frames.

Pre-registered gate (A, decisive tokens exact; stop if any fails):
  P1 text argmax == A's recorded token on every frame 0..133, function PAD on 0..132 and SOTC at 133;
  P2 phase 1: function argmax == A's call token at all 32 positions;
  P3 phase 2: the natural function token at 229 == EOTR;
  P4 free run 230..258: function argmax PAD on every frame (no new call); text argmax == A's recorded reply tokens on
     232..258 (BOS at 232, "I found a three-seat sofa for four hundred ninety-nine EUR. Would you like ..."), and
     not BOS / not SOTC on 230..231 (their recorded ids are not in the trace).
  RNNT turn-taking is not simulated: under the streaming config no forced BOS/EOS can fire on 230..258 (the RNNT
  blank counter restarts after the FC), which P4 checks on A.

Measures (per arm, per cell):
- the natural function token and logP(EOTR) at 229;
- per free frame: text/func argmax, logP(BOS), logP(SOTC);
- outcome: first reply BOS frame, or first SOTC frame (new call), within the window; reply text decoded.

Attribution cells for a front-end X (state 0..133 always X's own features):
  own      : X FC-silence embedding, X tail audio
  silA     : A's FC-silence embedding, X tail audio
  tailA    : X FC-silence embedding, A's tail audio
  silA_tailA: both from A
  If X opens a new call in `own` and replies in `silA_tailA`, the post-tool decision is driven by X's representation
  of silence; `silA`/`tailA` split it between the FC-silence embedding and the post-FC audio frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1_probe import (BOS, EOS, EOTC, PAD, SOTC, Thinker, fc_call, frame_tokens, logp_rows,  # noqa: E402
                      prev_shift, trace_session)

EOTR = 22
S_FC = 133
FREE_FROM = 230
GATE_TO = 258
TAIL_FROM_S = 10.5 + 5.6 + 0.6    # first tail frame used: caller end + encoder left context + margin


# ----------------------------------------------------------------------------------------------- pure schedule
def schedule(hist: list[int], s: int, call: list[int], resp: list[int]) -> list[dict]:
    """Positions 0 .. s + len(call) + len(resp) + 1 - 1 with their inputs and the forced function output (if any).
    call = A's call tokens AFTER SOTC (incl. EOTC); resp = TOOL_RESPONSE tokens (forced)."""
    rows = []
    for t in range(s + 1):
        rows.append({"pos": t, "kind": "main", "agent_in": PAD if t == 0 else hist[t - 1], "func_in": PAD,
                     "audio": ("feat", t)})
    for k in range(1, len(call) + 1):
        rows.append({"pos": s + k, "kind": "fc1", "agent_in": PAD, "func_in": SOTC if k == 1 else call[k - 2],
                     "audio": ("sil",), "expect_func": call[k - 1]})
    p0 = s + len(call) + 1
    for i in range(len(resp) + 1):
        rows.append({"pos": p0 + i, "kind": "fc2", "agent_in": PAD,
                     "func_in": call[-1] if i == 0 else resp[i - 1], "audio": ("sil",),
                     "forced_func": resp[i] if i < len(resp) else None})
    return rows


def free_run(step, cache, tail_rows, agent_prev: int, func_prev: int, n: int, stop_on_sotc: bool = True):
    """Greedy main-loop frames: own previous tokens as inputs. step(cache, audio_row, agent_prev, func_prev) ->
    (text_logits, func_logits, cache)."""
    out = []
    for k in range(n):
        tl, fl, cache = step(cache, tail_rows[k], agent_prev, func_prev)
        lt, lf = logp_rows(tl), logp_rows(fl)
        ta, fa = int(lt.argmax()), int(lf.argmax())
        out.append({"pos": FREE_FROM + k, "text": ta, "func": fa, "lp_BOS": round(float(lt[BOS]), 3),
                    "lp_SOTC": round(float(lf[SOTC]), 3), "lp_text_PAD": round(float(lt[PAD]), 3),
                    "lp_func_PAD": round(float(lf[PAD]), 3),
                    # margins vs the best ALTERNATIVE token of the same head (owner: not only the binary outcome)
                    "sotc_margin": round(float(lf[SOTC] - torch.cat([lf[:SOTC], lf[SOTC + 1:]]).max()), 3),
                    "bos_margin": round(float(lt[BOS] - torch.cat([lt[:BOS], lt[BOS + 1:]]).max()), 3),
                    "func_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lf.topk(2))],
                    "text_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lt.topk(2))]})
        if fa == SOTC and stop_on_sotc:
            break
        agent_prev, func_prev = ta, fa
    return out, cache


def outcome(rows: list[dict]) -> dict:
    bos = next((r["pos"] for r in rows if r["text"] == BOS), None)
    sotc = next((r["pos"] for r in rows if r["func"] == SOTC), None)
    first = None if bos is None and sotc is None else ("reply" if sotc is None or (bos is not None and bos < sotc)
                                                       else "new_call")
    return {"first": first, "reply_bos": bos, "new_call_sotc": sotc,
            "reply_ids": [r["text"] for r in rows if bos is not None and r["pos"] >= bos and r["text"] != PAD][:40]}


def gate_check(hist_pred, hist_rec, fc1_pred, call, nat229, free_rows, rec) -> dict:
    p1_text = [t for t in range(S_FC + 1) if hist_pred[t][0] != rec.get(t, PAD)]
    p1_func = all(hist_pred[t][1] == PAD for t in range(S_FC)) and hist_pred[S_FC][1] == SOTC
    p2 = [k for k, (a, b) in enumerate(zip(fc1_pred, call)) if a != b]
    by = {r["pos"]: r for r in free_rows}
    p4_func = [p for p in range(FREE_FROM, GATE_TO + 1) if p not in by or by[p]["func"] != PAD]
    p4_text = [p for p in range(232, GATE_TO + 1) if p not in by or by[p]["text"] != rec.get(p)]
    p4_pre = [p for p in (230, 231) if p not in by or by[p]["text"] in (BOS,) or by[p]["func"] == SOTC]
    ok = not p1_text and p1_func and not p2 and nat229 == EOTR and not p4_func and not p4_text and not p4_pre
    return {"pass": ok, "P1_text_mismatch": p1_text, "P1_func_ok": p1_func, "P2_mismatch": p2,
            "P3_nat229": nat229, "P4_func_not_pad": p4_func, "P4_text_mismatch": p4_text, "P4_230_231_bad": p4_pre}


# ----------------------------------------------------------------------------------------------- Thinker driving
def pos_summary(pos, kind, tl, fl) -> dict:
    """Per-position control log-probs, in the same keys as observe.py `logits` rows (for live-vs-replay diffs)."""
    lt, lf = logp_rows(tl), logp_rows(fl)
    return {"pos": pos, "kind": kind, "pred_text": int(lt.argmax()), "pred_func": int(lf.argmax()),
            "text": {"PAD": float(lt[PAD]), "BOS": float(lt[BOS]), "EOS": float(lt[EOS])},
            "func": {"PAD": float(lf[PAD]), "SOTC": float(lf[SOTC]), "EOTC": float(lf[EOTC])}}


def run_state(th, cache, sched, feats, sil, record=True, dump=None):
    """Run the schedule; returns (cache after the last position, per-main-frame (text, func) argmax, fc1 argmax,
    fc2 natural token at the last position, logP(EOTR) there, fc2 forced-position agreement)."""
    hist_pred, fc1, fc2_agree = [], [], []
    nat, lp_eotr = None, None
    for r in sched:
        audio = feats[r["audio"][1]] if r["audio"][0] == "feat" else sil
        p_before = int(getattr(cache, "_v2_pos", -1))
        tl, fl, cache = th.step(cache, audio, r["agent_in"], r["func_in"])
        if dump is not None:
            d = pos_summary(r["pos"], r["kind"], tl, fl)
            # consumed inputs and cache bookkeeping (replay side; the live trace has no equivalent fields)
            d.update({"agent_in": r["agent_in"], "func_in": r["func_in"], "audio": r["audio"][0],
                      "v2_pos_before": p_before, "v2_pos_after": int(getattr(cache, "_v2_pos", -1)),
                      "logit_dtype": str(tl.dtype)})
            dump.append(d)
        if r["kind"] == "main" and record:
            hist_pred.append((int(tl.argmax()), int(fl.argmax())))
        elif r["kind"] == "fc1":
            fc1.append(int(fl.argmax()))
        elif r["kind"] == "fc2":
            if r["forced_func"] is not None:
                fc2_agree.append(int(fl.argmax()) == r["forced_func"])
            else:
                lf = logp_rows(fl)
                nat, lp_eotr = int(lf.argmax()), round(float(lf[EOTR]), 3)
    return cache, hist_pred, fc1, nat, lp_eotr, (sum(fc2_agree), len(fc2_agree))


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-root", default=str(Path.home() / "mynah-s2s/ml/data/agentbench/tts_m1"))
    ap.add_argument("--a-trace", default=str(Path.home() / "mynah-s2s/ml/results/agentbench/m1-parity/vc_en_m03/trace.jsonl"))
    ap.add_argument("--budget-s", type=float, default=165.0)
    ap.add_argument("--n-free", type=int, default=33)
    ap.add_argument("--tail-s", type=float, default=10.0)
    ap.add_argument("--tail-mode", default="steady", choices=["steady", "continuation"],
                    help="post-FC audio: steady silence (job 254/260) or the frames right after the last regular frame "
                         "(134..), which is what the live main loop consumes after the FC (observe rows, jobs 252-259)")
    ap.add_argument("--post-fc-inputs", default="fc",
                    help="token inputs of the first main frame after the FC: 'fc' = what the FC thread produced "
                         "(agent PAD, function = its own token at 229, EOTR); 'live0' = what the live runtime consumes "
                         "(id 0 on both channels: context buffers initialised with 0 and never synced from the FC "
                         "thread's clones; s2s_context_manager.py:80, streaming_s2s_pipeline.py:1591, 1040-1062); "
                         "'pairs:0/0,12/12,12/0,0/22,12/22' = factorial, one branch per (text, function) pair from the "
                         "same cached state at the last FC position")
    ap.add_argument("--skip-gate-a", action="store_true",
                    help="no A gate cell: only for runs whose validity is checked against LIVE rows (--dump-positions)")
    ap.add_argument("--dump-positions", action="store_true", help="per-position control log-probs of every cell")
    ap.add_argument("--cells", default="", help="'arm:kind,...' override of the attribution plan")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    if (out / "t2.json").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "multilingual"))
    from e2e_qa import apply_variant, parse_variant
    from m0_runner import load_16k, render_prompt
    from perception_swap import FrontEnd, load_nemo_dir
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import iter_chunks
    rep, timing = {"cells": [], "skipped": []}, {}

    def save():
        rep["timing_s"], rep["elapsed_s"] = timing, round(time.monotonic() - T0, 1)
        (out / "t2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    torch.manual_seed(1234)
    t = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}), llm_cache=True)
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):     # decode_audio=False harness fix (decision D7)
        return _orig_log(frames, torch.zeros(1, 0) if audio_wave is None else audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    timing["engine_load"] = round(time.monotonic() - t, 1)
    spec = json.load(open(HERE / "tenant_furniture.json"))
    be = TenantBackend(spec)
    reg = ToolRegistry()
    for tl in spec["tools"]:
        reg.register(tl["name"], be.handlers()[tl["name"]], tl["description"], tl["parameters"])
    prompt = render_prompt(reg.schemas, spec["system_prompt"])
    assert len(prompt) == 2738, len(prompt)
    th = Thinker(eng, prompt)
    w = eng.wrapper
    # ---- tokens: A's call, the TOOL_RESPONSE exactly as the wrapper builds it
    sess = trace_session(a.a_trace)
    rec, (s, fids) = frame_tokens(sess), fc_call(sess)
    assert s == S_FC and fids[0] == SOTC and fids[-1] == EOTC
    call = fids[1:]
    result = be.search_catalog({"query": "3-seat sofa"})          # the handler string, as the live controller got it
    wrapped = f"<TOOL_RESPONSE>[{result}]</TOOL_RESPONSE>"       # pipeline wrapper (streaming_s2s_pipeline:1484)
    conv = w._convert_tool_response_nums_to_text(wrapped) if w._fc_convert_num_to_text else wrapped
    resp = list(w._build_fc_response_tokens(wrapped))
    rep["tokens"] = {"wrapped_chars": len(wrapped), "converted_chars": len(conv), "expect_chars": 201,
                     "converted": conv, "n_resp": len(resp),
                     "phase2_steps": len(resp) + 1, "expect_phase2_steps": 64, "call_ok": th.encode(th.decode(call[:-1])) == call[:-1]}
    if len(resp) + 1 != 64 or len(conv) != 201:        # A's trace: tool.result.injected steps=64, chars=201
        save()
        raise SystemExit("P0 failed: TOOL_RESPONSE length does not match A's recorded phase-2 steps")
    hist = [rec.get(t, PAD) for t in range(S_FC + 1)]
    sched = schedule(hist, S_FC, call, resp)
    ARMS = {"A": "vc",
            "C-ridge": "enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out.pt",
            "D-V2.1": "enc=ml;att=56;rnnt=ml;ln={R}/v21/distill/v21/encoder_ln_best.pt;adapter={R}/v21/distill/v21/adapter_best.pt",
            "D-E5": "enc=ml;att=56;rnnt=ml;ln={R}/distill/e5/encoder_ln_best.pt;adapter={R}/distill/e5/adapter_best.pt"}
    R = str(Path.home() / "mynah-s2s/ml/results")
    fe = FrontEnd(eng)
    fe.snapshot()
    ckpt = {"en": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-en"),
            "ml": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-ml")}
    man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(Path(a.audio_root) / "manifest.jsonl"))}
    x = load_16k(Path(a.audio_root) / man[("m03_t0", "en", "vivian")]["path"])
    wav = torch.cat([torch.zeros(int(7.0 * 16000)), x, torch.zeros(int(a.tail_s * 16000))])
    tail0 = int(round(TAIL_FROM_S / 0.08)) if a.tail_mode == "steady" else S_FC + 1
    rep["clip"] = {"caller_end_s": round(7.0 + x.numel() / 16000, 2), "tail_first_frame": tail0, "tail_mode": a.tail_mode}
    # live cosine(audio input, FC silence embedding) at the first two post-tool positions (observe feature rows):
    # mapping check for the continuation tail, pre-registered tolerance 0.02
    LIVE_COS = {"D-V2.1": [0.607, 0.603], "C-ridge": [0.507, 0.542]}
    rep["tail_mapping_check"] = {}
    feats, sils = {}, {}

    def use(arm):
        t = time.monotonic()
        apply_variant(fe, parse_variant(ARMS[arm].format(R=R)), ckpt)     # refreshes the silence embedding
        sils[arm] = th.silence().detach().clone()
        fx = []
        orig_if, orig_fu = w.model_llm_interface, w.fusion_module.forward

        def stub(*args, **kw):
            pad = torch.full((1,), PAD, dtype=torch.long, device=th.dev)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad,
                    "cache": kw.get("cache")}

        def fuse(*args, **kw):
            ua = kw.get("user_audio_embeds")
            if ua is not None and ua.dim() == 3 and ua.shape[1] == 1:
                fx.append(ua[0, 0].detach().clone())
            return orig_fu(*args, **kw)
        w.model_llm_interface, w.fusion_module.forward = stub, fuse
        try:
            eng.open(None)
            chunks = list(iter_chunks(wav, eng.chunk_samples))
            for i, c in enumerate(chunks):
                eng.step(c, is_last=(i == len(chunks) - 1))
            eng.close()
        finally:
            w.model_llm_interface, w.fusion_module.forward = orig_if, orig_fu
        feats[arm] = torch.stack(fx)
        cs = [round(float(torch.nn.functional.cosine_similarity(feats[arm][k].float().reshape(-1),
                                                                 sils[arm].float().reshape(-1), 0)), 3)
              for k in (tail0, tail0 + 1)]
        live = LIVE_COS.get(arm)
        rep["tail_mapping_check"][arm] = {"capture_cos": cs, "live_cos": live,
                                          "match": None if live is None else all(abs(c - l) < 0.02 for c, l in zip(cs, live))}
        assert feats[arm].shape[0] >= tail0 + a.n_free, (feats[arm].shape, tail0)
        timing.setdefault("swap_capture", []).append([arm, round(time.monotonic() - t, 1), feats[arm].shape[0]])

    def cell(arm, sil_from, tail_from, base=None):
        t = time.monotonic()
        dump = [] if a.dump_positions else None
        if base is None:
            d0 = [] if dump is not None else None
            c0, hp, _, _, _, _ = run_state(th, th.prefill(), sched[:S_FC + 1], feats[arm], sils[arm], dump=d0)
            base = (th.clone(c0), hp, d0)
        if dump is not None:
            dump.extend(base[2] or [])
        cache, _, fc1, nat, lp, agree = run_state(th, th.clone(base[0]), sched[S_FC + 1:], feats[arm], sils[sil_from],
                                                  dump=dump)
        tail = feats[tail_from][tail0:tail0 + a.n_free]
        factorial = None
        if a.post_fc_inputs.startswith("pairs:"):
            factorial = {}
            for pr in a.post_fc_inputs[6:].split(","):
                silent = pr.endswith("s")      # training-faithful: EOTR consumed in one INSERTED silent step
                ta, fa = (int(x) for x in pr.rstrip("s").split("/"))
                bc = th.clone(cache)
                pre = None
                if silent:                     # duplex_stt_model.py:1053-1073 + label_prep.py:286-288: the EOTR input
                    tl0, fl0, bc = th.step(bc, sils[sil_from], ta, fa)   # step is inserted (silence audio, PAD text)
                    lt0, lf0 = logp_rows(tl0), logp_rows(fl0)
                    pre = {"text": int(lt0.argmax()), "func": int(lf0.argmax()), "lp_BOS": round(float(lt0[BOS]), 3),
                           "lp_SOTC": round(float(lf0[SOTC]), 3), "lp_func_PAD": round(float(lf0[PAD]), 3),
                           "sotc_margin": round(float(lf0[SOTC] - torch.cat([lf0[:SOTC], lf0[SOTC + 1:]]).max()), 3)}
                    ta, fa = pre["text"], pre["func"]   # the main loop then resumes on the model's own outputs
                frows, fcache = free_run(th.step, bc, tail, ta, fa, a.n_free)
                fo = outcome(frows)
                if fo["first"] == "new_call":
                    prev, ids = SOTC, []
                    for _ in range(36):
                        _, fl2, fcache = th.step(fcache, sils[sil_from], PAD, prev)
                        prev = int(fl2.argmax())
                        ids.append(prev)
                        if prev == EOTC:
                            break
                    fo["new_call_text"] = th.decode([i for i in ids if i != EOTC])
                fo["reply_text"] = th.decode([i for i in fo["reply_ids"] if i not in (BOS, EOS)])
                factorial[pr] = {"silent_eotr_step": pre, "first_frame": frows[0] if frows else None, "outcome": fo}
            a0, f0 = PAD, nat if nat is not None else PAD                       # main row = the FC thread's own inputs
        else:
            a0, f0 = (0, 0) if a.post_fc_inputs == "live0" else (PAD, nat if nat is not None else PAD)
        rows, cache_end = free_run(th.step, cache, tail, a0, f0, a.n_free)
        o = outcome(rows)
        o["reply_text"] = th.decode([i for i in o["reply_ids"] if i not in (BOS, EOS)])
        if o["first"] == "new_call":           # decode the new call greedily (FC phase-1 inputs, same silence emb)
            prev, ids = SOTC, []
            for _ in range(48):
                _, fl, cache_end = th.step(cache_end, sils[sil_from], PAD, prev)
                prev = int(fl.argmax())
                ids.append(prev)
                if prev == EOTC:
                    break
            o["new_call_ids"], o["new_call_text"] = ids, th.decode([i for i in ids if i != EOTC])
        import hashlib
        hsh = hashlib.sha256(json.dumps([[r["agent_in"], r["func_in"], r.get("forced_func"), r["audio"][0]]
                                         for r in sched]).encode()).hexdigest()[:12]
        res = {"arm": arm, "sil": sil_from, "tail": tail_from,
               "controlled": {"token_history_sha": hsh, "kv_positions_before_free": th.P + len(sched),
                              "first_free_position": FREE_FROM, "rnnt_controller": "not simulated (identical: none)",
                              "changed_inputs": [x for x, src in (("fc_silence_embedding", sil_from),
                                                                  ("post_fc_audio_frames", tail_from)) if src != arm]},
               "first_post_tool_frame": rows[0] if rows else None, "post_fc_inputs": [a0, f0],
               "factorial": factorial, "fc1_match_A_call": sum(p == c for p, c in zip(fc1, call)),
               "fc2_forced_argmax_agree": agree, "nat229": nat, "lp_EOTR_229": lp, "outcome": o,
               "free": rows[:12], "s": round(time.monotonic() - t, 1)}
        if dump is not None:
            res["positions"] = dump + [{"pos": r["pos"], "kind": "free", "pred_text": r["text"], "pred_func": r["func"],
                                        "text": {"PAD": r["lp_text_PAD"], "BOS": r["lp_BOS"]},
                                        "func": {"PAD": r["lp_func_PAD"], "SOTC": r["lp_SOTC"]}} for r in rows]
        return res, base, (fc1, nat, rows)

    if a.skip_gate_a:
        rep["gate"] = "skipped (validity checked against the live observe rows of positions 0..133)"
        # A is captured only if a cell takes the silence embedding or the tail from A (job 263 lost its only cell
        # to the budget guard partly because of an unneeded A capture)
    # ---- gate on A
    if not a.skip_gate_a:
        use("A")
    if not a.skip_gate_a:
        r, baseA, (fc1, nat, rows) = cell("A", "A", "A")
        g = gate_check(baseA[1], hist, fc1, call, nat, rows, rec)
        rep["gate"], r["gate"] = g, True
        rep["cells"].append(r)
        save()
        print("GATE", "PASS" if g["pass"] else "FAILED", json.dumps(g)[:600], flush=True)
        if not g["pass"]:
            raise SystemExit(3)
    plan = [("D-V2.1", "own"), ("D-V2.1", "silA_tailA"), ("D-V2.1", "silA"), ("D-V2.1", "tailA"), ("C-ridge", "own"),
            ("D-E5", "own"), ("C-ridge", "silA_tailA")]
    if a.cells:
        plan = [tuple(c.split(":")) for c in a.cells.split(",")]
    costs, bases = {"base": [], "branch": []}, {}
    for i, (arm, kind) in enumerate(plan):
        # per-cell estimate (job 254 skipped everything with one max-based guess): a cell that needs the arm's
        # 0..133 base costs ~2x a branch from an existing base; capture/swap of a new arm ~10 s
        need_base = arm not in bases
        k = "base" if need_base else "branch"
        est = (max(costs[k]) if costs[k] else (32 if need_base else 14)) * 1.15 + (10 if arm not in feats else 0)
        if time.monotonic() - T0 + est > a.budget_s:
            rep["skipped"] = [list(p) for p in plan[i:]]
            print("BUDGET: skipping", len(plan) - i, "cells", flush=True)
            break
        if arm not in feats:
            use(arm)
        if kind in ("silA", "tailA", "silA_tailA") and "A" not in feats:
            use("A")
        sil_from = "A" if kind in ("silA", "silA_tailA") else arm
        tail_from = "A" if kind in ("tailA", "silA_tailA") else arm
        t = time.monotonic()
        r, bases[arm], _ = cell(arm, sil_from, tail_from, bases.get(arm))
        r["kind"] = kind
        costs[k].append(time.monotonic() - t)
        rep["cells"].append(r)
        save()
        o = r["outcome"]
        print(f"{arm:8} {kind:10} nat229={r['nat229']} lpEOTR={r['lp_EOTR_229']} first={o['first']} "
              f"bos={o['reply_bos']} sotc={o['new_call_sotc']} reply={o['reply_text'][:80]!r} ({r['s']}s)", flush=True)
    fe.restore()
    save()
    print("DONE", round(time.monotonic() - T0, 1), flush=True)


if __name__ == "__main__":
    main()
