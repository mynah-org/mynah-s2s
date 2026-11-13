"""T2 factorial across RESULT TYPES: the six resume contracts on several post-tool states, one engine load.

Owner GO 2026-11-12 (cap 5 GPU-min): the question is not which contract minimises SOTC but when a post-result SOTC is
a legitimate continuation and when it is an invented call. Per scenario the state (token history, call, tool result,
audio) is IDENTICAL across contracts and arms; only the perception front-end (the arm) and the token pair consumed at
the first main-loop frame after the FC (the contract) change.

Scenarios (history = the recorded consumed text ids of the named trace, teacher-forced for every arm):
  m03_empty  A's EN m03 history and call search_catalog("3-seat sofa"); the backend's REAL empty-result string
             (the handler output for a non-matching query), so the result type is the only change vs job 277.
  m08_vr     the EN m08 guided session (A's opener, then D-V2.1; job 269, fix-p2-on): call find_customer(full_name=
             "Lindvik") at 147, already an invented slot (the caller never says a name). The handler's
             verification_required result. Measures the POST-RESULT behaviour only, not the first wrong call.
  m07_chain  A's EN m07 history and call find_customer("Mario Rossi", "20135") -> verified; live A then chained
             prepare_return at 294 under (0, 0). The legitimate-chain control: a contract that kills this SOTC costs
             a real capability.

Contracts (text, function) at the first resume frame: 0/0 upstream, 12/12, 12/0, 0/22, 12/22, and 12/22s (EOTR
consumed in one inserted silent step with PAD text, the training layout, then the model's own outputs).

Pre-registered validity checks (recorded; a failing check invalidates only that scenario's numbers):
  V0 phase lengths: len(call) == trace `function.arguments.done steps`; len(resp)+1 == `tool.result.injected steps`
     and converted chars == its `chars` (m08, m07; m03_empty has no live reference);
  V1 A on its own trace (m03_empty, m07): text argmax == recorded on 0..s, SOTC at s, call argmax == recorded call;
  H5 live replication: m08 D-V2.1 under 12/22 (= live F1+F2) and m07 A under 0/0 (= live flags OFF) emit SOTC at the
     FIRST resume frame (live observe logits: m08 t=225 SOTC-PAD +1.25, m07 t=293 +4.0; the second call's
     function.arguments.done model_step is reported one frame later, 226 / 294, which job 278 wrongly pre-registered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1_probe import BOS, EOS, EOTC, PAD, SOTC, Thinker, fc_call, frame_tokens, logp_rows, trace_session  # noqa: E402

EOTR = 22
CONTRACTS = "0/0,12/12,12/0,0/22,12/22,12/22s"
# first-cell wall estimates for the budget guard (job 278: m08 41 s = 225 forced steps + 5 re-call decodes, ~85 ms/step)
FIRST_CELL_S = {"m08_vr": 42.0, "m07_chain": 55.0, "m03_empty": 40.0}
RES = Path.home() / "mynah-s2s/ml/results/agentbench"
SCEN = {
    "m03_empty": {"clip": "m03_t0", "trace": RES / "m1-parity/vc_en_m03/trace.jsonl", "trace_arm": "A",
                  "tool": "search_catalog", "args": {"query": "3-seat sofa"}, "result_args": {"query": "zzz qqq"},
                  "expect_result": '{"results": [], "next_step": "Ask the caller which product they mean."}',
                  "h5": None},
    "m08_vr": {"clip": "m08_t0", "trace": RES / "fix-p2-on/v21_en_m08_guided/trace.jsonl", "trace_arm": None,
               "tool": "find_customer", "args": {"full_name": "Lindvik"}, "result_args": None,
               "expect_result": None, "h5": ("D-V2.1", "12/22", "find_customer", 0)},
    "m07_chain": {"clip": "m07_t0", "audio_root": "tts_m0_en", "trace": RES / "rb-en/vc_en_m07/trace.jsonl", "trace_arm": "A",
                  "tool": "find_customer", "args": {"full_name": "Mario Rossi", "postcode": "20135"},
                  "result_args": None, "expect_result": None, "h5": ("A", "0/0", "prepare_return", 0)},
}


# ----------------------------------------------------------------------------------------------- pure helpers
def build_schedule(hist: list[int], s: int, call: list[int], resp: list[int]) -> list[dict]:
    """Same layout as t2_posttool.schedule (positions 0..s main, s+1.. FC phase 1, then phase 2 forced)."""
    from t2_posttool import schedule
    return schedule(hist, s, call, resp)


def first_free(s: int, call: list[int], resp: list[int]) -> int:
    """First main-loop frame after the FC: s + phase-1 steps + phase-2 steps (len(resp) + 1) + 1."""
    return s + len(call) + len(resp) + 2


def parse_pair(pr: str) -> tuple[int, int, bool]:
    silent = pr.endswith("s")
    ta, fa = (int(x) for x in pr.rstrip("s").split("/"))
    return ta, fa, silent


def trace_lengths(sess: list[dict]) -> dict:
    """Phase lengths and result chars of the FIRST FC in a trace."""
    d = {}
    for e in sess:
        if e["ev"] == "function.arguments.done" and "call_steps" not in d:
            d["call_steps"] = e.get("steps")
        elif e["ev"] == "tool.result.injected" and "result_steps" not in d:
            d["result_steps"], d["result_chars"] = e.get("steps"), e.get("chars")
        elif e["ev"] == "tool.result.consumed" and "consumed" not in d:
            d["consumed"] = e.get("model_step")
    return d


def margin(lp: torch.Tensor, k: int) -> float:
    return round(float(lp[k] - torch.cat([lp[:k], lp[k + 1:]]).max()), 3)


def frame_row(pos: int, tl, fl) -> dict:
    lt, lf = logp_rows(tl), logp_rows(fl)
    return {"pos": pos, "text": int(lt.argmax()), "func": int(lf.argmax()), "sotc_margin": margin(lf, SOTC),
            "sotc_minus_pad": round(float(lf[SOTC] - lf[PAD]), 3), "bos_margin": margin(lt, BOS),
            "lp_SOTC": round(float(lf[SOTC]), 3), "lp_func_PAD": round(float(lf[PAD]), 3),
            "func_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lf.topk(2))],
            "text_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lt.topk(2))]}


def summarise(rows: list[dict]) -> dict:
    bos = next((r["pos"] for r in rows if r["text"] == BOS), None)
    sotc = next((r["pos"] for r in rows if r["func"] == SOTC), None)
    first = None if bos is None and sotc is None else ("reply" if sotc is None or (bos is not None and bos < sotc)
                                                       else "new_call")
    ids = [r["text"] for r in rows if bos is not None and r["pos"] >= bos and r["text"] not in (PAD, BOS, EOS)]
    return {"first": first, "reply_bos": bos, "new_call_sotc": sotc, "reply_ids": ids,
            "max_sotc_margin": max((r["sotc_margin"] for r in rows), default=None)}


# ----------------------------------------------------------------------------------------------- GPU
def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "mynah-s2s/ml/data/agentbench"))
    ap.add_argument("--scenarios", default="m08_vr,m03_empty,m07_chain")
    ap.add_argument("--arms", default="A,D-V2.1,C-ridge")
    ap.add_argument("--contracts", default=CONTRACTS)
    ap.add_argument("--n-free", type=int, default=16)
    ap.add_argument("--call-cap", type=int, default=48)
    ap.add_argument("--tail-s", type=float, default=4.0)
    ap.add_argument("--lead-s", type=float, default=7.0)
    ap.add_argument("--budget-s", type=float, default=265.0)
    ap.add_argument("--plan", default="", help="ordered 'scenario:arm,...' (priority order under the budget guard); "
                                               "default: every scenario x every arm, scenario-major")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    if (out / "t2f.json").exists():
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
    rep = {"scenarios": {}, "cells": [], "skipped": [], "timing_s": {}}

    def save():
        rep["elapsed_s"] = round(time.monotonic() - T0, 1)
        (out / "t2f.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    torch.manual_seed(1234)
    t = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}), llm_cache=True)
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):     # decode_audio=False harness fix (decision D7)
        return _orig_log(frames, torch.zeros(1, 0) if audio_wave is None else audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    rep["timing_s"]["engine_load"] = round(time.monotonic() - t, 1)
    spec = json.load(open(HERE / "tenant_furniture.json"))
    reg, be0 = ToolRegistry(), TenantBackend(spec)
    for tl in spec["tools"]:
        reg.register(tl["name"], be0.handlers()[tl["name"]], tl["description"], tl["parameters"])
    prompt = render_prompt(reg.schemas, spec["system_prompt"])
    assert len(prompt) == 2738, len(prompt)
    th = Thinker(eng, prompt)
    w = eng.wrapper

    # ---- scenario states (CPU-side: tokens, schedule, checks)
    states = {}
    for sc in a.scenarios.split(","):
        cfg = SCEN[sc]
        sess = trace_session(str(cfg["trace"]))
        rec, (s, fids) = frame_tokens(sess), fc_call(sess)
        assert fids[0] == SOTC and fids[-1] == EOTC, sc
        call = fids[1:]
        be = TenantBackend(spec)                      # fresh backend: no call budget carried between scenarios
        result = be.handlers()[cfg["tool"]](cfg["result_args"] or cfg["args"])
        if cfg["expect_result"] is not None:
            assert result == cfg["expect_result"], (sc, result)
        wrapped = f"<TOOL_RESPONSE>[{result}]</TOOL_RESPONSE>"
        conv = w._convert_tool_response_nums_to_text(wrapped) if w._fc_convert_num_to_text else wrapped
        resp = list(w._build_fc_response_tokens(wrapped))
        tlen = trace_lengths(sess)
        call_text = th.decode(call[:-1])
        v0 = {"call_steps": [len(call), tlen.get("call_steps")], "result_steps": [len(resp) + 1, tlen.get("result_steps")],
              "result_chars": [len(conv), tlen.get("result_chars")],
              "call_has_args": all(str(v) in call_text for v in cfg["args"].values())}
        v0["pass"] = (v0["call_steps"][0] == v0["call_steps"][1] and v0["call_has_args"] and
                      (sc == "m03_empty" or (v0["result_steps"][0] == v0["result_steps"][1] and
                                             v0["result_chars"][0] == v0["result_chars"][1])))
        hist = [rec.get(t, PAD) for t in range(s + 1)]
        sched = build_schedule(hist, s, call, resp)
        ff = first_free(s, call, resp)
        if sc != "m03_empty":
            v0["first_free_vs_trace"] = [ff, (tlen.get("consumed") or -2) + 1]
            v0["pass"] = v0["pass"] and ff == (tlen.get("consumed") or -2) + 1
        states[sc] = {"s": s, "call": call, "resp": resp, "hist": hist, "rec": rec, "sched": sched, "ff": ff,
                      "sha": hashlib.sha256(json.dumps([[r["agent_in"], r["func_in"], r.get("forced_func"),
                                                         r["audio"][0]] for r in sched]).encode()).hexdigest()[:12]}
        rep["scenarios"][sc] = {"s": s, "call_text": call_text, "result": result, "converted": conv,
                                "n_resp": len(resp), "first_free": ff, "V0": v0, "schedule_sha": states[sc]["sha"],
                                "trace": str(cfg["trace"])}
        print("SCEN", sc, json.dumps(rep["scenarios"][sc]["V0"]), call_text[:80], flush=True)
    save()

    plan = ([tuple(c.split(":")) for c in a.plan.split(",")] if a.plan
            else [(sc, arm) for sc in states for arm in a.arms.split(",")])
    assert all(sc in states for sc, _ in plan), plan
    rep["plan"] = [list(p) for p in plan]
    ARMS = {"A": "vc",
            "C-ridge": "enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out.pt",
            "D-V2.1": "enc=ml;att=56;rnnt=ml;ln={R}/v21/distill/v21/encoder_ln_best.pt;adapter={R}/v21/distill/v21/adapter_best.pt"}
    R = str(Path.home() / "mynah-s2s/ml/results")
    fe = FrontEnd(eng)
    fe.snapshot()
    ckpt = {"en": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-en"),
            "ml": load_nemo_dir(Path.home() / "mynah-s2s/ml/ckpt/asr-ml")}
    def clip_path(sc):   # the audio root of the LIVE session each trace comes from (m07: tts_m0_en, others tts_m1)
        root = Path(a.data_root) / SCEN[sc].get("audio_root", "tts_m1")
        man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(root / "manifest.jsonl"))}
        return root / man[(SCEN[sc]["clip"], "en", "vivian")]["path"]
    feats, sils = {}, {}

    def capture(arm):
        """Apply the arm's front-end once and capture its user-audio embeddings for every scenario clip (the Thinker
        steps afterwards use only these tensors and the arm's FC silence embedding)."""
        t = time.monotonic()
        apply_variant(fe, parse_variant(ARMS[arm].format(R=R)), ckpt)
        sils[arm] = th.silence().detach().clone()
        orig_if, orig_fu = w.model_llm_interface, w.fusion_module.forward

        def stub(*args, **kw):
            pad = torch.full((1,), PAD, dtype=torch.long, device=th.dev)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad,
                    "cache": kw.get("cache")}
        for sc in [sc_ for sc_, arm_ in plan if arm_ == arm]:
            x = load_16k(clip_path(sc))
            wav = torch.cat([torch.zeros(int(a.lead_s * 16000)), x, torch.zeros(int(a.tail_s * 16000))])
            fx = []

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
            feats[(arm, sc)] = torch.stack(fx)
            need = states[sc]["s"] + 1 + a.n_free + 2
            assert feats[(arm, sc)].shape[0] >= need, (arm, sc, feats[(arm, sc)].shape, need)
        rep["timing_s"].setdefault("capture", []).append([arm, round(time.monotonic() - t, 1)])

    def decode_call(cache, sil):
        prev, ids = SOTC, []
        for _ in range(a.call_cap):
            _, fl, cache = th.step(cache, sil, PAD, prev)
            prev = int(fl.argmax())
            ids.append(prev)
            if prev == EOTC:
                break
        return ids, th.decode([i for i in ids if i != EOTC])

    def run_cell(arm, sc):
        t = time.monotonic()
        st, f, sil = states[sc], feats[(arm, sc)], sils[arm]
        s, sched = st["s"], st["sched"]
        cache = th.prefill()
        main_pred, fc1, agree, nat, lp_eotr = [], [], [], None, None
        for r in sched:
            audio = f[r["audio"][1]] if r["audio"][0] == "feat" else sil
            tl, fl, cache = th.step(cache, audio, r["agent_in"], r["func_in"])
            if r["kind"] == "main":
                main_pred.append((int(tl.argmax()), int(fl.argmax())))
            elif r["kind"] == "fc1":
                fc1.append(int(fl.argmax()))
            elif r["forced_func"] is not None:
                agree.append(int(fl.argmax()) == r["forced_func"])
            else:
                lf = logp_rows(fl)
                nat, lp_eotr = int(lf.argmax()), round(float(lf[EOTR]), 3)
        hm = [t_ for t_ in range(s + 1) if main_pred[t_][0] != st["rec"].get(t_, PAD)]
        res = {"arm": arm, "scenario": sc, "schedule_sha": st["sha"], "nat_last": nat, "lp_EOTR_last": lp_eotr,
               "own_sotc_at_s": main_pred[s][1] == SOTC, "own_func_pad_before_s": all(p[1] == PAD for p in main_pred[:s]),
               "own_text_mismatch_0_s": len(hm), "fc1_match": [sum(p == c for p, c in zip(fc1, st["call"])), len(st["call"])],
               "fc2_forced_agree": [sum(agree), len(agree)], "contracts": {}}
        if SCEN[sc]["trace_arm"] == arm:
            res["V1"] = {"text_mismatch": hm[:20], "sotc_at_s": main_pred[s][1] == SOTC,
                         "call_match": fc1 == st["call"], "pass": not hm and main_pred[s][1] == SOTC and fc1 == st["call"]}
        tail = f[s + 1:s + 1 + a.n_free]        # continuation: the live main loop resumes on frame s+1 of the audio
        for pr in a.contracts.split(","):
            ta, fa, silent = parse_pair(pr)
            bc, pre = th.clone(cache), None
            if silent:
                tl0, fl0, bc = th.step(bc, sil, ta, fa)
                pre = frame_row(-1, tl0, fl0)
                ta, fa = pre["text"], pre["func"]
            rows = []
            # job 278: when the silent EOTR step itself emits SOTC the FC starts there; do not feed the call tokens
            # through real-audio main frames (decode the call on FC phase-1 inputs instead)
            for k in range(0 if (pre is not None and pre["func"] == SOTC) else a.n_free):
                tl, fl, bc = th.step(bc, tail[k], ta, fa)
                rows.append(frame_row(st["ff"] + k, tl, fl))
                ta, fa = rows[-1]["text"], rows[-1]["func"]
                if fa == SOTC:
                    break
            o = summarise(rows) if rows else {"first": None, "reply_bos": None, "new_call_sotc": None, "reply_ids": [],
                                              "max_sotc_margin": None}
            if pre is not None and pre["func"] == SOTC:
                o["first"], o["new_call_sotc"] = "new_call", "silent_step"
            if o["first"] == "new_call":
                o["new_call_ids"], o["new_call_text"] = decode_call(bc, sil)
            o["reply_text"] = th.decode(o.pop("reply_ids"))
            res["contracts"][pr] = {"silent_step": pre, "first_frame": rows[0] if rows else pre, "frames": rows[:6],
                                    "outcome": o}
        res["s"] = round(time.monotonic() - t, 1)
        return res

    costs = {}
    for i, (sc, arm) in enumerate(plan):
        need_cap = arm not in sils
        n_clips = len({sc_ for sc_, arm_ in plan if arm_ == arm})
        est = costs.get(sc, FIRST_CELL_S.get(sc, 45.0)) * 1.1 + (8.0 * n_clips if need_cap else 0.0)
        if time.monotonic() - T0 + est > a.budget_s:
            rep["skipped"] = [list(p) for p in plan[i:]]
            print("BUDGET: skipping", rep["skipped"], flush=True)
            break
        if need_cap:
            capture(arm)
        r = run_cell(arm, sc)
        costs[sc] = max(costs.get(sc, 0.0), r["s"])
        h5 = SCEN[sc]["h5"]
        if h5 and h5[0] == arm:
            c = r["contracts"].get(h5[1], {})
            o = c.get("outcome", {})
            r["H5"] = {"contract": h5[1], "new_call_sotc": o.get("new_call_sotc"), "expect_sotc": states[sc]["ff"] + SCEN[sc]["h5"][3],
                       "new_call_text": o.get("new_call_text"), "expect_tool": h5[2],
                       "pass": o.get("new_call_sotc") == states[sc]["ff"] + SCEN[sc]["h5"][3] and h5[2] in (o.get("new_call_text") or "")}
        rep["cells"].append(r)
        save()
        print(f"{sc:10} {arm:8} V1={r.get('V1', {}).get('pass')} H5={r.get('H5', {}).get('pass')} ({r['s']}s)", flush=True)
        for pr, c in r["contracts"].items():
            o = c["outcome"]
            print(f"   {pr:7} m0={c['first_frame']['sotc_margin']:+.2f} first={o['first']} sotc={o['new_call_sotc']} "
                  f"call={o.get('new_call_text', '')[:70]!r} reply={o['reply_text'][:70]!r}", flush=True)
    fe.restore()
    save()
    print("DONE", round(time.monotonic() - T0, 1), flush=True)


if __name__ == "__main__":
    main()
