"""Pre-registered evaluation of the guided-opening A/B (CPU). See .work/multilingual/guided-open-design.md section 3
and the owner's precision of 2026-11-12.

S0 requires equivalence of:
- the injected token sequence;
- the controller state;
- the propagation of the injected tokens into the D-V2.1 KV cache, i.e. the Thinker's agent-text input.

It does NOT require logits or KV contents to be numerically identical to A: the acoustic front-end differs by
design.

Three outcomes are reported separately:
- tool call (S3);
- dialogue / turn-taking quality (S1, S2, latency);
- spoken audio (OmniASR, only when --asr is given).

Usage: guided_eval.py <session_dir> [--a-trace T] [--nemo-yaml Y] [--replay replay.json] [--asr asr_omni3b.json]
       [--out eval.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "multilingual"))
import guided_open as G  # noqa: E402

PAD, BOS, EOS = 12, 1, 2
UPTO = 94
REF_REPLY_FRAME = 232            # A and C-ridge: first agent BOS after the call (EN m03)
OK_REPLY_FRAME = REF_REPLY_FRAME + 25          # +2 s
PATHOLOGICAL_FRAME = REF_REPLY_FRAME + 62      # +5 s


def rows_by_t(rows, kind, fn=None):
    d = {}
    for r in rows:
        if r.get("kind") == kind and (fn is None or r.get("fn") == fn) and r.get("t") is not None:
            d.setdefault(r["t"], r)
    return d


def audio_activity(wav_path: Path, n_frames: int, rel_thr: float = 0.05) -> list[int]:
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(wav_path)
    x = x if x.ndim == 1 else x.mean(1)
    w = int(0.08 * sr)
    e = [float(np.sqrt(np.mean(x[i * w:(i + 1) * w] ** 2))) if (i + 1) * w <= len(x) else 0.0 for i in range(n_frames)]
    thr = max(e) * rel_thr if e and max(e) > 0 else 1.0
    return [i for i, v in enumerate(e) if v > thr]


def yaml_turn_cfg(nemo_yaml: Path) -> dict:
    import yaml
    s2s = yaml.safe_load(open(nemo_yaml))["s2s"]
    return {k: v for k, v in s2s.items() if not isinstance(v, (dict, list))}


def evaluate(sd: Path, a_trace: Path | None, nemo_yaml: Path | None, replay: dict | None, asr: dict | None) -> dict:
    from tenant_score import SPEC, score
    from agentbench import fact_spec
    from gold_v21 import hit
    summ = json.load(open(sd / "summary.json"))
    rows = [json.loads(x) for x in open(sd / "observe.jsonl")] if (sd / "observe.jsonl").exists() else []
    grows = [json.loads(x) for x in open(sd / "guided.jsonl")] if (sd / "guided.jsonl").exists() else []
    forcing = rows_by_t(rows, "forcing", "_apply_rnnt_turn_taking")
    logits = rows_by_t(rows, "logits")
    b0, b1 = summ["turn_bounds_s"][0]
    u_start, u_end = int(math.floor(b0 / 0.08 + 1e-9)), int(math.ceil(b1 / 0.08 - 1e-9))
    calls = summ.get("calls", [])
    call_f = calls[0]["model_step"] if calls else None
    out = {"session": sd.name, "opening": summ.get("opening", "native"), "fallback": summ.get("first_turn_fallback"),
           "user_frames": [u_start, u_end], "first_call_frame": call_f, "wall_s": summ.get("wall_s")}
    guided = summ.get("opening") == "guided"
    # ---------------------------------------------------------------- S0 validity (guided sessions only)
    if guided:
        toks = G.opener_from_trace(a_trace, upto=UPTO)
        gs = G.summarise(grows, toks)
        st95 = (forcing.get(UPTO + 1) or {}).get("rnnt") or {}
        s0 = {
            "1_injected_sequence": gs["injected"] == len(toks) and not gs["missing"] and not gs["errors"]
                                   and gs["post_equals_token"] and not gs["overridden_by_turn_logic"],
            "2_function_pad_on_opener": all(logits.get(t, {}).get("pred_func") == PAD for t in range(UPTO + 1)),
            "3_controller_state_95": st95.get("agent_speaking") is False and st95.get("first_turn") is False,
            "3b_kv_input_propagation": gs["kv_input_checked"] == len(toks) and gs["kv_input_ok"] == len(toks),
            "4_replay_0_mismatch": None if replay is None else replay.get("mismatches") == 0,
        }
        if nemo_yaml is not None:
            y = yaml_turn_cfg(nemo_yaml)
            tc = summ.get("turn_cfg") or {}
            diff = {k: (tc.get(k), y.get(k)) for k in tc if k in y and tc.get(k) != y.get(k)}
            s0["5_turn_cfg_equals_yaml"] = not diff
            out["turn_cfg_diff"] = diff
        out["S0"] = s0
        out["S0_pass"] = all(v is True for v in s0.values())
        out["injector"] = {k: gs[k] for k in ("injected", "missing", "thinker_agrees", "kv_input_ok", "kv_input_bad")}
        out["state_95"] = st95
    # ---------------------------------------------------------------- dialogue / turn-taking (S1, S2, latency)
    lim = call_f if call_f is not None else max(forcing) + 1 if forcing else 0
    bos_rows = [(t, r) for t, r in sorted(forcing.items()) if r["post"] == BOS]
    agent_bos_before_call = [{"t": t, "how": "forced" if r["pre"] != BOS else "spontaneous"}
                             for t, r in bos_rows if UPTO < t < lim]
    forced_bos_any_before_call = [t for t, r in bos_rows if t < lim and r["pre"] != BOS and
                                  not (guided and t <= UPTO)]
    act = audio_activity(sd / "output.wav", max(u_end + 1, 400)) if (sd / "output.wav").exists() else []
    overlap = [i for i in act if u_start <= i <= u_end]
    out["S1"] = {"agent_bos_95_to_call": agent_bos_before_call, "forced_bos_before_call": forced_bos_any_before_call,
                 "overlap_frames_inherited_le94": [i for i in overlap if i <= UPTO],
                 "overlap_frames_new_gt94": [i for i in overlap if i > UPTO]}
    out["S1_pass"] = not agent_bos_before_call and not forced_bos_any_before_call and not out["S1"]["overlap_frames_new_gt94"]
    conf = next((t for t, r in sorted(forcing.items()) if (r.get("rnnt") or {}).get("speech_confirmed")), None)
    s2_rows = [(t, forcing[t]["rnnt"]) for t in range(conf, u_end + 1) if t in forcing] if conf is not None else []
    out["S2"] = {"first_speech_confirmed": conf, "frames_checked": len(s2_rows),
                 "violations": [t for t, st in s2_rows if st.get("agent_speaking") or st.get("first_turn")][:10]}
    out["S2_pass"] = conf is not None and bool(s2_rows) and not out["S2"]["violations"]
    reply = next((t for t, r in bos_rows if call_f is not None and t > call_f), None)
    out["latency"] = {"first_reply_bos_frame": reply, "s": None if reply is None else round(reply * 0.08, 2),
                      "vs_ref_frames": None if reply is None else reply - REF_REPLY_FRAME,
                      "ok": reply is not None and reply <= OK_REPLY_FRAME,
                      "pathological": reply is None or reply > PATHOLOGICAL_FRAME}
    # ---------------------------------------------------------------- tool call (S3) + internal-text reply (S4a)
    spec = json.load(open(SPEC))
    sc = {s["id"]: s for s in spec["scenarios"]}[summ["scenario"]]
    prompt_text = spec.get("system_prompt", "") + " " + json.dumps(spec.get("tenant", {}))
    sco = score(sc, summ, prompt_text)
    out["calls"] = [{"name": c["name"], "args": c["args"], "model_step": c.get("model_step"),
                     "result_status": (c.get("result") or {}).get("status") if isinstance(c.get("result"), dict) else None}
                    for c in calls]
    out["slot_provenance"] = sco["slot_provenance"]
    out["redundant_calls"] = sco["redundant_calls"]
    out["score_checks"] = sco["checks"]
    out["score_pass"] = sco["pass"]
    out["agent_text"] = summ.get("agent_text")
    if summ["scenario"] == "m03":
        ck = sco["checks"]
        out["S3_pass"] = (len(calls) == 1 and calls[0]["name"] == "search_catalog" and ck.get("catalog_query_ok", False)
                          and ck.get("slots_grounded", False) and ck.get("no_redundant_call", False))
        out["S4_internal_text"] = bool(ck.get("says:Fjordby")) and bool(ck.get("says:499"))
    # ---------------------------------------------------------------- spoken audio (separate evidence)
    if asr is not None:
        sess = [x for x in asr.get("sessions", []) if x.get("scenario") == sd.name]   # asr_dirs.py output
        out["asr_segments"] = [[g["start_s"], g["end_s"], g["text"]] for x in sess for g in x.get("segments", [])]
        text = " ".join(g[2] for g in out["asr_segments"])
        out["asr_text"] = text
        if summ["scenario"] == "m03":
            out["S4_audio"] = bool(hit(text, fact_spec("Fjordby"))) and bool(hit(text, fact_spec("499")))
    if summ["scenario"] == "m03" and guided:
        out["S1_S4_pass"] = bool(out["S1_pass"] and out["S2_pass"] and out["S3_pass"] and out["S4_internal_text"]
                                 and out["latency"]["ok"] and out.get("S4_audio", None) is not False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="+")
    ap.add_argument("--a-trace", default=None)
    ap.add_argument("--nemo-yaml", default=None)
    ap.add_argument("--replay", default=None, help="test_guided_open.py --validate-run output (list)")
    ap.add_argument("--asr", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {r["session"]: r for r in json.load(open(a.replay))} if a.replay else {}
    asr = json.load(open(a.asr)) if a.asr else None
    res = [evaluate(Path(d), Path(a.a_trace) if a.a_trace else None, Path(a.nemo_yaml) if a.nemo_yaml else None,
                    rep.get(Path(d).name), asr) for d in a.session_dir]
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
