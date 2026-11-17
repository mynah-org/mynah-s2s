"""G3b: the G3 session with the ASR activity-window fix, in two arms, plus the offline content check.

  arm B  reactive user (e1/reactive_user.py): each utterance waits for the end of the agent's audio + 0.5 s; the
         last one is a deliberate barge-in 1 s into the agent's real reply (audio-stop latency on audible audio)
  arm A  the fixed track as in G0 / G3 (isolates the fix vs G3)
Both: INT4 Thinker with the exact profile verified token-identical in G2 (mamba_host_pos, no_timer_sync,
thinker_graph), bf16 Nano (holding phrase, constrained calls), live ASR every 8 frames / 12 s window with the
activity window >= the stride, the Flash talker in its own process with the plain code-predictor loop (codes
identical in G1), the holding phrase not cut, the audio-aware controller. One engine / Nano / talker load; the
talker is reset between arms. B runs first; A only if the time left allows.
Content: per played agent turn of G0, G3, G3b-B, G3b-A, the offline evaluation ASR (OmniASR CTC 3B, its own venv,
v2/scripts/asr_eval.py) vs the turn's text, if the time left allows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_codes_check(proc, ref_path):  # pragma: no cover - GPU
    """The G1 texts through a running talker process, one segment at a time, all text available: codes vs the G1
    plain-loop codes, step / decode statistics, codec frames per second."""
    ref = json.load(open(ref_path))
    proc.reset()
    same, frames, gen_s = [], 0, 0.0
    for i, (text, rs) in enumerate(zip(ref["texts"], ref["fast"]["segments"])):
        t_send = time.monotonic_ns()
        proc.submit_raw({"op": "start", "epoch": 0, "turn": i + 1, "seg": 0, "text": text})
        proc.submit_raw({"op": "final", "epoch": 0, "turn": i + 1, "seg": 0})
        done, t_end = None, time.monotonic() + 60
        while done is None and time.monotonic() < t_end:
            for ev in proc.poll():
                if ev.get("ev") == "done" and ev["turn"] == i + 1:
                    done = ev
            time.sleep(0.005)
        same.append(done is not None and done["codes"] == rs.get("codes"))
        if done is not None:
            frames += done["frames"]
            gen_s += (done["t_ns"] - t_send) / 1e9
    proc.reset()
    st = proc.stats or {}
    xs = sorted(st.get("step_ms", []))
    return {"segments": len(same), "identical": sum(same), "per_segment_identical": same,
            "step_ms_p50": xs[len(xs) // 2] if xs else None,
            "step_ms_mean": round(sum(xs) / len(xs), 2) if xs else None, "frames": frames,
            "fps": round(frames / gen_s, 2) if gen_s else None, "stream_priority": st.get("stream_priority"),
            "cp_graph": st.get("cp_graph"), "ref_fast": {"step_ms_mean": ref["fast"]["step_ms"]["mean"],
                                                          "fps": ref["fast"]["fps"]}}


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flash-python", required=True)
    ap.add_argument("--flash-harness", required=True)
    ap.add_argument("--flash-root", required=True)
    ap.add_argument("--flash-voice", required=True)
    ap.add_argument("--listen-dir", required=True)
    ap.add_argument("--g0-it", required=True)
    ap.add_argument("--g3-it", required=True)
    ap.add_argument("--omni-python", required=True)
    ap.add_argument("--omni-script", required=True)
    ap.add_argument("--omni-cache", required=True)
    ap.add_argument("--budget-s", type=float, default=575.0)
    ap.add_argument("--arm-a-min-left-s", type=float, default=170.0)
    ap.add_argument("--omni-min-left-s", type=float, default=80.0)
    # G4 (all off = G3b behaviour)
    ap.add_argument("--tag", default="g3b", help="run-dir prefix of the arms; content item prefix = TAG upper-cased")
    ap.add_argument("--talker-prio", action="store_true", help="G4: talker on a highest-priority CUDA stream")
    ap.add_argument("--brief", action="store_true", help="G4: brief spoken replies (prompt + sentence-aware cap)")
    ap.add_argument("--fast-barge", action="store_true", help="G4: 2 evidence frames + ASR stride 4 while audible")
    ap.add_argument("--codes-ref", default="", help="G4: g1.json; the G1 texts through this talker, codes vs G1 fast")
    ap.add_argument("--content-run", action="append", default=[], help="NAME=RUN_IT_DIR for the content check")
    ap.add_argument("--single-content", action="store_true", help="G4: one content check at the end (all runs)")
    ap.add_argument("--cp-graph", action="store_true", help="G5: the code-predictor loop as one CUDA graph")
    ap.add_argument("--standalone-bench", action="store_true",
                    help="G5: before the engine loads, the G1 texts through the talker alone (codes vs G1, speed)")
    ap.add_argument("--stop-if-codes-differ", action="store_true",
                    help="G5: no session if the standalone codes are not identical to G1")
    a = ap.parse_args()
    T0 = time.monotonic()
    left = lambda: a.budget_s - (time.monotonic() - T0)            # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "multilingual"))
    sys.path.insert(0, str(lab / "agentbench"))
    import llm_text_bench as B2
    B2.NANO = Path(a.nano_dir)
    from m0_runner import set_model_cfg
    from mynah_v2.model.perf import enable
    from mynah_v2.model.quant import quantize_thinker_int4
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.comp_timer import CompTimer
    from e1 import g0_session
    from e1.content_eval import build_items, score_items
    from e1.fast import safe_no_audio
    from e1.flash_client import FlashProc
    from e1.live_asr import LiveASR
    from e1.runner import mem_available_gb
    from e1.tools_bench import POLICY, tool_schemas
    res = {"commit": os.environ.get("FRANK_COMMIT"), "mem_before_gb": round(mem_available_gb(), 1)}

    def save():
        (out / "g3b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def make_proc():
        p = FlashProc([a.flash_python, str(Path(__file__).resolve().parent / "flash_server.py"), "--harness",
                       a.flash_harness, "--root", a.flash_root, "--voice", a.flash_voice, "--lang", "it",
                       "--fast-cp", *(["--prio", "high"] if a.talker_prio else []),
                       *(["--cp-graph"] if a.cp_graph else [])], env, str(out / "flash_server.log"))
        p.start()
        res["flash_ready"] = p.wait_ready(240)
        return p
    proc = None
    if a.standalone_bench:                               # G5: the talker alone on the GPU
        proc = make_proc()
        res["standalone"] = run_codes_check(proc, a.codes_ref)
        save()
        print("STANDALONE", json.dumps(res["standalone"]), flush=True)
        if a.stop_if_codes_differ and res["standalone"]["identical"] != res["standalone"]["segments"]:
            proc.close()
            res["status"] = "STOP: codes differ from the plain loop"
            save()
            print("STOP codes differ", flush=True)
            return
    t0 = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}),
                          llm_cache=True)
    safe_no_audio(eng)
    res["int4_thinker"] = quantize_thinker_int4(eng)
    res["profile"] = {n: enable(n, eng) for n in ("mamba_host_pos", "no_timer_sync", "thinker_graph")}
    res["engine_load_s"] = round(time.monotonic() - t0, 1)
    res["mem_after_engine_gb"] = round(mem_available_gb(), 1)
    t0 = time.monotonic()
    tok, m, _ = B2.load("nano")
    res["nano_load_s"] = round(time.monotonic() - t0, 1)
    res["mem_after_nano_gb"] = round(mem_available_gb(), 1)
    if proc is None:
        proc = make_proc()
    res["mem_after_flash_gb"] = round(mem_available_gb(), 1)
    live = LiveASR(str(lab / "multilingual"), lang="it", att_right=3, window_s=12.0, every=8)
    res["mem_after_asr_gb"] = round(mem_available_gb(), 1)
    save()
    print("LOADS", json.dumps({k: v for k, v in res.items() if k.endswith(("_s", "_gb"))}), flush=True)
    if res["mem_after_asr_gb"] < 20:
        proc.close()
        res["status"] = "INCOMPLETE: memory"
        save()
        return
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    w = eng.wrapper
    set_model_cfg(w.model_cfg, "force_turn_taking", False)
    timer = CompTimer()
    pcm = getattr(w, "perception_cache_mgr", None)
    if pcm is not None and hasattr(pcm, "step"):
        timer._wrap(pcm, "step", "perception")
    timer._wrap(w.model.stt_model, "forward", "thinker_b1")
    timer._wrap(w, "_rnnt_step", "rnnt")
    listen = Path(a.listen_dir)
    res["arms"] = {}

    def run_arm(arm, reactive):
        live.reset(window_s=12.0, every=8)
        t0 = time.monotonic()
        r = g0_session.run(out, f"{a.tag}_{arm}", eng, tok, m, spec, POLICY["it"], Path(a.asr_dir), 52.0,
                           lab / "agentbench", tool_schemas, live, proc, timer=timer, prebuffer_s=0.32,
                           listen_dir=listen / arm, holding_cut="never", audio_aware=True, reactive=reactive,
                           keep_proc=True, brief=a.brief, fast_barge=a.fast_barge)
        g = r["g0"]
        res["arms"][arm] = {"status": "DONE", "session_s": round(time.monotonic() - t0, 1)}
        save()
        print(f"ARM {arm}", json.dumps({k: g[k] for k in ("frames", "loop_s", "loop_lag_s", "barge_in",
                                                          "first_audio_processing_clock", "reactive_waits")},
                                       default=str)[:2500], flush=True)
        print(f"ARM {arm} PLAYBACK", json.dumps({k: v for k, v in g["playback"].items() if k != "gap_list"},
                                                default=str)[:1200], flush=True)
        print(f"ARM {arm} TALKER", json.dumps({k: g["talker"][k] for k in ("step_ms", "decode_ms")}), flush=True)

    def content(tag):
        items = out / f"asr_items_{tag}"
        refs = {}
        extra = [(x.split("=", 1)[0], Path(x.split("=", 1)[1])) for x in a.content_run]
        T = a.tag.upper().replace("G3B", "G3b")
        for name, run in [("G0", Path(a.g0_it)), ("G3", Path(a.g3_it))] + extra + [(T + "B", out / f"{a.tag}_B"),
                                                                                   (T + "A", out / f"{a.tag}_A")]:
            if (run / "summary.json").exists() and (run / "talker.jsonl").exists() and \
                    not any(k.startswith(name + "_t") for k in res.get("content_done", [])):
                refs.update(build_items(run, items, name))
        if not refs:
            return
        (out / f"asr_refs_{tag}.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False))
        if left() < a.omni_min_left_s:
            res[f"content_{tag}"] = {"status": "INCOMPLETE", "why": f"time left {left():.0f} s"}
            save()
            print("CONTENT SKIPPED", tag, res[f"content_{tag}"], flush=True)
            return
        oenv = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        oenv["FAIRSEQ2_CACHE_DIR"] = a.omni_cache
        try:
            p = subprocess.run([a.omni_python, a.omni_script, str(items)], env=oenv, capture_output=True, text=True,
                               timeout=max(30, left() - 10))
            (out / f"asr_eval_{tag}.log").write_text(p.stdout[-20000:] + "\n" + p.stderr[-20000:])
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            res[f"content_{tag}"] = {"status": "INCOMPLETE", "why": "evaluation ASR failed or timed out"}
        else:
            sc = score_items(items, refs)
            (out / f"content_score_{tag}.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False))
            res[f"content_{tag}"] = {"status": "DONE", **{k: v for k, v in sc.items() if k != "items"}}
            res.setdefault("content_done", []).extend(refs)
        save()
        print("CONTENT", tag, json.dumps(res[f"content_{tag}"])[:1500], flush=True)

    def codes_check():
        res["codes_check"] = run_codes_check(proc, a.codes_ref)
        save()
        print("CODES", json.dumps(res["codes_check"]), flush=True)

    run_arm("B", True)                                   # owner: B first
    if not a.single_content:
        content("1")                                     # then the content check (G0, G3, ..., B)
    if left() >= a.arm_a_min_left_s:
        run_arm("A", False)
        if a.codes_ref and left() > 60:
            codes_check()
        content("2" if not a.single_content else "all")  # A's turns (or everything), if time is left
    else:
        res["arms"]["A"] = {"status": "INCOMPLETE", "why": f"time left {left():.0f} s < {a.arm_a_min_left_s}"}
        save()
        print("ARM A SKIPPED", res["arms"]["A"], flush=True)
        if a.single_content:
            content("all")
    proc.close()


if __name__ == "__main__":
    main()
