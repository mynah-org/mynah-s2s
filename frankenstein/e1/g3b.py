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


def run_codes_check(proc, ref_path, audio_dir=None):  # pragma: no cover - GPU
    """The G1 texts through a running talker process, one segment at a time, all text available: codes vs the G1
    plain-loop codes, step / decode statistics, codec frames per second."""
    ref = json.load(open(ref_path))
    proc.reset()
    same, frames, gen_s, codes_all = [], 0, 0.0, []
    for i, (text, rs) in enumerate(zip(ref["texts"], ref["fast"]["segments"])):
        t_send = time.monotonic_ns()
        proc.submit_raw({"op": "start", "epoch": 0, "turn": i + 1, "seg": 0, "text": text})
        proc.submit_raw({"op": "final", "epoch": 0, "turn": i + 1, "seg": 0})
        done, t_end, pcm = None, time.monotonic() + 60, []
        while done is None and time.monotonic() < t_end:
            for ev in proc.poll():
                if ev.get("ev") == "done" and ev["turn"] == i + 1:
                    done = ev
                elif ev.get("ev") == "audio" and ev["turn"] == i + 1:
                    pcm.append(ev["pcm"])
            time.sleep(0.005)
        same.append(done is not None and done["codes"] == rs.get("codes"))
        if done is not None:
            codes_all.append(done["codes"])
        if audio_dir is not None and pcm:
            import numpy as np
            import soundfile as sf
            d = Path(audio_dir) / f"seg{i + 1:02d}"
            d.mkdir(parents=True, exist_ok=False)
            sf.write(d / "output.wav", np.concatenate(pcm), 24000, subtype="PCM_16")
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
            "cp_graph": st.get("cp_graph"), "step_graph": st.get("step_graph"),
            "ref_fast": {"step_ms_mean": ref["fast"]["step_ms"]["mean"], "fps": ref["fast"]["fps"]},
            "codes": codes_all}


def warm_graph(graph, spec, tok, m, B2):  # pragma: no cover - GPU
    """Capture the decode graph's buckets before any session (no capture while the frame loop runs)."""
    import torch
    from e1.nano_worker import NanoWorker
    wk = NanoWorker(tok, m, spec["system_prompt"] + " /no_think", [], {}, None, B2.parse_toolcall, max_new=8)
    wk.graph = graph
    out = {}
    with torch.inference_mode():
        for n in (10, 1300):
            ids = wk._ids([{"role": "user", "content": "parola " * n}])
            cache, h = wk._prefill(ids)
            for _ in range(3):
                h = wk._step(wk.llm.embeddings(torch.tensor([[1000]], device=wk.dev)), cache)
            out[str(len(ids))] = True
    out["stats"] = graph.stats()
    return out


def suite_check(a, spec, tok, m, B2, graph=None):  # pragma: no cover - GPU
    """G7: the G6 tool suite with the INT4 decode-only text LLM, compared to G6's bf16 reference rows."""
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from tenant_backend import TenantBackend
    from e1.g0_session import BRIEF_IT
    from e1.g6_bench import compare, run_suite, suite_items
    from e1.nano_worker import NanoWorker
    from e1.tools_bench import POLICY, tool_schemas
    brief = {"it": BRIEF_IT, "en": "Answer as on a phone call: at most two short sentences, no lists, no markdown."}
    ref = json.load(open(a.suite_ref))["bf16"]

    def make_worker(lang):
        said = []
        be = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        be._said = said
        reg = ToolRegistry()
        for t in spec["tools"]:
            reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
        bound = reg.bind(ToolLedger("g7", None), lambda: 0, lambda: None, timeout_s=None)
        system = spec["system_prompt"] + "\n" + POLICY[lang] + "\n" + brief[lang] + " /no_think"
        wk = NanoWorker(tok, m, system, tool_schemas(reg), bound, be, B2.parse_toolcall, max_new=60,
                        tool_step=True, constrained=True)
        wk.reply_soft_cap = 24
        wk.graph, wk.graph_reply_only = graph, graph is not None
        return wk
    r = run_suite(make_worker, suite_items(spec, a.suite_session or None),
                  "int4_graph_reply" if graph is not None else "int4")
    return {"per_token_ms_suite": r["per_token_ms_mean"], "vs_g6_bf16": compare(ref, r)}


def g6(a, res, save, spec, tok, m, B2, lab):  # pragma: no cover - GPU
    """G6 a) + b) + d): suite per Nano variant, per-token decode ms, NVFP4 probe. Returns the graph to use (or None)
    and leaves Int4DecodeOnly.active as decided."""
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from tenant_backend import TenantBackend
    from e1.g0_session import BRIEF_IT
    from e1.g6_bench import compare, nvfp4_probe, per_token_bench, run_suite, suite_items
    from e1.int4_dispatch import Int4DecodeOnly, quantize_int4_decode_only
    from e1.nano_graph import NanoGraph
    from e1.nano_worker import NanoWorker
    from e1.tools_bench import POLICY, tool_schemas
    brief = {"it": BRIEF_IT, "en": "Answer as on a phone call: at most two short sentences, no lists, no markdown."}

    def mk(graph):
        def make_worker(lang):
            said = []
            be = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
            be._said = said
            reg = ToolRegistry()
            for t in spec["tools"]:
                reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
            bound = reg.bind(ToolLedger("g6", None), lambda: 0, lambda: None, timeout_s=None)
            system = spec["system_prompt"] + "\n" + POLICY[lang] + "\n" + brief[lang] + " /no_think"
            wk = NanoWorker(tok, m, system, tool_schemas(reg), bound, be, B2.parse_toolcall, max_new=60,
                            tool_step=True, constrained=True)
            wk.reply_soft_cap, wk.graph = 24, graph
            return wk
        return make_worker
    items = suite_items(spec, a.suite_session or None)
    out = {"items": len(items)}
    r0 = run_suite(mk(None), items, "bf16")
    out["bf16"] = {"per_token_ms_suite": r0["per_token_ms_mean"], **per_token_bench(mk(None)("it"))}
    out["int4_wrap"] = quantize_int4_decode_only(m.backbone)
    r1 = run_suite(mk(None), items, "int4")
    out["int4"] = {"per_token_ms_suite": r1["per_token_ms_mean"], **per_token_bench(mk(None)("it")),
                   "vs_bf16": compare(r0, r1)}
    graph = None
    try:
        graph = NanoGraph(m.backbone, lmax=4096)
        w0 = mk(graph)("it")
        out["int4_graph_warm"] = per_token_bench(w0)                     # captures bucket 1024
        import torch
        with torch.inference_mode():                                    # and the next bucket, before the session
            ids = w0._ids([{"role": "user", "content": "parola " * 1300}])
            cache, h = w0._prefill(ids)
            for _ in range(3):
                h = w0._step(w0.llm.embeddings(torch.tensor([[1000]], device=w0.dev)), cache)
        out["warm_ctx_tokens"] = len(ids)
        r2 = run_suite(mk(graph), items, "int4_graph")
        out["int4_graph"] = {"per_token_ms_suite": r2["per_token_ms_mean"], **per_token_bench(mk(graph)("it")),
                             "vs_bf16": compare(r0, r2), "vs_int4": compare(r1, r2), "graph_stats": graph.stats()}
    except Exception as e:  # noqa: BLE001
        out["int4_graph"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        graph = None
    use_int4 = out["int4"]["vs_bf16"]["tools_ok"]
    use_graph = graph is not None and use_int4 and out["int4_graph"].get("vs_bf16", {}).get("tools_ok", False)
    Int4DecodeOnly.active = use_int4
    out["decision"] = {"int4_decode": use_int4, "nano_graph": use_graph}
    try:
        out["nvfp4"] = nvfp4_probe(m, a.wo_gemv or None)
    except Exception as e:  # noqa: BLE001
        out["nvfp4"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    res["g6"] = out
    save()
    (Path(a.out) / "g6_suite.json").write_text(json.dumps({"bf16": r0, "int4": r1,
                                                            "int4_graph": locals().get("r2")}, ensure_ascii=False))
    print("G6", json.dumps({k: v for k, v in out.items() if k != "nvfp4"}, default=str)[:3000], flush=True)
    print("NVFP4", json.dumps(out.get("nvfp4"), default=str)[:1500], flush=True)
    return graph if use_graph else None


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
    ap.add_argument("--g6-bench", action="store_true",
                    help="G6: Nano bf16 vs INT4 decode-only vs INT4 + decode graph on the tool suite, NVFP4 probe; "
                         "the session then uses what passed")
    ap.add_argument("--suite-session", default="", help="G6: summary.json whose tool prompts join the suite")
    ap.add_argument("--wo-gemv", default="", help="G6: copy of the web UI tree's Triton GEMV file (NVFP4 probe)")
    ap.add_argument("--ids-per-frame", type=int, default=1, help="G6: reply ids handed to the talker per frame")
    ap.add_argument("--nano-int4", action="store_true", help="G7: INT4 decode-only text LLM (as G6 chose)")
    ap.add_argument("--gpu-gate", action="store_true", help="G7: time-slice frames and text-LLM work (GpuGate)")
    ap.add_argument("--gate-budget-ms", type=float, default=0.0, help="G7b: text-LLM time after each frame")
    ap.add_argument("--nano-graph-reply", action="store_true",
                    help="G8: Nano decode graph for the reply step only (tool step eager); suite-checked")
    ap.add_argument("--suite-ref", default="", help="G7: g6_suite.json; the INT4 suite again, compared to its bf16")
    ap.add_argument("--thinker-ref", default="", help="G7: frames.jsonl of a fixed-track arm; Thinker ids compared")
    ap.add_argument("--standalone-bench", action="store_true",
                    help="G5: before the engine loads, the G1 texts through the talker alone (codes vs G1, speed)")
    ap.add_argument("--talker-variants", default="",
                    help="G9: comma list of talker variants benched alone before the engine loads "
                         "(cpgraph, eagerstatic, stepgraph); per segment audio kept for the content check")
    ap.add_argument("--talker-session", default="cpgraph", help="G9: talker variant for the sessions (or 'auto')")
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

    VARIANT = {"cpgraph": ["--cp-graph"], "eagerstatic": ["--step-eager-static"], "stepgraph": ["--step-graph"]}

    def make_proc(variant=None):
        extra = VARIANT[variant] if variant else (["--cp-graph"] if a.cp_graph else [])
        p = FlashProc([a.flash_python, str(Path(__file__).resolve().parent / "flash_server.py"), "--harness",
                       a.flash_harness, "--root", a.flash_root, "--voice", a.flash_voice, "--lang", "it",
                       "--fast-cp", *(["--prio", "high"] if a.talker_prio else []), *extra], env,
                      str(out / f"flash_server_{variant or 'session'}.log"))
        p.start()
        res["flash_ready"] = p.wait_ready(240)
        return p
    proc = None
    if a.talker_variants:                                # G9: talker variants alone on the GPU, before the engine
        res["talker_variants"] = {}
        for v in a.talker_variants.split(","):
            pv = make_proc(v)
            r = run_codes_check(pv, a.codes_ref, audio_dir=out / "talker_items" / v)
            pv.close()
            res["talker_variants"][v] = r
            print("TALKER", v, json.dumps({k: x for k, x in r.items() if k not in ("codes",)})[:900], flush=True)
        tv = res["talker_variants"]
        if "eagerstatic" in tv and "stepgraph" in tv:
            same = [x == y for x, y in zip(tv["eagerstatic"]["codes"], tv["stepgraph"]["codes"])]
            res["stepgraph_vs_eagerstatic"] = {"segments": len(same), "identical": sum(same)}
            print("STEPGRAPH_VS_EAGERSTATIC", res["stepgraph_vs_eagerstatic"], flush=True)
        try:                                             # content of each variant's audio (evaluation ASR)
            from e1.g23 import wer_cer
            dirs = [str(out / "talker_items" / v) for v in tv]
            oenv = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            oenv["FAIRSEQ2_CACHE_DIR"] = a.omni_cache
            subprocess.run([a.omni_python, a.omni_script, *dirs], env=oenv, capture_output=True, text=True, timeout=300)
            texts = json.load(open(a.codes_ref))["texts"]
            res["talker_content"] = {}
            for v in tv:
                rows = json.load(open(out / "talker_items" / v / "asr_omni3b.json"))["sessions"]
                hyp = {r["scenario"]: " ".join(x["text"] for x in r["segments"]) for r in rows}
                ref_all = " ".join(texts[int(k[3:]) - 1] for k in sorted(hyp))
                res["talker_content"][v] = wer_cer(ref_all, " ".join(hyp[k] for k in sorted(hyp)))
            print("TALKER_CONTENT", json.dumps(res["talker_content"]), flush=True)
        except Exception as e:  # noqa: BLE001
            res["talker_content"] = {"error": repr(e)[:300]}
        if a.talker_session == "auto":
            ok = res.get("stepgraph_vs_eagerstatic", {}).get("identical") == len(tv.get("stepgraph", {}).get("codes", [1]))
            # G9: the static-cache step graph was consistent but SLOWER than the dynamic cache + code-predictor graph
            # (attention over the whole static buffer); it only wins if it is also faster
            ok = ok and (tv.get("stepgraph", {}).get("step_ms_mean") or 1e9) < (tv.get("cpgraph", {}).get("step_ms_mean")
                                                                              or 0)
            a.talker_session = "stepgraph" if ok else "cpgraph"
        res["talker_session"] = a.talker_session
        for v in tv.values():
            v.pop("codes", None)
        save()
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
        # G5: a talker process that already ran (CUDA context, caching allocator, graph pool) made the engine load
        # fail with a CUDA OOM on this unified-memory host; load the engine first, then a fresh talker (G4 order)
        proc.close()
        proc = None
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
        proc = make_proc(a.talker_session if a.talker_variants else None)
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
    nano_graph = None
    gate = None
    if a.nano_int4:
        from e1.int4_dispatch import quantize_int4_decode_only
        res["int4_decode_only"] = quantize_int4_decode_only(m.backbone)
    if a.nano_graph_reply:
        import torch
        from e1.nano_graph import NanoGraph
        nano_graph = NanoGraph(m.backbone, lmax=4096)
        res["nano_graph_warm"] = warm_graph(nano_graph, spec, tok, m, B2)
    if a.suite_ref:
        res["suite"] = suite_check(a, spec, tok, m, B2, graph=nano_graph)
        save()
        print("SUITE", json.dumps(res["suite"])[:1500], flush=True)
    if a.gpu_gate:
        import torch
        from e1.gpu_gate import GpuGate
        gate = GpuGate(sync=torch.cuda.synchronize, unit_budget_ms=a.gate_budget_ms)
    if a.g6_bench:
        nano_graph = g6(a, res, save, spec, tok, m, B2, lab)
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
                           keep_proc=True, brief=a.brief, fast_barge=a.fast_barge, nano_graph=nano_graph,
                           ids_per_frame=a.ids_per_frame, gate=gate, graph_reply_only=a.nano_graph_reply)
        if gate is not None:
            res["arms"][arm + "_gate"] = dict(gate.stats)
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

    def thinker_check():
        ref = [json.loads(x) for x in open(a.thinker_ref)]
        new = [json.loads(x) for x in open(out / f"{a.tag}_A" / "frames.jsonl")]
        n = min(len(ref), len(new))
        same = sum(r["predicted"] == x["predicted"] for r, x in zip(ref[:n], new[:n]))
        res["thinker_identity"] = {"frames": n, "identical_predicted": same,
                                   "first_diff": next((i for i in range(n) if ref[i]["predicted"] != new[i]["predicted"]),
                                                      None)}
        save()
        print("THINKER", json.dumps(res["thinker_identity"]), flush=True)

    run_arm("B", True)                                   # owner: B first
    if not a.single_content:
        content("1")                                     # then the content check (G0, G3, ..., B)
    if left() >= a.arm_a_min_left_s:
        run_arm("A", False)
        if a.thinker_ref:
            thinker_check()
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
