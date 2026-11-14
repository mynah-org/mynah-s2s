"""Stream WAV file(s) through the V2 session runtime on the NVIDIA reference engine (native PyTorch).

Outputs per input (under --out/<stem>/): trace.jsonl, summary.json, agent.wav (22.05 kHz, as generated, no splicing),
combined.wav (user left / agent right, wall-clock aligned on the engine timeline).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import soundfile as sf
import torch
import torchaudio.functional as AF

from mynah_v2 import presets
from mynah_v2.agent.controller import Controller
from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolLedger, ToolRegistry, make_orders_db, orders_tool
from mynah_v2.model import tokens as T
from mynah_v2.model.reference import DEFAULT_NEMO, ReferenceEngine, build_cfg
from mynah_v2.runtime.fc_timeline import check_fc_timeline
from mynah_v2.runtime.session import EnhancedOptions, Mode, Session, run_paced
from mynah_v2.tracing.events import Tracer

sys.path.insert(0, str(DEFAULT_NEMO / "examples/speechlm2"))


def load_16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy()).mean(0)
    return AF.resample(wav.unsqueeze(0), sr, 16000)[0] if sr != 16000 else wav


def read_system_message(arg: str | None) -> str | None:
    """--system-message value: literal text, or @path to a UTF-8 text file. None keeps NVIDIA's default message."""
    if not arg:
        return None
    return Path(arg[1:]).read_text(encoding="utf-8").strip() if arg.startswith("@") else arg


def render_prompt(tools: list[dict], system_message: str | None = None) -> str:
    import offline_voicechat_fc_infer as fcmod
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import render_fc_system_prompt
    return render_fc_system_prompt(str(fcmod.DEFAULT_TEMPLATE), system_message or fcmod.DEFAULT_SYSTEM_MESSAGE, tools)


def fc_timeline_from_records(records) -> dict:
    """Rebuild the FC sub-timeline from the reference fast-path records and check reference invariants."""
    out = []
    recs = sorted(records, key=lambda r: (r.start_frame, r.phase))
    i = 0
    while i < len(recs):
        p1 = recs[i]
        p2 = recs[i + 1] if i + 1 < len(recs) and recs[i + 1].phase == 2 else None
        # phase 1: _run_fc_async_steps writes the call tokens and EOTC on frames [start, start+steps]. SOTC is either
        # the first of them (measured: en_s2/en_s4, start 101, func[0]=SOTC) or was predicted by the regular step at
        # start-1 (measured: NVIDIA FC sample SOTC 96/start 97; en_s3). Prepend the preceding frame only if it is SOTC.
        lead = p1.pre_func == T.SOTC_ID and (not p1.func or p1.func[0] != T.SOTC_ID)
        pre_t = [p1.pre_text] if lead else []
        pre_f = [p1.pre_func] if lead else []
        text = pre_t + list(p1.text[:p1.steps + 1])
        func = pre_f + list(p1.func[:p1.steps + 1])
        forced = [False] * len(func)
        if p2 is not None:
            n_resp = max(0, p2.steps - 1)
            text += p2.text[1:p2.steps + 1]
            func += p2.func[1:p2.steps + 1]
            forced += [True] * n_resp + [False] * (len(p2.func[1:p2.steps + 1]) - n_resp)
        rep = check_fc_timeline(text, func, forced_response=forced)
        ms = lambda r: round((r.t_end_ns - r.t_start_ns) / 1e6, 1) if r is not None and r.t_end_ns else None
        out.append({"start_frame": p1.start_frame, "phase1_steps": p1.steps, "phase2_steps": p2.steps if p2 else None,
                    # wall time per phase (FC thread): decode incl. on-hold TTS; gap = phase-1 end → phase-2 start
                    # (tool execution + hand-off); per-step = decode cost per FC token
                    "phase1_ms": ms(p1), "phase2_ms": ms(p2),
                    "gap_ms": round((p2.t_start_ns - p1.t_end_ns) / 1e6, 1) if p2 is not None and p1.t_end_ns else None,
                    "t_p1_start_ns": p1.t_start_ns, "t_end_ns": (p2 or p1).t_end_ns,
                    "func_ids": func, "text_ids": text, "violations": rep.violations,
                    "spans": [vars(s) for s in rep.spans]})
        i += 2 if p2 else 1
    return {"calls": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc-ctx-fix", default="auto", choices=["auto", "on", "off", "eotr_step"],
                    help="FC hand-back fix (selective): auto = ON for the official VoiceChat checkpoint only; off = rollback; "
                         "eotr_step = EXPERIMENT, the training layout's inserted EOTR slot (explicit only)")
    ap.add_argument("--strict", action="store_true",
                    help="fail if a requested optimisation cannot be enabled (default: continue without it, WARNING)")
    ap.add_argument("--preset", default="release", choices=list(presets.SINGLE),
                    help="release (default): enhanced mode + the fastest gated single-session optimisations (CUDA graphs, "
                         "int8 Thinker MLP, fused EarTTS norm, codec, RVQ cache, fast attention); release-exact: the "
                         "same with bit-exact attention; reference: NVIDIA's reference step. Explicit flags override it")
    ap.add_argument("--mode", default=None, choices=["reference", "enhanced"])
    ap.add_argument("--pace", default="realtime", choices=["realtime", "fast"])
    ap.add_argument("--tail-silence", type=float, default=6.0)
    ap.add_argument("--tools", default="orders,random", help="comma list: orders,random")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--system-message", default=None,
                    help="persona / instructions placed in NVIDIA's FC prompt template together with the tool schemas "
                         "(text, or @file.txt); default: NVIDIA's default system message")
    ap.add_argument("--tool-delay", type=float, default=0.0)
    ap.add_argument("--call-guard", action="store_true",
                    help="EXPERIMENT: per-session call registry + admission control in front of the executor "
                         "(mynah_v2.agent.call_guard); default off")
    ap.add_argument("--override", action="append", default=[], help="cfg key=value (OmegaConf dotlist)")
    presets.add_determinism_args(ap)
    ap.add_argument("--no-llm-cache", action="store_true",
                    help="diagnostic: NVIDIA's shipped no-cache Thinker (full-history recompute per step); "
                         "default is the V2 incremental Thinker (mynah_v2.model.llm_cache, gate scripts/cache_parity.py)")
    ap.add_argument("--tool-timeout", type=float, default=None,
                    help="ENHANCED: bounded tool wait in seconds (explicit timeout error to the model); "
                         "REFERENCE always keeps the pipeline's own timeout")
    ap.add_argument("--on-hold", action="store_true",
                    help="per-tool on-hold messages from the registry acks via the reference knob "
                         "s2s.fc_on_hold_messages_path (Layer 1 of the pipeline's on-hold state machine)")
    ap.add_argument("--fc-interrupt", default="",
                    help="test hook 'quit:S[:N]' or 'abort:S[:N]' (N = only the first N FC threads per item run): S seconds after an FC background thread starts, set the "
                         "pipeline's own quit_async_event / abort_event for it (deterministic stand-in for a barge-in "
                         "during FC, which the offline harness does not trigger: no BOU onset is detected during FC)")
    ap.add_argument("--close-kind", default="normal", choices=["normal", "cancel", "force"],
                    help="normal = drain an in-flight call before closing; cancel/force = close right after the input")
    ap.add_argument("--unregistered", default="",
                    help="comma list of tools kept in the prompt but WITHOUT a handler (unknown-tool scenario)")
    ap.add_argument("--rehydrate-after", type=float, default=None,
                    help="ENHANCED: restart the model stream with a facts summary after this many seconds of context")
    ap.add_argument("--concat", action="store_true",
                    help="treat all WAVs as ONE session (concatenated with 1 s of silence between them)")
    ap.add_argument("--tts-pad-silence", action="store_true",
                    help="ENHANCED: NVIDIA's idle-silence substitution in the TTS (inference_force_speech_silence_on_pad)")
    ap.add_argument("--mute-idle", type=float, default=10.0,
                    help="ENHANCED idle output gate in seconds; <= 0 disables it")
    ap.add_argument("--pad-silence", type=float, default=0.0, help="append this many seconds of silence to each input")
    ap.add_argument("--replay-at-consumption", action="store_true",
                    help="ENHANCED: replay buffered user audio as soon as the result is consumed (old behaviour); "
                         "default holds it until the agent finishes its post-call answer")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="performance option: torch.backends.cudnn.benchmark (codec 48 -> 5 ms/chunk on GB10); changes "
                         "numerics slightly (measured: one turn onset shifted by 1 frame), so never for parity runs")
    ap.add_argument("--orders-json", default=None, help="SQLite seed rows [[order_id,status,date,tracking],...]")
    ap.add_argument("--seed", type=int, default=None, help="reseed python/numpy/torch before each input (reproducible sampling)")
    ap.add_argument("--exp", default=None, help="comma list of mynah_v2.model.perf experiments (ENHANCED only), applied "
                    "after --perf-profile; non-exact ones (thinker_w8, heads_w8) change the model")
    ap.add_argument("--perf-profile", default=None, choices=["optimized-exact", "optimized-graph"],
                    help="validated performance profile (mynah_v2.model.perf.PROFILES); ENHANCED only")
    ap.add_argument("--thinker-int4", action="store_true",
                    help="EXPERIMENT: int4 weight-only Thinker (mynah_v2.model.quant); refused in REFERENCE mode")
    ap.add_argument("--int4-group", type=int, default=128, choices=[32, 64, 128])
    ap.add_argument("--int4-clip", action="store_true", help="per-group clipping search (MSE)")
    a = ap.parse_args()
    preset_info = presets.fill(a, presets.SINGLE, a.preset)
    print("PRESET", a.preset, preset_info, flush=True)
    if a.thinker_int4 and a.mode == "reference":
        ap.error("--thinker-int4 is experimental and never allowed in REFERENCE mode")
    if a.pace == "fast" and a.tools:
        ap.error("--pace fast is invalid with tools: the reference FC fast path runs on wall time while generate_step "
                 "returns at once, so unpaced input/drain closes sessions mid-call (golden v1 incident, 2026-11-11)")
    if (a.perf_profile or a.exp) and a.mode == "reference":
        ap.error("--perf-profile / --exp are never applied in REFERENCE mode")
    if a.cudnn_benchmark:
        import torch as _t
        _t.backends.cudnn.benchmark = True
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    db = make_orders_db(out_root / "orders.sqlite",
                        **({"rows": [tuple(r) for r in json.loads(Path(a.orders_json).read_text())]} if a.orders_json else {}))
    reg = ToolRegistry()
    names = a.tools.split(",") if a.tools else []
    if "orders" in names:
        reg.register("get_order_status", orders_tool(db, a.tool_delay),
                     "Get the status, delivery date and tracking code of a customer order by its order id.",
                     ORDERS_SCHEMA, ack_message="Let me check that order for you.")
    if "random" in names:
        import offline_voicechat_fc_infer as fcmod
        reg.schemas += fcmod.DEFAULT_TOOLS  # handler = NVIDIA builtin generate_random_number
    overrides = presets.determinism_overrides(a)
    for kv in a.override:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    if a.on_hold and reg.ack_messages:
        oh = out_root / "on_hold_messages.json"
        oh.write_text(json.dumps(reg.on_hold_messages(), indent=1))
        overrides["s2s.fc_on_hold_messages_path"] = str(oh)
    cfg = build_cfg(output_dir=str(out_root / "_nemo"), overrides=overrides)

    state = {"session": None}
    ledgers = {}

    def turn_id():
        s = state["session"]
        return s.turn_id if s else 0

    def step():
        s = state["session"]
        return s.model_step if s else None

    eng = ReferenceEngine(cfg, llm_cache=not a.no_llm_cache, fc_ctx_fix=a.fc_ctx_fix)
    det_info = presets.apply_determinism(a)
    cache_info = eng.llm_cache_info
    quant_info = None
    if a.thinker_int4:
        from mynah_v2.model.quant import quantize_thinker_int4
        quant_info = quantize_thinker_int4(eng, max_group=a.int4_group, clip=a.int4_clip)
        print("THINKER_QUANT", quant_info, flush=True)
    perf_info = None
    if a.perf_profile or a.exp:
        perf_info = presets.apply(eng, a.perf_profile, a.exp, strict=a.strict)
        print("PERF_PROFILE", perf_info, flush=True)
    print(f"engine loaded in {eng.load_s:.1f}s, chunk={eng.chunk_samples}", flush=True)
    prompt = None if a.no_prompt else render_prompt(reg.schemas, read_system_message(a.system_message))
    summary_all = []
    items = [(Path(w).stem, w, load_16k(w)) for w in a.wavs]
    if a.pad_silence > 0:
        items = [(st, w, torch.cat([wv, torch.zeros(int(a.pad_silence * 16000))])) for st, w, wv in items]
    if a.concat:
        gap = torch.zeros(16000)
        joined = [x for _, _, wv in items for x in (wv, gap)][:-1]
        items = [("+".join(st for st, _, _ in items), ",".join(w for _, w, _ in items), torch.cat(joined))]
    for stem, w, wav in items:
        od = out_root / stem
        od.mkdir(parents=True, exist_ok=True)
        tr = Tracer(od / "trace.jsonl", session_id=stem)
        led = ToolLedger(stem, tr)
        ledgers[stem] = led
        bound = reg.bind(led, turn_id, step, timeout_s=a.tool_timeout if a.mode == "enhanced" else None)
        session_tools = dict(bound)
        if "random" in names:  # wrap the NVIDIA builtin with the ledger too (same signature)
            builtin = eng.pipeline._tool_generate_random_number
            r2 = ToolRegistry()
            r2.register("generate_random_number", builtin, "", {})
            session_tools.update(r2.bind(led, turn_id, step, timeout_s=a.tool_timeout if a.mode == "enhanced" else None))
        eng.bind_session_tools(session_tools)              # unbound by engine.close() at session end
        for name in filter(None, a.unregistered.split(",")):
            eng.pipeline.tool_registry.pop(name, None)     # schema stays in the prompt, executor finds no handler
        agent_chunks, t_audio = [], []
        t0 = time.monotonic()
        enh = (EnhancedOptions(rehydrate_after_s=a.rehydrate_after, tts_pad_silence=a.tts_pad_silence,
                               replay_after_answer=not a.replay_at_consumption,
                               mute_idle_s=a.mute_idle if a.mute_idle > 0 else None)
               if a.mode == "enhanced" else None)
        sess = Session(eng, tr, Mode(a.mode), enh,
                       on_audio=lambda x: (agent_chunks.append(x.numpy()), t_audio.append(time.monotonic() - t0)),
                       controller=Controller(prompt, ledger=led, tools=dict(eng.pipeline.tool_registry)))
        state["session"] = sess
        if a.seed is not None:
            import random
            random.seed(a.seed)
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
        guard, guard_un = None, None
        if a.call_guard:                       # EXPERIMENT: refusals answer the model, the backend never sees them
            from mynah_v2.agent.call_guard import CallGuard, install
            guard = CallGuard(session_id=stem, ledger=led)
            _emit = tr.emit

            def _guard_emit(ev, _emit=_emit, _g=guard, **kw):   # caller speech (RNN-T user text) = new input epoch
                if ev == "asr.generated":
                    _g.on_user_text(kw.get("text", ""))
                return _emit(ev, **kw)
            tr.emit = _guard_emit
            guard_un = install(eng, guard, tracer=tr)
        sess.open(prompt)
        stop_hook = None
        if a.fc_interrupt:
            import threading
            parts = a.fc_interrupt.split(":")
            kind, sec, limit = parts[0], float(parts[1]), int(parts[2]) if len(parts) > 2 else 10**9
            stop_hook = threading.Event()

            def hook(kind=kind, sec=sec, stop=stop_hook, fired=set(), limit=limit):
                seen = {}
                while not stop.wait(0.02):
                    for sid, bg in list(getattr(eng.pipeline, "_fc_async_bg", {}).items()):
                        th = bg.get("thread")
                        if th is None or id(th) in fired or len(fired) >= limit:
                            continue
                        t0 = seen.setdefault(id(th), time.monotonic())
                        if time.monotonic() - t0 >= sec and th.is_alive():
                            ev = bg.get("quit_async_event" if kind == "quit" else "abort_event")
                            if ev is not None:
                                ev.set()
                                fired.add(id(th))
                                print(f"FC_INTERRUPT_INJECTED {kind} stream={sid} after={sec}s", flush=True)
            threading.Thread(target=hook, daemon=True).start()
        run_paced(sess, wav, pace=a.pace, tail_silence_s=a.tail_silence, drain=(a.close_kind == "normal"))
        if stop_hook is not None:
            stop_hook.set()
        wall = time.monotonic() - t0
        tt = eng.token_tensors()
        recs = list(eng.fc_records)
        sess.close(a.close_kind)
        cg = None
        if guard is not None:
            from mynah_v2.agent.call_guard import summary as guard_summary
            cg = guard_summary(guard)
            guard_un()
        tr.close()
        agent = np.concatenate(agent_chunks) if agent_chunks else np.zeros(0, np.float32)
        sf.write(od / "agent.wav", agent, T.OUTPUT_SR)
        sf.write(od / "output.wav", agent, T.OUTPUT_SR)  # name expected by scripts/asr_eval.py (OmniASR)
        user = AF.resample(wav.unsqueeze(0), 16000, T.OUTPUT_SR)[0].numpy()
        L = max(len(user), len(agent))
        sf.write(od / "combined.wav", np.stack([np.pad(user, (0, L - len(user))), np.pad(agent, (0, L - len(agent)))], 1),
                 T.OUTPUT_SR)
        st = sess.stats
        steps = np.array(st.step_ms_model) if st.step_ms_model else np.zeros(1)   # model-work steps only
        summ = {
            "wav": w, "mode": a.mode, "pace": a.pace, "audio_in_s": round(wav.numel() / 16000, 2),
            "wall_s": round(wall, 2), "audio_out_s": round(len(agent) / T.OUTPUT_SR, 2),
            "chunk_ms": 1000 * eng.chunk_samples / 16000,
            "step_ms_p50": round(float(np.percentile(steps, 50)), 1), "step_ms_p95": round(float(np.percentile(steps, 95)), 1),
            "rtf": round(float(steps.mean()) / (1000 * eng.chunk_samples / 16000), 3),
            "text": "".join(sess.text), "function_text": "".join(sess.func_text),
            "model_steps": sess.model_step, "turns": sess.turn_id,
            "rtf_definition": "p50/mean of per-chunk step latency over steps outside the FC fast path / 160 ms",
            "stats": {k: v for k, v in vars(st).items() if k not in ("step_ms", "step_ms_model")},
            "tool_calls": [{"call_id": c.call_id, "name": c.name, "args": c.arguments, "status": c.status.value,
                            "result": c.result, "error": c.error, "model_step": c.model_step,
                            "latency_ms": round(((c.t_end_ns or c.t_start_ns) - c.t_start_ns) / 1e6, 1),
                            "history": [h[1] for h in c.history], "late": [k for _, k in c.late] or None}
                           for c in led.calls.values()],
            "fc_timeline": fc_timeline_from_records(recs),
            "fc_records_raw": [{k: v for k, v in vars(r).items()} for r in recs],
            "tts_pad_silence": getattr(eng, "tts_pad_silence", lambda: None)(),
            "consumed": sess.controller.consumed, "grounding_violations": sess.controller.grounding_violations,
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "llm_cache": cache_info, "thinker_quant": quant_info, "perf_profile": perf_info,
            "preset": {"name": a.preset, **preset_info}, "determinism": det_info,
            "call_guard": cg,
        }
        summ["token_ids"] = {"text": list(sess.text_ids), "func": list(sess.func_ids)}
        if tt is not None:
            gt, ga, n, gf = tt
            summ["final_tokens"] = {"n": int(n), "text": gt[0, :n].tolist(), "asr": ga[0, :n].tolist(),
                                    "func": gf[0, :n].tolist() if gf is not None else None}
        (od / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
        print(json.dumps({k: summ[k] for k in ("wav", "mode", "text", "function_text", "rtf", "step_ms_p50",
                                                "audio_out_s", "tool_calls")}, ensure_ascii=False), flush=True)
        for c in summ["fc_timeline"]["calls"]:
            print("FC_TIMELINE", c["start_frame"], c["phase1_steps"], c["phase2_steps"], "violations:", c["violations"])
        summary_all.append(summ)
    (out_root / "summary_all.json").write_text(json.dumps(summary_all, indent=1, ensure_ascii=False))
    fx = getattr(eng.wrapper, "_v2_fc_ctx_fix", None)
    if fx is not None:                       # fc_ctx_fix: tokens the resumed steps ACTUALLY consumed + copy counters
        st = {k: v for k, v in fx["stats"].items() if k != "resume_inputs"}
        (out_root / "fc_ctx_fix.json").write_text(json.dumps({**st, "resume_inputs": fx["stats"]["resume_inputs"]}))
        print("FC_CTX_FIX " + json.dumps(st) + " resume_inputs=" + json.dumps(fx["stats"]["resume_inputs"][:20]),
              flush=True)
    fsg = getattr(eng, "_v2_full_graph", None)
    if fsg is not None:                                                   # EarTTS graph slots really allocated
        print("EARTTS_SLOTS " + json.dumps({"allocated": sum(x is not None for x in fsg.slots), "pool": fsg.n_slots,
                                            "eager_reasons": dict(fsg.stats.get("eager_reasons", {}))}), flush=True)
    if getattr(eng.wrapper, "_v2_fc_profile", None) is not None:        # --exp fc_profile (whole run, all items)
        from mynah_v2.model import fc_profile
        rep = fc_profile.report(eng)
        (out_root / "fc_profile.json").write_text(json.dumps(rep, indent=1))
        print("FC_PROFILE " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
