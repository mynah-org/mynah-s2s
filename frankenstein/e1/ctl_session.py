"""E1j: one controller-driven two-LLM session on the Italian track (GPU). Same inputs as e1i: the 52 s track, the
frozen multilingual-ASR emissions, the tenant prompt / tools / handlers, NVIDIA RNNT forced turn-taking off."""

from __future__ import annotations

import json
import time
from pathlib import Path


def run(out: Path, od_name: str, eng, tok, model, spec, policy_it: str, asr_dir: Path, cut_s: float,
        lab_agentbench: Path, tool_schemas, timer=None) -> dict:  # pragma: no cover - GPU
    import sys
    sys.path.insert(0, str(lab_agentbench))
    from m0_runner import load_16k, render_prompt, set_model_cfg
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.ctl_driver import CtlDriver
    from e1.m1force import FrameLog
    from e1.nano_worker import NanoWorker
    from e1.runner import adapter_summary, pump, seed_all
    from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter
    from frankenstein_s2s.controller import Controller, ControllerConfig
    from frankenstein_s2s.driver import FakeTalkerDriver, PlaybackGate
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein_s2s.tracing import Tracer as FTracer
    w = eng.wrapper
    L = session_language("it")
    asr = json.load(open(asr_dir / "asr_it.json"))
    wav = load_16k(asr_dir / "track_it.wav")[: int(cut_s * 16000)]
    od = out / od_name
    od.mkdir()
    said = []
    backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
    backend._said = said
    reg = ToolRegistry()
    for t in spec["tools"]:
        reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
    tr = V2Tracer(od / "trace.jsonl", session_id=od_name)
    led = ToolLedger(od_name, tr)
    bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
    duplex_prompt = render_prompt(reg.schemas, with_language(spec["system_prompt"], L))
    system = spec["system_prompt"] + "\n" + policy_it + " /no_think"
    worker = NanoWorker(tok, model, system, tool_schemas(reg), bound, backend, None, max_new=80, tool_step=True)
    import llm_text_bench as B2
    worker.parse = B2.parse_toolcall
    worker.start()
    ftr = FTracer(od / "fr_trace.jsonl", session_id=od_name)
    drv, gate = FakeTalkerDriver(tracer=ftr), PlaybackGate(tracer=ftr)
    ad = TalkerAdapter([drv, gate], decode=lambda ids: tok.decode(ids),
                       config=AdapterConfig(talker_lang=L.talker_lang, barge_in="thinker"), tracer=ftr)
    ctl = Controller(ControllerConfig(min_blank=2, debounce=8, reply_timeout=100, tool_timeout=150, merge_frames=12,
                                      barge_guard=6, barge_in_frames=3), tracer=ftr)
    frames, blog = FrameLog(od / "frames.jsonl"), FrameLog(od / "brain.jsonl")
    seed_all()
    d = CtlDriver(w, worker, tok, asr, ad, ftr, frames, blog, ctl)
    d.pump = lambda: pump(ad, drv, gate)
    old = set_model_cfg(w.model_cfg, "force_turn_taking", False)
    uninstall = d.install()
    eng.bind_session_tools({})
    if timer is not None:
        timer.reset()
    sess = Session(eng, tr, Mode.REFERENCE, None, on_audio=lambda x: None, controller=None)
    t0 = time.monotonic()
    try:
        sess.open(duplex_prompt)
        run_paced(sess, wav, pace="realtime", tail_silence_s=1.0, max_extra_s=10.0, drain=True)
        pump(ad, drv, gate)
        sess.close("normal")
    finally:
        uninstall()
        tr.close()
        set_model_cfg(w.model_cfg, "force_turn_taking", old)
        worker.close(timeout=30)
    frames.close()
    blog.close()
    rec = dict(d.summary(), arm=od_name, lang="it", cut_s=cut_s, calls=worker.calls, backend_audit=backend.audit,
               ledger=[{"name": c.name, "status": c.status.value} for c in led.calls.values()],
               adapter=adapter_summary(ad, gate), wall_s=round(time.monotonic() - t0, 1),
               comp_timer=timer.report() if timer is not None else None, worker_alive_after_close=worker.is_alive())
    ftr.close()
    (od / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    fr = [json.loads(x) for x in open(od / "frames.jsonl")]
    dt = sorted((b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(fr, fr[1:]))
    q = lambda p: round(dt[int(p * (len(dt) - 1))], 1)                       # noqa: E731
    speed = {"wall_s": rec["wall_s"], "frames": len(fr),
             "frame_ms_mean": round((fr[-1]["t_ns"] - fr[0]["t_ns"]) / 1e6 / max(1, len(fr) - 1), 1),
             "frame_dt_ms_p50": q(.5), "frame_dt_ms_p95": q(.95), "frame_dt_ms_max": round(dt[-1], 1),
             "comp": rec["comp_timer"]}
    (od / "speed.json").write_text(json.dumps(speed, indent=1, default=str))
    return {"summary": rec, "speed": speed}
