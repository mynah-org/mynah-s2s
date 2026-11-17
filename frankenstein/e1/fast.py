"""E1g (GPU job): the two-LLM session with the hot path cut down, profiled per component.

  * EarTTS off the hot path: decode_audio=False, with the streaming pipeline's log_output() made safe for
    audio_wave=None (instance-level wrap; the NVIDIA code is not edited);
  * Nano off the frame loop: e1/nano_worker.py (own thread and CUDA stream, streaming greedy decode, hybrid cache);
  * optional INT4 weight-only Thinker (mynah_v2.model.quant.quantize_thinker_int4) and the same INT4 path on the Nano
    backbone (same NemotronH mixers);
  * V2's CompTimer (mynah_v2.runtime.comp_timer) on perception / Thinker forward for the per-component budget.
The ASR stays the frozen CPU-built track (its live cost is not in this loop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def safe_no_audio(eng):
    """decode_audio=False crashes in StreamingS2SPipeline.log_output (audio_wave is None): give it an empty tensor."""
    import torch
    p = eng.pipeline
    orig = p.log_output

    def log_output(frames, audio_wave, ready_feats, text_pieces, asr_text_pieces=None):
        if audio_wave is None:
            audio_wave = torch.zeros(len(frames), 0, device="cuda")
        return orig(frames, audio_wave, ready_feats, text_pieces, asr_text_pieces)
    p.log_output = log_output


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut-s", type=float, default=52.0)
    ap.add_argument("--int4-thinker", action="store_true")
    ap.add_argument("--int4-nano", action="store_true")
    ap.add_argument("--ctl", action="store_true", help="E1j: controller-driven session(s) instead of the hook")
    ap.add_argument("--int4-nano-variant-after-s", type=float, default=0.0,
                    help="E1j: after the bf16 session, quantize Nano INT4 and rerun if the job is younger than this")
    ap.add_argument("--debug-first", action="store_true", help="E1h: check the worker decode against no-cache greedy "
                                                               "first; run the session only if it is exact")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    T_START = time.monotonic()
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "multilingual"))
    sys.path.insert(0, str(lab / "agentbench"))
    import llm_text_bench as B2
    B2.NANO = Path(a.nano_dir)
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.comp_timer import CompTimer
    from e1.runner import mem_available_gb
    from e1 import two_llm
    from e1.nano_worker import NanoWorker
    from e1.tools_bench import POLICY, tool_schemas
    import hashlib
    import os
    pkg = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()          # noqa: E731
    info = {"commit": os.environ.get("FRANK_COMMIT"), "argv": sys.argv,
            "harness_sha256": {str(p.relative_to(pkg)): sha(p) for p in sorted(pkg.rglob("*.py")) if "__pycache__" not in str(p)},
            "nano": str(B2.NANO), "nano_weights_note": "revision 6533e8de downloaded to this host's own cache (owner-accepted)",
            "mem_before_engine_gb": round(mem_available_gb(), 1)}
    t0 = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}), llm_cache=True)
    safe_no_audio(eng)
    info["engine_load_s"] = round(time.monotonic() - t0, 1)
    info["decode_audio"] = bool(eng.wrapper.decode_audio)
    if a.int4_thinker:
        from mynah_v2.model.quant import quantize_thinker_int4
        t0 = time.monotonic()
        info["int4_thinker"] = quantize_thinker_int4(eng)
        info["int4_thinker_s"] = round(time.monotonic() - t0, 1)
    info["mem_after_engine_gb"] = round(mem_available_gb(), 1)
    print("ENGINE", json.dumps(info, default=str)[-400:], flush=True)
    (out / "freeze.json").write_text(json.dumps(info, indent=1, default=str))
    t0 = time.monotonic()
    tok, m, ninfo = B2.load("nano")
    info["nano_load_s"] = round(time.monotonic() - t0, 1)
    if a.int4_nano:
        from mynah_v2.model.quant import quantize_thinker_int4
        t0 = time.monotonic()
        shim = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(llm=m.backbone))))
        info["int4_nano"] = quantize_thinker_int4(shim)
        info["int4_nano_s"] = round(time.monotonic() - t0, 1)
    info["mem_after_nano_gb"] = round(mem_available_gb(), 1)
    print("NANO", json.dumps({k: info[k] for k in info if "nano" in k}, default=str), flush=True)
    (out / "freeze.json").write_text(json.dumps(info, indent=1, default=str))
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    if a.debug_first:
        from e1 import nano_debug
        from mynah_v2.agent.tools import ToolRegistry
        from tenant_backend import TenantBackend
        reg = ToolRegistry()
        be = TenantBackend(spec)
        for t_ in spec["tools"]:
            reg.register(t_["name"], be.handlers()[t_["name"]], t_["description"], t_["parameters"])
        system = spec["system_prompt"] + "\n" + POLICY["it"] + " /no_think"
        t0 = time.monotonic()
        dbg, _wk = nano_debug.run_debug(tok, m, eng, system, tool_schemas(reg), n=20)
        dbg["debug_s"] = round(time.monotonic() - t0, 1)
        (out / "nano_debug.json").write_text(json.dumps(dbg, indent=1, ensure_ascii=False, default=str))
        print("DEBUG", json.dumps({k: v for k, v in dbg.items() if k.startswith(("div_", "exact", "debug_s"))}),
              json.dumps(dbg["text"], ensure_ascii=False)[:1200], flush=True)
        if not dbg["exact"]:
            print("DEBUG: worker decode not exact, session not run", flush=True)
            return
    timer = CompTimer()
    w = eng.wrapper
    pcm = getattr(w, "perception_cache_mgr", None)
    if pcm is not None and hasattr(pcm, "step"):
        timer._wrap(pcm, "step", "perception")
    timer._wrap(w.model.stt_model, "forward", "thinker_b1")
    timer._wrap(w, "_rnnt_step", "rnnt")

    if a.ctl:
        from e1 import ctl_session
        res = ctl_session.run(out, "ctl_bf16", eng, tok, m, spec, POLICY["it"], Path(a.asr_dir), a.cut_s,
                              lab / "agentbench", tool_schemas, timer=timer)
        print("SPEED ctl_bf16", json.dumps(res["speed"], default=str)[:1500], flush=True)
        el = time.monotonic() - T_START
        if a.int4_nano_variant_after_s and el < a.int4_nano_variant_after_s:
            from mynah_v2.model.quant import quantize_thinker_int4
            shim = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(llm=m.backbone))))
            t0 = time.monotonic()
            qi = quantize_thinker_int4(shim)
            print("INT4 nano", json.dumps(qi), round(time.monotonic() - t0, 1), "s", flush=True)
            res2 = ctl_session.run(out, "ctl_int4nano", eng, tok, m, spec, POLICY["it"], Path(a.asr_dir), a.cut_s,
                                   lab / "agentbench", tool_schemas, timer=timer)
            print("SPEED ctl_int4nano", json.dumps(res2["speed"], default=str)[:1500], flush=True)
        else:
            print("SKIP int4 nano variant, elapsed", round(el), flush=True)
        return

    def factory(tok_, model_, system, tools, bound, backend, parse):
        return NanoWorker(tok_, model_, system, tools, bound, backend, parse, max_new=80, tool_step=True)
    rec = two_llm.run_session(out, eng, tok, m, None, B2.parse_toolcall, spec, POLICY["it"], Path(a.asr_dir), a.cut_s,
                              lab / "agentbench", tool_schemas, fixes=True, brain_factory=factory, timer=timer,
                              od_name="fast_it")
    fr = [json.loads(x) for x in open(out / "fast_it" / "frames.jsonl")]
    dt = sorted((b["t_ns"] - a_["t_ns"]) / 1e6 for a_, b in zip(fr, fr[1:]))
    q = lambda p: round(dt[int(p * (len(dt) - 1))], 1)                       # noqa: E731
    span = (fr[-1]["t_ns"] - fr[0]["t_ns"]) / 1e6
    summ = {"wall_s": rec["wall_s"], "audio_s": a.cut_s, "frames": len(fr), "frame_ms_mean": round(span / max(1, len(fr) - 1), 1),
            "frame_dt_ms_p50": q(.5),
            "frame_dt_ms_p95": q(.95), "frame_dt_ms_max": round(dt[-1], 1), "comp": rec["comp_timer"],
            "async": rec["async"], "loads": {k: v for k, v in info.items() if k.endswith(("_s", "_gb"))}}
    (out / "speed.json").write_text(json.dumps(summ, indent=1, default=str))
    print("SPEED", json.dumps(summ, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
