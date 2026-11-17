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
    ap.add_argument("--reuse", action="store_true", help="E1k: Nano KV/SSM reuse across decision -> call -> reply")
    ap.add_argument("--microbench-int4", action="store_true", help="E1k: INT4 linear micro-bench (M=800, M=1)")
    ap.add_argument("--reuse-check", action="store_true", help="E1k: reuse vs no-reuse on one tool prompt first")
    ap.add_argument("--holding", action="store_true", help="E1l G1: holding phrase when Nano decides TOOL")
    ap.add_argument("--constrained", action="store_true", help="E1l G2: schema-constrained tool-call decoding")
    ap.add_argument("--call-ab", default="", help="E1l: summary.json of a previous session; free vs constrained call "
                                                  "on each of its tool prompts before the session")
    ap.add_argument("--int4-dispatch", action="store_true", help="E1k: the INT4 Nano variant uses Int4Dispatch")
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

    if a.microbench_int4:
        mb = int4_microbench(m)
        (out / "int4_microbench.json").write_text(json.dumps(mb, indent=1))
        print("MICROBENCH", json.dumps(mb), flush=True)
    if a.reuse_check:
        rc = reuse_check(tok, m, spec, POLICY["it"], lab, tool_schemas)
        (out / "reuse_check.json").write_text(json.dumps(rc, indent=1, ensure_ascii=False))
        print("REUSE_CHECK", json.dumps({k: v for k, v in rc.items() if k != "runs"}, ensure_ascii=False), flush=True)
        a.reuse = a.reuse and rc["same_call"] and rc["same_decision"]
        print("REUSE in session:", a.reuse, flush=True)
    if a.call_ab:
        ab = call_ab(tok, m, spec, POLICY["it"], tool_schemas, Path(a.call_ab))
        (out / "call_ab.json").write_text(json.dumps(ab, indent=1, ensure_ascii=False))
        print("CALL_AB", json.dumps(ab["summary"], ensure_ascii=False), flush=True)
    if a.ctl:
        from e1 import ctl_session
        res = ctl_session.run(out, "ctl_bf16", eng, tok, m, spec, POLICY["it"], Path(a.asr_dir), a.cut_s,
                              lab / "agentbench", tool_schemas, timer=timer, reuse=a.reuse, holding=a.holding,
                              constrained=a.constrained)
        print("SPEED ctl_bf16", json.dumps(res["speed"], default=str)[:1500], flush=True)
        el = time.monotonic() - T_START
        if a.int4_nano_variant_after_s and el < a.int4_nano_variant_after_s:
            t0 = time.monotonic()
            if a.int4_dispatch:
                from e1.int4_dispatch import quantize_int4_dispatch
                qi = quantize_int4_dispatch(m.backbone)
            else:
                from mynah_v2.model.quant import quantize_thinker_int4
                shim = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(llm=m.backbone))))
                qi = quantize_thinker_int4(shim)
            qi["mem_after_gb"] = round(mem_available_gb(), 1)
            print("INT4 nano", json.dumps(qi), round(time.monotonic() - t0, 1), "s", flush=True)
            res2 = ctl_session.run(out, "ctl_int4nano", eng, tok, m, spec, POLICY["it"], Path(a.asr_dir), a.cut_s,
                                   lab / "agentbench", tool_schemas, timer=timer, reuse=a.reuse)
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


def int4_microbench(m) -> dict:  # pragma: no cover - GPU
    """One large Nano linear: bf16 vs V2 Int4Linear vs Int4Dispatch, M=800 (prefill) and M=1 (decode)."""
    import torch
    import torch.nn.functional as F
    from mynah_v2.model.quant import Int4Linear
    from e1.int4_dispatch import Int4Dispatch, _nibbles
    lin = None
    for layer in m.backbone.layers:
        for name in ("up_proj", "in_proj"):
            mod = getattr(getattr(layer, "mixer", None), name, None)
            if isinstance(mod, torch.nn.Linear) and (lin is None or mod.weight.numel() > lin.weight.numel()):
                lin = mod
    q4 = Int4Linear(lin, max_group=128, clip=False)
    d = Int4Dispatch(q4, _nibbles(lin.weight.data.to(torch.bfloat16), q4.group), {})
    deq = d.dequant().float().clone()

    def bench(fn, x, n=20):
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn(x)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return round(sorted(ts)[len(ts) // 2], 3)
    res = {"shape": list(lin.weight.shape), "group": q4.group}
    with torch.inference_mode():
        for M in (800, 1):
            x = torch.randn(1, M, lin.in_features, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ deq.t()
            y_b = F.linear(x, lin.weight)
            y_o = q4(x)
            y_d = d(x)
            res[f"M{M}"] = {"bf16_ms": bench(lambda z: F.linear(z, lin.weight), x), "int4_old_ms": bench(q4, x),
                            "int4_dispatch_ms": bench(d, x),
                            "maxdiff_old_vs_ref": round(float((y_o.float() - ref).abs().max()), 4),
                            "maxdiff_dispatch_vs_ref": round(float((y_d.float() - ref).abs().max()), 4),
                            "maxdiff_bf16_vs_ref": round(float((y_b.float() - ref).abs().max()), 4),
                            "ref_absmax": round(float(ref.abs().max()), 3),
                            "dispatch_equals_old": bool(torch.equal(y_d, y_o))}
    del q4, d
    torch.cuda.empty_cache()
    return res


def reuse_check(tok, m, spec, policy_it, lab, tool_schemas) -> dict:  # pragma: no cover - GPU
    """One tool turn served with and without the KV/SSM reuse (sequential, default stream): same decision / call?"""
    import llm_text_bench as B2
    from mynah_v2.agent.tools import ToolRegistry
    from tenant_backend import TenantBackend
    from e1.nano_worker import NanoWorker
    msgs = [{"role": "assistant", "content": "Ciao! Come posso aiutarti oggi?"},
            {"role": "user", "content": "Buongiorno, vorrei sapere quando arriva il mio ordine. Mi chiamo Mario Rossi e "
                                        "il mio CAP è 10121"}]
    runs = {}
    for reuse in (False, True):
        said = []
        be = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        be._said = said
        reg = ToolRegistry()
        for t in spec["tools"]:
            reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
        bound = reg.bind(__import__("mynah_v2.agent.tools", fromlist=["ToolLedger"]).ToolLedger("rc", None),
                         lambda: 0, lambda: None, timeout_s=None)
        wk = NanoWorker(tok, m, spec["system_prompt"] + "\n" + policy_it + " /no_think", tool_schemas(reg), bound, be,
                        B2.parse_toolcall, max_new=60, tool_step=True, reuse=reuse)
        t0 = time.monotonic()
        wk._serve_job(1, [dict(x) for x in msgs], time.monotonic_ns())
        ev = wk.poll()
        done = [e for e in ev if e[0] == "done"][0]
        first = next((e[3] for e in ev if e[0] == "tok"), None)
        runs[str(reuse)] = {"text": done[2], "calls": [e[2]["name"] + json.dumps(e[2]["args"]) for e in ev if e[0] == "tool"],
                            "info": done[4], "wall_ms": round((time.monotonic() - t0) * 1e3, 1),
                            "first_tok_ms": round((first - done[4]["t_start_ns"]) / 1e6, 1) if first else None}
    a_, b_ = runs["False"], runs["True"]
    dec = lambda r: [s.get("out") for s in r["info"]["steps"] if s["step"] == "decide"]   # noqa: E731
    return {"same_decision": dec(a_) == dec(b_), "same_call": a_["calls"] == b_["calls"],
            "same_reply": a_["text"] == b_["text"], "first_tok_ms": {"no_reuse": a_["first_tok_ms"],
                                                                     "reuse": b_["first_tok_ms"]},
            "wall_ms": {"no_reuse": a_["wall_ms"], "reuse": b_["wall_ms"]}, "runs": runs}


def call_ab(tok, m, spec, policy_it, tool_schemas, summary_path: Path) -> dict:  # pragma: no cover - GPU
    """E1l G2: for every tool prompt of a recorded session (the history up to a user message whose answer was a
    call), the free call step vs the schema-constrained one: parsed call, generated tokens, time."""
    import llm_text_bench as B2
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from tenant_backend import TenantBackend
    from e1.nano_worker import NanoWorker
    msgs = json.load(open(summary_path))["messages"]
    be = TenantBackend(spec)
    reg = ToolRegistry()
    for t in spec["tools"]:
        reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
    bound = reg.bind(ToolLedger("ab", None), lambda: 0, lambda: None, timeout_s=None)
    wk = NanoWorker(tok, m, spec["system_prompt"] + "\n" + policy_it + " /no_think", tool_schemas(reg), bound, be,
                    B2.parse_toolcall, max_new=60, tool_step=True)
    pre = tok.encode("<TOOLCALL>[", add_special_tokens=False)
    rows = []
    import torch
    with torch.inference_mode():
        wk._greedy(0, wk._ids(msgs[:2], pre), 4, False)                       # warm-up (not measured)
        for i, mm in enumerate(msgs):
            if mm["role"] != "user" or i + 1 >= len(msgs) or not msgs[i + 1]["content"].startswith("<TOOLCALL>"):
                continue
            ids = wk._ids(msgs[: i + 1], pre)
            c, st_f = wk._greedy(0, ids, 60, False, stop_text="</TOOLCALL>")
            free = "<TOOLCALL>[" + tok.decode(c, skip_special_tokens=False)
            free = free[: free.index("</TOOLCALL>") + 11] if "</TOOLCALL>" in free else free
            con, st_c = wk._constrained_call(ids)
            pf, pc, rec = B2.parse_toolcall(free), B2.parse_toolcall(con), B2.parse_toolcall(msgs[i + 1]["content"])
            norm = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False) if x else None   # noqa: E731
            rows.append({"user": mm["content"], "recorded": msgs[i + 1]["content"], "free": free, "constrained": con,
                         "free_eq_recorded": norm(pf) == norm(rec), "constrained_eq_free": norm(pc) == norm(pf),
                         "free_tokens": st_f["n"], "free_ms": st_f["ms"], "constrained_generated": st_c["generated"],
                         "constrained_forced": st_c["forced"], "constrained_feeds": st_c["feeds"],
                         "constrained_ms": st_c["ms"], "constrained_prefill_ms": st_c["prefill_ms"]})
    n = len(rows)
    return {"summary": {"n": n, "constrained_eq_free": sum(r["constrained_eq_free"] for r in rows),
                        "free_eq_recorded": sum(r["free_eq_recorded"] for r in rows),
                        "free_tokens": [r["free_tokens"] for r in rows],
                        "constrained_generated": [r["constrained_generated"] for r in rows],
                        "free_ms": [r["free_ms"] for r in rows], "constrained_ms": [r["constrained_ms"] for r in rows]},
            "rows": rows}


if __name__ == "__main__":
    main()
