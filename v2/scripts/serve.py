"""Start the V2 realtime WebSocket server on the NVIDIA reference engine (one session at a time)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mynah_v2 import presets
from mynah_v2.agent.controller import Controller
from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolLedger, ToolRegistry, make_orders_db, orders_tool
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.transport.ws_server import ServerConfig, VoiceServer

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import read_system_message, render_prompt  # noqa: E402


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _backend_profile(args) -> str:
    if args.int4 or args.int4_pack or args.thinker_int4 or args.eartts_int4 or args.int4_extended:
        return "int4"
    if args.int8 or args.int8_pack:
        return "q8"
    return "bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--system-message", default=None,
                    help="default persona / instructions for sessions that do not send their own system_prompt "
                         "(text, or @file.txt); rendered with the tool schemas in NVIDIA's FC template")
    ap.add_argument("--strict", action="store_true",
                    help="fail if a requested optimisation cannot be enabled (default: continue without it, WARNING)")
    ap.add_argument("--preset", default="release", choices=list(presets.SINGLE),
                    help="release (default): enhanced sessions + the fastest gated single-session optimisations; "
                         "release-exact: the same with bit-exact attention; reference: NVIDIA's reference step. "
                         "Explicit flags override it")
    ap.add_argument("--mode", default=None, choices=["reference", "enhanced"],
                    help="default session mode (a client may request another one in session.start)")
    ap.add_argument("--tool-delay", type=float, default=0.0)
    ap.add_argument("--max-backlog", type=float, default=10.0,
                    help="seconds of inbound audio queued before the oldest is dropped (raise when slower than realtime)")
    ap.add_argument("--idle-timeout", type=float, default=60.0,
                    help="close a session (as a cancel) when the client sends nothing for this many seconds")
    ap.add_argument("--close-timeout", type=float, default=180.0,
                    help="if closing a session takes longer (stuck engine), refuse new sessions until restarted")
    ap.add_argument("--no-llm-cache", action="store_true",
                    help="diagnostic: NVIDIA's shipped no-cache Thinker (default: V2 incremental Thinker)")
    ap.add_argument("--int4", action="store_true", default=_env_flag("MYNAH_INT4"),
                    help="opt-in INT4 preset: standard paths plus tested MoG/RVQ/codec pointwise kernels; temporal conv/LSTM stay BF16")
    ap.add_argument("--int4-specialized", action="store_true", default=_env_flag("MYNAH_INT4_SPECIALIZED"),
                    help="enable the measured selected-MoG and RVQ gather kernels (implied by --int4; dynamic mode only)")
    ap.add_argument("--thinker-int4", action="store_true", default=_env_flag("MYNAH_THINKER_INT4"),
                    help="EXPERIMENT: replace Thinker backbone linears with native int4 tinygemm; lower memory and may improve GB10 decode speed")
    ap.add_argument("--thinker-int4-pack", default=os.environ.get("MYNAH_THINKER_INT4_PACK"),
                    help="legacy alias for --int4-pack")
    ap.add_argument("--int4-pack", default=os.environ.get("MYNAH_INT4_PACK"),
                    help="load one offline .safetensors + sibling .json INT4 bundle (Thinker + EarTTS)")
    ap.add_argument("--int4-heads", action="store_true", default=_env_flag("MYNAH_INT4_HEADS"),
                    help="also quantize the text/function output heads with --thinker-int4; keeps caches and embeddings unchanged")
    ap.add_argument("--int4-group", type=int, default=128, choices=[32, 64, 128],
                    help="int4 weight group size for --thinker-int4 (smaller groups preserve more quality but use more metadata)")
    ap.add_argument("--int4-clip", action="store_true",
                    help="use a per-group clipping search for --thinker-int4 (slower startup, lower quantization error)")
    ap.add_argument("--eartts-int4", action="store_true", default=_env_flag("MYNAH_EARTTS_INT4"),
                    help="EXPERIMENT: graph-safe native INT4 for the EarTTS Gemma backbone only; requires enhanced mode")
    ap.add_argument("--int4-extended", action="store_true", default=_env_flag("MYNAH_INT4_EXTENDED"),
                    help="EXPERIMENT: also quantize perception Linear, RNNT joint, CAS/fusion/MoG Linear paths")
    ap.add_argument("--int8", action="store_true", default=_env_flag("MYNAH_INT8"),
                    help="opt-in Q8 parity preset: Thinker, EarTTS, perception, RNNT joint, codec 1x1, and specialized Q8 paths")
    ap.add_argument("--int8-protected", action="store_true", default=_env_flag("MYNAH_INT8_PROTECTED"),
                    help="EXPERIMENT: W8A16 with Thinker text/function heads left in BF16; dynamic Q8 only")
    ap.add_argument("--int8-pack", default=os.environ.get("MYNAH_INT8_PACK"),
                    help="load one offline .safetensors + sibling .json Q8 bundle; specialized Q8 paths remain runtime opt-ins")
    ap.add_argument("--int8-specialized", action="store_true", default=_env_flag("MYNAH_INT8_SPECIALIZED"),
                    help="enable Q8 selected-MoG, RVQ, depthwise codec, and RNNT prediction paths (implied by --int8)")
    ap.add_argument("--tool-timeout", type=float, default=8.0,
                    help="ENHANCED: max seconds a tool may run before an explicit timeout error goes to the model")
    ap.add_argument("--exp", default=None, help="comma list of extra perf experiments (ENHANCED only)")
    ap.add_argument("--perf-profile", default=None, choices=["optimized-exact", "optimized-graph", "optimized-bf16"],
                    help="validated performance profile (mynah_v2.model.perf.PROFILES); ENHANCED only")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="performance option: torch.backends.cudnn.benchmark (codec 48 -> 5 ms/chunk on GB10); changes "
                         "numerics slightly (measured: one turn onset shifted by 1 frame), so never for parity runs")
    presets.add_determinism_args(ap)
    a = ap.parse_args()
    print("PRESET", a.preset, presets.fill(a, presets.SINGLE, a.preset), flush=True)
    if a.thinker_int4_pack and a.int4_pack:
        ap.error("use only one of --int4-pack and --thinker-int4-pack")
    if a.int4_pack is None:
        a.int4_pack = a.thinker_int4_pack
    int8_requested = a.int8
    if a.int8_pack:
        if int8_requested or a.int4 or a.thinker_int4 or a.eartts_int4 or a.int4_extended or a.int4_pack:
            ap.error("--int8-pack cannot be combined with --int8 or INT4 options")
        a.int8 = True
        a.int8_specialized = True
    if a.int8_protected and a.int8_pack:
        ap.error("--int8-protected is currently supported only for the dynamic Q8 path")
    if a.int8_protected and not a.int8:
        ap.error("--int8-protected requires --int8")
    if a.int4:
        if a.int8:
            ap.error("--int4 and --int8 are mutually exclusive")
        a.thinker_int4 = True
        a.int4_heads = True
        a.eartts_int4 = True
        a.int4_extended = True
        if os.environ.get("MYNAH_INT4_SPECIALIZED") != "0":
            a.int4_specialized = True
    if a.int8:
        if (a.thinker_int4 or a.eartts_int4 or a.int4_extended or a.int4_pack
                or a.int4_specialized or a.int4_heads):
            ap.error("--int4 options and --int8 are mutually exclusive")
        a.int8_specialized = True
    if a.int4_pack:
        # A bundle already contains both graph-safe families.  It is valid to
        # combine it with --int4 as a convenient preset, but never requantize
        # or stack the runtime TTS replacement on top of the loaded modules.
        a.thinker_int4_pack = a.int4_pack
        a.thinker_int4 = True
        a.int4_heads = False
        a.eartts_int4 = False
        a.int4_extended = False
        a.int4_specialized = False
    if a.int4_specialized:
        extra = ["eartts_mog_q4", "eartts_rvq_q4"]
        if os.environ.get("MYNAH_INT4_RNNT", "0") == "1":
            extra.append("rnnt_decoder_q4")
        current = [x for x in (a.exp or "").split(",") if x]
        a.exp = ",".join(current + [x for x in extra if x not in current])
    if a.int8_specialized:
        extra = ["eartts_mog_w8", "eartts_rvq_w8", "codec_depthwise_w8", "rnnt_decoder_w8"]
        current = [x for x in (a.exp or "").split(",") if x]
        a.exp = ",".join(current + [x for x in extra if x not in current])
    if a.int4_heads and not a.thinker_int4:
        ap.error("--int4-heads requires --thinker-int4")
    if a.thinker_int4 and a.mode == "reference":
        ap.error("--thinker-int4 is experimental and requires --mode enhanced")
    if a.eartts_int4 and a.mode == "reference":
        ap.error("--eartts-int4 is experimental and requires --mode enhanced")
    if a.int4_extended and a.mode == "reference":
        ap.error("--int4-extended is experimental and requires --mode enhanced")
    if a.int8 and a.mode == "reference":
        ap.error("--int8 is experimental and requires --mode enhanced")
    if (a.perf_profile or a.exp) and a.mode == "reference":
        ap.error("--perf-profile / --exp are never applied in REFERENCE mode")
    if a.cudnn_benchmark:
        import torch as _t
        _t.backends.cudnn.benchmark = True
    td = Path(a.trace_dir)
    td.mkdir(parents=True, exist_ok=True)
    db = make_orders_db(td / "orders.sqlite")
    reg = ToolRegistry()
    reg.register("get_order_status", orders_tool(db, a.tool_delay),
                 "Get the status, delivery date and tracking code of a customer order by its order id.", ORDERS_SCHEMA)
    eng = ReferenceEngine(build_cfg(output_dir=str(td / "_nemo"), overrides=presets.determinism_overrides(a)),
                          llm_cache=not a.no_llm_cache)
    presets.apply_determinism(a)
    print("LLM_CACHE", eng.llm_cache_info, flush=True)
    if a.thinker_int4:
        if a.thinker_int4_pack:
            from mynah_v2.model.quant import load_int4_pack
            quant_info = load_int4_pack(eng, a.thinker_int4_pack)
        else:
            from mynah_v2.model.quant import quantize_thinker_int4
            quant_info = quantize_thinker_int4(eng, max_group=a.int4_group, clip=a.int4_clip,
                                               include_heads=a.int4_heads)
        print("THINKER_QUANT", quant_info, flush=True)
    if a.eartts_int4:
        from mynah_v2.model.eartts_q4 import quantize_eartts_int4
        print("EARTTS_QUANT", quantize_eartts_int4(eng), flush=True)
    if a.int4_extended:
        from mynah_v2.model.quant import quantize_extended_int4
        print("EXTENDED_QUANT", quantize_extended_int4(eng), flush=True)
    if a.int8:
        if a.int8_pack:
            from mynah_v2.model.quant import load_w8_bundle
            print("INT8_PACK", load_w8_bundle(eng, a.int8_pack), flush=True)
        else:
            from mynah_v2.model.quant import quantize_extended_w8, quantize_thinker_w8
            from mynah_v2.model.eartts_q8 import quantize_eartts_w8
            print("THINKER_Q8", quantize_thinker_w8(eng, include_heads=not a.int8_protected), flush=True)
            print("EARTTS_Q8", quantize_eartts_w8(eng), flush=True)
            print("EXTENDED_Q8", quantize_extended_w8(eng), flush=True)
    if a.perf_profile or a.exp:
        exp = a.exp
        if a.thinker_int4:
            # Int4 replaces backbone Linear objects; do not stack the default w8 MLP swap on top.
            exp = ",".join(x for x in (exp or "").split(",") if x and not x.startswith(("wo:", "wor:")))
        print("PERF_PROFILE", presets.apply(eng, a.perf_profile, exp, strict=a.strict), flush=True)

    def controller_factory(sid, tracer, mode, session_ref):
        """Composition root: a fresh controller + ledger per session; ENHANCED gets the bounded tool wait
        (explicit timeout error to the model)."""
        led = ToolLedger(sid, tracer)
        bound = reg.bind(
            led, lambda: (session_ref().turn_id if session_ref() else 0),
            lambda: (session_ref().model_step if session_ref() else None),
            timeout_s=a.tool_timeout if mode == "enhanced" else None)
        eng.bind_session_tools(bound)          # owned by this session; unbound by engine.close()
        return Controller(None, ledger=led, tools=bound)
    cfg = ServerConfig(host=a.host, port=a.port, trace_dir=str(td), default_mode=a.mode, backend_profile=_backend_profile(a), max_backlog_s=a.max_backlog, idle_timeout_s=a.idle_timeout, close_timeout_s=a.close_timeout,
                       default_prompt=render_prompt(reg.schemas, read_system_message(a.system_message)), controller_factory=controller_factory)
    srv = VoiceServer(eng, cfg)
    asyncio.run(srv.serve(ready=lambda: print(f"READY ws://{a.host}:{a.port} (engine load {eng.load_s:.0f}s)", flush=True)))


if __name__ == "__main__":
    main()
