"""Start the V2 realtime WebSocket server on the NVIDIA reference engine (one session at a time)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mynah_v2 import presets
from mynah_v2.agent.controller import Controller
from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolLedger, ToolRegistry, make_orders_db, orders_tool
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.transport.ws_server import ServerConfig, VoiceServer

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import read_system_message, render_prompt  # noqa: E402


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
    ap.add_argument("--thinker-int4", action="store_true",
                    help="EXPERIMENT: replace Thinker backbone linears with native int4 tinygemm; lower memory and may improve GB10 decode speed")
    ap.add_argument("--int4-group", type=int, default=128, choices=[32, 64, 128],
                    help="int4 weight group size for --thinker-int4 (smaller groups preserve more quality but use more metadata)")
    ap.add_argument("--int4-clip", action="store_true",
                    help="use a per-group clipping search for --thinker-int4 (slower startup, lower quantization error)")
    ap.add_argument("--tool-timeout", type=float, default=8.0,
                    help="ENHANCED: max seconds a tool may run before an explicit timeout error goes to the model")
    ap.add_argument("--exp", default=None, help="comma list of extra perf experiments (ENHANCED only)")
    ap.add_argument("--perf-profile", default=None, choices=["optimized-exact", "optimized-graph"],
                    help="validated performance profile (mynah_v2.model.perf.PROFILES); ENHANCED only")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="performance option: torch.backends.cudnn.benchmark (codec 48 -> 5 ms/chunk on GB10); changes "
                         "numerics slightly (measured: one turn onset shifted by 1 frame), so never for parity runs")
    presets.add_determinism_args(ap)
    a = ap.parse_args()
    print("PRESET", a.preset, presets.fill(a, presets.SINGLE, a.preset), flush=True)
    if a.thinker_int4 and a.mode == "reference":
        ap.error("--thinker-int4 is experimental and requires --mode enhanced")
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
        from mynah_v2.model.quant import quantize_thinker_int4
        quant_info = quantize_thinker_int4(eng, max_group=a.int4_group, clip=a.int4_clip)
        print("THINKER_QUANT", quant_info, flush=True)
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
    cfg = ServerConfig(host=a.host, port=a.port, trace_dir=str(td), default_mode=a.mode, max_backlog_s=a.max_backlog, idle_timeout_s=a.idle_timeout, close_timeout_s=a.close_timeout,
                       default_prompt=render_prompt(reg.schemas, read_system_message(a.system_message)), controller_factory=controller_factory)
    srv = VoiceServer(eng, cfg)
    asyncio.run(srv.serve(ready=lambda: print(f"READY ws://{a.host}:{a.port} (engine load {eng.load_s:.0f}s)", flush=True)))


if __name__ == "__main__":
    main()
