"""Frankenstein v2 realtime WebSocket server: V2's VoiceServer (protocol unchanged) on Frankenstein2Engine.

  python frankenstein2/server/serve.py --trace-dir DIR --port N  (+ the resource options of server/build.py)

Protocol additions (optional fields only, as Frankenstein v1): session.start {"language": "it"|"en"|"fr",
"voice": NAME}, also as URL query parameters. One language per session, declared at start (never detected); it
drives ASR prompt, LLM policy, normaliser, spoken-numbers parsing, talker language, holding phrase and backchannel.
Prints ``READY port=N`` when it accepts connections.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frankenstein"))


def observability(a, td: Path):
    """The server's trace listeners: the SQLite store and the short event log (frankenstein2/obs.py)."""
    import logging
    from frankenstein2.obs import LogSink, TraceStore
    sinks, store = [], None
    if a.trace_db != "off":
        import atexit
        store = TraceStore(a.trace_db or td / "f2_traces.sqlite", level=a.trace_db_level)
        atexit.register(store.close)                 # flush the last batch on exit
        sinks.append(store.sink)
    if a.log_events != "off":
        lg = logging.getLogger("frankenstein2.trace")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        h = logging.StreamHandler(sys.stdout) if a.log_events == "stdout" else logging.FileHandler(td / "events.log")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(h)
        sinks.append(LogSink(lg).sink)
    return sinks, store


def main():  # pragma: no cover - GPU
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--idle-timeout", type=float, default=60.0)
    ap.add_argument("--max-sessions", type=int, default=4, help="concurrent sessions (each with its own language)")
    ap.add_argument("--trace-db", default="", help="SQLite trace store (default: TRACE_DIR/f2_traces.sqlite); "
                    "'off' disables it. View it with: python -m frankenstein2.obs DB")
    ap.add_argument("--trace-db-level", default="normal", choices=["normal", "all"],
                    help="all = also per-frame / per-hop events (large)")
    ap.add_argument("--trace-no-text", action="store_true", help="drop user / agent text from every trace")
    ap.add_argument("--log-events", default="file", choices=["file", "stdout", "off"],
                    help="short per-event lines to TRACE_DIR/events.log (file), the console, or nowhere")
    ap.add_argument("--v2-server", action="store_true",
                    help="V2's single-session VoiceServer instead of the multi-session server")
    B.add_args(ap)
    a = ap.parse_args()
    B.resolve_llm(a)
    td = Path(a.trace_dir)
    td.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ears = B.start_ears(a, td)
    spec = B.tenant(a)
    pocket = B.start_pocket(a, td)
    qproc = None
    if a.llm_backend == "qwen":                  # Qwen3.5 on our own inference engine (default), in its process
        qproc = B.start_qwen(a, td)
        make_llm, llm_info = B.qwen_factory(a, qproc, spec), qproc.ready
    else:                                        # Nemotron Nano 9B v2 in this process (fallback)
        from frankenstein2.llm.nano import NanoShared
        shared = NanoShared(a.nano_dir, str(Path(a.lab_scripts) / "multilingual"), a.nano_precision, a.nano_graph)
        make_llm, llm_info = B.llm_factory(a, shared, spec), shared.mem
    from frankenstein2.server.engine import Frankenstein2Engine, Resources, make_pipeline
    sinks, store = observability(a, td)
    res = Resources(ears=ears, make_llm=make_llm, make_talker=B.talker_factory(a, pocket),
                    voices=B.voices(a), ears_thr=B.thresholds(a), trace_dir=str(td), sinks=sinks,
                    store_text=not a.trace_no_text)
    cat = getattr(getattr(pocket, "catalog", None), "languages", None)
    if cat is not None:                          # packages: every language folder is served, found by scanning
        res.languages = cat
    loads = {"load_s": round(time.monotonic() - t0, 1), "ears": ears.ready, "talker": pocket.ready,
             "llm": llm_info}
    (td / "server_loads.json").write_text(json.dumps(loads, indent=1, default=str))
    import asyncio
    if not a.v2_server:
        from frankenstein2.server.multi import MultiServer
        srv = MultiServer(lambda sid, lang, voice, prompt, sink: make_pipeline(res, sid, lang, voice, prompt, sink),
                          max_sessions=a.max_sessions, idle_timeout_s=a.idle_timeout,
                          profile=f"frankenstein2-{a.llm_backend}")
        try:
            asyncio.run(srv.serve(a.host, a.port, ready=lambda: print(
                f"READY port={a.port} " + json.dumps(loads, default=str), flush=True)))
        finally:
            pocket.close()
            ears.close()
            if qproc is not None:
                qproc.close()
        return
    eng = Frankenstein2Engine(res)
    from server.serve import make_server                     # v1's server wrapper (language / voice fields)
    from mynah_v2.transport.ws_server import ServerConfig
    cfg = ServerConfig(host=a.host, port=a.port, trace_dir=str(td), idle_timeout_s=a.idle_timeout)
    cfg.backend_profile = f"frankenstein2-{a.llm_backend}"
    srv = make_server(eng, cfg)
    try:
        asyncio.run(srv.serve(ready=lambda: print(f"READY port={a.port} " + json.dumps(loads, default=str),
                                                  flush=True)))
    finally:
        pocket.close()
        ears.close()


if __name__ == "__main__":
    main()
