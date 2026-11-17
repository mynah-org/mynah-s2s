"""Frankenstein realtime WebSocket server: V2's VoiceServer (protocol unchanged) on the FrankensteinEngine.

  python frankenstein/server/serve.py --trace-dir DIR --port N --lab-scripts ... --nano-dir ... --flash-python ...
         --flash-harness ... --flash-root ... --flash-voice NAME_OR_WAV [--clone NAME=WAV ...]

Protocol additions (optional fields only, V2 clients keep working):
  session.start  {"language": "it" | "en" | ... (Qwen3-TTS codec languages), "voice": <clone or built-in name>}
  also accepted as URL query parameters ?language=..&voice=.. (for bridges that forward the URL only)
  events         engine.session (language, voice, session index), audio.flush (on a cancel: drop buffered audio),
                 engine.teardown (worker joined, CUDA memory before / after), plus the controller / adapter / talker
                 events; tool.* events carry call_id (V2 ToolLedger); every event carries session_id.
Prints ``READY port=N`` when it accepts connections (the web UI launcher waits for it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class WsProxy:
    """Wraps the connection to read Frankenstein's optional session.start fields before V2's handler sees them."""

    def __init__(self, ws, on_start):
        self._ws, self._on_start, self._first = ws, on_start, True

    def __getattr__(self, name):
        return getattr(self._ws, name)

    async def recv(self):
        msg = await self._ws.recv()
        if self._first:
            self._first = False
            if isinstance(msg, str):
                try:
                    m = json.loads(msg)
                except ValueError:
                    m = {}
                if m.get("type") == "session.start":
                    self._on_start(m)
        return msg


def options_from(query: str, start: dict | None) -> dict:
    qs = parse_qs(query or "")
    out = {"language": (qs.get("language") or qs.get("lang") or [None])[0], "voice": (qs.get("voice") or [None])[0]}
    for k in ("language", "voice"):
        if start and start.get(k):
            out[k] = start[k]
    return {k: v for k, v in out.items() if v}


def make_server(engine, cfg):  # pragma: no cover - needs the V2 runtime
    from mynah_v2.transport.ws_server import VoiceServer

    class FrankServer(VoiceServer):
        def __init__(self, engine, cfg):
            super().__init__(engine, cfg, tracer_factory=self._tracer)

        def _tracer(self, sid):
            t = self._default_tracer(sid)
            self.engine.attach_tracer(t)
            return t

        async def handler(self, ws):
            path = getattr(getattr(ws, "request", None), "path", None) or getattr(ws, "path", "/")
            query = urlparse(path).query
            if not parse_qs(query).get("probe"):
                self.engine.set_session_options(**options_from(query, None))

            def on_start(m):
                self.engine.set_session_options(**options_from(query, m))
            return await super().handler(WsProxy(ws, on_start))
    return FrankServer(engine, cfg)


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--flash-python", required=True)
    ap.add_argument("--flash-harness", required=True)
    ap.add_argument("--flash-root", required=True)
    ap.add_argument("--flash-voice", required=True)
    ap.add_argument("--clone", action="append", default=[], help="NAME=reference.wav (selectable voice)")
    ap.add_argument("--max-session-s", type=float, default=1200.0)
    ap.add_argument("--idle-timeout", type=float, default=60.0)
    a = ap.parse_args()
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "multilingual"))
    sys.path.insert(0, str(lab / "agentbench"))
    import llm_text_bench as B2
    B2.NANO = Path(a.nano_dir)
    from mynah_v2.model.perf import enable
    from mynah_v2.model.quant import quantize_thinker_int4
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.transport.ws_server import ServerConfig
    from e1.fast import safe_no_audio
    from e1.flash_client import FlashProc
    from e1.g0_session import BRIEF_IT
    from e1.int4_dispatch import quantize_int4_decode_only
    from e1.live_asr import LiveASR
    from e1.runner import mem_available_gb
    from e1.tools_bench import POLICY
    from server.engine import FrankensteinEngine
    t0 = time.monotonic()
    td = Path(a.trace_dir)
    td.mkdir(parents=True, exist_ok=True)
    loads = {"mem_before_gb": round(mem_available_gb(), 1)}
    ref = ReferenceEngine(build_cfg(output_dir=str(td / "_nemo"), overrides={"s2s.decode_audio": False}),
                          llm_cache=True)
    safe_no_audio(ref)
    quantize_thinker_int4(ref)
    for n in ("mamba_host_pos", "no_timer_sync", "thinker_graph"):
        enable(n, ref)
    loads["mem_after_engine_gb"] = round(mem_available_gb(), 1)
    tok, nano, _ = B2.load("nano")
    quantize_int4_decode_only(nano.backbone)
    loads["mem_after_nano_gb"] = round(mem_available_gb(), 1)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = FlashProc([a.flash_python, str(Path(__file__).resolve().parents[1] / "e1" / "flash_server.py"),
                      "--harness", a.flash_harness, "--root", a.flash_root, "--voice", a.flash_voice, "--lang", "it",
                      "--fast-cp", "--cp-graph", "--prio", "high", *sum((["--clone", c] for c in a.clone), [])],
                     env, str(td / "flash_server.log"))
    proc.start()
    proc.wait_ready(240)
    live = LiveASR(str(lab / "multilingual"), lang="it", att_right=3, window_s=12.0, every=8)
    loads["mem_after_all_gb"] = round(mem_available_gb(), 1)
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    brief = {"it": BRIEF_IT, "en": "Answer as on a phone call: at most two short sentences, no lists, no markdown."}
    eng = FrankensteinEngine(ref, tok, nano, spec, live, proc, lab / "agentbench", POLICY, brief,
                             max_session_s=a.max_session_s)
    cfg = ServerConfig(host=a.host, port=a.port, trace_dir=str(td), idle_timeout_s=a.idle_timeout)
    cfg.backend_profile = "frankenstein-int4"
    srv = make_server(eng, cfg)
    loads["load_s"] = round(time.monotonic() - t0, 1)
    (td / "server_loads.json").write_text(json.dumps({**loads, "talker": proc.ready}, indent=1, default=str))
    import asyncio
    try:
        asyncio.run(srv.serve(ready=lambda: print(f"READY port={a.port} " + json.dumps(loads), flush=True)))
    finally:
        proc.close()


if __name__ == "__main__":
    main()
