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


def main():  # pragma: no cover - GPU
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--idle-timeout", type=float, default=60.0)
    B.add_args(ap)
    a = ap.parse_args()
    td = Path(a.trace_dir)
    td.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ears = B.start_ears(a, td)
    spec = B.tenant(a)
    from frankenstein2.llm.nano import NanoShared
    shared = NanoShared(a.nano_dir, str(Path(a.lab_scripts) / "multilingual"), a.nano_precision, a.nano_graph)
    pocket = B.start_pocket(a, td)
    from frankenstein2.server.engine import Frankenstein2Engine, Resources
    res = Resources(ears=ears, make_llm=B.llm_factory(a, shared, spec), make_talker=B.talker_factory(a, pocket),
                    voices=B.voices(a), ears_thr=B.thresholds(a), trace_dir=str(td))
    eng = Frankenstein2Engine(res)
    from server.serve import make_server                     # v1's server wrapper (language / voice fields)
    from mynah_v2.transport.ws_server import ServerConfig
    cfg = ServerConfig(host=a.host, port=a.port, trace_dir=str(td), idle_timeout_s=a.idle_timeout)
    cfg.backend_profile = f"frankenstein2-{a.nano_precision}"
    srv = make_server(eng, cfg)
    loads = {"load_s": round(time.monotonic() - t0, 1), "ears": ears.ready, "talker": pocket.ready,
             "nano": shared.mem}
    (td / "server_loads.json").write_text(json.dumps(loads, indent=1, default=str))
    import asyncio
    try:
        asyncio.run(srv.serve(ready=lambda: print(f"READY port={a.port} " + json.dumps(loads, default=str),
                                                  flush=True)))
    finally:
        pocket.close()
        ears.close()


if __name__ == "__main__":
    main()
