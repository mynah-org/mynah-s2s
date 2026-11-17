"""Model-free parts of the Frankenstein server engine (frankenstein/server)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.engine import BridgeTracer, SessionOptions, _Out  # noqa: E402
from server.serve import WsProxy, options_from  # noqa: E402


def test_output_queue_is_epoch_gated_and_flushed_on_cancel():
    o = _Out()
    assert o.audio(0, [1, 2, 3]) and o.audio(0, [4])
    assert o.cancel(1) == 4 and o.take() == []
    assert not o.audio(0, [5]) and o.stale == 1            # a chunk of the cancelled epoch never reaches the client
    assert o.audio(1, [6]) and o.take() == [[6]]


def test_bridge_tracer_maps_frame_and_drops_none():
    got = []

    class T:
        def emit(self, name, **f):
            got.append((name, f))
    b = BridgeTracer(T())
    b.emit("ctl.action", 12, kind="OPEN_TURN", turn_id=2, epoch=None)
    b.write({"ev": "brain.submit", "k": 30, "rid": 4, "user": "ciao", "info": {"x": 1}})
    assert got[0] == ("ctl.action", {"kind": "OPEN_TURN", "turn_id": 2, "model_step": 12})
    assert got[1] == ("brain.submit", {"rid": 4, "user": "ciao", "model_step": 30})
    BridgeTracer(None).emit("x")                           # no tracer attached: no-op


def test_session_options_from_start_message_and_query():
    assert options_from("language=fr&voice=a", None) == {"language": "fr", "voice": "a"}
    assert options_from("lang=en", {"type": "session.start", "language": "it", "voice": "b"}) == \
        {"language": "it", "voice": "b"}
    assert SessionOptions().language == "it"


def test_ws_proxy_reads_session_start_once_and_passes_messages_through():
    seen = []

    class WS:
        def __init__(self):
            self.msgs = [json.dumps({"type": "session.start", "language": "en", "voice": "v"}), b"\x00\x00"]
            self.request = type("R", (), {"path": "/v1/session"})()

        async def recv(self):
            return self.msgs.pop(0)
    p = WsProxy(WS(), seen.append)
    m1 = asyncio.run(p.recv())
    m2 = asyncio.run(p.recv())
    assert json.loads(m1)["language"] == "en" and m2 == b"\x00\x00"
    assert seen == [{"type": "session.start", "language": "en", "voice": "v"}] and p.request.path == "/v1/session"
