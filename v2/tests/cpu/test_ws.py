"""WebSocket transport against the fake engine: lifecycle, PCM in/out, events, busy rejection, cleanup."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_session import CS, FakeEngine  # noqa: E402

from mynah_v2.transport.ws_server import ServerConfig, VoiceServer  # noqa: E402


async def _session(port, n_chunks, mode="reference"):
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
        await ws.send(json.dumps({"type": "session.start", "mode": mode, "system_prompt": "p"}))
        created = json.loads(await ws.recv())
        pcm = (np.ones(CS * n_chunks) * 1000).astype("<i2").tobytes()
        for i in range(0, len(pcm), 1000):           # odd packet size on purpose
            await ws.send(pcm[i:i + 1000])
        await ws.send(json.dumps({"type": "session.end"}))
        audio, evs = bytearray(), []
        async for m in ws:
            if isinstance(m, bytes):
                audio.extend(m)
            else:
                d = json.loads(m)
                evs.append(d)
                if d.get("type") == "session.closed":
                    break
        return created, audio, evs


def test_ws_lifecycle_audio_and_events(tmp_path):
    eng = FakeEngine(fc_from=2, fc_to=4)
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0, trace_dir=str(tmp_path), post_fc_s=0.0))  # call ends in-input

    async def main():
        import websockets
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            created, audio, evs = await _session(port, 6)
            # second session reuses the same engine after cleanup
            created2, audio2, _ = await _session(port, 2)
            return created, audio, evs, created2, audio2

    created, audio, evs, created2, audio2 = asyncio.run(main())
    assert created["type"] == "session.created" and created["output_rate"] == 22050
    assert len(audio) == 6 * 1764 * 2 * 2          # fake engine: 2*1764 samples per chunk, int16
    kinds = [e.get("ev", e.get("type")) for e in evs]
    assert "session.opened" in kinds and "function.started" in kinds and "tool.result.consumed" in kinds
    assert kinds[-1] == "session.closed" and evs[-1]["chunks_in"] == 6
    assert created2["session_id"] != created["session_id"] and len(audio2) == 2 * 1764 * 2 * 2
    traces = list(tmp_path.glob("ws-*.jsonl"))
    assert len(traces) == 2


def test_ws_rejects_second_concurrent_session(tmp_path):
    eng = FakeEngine()
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0))

    async def main():
        import websockets
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws1:
                await ws1.send(json.dumps({"type": "session.start"}))
                await ws1.recv()
                async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws2:
                    msg = json.loads(await ws2.recv())
                await ws1.send(json.dumps({"type": "session.end"}))
                async for m in ws1:
                    if isinstance(m, str) and json.loads(m).get("type") == "session.closed":
                        break
                return msg

    assert asyncio.run(main())["error"] == "busy"


def test_ws_session_end_during_a_call_drains_it_before_closing(tmp_path):
    """Live regression (GB10 ws demo): the client ended while the tool call was running; the call leaked into the
    next session. The worker must keep the line open until the call is consumed, then close cleanly."""
    eng = FakeEngine(fc_from=4, fc_to=9)
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0, trace_dir=str(tmp_path), post_fc_s=0.32,
                                        max_drain_s=10))

    async def main():
        import websockets
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await _session(port, 5)

    _, audio, evs = asyncio.run(main())
    kinds = [e.get("ev", e.get("type")) for e in evs]
    assert "tool.result.consumed" in kinds and kinds[-1] == "session.closed"
    assert not eng.fc_active() and eng.k >= 8 + 2       # >= post_fc_s after the last call-active chunk (8)


def test_resampler_streaming_48k_to_16k_keeps_tone_and_length():
    from mynah_v2.transport.ws_server import _Resampler
    sr, f = 48000, 1000.0
    x = (np.sin(2 * np.pi * f * np.arange(sr) / sr) * 8000).astype("<i2").tobytes()
    rs = _Resampler(sr)
    out = b"".join(rs(x[i:i + 1001]) for i in range(0, len(x), 1001)) + rs(b"", last=True)  # odd packets
    y = np.frombuffer(out, dtype="<i2").astype(np.float32)
    assert abs(len(y) - 16000) <= 2
    spec = np.abs(np.fft.rfft(y[1000:15000]))
    peak_hz = np.argmax(spec) * 16000 / len(y[1000:15000])
    assert abs(peak_hz - f) < 5 and np.sqrt(np.mean(y[1000:15000] ** 2)) > 5000   # tone kept, no level loss
    assert _Resampler(16000)(x[:100]) == x[:100]                                     # 16 kHz passthrough


def test_ws_session_at_48k_reaches_the_engine_as_16k_chunks(tmp_path):
    import websockets
    eng = FakeEngine()
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0, trace_dir=str(tmp_path), post_fc_s=0.0))

    async def main():
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                await ws.send(json.dumps({"type": "session.start", "input_rate": 48000}))
                created = json.loads(await ws.recv())
                pcm = (np.ones(CS * 3 * 4) * 1000).astype("<i2").tobytes()     # 4 engine chunks at 48 kHz
                for i in range(0, len(pcm), 1500):
                    await ws.send(pcm[i:i + 1500])
                await ws.send(json.dumps({"type": "session.end"}))
                async for m in ws:
                    if isinstance(m, str) and json.loads(m).get("type") == "session.closed":
                        return created, json.loads(m)

    created, closed = asyncio.run(main())
    assert created["input_rate"] == 48000 and closed["chunks_in"] == 4 and closed["kind"] == "normal"


def test_ws_disconnect_without_session_end_is_a_cancel_close(tmp_path):
    import websockets
    eng = FakeEngine(fc_from=2, fc_to=200)
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0, trace_dir=str(tmp_path), max_drain_s=30))

    async def main():
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                await ws.send(json.dumps({"type": "session.start"}))
                await ws.recv()
                await ws.send((np.ones(CS * 3) * 1000).astype("<i2").tobytes())
                await asyncio.sleep(0.3)
            for _ in range(100):                        # hung up: the server must close without draining 30 s
                if srv.active == 0:
                    return True
                await asyncio.sleep(0.05)
            return False

    assert asyncio.run(main())
    rows = [json.loads(l) for f in tmp_path.glob("ws-*.jsonl") for l in f.read_text().splitlines()]
    closed = [r for r in rows if r["ev"] == "session.closed"][0]
    assert closed["kind"] == "cancel" and eng.k < 20


def test_ws_silent_client_is_closed_after_the_idle_timeout(tmp_path):
    import websockets
    eng = FakeEngine()
    srv = VoiceServer(eng, ServerConfig(host="127.0.0.1", port=0, trace_dir=str(tmp_path), idle_timeout_s=0.3))

    async def main():
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                await ws.send(json.dumps({"type": "session.start"}))
                await ws.recv()                          # session.created, then silence (no audio, no end)
                msgs = []
                async for m in ws:
                    if isinstance(m, str):
                        msgs.append(json.loads(m))
                        if msgs[-1].get("type") == "session.closed":
                            break
            for _ in range(100):
                if srv.active == 0 and not srv.engine_lock.locked():
                    return msgs
                await asyncio.sleep(0.05)
            return None

    msgs = asyncio.run(main())
    assert msgs is not None, "slot not released after the idle timeout"
    assert any(m.get("error") == "idle_timeout" for m in msgs)
    assert [m for m in msgs if m.get("type") == "session.closed"][0]["kind"] == "cancel"


def test_ws_client_that_never_sends_is_closed_after_the_idle_timeout(tmp_path):
    import websockets
    srv = VoiceServer(FakeEngine(), ServerConfig(host="127.0.0.1", port=0, idle_timeout_s=0.3))

    async def main():
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                msg = json.loads(await ws.recv())        # nothing sent at all
            for _ in range(100):
                if srv.active == 0 and not srv.engine_lock.locked():
                    return msg
                await asyncio.sleep(0.05)
            return None

    assert asyncio.run(main())["error"] == "idle_timeout"


def test_ws_stuck_close_turns_the_server_unhealthy_and_keeps_the_slot(tmp_path):
    import threading

    import websockets

    release = threading.Event()

    class StuckClose(FakeEngine):
        def close(self):
            release.wait(10)                             # a hung engine step / close
            super().close()

    srv = VoiceServer(StuckClose(), ServerConfig(host="127.0.0.1", port=0, close_timeout_s=0.3))

    async def main():
        async with websockets.serve(srv.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                await ws.send(json.dumps({"type": "session.start"}))
                await ws.recv()
                await ws.send(json.dumps({"type": "session.cancel"}))
                await asyncio.sleep(0.8)
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws2:
                return json.loads(await ws2.recv())

    try:
        msg = asyncio.run(main())
    finally:
        release.set()
    assert msg["error"].startswith("unhealthy:")
    assert srv.unhealthy is not None and srv.engine_lock.locked()
