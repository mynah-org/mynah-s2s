"""Multi-session WebSocket server with fakes: two concurrent sessions in different languages over the wire."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

websockets = pytest.importorskip("websockets")

from frankenstein2.core import Pipeline, PipelineConfig  # noqa: E402
from frankenstein2.ears.client import EarsLink  # noqa: E402
from frankenstein2.llm.base import ScriptedLLM  # noqa: E402
from frankenstein2.server.multi import MultiServer  # noqa: E402
from frankenstein2.tests.fakes import FakeEarsBackend, pieces  # noqa: E402
from frankenstein2.tracing import Tracer  # noqa: E402
from frankenstein2.tts.base import FakeTalker  # noqa: E402

TEXT = {"it": "buongiorno vorrei un divano", "en": "hello I need a sofa"}
REPLY = {"it": "Certo, ecco il divano Alba.", "en": "Sure, here is the Alba sofa."}


def build():
    lang_of = {}
    be = FakeEarsBackend(lambda s, n, mode, reset, u, a: {
        2: {"pieces": pieces(4, TEXT[lang_of[s]]), "events": [{"kind": "ENDPOINT", "k": 11}]}}.get(n))
    made = {}

    def make(sid, lang, voice, prompt, sink):
        lang_of[sid] = lang
        llm = ScriptedLLM(lambda m, lang=lang: [("text", REPLY[lang])], lang=lang)
        p = Pipeline(sid, EarsLink(be, sid, lang), llm, FakeTalker(lang=lang), PipelineConfig(lang=lang),
                     Tracer(session_id=sid, sink=sink))
        p.open()
        made[sid] = (p, llm)
        return p
    return make, made


async def client(port, lang, n_chunks=60):
    msgs, audio = [], bytearray()
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
        await ws.send(json.dumps({"type": "session.start", "language": lang}))
        for _ in range(n_chunks):
            await ws.send(np.zeros(1280, "<i2").tobytes())
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "session.end"}))
        async for m in ws:
            if isinstance(m, bytes):
                audio.extend(m)
            else:
                d = json.loads(m)
                msgs.append(d)
                if d.get("type") == "session.closed":
                    break
    return msgs, audio


def test_two_languages_concurrently():
    make, made = build()
    srv = MultiServer(make, max_sessions=2)

    async def main():
        import websockets as W
        async with W.serve(srv.handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            return await asyncio.gather(client(port, "it"), client(port, "en"))
    (m_it, a_it), (m_en, a_en) = asyncio.run(main())
    for msgs, lang in ((m_it, "it"), (m_en, "en")):
        created = [m for m in msgs if m.get("type") == "session.created"][0]
        assert created["language"] == lang and created["output_rate"] == 22050
        turns = [m for m in msgs if m.get("ev") == "asr.turn"]
        assert turns and turns[0]["text"] == TEXT[lang]
        assert all(m.get("session_id") == created["session_id"] for m in msgs if m.get("type") == "event")
    assert len(a_it) > 0 and len(a_en) > 0
    hist = {p.lang: p.conv.messages for p, _ in made.values()}
    assert hist["it"][-1]["content"] == REPLY["it"] and hist["en"][-1]["content"] == REPLY["en"]


def test_health_probe_lists_the_served_languages():
    make, _ = build()
    langs = [{"code": "it", "name": "Italian", "tested": True}, {"code": "ko", "name": "Korean", "tested": False}]
    srv = MultiServer(make, max_sessions=2, languages=lambda: langs)

    async def main():
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        ready = asyncio.Event()
        task = asyncio.create_task(srv.serve("127.0.0.1", port, ready=ready.set))
        await asyncio.wait_for(ready.wait(), 5)
        async with websockets.connect(f"ws://127.0.0.1:{port}/v1/session?probe=1") as ws:
            msg = json.loads(await ws.recv())
        task.cancel()
        return msg
    msg = asyncio.run(main())
    assert msg["type"] == "server.ready" and msg["languages"] == langs


def test_demo_dialogue_caller_lines_come_back_as_audio():
    """demo.say: the server speaks the caller's line with the caller talker and returns it (demo.audio ... demo.done);
    the probe says the demo dialogue is available and lists the voices."""
    import base64
    make, made = build()
    callers = []

    def make_caller(sid, lang, voice):
        t = FakeTalker(lang=lang)
        callers.append((sid, lang, voice, t))
        return t
    srv = MultiServer(make, max_sessions=2, voices=lambda: ["agent", "caller"], make_caller=make_caller)

    async def main():
        import websockets as W
        async with W.serve(srv.handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            async with W.connect(f"ws://127.0.0.1:{port}/v1/session?probe=1") as ws:
                probe = json.loads(await ws.recv())
            got = []
            async with W.connect(f"ws://127.0.0.1:{port}/v1/session") as ws:
                await ws.send(json.dumps({"type": "session.start", "language": "it", "voice": "agent"}))
                await ws.send(json.dumps({"type": "demo.say", "id": 7, "text": "Buongiorno, avete un divano?"}))
                while True:
                    await ws.send(np.zeros(1280, "<i2").tobytes())
                    try:
                        m = await asyncio.wait_for(ws.recv(), 0.05)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(m, str):
                        d = json.loads(m)
                        if d.get("type", "").startswith("demo."):
                            got.append(d)
                            if d["type"] == "demo.done":
                                break
                await ws.send(json.dumps({"type": "session.end"}))
                async for m in ws:                      # drain until the server closes the session
                    if isinstance(m, str) and json.loads(m).get("type") == "session.closed":
                        break
            return probe, got
    probe, got = asyncio.run(main())
    assert probe["demo_dialogue"] is True and probe["voices"] == ["agent", "caller"]
    assert callers and callers[0][1:3] == ("it", "caller")          # another voice than the agent's
    audio = [d for d in got if d["type"] == "demo.audio"]
    assert audio and all(d["id"] == 7 and d["sr"] == 24000 for d in audio)
    assert sum(len(base64.b64decode(d["pcm"])) for d in audio) > 0
    assert got[-1] == {"type": "demo.done", "id": 7, "audio_s": got[-1]["audio_s"], "complete": True}
