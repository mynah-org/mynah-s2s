"""Multi-session WebSocket server for Frankenstein v2: the V2 wire format, several concurrent sessions, each with its
own language, on one set of shared resources (ears process, LLM model, talker process).

Wire (as mynah_v2.transport.ws_server, optional fields added):
  client -> server   {"type": "session.start", "language": "it"|"en"|"fr", "voice"?, "system_prompt"?,
                      "input_rate"?: 16000}  then binary PCM16 mono; {"type": "session.end"} | {"type": "session.cancel"}
                     URL ?probe=1: {"type": "server.ready", "sessions", "max_sessions"} and close (health probe)
  server -> client   {"type": "session.created", "session_id", "language", "voice", "input_rate", "output_rate": 22050,
                      "backend_profile"} | binary PCM16 22.05 kHz | {"type": "event", "ev", ...} (ears.frame and
                      audio.release only with ?verbose=1) | {"type": "session.closed", ...} | {"type": "error", ...}
One thread per session steps its Pipeline as its 80 ms input chunks arrive (the client streams in real time). Shared
process links are thread-safe; the LLM serialises requests across sessions; nothing session-specific is global.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

import numpy as np

QUIET = {"ears.frame", "audio.release", "ctl.intent", "asr.piece", "llm.text"}
STEP = 1280
OUT_SR = 22050


class SessionRunner(threading.Thread):
    def __init__(self, pipeline, send, out_rs, in_rs):
        super().__init__(daemon=True, name=f"f2-{pipeline.sid}")
        self.p, self.send, self.out_rs, self.in_rs = pipeline, send, out_rs, in_rs
        self.q: queue.Queue = queue.Queue()
        self.buf = np.zeros(0, np.float32)
        self.stop_kind = None
        self.steps = 0
        self.step_ms: list = []

    def push(self, pcm16: bytes) -> None:
        self.q.put(pcm16)

    def end(self, kind: str) -> None:
        self.stop_kind = kind
        self.q.put(None)

    def run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                x = np.frombuffer(item, "<i2").astype(np.float32) / 32768.0
                self.buf = np.concatenate([self.buf, self.in_rs(x)])
                while len(self.buf) >= STEP:
                    ch, self.buf = self.buf[:STEP], self.buf[STEP:]
                    t0 = time.monotonic()
                    r = self.p.step(ch)
                    self.step_ms.append((time.monotonic() - t0) * 1e3)
                    self.steps += 1
                    if r["audio"].any():
                        y = self.out_rs(r["audio"])
                        self.send((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
                    elif len(r["audio"]):
                        y = self.out_rs(r["audio"])
                        self.send(np.zeros(len(y), "<i2").tobytes())
        finally:
            self.p.close()


class MultiServer:
    def __init__(self, make_pipeline, max_sessions: int = 4, idle_timeout_s: float = 60.0, profile: str = "f2"):
        self.make_pipeline, self.max_sessions, self.idle = make_pipeline, max_sessions, idle_timeout_s
        self.profile = profile
        self.active: dict = {}
        self.lock = threading.Lock()

    async def handler(self, ws):
        path = getattr(getattr(ws, "request", None), "path", None) or getattr(ws, "path", "/")
        qs = parse_qs(urlparse(path).query)
        if qs.get("probe", ["0"])[0] == "1":
            await ws.send(json.dumps({"type": "server.ready", "sessions": len(self.active),
                                      "max_sessions": self.max_sessions}))
            await ws.close(code=1000, reason="health probe")
            return
        verbose = qs.get("verbose", ["0"])[0] == "1"
        with self.lock:
            if len(self.active) >= self.max_sessions:
                busy = True
            else:
                busy = False
                sid = "f2-" + uuid.uuid4().hex[:10]
                self.active[sid] = None
        if busy:
            await ws.send(json.dumps({"type": "error", "error": "busy", "max_sessions": self.max_sessions}))
            await ws.close(code=1013, reason="busy")
            return
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()

        def send_threadsafe(m):
            loop.call_soon_threadsafe(out_q.put_nowait, m)

        async def sender():
            while True:
                m = await out_q.get()
                if m is None:
                    return
                try:
                    await ws.send(m)
                except Exception:  # noqa: BLE001 - client gone: drop
                    pass
        send_task = asyncio.create_task(sender())
        runner, kind = None, "cancel"
        try:
            first = await asyncio.wait_for(ws.recv(), self.idle)
            start = json.loads(first) if isinstance(first, str) else {}
            lang = start.get("language") or qs.get("language", ["it"])[0]
            voice = start.get("voice") or qs.get("voice", [""])[0]
            in_rate = int(start.get("input_rate", 16000))

            def sink(name, d):
                if verbose or name not in QUIET:
                    send_threadsafe(json.dumps({"type": "event", **d}, default=str))
            p = await loop.run_in_executor(None, self.make_pipeline, sid, lang, voice, start.get("system_prompt"),
                                           sink)
            from frankenstein2.audio.playout import Resampler
            runner = SessionRunner(p, send_threadsafe, Resampler(p.talker.sr, OUT_SR), Resampler(in_rate, 16000))
            send_threadsafe(json.dumps({"type": "session.created", "session_id": sid, "language": lang, "voice": voice,
                                        "mode": "frankenstein2", "input_rate": in_rate, "output_rate": OUT_SR,
                                        "backend_profile": self.profile}))
            runner.start()
            if isinstance(first, bytes):
                runner.push(first)
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), self.idle)
                except asyncio.TimeoutError:
                    send_threadsafe(json.dumps({"type": "error", "error": "idle_timeout"}))
                    break
                except Exception:  # noqa: BLE001 - disconnected
                    break
                if isinstance(msg, bytes):
                    runner.push(msg)
                else:
                    t = json.loads(msg).get("type")
                    if t in ("session.end", "session.cancel"):
                        kind = "normal" if t == "session.end" else "cancel"
                        break
        except Exception as e:  # noqa: BLE001
            send_threadsafe(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}))
        finally:
            if runner is not None:
                runner.end(kind)
                await loop.run_in_executor(None, runner.join, 60)
                xs = sorted(runner.step_ms)
                send_threadsafe(json.dumps({"type": "session.closed", "session_id": sid, "kind": kind,
                                            "steps": runner.steps,
                                            "step_ms_p50": round(xs[len(xs) // 2], 2) if xs else None,
                                            "step_ms_p95": round(xs[int(.95 * (len(xs) - 1))], 2) if xs else None}))
            send_threadsafe(None)
            try:
                await asyncio.wait_for(send_task, 10)
            except Exception:  # noqa: BLE001
                send_task.cancel()
            with self.lock:
                self.active.pop(sid, None)

    async def serve(self, host: str, port: int, ready=None):
        import websockets
        async with websockets.serve(self.handler, host, port, max_size=2 ** 24):
            if ready:
                ready()
            await asyncio.Future()
