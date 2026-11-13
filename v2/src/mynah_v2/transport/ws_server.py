"""Minimal realtime WebSocket transport (no new protocol framework; a thin adapter over the Session).

Wire format (one connection = one session; ``max_sessions`` concurrent, default 1 because one engine = one GPU):
  client -> server
    text   {"type": "session.start", "system_prompt"?: str, "mode"?: "reference"|"enhanced",
            "input_rate"?: int (8000-48000, default 16000)}                                       (optional, first)
    binary PCM16 little-endian mono at input_rate, any length (resampled to 16 kHz with a streaming soxr resampler,
           as in the paper's App. B client contract; chunked server-side into 160 ms engine chunks)
    text   {"type": "session.end"}     graceful end: an in-flight tool call is answered first (drain), then close
    text   {"type": "session.cancel"}  immediate end: open tool calls cancelled, no drain (also: client disconnect)
  server -> client
    text   {"type": "session.created", "session_id", "mode", "input_rate": 16000, "output_rate": 22050}
    binary PCM16 little-endian mono 22.05 kHz agent audio, in generation order
    text   {"type": "event", "ev": <trace event name>, ...trace fields}   (audio.received/thinker.step suppressed
           unless ?verbose=1)
    text   {"type": "session.closed", ...stats}
Backpressure: inbound audio waits in a bounded queue (``max_backlog_s``); beyond it the oldest audio is dropped and
an ``audio.dropped`` notice is sent. Cleanup: disconnect -> the worker drains, closes the session, the engine stays
loaded for the next connection.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from mynah_v2.runtime.session import EnhancedOptions, Mode, Session
from mynah_v2.tracing import events as E

QUIET = {E.AUDIO_RECEIVED, E.THINKER_STEP, E.AUDIO_GENERATED, E.AUDIO_BUFFERED}


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    max_sessions: int = 1
    max_backlog_s: float = 10.0
    post_fc_s: float = 4.0          # line kept open after a tool call when the client ends the session
    max_drain_s: float = 90.0
    trace_dir: str | None = None
    default_mode: str = "reference"
    default_prompt: str | None = None
    # Composition hook, called per session: (session_id, tracer, mode, session_ref) -> conversation controller
    # (application state: tool ledger, facts, grounding). The transport never interprets tools or model behaviour.
    # session_ref() returns the live Session once created. None = no controller.
    controller_factory: Callable[..., object] | None = None


class _Stop:
    pass


class _Resampler:
    """PCM16 bytes at ``rate`` -> PCM16 bytes at 16 kHz, stateful across packets (no boundary artefacts)."""

    def __init__(self, rate: int):
        self.rate = rate
        self.rs = None
        if rate != 16000:
            import soxr
            self.rs = soxr.ResampleStream(rate, 16000, 1, dtype="int16", quality="HQ")
        self._odd = b""

    def __call__(self, data: bytes, last: bool = False) -> bytes:
        if self.rs is None:
            return data
        data = self._odd + data
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        else:
            self._odd = b""
        x = np.frombuffer(data, dtype="<i2")
        return self.rs.resample_chunk(x, last=last).astype("<i2").tobytes()


class SessionWorker(threading.Thread):
    """Runs the (blocking, GPU) engine for one connection, fed by a bounded queue of PCM bytes."""

    def __init__(self, session: Session, send_bytes: Callable[[bytes], None], max_backlog_bytes: int,
                 on_drop: Callable[[int], None], post_fc_s: float = 4.0, max_drain_s: float = 90.0):
        super().__init__(daemon=True)
        self.session = session
        self.q: queue.Queue = queue.Queue()
        self.backlog = 0
        self.max_backlog = max_backlog_bytes
        self.lock = threading.Lock()
        self.on_drop = on_drop
        self.post_fc_s, self.max_drain_s = post_fc_s, max_drain_s
        session.on_audio = lambda a: send_bytes((np.clip(a.numpy(), -1, 1) * 32767).astype("<i2").tobytes())

    def push(self, data: bytes) -> None:
        with self.lock:
            self.backlog += len(data)
            self.q.put(data)
            dropped = 0
            while self.backlog > self.max_backlog:
                try:
                    old = self.q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(old, _Stop):
                    self.q.put(old)
                    break
                self.backlog -= len(old)
                dropped += len(old)
        if dropped:
            self.on_drop(dropped)

    def stop(self, kind: str = "normal") -> None:
        """kind "normal": finish queued audio, drain an in-flight call, close. "cancel": drop queued audio and
        cancel open calls now."""
        if kind == "cancel":
            with self.lock:
                while True:
                    try:
                        old = self.q.get_nowait()
                    except queue.Empty:
                        break
                    if not isinstance(old, _Stop):
                        self.backlog -= len(old)
        self.close_kind = kind
        self.q.put(_Stop())

    def run(self) -> None:
        while True:
            item = self.q.get()
            if isinstance(item, _Stop):
                break
            with self.lock:
                self.backlog -= len(item)
            self.session.feed_pcm16(item)
        kind = getattr(self, "close_kind", "normal")
        if kind == "normal":
            self.session.drain_fc(self.post_fc_s, self.max_drain_s)
            self.session.flush(0.0)
        self.session.close(kind)


class VoiceServer:
    def __init__(self, engine, cfg: ServerConfig, tracer_factory: Callable[[str], E.Tracer] | None = None):
        self.engine = engine
        self.cfg = cfg
        self.active = 0
        self.engine_lock = threading.Lock()
        self.tracer_factory = tracer_factory or self._default_tracer

    def _default_tracer(self, sid: str) -> E.Tracer:
        path = Path(self.cfg.trace_dir) / f"{sid}.jsonl" if self.cfg.trace_dir else None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        return E.Tracer(path, session_id=sid, keep=False)

    async def handler(self, ws) -> None:
        path = getattr(getattr(ws, "request", None), "path", None) or getattr(ws, "path", "/")
        qs = parse_qs(urlparse(path).query)
        verbose = qs.get("verbose", ["0"])[0] == "1"
        if self.active >= self.cfg.max_sessions or not self.engine_lock.acquire(blocking=False):
            await ws.send(json.dumps({"type": "error", "error": "busy"}))
            await ws.close(code=1013, reason="busy")
            return
        self.active += 1
        loop = asyncio.get_running_loop()
        sid = f"ws-{uuid.uuid4().hex[:8]}"
        tracer = self.tracer_factory(sid)
        out_q: asyncio.Queue = asyncio.Queue()

        def send_threadsafe(msg) -> None:
            loop.call_soon_threadsafe(out_q.put_nowait, msg)

        def on_ev(ev: E.Event) -> None:
            if verbose or ev.name not in QUIET:
                d = {"type": "event", "ev": ev.name, "t_ns": ev.t_ns}
                d.update(ev.fields)
                send_threadsafe(json.dumps(d, ensure_ascii=False, default=str))
        tracer.subscribe(on_ev)

        async def sender():
            while True:
                msg = await out_q.get()
                if msg is None:
                    return
                await ws.send(msg)

        send_task = asyncio.create_task(sender())
        worker = None
        end_kind = "cancel"                  # disconnect without session.end: nobody is listening any more
        try:
            mode = qs.get("mode", [self.cfg.default_mode])[0]
            prompt = self.cfg.default_prompt
            first = await ws.recv()
            pending_audio = None
            in_rate = 16000
            if isinstance(first, str):
                msg = json.loads(first)
                if msg.get("type") == "session.start":
                    mode = msg.get("mode", mode)
                    prompt = msg.get("system_prompt", prompt)
                    in_rate = int(msg.get("input_rate", 16000))
                    if not 8000 <= in_rate <= 48000:
                        raise ValueError(f"input_rate {in_rate} outside 8000-48000")
            else:
                pending_audio = first
            to16k = _Resampler(in_rate)
            holder: dict = {}
            controller = (self.cfg.controller_factory(sid, tracer, mode, lambda: holder.get("s"))
                          if self.cfg.controller_factory else None)
            sess = holder["s"] = Session(self.engine, tracer, Mode(mode), EnhancedOptions() if mode == "enhanced" else None,
                           controller=controller)
            # through the ordered outbound queue, so it precedes every event of this session
            send_threadsafe(json.dumps({"type": "session.created", "session_id": sid, "mode": mode,
                                        "input_rate": in_rate, "output_rate": 22050}))
            await loop.run_in_executor(None, sess.open, prompt)
            worker = SessionWorker(sess, send_threadsafe, int(self.cfg.max_backlog_s * 16000 * 2),
                                   lambda n: send_threadsafe(json.dumps({"type": "audio.dropped", "bytes": n})),
                                   self.cfg.post_fc_s, self.cfg.max_drain_s)
            worker.start()
            if pending_audio is not None:
                worker.push(to16k(pending_audio))
            async for msg in ws:
                if isinstance(msg, bytes):
                    worker.push(to16k(msg))
                else:
                    m = json.loads(msg)
                    if m.get("type") == "session.end":
                        end_kind = "normal"
                        tail = to16k(b"", last=True)
                        if tail:
                            worker.push(tail)
                        break
                    if m.get("type") == "session.cancel":
                        break
        except Exception as e:  # noqa: BLE001 - report and clean up
            send_threadsafe(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}))
        finally:
            if worker is not None:
                worker.stop(end_kind)
                await loop.run_in_executor(None, worker.join)
                s = worker.session.stats
                send_threadsafe(json.dumps({"type": "session.closed", "session_id": sid,
                                            "chunks_in": s.chunks_in, "buffered": s.chunks_buffered,
                                            "replayed": s.chunks_replayed, "dropped": s.chunks_dropped,
                                            "kind": getattr(worker, "close_kind", None),
                                            "cancelled_calls": s.calls_cancelled_on_close}))
            send_threadsafe(None)
            try:
                await asyncio.wait_for(send_task, timeout=10)
            except Exception:  # noqa: BLE001 - client may be gone
                send_task.cancel()
            tracer.close()
            self.active -= 1
            self.engine_lock.release()

    async def serve(self, ready: Callable[[], None] | None = None) -> None:
        import websockets
        async with websockets.serve(self.handler, self.cfg.host, self.cfg.port, max_size=2**24):
            if ready:
                ready()
            await asyncio.Future()
