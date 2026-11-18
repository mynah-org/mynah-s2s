"""Small browser bridge for the Mynah V2 realtime WebSocket server.

The bridge serves the static demo and optionally proxies the browser's binary
PCM stream to a GPU-backed V2 server. With no ``--v2-ws`` it stays entirely
local and emits a deterministic scenario trace, which makes the UI easy to
develop and demo without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import socket
import socketserver
import struct
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

SCENARIOS = {
    "ikea-orders": {
        "id": "ikea-orders",
        "eyebrow": "ORDERS · RETAIL",
        "name": "Mynah Home",
        "description": "A calm assistant for order status, delivery, and returns.",
        "accent": "amber",
        "suggestion": "Try asking: where is my order ABC123?",
        "prompt": """You are the Mynah Home voice assistant. Help with orders and delivery in a clear, concise way.\n\nWhen the user asks about an order, use get_order_status and speak only from the tool result. If the ID is missing, ask for it. Never invent a status.""",
        "tool": {"name": "get_order_status", "label": "Look up order", "args": {"order_id": "ABC123"}, "result": {"order_id": "ABC123", "status": "out for delivery", "eta": "tomorrow, 6 PM", "tracking": "MYN-4402"}},
        "answer": "I found order ABC123: it is out for delivery and should arrive tomorrow by 6 PM. The tracking code is MYN-4402.",
        "example_user": "Where is my order ABC123?",
        "samples": [
            {"id": "t02", "label": "Ryan · order ABC456", "path": "corpus_v1/t02.wav", "text": "Can you look up order A B C four five six for me?", "duration_s": 30.2},
            {"id": "t01", "label": "Vivian · order XYZ123", "path": "corpus_v1/t01.wav", "text": "Hi, I'd like to know where my order is. The order number is X Y Z one two three.", "duration_s": 30.2},
        ],
    },
    "insurance": {
        "id": "insurance",
        "eyebrow": "POLICIES · SUPPORT",
        "name": "Faro Insurance",
        "description": "An assistant for claims, coverage, and next steps.",
        "accent": "violet",
        "suggestion": "Try asking: what is the status of claim SIN-2048?",
        "prompt": """You are the Faro Insurance voice assistant. Explain one thing at a time and do not give legal advice.\n\nFor a claim status, use get_claim_status. Summarize only the tool response and say when a human agent is needed.""",
        "tool": {"name": "get_claim_status", "label": "Look up claim", "args": {"claim_id": "SIN-2048"}, "result": {"claim_id": "SIN-2048", "status": "under review", "next_step": "assessment scheduled for November 21"}},
        "answer": "Claim SIN-2048 is under review. An assessment is scheduled for November 21; you will receive the next update after that.",
        "example_user": "What is the status of claim SIN-2048?",
    },
    "travel": {
        "id": "travel",
        "eyebrow": "TRAVEL · CONCIERGE",
        "name": "Mynah Travel",
        "description": "A voice concierge for bookings and changing plans.",
        "accent": "cyan",
        "suggestion": "Try asking: is flight MX204 on time?",
        "prompt": """You are a voice travel concierge. Be concise, concrete, and transparent.\n\nFor flight status, use get_flight_status. Never say a flight is on time without the tool result.""",
        "tool": {"name": "get_flight_status", "label": "Check flight", "args": {"flight": "MX204"}, "result": {"flight": "MX204", "status": "on time", "gate": "B17", "departure": "19:40"}},
        "answer": "Flight MX204 is on time. Departure is at 7:40 PM from gate B17.",
        "example_user": "Is flight MX204 on time?",
    },
}


def json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class StaticHandler(SimpleHTTPRequestHandler):
    """Serve webui assets plus a tiny JSON API without another dependency."""

    server_version = "MynahWebUI/0.1"

    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/scenarios":
            scenarios = []
            for scenario in SCENARIOS.values():
                item = dict(scenario)
                item["samples"] = [dict(sample, available=self._sample_path(sample) is not None)
                                    for sample in scenario.get("samples", [])]
                scenarios.append(item)
            self._json({"scenarios": scenarios})
            return
        if path == "/api/config":
            configured = bool(self.server.v2_ws)
            self._json({
                "gpu_configured": configured,
                "v2_reachable": probe_v2(self.server.v2_ws) if configured else False,
                "profile": self.server.profile,
                "engine": self.server.engine,
                "profile_label": profile_label(self.server.profile),
                "audio_samples": bool(self.server.audio_root and self.server.audio_root.is_dir()),
                "ws_path": "/ws",
                "ws_port": self.server.ws_port,
            })
            return
        if path.startswith("/api/samples/"):
            parts = path.split("/")
            if len(parts) == 5:
                scenario = SCENARIOS.get(parts[3])
                sample = next((s for s in (scenario or {}).get("samples", []) if s["id"] == parts[4]), None)
                sample_path = self._sample_path(sample) if sample else None
                if sample_path:
                    body = sample_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_error(404, "sample unavailable")
            return
        if path == "/" or path == "":
            self.path = "/index.html"
        super().do_GET()

    def _json(self, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sample_path(self, sample: dict | None) -> Path | None:
        if not sample or not self.server.audio_root:
            return None
        root = self.server.audio_root.resolve()
        candidate = (root / sample["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def log_message(self, fmt, *args):
        if self.path.startswith("/api"):
            super().log_message(fmt, *args)


class WebUIHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler, *, v2_ws: str | None, profile: str, audio_root: Path | None):
        self.v2_ws = v2_ws
        self.profile = profile
        self.audio_root = audio_root
        super().__init__(address, handler)


def profile_label(profile: str) -> str:
    return {"int4": "INT4", "q8": "Q8", "bf16": "BF16", "demo": "LOCAL"}.get(profile, profile.upper())


def probe_v2(v2_ws: str | None) -> bool:
    """Check the V2 WebSocket protocol without creating a model session.

    The TCP fallback is intentionally kept for minimal/demo environments, but a
    real WebSocket probe is preferred so a listening-but-broken port is not shown
    as a live GPU backend.
    """
    if not v2_ws:
        return False
    target = urlparse(v2_ws)
    if target.hostname is None or target.port is None:
        return False
    try:
        import websockets

        query = dict(parse_qsl(target.query, keep_blank_values=True))
        query["probe"] = "1"
        probe_url = urlunparse(target._replace(query=urlencode(query)))

        async def check():
            async with websockets.connect(probe_url, open_timeout=0.4, close_timeout=0.4, max_size=1 << 20) as ws:
                message = await asyncio.wait_for(ws.recv(), timeout=0.6)
                payload = json.loads(message)
                return payload.get("type") == "server.ready" and not payload.get("unhealthy")

        return bool(asyncio.run(check()))
    except Exception:  # noqa: BLE001 - health checks must fail closed and never break /api/config
        try:
            with socket.create_connection((target.hostname, target.port), timeout=0.4):
                return True
        except OSError:
            return False


async def emit(ws, ev: str, **fields):
    await ws.send(json_line({"type": "event", "ev": ev, "t_ns": time.monotonic_ns(), **fields}))


def demo_steps(scenario: dict, sid: str):
    tool = scenario["tool"]
    call_id = f"{sid}-c1"
    def event(ev, **fields):
        return {"type": "event", "ev": ev, "t_ns": time.monotonic_ns(), **fields}
    return [
        (0.25, event("asr.generated", text=scenario["example_user"], turn_id=1)),
        (0.0, event("turn.started", who="user", turn_id=1)),
        (0.25, event("thinker.step", model_step=18, phase="understand", status="thinking")),
        (0.35, event("function.started", model_step=26, turn_id=1)),
        (0.0, event("tool.pending", call_id=call_id, tool=tool["name"], args=tool["args"], turn_id=1)),
        (0.65, event("tool.running", call_id=call_id, tool=tool["name"])),
        (0.55, event("tool.completed", call_id=call_id, tool=tool["name"], status="completed", result=tool["result"], latency_ms=1180)),
        (0.0, event("tool.result.injected", call_id=call_id, tool=tool["name"])),
        (0.35, event("thinker.step", model_step=44, phase="grounded answer", status="thinking")),
        (0.35, event("text.generated", text=scenario["answer"], turn_id=1, grounded_by=tool["name"])),
        (0.0, event("tool.result.consumed", call_id=call_id, tool=tool["name"])),
    ]


async def simulate(ws, scenario: dict, sid: str | None = None):
    """Play one guided tool-call trace; it is intentionally inspectable, not random."""
    if sid is None:
        sid = f"demo-{int(time.time() * 1000) % 100000:05d}"
        await ws.send(json_line({"type": "session.created", "session_id": sid, "mode": "demo", "input_rate": 16000, "output_rate": 22050}))
    for delay, payload in demo_steps(scenario, sid):
        await asyncio.sleep(delay)
        await ws.send(json_line(payload))


async def open_demo(ws, scenario: dict) -> str:
    """Open an idle demo session; the example starts only after demo.request."""
    sid = f"demo-{int(time.time() * 1000) % 100000:05d}"
    await ws.send(json_line({"type": "session.created", "session_id": sid, "mode": "demo", "input_rate": 16000, "output_rate": 22050}))
    await ws.send(json_line({"type": "event", "ev": "session.opened", "t_ns": time.monotonic_ns(), "mode": "demo", "scenario": scenario["id"]}))
    return sid


def ws_read_frame(conn):
    head = conn.recv(2)
    if len(head) != 2:
        return None
    fin_opcode, flags = head
    length = flags & 127
    if length == 126:
        length = struct.unpack("!H", conn.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", conn.recv(8))[0]
    mask = conn.recv(4) if flags & 128 else b""
    data = bytearray()
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    if mask:
        data = bytearray(value ^ mask[i % 4] for i, value in enumerate(data))
    return fin_opcode & 15, bytes(data)


def ws_send_frame(conn, payload: bytes, opcode: int = 1):
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    conn.sendall(header + payload)


class FallbackWebSocketHandler(socketserver.BaseRequestHandler):
    """Tiny stdlib-only fallback for the local demo when websockets is unavailable."""
    def handle(self):
        conn = self.request
        conn.settimeout(60)
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = conn.recv(4096)
            if not chunk:
                return
            request += chunk
        headers = dict(line.split(": ", 1) for line in request.decode().split("\r\n")[1:] if ": " in line)
        key = headers.get("Sec-WebSocket-Key")
        if not key:
            return
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        first = ws_read_frame(conn)
        if not first or first[0] != 1:
            return
        request = json.loads(first[1].decode())
        scenario = SCENARIOS.get(request.get("scenario", "ikea-orders"), SCENARIOS["ikea-orders"])
        if request.get("transport", "demo") == "gpu":
            ws_send_frame(conn, json_line({"type": "error", "error": "GPU proxy needs the websockets package in the uv environment"}).encode())
            return
        sid = f"demo-{int(time.time() * 1000) % 100000:05d}"
        ws_send_frame(conn, json_line({"type": "session.created", "session_id": sid, "mode": "demo", "input_rate": 16000, "output_rate": 22050}).encode())
        ws_send_frame(conn, json_line({"type": "event", "ev": "session.opened", "t_ns": time.monotonic_ns(), "mode": "demo", "scenario": scenario["id"]}).encode())
        while True:
            frame = ws_read_frame(conn)
            if not frame:
                return
            if frame[0] == 8:
                return
            if frame[0] != 1:
                continue
            msg = json.loads(frame[1].decode())
            if msg.get("type") in {"session.end", "session.cancel"}:
                ws_send_frame(conn, json_line({"type": "session.closed", "kind": msg["type"].split(".")[1]}).encode())
                return
            if msg.get("type") == "demo.request":
                next_scenario = SCENARIOS.get(msg.get("scenario"), scenario)
                for delay, payload in demo_steps(next_scenario, sid):
                    time.sleep(delay)
                    ws_send_frame(conn, json_line(payload).encode())


class FallbackWebSocketServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


async def proxy_session(browser, first: str | bytes, v2_ws: str, profile: str):
    """Relay JSON and binary frames to the real V2 transport, preserving frame order."""
    import websockets

    async with websockets.connect(v2_ws, max_size=2**24) as upstream:
        await upstream.send(first)
        browser_ended = False
        upstream_ended = False

        async def browser_to_gpu():
            nonlocal browser_ended
            async for message in browser:
                await upstream.send(message)
                if isinstance(message, str) and json.loads(message).get("type") in {"session.end", "session.cancel"}:
                    browser_ended = True
                    return

        async def gpu_to_browser():
            nonlocal upstream_ended
            try:
                async for message in upstream:
                    if isinstance(message, str):
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            payload = None
                        if isinstance(payload, dict) and payload.get("type") == "session.created":
                            actual_profile = payload.setdefault("backend_profile", profile)
                            payload["backend_profile_label"] = profile_label(actual_profile)
                            message = json_line(payload)
                    await browser.send(message)
                    if isinstance(message, str) and json.loads(message).get("type") == "session.closed":
                        return
            finally:
                upstream_ended = True

        tasks = [asyncio.create_task(browser_to_gpu()), asyncio.create_task(gpu_to_browser())]
        browser_task, gpu_task = tasks
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if browser_task in done and not upstream_ended:
                # A normal end/cancel must still receive V2's session.closed;
                # an unexpected browser disconnect gets a short cleanup window.
                if not browser_ended:
                    try:
                        await upstream.send(json_line({"type": "session.cancel"}))
                    except Exception:  # noqa: BLE001 - upstream may already be closed
                        pass
                try:
                    await asyncio.wait_for(gpu_task, timeout=90.0 if browser_ended else 5.0)
                except asyncio.TimeoutError:
                    gpu_task.cancel()
                    await asyncio.gather(gpu_task, return_exceptions=True)
            else:
                for task in done:
                    if task.exception():
                        raise task.exception()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def websocket_handler(ws, v2_ws: str | None, profile: str):
    try:
        first = await asyncio.wait_for(ws.recv(), timeout=30)
        if isinstance(first, bytes):
            await ws.send(json_line({"type": "error", "error": "session.start is required before audio"}))
            return
        request = json.loads(first)
        scenario = SCENARIOS.get(request.get("scenario", "ikea-orders"), SCENARIOS["ikea-orders"])
        transport = request.get("transport", "demo")
        if transport == "gpu" and v2_ws:
            await proxy_session(ws, first, v2_ws, profile=profile)
            return
        if transport == "gpu" and not v2_ws:
            await ws.send(json_line({"type": "error", "error": "GPU bridge is not configured: start with --v2-ws"}))
            return

        sid = await open_demo(ws, scenario)
        async for message in ws:
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("type") in {"session.end", "session.cancel"}:
                    await ws.send(json_line({"type": "session.closed", "kind": msg["type"].split(".")[1]}))
                    return
                if msg.get("type") == "demo.request":
                    await simulate(ws, SCENARIOS.get(msg.get("scenario"), scenario), sid)
    except Exception as exc:  # noqa: BLE001 - bridge errors are sent to the UI
        try:
            await ws.send(json_line({"type": "error", "error": f"{type(exc).__name__}: {exc}"}))
        except Exception:
            pass


async def websocket_server(host: str, port: int, v2_ws: str | None, profile: str):
    try:
        import websockets
    except ImportError:
        fallback = FallbackWebSocketServer((host, port), FallbackWebSocketHandler)
        print(f"WebSocket bridge: ws://{host}:{port}/ws (stdlib demo fallback)", flush=True)
        await asyncio.get_running_loop().run_in_executor(None, fallback.serve_forever)
        return
    async with websockets.serve(lambda ws: websocket_handler(ws, v2_ws, profile), host, port, max_size=2**24):
        print(f"WebSocket bridge: ws://{host}:{port}/ws", flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="Mynah S2S local web UI bridge")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 for LAN")
    parser.add_argument("--advertise-host", default=None, help="host printed as the LAN URL (e.g. 192.0.2.20)")
    parser.add_argument("--port", type=int, default=8787, help="static UI port")
    parser.add_argument("--engine", choices=["voicechat", "frankenstein", "frankenstein2"], default="voicechat",
                        help="which S2S server the GPU backend talks to: the official English VoiceChat engine or the multilingual Frankenstein engine (language and voice selectors)")
    parser.add_argument("--ws-port", type=int, default=8788, help="browser WebSocket port")
    parser.add_argument("--v2-ws", default=None, help="GPU server URL, e.g. ws://127.0.0.1:8765/v1/session")
    parser.add_argument("--profile", choices=["int4", "q8", "bf16", "demo"], default=None,
                        help="runtime profile label shown in the UI; it must match the V2 server flags")
    parser.add_argument("--audio-root", default=os.environ.get("MYNAH_AUDIO_ROOT", str(Path.home() / "mynah-s2s" / "data" / "agent")),
                        help="directory containing corpus_v1/ WAVs for guided real-audio input")
    args = parser.parse_args()
    args.profile = args.profile or ("int4" if args.v2_ws else "demo")
    root = Path(__file__).parent

    audio_root = Path(args.audio_root).expanduser() if args.audio_root else None
    httpd = WebUIHTTPServer((args.host, args.port), lambda *a, **kw: StaticHandler(*a, directory=str(root), **kw),
                            v2_ws=args.v2_ws, profile=args.profile, audio_root=audio_root)
    httpd.engine = args.engine                   # reported by /api/config: the page shows LANGUAGE/VOICE only for frankenstein
    httpd.ws_port = args.ws_port                 # reported by /api/config: the page connects to it (any UI port)
    http_thread = threading.Thread(target=httpd.serve_forever, name="webui-http", daemon=True)
    http_thread.start()
    browser_host = args.advertise_host or ("127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host)
    print(f"Web UI local: http://127.0.0.1:{args.port}", flush=True)
    if args.host in {"0.0.0.0", "::"}:
        print(f"Web UI LAN:   http://{browser_host}:{args.port}  (WebSocket :{args.ws_port})", flush=True)
    try:
        asyncio.run(websocket_server(args.host, args.ws_port, args.v2_ws, args.profile))
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
