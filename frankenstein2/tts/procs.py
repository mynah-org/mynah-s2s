"""Talker plugins backed by processes in their own environments.

``PocketProc`` + ``PocketTalker``  Pocket-TTS (pocket_server.py): one process for every session; a session's talker
                                  fixes its language and voice (a reference wav = zero-shot clone, or a stock name).
                                  Whole-text segments ("sentence" chunking by default, "whole" possible).
``FlashTalker``                   Sophon Flash / Qwen3-TTS through Frankenstein v1's talker process
                                  (frankenstein/e1/flash_server.py, imported read-only): one session at a time (the
                                  v1 process keeps one session's language/voice), segments as start + final.
pocket_swap (language slots with VRAM / RAM LRU) plugs in behind the same ``Talker`` interface when ready.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
import time

from .base import parse_sid


class PocketProc:  # pragma: no cover - process plumbing
    def __init__(self, argv: list, env: dict, log_path: str):
        self.argv, self.env, self.log_path = argv, env, log_path
        self.boxes: dict = {}
        self.misc: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.open_lock = threading.Lock()                # one session open (voice load) at a time
        self.ready = None

    def start(self):
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  env=self.env, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _box(self, ss):
        with self.lock:
            return self.boxes.setdefault(ss, queue.Queue())

    def _read(self):
        import numpy as np
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            m["t_recv_ns"] = time.monotonic_ns()
            if m.get("ev") == "audio":
                m["pcm"] = np.frombuffer(base64.b64decode(m.pop("pcm16")), dtype="<i2").astype(np.float32) / 32767
            (self._box(m["ss"]) if m.get("ss") is not None else self.misc).put(m)
        self.misc.put({"ev": "eof"})

    def send(self, m):
        with self.lock:                                  # several sessions' threads share the process
            self.p.stdin.write(json.dumps(m) + "\n")
            self.p.stdin.flush()

    def wait(self, ev: str, timeout: float = 300.0, pred=None) -> dict:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            try:
                m = self.misc.get(timeout=0.5)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError(f"talker process exited rc={self.p.returncode}")
                continue
            if m.get("ev") == ev and (pred is None or pred(m)):
                return m
            if m.get("ev") == "error":
                raise RuntimeError(f"talker process error: {m.get('error')}")
            if m.get("ev") == "eof":
                raise RuntimeError("talker process closed its output")
        raise TimeoutError(ev)

    def wait_ready(self, timeout=300.0):
        self.ready = self.wait("ready", timeout)
        return self.ready

    def poll(self, ss) -> list:
        box, out = self._box(ss), []
        while True:
            try:
                out.append(box.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        if self.p.poll() is None:
            try:
                self.send({"op": "quit"})
                self.p.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self.p.terminate()
        self.log.close()


class PocketTalker:  # pragma: no cover - needs the process
    sr = 24000

    def __init__(self, proc: PocketProc, ss: str, lang: str, voice: str = "", chunking: str = "sentence",
                 load_timeout: float = 120.0):
        self.proc, self.ss, self.lang, self.voice, self.chunking = proc, ss, lang, voice, chunking
        with proc.open_lock:
            proc.send({"op": "voice", "id": ss, "lang": lang, "voice": voice})
            self.info = proc.wait("voice", load_timeout, pred=lambda m: m.get("id") == ss)
        self.done_info: list = []

    def say(self, sid, text, epoch):
        self.proc.send({"op": "say", "ss": self.ss, "sid": sid, "lang": self.lang, "voice": self.voice,
                        "text": text, "epoch": epoch})

    def cancel(self, epoch):
        self.proc.send({"op": "cancel", "ss": self.ss, "epoch": epoch})

    def poll(self):
        out = []
        for m in self.proc.poll(self.ss):
            if m.get("ev") == "done":
                self.done_info.append(m)
            if m.get("ev") in ("audio", "done", "error"):
                out.append(m)
        return out

    def close(self):
        try:
            self.proc.send({"op": "close", "ss": self.ss})       # pocket_swap: release the session's slot binding
        except Exception:  # noqa: BLE001
            pass


class FlashTalker:  # pragma: no cover - needs the v1 talker process
    """Sophon Flash through v1's FlashProc (events carry epoch / turn / seg; we keep the lane in a side map)."""
    sr = 24000
    chunking = "sentence"

    def __init__(self, flash_proc, lang: str, voice: str = ""):
        self.p, self.lang = flash_proc, lang
        self.p.reset()
        self.p.submit_raw({"op": "session", "lang": lang, "voice": voice})
        self.lanes: dict = {}
        self.segs: dict = {}

    def say(self, sid, text, epoch):
        e, t, s, lane = parse_sid(sid)
        key = (e, t, s * 2 + (1 if lane == "bc" else 0))
        self.lanes[key] = sid
        self.p.submit_raw({"op": "start", "epoch": e, "turn": t, "seg": key[2], "text": text})
        self.p.submit_raw({"op": "final", "epoch": e, "turn": t, "seg": key[2]})

    def cancel(self, epoch):
        self.p.submit_raw({"op": "cancel", "epoch": epoch - 1, "new_epoch": epoch})

    def poll(self):
        out = []
        for ev in self.p.poll():
            key = (ev.get("epoch"), ev.get("turn"), ev.get("seg"))
            sid = self.lanes.get(key)
            if sid is None:
                continue
            if ev.get("ev") == "audio":
                out.append({"ev": "audio", "sid": sid, "pcm": ev["pcm"]})
            elif ev.get("ev") == "done":
                out.append({"ev": "done", "sid": sid})
        return out

    def close(self):
        self.p.reset()
