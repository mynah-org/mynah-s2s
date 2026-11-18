"""Engine side of the ears: the hop clock, the process plumbing and the user transcript.

``EarsLink``     per session: collects 80 ms input steps into 320 ms hops (user mic + 2nd ear), keeps at most one
                 hop in flight (a reset or mode change asked by the controller applies to the next hop, in order),
                 and turns hop results into (events, pieces, frames). Talks to a backend with ``send(msg)`` and
                 ``poll(sid) -> [msg]`` (the ears process, or a fake in tests).
``EarsProc``     the ears process (frankenstein2/ears/server.py, its own Python): JSON lines, one reader thread,
                 replies routed by session id, so several sessions (each with its own language) share one process.
``UserText``     the user transcript from the streaming ASR pieces, with frame stamps and a turn boundary.
Pure Python except EarsProc's process plumbing.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field

HOP = 5120
STEP = 1280


def f32b64(x) -> str:
    import numpy as np
    return base64.b64encode(np.asarray(x, dtype="<f4").tobytes()).decode()


@dataclass
class UserText:
    """ASR pieces (SentencePiece, "▁" = word start) stamped with their encoder frame. ``take`` returns the text
    since the boundary and moves it; ``drop_before(k)`` forgets pieces older than frame k (speech heard while the
    agent spoke and not part of a barge-in)."""
    pieces: list = field(default_factory=list)       # (frame, piece)
    boundary: int = 0                                # index into pieces

    def add(self, items) -> str:
        new = [(int(p["f"]), p["p"]) for p in items if not _is_tag(p["p"])]
        self.pieces += new
        return _join([p for _, p in new], leading=bool(self.pieces[:-len(new)] if new else self.pieces))

    def pending(self) -> str:
        return _join([p for _, p in self.pieces[self.boundary:]]).strip()

    def take(self) -> str:
        txt = self.pending()
        self.boundary = len(self.pieces)
        return txt

    def drop_before(self, k: int) -> int:
        """Move the boundary past every pending piece of a word that started before frame k. -> pieces dropped."""
        i = self.boundary
        n0 = i
        while i < len(self.pieces) and self.pieces[i][0] < k:
            i += 1
        while i < len(self.pieces) and not self.pieces[i][1].startswith("▁") and i > n0:
            i += 1                                     # never start in the middle of a word
        self.boundary = i
        return i - n0


def _is_tag(p: str) -> bool:
    s = p.strip().replace("▁", "")
    return s.startswith("<") and s.endswith(">")


def _join(pieces, leading: bool = False) -> str:
    s = "".join(pieces).replace("▁", " ")
    return s if leading else s.lstrip()


class EarsLink:
    def __init__(self, backend, sid: str, lang: str, thr: dict | None = None, gru_features: str = "window",
                 sync: bool = False, sync_timeout_s: float = 2.0):
        self.b, self.sid, self.lang = backend, sid, lang
        self.thr, self.gru_features = thr or {}, gru_features
        self.sync, self.sync_timeout_s = sync, sync_timeout_s
        self.u: list = []
        self.a: list = []
        self.n_sent = 0
        self.in_flight = None              # hop index awaiting its result
        self.queued: list = []             # complete hops not yet sent (one in flight at a time)
        self.reset_next = False
        self.results: list = []
        self.lag_hops_max = 0
        self.errors: list = []

    def open(self) -> None:
        self.b.send({"op": "open", "sid": self.sid, "lang": self.lang, "thr": self.thr,
                     "gru_features": self.gru_features})

    def close(self) -> None:
        self.b.send({"op": "close", "sid": self.sid})

    def request_reset(self) -> None:
        self.reset_next = True

    def push(self, user_step, agent_step, mode: str) -> None:
        """One 80 ms step of both channels (16 kHz float). ``mode`` is the controller's mode at this step; a hop
        takes the mode of its last step."""
        self.u.append(user_step)
        self.a.append(agent_step)
        if len(self.u) == HOP // STEP:
            import numpy as np
            self.queued.append((np.concatenate(self.u), np.concatenate(self.a), mode))
            self.u, self.a = [], []
        self._send_next()

    def _send_next(self) -> None:
        while self.in_flight is None and self.queued:
            u, a, mode = self.queued.pop(0)
            self.b.send({"op": "hop", "sid": self.sid, "n": self.n_sent, "mode": mode, "reset": self.reset_next,
                         "u": f32b64(u), "a": f32b64(a)})
            self.reset_next = False
            self.in_flight = self.n_sent
            self.n_sent += 1
            if not self.sync:
                return
            self._wait()

    def _wait(self) -> None:
        t_end = time.monotonic() + self.sync_timeout_s
        while self.in_flight is not None and time.monotonic() < t_end:
            self._drain(block=True)

    def _drain(self, block: bool = False) -> None:
        for m in self.b.poll(self.sid, block=block):
            if m.get("ev") == "hop":
                if m.get("n") == self.in_flight:
                    self.in_flight = None
                self.results.append(m)
            elif m.get("ev") == "error":
                self.errors.append(m)
                if m.get("op") == "hop" and m.get("n") == self.in_flight:
                    self.in_flight = None

    def poll(self) -> list:
        """Hop results that arrived (in order); sends the next queued hop when the previous one is back."""
        self._drain()
        self._send_next()
        self.lag_hops_max = max(self.lag_hops_max, len(self.queued))
        out, self.results = self.results, []
        return out


class EarsProc:  # pragma: no cover - process plumbing, exercised on the GPU host
    def __init__(self, argv: list, env: dict, log_path: str):
        self.argv, self.env, self.log_path = argv, env, log_path
        self.boxes: dict = {}
        self.lock = threading.Lock()
        self.misc: queue.Queue = queue.Queue()
        self.ready = None
        self.p = None

    def start(self):
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  env=self.env, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _box(self, sid) -> queue.Queue:
        with self.lock:
            return self.boxes.setdefault(sid, queue.Queue())

    def _read(self):
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            m["t_recv_ns"] = time.monotonic_ns()
            (self._box(m["sid"]) if m.get("sid") is not None else self.misc).put(m)
        self.misc.put({"ev": "eof"})

    def wait_misc(self, ev: str, timeout: float):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            try:
                m = self.misc.get(timeout=0.5)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError(f"ears process exited rc={self.p.returncode}")
                continue
            if m.get("ev") == ev:
                return m
            if m.get("ev") == "eof":
                raise RuntimeError("ears process closed its output")
        raise TimeoutError(ev)

    def wait_ready(self, timeout: float = 300.0) -> dict:
        self.ready = self.wait_misc("ready", timeout)
        return self.ready

    def send(self, msg: dict) -> None:
        with self.lock:                                  # several sessions' threads share the process
            self.p.stdin.write(json.dumps(msg) + "\n")
            self.p.stdin.flush()

    def poll(self, sid, block: bool = False) -> list:
        box, out = self._box(sid), []
        if block:
            try:
                out.append(box.get(timeout=0.05))
            except queue.Empty:
                return out
        while True:
            try:
                out.append(box.get_nowait())
            except queue.Empty:
                return out

    def request(self, msg: dict, ev: str, timeout: float = 60.0) -> dict:
        self.send(msg)
        return self.wait_misc(ev, timeout)

    def close(self):
        if self.p is not None and self.p.poll() is None:
            try:
                self.send({"op": "quit"})
                self.p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.p.terminate()
        self.log.close()


class EarsPool:
    """Several ears processes behind the EarsProc interface: a session is placed on the process with the fewest
    sessions when it opens and stays there (its caches and GRU state live in that process). On GPUs where the ears are
    bound by per-session kernel launches (small batches), processes run those launches in parallel."""

    def __init__(self, procs: list):
        self.procs = procs
        self.where: dict = {}
        self.ready = procs[0].ready if procs else None

    def _proc(self, sid):
        if sid not in self.where:
            load = [sum(1 for v in self.where.values() if v == i) for i in range(len(self.procs))]
            self.where[sid] = load.index(min(load))
        return self.procs[self.where[sid]]

    def send(self, msg: dict) -> None:
        sid = msg.get("sid")
        p = self._proc(sid) if sid is not None else self.procs[0]
        p.send(msg)
        if msg.get("op") == "close":
            self.where.pop(sid, None)

    def poll(self, sid, block: bool = False) -> list:
        return self._proc(sid).poll(sid, block=block)

    def request(self, msg: dict, ev: str, timeout: float = 60.0) -> dict:
        return self.procs[0].request(msg, ev, timeout)

    def close(self):
        for p in self.procs:
            p.close()
