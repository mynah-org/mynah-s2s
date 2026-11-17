"""G0: engine-side half of the real talker: the adapter sink that drives the Flash talker process (e1/flash_server.py)
and a playback model on the wall clock.

``FlashProc``  TalkerDriver sink: adapter commands -> JSON lines on the talker process's stdin; a reader thread turns
               its stdout into events (audio chunks with their CLOCK_MONOTONIC delivery time, segment done, cancel).
``Player``     what a client playing the delivered audio would hear: chunks of the current epoch are scheduled back
               to back; the first chunk of a segment that starts while nothing is playing waits ``prebuffer_s``; a
               chunk that arrives after the scheduled audio ran out is an UNDERRUN (gap). A cancel at wall time t
               stops playback at t (the rest of a chunk in progress and every later chunk are dropped) and moves to
               the new epoch: chunks of an old epoch delivered afterwards are dropped (stale). The model has no
               transport delay; it is the lower bound of what a real client adds.
Pure Python except the process plumbing; ``Player`` is unit-tested.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field

from frankenstein_s2s.events import AppendText, Cancel, FinalizeSegment, StartSegment


@dataclass
class Item:
    t_start: float
    n: int                        # samples
    epoch: int
    turn: int
    seg: int
    a: int
    b: int
    t_ready: float
    keep: int = -1                # samples kept after a cancel cut (-1 = all)
    dropped: bool = False
    pcm: object = None


@dataclass
class Player:
    sr: int = 24000
    prebuffer_s: float = 0.32
    epoch: int = 0
    end: float | None = None
    cur: tuple | None = None
    items: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    stale: list = field(default_factory=list)
    cuts: list = field(default_factory=list)

    def on_cancel(self, t: float, new_epoch: int) -> None:
        playing = self.end is not None and self.end > t
        for it in self.items:
            if it.dropped or it.keep == 0:
                continue
            if it.t_start >= t:
                it.dropped = True
            elif it.t_start + it.n / self.sr > t:
                it.keep = int((t - it.t_start) * self.sr)
        self.cuts.append({"t": t, "was_playing": playing, "new_epoch": new_epoch})
        self.epoch = new_epoch
        if self.end is not None:
            self.end = min(self.end, t)
        self.cur = None

    def on_audio(self, t_ready: float, epoch: int, turn: int, seg: int, a: int, b: int, n: int, pcm=None) -> Item | None:
        if epoch != self.epoch:
            self.stale.append({"t": t_ready, "epoch": epoch, "turn": turn, "seg": seg, "a": a, "b": b})
            return None
        key = (epoch, turn, seg)
        idle = self.end is None or self.end <= t_ready
        if idle and key != self.cur:
            start = t_ready + self.prebuffer_s          # a new segment, nothing audible: prebuffer
        elif idle:
            start = t_ready                             # same segment ran dry: underrun
            self.gaps.append({"t": self.end, "gap_s": round(t_ready - self.end, 4), "turn": turn, "seg": seg,
                              "a": a})
        else:
            start = self.end
        it = Item(start, n, epoch, turn, seg, a, b, t_ready, pcm=pcm)
        self.items.append(it)
        self.end = start + n / self.sr
        self.cur = key
        return it

    def audible(self, t: float) -> bool:
        return self.end is not None and self.end > t

    def render(self, t0: float, total_s: float):
        """The played agent channel on the session clock (t0 = wall time of user-audio sample 0)."""
        import numpy as np
        out = np.zeros(int(total_s * self.sr) + 1, np.float32)
        for it in self.items:
            if it.dropped or it.pcm is None:
                continue
            x = it.pcm if it.keep < 0 else it.pcm[: it.keep]
            i = int(round((it.t_start - t0) * self.sr))
            if i < 0 or i >= len(out):
                continue
            x = x[: len(out) - i]
            out[i: i + len(x)] += x
        return out

    def per_turn(self) -> dict:
        res: dict = {}
        for it in self.items:
            if it.dropped or it.keep == 0:
                continue
            r = res.setdefault(it.turn, {"first_play": it.t_start, "first_ready": it.t_ready, "samples": 0})
            r["samples"] += it.n if it.keep < 0 else it.keep
        return res


class FlashProc:  # pragma: no cover - process plumbing, exercised on the GPU host
    def __init__(self, argv: list, env: dict, log_path: str):
        self.argv, self.env, self.log_path = argv, env, log_path
        self.events: queue.Queue = queue.Queue()
        self.sent: list = []
        self.p = None
        self.ready = None
        self.stats = None

    def start(self):
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  env=self.env, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.events.put({"ev": "eof"})

    def wait_ready(self, timeout: float) -> dict:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            try:
                ev = self.events.get(timeout=1.0)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError(f"talker process exited rc={self.p.returncode}")
                continue
            if ev.get("ev") == "ready":
                self.ready = ev
                return ev
            if ev.get("ev") == "eof":
                raise RuntimeError("talker process closed its output before ready")
        raise TimeoutError("talker not ready")

    def _send(self, obj):
        obj["t_ns"] = time.monotonic_ns()
        self.sent.append(obj)
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def submit(self, cmd) -> None:   # TalkerDriver sink
        if isinstance(cmd, StartSegment):
            self._send({"op": "start", "epoch": cmd.epoch, "turn": cmd.turn, "seg": cmd.segment, "text": cmd.text})
        elif isinstance(cmd, AppendText):
            self._send({"op": "append", "epoch": cmd.epoch, "turn": cmd.turn, "seg": cmd.segment, "text": cmd.text})
        elif isinstance(cmd, FinalizeSegment):
            self._send({"op": "final", "epoch": cmd.epoch, "turn": cmd.turn, "seg": cmd.segment})
        elif isinstance(cmd, Cancel):
            self._send({"op": "cancel", "epoch": cmd.epoch, "new_epoch": cmd.new_epoch})

    def poll(self) -> list:
        import numpy as np
        out = []
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return out
            if ev.get("ev") == "audio":
                ev["pcm"] = np.frombuffer(base64.b64decode(ev.pop("pcm16")), dtype=np.int16).astype(np.float32) / 32767
            elif ev.get("ev") == "stats":
                self.stats = ev
            out.append(ev)

    def close(self, timeout: float = 20.0):
        try:
            self._send({"op": "quit"})
        except Exception:  # noqa: BLE001
            pass
        t_end = time.monotonic() + timeout
        rest = []
        while time.monotonic() < t_end and self.stats is None:
            rest += self.poll()
            time.sleep(0.05)
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.terminate()
        self.log.close()
        return rest
