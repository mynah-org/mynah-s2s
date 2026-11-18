"""Fakes for the CPU tests: an ears backend driven by a script, a multi-session talker backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.tts.base import FakeTalker  # noqa: E402


class FakeEarsBackend:
    """Shared by several sessions (like the ears process). ``script(sid, n, mode, reset, u, a) -> dict(events,
    pieces)``; events / pieces carry absolute frames. Replies are delivered on the next poll (or ``delay`` polls
    later). Records every hop it receives."""

    def __init__(self, script, delay: int = 0):
        self.script, self.delay = script, delay
        self.boxes: dict = {}
        self.hops: list = []
        self.opened: dict = {}
        self.closed: list = []

    def send(self, m):
        sid = m.get("sid")
        if m["op"] == "open":
            self.opened[sid] = m
        elif m["op"] == "close":
            self.closed.append(sid)
        elif m["op"] == "hop":
            import base64
            u = np.frombuffer(base64.b64decode(m["u"]), "<f4")
            a = np.frombuffer(base64.b64decode(m["a"]), "<f4")
            self.hops.append({"sid": sid, "n": m["n"], "mode": m["mode"], "reset": m["reset"],
                              "u_rms": float(np.sqrt((u ** 2).mean())), "a_rms": float(np.sqrt((a ** 2).mean()))})
            r = self.script(sid, m["n"], m["mode"], m["reset"], u, a) or {}
            msg = {"ev": "hop", "sid": sid, "n": m["n"], "frames": [], "events": r.get("events", []),
                   "pieces": r.get("pieces", []), "ms": 1.0}
            self.boxes.setdefault(sid, []).append([self.delay, msg])

    def poll(self, sid, block=False):
        box = self.boxes.get(sid, [])
        out = []
        for item in list(box):
            if item[0] <= 0:
                out.append(item[1])
                box.remove(item)
            else:
                item[0] -= 1
        return out


def pieces(frame: int, text: str):
    """Words -> SentencePiece-like pieces, one word per frame from ``frame``."""
    return [{"f": frame + i, "p": "▁" + w, "id": 0} for i, w in enumerate(text.split())]


class SharedTalkerBackend:
    """One 'process' for several sessions: per-session FakeTalker state, generation interleaved one chunk at a
    time across sessions (like segments of different sessions sharing a GPU)."""

    def __init__(self):
        self.talkers: dict = {}

    def talker(self, ss, lang, **kw):
        t = FakeTalker(lang=lang, **kw)
        self.talkers[ss] = t
        return t
