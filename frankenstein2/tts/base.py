"""Talker plugin interface and a fake for tests.

A talker instance serves ONE session (language + voice fixed at open); several instances may share one process /
model (Pocket keeps per-language models and voice states; pocket_swap will manage them as LRU language slots).

  sr, chunking            output sample rate; "sentence" | "whole" | "progressive" (text/chunker.py)
  say(sid, text, epoch)   synthesise one segment (already normalised); segments of a session are produced in order
  cancel(epoch)           drop every queued / running segment of an epoch below ``epoch`` (barge-in: the engine
                          also flushes its playout queue, so nothing already generated is ever played)
  poll() -> [events]      {"ev": "audio", "sid", "pcm": float32 array at ``sr``} | {"ev": "done", "sid", ...}
                          | {"ev": "error", "sid", "error"}
  close()
Segment ids are strings "epoch:turn:seg:lane" so a reply can be routed and checked for staleness.
"""

from __future__ import annotations

from typing import Protocol


class Talker(Protocol):
    sr: int
    chunking: str

    def say(self, sid: str, text: str, epoch: int) -> None: ...
    def cancel(self, epoch: int) -> None: ...
    def poll(self) -> list: ...
    def close(self) -> None: ...


def seg_id(epoch: int, turn: int, seg: int, lane: str) -> str:
    return f"{epoch}:{turn}:{seg}:{lane}"


def parse_sid(sid: str):
    e, t, s, lane = sid.split(":")
    return int(e), int(t), int(s), lane


class FakeTalker:
    """Text -> a tone of ``s_per_char`` seconds per character, delivered in ``chunk`` sample pieces, one per poll
    (``chunks_per_poll`` > 1 = faster than real time). Records every request."""

    def __init__(self, sr: int = 24000, s_per_char: float = 0.06, chunk: int = 1920, chunks_per_poll: int = 3,
                 chunking: str = "sentence", lang: str = "en"):
        self.sr, self.s_per_char, self.chunk, self.cpp = sr, s_per_char, chunk, chunks_per_poll
        self.chunking, self.lang = chunking, lang
        self.q: list = []
        self.said: list = []
        self.cancels: list = []
        self.min_epoch = 0

    def say(self, sid, text, epoch):
        import numpy as np
        if epoch < self.min_epoch:
            return
        n = max(self.chunk, int(len(text) * self.s_per_char * self.sr))
        amp = 0.1 + 0.01 * (hash(self.lang) % 10)
        pcm = (amp * np.sin(np.arange(n) * 2 * np.pi * 220 / self.sr)).astype(np.float32)
        self.said.append({"sid": sid, "text": text, "epoch": epoch, "lang": self.lang})
        self.q.append([sid, epoch, pcm, 0])

    def cancel(self, epoch):
        self.cancels.append(epoch)
        self.min_epoch = max(self.min_epoch, epoch)
        self.q = [x for x in self.q if x[1] >= epoch]

    def poll(self):
        out = []
        for _ in range(self.cpp):
            if not self.q:
                break
            sid, ep, pcm, pos = self.q[0]
            out.append({"ev": "audio", "sid": sid, "pcm": pcm[pos:pos + self.chunk]})
            self.q[0][3] = pos + self.chunk
            if self.q[0][3] >= len(pcm):
                self.q.pop(0)
                out.append({"ev": "done", "sid": sid})
        return out

    def close(self):
        self.q = []
