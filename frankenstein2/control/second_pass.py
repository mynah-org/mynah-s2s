"""Second ASR pass on an uncertain turn. Model-free; unit-tested.

The streaming ASR sometimes drops the first words of a turn ("Perfekt, vielen Dank und auf Wiedersehen" heard as "und
auf Wiedersehen"): sporadic, in any language, and the same audio transcribed on its own is heard whole
(frankenstein2/lab/asr_onset_replay.py, asr_silence_probe.py). The ears tell when it happens: the user's channel was
voiced for a while before the first transcribed piece, or a turn ends with voice but no text. On such a turn the
pipeline transcribes the turn's own audio offline (one request to the ears, tens of ms) and holds the endpoint until
the transcript is back or a short deadline passes; the fuller of the two texts goes to the LLM.

    Suspicion   per turn: voiced listen-frames (ears frame dB) and the first piece's frame -> a reason or None
    TurnAudio   the user's recent audio by 80 ms step, to cut the turn out
    choose      the streaming text or the offline one
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

STEP = 1280                                  # 80 ms at 16 kHz, one engine step = one ears frame


@dataclass
class SecondPassConfig:
    enabled: bool = True
    voiced_db: float = -42.0                 # an ears frame louder than this (listen mode) is voiced
    lag_frames: int = 3                      # the streaming ASR's normal delay before a word's first piece
    min_unheard_frames: int = 5              # voiced frames before the first piece (minus the lag) = suspicious
    min_voiced_empty: int = 6                # an empty transcript after this many voiced frames = suspicious
    timeout_steps: int = 6                   # the longest the endpoint waits for the offline transcript (480 ms)
    pre_frames: int = 4                      # audio kept before the voiced onset
    keep_s: float = 30.0                     # user audio kept in memory


class Suspicion:
    """Voiced frames and transcribed pieces since the turn's boundary (the last text taken by a turn)."""

    def __init__(self, cfg: SecondPassConfig):
        self.cfg = cfg
        self.boundary = 0
        self.voiced: list = []               # frame indices, ascending
        self.first_piece = None

    def reset(self, boundary_frame: int) -> None:
        self.boundary = int(boundary_frame)
        self.voiced = [f for f in self.voiced if f >= self.boundary]
        self.first_piece = None

    def frame(self, k: int, db, mode: str = "listen") -> None:
        if mode == "listen" and db is not None and db > self.cfg.voiced_db and k >= self.boundary:
            self.voiced.append(int(k))

    def pieces(self, frames) -> None:
        for f in frames:
            if f >= self.boundary and (self.first_piece is None or f < self.first_piece):
                self.first_piece = int(f)

    def onset(self):
        return self.voiced[0] if self.voiced else None

    def verdict(self, text: str):
        """-> "empty_with_voice" | "unheard_onset" | None for the turn ending now with the streaming ``text``."""
        c = self.cfg
        if not _norm(text) and len(self.voiced) >= c.min_voiced_empty:
            return "empty_with_voice"
        if self.first_piece is not None:
            unheard = sum(1 for v in self.voiced if v < self.first_piece - c.lag_frames)
            if unheard >= c.min_unheard_frames:
                return "unheard_onset"
        return None


class TurnAudio:
    """The user's audio by step (16 kHz float32), the last ``keep_s`` seconds."""

    def __init__(self, keep_s: float = 30.0):
        self.keep = max(1, int(keep_s / 0.08))
        self.first = 0
        self.chunks: list = []

    def push(self, k: int, x) -> None:
        if not self.chunks:
            self.first = int(k)
        self.chunks.append(np.asarray(x, np.float32))
        if len(self.chunks) > self.keep:
            drop = len(self.chunks) - self.keep
            self.chunks = self.chunks[drop:]
            self.first += drop

    def cut(self, k0: int, k1: int) -> np.ndarray:
        """Steps [k0, k1) as one array (clipped to what is kept)."""
        a, b = max(int(k0), self.first), min(int(k1), self.first + len(self.chunks))
        if b <= a:
            return np.zeros(0, np.float32)
        return np.concatenate(self.chunks[a - self.first:b - self.first])


def _norm(s: str) -> str:
    return "".join(re.findall(r"\w", (s or "").lower()))


def choose(streaming: str, offline) -> tuple:
    """-> (text, which): the offline transcript when it heard clearly more than the stream (at least 3 more
    letters or digits), else the streaming text. A missing or empty offline transcript keeps the stream."""
    if not offline or not _norm(offline):
        return streaming, "streaming"
    if len(_norm(offline)) >= len(_norm(streaming)) + 3:
        return offline.strip(), "offline"
    return streaming, "streaming"
