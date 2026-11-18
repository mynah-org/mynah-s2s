"""Reply text -> talker segments, per talker policy, normalised for speech.

  "sentence"     each complete sentence is a segment (a sentence ends at . ! ? … followed by a space, or at the end of
                 the reply; "3.5" or "49,90" never end one); a sentence longer than ``max_chars`` is cut at the last
                 clause mark (, ; :) or space before it. Whole-text talkers (Pocket, VoxCPM2).
  "whole"        the whole reply is one segment, at the end (best prosody, slowest first audio).
  "progressive"  text is committed as it grows, up to ``split_safe`` (never inside a number / price / time / code),
                 in pieces of at least ``min_piece`` characters (Flash / Qwen3-TTS streaming text input).
Every segment is normalised (numbers, prices, times, dates, codes as words in the session language) BEFORE any
other text clean-up, then stripped of markdown symbols. Model-free; unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import LANGS, normalize, split_safe

_END = re.compile(r"[.!?…]+[\"')\]]*\s")


def clean_for_speech(text: str) -> str:
    t = re.sub(r"[*_#`>|]+", " ", text)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    return re.sub(r"\s{2,}", " ", t).strip()


def speech_text(text: str, lang: str) -> str:
    t = normalize(text, lang) if lang in LANGS else text
    return clean_for_speech(t)


@dataclass
class Chunker:
    lang: str
    mode: str = "sentence"
    max_chars: int = 220
    min_piece: int = 12
    buf: str = ""
    out: list = field(default_factory=list)          # (original text, speech text)

    def __post_init__(self):
        if self.mode not in ("sentence", "whole", "progressive"):
            raise ValueError(self.mode)

    def _emit(self, raw: str) -> list:
        raw = raw.strip()
        if not raw:
            return []
        seg = (raw, speech_text(raw, self.lang))
        self.out.append(seg)
        return [seg]

    def feed(self, delta: str) -> list:
        """More reply text -> the segments now complete."""
        self.buf += delta
        if self.mode == "whole":
            return []
        res = []
        if self.mode == "sentence":
            while True:
                m = _END.search(self.buf)
                if m and m.end() <= self.max_chars:
                    res += self._emit(self.buf[:m.end()])
                    self.buf = self.buf[m.end():]
                    continue
                if len(self.buf) > self.max_chars:
                    cut = max(self.buf.rfind(c, 0, self.max_chars) for c in ",;:")
                    if cut <= 0:
                        cut = self.buf.rfind(" ", 0, self.max_chars)
                    if cut > 0:
                        res += self._emit(self.buf[:cut + 1])
                        self.buf = self.buf[cut + 1:]
                        continue
                break
            return res
        i = split_safe(self.buf)                           # progressive
        if i >= self.min_piece:
            res += self._emit(self.buf[:i])
            self.buf = self.buf[i:]
        return res

    def flush(self) -> list:
        raw, self.buf = self.buf, ""
        return self._emit(raw)
