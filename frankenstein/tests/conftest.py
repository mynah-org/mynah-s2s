"""Synthetic VoiceChat token streams and fake tokenizers (no model, no torch)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frankenstein_s2s import tokens as T  # noqa: E402
from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter  # noqa: E402
from frankenstein_s2s.driver import FakeTalkerDriver, PlaybackGate  # noqa: E402
from frankenstein_s2s.events import Frame  # noqa: E402


class FakeVocab:
    """Sub-word pieces <-> ids >= 1000 (Nemotron-like: a leading space marks a word start)."""

    def __init__(self):
        self.piece_to_id: dict = {}
        self.id_to_piece: dict = {}

    def id(self, piece: str) -> int:
        if piece not in self.piece_to_id:
            i = T.FIRST_TEXT_ID + len(self.piece_to_id)
            self.piece_to_id[piece] = i
            self.id_to_piece[i] = piece
        return self.piece_to_id[piece]

    def decode(self, ids) -> str:
        return "".join(self.id_to_piece[i] for i in ids if i >= T.FIRST_TEXT_ID)


def qwen_encode(text: str) -> list:
    """Prefix-stable fake Qwen tokenizer: one id per (leading-space + word) or punctuation mark."""
    return [hash_id(p) for p in re.findall(r"\s?[^\s.,;:!?]+|[.,;:!?]", text)]


_QIDS: dict = {}


def hash_id(piece: str) -> int:
    return _QIDS.setdefault(piece, 50000 + len(_QIDS))


SPECIAL = {"BOS": T.BOS_ID, "EOS": T.EOS_ID, "PAD": T.PAD_ID, "MISS": None}
FUNC = {"SOTC": T.SOTC_ID, "EOTC": T.EOTC_ID, "EOTR": T.EOTR_ID}


def stream(vocab: FakeVocab, spec: list, start_k: int = 0, speech: frozenset = frozenset()) -> list:
    """Build frames from a compact spec: "BOS", "EOS", "PAD", "PAD*5", "MISS", "f:SOTC" (function channel, text PAD),
    or a text piece like " Your" (one token per frame, as VoiceChat emits its burst)."""
    frames, k = [], start_k
    for item in spec:
        m = re.fullmatch(r"(PAD|MISS)\*(\d+)", item)
        if m:
            for _ in range(int(m.group(2))):
                frames.append(Frame(k, SPECIAL[m.group(1)], user_speech=k in speech))
                k += 1
            continue
        if item in SPECIAL:
            frames.append(Frame(k, SPECIAL[item], user_speech=k in speech))
        elif item.startswith("f:"):
            frames.append(Frame(k, T.PAD_ID, func_id=FUNC[item[2:]], user_speech=k in speech))
        else:
            frames.append(Frame(k, vocab.id(item), user_speech=k in speech))
        k += 1
    return frames


class Rig:
    """Adapter + fake talker + playback gate wired as in the engine."""

    def __init__(self, **cfg):
        self.vocab = FakeVocab()
        self.driver = FakeTalkerDriver()
        self.gate = PlaybackGate()
        self.adapter = TalkerAdapter([self.driver, self.gate], decode=self.vocab.decode, encode=qwen_encode,
                                     config=AdapterConfig(**cfg))

    def run(self, spec, start_k=0, speech=frozenset(), pump=False):
        frames = stream(self.vocab, spec, start_k, speech)
        for f in frames:
            self.adapter.feed(f)
            if pump:
                self.pump()
        return frames[-1].k + 1 if frames else start_k

    def pump(self):
        """Talker produces what it can; the transport delivers it; playback consumes it."""
        self.driver.produce()
        for c in self.driver.take():
            self.gate.accept(c)
        for d in self.driver.done:
            self.adapter.talker_done(d)
        self.driver.done.clear()
        self.gate.play_all()

    def cmds(self, kind=None):
        return [c for c in self.adapter.commands if kind is None or type(c).__name__ == kind]

    def logs(self, what=None):
        return [e for e in self.adapter.log if what is None or e.what == what]

    def spoken_text(self, epoch=None):
        out = []
        for c in self.adapter.commands:
            if type(c).__name__ in ("StartSegment", "AppendText") and (epoch is None or c.epoch == epoch):
                out.append(c.text)
        return "".join(out)


@pytest.fixture
def rig():
    return Rig()


@pytest.fixture
def make_rig():
    return Rig
