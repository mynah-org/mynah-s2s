"""User side: streaming-ASR pieces -> one Thinker token per 80 ms frame, plus turn-taking.

The multilingual streaming ASR (cache-aware FastConformer RNNT, 80 ms encoder frame = the Thinker step) emits 0..n
sub-word pieces per frame from its own vocabulary. The Thinker reads Nemotron ids. This module:

* rebuilds words from pieces (a piece starting with the word marker opens a new word);
* closes a word when the next word starts, or ``close_after`` frames after its last piece;
* re-tokenizes each closed word with the Thinker tokenizer (as ' word') and queues the ids FIFO;
* releases at most ONE id per frame (``step``: per-frame channel feeding, M1_CHAN) or hands each closed word to
  ``on_close`` (M1_CTX: ``TranscriptInserter`` turns closed words into inserted steps);
* drops language tags (``<it-IT>``), standalone late punctuation, and any id in the special range (< 1000);
* exposes ``frame_inputs``: what each fusion term of the Thinker receives in each controlled mode M0-M3.

Causality: a word is released only after it is closed, so the id injected at frame k never depends on audio after
frame k (plus the ASR's own right context). The streaming RNNT is greedy and monotonic: emitted pieces are never
retracted, so nothing injected ever needs to be taken back (a revising ASR would need a stable-prefix policy first).

``TurnTaker`` mirrors the wrapper's RNNT counters (EOU after N blank frames, BOU / barge-in after M speech frames) on
the external ASR's emissions; it is needed for barge-in evidence in every mode and for the turn itself in M3.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from . import tokens as T
from .tracing import NullTracer

WORD_MARKERS = ("▁", " ")
LANG_TAG = re.compile(r"^<[a-z]{2,3}(-[A-Za-z]{2,4})?>$")
PUNCT_ONLY = re.compile(r"^[\s▁]*[.,;:!?¿¡]+$")


@dataclass
class UserTextScheduler:
    encode_word: Callable[[str], list]   # Thinker tokenizer: ' word' -> ids (no special tokens)
    close_after: int = 3
    queue: deque = field(default_factory=deque)
    word: str = ""
    last_piece_k: Optional[int] = None
    dropped: list = field(default_factory=list)
    words: list = field(default_factory=list)
    on_close: Optional[Callable[[int, str, list], None]] = None   # (frame, word, kept ids): feeds TranscriptInserter
    tracer: object = field(default_factory=NullTracer)

    def on_asr(self, k: int, pieces: list) -> None:
        """Pieces emitted by the streaming ASR at frame k (in order)."""
        for p in pieces:
            if LANG_TAG.match(p.strip()):
                self.dropped.append((k, p, "lang_tag"))
                self.tracer.emit("asr.drop", k, piece=p, why="lang_tag")
                continue
            if PUNCT_ONLY.match(p):
                if self.word:
                    self.word += p.strip("▁ ")
                else:
                    self.dropped.append((k, p, "late_punctuation"))
                    self.tracer.emit("asr.drop", k, piece=p, why="late_punctuation")
                continue
            if p.startswith(WORD_MARKERS):
                self._close(k)
                self.word = p.lstrip("▁ ")
            else:
                self.word += p
            self.last_piece_k = k

    def step(self, k: int) -> Optional[int]:
        """Call once per Thinker frame (after on_asr for that frame). Returns the id to inject now, or None."""
        if self.word and self.last_piece_k is not None and k - self.last_piece_k >= self.close_after:
            self._close(k)
        return self.queue.popleft() if self.queue else None

    def flush(self, k: int) -> None:
        """End of user turn (frame k): close the open word now."""
        self._close(k)

    def _close(self, k: int) -> None:
        if not self.word:
            return
        word, kept = self.word, []
        self.words.append(word)
        for i in self.encode_word(" " + word):
            if i < T.FIRST_TEXT_ID:
                self.dropped.append((k, i, "special_id"))
            else:
                kept.append(i)
        self.queue.extend(kept)
        self.word = ""
        self.tracer.emit("asr.word", k, word=word, ids=kept)
        if self.on_close is not None and kept:
            self.on_close(k, word, kept)

    @property
    def backlog(self) -> int:
        return len(self.queue)


@dataclass
class TurnTaker:
    """External turn-taking on ASR activity (non-blank = at least one piece emitted at that frame).

    eou_frames         blank frames after confirmed speech -> user end of utterance (force agent BOS)
    bou_frames         consecutive speech frames -> user speech confirmed (barge-in evidence)
    The wrapper's defaults (code) are eou 10 / bou 3; the shipped streaming config uses 40 / 40 (3.2 s)."""

    eou_frames: int = 10
    bou_frames: int = 3
    run: int = 0
    blank: int = 0
    speech_confirmed: bool = False
    tracer: object = field(default_factory=NullTracer)

    def step(self, k: int, nonblank: bool) -> list:
        events = []
        if nonblank:
            self.run += 1
            self.blank = 0
            if not self.speech_confirmed and self.run >= self.bou_frames:
                self.speech_confirmed = True
                events.append("bou")
                self.tracer.emit("turn.bou", k, source="asr", run_frames=self.run)
        else:
            self.run = 0
            self.blank += 1
            if self.speech_confirmed and self.blank >= self.eou_frames:
                self.speech_confirmed = False
                events.append("eou")
                self.tracer.emit("turn.eou", k, source="asr", blank_frames=self.blank)
        return events

    @property
    def user_speaking(self) -> bool:
        return self.speech_confirmed


class InjectionMode(Enum):
    """Controlled modes of the user side (DESIGN.md "Controlled modes"). Agent-text and function terms are unchanged
    on real frames; inserted steps carry agent PAD and function PAD."""

    M0_BASELINE = "m0"   # original perception, no transcript (what the released checkpoint runs)
    M1_FORCE = "m1force" # PRIMARY: original perception; agent channel teacher-forced with the text-mode reply
                         # (forcing.AgentChannelForcer); the transcript reaches only the text-mode brain
    M1_CTX = "m1ctx"     # PRIMARY candidate: original perception on real frames + transcript INSERTED as extra
                         # steps (embed_tokens(id) in the user-audio slot, agent PAD: the system-prompt layout)
    M1_CHAN = "m1chan"   # transcript on the 4th fusion term (user-text channel); untrained in this checkpoint:
                         # documented negative zero-shot, kept only for completeness
    M2_DAMPED = "m2"     # M1_CTX with the perception term scaled down on real frames (only if M1_CTX shows interference)
    M3_BYPASS = "m3"     # silence embedding on real frames + M1_CTX inserts + external turn-taking (experimental)


def frame_inputs(mode: InjectionMode, user_token: Optional[int], inserted: bool = False,
                 damped_audio_weight: float = 0.5) -> dict:
    """Contract of one Thinker input step on the user side.

    ``inserted`` = an extra step that consumes NO audio frame (like the wrapper's FC steps); only M1_CTX / M2 / M3
    have inserted steps, and only they may carry a transcript id there. ``user_token`` must have been produced from
    audio up to the current frame (causal).

    Returns {"user_audio": "perception" | "silence" | ("token", id), "audio_weight": float,
             "user_text": None | ("token", id), "user_text_weight": float, "agent": "feedback" | "pad"}.
    "silence" = the perception output for digital silence (the wrapper's cached FC silence embedding)."""
    if user_token is not None and user_token < T.FIRST_TEXT_ID:
        raise ValueError("special ids are never injected on the user side")
    tok = ("token", user_token) if user_token is not None else None
    if inserted:
        if mode not in (InjectionMode.M1_CTX, InjectionMode.M2_DAMPED, InjectionMode.M3_BYPASS) or tok is None:
            raise ValueError("inserted steps exist only in M1_CTX / M2 / M3 and always carry a transcript id")
        return {"user_audio": tok, "audio_weight": 1.0, "user_text": None, "user_text_weight": 0.0, "agent": "pad"}
    if tok is not None and mode is not InjectionMode.M1_CHAN:
        raise ValueError(f"{mode.name}: transcript ids go into inserted steps, never onto a real audio frame")
    base = {"user_audio": "perception", "audio_weight": 1.0, "user_text": None, "user_text_weight": 0.0,
            "agent": "feedback"}
    if mode is InjectionMode.M1_FORCE:
        base.update(agent="forced")
    elif mode is InjectionMode.M1_CHAN:
        base.update(user_text=tok, user_text_weight=1.0 if tok else 0.0)
    elif mode is InjectionMode.M2_DAMPED:
        base.update(audio_weight=damped_audio_weight)
    elif mode is InjectionMode.M3_BYPASS:
        base.update(user_audio="silence")
    return base


@dataclass(frozen=True)
class InsertSpan:
    """Transcript ids to insert as extra Thinker steps right after real frame ``after_k`` (no audio consumed)."""

    after_k: int
    ids: tuple
    words: tuple
    reason: str          # "word" (incremental) | "eou" (end of user utterance)


@dataclass
class TranscriptInserter:
    """Plans M1_CTX insertions from closed ASR words. Causal by construction: a word can only be inserted after the
    frame at which it was closed; nothing is ever inserted for words not yet heard.

    policy "eou":  insert the whole utterance once, at the external end-of-utterance frame (one span per user turn);
    policy "word": insert each closed word as soon as possible (earlier, more spans, more inserted steps).
    ``prefix_ids`` / ``suffix_ids`` wrap a span (e.g. the prompt layout's <s> ... </s>, or a "User:" marker); which
    wrapping the Thinker reads best is an open question for the GPU gate."""

    policy: str = "eou"
    prefix_ids: tuple = ()
    suffix_ids: tuple = ()
    pending_ids: list = field(default_factory=list)
    pending_words: list = field(default_factory=list)
    last_word_k: Optional[int] = None
    spans: list = field(default_factory=list)
    tracer: object = field(default_factory=NullTracer)

    def __post_init__(self):
        if self.policy not in ("eou", "word"):
            raise ValueError("policy must be 'eou' or 'word'")

    def on_word(self, k: int, word: str, ids: list) -> Optional[InsertSpan]:
        """A word closed at frame k with its Thinker ids."""
        if any(i < T.FIRST_TEXT_ID for i in ids):
            raise ValueError("special ids are never inserted")
        self.pending_ids += list(ids)
        self.pending_words.append(word)
        self.last_word_k = k
        if self.policy == "word":
            return self._emit(k, "word")
        return None

    def on_eou(self, k: int) -> Optional[InsertSpan]:
        if self.last_word_k is not None and k < self.last_word_k:
            raise ValueError("end of utterance before the last word closed")
        return self._emit(k, "eou") if self.pending_ids else None

    @property
    def hold_bos(self) -> bool:
        """True while heard words are not yet in the Thinker context: an agent BOS now would answer without them."""
        return bool(self.pending_ids)

    def _emit(self, k: int, reason: str) -> InsertSpan:
        span = InsertSpan(k, tuple(self.prefix_ids) + tuple(self.pending_ids) + tuple(self.suffix_ids),
                          tuple(self.pending_words), reason)
        self.pending_ids, self.pending_words = [], []
        self.spans.append(span)
        self.tracer.emit("insert.span", k, after_k=k, ids=span.ids, words=span.words, reason=reason)
        return span
