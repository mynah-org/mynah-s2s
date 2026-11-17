"""Word-level commits and prefix-stable re-tokenization (ported from the transplant replay planner).

The Thinker writes Nemotron sub-word ids; the talker reads Qwen ids. Text is moved between the two only as
COMPLETE words, so nothing that was handed to the talker ever has to be taken back:

* a word is committed at the frame where the next separator arrives (space or punctuation);
* the remaining tail is committed only when the text is declared final (burst commit, EOS, tool call, segment cut);
* Qwen ids for a commit are the prefix-diff of encoding the whole segment text so far, so the talker sees the same
  ids it would see if the segment had been encoded at once (when the tokenizer is prefix-stable at word boundaries;
  when it is not, the old prefix is kept and the event is logged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

PUNCT = ".,;:!?"
SENTENCE_END = ".!?"


def complete_word_cut(new: str) -> int:
    """Index up to which ``new`` (uncommitted text) consists of complete words; 0 = nothing complete yet."""
    space = new.rfind(" ")
    punct = max((new.rfind(p) for p in PUNCT), default=-1) + 1
    return max(space, punct, 0)


@dataclass
class WordCommitter:
    """Tracks one turn's cumulative decoded text and how much of it was committed."""

    text: str = ""
    committed: int = 0
    rewrites: int = 0

    def update(self, text: str) -> str:
        """New cumulative text; returns the newly complete-word chunk (may be '')."""
        if not text.startswith(self.text[: self.committed]):
            # The detokenizer rewrote already-committed text (should not happen with byte-level BPE). Keep the commit
            # point; the caller logs it through ``rewrites``.
            self.rewrites += 1
        self.text = text
        new = text[self.committed:]
        cut = complete_word_cut(new)
        if cut <= 0:
            return ""
        self.committed += cut
        return new[:cut]

    def flush(self) -> str:
        """Commit everything left (the text is final)."""
        rest = self.text[self.committed:]
        self.committed = len(self.text)
        return rest

    @property
    def pending(self) -> str:
        return self.text[self.committed:]


@dataclass
class Retokenizer:
    """Prefix-diff encoder for one talker segment."""

    encode: Optional[Callable[[str], list]] = None
    text: str = ""
    ids: list = field(default_factory=list)
    prefix_changes: int = 0

    def add(self, chunk: str) -> tuple:
        if self.encode is None:
            self.text += chunk
            return ()
        full = self.encode(self.text + chunk)
        if full[: len(self.ids)] != self.ids:
            self.prefix_changes += 1
            # keep what the talker already has; append only the part after the longest common prefix of the text
            fresh = self.encode(chunk)
            self.ids = self.ids + list(fresh)
            self.text += chunk
            return tuple(fresh)
        new = full[len(self.ids):]
        self.ids = list(full)
        self.text += chunk
        return tuple(new)
