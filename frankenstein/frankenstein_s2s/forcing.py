"""M1-force: the duplex loop decides WHEN, a text-mode pass of the same Thinker weights decides WHAT.

Per real 80 ms frame the duplex Thinker predicts an agent text id from the original user audio. That prediction is
kept only as a TIMING signal (BOS = start speaking, EOS = stop, PAD = keep quiet). The reply CONTENT comes from a
text-mode generation (Nano chat template, authoritative multilingual transcripts as user turns) and is teacher-forced
into the agent text channel one id per frame, the same way the wrapper already forces its own "redirect" phrase
(wrapper ``_apply_rnnt_turn_taking``: ``gen_text[b, t] = next_tok`` from ``_redirect_tokens_queue``). The forced id is
what the duplex loop feeds back at the next frame, so its state sees the reply it "said".

``AgentChannelForcer`` is the per-frame decision, pure Python:

  LISTEN    model BOS            -> keep BOS, request a reply                     -> WAIT_TEXT / FORCING
            model text (no BOS)  -> force PAD (the duplex loop's own content is never spoken)
            user EOU + no BOS for ``bos_fallback_frames`` -> force BOS (timing fallback)
  WAIT_TEXT reply not ready      -> force PAD (adapter sees "no text": nothing is committed)  ... cap -> force EOS
  FORCING   next reply id        -> force it (override counted when the model wanted something else)
            reply ids exhausted, generation still running -> force PAD, adapter gets "no text" (underrun)
            cut() (barge-in)     -> force EOS, drop the rest of the reply
  TAIL      reply complete       -> keep the model's PAD / EOS; any other text -> PAD; ``tail_max`` -> force EOS

Each step returns ``ForcedStep(write, to_adapter)``: ``write`` = id to place in ``gen_text[b, t]`` (None = keep the
model's prediction), ``to_adapter`` = the text-channel id the talker adapter must see for this frame (None = no text,
which never triggers a burst commit).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from . import tokens as T
from .tracing import NullTracer


class ForceState(Enum):
    LISTEN = "listen"
    WAIT_TEXT = "wait_text"
    FORCING = "forcing"
    TAIL = "tail"


@dataclass(frozen=True)
class ForcedStep:
    write: Optional[int]
    to_adapter: Optional[int]


@dataclass
class AgentChannelForcer:
    bos_fallback_frames: int = 6     # after the user's EOU, force BOS if the duplex loop has not opened a turn
    wait_max: int = 25               # 2 s without reply text after BOS -> give up the turn (EOS)
    tail_max: int = 60               # PAD tail cap after the forced reply
    eos_hold: Optional[int] = None   # controller-owned EOS: None = wait for the Thinker's EOS (up to tail_max);
                                     # n = force EOS n PAD frames after the last forced id (0 = the next frame)
    state: ForceState = ForceState.LISTEN
    queue: deque = field(default_factory=deque)
    reply_done: bool = False
    need_reply: bool = False         # set at BOS; the runner starts / attaches the text-mode generation
    eou_k: Optional[int] = None
    wait: int = 0
    tail: int = 0
    turn_ids: list = field(default_factory=list)   # ids actually forced in the current turn (for history on a cut)
    overrides: int = 0
    agreements: int = 0
    log: list = field(default_factory=list)
    tracer: object = field(default_factory=NullTracer)
    _cut: bool = False

    # ------------------------------------------------------------------------------ inputs from the runner
    def on_eou(self, k: int) -> None:
        """External end of user utterance (multilingual ASR TurnTaker)."""
        self.eou_k = k

    def add_reply(self, ids: list, done: bool = False) -> None:
        """Text-mode ids as they are generated (specials and id 12 must already be banned / stripped)."""
        for i in ids:
            if i < T.FIRST_TEXT_ID:
                raise ValueError(f"special id {i} in a forced reply")
            self.queue.append(i)
        self.reply_done = self.reply_done or done

    def cut(self) -> None:
        """Barge-in decided outside (adapter / TurnTaker): stop the reply at the next frame (also in the PAD tail)."""
        if self.state in (ForceState.WAIT_TEXT, ForceState.FORCING, ForceState.TAIL):
            self._cut = True

    # ------------------------------------------------------------------------------ per frame
    def step(self, k: int, predicted: int) -> ForcedStep:
        s = self.state
        if s is ForceState.LISTEN:
            if predicted == T.BOS_ID:
                return self._open(k, write=None, why="model_bos")
            if self.eou_k is not None and k - self.eou_k >= self.bos_fallback_frames:
                return self._open(k, write=T.BOS_ID, why="eou_fallback")
            if T.kind_of(predicted) == "tok":
                self._override(k, predicted, T.PAD_ID, "model_text_while_listening")
                return ForcedStep(T.PAD_ID, T.PAD_ID)
            if predicted == T.EOS_ID:
                return ForcedStep(T.PAD_ID, T.PAD_ID)
            return ForcedStep(None, predicted)

        if self._cut:
            return self._close(k, "barge_in")

        if s is ForceState.WAIT_TEXT:
            if self.queue:
                self.state = ForceState.FORCING
                return self._force_next(k, predicted)
            self.wait += 1
            if self.wait > self.wait_max:
                return self._close(k, "no_reply_text")
            return ForcedStep(T.PAD_ID if predicted != T.PAD_ID else None, None)

        if s is ForceState.FORCING:
            if self.queue:
                return self._force_next(k, predicted)
            if self.reply_done:
                self.state = ForceState.TAIL
                return self._tail(k, predicted)
            self.log.append((k, "reply_underrun", "PAD to the model, no text to the adapter"))
            self.tracer.emit("force.underrun", k, predicted=predicted)
            return ForcedStep(T.PAD_ID if predicted != T.PAD_ID else None, None)

        if self._cut:
            return self._close(k, "barge_in")
        return self._tail(k, predicted)

    # ------------------------------------------------------------------------------ helpers
    def _open(self, k: int, write: Optional[int], why: str) -> ForcedStep:
        self.state = ForceState.WAIT_TEXT if not self.queue else ForceState.FORCING
        self.need_reply = True
        self.eou_k = None
        self.wait = self.tail = 0
        self.turn_ids = []
        self.log.append((k, "turn_open", why))
        self.tracer.emit("force.turn_open", k, source=why, reply_ready=len(self.queue))
        return ForcedStep(write, T.BOS_ID)

    def _force_next(self, k: int, predicted: int) -> ForcedStep:
        tid = self.queue.popleft()
        self.turn_ids.append(tid)
        if predicted == tid:
            self.agreements += 1
            return ForcedStep(None, tid)
        self._override(k, predicted, tid, "forced_reply_token")
        return ForcedStep(tid, tid)

    def _tail(self, k: int, predicted: int) -> ForcedStep:
        self.tail += 1
        if predicted == T.EOS_ID:
            return self._close(k, "model_eos", write=None)
        if self.eos_hold is not None and self.tail > self.eos_hold:
            return self._close(k, "controller_eos")
        if self.tail > self.tail_max:
            return self._close(k, "tail_cap")
        if predicted != T.PAD_ID:
            self._override(k, predicted, T.PAD_ID, "model_text_after_reply")
            return ForcedStep(T.PAD_ID, T.PAD_ID)
        return ForcedStep(None, T.PAD_ID)

    def _close(self, k: int, why: str, write: Optional[int] = T.EOS_ID) -> ForcedStep:
        dropped = len(self.queue)
        self.queue.clear()
        self.state = ForceState.LISTEN
        self.reply_done = self.need_reply = self._cut = False
        self.log.append((k, "turn_close", why, {"dropped_reply_ids": dropped, "forced": len(self.turn_ids)}))
        self.tracer.emit("force.turn_close", k, why=why, dropped_reply_ids=dropped, forced=len(self.turn_ids),
                         wrote=write)
        return ForcedStep(write, T.EOS_ID)

    def _override(self, k: int, predicted: int, forced: int, why: str) -> None:
        self.overrides += 1
        self.log.append((k, why, f"model {predicted} -> {forced}"))
        self.tracer.emit("force.override", k, predicted=predicted, forced=forced, why=why)
