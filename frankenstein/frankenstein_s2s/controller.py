"""Controller: the single writer of turn state (CONTROL.md).

Sources post INTENTS; the controller turns them into ACTIONS. Nothing else changes turn state.

* Every intent carries its source, a model-frame stamp ``k`` (Thinker / ASR: the frame it refers to; Nano / tools /
  talker / app: the frame at which it was delivered to the controller) and, when it answers an earlier action, the
  ``epoch`` and ``request_id`` / ``call_id`` it belongs to.
* Intents are queued with ``post`` and decided once per frame with ``tick(k)``. Within a tick the order of arrival
  does not matter: intents are sorted by (priority, source rank, k, stable key) before they are applied, so the same
  set of intents always gives the same actions (race-free by construction, tested with permutations).
* Priority: barge-in (confirmed user speech) > cancel > tool (call / result) > speak (reply tokens) > the rest.
* Stale = an answer to a cancelled epoch, an old request or an unknown call: dropped and traced, never applied.
* Idempotent: an intent with an id already applied (``uid``) is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .tracing import NullTracer


class St(Enum):
    LISTENING = "listening"      # no agent turn
    PENDING = "pending"          # turn opened, reply requested, nothing spoken yet
    SPEAKING = "speaking"        # reply tokens are flowing to the talker
    TOOL_WAIT = "tool_wait"      # a tool call is out; the turn stays open
    DRAINING = "draining"        # G3 (audio_aware): the reply text is complete, its audio is still playing


# intent kinds -> priority (lower = first)
PRIORITY = {
    "USER_SPEECH_START": 0, "USER_SPEECH": 0,         # barge-in evidence
    "CANCEL": 1,
    "TOOL_DECIDED": 2, "TOOL_CALL": 2, "TOOL_RESULT": 2, "TOOL_TIMEOUT": 2,
    "REPLY_TOKENS": 3, "REPLY_DONE": 4,               # tokens of a frame before the end that follows them
    "YIELD": 5,
    "WORDS": 6,                                       # a word closed (before a new one opens in the same frame)
    "WORD_OPEN": 7,                                   # the ASR has emitted pieces of a word not closed yet
    "EOU": 8, "TAKE_TURN": 9, "USER_SPEECH_END": 10,
    "AUDIO_STARTED": 11, "AUDIO_ENDED": 11, "UNDERRUN": 11,
}
SOURCE_RANK = {"app": 0, "asr": 1, "tool": 2, "nano": 3, "thinker": 4, "talker": 5}


@dataclass(frozen=True)
class Intent:
    kind: str
    source: str
    k: int
    payload: Any = None
    epoch: Optional[int] = None
    request_id: Optional[int] = None
    call_id: Optional[str] = None
    uid: Optional[str] = None

    def __post_init__(self):
        if self.kind not in PRIORITY:
            raise ValueError(f"unknown intent {self.kind!r}")
        if self.source not in SOURCE_RANK:
            raise ValueError(f"unknown source {self.source!r}")

    def sort_key(self, seq: int = 0):
        # seq = arrival order WITHIN the same source: each source is one ordered stream (E1j: several REPLY_TOKENS
        # delivered in one frame were reordered when the payload broke the tie). Across sources the order is fixed
        # by priority and source rank, so arrival order between sources never matters.
        return (PRIORITY[self.kind], SOURCE_RANK[self.source], self.k, seq)


@dataclass
class Action:
    kind: str                    # OPEN_TURN | REQUEST_REPLY | SPEAK | HOLD | CANCEL | CALL_TOOL | RESUME_AFTER_TOOL |
    k: int                       # CLOSE_TURN | DROP_STALE | DEFER
    epoch: int
    turn: int
    reason: str
    data: dict = field(default_factory=dict)


@dataclass
class ControllerConfig:
    barge_in_frames: int = 3       # consecutive frames of user speech that make a barge-in (shorter = back-channel)
    barge_in_frames_audible: int = 0  # G4: if > 0, the run needed while OUR audio is audible (faster stop)
    min_blank: int = 2             # ASR blank frames before a Thinker TAKE_TURN may open the turn
    take_turn_window: int = 12     # a deferred TAKE_TURN is kept this many frames, then dropped
    debounce: int = 8              # no new turn this many frames after a cancel
    reply_timeout: int = 40        # PENDING without reply tokens: close the turn (3.2 s)
    tool_timeout: int = 100        # TOOL_WAIT without a result: resume with an error (8 s)
    eou_fallback: bool = True      # an ASR EOU opens the turn when the Thinker stays silent
    merge_frames: int = 12         # caller words arriving this soon after an open, before anything was spoken,
                                   # are merged into a NEW request that supersedes the pending one
    barge_guard: int = 0           # user speech counts for barge-in only from this many frames after the turn opened
                                   # (the ASR tail of the words just spoken is not a barge-in)
    audio_aware: bool = False      # G3: the turn closes at the END OF ITS AUDIO (AUDIO_ENDED), not at the end of the
                                   # text; user speech during the remaining audio is a barge-in; a take-turn waits
    audio_timeout: int = 600       # DRAINING without AUDIO_ENDED: close anyway (48 s)
    holding_phrase: bool = False   # E1l: speak a short per-language holding phrase once per turn when the text LLM
                                   # decides to call a tool (before the call is generated)
    barge_after_open: bool = False # E1k: only a speech run that STARTS after the turn opened can be a barge-in (a run
                                   # that began before the open is the tail of the caller's own turn)


class Controller:
    def __init__(self, config: Optional[ControllerConfig] = None, tracer=None):
        self.cfg = config or ControllerConfig()
        self.tr = tracer or NullTracer()
        self.state = St.LISTENING
        self.epoch = 0
        self.turn = 0
        self.request_id = 0
        self.inbox: list = []
        self._seq: dict = {}
        self.applied: set = set()
        self.actions: list = []
        self.calls: dict = {}            # call_id -> status
        self.speech_run = 0
        self.run_start_k = None
        self.last_speech_k = -10 ** 9
        self.heard_since_turn = False
        self.deferred: Optional[Intent] = None
        self.debounce_until = -1
        self.opened_k = None
        self.last_reply_k = None
        self.tool_since_k = None
        self.open_word = False
        self.spoken_in_turn = False
        self.drain_since_k = None
        self.agent_audible = False       # G4: set by the driver every frame (playback model)
        self.audio_end_k = None          # frame of the last agent audio end (G3)
        self.holding_done = False        # a holding phrase was already spoken in this turn (never repeated)
        self.held: list = []             # current-request reply intents that arrived during TOOL_WAIT (E1k)

    # ------------------------------------------------------------------ input
    def post(self, it: Intent) -> None:
        self.tr.emit("ctl.intent", it.k, kind=it.kind, source=it.source, epoch=it.epoch, request_id=it.request_id,
                     call_id=it.call_id, uid=it.uid)
        self._seq[it.source] = self._seq.get(it.source, 0) + 1
        self.inbox.append((it, self._seq[it.source]))

    def tick(self, k: int) -> list:
        """Apply every intent posted for frames <= k, in priority order; then the per-frame timers."""
        due = [i for i, _ in sorted([x for x in self.inbox if x[0].k <= k], key=lambda x: x[0].sort_key(x[1]))]
        self.inbox = [x for x in self.inbox if x[0].k > k]
        out_start = len(self.actions)
        speech_now = any(i.kind in ("USER_SPEECH_START", "USER_SPEECH") for i in due)
        for it in due:
            if it.uid is not None:
                if it.uid in self.applied:
                    continue
                self.applied.add(it.uid)
            getattr(self, "_on_" + it.kind.lower())(it, k)
        if not speech_now:
            self.speech_run = 0
        self._timers(k)
        return self.actions[out_start:]

    # ------------------------------------------------------------------ helpers
    def _act(self, kind: str, k: int, reason: str, **data) -> Action:
        a = Action(kind, k, self.epoch, self.turn, reason, data)
        self.actions.append(a)
        self.tr.emit("ctl.action", k, kind=kind, epoch=self.epoch, turn_id=self.turn, reason=reason,
                     **{x: v for x, v in data.items() if isinstance(v, (int, str, float, bool, type(None)))})
        return a

    def _stale(self, it: Intent, k: int, why: str) -> None:
        self._act("DROP_STALE", k, why, intent=it.kind, source=it.source, intent_epoch=it.epoch,
                  request_id=it.request_id, call_id=it.call_id)

    def _is_stale(self, it: Intent) -> Optional[str]:
        if it.epoch is not None and it.epoch != self.epoch:
            return "old_epoch"
        if it.request_id is not None and it.request_id != self.request_id:
            return "old_request"
        return None

    def _open(self, k: int, reason: str, user_text: str = "") -> None:
        self.turn += 1
        self.request_id += 1
        self.state = St.PENDING
        self.opened_k = k
        self.last_reply_k = None
        self.heard_since_turn = False
        self.spoken_in_turn = False
        self.deferred = None
        self.held = []
        self.holding_done = False
        self._act("OPEN_TURN", k, reason)
        self._act("REQUEST_REPLY", k, reason, request_id=self.request_id, user_text=user_text)

    def _cancel(self, k: int, reason: str) -> None:
        self._act("CANCEL", k, reason, flush=True, new_epoch=self.epoch + 1)
        self.epoch += 1
        self.request_id += 1               # any answer to the old request is now stale
        for cid, st in self.calls.items():
            if st == "pending":
                self.calls[cid] = "cancelled"
        self.state = St.LISTENING
        self.deferred = None
        self.held = []
        self.debounce_until = k + self.cfg.debounce

    def _close(self, k: int, reason: str) -> None:
        self._act("CLOSE_TURN", k, reason)
        self.state = St.LISTENING

    def _may_open(self, k: int) -> Optional[str]:
        if k < self.debounce_until:
            return "debounce_after_cancel"
        if k - self.last_speech_k < self.cfg.min_blank:
            return "asr_speech_recent"
        if self.open_word:
            return "asr_word_open"
        if self.turn > 0 and not self.heard_since_turn:
            return "no_new_caller_words"
        return None

    # ------------------------------------------------------------------ intents
    def _on_user_speech_start(self, it, k):
        self._on_user_speech(it, k)

    def _on_user_speech(self, it, k):
        self.speech_run += 1
        if self.speech_run == 1:
            self.run_start_k = it.k
        self.last_speech_k = it.k
        started_after = (not self.cfg.barge_after_open or self.opened_k is None or
                         (self.run_start_k is not None and self.run_start_k > self.opened_k))
        if self.state in (St.PENDING, St.SPEAKING, St.TOOL_WAIT, St.DRAINING) and started_after and \
                (self.opened_k is None or it.k - self.opened_k >= self.cfg.barge_guard):
            need = self.cfg.barge_in_frames_audible if (self.agent_audible and self.cfg.barge_in_frames_audible) \
                else self.cfg.barge_in_frames
            if self.speech_run >= need:
                self._cancel(k, "barge_in")
            elif self.state is St.SPEAKING:
                self._act("HOLD", k, "backchannel_candidate", run=self.speech_run)

    def _on_user_speech_end(self, it, k):
        pass

    def _on_words(self, it, k):
        self.heard_since_turn = True
        self.open_word = False
        if self.state is St.PENDING and not self.spoken_in_turn and self.opened_k is not None \
                and k - self.opened_k <= self.cfg.merge_frames:
            old = self.request_id
            self.request_id += 1
            self._act("REQUEST_REPLY", k, "merge_fragment", request_id=self.request_id, supersedes=old)

    def _on_word_open(self, it, k):
        self.open_word = True

    def _on_cancel(self, it, k):
        if self.state is not St.LISTENING:
            self._cancel(k, f"app:{it.payload or 'cancel'}")

    def _on_take_turn(self, it, k):
        if self.state is St.DRAINING:                    # our audio is still playing: retry when it has ended
            self.deferred = it
            self._act("DEFER", k, "agent_audio_playing", signal=it.payload or "bos")
            return
        if self.state is not St.LISTENING:
            return                                       # already in a turn: idempotent
        why = self._may_open(k)
        if why is None:
            self._open(k, f"thinker_{it.payload or 'bos'}")
        elif why in ("asr_speech_recent", "asr_word_open"):
            self.deferred = it                           # sources disagree: keep it, retry when the ASR is blank
            self._act("DEFER", k, why, signal=it.payload or "bos")
        else:
            self._act("HOLD", k, why, signal=it.payload or "bos")

    def _on_eou(self, it, k):
        if self.state is St.LISTENING and self.cfg.eou_fallback and self.heard_since_turn:
            why = self._may_open(k)
            if why is None:
                self._open(k, "asr_eou_fallback")

    def _on_yield(self, it, k):
        pass                                             # the Thinker's EOS is advisory; the controller owns EOS

    def _hold(self, it, k) -> bool:
        """A reply of the CURRENT request during TOOL_WAIT is not stale: the worker ran the tool itself and is already
        streaming the answer, while the TOOL_RESULT intent lands a frame later (E1k: with KV reuse the first word
        arrived in the same frame as the call and was dropped). Keep it and release it on resume, in arrival order."""
        if self.state is St.TOOL_WAIT and self._is_stale(it) is None:
            self.held.append(it)
            self._act("HOLD", k, "reply_during_tool_wait", intent=it.kind, request_id=it.request_id)
            return True
        return False

    def _release(self, k: int) -> None:
        held, self.held = self.held, []
        for it in held:
            (self._on_reply_tokens if it.kind == "REPLY_TOKENS" else self._on_reply_done)(it, k)

    def _on_reply_tokens(self, it, k):
        if self._hold(it, k):
            return
        why = self._is_stale(it)
        if why or self.state not in (St.PENDING, St.SPEAKING):
            return self._stale(it, k, why or f"state_{self.state.value}")
        self.state = St.SPEAKING
        self.spoken_in_turn = True
        self.last_reply_k = k
        self._act("SPEAK", k, "reply_tokens", ids=list(it.payload or []), request_id=it.request_id)

    def _on_reply_done(self, it, k):
        if self._hold(it, k):
            return
        why = self._is_stale(it)
        if why or self.state not in (St.PENDING, St.SPEAKING):
            return self._stale(it, k, why or f"state_{self.state.value}")
        if self.cfg.audio_aware and self.spoken_in_turn:
            self._act("END_TEXT", k, "reply_done")       # the text is final (EOS to the talker); the turn stays open
            self.state = St.DRAINING
            self.drain_since_k = k
            return
        self._close(k, "reply_done")

    def _on_tool_decided(self, it, k):
        """The text LLM decided to call a tool (before generating the call). With ``holding_phrase``: say a short
        holding phrase now, once per turn, only if nothing of the reply was spoken yet."""
        why = self._is_stale(it)
        if why or self.state not in (St.PENDING, St.TOOL_WAIT):
            return self._stale(it, k, why or f"state_{self.state.value}")
        if not self.cfg.holding_phrase or self.holding_done or self.spoken_in_turn:
            return
        self.holding_done = True
        self._act("HOLDING_PHRASE", k, "tool_decided", request_id=it.request_id)

    def _on_tool_call(self, it, k):
        why = self._is_stale(it)
        if why or self.state not in (St.PENDING, St.SPEAKING):
            return self._stale(it, k, why or f"state_{self.state.value}")
        if it.call_id in self.calls:
            return                                       # idempotent
        self.calls[it.call_id] = "pending"
        self.state = St.TOOL_WAIT
        self.tool_since_k = k
        self._act("CALL_TOOL", k, "nano_tool_call", call_id=it.call_id, name=(it.payload or {}).get("name"))

    def _on_tool_result(self, it, k):
        if self.calls.get(it.call_id) != "pending" or (it.epoch is not None and it.epoch != self.epoch):
            return self._stale(it, k, f"call_{self.calls.get(it.call_id, 'unknown')}")
        self.calls[it.call_id] = "done"
        if self.state is St.TOOL_WAIT and all(v != "pending" for v in self.calls.values()):
            self.state = St.PENDING
            self._act("RESUME_AFTER_TOOL", k, "tool_result", call_id=it.call_id, request_id=self.request_id)
            self._release(k)

    def _on_tool_timeout(self, it, k):
        if self.calls.get(it.call_id) == "pending":
            self.calls[it.call_id] = "timeout"
            self.state = St.PENDING
            self._act("RESUME_AFTER_TOOL", k, "tool_timeout", call_id=it.call_id, request_id=self.request_id)
            self._release(k)

    def _on_audio_started(self, it, k):
        if it.epoch is not None and it.epoch != self.epoch:
            self._stale(it, k, "old_epoch_audio")

    def _on_audio_ended(self, it, k):
        if it.epoch is not None and it.epoch != self.epoch:
            return self._stale(it, k, "old_epoch_audio")
        self.audio_end_k = k
        if self.state is St.DRAINING:
            self._close(k, "audio_end")
            if self.deferred is not None:                # a take-turn that waited for our audio: its window starts now
                import dataclasses
                self.deferred = dataclasses.replace(self.deferred, k=k)

    def _on_underrun(self, it, k):
        pass

    # ------------------------------------------------------------------ timers
    def _timers(self, k: int) -> None:
        if self.state is St.DRAINING and self.drain_since_k is not None and \
                k - self.drain_since_k > self.cfg.audio_timeout:
            self._close(k, "audio_timeout")
        if self.deferred is not None and self.state is St.LISTENING:
            if k - self.deferred.k > self.cfg.take_turn_window:
                self._act("HOLD", k, "deferred_take_turn_expired")
                self.deferred = None
            elif self._may_open(k) is None:
                it, self.deferred = self.deferred, None
                self._open(k, f"thinker_{it.payload or 'bos'}_deferred")
        if self.state is St.PENDING and self.last_reply_k is None and self.opened_k is not None \
                and k - self.opened_k > self.cfg.reply_timeout:
            self._close(k, "reply_timeout")
        if self.state is St.TOOL_WAIT and self.tool_since_k is not None and k - self.tool_since_k > self.cfg.tool_timeout:
            for cid, st in self.calls.items():
                if st == "pending":
                    self.calls[cid] = "timeout"
            self.state = St.PENDING
            self.opened_k = k
            self._act("RESUME_AFTER_TOOL", k, "tool_timeout", request_id=self.request_id)
            self._release(k)
