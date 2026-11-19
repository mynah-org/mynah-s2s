"""Controller v2: the single writer of turn state, driven by the ears' events (CONTROL.md ideas of v1).

Sources post INTENTS stamped with the 80 ms step ``k`` (ears events carry the frame they refer to; the others the
step at which they were delivered) and, when they answer an earlier action, the ``epoch`` / ``request_id``. Once per
step ``tick(k)`` sorts the due intents by (priority, source rank, k, arrival within a source) and applies them, so the
same set of intents always gives the same actions. Priority: barge-in > cancel > tool > text > done > endpoint > rest.
Stale answers (old epoch / old request) are dropped and traced; ``uid`` makes an intent idempotent.

States:   LISTENING  no agent turn (ears in "listen" mode)
          THINKING   turn opened, reply requested, nothing queued for playback yet
          SPEAKING   reply / holding audio queued or playing (until the turn's audio has played out)
The ears are in "speaking" mode in every state but LISTENING (Tyger: from the endpoint until played out), so the
yield head can detect a user who was not finished (an early endpoint) as well as an interruption.

Intents                          ->  actions
  ENDPOINT (ears, listen)            OPEN_TURN + REQUEST_REPLY(user_text)        (empty transcript: HOLD)
  BARGE (ears, speaking)             CANCEL(flush, new epoch) + EARS_RESET; history keeps what was heard; a barge
                                     before anything was heard merges the cancelled user text into the next turn
  USER_CONTINUES (asr: words after the endpoint frame, soon after the open, before any reply audio)
                                     CANCEL as an unheard barge: the endpoint was early (a pause inside the
                                     turn); the user text carries into the next endpoint
  BACKCHANNEL (ears, listen)         BACKCHANNEL (short audio, not in the 2nd ear, not in history)
  LLM_DECIDED (tool step chose a call)  HOLDING_PHRASE once per turn if nothing was spoken yet
  LLM_WAIT (a tool is still running)  keeps the reply timer alive; HOLDING_PHRASE(reminder) when the agent has
                                     been silent ``reminder_silence`` steps, at most ``max_reminders`` per turn,
                                     ``reminder_gap`` steps apart (tools run asynchronously, the caller can barge in)
  LLM_TOOL                           TOOL_RECORD (history: the call and its result, kept even if later barged)
  LLM_TEXT                           SPEAK(text) (-> chunker -> normaliser -> talker)
  LLM_DONE                           END_TEXT (chunker flush); the turn closes when its audio has played out
  AUDIO_STARTED                      state SPEAKING
  PLAYED_OUT                         CLOSE_TURN + EARS_RESET (only after LLM_DONE)
  CANCEL (app)                       CANCEL
Timers: reply_timeout (THINKING without text or tool), audio_timeout (text done, audio never played out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class St(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


PRIORITY = {"BARGE": 0, "CANCEL": 1, "USER_CONTINUES": 1, "LLM_DECIDED": 2, "LLM_WAIT": 2, "LLM_TOOL": 2, "LLM_TEXT": 3, "LLM_DONE": 4,
            "AUDIO_STARTED": 5, "PLAYED_OUT": 6, "ENDPOINT": 7, "BACKCHANNEL": 8}
SOURCE_RANK = {"app": 0, "ears": 1, "llm": 2, "talker": 3, "playout": 4}


@dataclass(frozen=True)
class Intent:
    kind: str
    source: str
    k: int
    payload: Any = None
    epoch: Optional[int] = None
    request_id: Optional[int] = None
    uid: Optional[str] = None

    def __post_init__(self):
        if self.kind not in PRIORITY:
            raise ValueError(f"unknown intent {self.kind!r}")
        if self.source not in SOURCE_RANK:
            raise ValueError(f"unknown source {self.source!r}")


@dataclass
class Action:
    kind: str
    k: int
    epoch: int
    turn: int
    reason: str
    data: dict = field(default_factory=dict)


@dataclass
class ControllerConfig:
    holding_phrase: bool = True
    backchannel: bool = True
    reply_timeout: int = 150           # steps (12 s) THINKING without any text or tool event
    audio_timeout: int = 750           # steps (60 s) after LLM_DONE without PLAYED_OUT
    min_user_chars: int = 1            # an endpoint with a shorter transcript does not open a turn
    merge_unheard: bool = True         # a barge before anything was heard: the user's text carries into the next turn
    continue_window: int = 25          # steps (2 s) after an open in which new user words mean "not finished"
    max_reminders: int = 2             # on-hold reminders per turn while a tool runs
    reminder_silence: int = 12         # steps (~1 s) of agent silence before a reminder
    reminder_gap: int = 50             # steps (4 s) between two reminders


class Controller:
    def __init__(self, config: Optional[ControllerConfig] = None, tracer=None):
        self.cfg = config or ControllerConfig()
        self.tr = tracer
        self.state = St.LISTENING
        self.epoch = 0
        self.turn = 0
        self.request_id = 0
        self.inbox: list = []
        self._seq: dict = {}
        self.applied: set = set()
        self.actions: list = []
        self.opened_k = None
        self.text_done = False
        self.spoken = False               # reply text handed to the talker in this turn
        self.holding_done = False
        self.reminders = 0
        self.last_reminder_k = None
        self.last_activity_k = None
        self.done_k = None
        self.carry = ""                   # user text of a turn cancelled before anything was heard
        self.user_text = ""               # the current turn's user text (as sent to the LLM)

    # ------------------------------------------------------------------ io
    def _emit(self, ev, k, **f):
        if self.tr is not None:
            self.tr.emit(ev, k, **f)

    @property
    def ears_mode(self) -> str:
        return "listen" if self.state is St.LISTENING else "speaking"

    def post(self, it: Intent) -> None:
        self._emit("ctl.intent", it.k, kind=it.kind, source=it.source, epoch=it.epoch, request_id=it.request_id,
                   uid=it.uid)
        self._seq[it.source] = self._seq.get(it.source, 0) + 1
        self.inbox.append((it, self._seq[it.source]))

    def tick(self, k: int) -> list:
        due = [x for x in self.inbox if x[0].k <= k]
        self.inbox = [x for x in self.inbox if x[0].k > k]
        due.sort(key=lambda x: (PRIORITY[x[0].kind], SOURCE_RANK[x[0].source], x[0].k, x[1]))
        n0 = len(self.actions)
        for it, _ in due:
            if it.uid is not None:
                if it.uid in self.applied:
                    continue
                self.applied.add(it.uid)
            getattr(self, "_on_" + it.kind.lower())(it, k)
        self._timers(k)
        return self.actions[n0:]

    def _act(self, kind, k, reason, **data) -> Action:
        a = Action(kind, k, self.epoch, self.turn, reason, data)
        self.actions.append(a)
        self._emit("ctl.action", k, kind=kind, epoch=self.epoch, turn_id=self.turn, reason=reason,
                   **{x: v for x, v in data.items() if isinstance(v, (int, str, float, bool))})
        return a

    def _stale(self, it, k, why):
        self._act("DROP_STALE", k, why, intent=it.kind, source=it.source, intent_epoch=it.epoch,
                  request_id=it.request_id)

    def _is_stale(self, it) -> Optional[str]:
        if it.epoch is not None and it.epoch != self.epoch:
            return "old_epoch"
        if it.request_id is not None and it.request_id != self.request_id:
            return "old_request"
        return None

    # ------------------------------------------------------------------ transitions
    def _open(self, k, user_text, reason):
        self.turn += 1
        self.request_id += 1
        self.state = St.THINKING
        self.opened_k = self.last_activity_k = k
        self.text_done = self.spoken = self.holding_done = False
        self.reminders, self.last_reminder_k = 0, None
        self.done_k = None
        self.user_text = user_text
        self._act("OPEN_TURN", k, reason, user_text=user_text)
        self._act("REQUEST_REPLY", k, reason, request_id=self.request_id, user_text=user_text)

    def _cancel(self, k, reason, **data):
        heard_any = self.state is St.SPEAKING
        self._act("CANCEL", k, reason, flush=True, new_epoch=self.epoch + 1, cancelled_request=self.request_id,
                  heard_possible=heard_any, **data)
        self.epoch += 1
        self.request_id += 1
        self.state = St.LISTENING
        self._act("EARS_RESET", k, reason)

    # ------------------------------------------------------------------ intents
    def _on_endpoint(self, it, k):
        if self.state is not St.LISTENING:
            return self._act("HOLD", k, f"endpoint_in_{self.state.value}")
        text = (it.payload or {}).get("text", "").strip()
        if self.carry:
            text = (self.carry + " " + text).strip()
        if len(text) < self.cfg.min_user_chars:
            return self._act("HOLD", k, "empty_transcript")
        merged = bool(self.carry)
        self.carry = ""
        self._open(k, text, "endpoint_merged" if merged else "endpoint")

    def _on_barge(self, it, k):
        if self.state is St.LISTENING:
            return self._act("HOLD", k, "barge_while_listening")
        p = it.payload or {}
        unheard = self.state is St.THINKING and not self.spoken
        if unheard and self.cfg.merge_unheard:
            self.carry = self.user_text
        self._cancel(k, "barge_in", onset_s=p.get("onset_s", 0.0), barge_k=it.k, unheard=unheard,
                     drop_user_turn=unheard and self.cfg.merge_unheard)

    def _on_user_continues(self, it, k):
        p = it.payload or {}
        if self.state is St.LISTENING or self.opened_k is None or k - self.opened_k > self.cfg.continue_window \
                or p.get("reply_heard"):
            return self._act("HOLD", k, "user_words_ignored")
        if self.cfg.merge_unheard:
            self.carry = self.user_text
        self._cancel(k, "user_continues", unheard=True, drop_user_turn=self.cfg.merge_unheard,
                     barge_k=it.k, onset_s=0.0)

    def _on_cancel(self, it, k):
        if self.state is not St.LISTENING:
            self._cancel(k, f"app:{it.payload or 'cancel'}")

    def _on_backchannel(self, it, k):
        if self.state is St.LISTENING and self.cfg.backchannel:
            self._act("BACKCHANNEL", k, "ears_backchannel")

    def _on_llm_decided(self, it, k):
        why = self._is_stale(it)
        if why or self.state is St.LISTENING:
            return self._stale(it, k, why or "state_listening")
        self.last_activity_k = k
        if self.cfg.holding_phrase and not self.holding_done and not self.spoken:
            self.holding_done = True
            self._act("HOLDING_PHRASE", k, "tool_decided", request_id=it.request_id)

    def _on_llm_wait(self, it, k):
        why = self._is_stale(it)
        if why or self.state is St.LISTENING:
            return self._stale(it, k, why or "state_listening")
        self.last_activity_k = k
        p = it.payload or {}
        silent = p.get("silent_steps", self.cfg.reminder_silence)      # unknown silence: assume silent
        if self.cfg.holding_phrase and self.reminders < self.cfg.max_reminders and \
                silent >= self.cfg.reminder_silence and \
                (self.last_reminder_k is None or k - self.last_reminder_k >= self.cfg.reminder_gap):
            self.reminders += 1
            self.last_reminder_k = k
            self._act("HOLDING_PHRASE", k, "tool_wait", request_id=it.request_id, reminder=self.reminders,
                      waited_s=(it.payload or {}).get("s"))

    def _on_llm_tool(self, it, k):
        why = self._is_stale(it)
        if why:
            # the tool DID run (the worker executes it before we see it): its record stays in the history
            self._act("TOOL_RECORD", k, "tool_after_cancel", stale=True, **_tool_fields(it.payload))
            return
        self.last_activity_k = k
        self._act("TOOL_RECORD", k, "tool_result", stale=False, **_tool_fields(it.payload))

    def _on_llm_text(self, it, k):
        why = self._is_stale(it)
        if why or self.state is St.LISTENING:
            return self._stale(it, k, why or "state_listening")
        self.last_activity_k = k
        self.spoken = True
        self._act("SPEAK", k, "reply_text", text=(it.payload or {}).get("text", ""), request_id=it.request_id)

    def _on_llm_done(self, it, k):
        why = self._is_stale(it)
        if why or self.state is St.LISTENING:
            return self._stale(it, k, why or "state_listening")
        self.text_done = True
        self.done_k = k
        self._act("END_TEXT", k, "llm_done", request_id=it.request_id, text=(it.payload or {}).get("text", ""),
                  spoken=self.spoken)

    def _on_audio_started(self, it, k):
        if self._is_stale(it):
            return self._stale(it, k, "old_epoch_audio")
        if self.state is St.THINKING:
            self.state = St.SPEAKING

    def _on_played_out(self, it, k):
        if self._is_stale(it):
            return self._stale(it, k, "old_epoch_audio")
        if self.state is not St.LISTENING and self.text_done:
            self.state = St.LISTENING
            self._act("CLOSE_TURN", k, "played_out")
            self._act("EARS_RESET", k, "turn_end")

    # ------------------------------------------------------------------ timers
    def _timers(self, k):
        if self.state is St.THINKING and not self.text_done and self.last_activity_k is not None and \
                k - self.last_activity_k > self.cfg.reply_timeout:
            self._cancel(k, "reply_timeout")
        if self.state is not St.LISTENING and self.text_done and self.done_k is not None and \
                k - self.done_k > self.cfg.audio_timeout:
            self.state = St.LISTENING
            self._act("CLOSE_TURN", k, "audio_timeout")
            self._act("EARS_RESET", k, "turn_end")


def _tool_fields(p) -> dict:
    p = p or {}
    return {"name": p.get("name"), "call_id": p.get("call_id")}
