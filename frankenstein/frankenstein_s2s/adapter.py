"""Talker adapter: the VoiceChat Thinker's duplex text stream -> commands for a streaming TTS talker.

Input, per 80 ms Thinker step: the text-channel id (BOS / text / PAD / EOS), the function-channel id (SOTC / EOTC /
EOTR), and external user-speech evidence; plus out-of-band tool events and talker completion notices. Output:
``StartSegment`` / ``AppendText`` / ``FinalizeSegment`` / ``Cancel`` commands sent to every sink (talker driver,
playback gate, tracer), and a ``log`` of every non-trivial decision.

Rules (DESIGN.md "Adapter state machine" has the full table):

* VoiceChat writes a reply as a BURST (one token per frame), then a PAD tail sized for the old talker, then EOS.
  Text moves to the talker as complete words; the turn's text is declared final ``commit_pads`` PAD frames after
  the burst ("burst commit") instead of waiting for EOS, which can be seconds later.
* PAD is never forwarded as ``tts_pad``: a text underrun is a talker BLOCK (driver side).
* EOS after the burst commit = natural end: the talker drains. EOS before it is ambiguous (VoiceChat uses one EOS for
  a natural end and for an interruption): with user speech in the last ``eos_speech_window`` frames it is an
  interruption (cancel), otherwise a natural end (finalize what was said).
* Barge-in (``barge_in="immediate"``): ``barge_in_frames`` consecutive frames of user speech while a reply is open
  or audible -> cancel. With ``"thinker"`` the adapter waits for the Thinker's own EOS.
* Cancel bumps the epoch; every later command carries the new epoch; the PlaybackGate drops older audio.
* SOTC on the function channel: the text said so far is finalized (spoken preamble), the turn closes, the adapter
  waits for the tool; the answer arrives as a new BOS turn (or, defensively, as text without BOS).
* Text after the burst commit (multi-burst turn) opens a NEW talker segment: ``tts_eos`` was already fed.
* Long replies are cut into segments at a sentence end once ``segment_max_chars`` is reached (each segment is a new
  talker prefill; Sophon segments long utterances the same way).

No torch, no model: tokenizers are injected as callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

from . import tokens as T
from .events import (AppendText, Cancel, FinalizeSegment, Frame, LogEntry, StartSegment, TalkerDone, ToolEvent)
from .textunits import SENTENCE_END, Retokenizer, WordCommitter
from .tracing import NullTracer


class State(Enum):
    IDLE = "idle"            # no reply open, no audio pending
    ARMED = "armed"          # BOS seen, no complete word yet (no talker request)
    SPEAKING = "speaking"    # talker request(s) open, text still streaming
    FINAL = "final"          # turn text final (tts_eos fed), waiting for the Thinker's EOS
    DRAINING = "draining"    # turn closed, talker still producing / audio still pending
    TOOL_WAIT = "tool_wait"  # a tool call is in flight; no reply text expected


@dataclass
class AdapterConfig:
    talker_lang: str = "english"     # Qwen3-TTS codec language key (lang.SessionLanguage.talker_lang)
    voice: str = "default"
    commit_pads: int = 2
    barge_in: str = "immediate"      # "immediate" | "thinker"
    barge_in_frames: int = 3         # 240 ms of continuous user speech
    eos_speech_window: int = 5       # frames: speech this recent makes an early EOS an interruption
    segment_max_chars: int = 400     # cut long replies at the next sentence end after this many characters

    def __post_init__(self):
        if self.barge_in not in ("immediate", "thinker"):
            raise ValueError("barge_in must be 'immediate' or 'thinker'")
        if self.commit_pads < 1 or self.barge_in_frames < 1:
            raise ValueError("commit_pads and barge_in_frames must be >= 1")


@dataclass
class _Turn:
    id: int
    bos_k: int
    words: WordCommitter = field(default_factory=WordCommitter)
    ids: list = field(default_factory=list)
    raw_text: str = ""               # text-mode (cascade) source
    segment: int = -1
    seg_open: bool = False
    seg_chars: int = 0
    retok: Optional[Retokenizer] = None
    pads: int = 0
    final: bool = False
    final_k: Optional[int] = None
    has_text: bool = False


class TalkerAdapter:
    def __init__(self, sinks: Sequence, decode: Optional[Callable[[list], str]] = None,
                 encode: Optional[Callable[[str], list]] = None, config: Optional[AdapterConfig] = None,
                 tracer=None):
        self.sinks = list(sinks)
        self.tracer = tracer or NullTracer()
        self.decode = decode
        self.encode = encode
        self.cfg = config or AdapterConfig()
        self.epoch = 0
        self.turn: Optional[_Turn] = None
        self.turns_started = 0
        self.busy: set = set()           # (turn, segment) of the current epoch not yet reported done
        self.tool_wait = False
        self.last_close: Optional[str] = None
        self.speech_run = 0
        self.last_speech_k: Optional[int] = None
        self._barged = False
        self.log: list = []
        self.commands: list = []

    # ------------------------------------------------------------------------------------------------ state
    @property
    def state(self) -> State:
        if self.turn is None:
            if self.tool_wait:
                return State.TOOL_WAIT
            return State.DRAINING if self.busy else State.IDLE
        if self.turn.final:
            return State.FINAL
        return State.SPEAKING if self.turn.segment >= 0 else State.ARMED

    # ------------------------------------------------------------------------------------------------ inputs
    def feed(self, f: Frame) -> None:
        """One Thinker step (VoiceChat duplex source)."""
        self._user_speech(f.k, f.user_speech)
        if f.func_id is not None and f.func_id != T.PAD_ID:
            self._function(f.k, f.func_id)
        kind = T.kind_of(f.text_id)
        if kind == "bos":
            self._bos(f.k)
        elif kind == "eos":
            self._eos(f.k)
        elif kind == "pad":
            self._pad(f.k)
        elif kind == "tok":
            self._tok(f.k, f.text_id)
        elif kind == "missing":
            self._log(f.k, "missing_text_id", "treated as no text (neither text nor PAD)")
        else:  # sotc/eotc/eotr/special on the TEXT channel: never forwarded
            self._log(f.k, f"{kind}_on_text_channel", "dropped", id=f.text_id)

    def tool(self, ev: ToolEvent) -> None:
        if ev.kind == "pending":
            if not self.tool_wait:
                self._enter_tool_wait(ev.k, f"tool.pending {ev.call_id}")
        elif ev.kind == "result":
            self._log(ev.k, "tool_result", "wait for the Thinker to resume (EOTR / BOS)", call_id=ev.call_id)
        else:  # resumed | timeout | cancelled
            if self.tool_wait:
                self.tool_wait = False
                self._log(ev.k, f"tool_{ev.kind}", "leave TOOL_WAIT", call_id=ev.call_id)

    def talker_done(self, d: TalkerDone) -> None:
        if d.epoch == self.epoch:
            self.busy.discard((d.turn, d.segment))
        self.tracer.emit("adapter.talker_done", None, turn_id=d.turn, segment=d.segment, epoch=d.epoch,
                         stale=d.epoch != self.epoch)

    # cascade source (fallback architecture): plain text deltas from a text LLM, same rules downstream
    def begin_turn(self, k: int) -> None:
        self._bos(k)

    def add_text(self, k: int, delta: str) -> None:
        if self.turn is None:
            self._log(k, "text_outside_turn", "implicit turn")
            self._bos(k)
        t = self.turn
        if t.final:
            self._multi_burst(k)
        t.raw_text += delta
        self._text_update(k, t.raw_text)

    def end_turn(self, k: int, interrupted: bool = False) -> None:
        if interrupted:
            if self.turn is not None or self.busy:
                self._cancel(k, "end_turn_interrupted")
            return
        if self.turn is not None and self.turn.has_text and not self.turn.final:
            self._finalize_text(k, "end_turn")
        self._close_turn("end_turn", k)

    def inject_utterance(self, k: int, text: str, reason: str = "app") -> bool:
        """Speak an application-provided utterance (e.g. an on-hold acknowledgement) as its own short turn.
        Only when no reply is open and nothing is audible; returns False (and logs) otherwise."""
        if self.turn is not None or self.busy:
            self._log(k, "inject_refused", "a reply is open or audible", reason=reason)
            return False
        tool_wait = self.tool_wait
        self._bos(k)
        self.tool_wait = tool_wait
        self.turn.raw_text = text
        self._text_update(k, text)
        self._finalize_text(k, f"inject:{reason}")
        self._close_turn(f"inject:{reason}", k)
        return True

    # ------------------------------------------------------------------------------------------------ text channel
    def _bos(self, k: int) -> None:
        t = self.turn
        if t is not None:
            if t.final or not t.has_text:
                self._log(k, "bos_inside_open_turn", "previous turn closed without EOS (its audio drains)",
                          turn=t.id)
                self._close_turn("bos", k)
            else:
                self._log(k, "bos_inside_open_turn", "previous turn cancelled (text was not final)", turn=t.id)
                self._cancel(k, "bos_restart")
        if self.tool_wait:
            self.tool_wait = False
            self._log(k, "bos_after_tool", "leave TOOL_WAIT: the answer starts")
        self.turns_started += 1
        self.turn = _Turn(id=self.turns_started, bos_k=k)
        self.tracer.emit("turn.started", k, turn_id=self.turn.id, who="agent", epoch=self.epoch)
        self.tracer.emit("adapter.turn_open", k, turn_id=self.turn.id, epoch=self.epoch)

    def _tok(self, k: int, tid: int) -> None:
        if self.turn is None:
            if self.tool_wait or self.last_close == "tool_call":
                self._log(k, "text_without_bos", "implicit turn after a tool call")
                self._bos(k)
            else:
                self._log(k, "tok_outside_turn", "dropped", id=tid)
                return
        t = self.turn
        if t.final:
            self._multi_burst(k)
        t.ids.append(tid)
        if self.decode is None:
            raise RuntimeError("feed() with token ids needs a decode callable")
        self._text_update(k, self.decode(t.ids))

    def _multi_burst(self, k: int) -> None:
        t = self.turn
        self._log(k, "text_after_commit", "new talker segment (tts_eos already fed)", turn=t.id)
        t.final = False
        t.final_k = None
        t.pads = 0

    def _text_update(self, k: int, text: str) -> None:
        t = self.turn
        t.pads = 0
        t.has_text = t.has_text or bool(text.strip())
        before = t.words.rewrites
        chunk = t.words.update(text)
        if t.words.rewrites != before:
            self._log(k, "detokenizer_rewrote_committed_text", "kept the committed prefix")
        if chunk:
            self._emit_text(k, chunk)

    def _pad(self, k: int) -> None:
        t = self.turn
        if t is None or not t.has_text or t.final:
            return
        t.pads += 1
        if t.pads >= self.cfg.commit_pads:
            self._finalize_text(k, "burst_commit")
            self._log(k, f"{self.cfg.commit_pads}_pads_after_text", "burst commit: tts_eos", turn=t.id)

    def _eos(self, k: int) -> None:
        t = self.turn
        if t is None:
            self._log(k, "eos_outside_turn", "dropped")
            return
        if not t.has_text:
            self._log(k, "empty_turn", "closed, nothing spoken", turn=t.id)
        elif t.final:
            self._log(k, "eos_after_commit", "drain (let the talker finish)", turn=t.id,
                      pad_tail_frames=k - t.final_k)
        else:
            recent = self.last_speech_k is not None and k - self.last_speech_k <= self.cfg.eos_speech_window
            if recent:
                self._log(k, "eos_before_commit", "AMBIGUOUS -> interruption (user speech is recent): cancel",
                          turn=t.id)
                self._cancel(k, "eos_with_user_speech")
                return
            self._log(k, "eos_before_commit", "AMBIGUOUS -> natural end (no recent user speech): finalize",
                      turn=t.id)
            self._finalize_text(k, "eos")
        self._close_turn("eos", k)

    # ------------------------------------------------------------------------------------------------ function channel
    def _function(self, k: int, fid: int) -> None:
        if fid == T.SOTC_ID:
            self._enter_tool_wait(k, "sotc")
        elif fid == T.EOTC_ID:
            self._log(k, "eotc", "call text complete; tool runs")
        elif fid == T.EOTR_ID:
            if self.tool_wait:
                self.tool_wait = False
                self._log(k, "eotr", "leave TOOL_WAIT: the Thinker consumed the result")

    def _enter_tool_wait(self, k: int, why: str) -> None:
        t = self.turn
        if t is not None:
            if t.has_text and not t.final:
                self._finalize_text(k, "tool_call")
            self._log(k, "tool_call_start", "preamble finalized, turn closed", turn=t.id, why=why)
            self._close_turn("tool_call", k)
        else:
            self._log(k, "tool_call_start", "no open turn", why=why)
            self.last_close = "tool_call"
        self.tool_wait = True

    # ------------------------------------------------------------------------------------------------ user side
    def _user_speech(self, k: int, speaking: bool) -> None:
        if not speaking:
            self.speech_run = 0
            self._barged = False
            return
        self.speech_run += 1
        self.last_speech_k = k
        if (self.cfg.barge_in == "immediate" and not self._barged and self.speech_run >= self.cfg.barge_in_frames
                and (self.turn is not None or self.busy)):
            self._barged = True
            self._log(k, "barge_in", "cancel reply and flush audio", run_frames=self.speech_run)
            self._cancel(k, "barge_in")

    # ------------------------------------------------------------------------------------------------ actions
    def _emit_text(self, k: int, chunk: str) -> None:
        t = self.turn
        if not t.seg_open:
            chunk = chunk.lstrip()
            if not chunk:
                return
            t.segment += 1
            t.seg_open = True
            t.seg_chars = 0
            t.retok = Retokenizer(self.encode)
            ids = t.retok.add(chunk)
            self._send(StartSegment(k, self.epoch, t.id, t.segment, self.cfg.talker_lang, self.cfg.voice, chunk, ids))
            self.busy.add((t.id, t.segment))
        else:
            before = t.retok.prefix_changes
            ids = t.retok.add(chunk)
            if t.retok.prefix_changes != before:
                self._log(k, "retokenization_changed_prefix", "kept old prefix, appended the chunk's own ids")
            self._send(AppendText(k, self.epoch, t.id, t.segment, chunk, ids))
        t.seg_chars += len(chunk)
        if t.seg_chars >= self.cfg.segment_max_chars and chunk.rstrip()[-1:] in SENTENCE_END:
            self._send(FinalizeSegment(k, self.epoch, t.id, t.segment, "segment_length"))
            t.seg_open = False
            self._log(k, "segment_cut", "long reply: next text opens a new segment", turn=t.id, chars=t.seg_chars)

    def _finalize_text(self, k: int, reason: str) -> None:
        t = self.turn
        rest = t.words.flush()
        if rest.strip():
            self._emit_text(k, rest)
        if t.seg_open:
            self._send(FinalizeSegment(k, self.epoch, t.id, t.segment, reason))
            t.seg_open = False
        t.final = True
        t.final_k = k

    def _close_turn(self, why: str, k: Optional[int] = None) -> None:
        if self.turn is not None:
            self.tracer.emit("adapter.turn_close", k, turn_id=self.turn.id, epoch=self.epoch, why=why)
        self.turn = None
        self.last_close = why

    def _cancel(self, k: int, reason: str) -> None:
        turn = self.turn.id if self.turn is not None else None
        self._send(Cancel(k, self.epoch, self.epoch + 1, reason))
        self.epoch += 1
        self.tracer.emit("adapter.epoch", k, epoch=self.epoch, reason=reason)
        self.busy.clear()
        if self.turn is not None:
            self.tracer.emit("adapter.turn_close", k, turn_id=turn, epoch=self.epoch - 1, why=f"cancel:{reason}")
        self.turn = None
        self.last_close = f"cancel:{reason}"

    def _send(self, cmd) -> None:
        self._trace_cmd(cmd)
        self.commands.append(cmd)
        for s in self.sinks:
            s.submit(cmd)

    def _log(self, k: int, what: str, action: str, **detail) -> None:
        self.log.append(LogEntry(k, what, action, dict(detail)))
        ev = _LOG_EVENT.get(what) or ("adapter.burst_commit" if what.endswith("_pads_after_text") else
                                      "adapter.tool_wait" if what.startswith(("tool", "eot")) else "adapter.drop")
        turn = detail.pop("turn", None)
        if turn is None and self.turn is not None:
            turn = self.turn.id
        self.tracer.emit(ev, k, turn_id=turn, epoch=self.epoch, what=what, action=action, **detail)

    def _trace_cmd(self, cmd) -> None:
        tr = self.tracer
        if isinstance(cmd, StartSegment):
            tr.emit("adapter.segment_start", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch,
                    lang=cmd.lang, voice=cmd.voice)
            tr.emit("adapter.commit_word", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch,
                    text=cmd.text, qwen_ids=cmd.qwen_ids, first=True)
        elif isinstance(cmd, AppendText):
            tr.emit("adapter.commit_word", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch,
                    text=cmd.text, qwen_ids=cmd.qwen_ids)
        elif isinstance(cmd, FinalizeSegment):
            tr.emit("adapter.finalize", cmd.k, turn_id=cmd.turn, segment=cmd.segment, epoch=cmd.epoch,
                    reason=cmd.reason)
        elif isinstance(cmd, Cancel):
            tr.emit("adapter.cancel", cmd.k, epoch=cmd.epoch, new_epoch=cmd.new_epoch, reason=cmd.reason)


_LOG_EVENT = {
    "bos_inside_open_turn": "adapter.turn_restart",
    "text_without_bos": "adapter.turn_implicit",
    "text_outside_turn": "adapter.turn_implicit",
    "text_after_commit": "adapter.segment_restart",
    "segment_cut": "adapter.segment_restart",
    "eos_after_commit": "adapter.eos_drain",
    "eos_before_commit": "adapter.eos_ambiguous",
    "empty_turn": "adapter.drop",
    "barge_in": "adapter.barge_in",
    "bos_after_tool": "adapter.tool_wait",
    "inject_refused": "adapter.inject",
    "detokenizer_rewrote_committed_text": "adapter.retokenize",
    "retokenization_changed_prefix": "adapter.retokenize",
}
