"""Frankenstein v2 session pipeline (model-free orchestration; every model sits behind a plugin).

One ``Pipeline`` per session, advanced by ``step(user_pcm16k)`` once per 80 ms input step (the session clock):

  mic 16 kHz ──┐                                   ┌── 2nd ear (agent audio as released, delayed, 16 kHz)
               ▼                                   ▼
          EarsLink: 320 ms hops ──► ears process: streaming Nemotron ASR (transcript pieces) + GRU on layer 15
               │                                        (window regime) + rules -> ENDPOINT / BARGE / BACKCHANNEL
               ▼
          Controller (single writer; intents -> actions; epochs; ears mode = listen | speaking)
               │ REQUEST_REPLY(history)            ▲ LLM text / decided / tool / done
               ▼                                   │
          LLM plugin (Nano + tools + spoken-numbers -> digits)
               │ reply text
               ▼
          Chunker (per talker policy) -> normaliser -> Talker plugin (Pocket / Flash) -> Playout (real-time
          release; flush on cancel; played cursor -> history after barge + 2nd ear) -> transport

Everything here is per session: ears state and language, controller, conversation / history, LLM context and tool
ledger (inside the LLM plugin), chunker + normaliser locale, talker language / voice, playout, epochs. The session
language is set once at open and drives ASR prompt, LLM policy, normaliser, spoken-number parsing, talker language,
holding phrases and backchannels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio.playout import Playout, SecondEar
from .control.controller import Controller, ControllerConfig, Intent, St
from .control.history import Conversation
from .ears.client import UserText
from .text.chunker import Chunker, speech_text
from .tts.base import parse_sid, seg_id

# Fixed phrases per language live in langs.py (built-in table + language.json of a package folder); these dicts are
# the built-in values, kept for callers and tests that read them.
from . import langs as _langs  # noqa: E402

HOLDING = {c: x.phrases["holding"] for c, x in _langs.BUILTIN.items()}
HOLD_REMINDER = {c: x.phrases["reminder"] for c, x in _langs.BUILTIN.items()}
BACKCHANNEL = {c: x.phrases["backchannel"] for c, x in _langs.BUILTIN.items()}
STEP = 1280


def _tag(p: str) -> bool:
    s = p.strip().replace("\u2581", "")
    return s.startswith("<") and s.endswith(">")


@dataclass
class PipelineConfig:
    lang: str = "it"
    chunking: str | None = None            # None = the talker's policy
    client_delay_steps: int = 1            # transport + client playback buffer, in 80 ms steps
    overlap_grace_frames: int = 6          # speech this close to a turn's end is kept for the next user turn
    trace_frames: bool = True
    barge_assist: bool = True              # echo-aware energy onset + ASR word confirmation (control/barge.py)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass
class TurnOut:
    epoch: int
    turn: int
    chunker: Chunker
    n_seg: int = 0
    text: str = ""
    segs: list = field(default_factory=list)
    started: bool = False
    played_out_posted: bool = False


class Pipeline:
    def __init__(self, sid: str, ears, llm, talker, cfg: PipelineConfig, tracer):
        self.sid, self.ears, self.llm, self.talker, self.cfg, self.tr = sid, ears, llm, talker, cfg, tracer
        self.lang = cfg.lang
        self.ctl = Controller(cfg.controller, tracer=tracer)
        self.conv = Conversation()
        self.user = UserText()
        self.playout = Playout(client_delay_steps=cfg.client_delay_steps)
        self.ear = SecondEar(cfg.client_delay_steps, sr_in=talker.sr)
        self.k = 0
        self.turns: dict = {}                  # (epoch, turn) -> TurnOut
        self.bc_n = 0
        self.stats = {"hops": 0, "ears_ms": [], "endpoints": [], "barges": [], "backchannels": [], "cancels": [],
                      "continues": []}
        self.closed = False
        from .control.barge import BargeAssist
        self.barge = BargeAssist()
        self.barge_cand = None
        self.last_ep_frame = None
        self.continues_posted = False

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        self.ears.open()
        self.tr.emit("engine.session", 0, language=self.lang, chunking=self.cfg.chunking or self.talker.chunking,
                     client_delay_steps=self.cfg.client_delay_steps)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.llm.cancel(self.ctl.request_id)
            self.talker.cancel(self.ctl.epoch + 1)
        finally:
            self.ears.close()

    # ------------------------------------------------------------------ helpers
    def _turn_out(self, epoch=None, turn=None) -> TurnOut:
        key = (self.ctl.epoch if epoch is None else epoch, self.ctl.turn if turn is None else turn)
        if key not in self.turns:
            self.turns[key] = TurnOut(key[0], key[1], Chunker(self.lang, self.cfg.chunking or self.talker.chunking))
        return self.turns[key]

    def _say(self, to: TurnOut, raw: str, speech: str, lane: str = "agent", holding: bool = False) -> None:
        if lane == "bc":
            self.bc_n += 1
            n = 10000 + self.bc_n
        else:
            n = to.n_seg
            to.n_seg += 1
        sid = seg_id(to.epoch, to.turn, n, lane)
        self.playout.register(to.epoch, to.turn, n, lane, raw, holding=holding)
        if lane == "agent":
            to.segs.append(sid)
        self.talker.say(sid, speech, to.epoch)
        self.tr.emit("tts.segment", self.k, sid=sid, lane=lane, holding=holding, text=raw, speech=speech,
                     epoch=to.epoch, turn_id=to.turn)

    # ------------------------------------------------------------------ the step
    def step(self, user: np.ndarray) -> dict:
        k = self.k
        user = np.asarray(user, np.float32)
        asr_delta = []
        heard = self.ear.take()
        self.ears.push(user, heard, self.ctl.ears_mode)
        if self.cfg.barge_assist:
            self.barge_cand = self.barge.step(k, user, heard,
                                              self.ctl.state is St.SPEAKING and self.playout.audible())
        for r in self.ears.poll():
            self._on_hop(r, k, asr_delta)
        if self.barge_cand is not None:
            on = self.barge_cand["onset"]
            to = self.turns.get((self.ctl.epoch, self.ctl.turn))
            why = self.barge.decide(self.barge_cand, [{"f": f, "p": p} for f, p in self.user.pieces if f >= on - 1],
                                    agent_text=(to.text if to else "") + " " + self._canned())
            if why:
                self.stats["barges"].append({"frame": on, "step": k, "by": why})
                self.tr.emit("ears.event", k, kind="BARGE", by=why, onset_frame=on,
                             onset_s=round((k - on + 1) * 0.08, 2), **{x: self.barge_cand[x] for x in
                                                                     ("mic_db", "agent_db", "coupling_db")})
                self.ctl.post(Intent("BARGE", "ears", k, payload={"onset_s": round((k - on + 1) * 0.08, 2),
                                                                  "frame": on, "by": why}))
        for e in self.llm.poll():
            self._on_llm(e, k)
        for e in self.talker.poll():
            self._on_tts(e, k)
        for a in self.ctl.tick(k):
            self._do(a, k)
        self._check_played_out(k)
        mix, agent, info = self.playout.release()
        self.ear.push(agent)
        if info["agent"] or info["bc"]:
            self.tr.emit("audio.release", k, agent=info["agent"], bc=info["bc"], epoch=self.playout.epoch)
        for key, to in self.turns.items():
            if not to.started and key[0] == self.ctl.epoch and \
                    self.playout.turn_first_step(key[0], key[1]) is not None:
                to.started = True
                self.ctl.post(Intent("AUDIO_STARTED", "playout", k + 1, epoch=key[0], payload={"turn": key[1]}))
        if self.playout.underruns and self.playout.underruns[-1]["step"] == k:
            self.tr.emit("audio.underrun", k, missing=self.playout.underruns[-1]["missing"])
        self.k += 1
        return {"audio": mix, "agent": agent, "asr_delta": "".join(asr_delta), "state": self.ctl.state.value}

    # ------------------------------------------------------------------ inputs
    def _on_hop(self, r: dict, k: int, asr_delta: list) -> None:
        self.stats["hops"] += 1
        self.stats["ears_ms"].append(r.get("ms"))
        self.tr.emit("ears.hop", k, n=r.get("n"), ms=r.get("ms"), enc_ms=r.get("enc_ms"), win_ms=r.get("win_ms"),
                     lag_steps=k - (4 * r.get("n", 0) + 3))
        if self.cfg.trace_frames:
            for f in r.get("frames", []):
                self.tr.emit("ears.frame", f.get("k"), **{x: f[x] for x in ("mode", "state", "p_end", "yield_prob",
                                                                          "bc_prob", "hazard", "db") if x in f})
        if r.get("pieces"):
            asr_delta.append(self.user.add(r["pieces"]))
            to = self.turns.get((self.ctl.epoch, self.ctl.turn))
            self.barge.words(r["pieces"], (to.text if to else "") + " " + self._canned())
            self.tr.emit("asr.piece", k, pieces=[[p["f"], p["p"]] for p in r["pieces"]])
            c = self.ctl
            if c.state is not St.LISTENING and self.last_ep_frame is not None and not self.continues_posted and \
                    any(p["f"] > self.last_ep_frame + 2 and any(ch.isalnum() for ch in p["p"]) and
                        not _tag(p["p"]) for p in r["pieces"]):
                self.continues_posted = True
                self.stats["continues"].append({"step": k})
                heard = self.playout.turn_first_step(c.epoch, c.turn, holding=False) is not None
                c.post(Intent("USER_CONTINUES", "ears", k, payload={"reply_heard": heard}))
        for e in r.get("events", []):
            self.tr.emit("ears.event", e.get("k"), kind=e["kind"], **{x: v for x, v in e.items()
                                                                       if x not in ("kind", "k")})
            if e["kind"] == "ENDPOINT":
                self.stats["endpoints"].append({"frame": e["k"], "step": k})
                self.last_ep_frame, self.continues_posted = e["k"], False
                self.ctl.post(Intent("ENDPOINT", "ears", k, payload={"text": self.user.pending(), "frame": e["k"]}))
            elif e["kind"] == "BARGE":
                self.stats["barges"].append({"frame": e["k"], "step": k, "onset_s": e.get("onset_s")})
                self.ctl.post(Intent("BARGE", "ears", k, payload={"onset_s": e.get("onset_s", 0.0),
                                                                  "frame": e["k"]}))
            elif e["kind"] == "BACKCHANNEL":
                self.stats["backchannels"].append({"frame": e["k"], "step": k})
                self.ctl.post(Intent("BACKCHANNEL", "ears", k, payload={"frame": e["k"]}))

    def _canned(self) -> str:
        """Fixed phrases the agent may be saying (the barge check must not take their echo for the user)."""
        return _langs.phrase(self.lang, "holding") + " " + _langs.phrase(self.lang, "reminder")

    def _on_llm(self, e: dict, k: int) -> None:
        kind, rid = e["ev"], e.get("rid")
        if kind == "text":
            self.tr.emit("llm.text", k, request_id=rid, text=e["text"])
            self.ctl.post(Intent("LLM_TEXT", "llm", k, payload={"text": e["text"]}, request_id=rid))
        elif kind == "decided":
            self.tr.emit("llm.decided", k, request_id=rid)
            self.ctl.post(Intent("LLM_DECIDED", "llm", k, request_id=rid))
        elif kind == "tool_start":
            self.tr.emit("tool.started", k, request_id=rid, name=e["name"], args=e.get("args"),
                         call_id=e.get("call_id"), turn_id=self.ctl.turn)
        elif kind == "waiting":
            po = self.playout
            silent = 0 if po.audible() else po.step - (po.last_agent_step() + po.client_delay_steps)
            self.tr.emit("tool.waiting", k, request_id=rid, s=e.get("s"), names=e.get("names"),
                         silent_steps=min(silent, 10 ** 4))
            self.ctl.post(Intent("LLM_WAIT", "llm", k, payload={"s": e.get("s"), "names": e.get("names"),
                                                                "silent_steps": silent}, request_id=rid))
        elif kind == "tool":
            self.conv.tool(e["name"], e.get("args") or {}, e.get("result", ""),
                           markup=getattr(self.llm, "call_markup", None))
            self.tr.emit("tool.completed", k, request_id=rid, name=e["name"], args=e.get("args"),
                         args_raw=e.get("args_raw"), result=e.get("result"), call_id=e.get("call_id"),
                         status=e.get("status", "completed"), ms=e.get("ms"),
                         turn_id=self.ctl.turn)
            self.ctl.post(Intent("LLM_TOOL", "llm", k, payload={"name": e["name"], "call_id": e.get("call_id")},
                                 request_id=rid))
        elif kind == "done":
            self.tr.emit("llm.done", k, request_id=rid, text=e.get("text"),
                         info={x: v for x, v in (e.get("info") or {}).items() if x in ("steps", "error", "skipped")})
            self.ctl.post(Intent("LLM_DONE", "llm", k, payload={"text": e.get("text", "")}, request_id=rid))

    def _on_tts(self, e: dict, k: int) -> None:
        sid = e.get("sid")
        if sid is None:
            return
        epoch, turn, seg, lane = parse_sid(sid)
        if e["ev"] == "audio":
            s = self.playout.segs.get((epoch, turn, seg))
            first = s is not None and s.queued == 0
            if not self.playout.enqueue(epoch, turn, seg, e["pcm"]):
                self.tr.emit("audio.stale", k, sid=sid, samples=len(e["pcm"]))
            elif first:
                self.tr.emit("tts.first_audio", k, sid=sid, lane=lane)
        elif e["ev"] == "done":
            self.playout.seg_done(epoch, turn, seg)
            self.tr.emit("tts.done", k, sid=sid, **{x: e[x] for x in ("first_audio_ms", "audio_s", "gen_s") if x in e})
        elif e["ev"] == "error":
            self.playout.seg_done(epoch, turn, seg)
            self.tr.emit("tts.error", k, sid=sid, error=e.get("error"))

    # ------------------------------------------------------------------ actions
    def _do(self, a, k: int) -> None:
        d = a.data
        if a.kind == "OPEN_TURN":
            self.user.take()
            self.conv.open_turn(a.turn, d["user_text"])
            self.tr.emit("asr.turn", k, turn_id=a.turn, text=d["user_text"], reason=a.reason)
        elif a.kind == "REQUEST_REPLY":
            self.llm.submit(d["request_id"], self.conv.for_llm())
            self.tr.emit("llm.request", k, request_id=d["request_id"], turn_id=a.turn,
                         n_messages=len(self.conv.messages))
        elif a.kind == "HOLDING_PHRASE":
            txt = _langs.phrase(self.lang, "reminder" if a.reason == "tool_wait" else "holding")
            self._say(self._turn_out(), txt, speech_text(txt, self.lang), holding=True)
        elif a.kind == "SPEAK":
            to = self._turn_out()
            to.text += d.get("text", "")
            for raw, sp in to.chunker.feed(d.get("text", "")):
                self._say(to, raw, sp)
        elif a.kind == "END_TEXT":
            to = self._turn_out()
            for raw, sp in to.chunker.flush():
                self._say(to, raw, sp)
            if not to.text and d.get("text"):
                to.text = d["text"]
        elif a.kind == "BACKCHANNEL":
            txt = _langs.phrase(self.lang, "backchannel")
            self._say(self._turn_out(turn=self.ctl.turn), txt, txt, lane="bc")
        elif a.kind == "CANCEL":
            old_epoch, turn = a.epoch, a.turn
            self.llm.cancel(d.get("cancelled_request"))
            self.talker.cancel(d["new_epoch"])
            heard, complete = self.playout.heard_text(old_epoch, turn)
            dropped = self.playout.flush(d["new_epoch"])
            self.stats["cancels"].append({"step": k, "reason": a.reason, "dropped": dropped, "heard": heard})
            self.tr.emit("audio.flush", k, epoch=old_epoch, new_epoch=d["new_epoch"], dropped_samples=dropped,
                         reason=a.reason)
            self.tr.emit("tts.cancel", k, epoch=d["new_epoch"])
            if d.get("drop_user_turn"):
                self.conv.drop_user_turn(turn)
                self.tr.emit("history.drop_user", k, turn_id=turn)
            elif heard:
                self.conv.agent(heard, cut=not complete)
                self.tr.emit("history.agent", k, turn_id=turn, heard=heard, cut=not complete)
            if a.reason == "barge_in" and not d.get("unheard"):
                onset_frames = int(round(float(d.get("onset_s") or 0.0) / 0.08))
                n = self.user.drop_before(int(d.get("barge_k", k)) - onset_frames - 2)
                if n:
                    self.tr.emit("asr.drop", k, pieces_dropped=n, why="before_barge_onset")
        elif a.kind == "CLOSE_TURN":
            to = self.turns.get((a.epoch, a.turn))
            if to is not None and to.text:
                heard, complete = self.playout.heard_text(a.epoch, a.turn)
                self.conv.agent(to.text if complete or not to.segs else heard, cut=False)
                self.tr.emit("history.agent", k, turn_id=a.turn, heard=to.text, cut=False)
            n = self.user.drop_before(k - self.cfg.overlap_grace_frames)
            if n:
                self.tr.emit("asr.drop", k, pieces_dropped=n, why="spoken_over_agent")
        elif a.kind == "EARS_RESET":
            self.ears.request_reset(keep_turn=a.reason == "user_continues")
            self.tr.emit("ears.reset", k, reason=a.reason)

    def _check_played_out(self, k: int) -> None:
        c = self.ctl
        if c.state is St.LISTENING or not c.text_done:
            return
        to = self._turn_out()
        if to.played_out_posted or to.chunker.buf:
            return
        segs = [s for s in self.playout.segs.values() if s.epoch == c.epoch and s.turn == c.turn and s.lane == "agent"]
        if not segs or self.playout.turn_audio_done(c.epoch, c.turn):
            to.played_out_posted = True
            c.post(Intent("PLAYED_OUT", "playout", k + 1, epoch=c.epoch, payload={"turn": c.turn}))
