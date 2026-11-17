"""E1j: the controller (frankenstein_s2s/controller.py, CONTROL.md) drives the two-LLM session.

The duplex Thinker runs free (its text channel is never written: timing shell). Per real frame k, inside the
wrapper's loop (instance-level wraps, no NVIDIA / v2 code edited):
  sources -> intents   Thinker: BOS -> TAKE_TURN(bos), tool-call start id 20 -> TAKE_TURN(tool_call_start)
                                (suppressed before the engine's FC state machine), EOS -> YIELD
                       ASR (frozen causal track): activity -> USER_SPEECH, a piece of a not-yet-closed word ->
                                WORD_OPEN, a closed word -> WORDS, end of utterance -> EOU
                       Nano worker (polled, never awaited): tokens -> REPLY_TOKENS, a tool run -> TOOL_CALL (k) and
                                TOOL_RESULT (k + 1), end -> REPLY_DONE; all tagged with the request id they answer
  controller.tick(k)   -> actions, executed here: OPEN_TURN (BOS to the talker adapter), REQUEST_REPLY (submit to
                                the worker; a superseded request is cancelled there), SPEAK (ids queued, one per
                                frame to the adapter), CANCEL (adapter cancel + flush, worker cancel), CLOSE_TURN
                                (EOS to the adapter once the queued ids are out), HOLDING_PHRASE (E1l: the
                                per-language phrase as its own short adapter turn; when the real reply starts while
                                the phrase is still audible, the phrase is cut with an adapter cancel = new epoch,
                                and the reply opens a fresh turn in that epoch)
The controller is the only writer of turn state; the adapter's own barge-in is off (barge_in="thinker").
"""

from __future__ import annotations

import json
import time

from frankenstein_s2s import tokens as T
from frankenstein_s2s.controller import Controller, Intent

TAG = ("<", ">")


class CtlDriver:
    def __init__(self, wrapper, worker, tok, asr: dict, adapter, ftr, frames, brain_log, controller: Controller,
                 sotc_turn: bool = True, lang: str = "it", live=None, wav=None):
        self.w, self.worker, self.tok, self.adapter, self.ftr = wrapper, worker, tok, adapter, ftr
        self.frames, self.blog, self.ctl, self.sotc_turn = frames, brain_log, controller, sotc_turn
        self.pieces = {int(k): [p for p in v if not (p.strip().startswith(TAG[0]) and p.strip().endswith(TAG[1]))]
                       for k, v in (asr.get("pieces") or {}).items()}
        self.words_at: dict = {}
        for wd in asr["words"]:
            self.words_at.setdefault(int(wd["k"]), []).append(wd["word"])
        # speech evidence = LEXICAL ASR activity (E1k): a piece with a letter or digit in the last 3 frames.
        # Punctuation / space / language-tag emissions come late after the caller stopped (e1j: '.' '<it-IT>' at
        # frame 611, 13 frames after the end of U4) and are not speech.
        lex = sorted(k for k, v in self.pieces.items() if any(ch.isalnum() for p in v for ch in p))
        lexset = set(lex)
        self.nonblank = {k for k in range(0, (max(lex) + 3) if lex else 0) if any(j in lexset for j in (k, k - 1, k - 2))}
        self.eou = set(asr["eou"])
        self.messages: list = []
        self.base_k = -1                 # caller text of the current turn = pieces in (base_k, k]
        self.req: dict = {}              # rid -> record
        self.active_rid = None
        self.queue: list = []
        self.close_pending = False
        self.pending_bos = False
        self.spoken: list = []
        self.n_hook = 0
        self.last_t = -1
        self.gaps: list = []
        self.stale = 0
        self.fc_suppressed: list = []
        self.pump = None
        self.pump_every = 5
        self.audible = None              # G0: callable -> is agent audio still playing (default: adapter.busy)
        self.holding_cut = "audible"     # "audible": cut a still-audible phrase when the reply starts (E1l);
                                         # "never": the reply queues behind the phrase (G0: a cut in a talker
                                         # underrun dropped the end of the phrase)
        self._orig = {}
        self.lang = lang
        self.live, self.wav = live, wav  # G0: live streaming ASR (e1/live_asr.py) instead of the frozen track
        self.lex_live: set = set()
        self.nonblank = set(self.nonblank)
        self.eou = set(self.eou)
        self.holding: list = []          # one record per holding phrase said
        self.reply_bos_needed = False    # a holding phrase closed the turn: the reply opens a new one

    # ------------------------------------------------------------------ install
    def install(self):
        w = self.w
        self._orig["rnnt"], self._orig["fc"] = w._apply_rnnt_turn_taking, w._apply_fc_state_machine
        w._apply_rnnt_turn_taking = self._rnnt_hook
        w._apply_fc_state_machine = self._fc_hook

        def uninstall():
            for attr in ("_apply_rnnt_turn_taking", "_apply_fc_state_machine"):
                if attr in w.__dict__:
                    delattr(w, attr)
        return uninstall

    def _fc_hook(self, *args, **kw):
        t = int(kw["current_frame_idx"])
        g = kw["gen_function_text"]
        wanted = int(g[0, t])
        if wanted != T.PAD_ID:
            if wanted == T.SOTC_ID and self.sotc_turn:
                self.ctl.post(Intent("TAKE_TURN", "thinker", t, payload="tool_call_start"))
            g[0, t] = T.PAD_ID
            kw["function_predicted_tokens"][:, kw["frame_offset"]] = T.PAD_ID
            kw["function_predicted_token"] = g[:, t]
            self.fc_suppressed.append({"k": t, "wanted": wanted})
        return self._orig["fc"](**kw)

    # ------------------------------------------------------------------ per frame
    def _rnnt_hook(self, t, gen_text, *rest, **kw):
        self.n_hook += 1
        t = int(t)
        if self.live is not None:
            self._ingest_live(t, self.live.step(t, self.wav))
        predicted = int(gen_text[0, t])
        out = self._orig["rnnt"](t, gen_text, *rest, **kw)
        emitted = int(gen_text[0, t])
        if predicted == T.BOS_ID or emitted == T.BOS_ID:
            self.ctl.post(Intent("TAKE_TURN", "thinker", t, payload="bos" if predicted == T.BOS_ID else "wrapper_bos"))
        elif predicted == T.EOS_ID:
            self.ctl.post(Intent("YIELD", "thinker", t))
        speech = t in self.nonblank
        if speech:
            self.ctl.post(Intent("USER_SPEECH", "asr", t))
        for wd in self.words_at.get(t, []):
            self.ctl.post(Intent("WORDS", "asr", t, payload=wd))
        if self.pieces.get(t):
            self.ctl.post(Intent("WORD_OPEN", "asr", t))
        if t in self.eou:
            self.ctl.post(Intent("EOU", "asr", t))
        self._poll_worker(t)
        self._audio_end_check(t)
        for a in self.ctl.tick(t):
            self._do(a, t)
        to_adapter = self._adapter_id()
        self.adapter.feed(_frame(t, to_adapter))
        self.frames.write({"k": t, "predicted": predicted, "emitted": emitted, "to_adapter": to_adapter,
                           "state": self.ctl.state.value, "epoch": self.ctl.epoch, "speech": speech,
                           "eou": t in self.eou, "t_ns": time.monotonic_ns()})
        if self.last_t >= 0 and t != self.last_t + 1:
            self.gaps.append({"k": t, "gap_from": self.last_t})
        self.last_t = t
        if self.pump is not None and t % self.pump_every == 0:
            self.pump()
        return out

    def _audio_end_check(self, t):
        """G3: while the controller drains (text final, audio playing), report the end of our audio once every queued
        id was handed over, the talker is idle and nothing is audible."""
        if self.ctl.state.value != "draining" or self.queue or self.close_pending or self.pending_bos:
            return
        audible = self.audible() if self.audible is not None else bool(self.adapter.busy)
        if not audible and not self.adapter.busy:
            self.ctl.post(Intent("AUDIO_ENDED", "talker", t, epoch=self.ctl.epoch))
            self.blog.write({"ev": "audio.ended", "k": t})

    def _ingest_live(self, t, ev):
        """Live ASR events of frame t -> the same per-frame tables the frozen track fills (lexical activity smoothed
        over 3 frames, closed words, EOU)."""
        if ev["pieces"]:
            self.pieces[t] = list(ev["pieces"])
            if any(ch.isalnum() for p in ev["pieces"] for ch in p):
                self.lex_live.add(t)
        if any(j in self.lex_live for j in (t, t - 1, t - 2)):
            self.nonblank.add(t)
        for w in ev["words"]:
            self.words_at.setdefault(t, []).append(w)
        if ev["eou"]:
            self.eou.add(t)

    def _poll_worker(self, t):
        for ev in self.worker.poll():
            kind, rid = ev[0], ev[1]
            r = self.req.get(rid)
            ep = r["epoch"] if r else -1
            if kind == "tok":
                if ev[2] < T.FIRST_TEXT_ID:
                    self.ftr.emit("adapter.drop", t, what="brain_special_id", action="dropped", id=ev[2], rid=rid)
                    continue
                if r is not None and r.get("first_tok_k") is None:
                    r["first_tok_k"], r["first_tok_ns"] = t, ev[3]
                self.ctl.post(Intent("REPLY_TOKENS", "nano", t, payload=[ev[2]], request_id=rid, epoch=ep))
            elif kind == "tool":
                cid = f"r{rid}-{len(r.get('calls', [])) if r else 0}"
                if r is not None:
                    r.setdefault("calls", []).append(ev[2])
                self.ctl.post(Intent("TOOL_CALL", "nano", t, payload={"name": ev[2]["name"]}, request_id=rid,
                                     epoch=ep, call_id=cid))
                self.ctl.post(Intent("TOOL_RESULT", "tool", t + 1, call_id=cid, epoch=ep))
                self.blog.write({"ev": "brain.tool", "k": t, "rid": rid, "call": ev[2]})
            elif kind == "done":
                if r is not None:
                    r.update(done_k=t, text=ev[2], info=ev[4], new_msgs=ev[3])
                self.ctl.post(Intent("REPLY_DONE", "nano", t, request_id=rid, epoch=ep))
                self.blog.write({"ev": "brain.done", "k": t, "rid": rid, "reply": ev[2], "info": ev[4]})
            elif kind == "decided":
                self.ctl.post(Intent("TOOL_DECIDED", "nano", t, request_id=rid, epoch=ep))
                self.blog.write({"ev": "brain.decided", "k": t, "rid": rid, "what": ev[2]})
            elif kind == "skipped":
                self.blog.write({"ev": "brain.skipped", "k": t, "rid": rid})

    def _caller_text(self, k):
        ps = [p for j in sorted(self.pieces) if self.base_k < j <= k for p in self.pieces[j]]
        return " ".join("".join(ps).replace("▁", " ").split())

    def _do(self, a, t):
        if a.kind == "OPEN_TURN":
            self.pending_bos = True
            self.queue, self.close_pending, self.spoken = [], False, []
            self.reply_bos_needed = False
        elif a.kind == "HOLDING_PHRASE":
            from frankenstein_s2s.holding import holding_phrase
            text = holding_phrase(self.lang)
            self.pending_bos = False                 # the phrase is its own adapter turn
            self.adapter.end_turn(t)                 # closes the empty reply turn opened by BOS (if any)
            ok = self.adapter.inject_utterance(t, text, reason="holding")
            self.holding.append({"k": t, "text": text, "ok": ok, "epoch": self.adapter.epoch,
                                 "adapter_turn": self.adapter.turns_started if ok else None,
                                 "request_id": a.data.get("request_id"), "cut_k": None})
            self.reply_bos_needed = True
            self.blog.write({"ev": "brain.holding", "k": t, "text": text, "ok": ok})
        elif a.kind == "REQUEST_REPLY":
            text = self._caller_text(t)
            sup = a.data.get("supersedes")
            if sup is not None:
                self.worker.cancel(sup)
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages[-1]["content"] = text
                elif text:
                    self.messages.append({"role": "user", "content": text})
            elif text:
                self.messages.append({"role": "user", "content": text})
            rid = a.data["request_id"]
            self.req[rid] = {"rid": rid, "k": t, "user": text, "epoch": self.ctl.epoch, "reason": a.reason,
                             "supersedes": sup, "t_submit_ns": time.monotonic_ns()}
            self.active_rid = rid
            self.worker.submit(rid, self.messages)
            self.blog.write({"ev": "brain.submit", "k": t, "rid": rid, "user": text, "supersedes": sup})
        elif a.kind == "SPEAK":
            if self.reply_bos_needed:
                self.reply_bos_needed = False
                if self.holding_cut == "audible" and (self.audible() if self.audible is not None
                                                      else self.adapter.busy):           # phrase still audible: cut
                    self.adapter.cancel(t, "holding_cut")
                    if self.holding:
                        self.holding[-1]["cut_k"] = t
                self.pending_bos = True
            self.queue += list(a.data.get("ids", []))
        elif a.kind == "END_TEXT":                    # G3: text final, the turn stays open until its audio ends
            self.close_pending = True
            self.text_closed = True
            self._history(t, "reply_done")
        elif a.kind == "CLOSE_TURN":
            if getattr(self, "text_closed", False):     # after END_TEXT: EOS and history are already done
                self.text_closed = False
            else:
                self.close_pending = True
                self._history(t, a.reason)
        elif a.kind == "CANCEL":
            self.reply_bos_needed = False
            self.adapter.cancel(t, a.reason)
            self.worker.cancel(self.active_rid)
            self.queue, self.close_pending, self.pending_bos = [], False, False
            if getattr(self, "text_closed", False):     # barge-in on the audio tail: the text is already in history
                self.text_closed = False
                self.blog.write({"ev": "history.agent", "k": t, "why": a.reason, "spoken": "", "audio_cut": True})
            else:
                self._history(t, a.reason)
        elif a.kind == "DROP_STALE" and a.data.get("intent") == "REPLY_TOKENS":
            self.stale += 1

    def _history(self, t, why):
        r = self.req.get(self.active_rid) or {}
        self.messages.extend(r.get("new_msgs") or [])
        ids = self.spoken + (self.queue if why == "reply_done" else [])   # queued ids of a closed turn still play
        spoken = self.tok.decode(ids) if ids else ""
        if spoken:
            self.messages.append({"role": "assistant", "content": spoken})
        self.blog.write({"ev": "history.agent", "k": t, "why": why, "spoken": spoken})
        self.base_k = t

    def _adapter_id(self):
        if self.pending_bos:
            self.pending_bos = False
            return T.BOS_ID
        if self.queue:
            tid = self.queue.pop(0)
            self.spoken.append(tid)
            return tid
        if self.close_pending:
            self.close_pending = False
            return T.EOS_ID
        return None                         # waiting: neither text nor PAD (never a premature burst commit)

    def summary(self) -> dict:
        return {"hook_calls": self.n_hook, "frames_last": self.last_t, "n_gaps": len(self.gaps), "gaps": self.gaps[:20],
                "fc_suppressed": self.fc_suppressed, "stale_tokens": self.stale, "holding": self.holding,
                "requests": [{k: v for k, v in r.items() if k != "new_msgs"} for r in self.req.values()],
                "actions": [{"k": a.k, "kind": a.kind, "reason": a.reason, "epoch": a.epoch, "turn": a.turn,
                             **{x: y for x, y in a.data.items() if x != "ids"}} for a in self.ctl.actions],
                "messages": self.messages}


def _frame(k, text_id):
    from frankenstein_s2s.events import Frame
    return Frame(k, text_id, user_speech=False)
