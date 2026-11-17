"""M1-force runtime hook: instance-level wraps on a live NVIDIA VoiceChat wrapper (no change to v2/ or NeMo).

Per real frame t, inside the wrapper's own loop (``_apply_rnnt_turn_taking`` is called once per frame right after the
RNNT step, W:2503-2513; NVIDIA's redirect forcing lives at its top, W:3034-3044):
  1. predicted = gen_text[0, t]   (the duplex model's own id after FC / forced-turn-taking handling)
  2. causal user side: ASR words closed at <= t, ASR EOU at t, ASR non-blank at t
  3. brain output whose availability frame <= t is handed to the AgentChannelForcer
  4. s = forcer.step(t, predicted); if s.write is set: gen_text[0, t] = s.write   (same assignment as the redirect)
  5. the ORIGINAL _apply_rnnt_turn_taking runs: it reads the written id (agent_speaking updates, self-play
     suppression, RNNT EOU forced BOS, RNNT barge-in forced EOS, max-response caps)
  6. emitted = gen_text[0, t] after step 5 (= the id the wrapper reports as output, W:2514); an EOS written by the
     wrapper while the forcer was speaking = duplex barge-in -> forcer.cut()
  7. the TalkerAdapter (CPU, fake talker + PlaybackGate) gets the frame; an adapter barge-in also cuts the forcer
  8. if the forcer asked for a reply: the brain runs NOW on the user words closed so far (causal) with its own cache;
     the duplex cache is fingerprinted before and after; reply ids become available at
     t + ceil(measured wall ms / 80) (as if the brain ran beside the loop at its measured speed).
``_apply_fc_state_machine`` is wrapped so the duplex never starts its own FC: a non-PAD function id is replaced by PAD
before the state machine sees it (logged as the duplex's wanted id). Tools go through the brain (T1).
``Observer`` = the same wrap without any write (M0), so M0 has the same per-frame trace.
"""

from __future__ import annotations

import json
import math
import sys
import time

from frankenstein_s2s import tokens as T
from frankenstein_s2s.events import Frame
from frankenstein_s2s.forcing import AgentChannelForcer, ForceState

OPENER_USER = "[call connected]"


class FrameLog:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)

    def write(self, row):
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()


def _caller_cache():
    """The duplex loop's live cache, read from the calling wrapper frame (``dynamic_cache`` local of
    infer_one_step), for the cache-isolation check. None if not found."""
    f = sys._getframe(2)
    for _ in range(4):
        if f is None:
            return None
        if "dynamic_cache" in f.f_locals:
            return f.f_locals["dynamic_cache"]
        f = f.f_back
    return None


class UserSide:
    """Causal view of the frozen ASR track: what is known at frame k."""

    def __init__(self, asr: dict):
        self.words = sorted(asr["words"], key=lambda w: w["k"])
        self.nonblank = set(asr["nonblank"])
        self.eou = set(asr["eou"])
        self.i = 0
        self.pending: list = []
        self.run = 0

    def advance(self, k: int) -> dict:
        new = []
        while self.i < len(self.words) and self.words[self.i]["k"] <= k:
            new.append(self.words[self.i]["word"])
            self.i += 1
        self.pending += new
        self.run = self.run + 1 if k in self.nonblank else 0
        return {"new_words": new, "eou": k in self.eou, "speech": k in self.nonblank, "run": self.run}

    def take_turn(self) -> str:
        text = " ".join(self.pending).strip()
        self.pending = []
        return text


class M1Force:
    def __init__(self, wrapper, brain, tok, asr: dict, adapter, ftr, frames: FrameLog, brain_log: FrameLog,
                 execute_tool, mode: str = "force", tool_turns: bool = True):
        assert mode in ("force", "observe")
        self.w, self.brain, self.tok, self.adapter, self.ftr = wrapper, brain, tok, adapter, ftr
        self.frames, self.brain_log, self.execute_tool = frames, brain_log, execute_tool
        self.mode = mode
        self.user = UserSide(asr)
        self.forcer = AgentChannelForcer(bos_fallback_frames=8, wait_max=150, tail_max=60, tracer=ftr)
        self.messages: list = []
        self.sched: list = []          # (avail_k, [ids], done)
        self.job_active = False
        self.tool_turns = tool_turns
        self.n_hook = 0
        self.cache_checks: list = []
        self.fc_suppressed: list = []
        self.mismatch: list = []       # frames where the ORIGINAL wrapper logic changed the id we wrote
        self.gaps: list = []           # hook calls that were not consecutive frames
        self.last_t = -1
        self.pump = None               # optional callable: deliver fake-talker audio through the PlaybackGate
        self._orig = {}

    # ---------------------------------------------------------------------------------------- install
    def install(self):
        w = self.w
        self._orig["rnnt"] = w._apply_rnnt_turn_taking
        self._orig["fc"] = w._apply_fc_state_machine
        w._apply_rnnt_turn_taking = self._rnnt_hook
        if self.mode == "force":
            w._apply_fc_state_machine = self._fc_hook

        def uninstall():
            for k, attr in (("rnnt", "_apply_rnnt_turn_taking"), ("fc", "_apply_fc_state_machine")):
                if attr in w.__dict__:
                    delattr(w, attr)
        return uninstall

    # ---------------------------------------------------------------------------------------- hooks
    def _fc_hook(self, *args, **kw):
        if args:
            raise RuntimeError("_apply_fc_state_machine called positionally: hook layout changed")
        t = int(kw["current_frame_idx"])
        g = kw["gen_function_text"]
        wanted = int(g[0, t])
        if wanted != T.PAD_ID:
            g[0, t] = T.PAD_ID
            kw["function_predicted_tokens"][:, kw["frame_offset"]] = T.PAD_ID
            kw["function_predicted_token"] = g[:, t]
            self.fc_suppressed.append({"k": t, "wanted": wanted})
            self.ftr.emit("thinker.frame", t, predicted=None, fed=None, func_id=T.PAD_ID, func_wanted=wanted,
                          source="fc_suppressed")
        return self._orig["fc"](**kw)

    def _rnnt_hook(self, t, gen_text, *rest, **kw):
        self.n_hook += 1
        t = int(t)
        predicted = int(gen_text[0, t])
        u = self.user.advance(t)
        if u["eou"]:
            self.forcer.on_eou(t)
            self.ftr.emit("turn.eou", t, source="asr")
        elif u["speech"] and self.forcer.eou_k is not None:
            self.forcer.eou_k = None            # the caller resumed: no BOS fallback on a mid-utterance pause
        s = None
        written = None
        state_before = None
        if self.mode == "force":
            while self.sched and self.sched[0][0] <= t:
                _, ids, done = self.sched.pop(0)
                self.forcer.add_reply(ids, done=done)
            state_before = self.forcer.state
            s = self.forcer.step(t, predicted)
            if s.write is not None:
                gen_text[0, t] = s.write
                written = s.write
        out = self._orig["rnnt"](t, gen_text, *rest, **kw)
        emitted = int(gen_text[0, t])
        expected = written if written is not None else predicted
        duplex_change = emitted != expected
        if duplex_change and self.mode == "force":
            self.mismatch.append({"k": t, "expected": expected, "emitted": emitted})
            if emitted == T.BOS_ID and state_before is ForceState.LISTEN and self.forcer.state is ForceState.LISTEN:
                self.forcer._open(t, write=None, why="duplex_wrapper_bos")   # the audio path's own timing decision
            if emitted == T.EOS_ID and state_before in (ForceState.WAIT_TEXT, ForceState.FORCING):
                self.ftr.emit("adapter.barge_in", t, source="duplex_rnnt", what="wrapper_forced_eos",
                              action="cut forced reply")
                self.forcer._close(t, "duplex_rnnt_eos", write=None)   # the wrapper already wrote EOS at t
        # adapter (CPU) sees what the talker would get
        if self.mode == "force":
            to_adapter = s.to_adapter if not duplex_change else emitted
        else:
            to_adapter = emitted
        n_cmd = len(self.adapter.commands)
        self.adapter.feed(Frame(t, to_adapter, user_speech=u["speech"]))
        if self.mode == "force" and any(type(c).__name__ == "Cancel" for c in self.adapter.commands[n_cmd:]):
            self.forcer.cut()
        piece = self.tok.decode([emitted]) if emitted >= T.FIRST_TEXT_ID else None
        self.ftr.emit("thinker.frame", t, predicted=predicted, fed=emitted, forced=written is not None and written != predicted,
                      piece=piece, source=self.mode)
        row = {"k": t, "predicted": predicted, "written": written, "emitted": emitted, "to_adapter": to_adapter,
               "forcer": self.forcer.state.value if self.mode == "force" else None,
               "speech": u["speech"], "eou": u["eou"], "new_words": u["new_words"] or None,
               "t_ns": time.monotonic_ns()}
        self.frames.write(row)
        if self.last_t >= 0 and t != self.last_t + 1:
            self.gaps.append({"k": t, "gap_from": self.last_t})
        self.last_t = t
        if self.pump is not None and t % 5 == 0:
            self.pump()
        # history bookkeeping on turn close
        if self.mode == "force":
            self._maybe_close_history(t)
            if self.forcer.need_reply and not self.job_active:
                self._start_brain(t)
        return out

    # ---------------------------------------------------------------------------------------- brain
    def _maybe_close_history(self, t):
        if getattr(self, "_open_turn", None) is not None and self.forcer.state is ForceState.LISTEN:
            dropped = sum(len(x[1]) for x in self.sched)
            self.sched = []                    # ids of a closed turn must never reach the next one
            if dropped:
                self.ftr.emit("force.turn_close", t, why="drop_unreleased_reply_ids", dropped_reply_ids=dropped)
            spoken = self.tok.decode(self.forcer.turn_ids) if self.forcer.turn_ids else ""
            why = self.forcer.log[-1][2] if self.forcer.log and self.forcer.log[-1][1] == "turn_close" else "?"
            self.messages.append({"role": "assistant", "content": spoken})
            self.brain_log.write({"ev": "history.agent", "k": t, "spoken": spoken, "why": why,
                                  "planned": self._open_turn.get("text")})
            self._open_turn = None
            self.job_active = False

    def _start_brain(self, t):
        self.forcer.need_reply = False
        self.job_active = True
        user = self.user.take_turn()
        self.messages.append({"role": "user", "content": user or OPENER_USER})
        dup = _caller_cache()
        fp0 = self.brain_fp(dup)
        t0 = time.monotonic()
        self.ftr.emit("brain.request", t, user_text=user or OPENER_USER, n_messages=len(self.messages))
        r = self.brain.run(self.messages, decide_tool=self.tool_turns and bool(user))
        rec = {"ev": "brain.run", "k": t, "user": user, "prefill_ms": r["prefill_ms"], "ctx_len": r["ctx_len"],
               "tool_margin": r["tool_margin"], "func_top1": r["func_top1"]}
        ms = (time.monotonic() - t0) * 1e3
        if r.get("tool"):
            call = r["tool"]
            rec["call"] = call
            self.ftr.emit("brain.tool_decision", t, decision="call", margin=r["tool_margin"], call_text=call["text"])
            result, exec_ms = self.execute_tool(call.get("parsed"))
            rec["tool_result"], rec["tool_exec_ms"] = result, exec_ms
            self.messages.append({"role": "assistant", "content": call["text"]})
            self.messages.append({"role": "user", "content": f"<TOOL_RESPONSE>[{result}]</TOOL_RESPONSE>"})
            r2 = self.brain.run(self.messages, decide_tool=False)
            self.messages = self.messages[:-2]          # history keeps call + result folded into the reply turn below
            self.messages.append({"role": "assistant", "content": call["text"]})
            self.messages.append({"role": "user", "content": f"<TOOL_RESPONSE>[{result}]</TOOL_RESPONSE>"})
            rec["resume"] = {"prefill_ms": r2["prefill_ms"], "text": r2["text"]}
            r = r2
        else:
            self.ftr.emit("brain.tool_decision", t, decision="none", margin=r["tool_margin"])
        ms_total = (time.monotonic() - t0) * 1e3
        fp1 = self.brain_fp(dup)
        self.cache_checks.append({"k": t, "before": fp0, "after": fp1, "same": fp0 == fp1,
                                  "distinct_object": dup is not self.brain.last_cache})
        ids = [i for i in r["ids"] if i >= T.FIRST_TEXT_ID]
        dropped = [i for i in r["ids"] if i < T.FIRST_TEXT_ID]
        per_tok = r.get("decode_ms_per_tok", 0.0)
        first_ms = max(ms_total - per_tok * max(0, len(ids) - 1), r.get("prefill_ms", 0.0) + per_tok)
        for i, tid in enumerate(ids):
            avail = t + math.ceil((first_ms + per_tok * i) / 80.0)
            self.sched.append((avail, [tid], i == len(ids) - 1))
        if not ids:
            self.sched.append((t + math.ceil(ms_total / 80.0), [], True))
        self.sched.sort(key=lambda x: x[0])
        self._open_turn = {"k": t, "text": r["text"]}
        rec.update({"reply": r["text"], "n_ids": len(ids), "dropped_special": dropped, "wall_ms": round(ms_total, 1),
                    "first_avail_k": self.sched[0][0] if self.sched else None, "decode_ms_per_tok": per_tok,
                    "duplex_cache_same": fp0 == fp1})
        self.ftr.emit("brain.first_token", t, avail_k=rec["first_avail_k"], wall_ms=round(first_ms, 1))
        self.ftr.emit("brain.last_token", t, avail_k=self.sched[-1][0] if self.sched else None)
        self.ftr.emit("brain.text", t, text=r["text"])
        self.brain_log.write(rec)

    @staticmethod
    def brain_fp(cache):
        from e1.brain import fingerprint
        return fingerprint(cache)
