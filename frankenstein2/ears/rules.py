# SPDX-License-Identifier: Apache-2.0
# Derived from tyger-s2s `app/plugins/ears/tyger/rules.py`, `session.py` and `constants.py` @97d011d
# (Copyright the tyger-s2s authors, Apache License 2.0, see LICENSES/Apache-2.0.txt). Modified for Frankenstein v2:
# the audio buffers (8 s windows, turn audio, barge seed) are removed because the ears stream and transcribe
# continuously; only the per-frame decision bookkeeping is kept, in pure Python (no numpy), so the same code runs in
# the ears process and in CPU unit tests with synthetic head outputs.
"""Turn rules around the turn-taking GRU: what one frame's head outputs mean for the turn.

Per 80 ms frame the GRU gives [state logits x4, hazard, yield, bc, p_end]; the rules add the frame's user energy and
the mode the controller forces ("listen" or "speaking"):

  * listen:   ENDPOINT when p_end >= ep_thr for ep_deb frames, once the user has spoken MIN_SPEECH_MS (no silence
              rule: only the head ends a turn); BACKCHANNEL when bc >= bc_thr for bc_deb frames (then a refractory
              period of bc_refractory frames);
  * speaking: BARGE when yield >= barge_thr for barge_deb frames, with ``onset_s``: how long the user had been
              talking by then (energy stretch with gaps up to ONSET_GAP_FRAMES, capped at ONSET_MAX_S);
  * both:     the first WARMUP_FRAMES frames after a reset or a mode switch never decide.

``on_frame`` returns (events, reset): ``reset`` = the GRU context must start over (after an ENDPOINT). A mode switch
(``switch_mode``) and an explicit ``reset()`` also restart the GRU context; the caller zeroes the GRU state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FRAME_S = 0.08
HOP_FRAMES = 4
SR = 16000
LISTEN, HOLD, SPEAK, BACKCHANNEL = 0, 1, 2, 3
STATE_NAMES = ("LISTEN", "HOLD", "SPEAK", "BACKCHANNEL")
VAD_DB = -50.0
MIN_SPEECH_MS = 350.0
WARMUP_FRAMES = 6
ONSET_GAP_FRAMES = 8
ONSET_MAX_S = 4.0


@dataclass
class Thresholds:
    ep_thr: float = 0.915
    ep_deb: int = 2
    barge_thr: float = 0.4
    barge_deb: int = 2
    bc_thr: float = 0.4
    bc_deb: int = 2
    bc_refractory: int = 15          # 1.2 s
    theta: float = 0.2               # hazard threshold of the display-only state decision
    fallback_sil_frames: int = 15    # the bundle's own policy (config.json default_policy): after the user spoke,
                                     # this many unvoiced frames (1.2 s) end the turn even if p_end stays under the
                                     # threshold (Tyger's live config leaves it off; 0 = off). v2 runs showed p_end
                                     # plateaus of 0.89-0.90 for 8 s after an Italian turn.
    floor_margin_db: float = 9.0     # voiced = above max(VAD_DB, tracked noise floor + margin)


def energy_db(frame) -> float:
    """dB of one frame of float samples (the rules' and the GRU's energy input)."""
    n = len(frame)
    if n == 0:
        return -120.0
    ms = sum(float(x) * float(x) for x in frame) / n
    return 20.0 * math.log10(math.sqrt(ms + 1e-12) + 1e-9)


def state_decision(state_logits, hazard, prev, theta, bc_thresh=0.5):
    """The 4-way state head's decision (display only: the ep / yield / bc heads make the real decisions)."""
    mx = max(state_logits)
    e = [math.exp(v - mx) for v in state_logits]
    s = sum(e)
    p = [v / s for v in e]
    if prev in (LISTEN, HOLD) and p[BACKCHANNEL] >= bc_thresh and p[BACKCHANNEL] == max(p):
        return BACKCHANNEL
    if prev in (LISTEN, HOLD):
        if hazard >= theta:
            return SPEAK
        return HOLD if p[HOLD] > p[LISTEN] else LISTEN
    if prev == SPEAK:
        return SPEAK if p[SPEAK] >= max(p[LISTEN], p[HOLD]) else LISTEN
    if prev == BACKCHANNEL:
        return BACKCHANNEL if p[BACKCHANNEL] == max(p) else LISTEN
    return LISTEN


@dataclass
class RulesState:
    thr: Thresholds = field(default_factory=Thresholds)
    mode: str = "listen"
    prev_state: int = LISTEN
    warm: int = 0
    ep_run: int = 0
    yield_run: int = 0
    bc_run: int = 0
    bc_cooldown: int = 0
    voiced_ms: float = 0.0
    in_turn: bool = False
    talk_frames: int = 0
    quiet_frames: int = 0
    frame: int = 0                   # frames seen since the session start (never reset)
    sil_run: int = 0                 # listen mode: unvoiced frames since the user's last voiced frame
    floor_db: float = -70.0          # tracked noise floor of the mic (slow up, fast down)

    def reset(self, keep_turn: bool = False) -> None:
        """A new turn (Tyger: Conversation resets the session at a turn end and after a barge). keep_turn: the user's
        turn is still open (the controller cancelled a reply because the user went on speaking): the next silence
        must end it, even when the continuation was already over by the time the reset arrived (else no endpoint
        ever fires and the turn waits for the user's next utterance)."""
        self.prev_state = SPEAK if self.mode == "speaking" else LISTEN
        self.warm = self.ep_run = self.yield_run = self.bc_run = self.bc_cooldown = 0
        self.voiced_ms, self.in_turn = (MIN_SPEECH_MS, True) if keep_turn else (0.0, False)
        self.talk_frames = self.quiet_frames = 0
        self.sil_run = 0

    def voiced(self, db: float) -> bool:
        return db > max(VAD_DB, self.floor_db + self.thr.floor_margin_db)

    def _track_floor(self, db: float) -> None:
        """Noise floor: follows a quieter frame at once, a louder one by 0.05 dB per frame (~0.6 dB/s), so speech
        never lifts it but a noisier line does over tens of seconds."""
        self.floor_db = db if db < self.floor_db else self.floor_db + min(0.05, db - self.floor_db)

    def switch_mode(self, mode: str) -> bool:
        """listen <-> speaking: fresh decision runs, previous state seeded from the new mode. True if it changed."""
        if mode not in ("listen", "speaking"):
            raise ValueError(mode)
        if mode == self.mode:
            return False
        self.warm = self.yield_run = self.bc_run = self.bc_cooldown = self.ep_run = 0
        self.talk_frames = self.quiet_frames = 0
        self.prev_state = SPEAK if mode == "speaking" else LISTEN
        self.mode = mode
        return True

    def gru_prev_state(self) -> int:
        """The GRU's previous-state input: forced by the mode, never the model's own guess."""
        return SPEAK if self.mode == "speaking" else LISTEN

    def _track_talk(self, voiced: bool) -> None:
        if self.talk_frames:
            self.talk_frames += 1
            self.quiet_frames = 0 if voiced else self.quiet_frames + 1
            if self.quiet_frames > ONSET_GAP_FRAMES:
                self.talk_frames = self.quiet_frames = 0
        elif voiced:
            self.talk_frames, self.quiet_frames = 1, 0

    def on_frame(self, outputs, act_db: float, left: int = 0):
        """outputs: 8 floats [state logits x4, hazard, yield, bc, p_end]; act_db: the frame's user energy; left: frames
        of the hop after this one (a barge's onset counts to the hop end). -> (events, reset)."""
        slog = list(outputs[:4])
        haz, yp, bcp, pep = (float(v) for v in outputs[4:8])
        k = self.frame
        self.frame += 1
        st = state_decision(slog, haz, self.prev_state, self.thr.theta)
        voiced = self.voiced(act_db)
        self._track_floor(act_db)
        events = []
        frame_ev = dict(kind="FRAME", k=k, mode=self.mode, hazard=round(haz, 4), yield_prob=round(yp, 4),
                        bc_prob=round(bcp, 4), p_end=round(pep, 4), db=round(act_db, 1))
        self.warm += 1
        if self.mode == "speaking":
            self._track_talk(voiced)
            if self.warm > WARMUP_FRAMES:
                self.yield_run = self.yield_run + 1 if yp >= self.thr.barge_thr else 0
                if self.yield_run >= self.thr.barge_deb:
                    onset_s = min((self.talk_frames + left) * FRAME_S, ONSET_MAX_S) if self.talk_frames else 0.0
                    events.append(dict(kind="BARGE", k=k, yield_prob=round(yp, 3), onset_s=round(onset_s, 2)))
                    self.yield_run = 0
            frame_ev["state"] = "SPEAK"
            events.append(frame_ev)
            self.prev_state = SPEAK
            return events, False
        if voiced:
            self.voiced_ms += FRAME_S * 1e3
            self.in_turn = True
            self.sil_run = 0
        elif self.in_turn:
            self.sil_run += 1
        spoke = self.in_turn and self.voiced_ms >= MIN_SPEECH_MS
        if self.bc_cooldown > 0:
            self.bc_cooldown -= 1
            self.bc_run = 0
        elif spoke and self.warm > WARMUP_FRAMES:
            self.bc_run = self.bc_run + 1 if bcp >= self.thr.bc_thr else 0
            if self.bc_run >= self.thr.bc_deb:
                events.append(dict(kind="BACKCHANNEL", k=k, bc_prob=round(bcp, 3)))
                self.bc_run = 0
                self.bc_cooldown = self.thr.bc_refractory
        if self.warm > WARMUP_FRAMES:
            self.ep_run = self.ep_run + 1 if pep >= self.thr.ep_thr else 0
        else:
            self.ep_run = 0
        by_sil = bool(self.thr.fallback_sil_frames) and self.sil_run >= self.thr.fallback_sil_frames
        if spoke and (self.ep_run >= self.thr.ep_deb or by_sil):
            events.append(dict(kind="ENDPOINT", k=k, p_end=round(pep, 3), voiced_ms=round(self.voiced_ms),
                               by="head" if self.ep_run >= self.thr.ep_deb else "silence"))
            self.voiced_ms, self.in_turn, self.sil_run = 0.0, False, 0
            self.prev_state = LISTEN
            self.warm = self.bc_run = self.bc_cooldown = self.ep_run = 0
            frame_ev["state"] = "idle"
            events.append(frame_ev)
            return events, True
        frame_ev["state"] = STATE_NAMES[st] if self.in_turn else ("idle" if not voiced else "LISTEN")
        events.append(frame_ev)
        self.prev_state = LISTEN
        return events, False
