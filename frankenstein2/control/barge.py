"""Barge-in assist: a cheap, echo-aware user-speech detector that runs beside the GRU's yield head while the agent's
audio is audible. It needs no model: the mic and the exact agent audio the user hears (the 2nd ear) are both here,
frame by frame, and the streaming ASR is already running.

Per 80 ms step (agent audible):
  * mic level m and 2nd-ear level a (dB); the echo coupling c = median of (m - a) over steps where the agent spoke
    and the user did not (no recent ASR words), starting from ``coupling0_db``;
  * the step is user speech when m > max(floor + floor_margin, a + c + echo_margin) - the mic is clearly louder than
    the agent's own echo predicts (a clean mic: c is very low and any speech counts);
  * a candidate onset = the first of ``min_frames`` consecutive user-speech steps (short pauses tolerated);
  * BARGE when an ASR word confirms it: a piece from the onset frame on that starts a word longer than one
    character and is not a backchannel ("sì", "ok", "mh", "yeah", "oui", ...), or when the user speech run lasts
    ``force_frames`` without any word (a long cut-in the ASR has not caught yet).
Backchannels in the agent's pauses do not barge (Tyger's I-18); the agent's own echo does not either (I-26), as long as
the coupling estimate holds. Model-free; unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

BACKCHANNEL_WORDS = {
    "sì", "si", "mh", "mhm", "mm", "mmh", "ok", "okay", "certo", "ah", "eh", "oh", "bene", "va", "d'accordo",
    "yes", "yeah", "yep", "uh", "huh", "hmm", "right", "sure", "oui", "ouais", "d'accord", "hm", "ja", "no",
}


def db(x) -> float:
    n = len(x)
    if not n:
        return -120.0
    return 10.0 * math.log10(sum(float(v) * float(v) for v in x) / n + 1e-12)


@dataclass
class BargeAssist:
    min_frames: int = 3
    gap_frames: int = 2              # unvoiced steps tolerated inside a run
    force_frames: int = 12           # 0.96 s of user speech over the agent without a confirming word
    floor_margin_db: float = 9.0
    echo_margin_db: float = 10.0
    coupling0_db: float = -30.0
    min_agent_db: float = -50.0      # the agent is "speaking" in this step
    floor_db: float = -70.0
    coup: list = field(default_factory=list)
    run: int = 0
    gap: int = 0
    onset: int | None = None
    fired: bool = False
    last_word_k: int = -10 ** 9
    log: list = field(default_factory=list)

    prior_n: int = 20                # the prior counts as this many observations in the median
    learned_n: int = 20              # long_speech needs this many observed echo steps (a learned coupling)

    @property
    def coupling(self) -> float:
        xs = sorted([self.coupling0_db] * self.prior_n + self.coup)
        return xs[len(xs) // 2]

    def reset(self) -> None:
        self.run = self.gap = 0
        self.onset = None
        self.fired = False

    def step(self, k: int, mic, agent, audible: bool) -> dict | None:
        """One step. -> a candidate dict {"onset": k0, "frames": n} while a user-speech run is open, else None."""
        m, a = db(mic), db(agent)
        self.floor_db = m if m < self.floor_db else self.floor_db + min(0.05, m - self.floor_db)
        if not audible:
            self.reset()
            return None
        agent_on = a > self.min_agent_db
        if agent_on and k - self.last_word_k > 8:
            # echo level: the mic over agent-on steps with no user word around (the ASR lags a few frames, so a few
            # user steps get in: the median over up to 100 steps + the prior ignores them)
            self.coup.append(m - a)
            self.coup = self.coup[-100:]
        thr = self.floor_db + self.floor_margin_db
        if agent_on:
            thr = max(thr, a + self.coupling + self.echo_margin_db)
        user = m > max(thr, -50.0)
        if user:
            if self.run == 0:
                self.onset = k
            self.run += 1
            self.gap = 0
        elif self.run:
            self.gap += 1
            if self.gap > self.gap_frames:
                self.run, self.gap, self.onset = 0, 0, None
        if self.run >= self.min_frames:
            return {"onset": self.onset, "frames": self.run, "now": user, "mic_db": round(m, 1), "agent_db": round(a, 1),
                    "coupling_db": round(self.coupling, 1)}
        return None

    def words(self, pieces, agent_text: str = "") -> None:
        """ASR pieces: a user word (not an echo of the agent's text) pauses the echo-level learning."""
        echo = agent_text.lower()
        for p in pieces:
            w = p["p"].replace("\u2581", "").strip(".,;:!?").lower()
            if any(ch.isalnum() for ch in w) and not (len(w) > 2 and w in echo):
                self.last_word_k = max(self.last_word_k, int(p["f"]))

    def decide(self, cand: dict | None, pieces_since_onset: list, agent_text: str = "") -> str | None:
        """-> "asr_word" | "long_speech" | None. Fires once per audible stretch. ``agent_text``: what the agent is
        saying; a word of it heard in the mic is its echo, not the user."""
        if cand is None or self.fired:
            return None
        text = "".join(p["p"] for p in pieces_since_onset).replace("\u2581", " ").lower()
        words = [w.strip(".,;:!?") for w in text.split()]
        echo = {w.strip(".,;:!?\"'") for w in agent_text.lower().split()}
        real = [w for w in words if len(w) > 1 and any(c.isalpha() for c in w) and w not in BACKCHANNEL_WORDS
                and w not in echo]
        if real:
            self.fired = True
            return "asr_word"
        if cand["frames"] >= self.force_frames and cand.get("now") and len(self.coup) >= self.learned_n:
            self.fired = True
            return "long_speech"
        return None
