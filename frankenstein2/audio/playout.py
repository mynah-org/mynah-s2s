"""Playout: the engine releases agent audio at real-time pace, so it always knows what was played.

The talker may generate far ahead of real time (Pocket on a GPU: ~5x). Audio is queued here, per epoch, and
``release()`` hands exactly one input step's worth (80 ms) to the transport each time the session clock advances.
Hence:
  * a cancel (barge-in) only has to flush THIS queue: nothing is buffered ahead in the transport / client beyond
    the client's own playback buffer (``client_delay_steps``);
  * "played up to sample N" is exact on the session clock (up to that fixed client delay), which feeds
    - the 2nd ear: the agent channel the ears hear = agent audio as released, delayed by the client delay,
      resampled to 16 kHz, sample-aligned with the user mic; backchannel audio is mixed into the output but kept
      OUT of the 2nd ear (and out of the history);
    - the history after a barge-in: the text of what was heard (segments fully played + the heard part of the
      segment being played, cut at a word boundary in proportion to its played samples).
Two lanes: "agent" (replies, holding phrases) and "bc" (backchannels, mixed on top). Model-free; unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OUT_SR = 24000
STEP_S = 0.08


class Resampler:
    """Stateful mono resampler: soxr when available, else linear interpolation on a running time base (tests)."""

    def __init__(self, sr_in: int, sr_out: int, use_soxr: bool = True):
        self.sr_in, self.sr_out = sr_in, sr_out
        self.s = None
        if sr_in != sr_out and use_soxr:
            try:
                import soxr
                self.s = soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality="HQ")
            except Exception:  # noqa: BLE001
                self.s = None
        self.t_in = 0
        self.t_out = 0
        self.tail = np.zeros(0, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.sr_in == self.sr_out:
            return x
        if self.s is not None:
            return self.s.resample_chunk(x)
        buf = np.concatenate([self.tail, x])
        base = self.t_in - len(self.tail)
        self.t_in += len(x)
        n_out = int(self.t_in * self.sr_out // self.sr_in) - self.t_out
        if n_out <= 0 or len(buf) == 0:
            self.tail = buf[-1:]
            return np.zeros(0, np.float32)
        pos = (np.arange(self.t_out, self.t_out + n_out) * self.sr_in / self.sr_out) - base
        y = np.interp(pos, np.arange(len(buf)), buf).astype(np.float32)
        self.t_out += n_out
        self.tail = buf[-1:]
        return y


@dataclass
class Seg:
    epoch: int
    turn: int
    seg: int
    lane: str
    text: str = ""
    queued: int = 0                 # samples received from the talker
    released: int = 0               # samples handed to the transport
    done: bool = False              # the talker finished this segment
    first_step: int | None = None
    holding: bool = False


@dataclass
class Playout:
    step_samples: int = int(OUT_SR * STEP_S)        # 1920 at 24 kHz
    client_delay_steps: int = 1
    bc_gain: float = 0.7
    epoch: int = 0
    segs: dict = field(default_factory=dict)        # (epoch, turn, seg) -> Seg
    q_agent: list = field(default_factory=list)     # [(key, pcm)] in order
    q_bc: list = field(default_factory=list)
    step: int = 0
    underruns: list = field(default_factory=list)
    flushes: list = field(default_factory=list)
    stale: int = 0
    released_log: list = field(default_factory=list)  # (step, key or None, n) per release (agent lane)
    _starving: bool = False

    # ------------------------------------------------------------------ input
    def register(self, epoch: int, turn: int, seg: int, lane: str, text: str, holding: bool = False) -> None:
        if epoch == self.epoch:
            self.segs[(epoch, turn, seg)] = Seg(epoch, turn, seg, lane, text, holding=holding)

    def enqueue(self, epoch: int, turn: int, seg: int, pcm) -> bool:
        key = (epoch, turn, seg)
        s = self.segs.get(key)
        if epoch != self.epoch or s is None:
            self.stale += 1
            return False
        pcm = np.asarray(pcm, np.float32)
        s.queued += len(pcm)
        (self.q_bc if s.lane == "bc" else self.q_agent).append((key, pcm))
        return True

    def seg_done(self, epoch: int, turn: int, seg: int) -> None:
        s = self.segs.get((epoch, turn, seg))
        if s is not None:
            s.done = True

    # ------------------------------------------------------------------ cancel
    def flush(self, new_epoch: int) -> int:
        """Drop everything not released yet; later audio of older epochs is stale. -> samples dropped."""
        n = sum(len(p) for _, p in self.q_agent) + sum(len(p) for _, p in self.q_bc)
        self.q_agent, self.q_bc = [], []
        self.flushes.append({"step": self.step, "dropped": n, "new_epoch": new_epoch})
        self.epoch = new_epoch
        self._starving = False
        return n

    # ------------------------------------------------------------------ output
    def _take(self, q: list, n: int):
        out = np.zeros(n, np.float32)
        got, parts = 0, []
        while q and got < n:
            key, pcm = q[0]
            k = min(n - got, len(pcm))
            out[got:got + k] = pcm[:k]
            parts.append((key, k))
            got += k
            if k == len(pcm):
                q.pop(0)
            else:
                q[0] = (key, pcm[k:])
        return out, got, parts

    def release(self):
        """One step of output. -> (mix 24 kHz, agent-only 24 kHz, info)."""
        n = self.step_samples
        a, got, parts = self._take(self.q_agent, n)
        b, got_b, parts_b = self._take(self.q_bc, n)
        for key, k in parts + parts_b:
            s = self.segs[key]
            if s.first_step is None:
                s.first_step = self.step
            s.released += k
            if s.lane == "agent":
                self.released_log.append((self.step, key, k))
        if got == 0:
            self.released_log.append((self.step, None, 0))
        open_turn = [s for s in self.segs.values() if s.epoch == self.epoch and s.lane == "agent" and
                     s.released > 0 and (not s.done or s.released < s.queued)]
        if got < n and open_turn and not self._starving:
            self.underruns.append({"step": self.step, "missing": n - got})
            self._starving = True
        elif got == n:
            self._starving = False
        mix = a + self.bc_gain * b
        self.step += 1
        return mix, a, {"agent": got, "bc": got_b}

    # ------------------------------------------------------------------ queries
    def pending(self, lane: str = "agent") -> int:
        return sum(len(p) for _, p in (self.q_agent if lane == "agent" else self.q_bc))

    def turn_audio_done(self, epoch: int, turn: int) -> bool:
        """Every registered agent segment of the turn generated and released (plus the client delay)."""
        segs = [s for s in self.segs.values() if s.epoch == epoch and s.turn == turn and s.lane == "agent"]
        return bool(segs) and all(s.done and s.released >= s.queued for s in segs) and \
            self.pending("agent") == 0 and self.last_agent_step() + self.client_delay_steps < self.step

    def last_agent_step(self) -> int:
        for st, key, k in reversed(self.released_log):
            if key is not None and k:
                return st
        return -10 ** 9

    def audible(self) -> bool:
        return self.pending("agent") > 0 or self.last_agent_step() + self.client_delay_steps >= self.step - 1

    def turn_first_step(self, epoch: int, turn: int, holding=None):
        xs = [s.first_step for s in self.segs.values() if s.epoch == epoch and s.turn == turn and s.lane == "agent"
              and s.first_step is not None and (holding is None or s.holding == holding)]
        return min(xs) if xs else None

    def heard_text(self, epoch: int, turn: int, include_holding: bool = False) -> tuple:
        """What the user heard of an agent turn: (text, complete). Played = released minus the client delay."""
        played_total = max(0, sum(k for st, key, k in self.released_log
                                  if key is not None and key[0] == epoch and key[1] == turn
                                  and st < self.step - self.client_delay_steps))
        out, complete = [], True
        for key in sorted(k for k in self.segs if k[0] == epoch and k[1] == turn):
            s = self.segs[key]
            if s.lane != "agent" or (s.holding and not include_holding):
                if s.lane == "agent" and s.holding:
                    played_total -= min(played_total, s.released)
                continue
            heard = min(played_total, s.released)
            played_total -= heard
            if heard <= 0:
                complete = False
                break
            total = s.queued if s.done else max(s.queued, 1)
            if heard >= total and s.done:
                out.append(s.text)
                continue
            words = s.text.split()
            keep = int(len(words) * heard / max(total, 1))
            if keep:
                out.append(" ".join(words[:keep]))
            complete = False
            break
        return " ".join(x for x in out if x).strip(), complete


class SecondEar:
    """The agent channel the ears hear: agent-lane audio released at step s is heard during input step
    s + client_delay_steps, resampled 24 -> 16 kHz (1280 samples per step)."""

    def __init__(self, client_delay_steps: int = 1, sr_in: int = OUT_SR, step16: int = 1280):
        self.delay = client_delay_steps
        self.rs = Resampler(sr_in, 16000, use_soxr=False)   # zero-latency (alignment with the mic matters more
        #                                                     than band-limiting for the GRU's agent channel)
        self.step16 = step16
        self.fifo = [np.zeros(step16, np.float32) for _ in range(client_delay_steps)]
        self.buf = np.zeros(0, np.float32)

    def push(self, agent24: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, self.rs(agent24)])
        if len(self.buf) >= self.step16:
            x, self.buf = self.buf[:self.step16], self.buf[self.step16:]
        else:
            x = np.concatenate([self.buf, np.zeros(self.step16 - len(self.buf), np.float32)])
            self.buf = np.zeros(0, np.float32)
        self.fifo.append(x)

    def take(self) -> np.ndarray:
        """The agent channel for the current input step."""
        return self.fifo.pop(0) if self.fifo else np.zeros(self.step16, np.float32)
