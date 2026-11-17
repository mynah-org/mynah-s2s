"""G3b arm B: a reactive user. The same four Italian utterances as the fixed track, but each one starts only after
the agent's audio for the previous turn has ended (+ ``pad_s``); the last one is a deliberate barge-in that starts
``barge_after_s`` after the agent's real reply (not the holding phrase) to the previous utterance became audible, so
the audio-stop latency of a cancel on audible audio can be measured.

The scripted track of E1/G0/G3 has fixed gaps: with a talker slower than real time and a turn that stays open until
its audio ends, the caller's next sentence lands on the agent's audio and is (correctly) a barge-in. This source
removes those scripted overlaps.

Chunk source for the paced feeder: ``next_chunk(now, world)`` returns the next ``chunk_samples`` of user audio
(speech or noise-floor silence). ``world`` answers: ``audible(now)``, ``agent_idle()`` (controller listening, talker
idle, nothing audible), ``audio_starts_after(t, real_only)`` -> first agent playback start after wall time t.
Model-free; tests in tests/test_reactive_user.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Utt:
    label: str
    clip: object                    # 1-D float array / tensor at 16 kHz
    barge: bool = False


@dataclass
class ReactiveUser:
    utts: list
    chunk_samples: int = 2560
    sr: int = 16000
    pad_s: float = 0.5
    wait_cap_s: float = 25.0
    barge_after_s: float = 1.0
    tail_cap_s: float = 20.0
    noise: object = None            # callable(n) -> silence chunk with the noise floor
    cat: object = None              # callable([pieces]) -> one chunk
    i: int = 0                      # next utterance
    pos: int = -1                   # sample position inside the current utterance (-1 = waiting)
    t_ref: float | None = None      # wall time the wait started (end of the previous utterance / session start)
    t_ready: float | None = None    # wall time the condition was met (then pad_s)
    sent: int = 0                   # samples sent so far
    timeline: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    done: bool = False

    def _cond(self, now, world) -> str | None:
        u = self.utts[self.i] if self.i < len(self.utts) else None
        if u is not None and u.barge:
            t = world.audio_starts_after(self.t_ref, True)
            if t is not None and now - t >= self.barge_after_s and world.audible(now):
                return "barge_over_audio"
            return "timeout" if now - self.t_ref > self.wait_cap_s else None
        started = world.audio_starts_after(self.t_ref, False) is not None
        if started and world.agent_idle(now):
            return "agent_audio_ended"
        cap = self.tail_cap_s if u is None else self.wait_cap_s
        return "timeout" if now - self.t_ref > cap else None

    def next_chunk(self, now: float, world):
        n = self.chunk_samples
        if self.t_ref is None:
            self.t_ref = now
        out = []
        while n > 0:
            if self.pos < 0:                                   # waiting
                if self.t_ready is None:
                    why = self._cond(now, world)
                    if why is not None:
                        self.t_ready = now
                        self.waits.append({"before": self.utts[self.i].label if self.i < len(self.utts) else "end",
                                           "why": why, "wait_s": round(now - self.t_ref, 2)})
                pad = 0.0 if (self.i < len(self.utts) and self.utts[self.i].barge) else self.pad_s
                if self.t_ready is not None and now - self.t_ready >= pad:
                    if self.i >= len(self.utts):
                        self.done = True
                        out.append(self.noise(n))
                        self.sent += n
                        break
                    self.pos = 0
                    self.timeline.append({"label": self.utts[self.i].label, "start_sample": self.sent,
                                          "start_k": self.sent // 1280, "barge": self.utts[self.i].barge})
                    continue
                out.append(self.noise(n))
                self.sent += n
                n = 0
            else:
                clip = self.utts[self.i].clip
                take = min(n, len(clip) - self.pos)
                out.append(clip[self.pos:self.pos + take])
                self.pos += take
                self.sent += take
                n -= take
                if self.pos >= len(clip):
                    tl = self.timeline[-1]
                    tl["end_sample"] = self.sent
                    tl["end_k"] = (self.sent - 1) // 1280
                    self.i += 1
                    self.pos = -1
                    self.t_ref, self.t_ready = now, None
        return self.cat(out)
