"""G3b: the reactive user source (model-free)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.reactive_user import ReactiveUser, Utt  # noqa: E402


class World:
    """Agent audio: after each user utterance ends, the agent plays [end + 1 s, end + 3 s] (holding first 0.5 s)."""

    def __init__(self):
        self.plays = []            # (start, end, real)

    def audible(self, now):
        return any(a <= now < b for a, b, _ in self.plays)

    def agent_idle(self, now):
        return not self.audible(now) and all(b <= now for _, b, _ in self.plays)

    def audio_starts_after(self, t, real_only):
        xs = [a for a, _, r in self.plays if a > t and (r or not real_only)]
        return min(xs) if xs else None


def drive(utts, steps=400, dt=0.16):
    w = World()
    ru = ReactiveUser(utts, chunk_samples=2560, noise=lambda n: [0.0] * n, cat=lambda xs: sum(map(list, xs), []))
    ends = 0
    w.plays.append((0.5, 2.0, True))                   # the greeting
    for i in range(steps):
        now = i * dt
        c = ru.next_chunk(now, w)
        assert len(c) == 2560
        if len(ru.timeline) > ends and "end_sample" in ru.timeline[-1]:
            ends += 1
            w.plays += [(now + 1.0, now + 1.5, False), (now + 1.5, now + 3.0, True)]
        if ru.done:
            break
    return ru, w


def test_each_utterance_waits_for_the_agent_audio_end_plus_pad():
    utts = [Utt(f"U{i}", [1.0] * 8000) for i in range(1, 4)]
    ru, w = drive(utts)
    assert [t["label"] for t in ru.timeline] == ["U1", "U2", "U3"] and ru.done
    assert all(x["why"] == "agent_audio_ended" for x in ru.waits)
    # U1 starts after the greeting ended (2.0 s) + 0.5 s
    assert ru.timeline[0]["start_sample"] / 16000 >= 2.5 - 0.16


def test_last_utterance_barges_in_over_the_real_reply():
    utts = [Utt("U1", [1.0] * 8000), Utt("U4", [1.0] * 8000, barge=True)]
    ru, w = drive(utts)
    b = [x for x in ru.waits if x["before"] == "U4"][0]
    assert b["why"] == "barge_over_audio"
    t_u4 = ru.timeline[1]["start_sample"] / 16000
    real_start = [a for a, _, r in w.plays if r][1]
    assert real_start + 1.0 - 0.2 <= t_u4 <= real_start + 1.5
