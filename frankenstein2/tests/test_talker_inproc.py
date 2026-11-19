"""The one-process talker with its language cache and the real client (PocketTalker), on fake models (torch CPU):
a cold language loads while another session keeps receiving audio, and the client protocol works end to end."""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch = pytest.importorskip("torch")

from frankenstein2.runtime import InProcTalker, _Lang  # noqa: E402
from frankenstein2.tts.lang_cache import TalkerLangs  # noqa: E402
from frankenstein2.tts.procs import PocketTalker  # noqa: E402


class FakeModel:
    def chunks(self, text):
        return [(text, 1)]

    def voice_from_file(self, path):
        return "voice"


class FakeBatch:
    def __init__(self):
        self.rows = {}

    def start(self, key, voice, text, after):
        self.rows[key] = 3                       # three 80 ms frames per segment

    def stop(self, key):
        self.rows.pop(key, None)

    def step(self, keys):
        out = {}
        for k in keys:
            if self.rows[k] == 0:
                out[k] = None
            else:
                self.rows[k] -= 1
                out[k] = torch.full((1920,), 0.1)
        return out


class Sched:
    def __init__(self):
        self.talker = None

    def submit(self, item):
        self.talker.control(item[1])            # synchronous: the test drives run() itself

    def kick(self):
        pass

    def mem(self):
        return {}


def make(gates):
    def fetch(name):
        if name in gates:
            gates[name].wait(5)
        return name

    def build(name, h):
        return _Lang(FakeModel(), FakeBatch(), {"": "voice"})
    sched = Sched()
    t = InProcTalker(sched, TalkerLangs(fetch, build, vram_langs=2, host_langs=2), names=lambda lang, ss: lang)
    sched.talker = t
    return t


def test_a_cold_language_loads_while_another_session_keeps_talking():
    gate = threading.Event()
    t = make({"fr": gate})
    t.preload("it")
    it = PocketTalker(t, "s_it", "it")                      # the real client: open waits for the "voice" reply
    fr = PocketTalker(t, "s_fr", "fr")                      # returns at once although fr is not loaded
    assert fr.info["status"] == "loading" and it.info["status"] == "ready"
    fr.say("e0t1s0", "Bonjour.", 0)
    got = {"s_it": 0, "s_fr": 0}
    for i in range(6):                                      # it keeps producing audio, fr waits for its load
        it.say(f"e0t1s{i}", "Ciao.", 0)
        for _ in range(5):
            t.run(40.0)
            t._ahead = staticmethod(lambda c, now: -1.0)    # never "ahead": step every turn
        for ss in got:
            got[ss] += sum(1 for m in t.poll(ss) if m["ev"] == "audio")
    assert got["s_it"] > 0 and got["s_fr"] == 0
    gate.set()
    t_end = time.monotonic() + 3
    while got["s_fr"] == 0 and time.monotonic() < t_end:
        t.run(40.0)
        got["s_fr"] += sum(1 for m in t.poll("s_fr") if m["ev"] == "audio")
        time.sleep(0.005)
    assert got["s_fr"] > 0
    done = [m for m in t.poll("s_fr") if m["ev"] == "done"]
    assert t.waits and max(t.waits) > 0                     # the fr text waited for its language
    _ = done
