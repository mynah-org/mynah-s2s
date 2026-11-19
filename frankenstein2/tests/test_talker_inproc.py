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


def test_ears_on_their_own_thread_open_hop_and_transcribe():
    """ThreadedEars drives InProcEars in place of the scheduler: opens, hops and a transcription all come back."""
    import base64
    import numpy as np
    from frankenstein2.runtime import InProcEars, ThreadedEars

    class E:
        def __init__(self):
            self.sess = {}

        def open(self, sid, lang, thr):
            self.sess[sid] = lang

        def close(self, sid):
            self.sess.pop(sid, None)

        def transcribe(self, x, lang):
            return f"{lang}:{len(x)}"

        def hop(self, items):
            return [{"frames": 1, "events": [], "pieces": []} for _ in items]

    class S:
        def mem(self):
            return {}

    t = ThreadedEars(S())
    ears = InProcEars(E(), t)
    t.ears = ears
    t.start()
    try:
        ears.send({"op": "open", "sid": "s1", "lang": "it"})
        z = base64.b64encode(np.zeros(5120, "<f4").tobytes()).decode()
        deadline = time.monotonic() + 5
        while "s1" not in ears.e.sess and time.monotonic() < deadline:
            time.sleep(0.01)
        ears.send({"op": "hop", "sid": "s1", "n": 0, "mode": "listen", "u": z, "a": z})
        got = []
        while not got and time.monotonic() < deadline:
            got = [m for m in ears.poll("s1") if m.get("ev") == "hop"]
            time.sleep(0.01)
        assert got and got[0]["n"] == 0
        x = base64.b64encode(np.zeros(160, "<f4").tobytes()).decode()
        assert ears.request({"op": "transcribe", "id": "a", "lang": "fr", "pcm": x}, "transcript", 5)["text"] == "fr:160"
        assert t.err is None and t.ms
    finally:
        t.close()
