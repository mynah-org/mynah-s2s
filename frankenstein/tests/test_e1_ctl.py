"""E1j: the controller-driven session hook (e1/ctl_driver.py) and the model-free worker base (QueueWorker)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import FakeVocab  # noqa: E402
from e1.ctl_driver import CtlDriver  # noqa: E402
from e1.nano_worker import QueueWorker  # noqa: E402
from frankenstein_s2s import tokens as T  # noqa: E402
from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter  # noqa: E402
from frankenstein_s2s.controller import Controller, ControllerConfig  # noqa: E402
from frankenstein_s2s.driver import FakeTalkerDriver, PlaybackGate  # noqa: E402
from frankenstein_s2s.tracing import Tracer  # noqa: E402
from test_e1_hook import FakeWrapper, G, Log, Tok  # noqa: E402


class FakeWorker:
    """rid-aware: after `delay` polls a request yields one id per poll; cancel() stops it, but one id that was
    already 'in flight' is still delivered (to exercise the stale path)."""

    def __init__(self, replies, delay=2):
        self.replies, self.delay = dict(replies), delay
        self.jobs, self.cancelled, self.submitted = [], set(), []

    def submit(self, rid, messages):
        self.submitted.append((rid, messages[-1]["content"] if messages else ""))
        self.jobs.append({"rid": rid, "ids": list(self.replies.get(rid, [])), "wait": self.delay, "leak": 1})

    def cancel(self, rid):
        self.cancelled.add(rid)

    def poll(self):
        out = []
        for j in self.jobs:
            if j["wait"] > 0:
                j["wait"] -= 1
                continue
            if j["rid"] in self.cancelled:
                if j["leak"] and j["ids"]:
                    out.append(("tok", j["rid"], j["ids"].pop(0), 0))
                    j["leak"] = 0
                continue
            if j["ids"]:
                out.append(("tok", j["rid"], j["ids"].pop(0), 0))
                if not j["ids"]:
                    out.append(("done", j["rid"], "x", [], {}))
        return out


def rig(asr, model, replies, cfg=None, delay=2):
    v = FakeVocab()
    ids = {rid: [v.id(p) for p in r] for rid, r in replies.items()}
    w = FakeWrapper()
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode,
                       config=AdapterConfig(barge_in="thinker"), tracer=tr)
    wk = FakeWorker(ids, delay)
    ctl = Controller(cfg or ControllerConfig(reply_timeout=100), tracer=tr)
    d = CtlDriver(w, wk, Tok(v), asr, ad, tr, Log(), Log(), ctl)
    un = d.install()
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    return d, wk, ad, tr, ids, g


# the e1i U1 sequence (frames shifted): caller words close at 140 ("cosa"), pieces "▁a" "ve" (148-151) are an OPEN
# word when the Thinker wants the turn at 154; "te" arrives at 155 and the word closes at 158 (close_after 3)
U1 = {"words": [{"k": 140, "word": "Cerco un divano a tre posti, cosa"}, {"k": 158, "word": "avete"}],
      "pieces": {"148": ["▁a"], "151": ["ve"], "155": ["te"]},
      "nonblank": [146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157], "eou": [167]}


def test_split_user_turn_regression_e1i_u1():
    model = [T.PAD_ID] * 154 + [T.BOS_ID] + [T.PAD_ID] * 30
    d, wk, ad, tr, ids, g = rig(U1, model, {1: [" Ok", "."]})
    acts = [(a.k, a.kind, a.reason) for a in d.ctl.actions]
    assert acts[0][:2] == (154, "DEFER") and acts[0][2] in ("asr_speech_recent", "asr_word_open")
    opened = [a for a in acts if a[1] == "OPEN_TURN"]
    assert len(opened) == 1 and opened[0][0] >= 158 and opened[0][2] == "thinker_bos_deferred"
    assert len(wk.submitted) == 1 and wk.submitted[0][1].endswith("avete")      # ONE request, the whole turn


def test_fragment_merged_into_a_new_request_and_old_tokens_stale():
    asr = {"words": [{"k": 10, "word": "vorrei"}, {"k": 30, "word": "sapere"}],
           "pieces": {"8": ["▁vorrei"], "29": ["▁sapere"]}, "nonblank": [8, 9, 29], "eou": []}
    model = [T.PAD_ID] * 20 + [T.BOS_ID] + [T.PAD_ID] * 40
    d, wk, ad, tr, ids, g = rig(asr, model, {1: [" Uno", " due", " tre", "."], 2: [" Si", "."]}, delay=12,
                                cfg=ControllerConfig(reply_timeout=100, barge_in_frames=4))
    reqs = [a for a in d.ctl.actions if a.kind == "REQUEST_REPLY"]
    assert [a.data.get("supersedes") for a in reqs] == [None, 1] and 1 in wk.cancelled
    assert wk.submitted[1][1] == "vorrei sapere"
    assert any(a.kind == "DROP_STALE" and a.data.get("request_id") == 1 for a in d.ctl.actions)
    spoken = "".join(c.text for c in ad.commands if hasattr(c, "text"))
    assert spoken == "Si." and d.stale >= 1
    assert tr.of("ctl.intent") and tr.of("ctl.action")


def test_controller_cancel_reaches_adapter_and_worker():
    asr = {"words": [{"k": 4, "word": "ciao"}, {"k": 40, "word": "aspetta"}],
           "pieces": {"3": ["▁ciao"], "18": ["▁asp"], "19": ["et"], "20": ["ta"]},
           "nonblank": [], "eou": [14]}
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [T.PAD_ID] * 40
    d, wk, ad, tr, ids, g = rig(asr, model, {1: [" a", " b", " c", " d", " e", " f", " g", " h", " i", " l", "."]},
                                cfg=ControllerConfig(reply_timeout=100, barge_guard=6))
    kinds = [a.kind for a in d.ctl.actions]
    assert "CANCEL" in kinds and 1 in wk.cancelled
    assert any(type(c).__name__ == "Cancel" for c in ad.commands)


class EchoWorker(QueueWorker):
    def __init__(self):
        super().__init__("echo")

    def _serve(self, rid, msgs, t_req):
        for i in range(5):
            if rid in self.cancelled:
                break
            self.q_out.put(("tok", rid, 1000 + i, time.monotonic_ns()))
            time.sleep(0.01)
        self.q_out.put(("done", rid, "", [], {}))


def test_queue_worker_preemption_and_close_join():
    wk = EchoWorker()
    wk.cancel(1)
    wk.start()
    wk.submit(1, [])                      # superseded before it started: never served
    wk.submit(2, [])
    time.sleep(0.2)
    ev = wk.poll()
    assert ("skipped", 1) in ev and [e[2] for e in ev if e[0] == "tok"] == [1000, 1001, 1002, 1003, 1004]
    assert wk.served == [2]
    wk.close(timeout=2)
    assert not wk.is_alive()


def test_open_word_alone_defers_the_take_turn():
    # ASR blank long enough, but a word is still open (pieces, no close yet): the take-turn waits for the close
    asr = {"words": [{"k": 4, "word": "ciao"}, {"k": 20, "word": "grazie"}],
           "pieces": {"3": ["▁ciao"], "10": ["▁gra"]}, "nonblank": [3, 10], "eou": []}
    model = [T.PAD_ID] * 15 + [T.BOS_ID] + [T.PAD_ID] * 15
    d, *_ = rig(asr, model, {1: [" Si", "."]})
    acts = [(a.k, a.kind, a.reason) for a in d.ctl.actions]
    assert (15, "DEFER", "asr_word_open") in acts and any(a[1] == "OPEN_TURN" and a[0] >= 20 for a in acts)


# e1j U4 (frames as recorded): caller words until 602, the reply opened at 607 (Thinker BOS), then a late
# punctuation + language-tag emission at 611 made a 3-frame "speech" run and a FALSE barge-in at 613.
U4 = {"words": [{"k": 597, "word": "numero"}, {"k": 605, "word": "d'ordine"}],
      "pieces": {"594": ["▁nu"], "595": ["me", "ro"], "597": ["▁", "d", "'"], "599": ["ord"], "601": ["in"],
                 "602": ["e"], "611": [".", "▁", "<it-IT>"]},
      "nonblank": [594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 611, 612, 613], "eou": [623]}


def test_e1j_u4_late_punctuation_is_not_a_barge_in():
    model = [T.PAD_ID] * 607 + [T.BOS_ID] + [T.PAD_ID] * 30
    reply = [" Mi", " serve", " il", " nome", " e", " il", " CAP", ".", " Grazie", "."]
    d, wk, ad, tr, ids, g = rig(U4, model, {1: reply},
                                cfg=ControllerConfig(reply_timeout=100, barge_after_open=True, eou_fallback=False))
    assert "CANCEL" not in [a.kind for a in d.ctl.actions]
    assert "".join(c.text for c in ad.commands if hasattr(c, "text")) == "Mi serve il nome e il CAP. Grazie."


def test_true_barge_in_after_open_still_cancels():
    asr = {"words": [{"k": 4, "word": "ciao"}, {"k": 22, "word": "aspetta"}],
           "pieces": {"3": ["▁ciao"], "18": ["▁asp"], "19": ["et"], "20": ["ta"]},
           "nonblank": [], "eou": [14]}
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [T.PAD_ID] * 30
    d, wk, ad, tr, ids, g = rig(asr, model, {1: [" a", " b", " c", " d", " e", " f", " g", " h", " i", " l", "."]},
                                cfg=ControllerConfig(reply_timeout=100, barge_after_open=True))
    c = [a for a in d.ctl.actions if a.kind == "CANCEL"]
    assert c and c[0].k == 20 and c[0].reason == "barge_in"          # 18, 19, 20: three lexical frames


class ToolWorker(FakeWorker):
    """Says 'decided TOOL' after `delay` polls, then waits `call` polls (the call step) before streaming the reply."""

    def __init__(self, replies, delay=2, call=6):
        super().__init__(replies, delay)
        self.call = call

    def submit(self, rid, messages):
        super().submit(rid, messages)
        self.jobs[-1].update(decided=False, call=self.call)

    def poll(self):
        out = []
        for j in self.jobs:
            if j["wait"] == 0 and not j["decided"] and j["rid"] not in self.cancelled:
                j["decided"] = True
                out.append(("decided", j["rid"], "TOOL"))
                continue
            if j["decided"] and j["call"] > 0:
                j["call"] -= 1
                continue
            out += FakeWorker.poll(_One(self, j))
        return out


class _One:
    def __init__(self, w, j):
        self.jobs, self.cancelled = [j], w.cancelled


def tool_rig(cfg, n_model=60, call=6, holding_cut="audible"):
    asr = {"words": [{"k": 4, "word": "dov'è il mio ordine"}], "pieces": {"3": ["▁dov"]}, "nonblank": [],
           "eou": [8]}
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [T.PAD_ID] * n_model
    v = FakeVocab()
    reply = [v.id(p) for p in [" Il", " suo", " ordine", " arriva", " domani", "."]]
    w, tr = FakeWrapper(), Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode,
                       config=AdapterConfig(barge_in="thinker"), tracer=tr)
    wk = ToolWorker({1: reply}, delay=2, call=call)
    ctl = Controller(cfg, tracer=tr)
    d = CtlDriver(w, wk, Tok(v), asr, ad, tr, Log(), Log(), ctl, lang="it")
    d.holding_cut = holding_cut
    un = d.install()
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    return d, ad


def _segments(ad):
    return [(type(c).__name__, c.epoch, getattr(c, "turn", None), getattr(c, "text", "")) for c in ad.commands]


def test_holding_phrase_once_then_cut_by_the_reply_in_a_new_epoch():
    d, ad = tool_rig(ControllerConfig(reply_timeout=100, holding_phrase=True))
    kinds = [a.kind for a in d.ctl.actions]
    assert kinds.count("HOLDING_PHRASE") == 1
    h = d.holding[0]
    assert h["ok"] and h["text"] == "Un momento, controllo." and h["cut_k"] is not None
    seg = _segments(ad)
    starts = [s for s in seg if s[0] == "StartSegment"]
    assert starts[0][3] == "Un momento, controllo." and starts[0][1] == 0
    cancel = [s for s in seg if s[0] == "Cancel"]
    assert len(cancel) == 1                                         # the cut
    assert starts[1][1] == 1 and starts[1][2] != starts[0][2]      # the reply: new epoch, new adapter turn
    reply = "".join(s[3] for s in seg if s[0] in ("StartSegment", "AppendText") and s[1] == 1)
    assert reply == "Il suo ordine arriva domani."


def test_holding_phrase_off_by_default_and_reply_unchanged():
    d, ad = tool_rig(ControllerConfig(reply_timeout=100))
    assert "HOLDING_PHRASE" not in [a.kind for a in d.ctl.actions] and d.holding == []
    assert "".join(c.text for c in ad.commands if hasattr(c, "text")) == "Il suo ordine arriva domani."
    assert not any(type(c).__name__ == "Cancel" for c in ad.commands)


def test_holding_phrase_not_repeated_and_not_after_speech():
    from frankenstein_s2s.controller import Intent
    c = Controller(ControllerConfig(holding_phrase=True))
    c.post(Intent("WORDS", "asr", 1))
    c.tick(1)
    c.post(Intent("TAKE_TURN", "thinker", 4))
    c.tick(4)
    for k in (5, 6):
        c.post(Intent("TOOL_DECIDED", "nano", k, request_id=1, epoch=0))
        c.tick(k)
    assert [a.kind for a in c.actions].count("HOLDING_PHRASE") == 1
    c2 = Controller(ControllerConfig(holding_phrase=True))
    c2.post(Intent("WORDS", "asr", 1))
    c2.tick(1)
    c2.post(Intent("TAKE_TURN", "thinker", 4))
    c2.tick(4)
    c2.post(Intent("REPLY_TOKENS", "nano", 5, payload=[1001], request_id=1, epoch=0))
    c2.tick(5)
    c2.post(Intent("TOOL_DECIDED", "nano", 6, request_id=1, epoch=0))
    c2.post(Intent("TOOL_DECIDED", "nano", 7, request_id=0, epoch=0))   # stale request
    c2.tick(7)
    assert "HOLDING_PHRASE" not in [a.kind for a in c2.actions]


def test_holding_phrase_never_cut_reply_queues_behind_it():
    d, ad = tool_rig(ControllerConfig(reply_timeout=100, holding_phrase=True), holding_cut="never")
    seg = _segments(ad)
    assert not any(x[0] == "Cancel" for x in seg) and d.holding[0]["cut_k"] is None
    starts = [x for x in seg if x[0] == "StartSegment"]
    assert starts[0][3] == "Un momento, controllo." and starts[1][2] != starts[0][2] and starts[1][1] == 0
    assert "".join(x[3] for x in seg if x[0] in ("StartSegment", "AppendText"))[len("Un momento, controllo."):] \
        == "Il suo ordine arriva domani."
