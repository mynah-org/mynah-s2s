"""CPU dry run of the E1 M1-force hook against a fake wrapper that mimics the NVIDIA per-frame order:
model prediction in gen_text[0, t] -> _apply_rnnt_turn_taking (reads the written id, may force EOS on barge-in)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import FakeVocab  # noqa: E402
from e1.m1force import M1Force  # noqa: E402
from frankenstein_s2s import tokens as T  # noqa: E402
from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter  # noqa: E402
from frankenstein_s2s.driver import FakeTalkerDriver, PlaybackGate  # noqa: E402
from frankenstein_s2s.tracing import Tracer  # noqa: E402


class G:
    """2-D int grid with tensor-style [b, t] indexing."""

    def __init__(self, n):
        self.d = [[T.PAD_ID] * n]

    def __getitem__(self, ix):
        return self.d[ix[0]][ix[1]]

    def __setitem__(self, ix, v):
        self.d[ix[0]][ix[1]] = v


class FakeWrapper:
    def __init__(self, barge_eos_at=None):
        self.barge_eos_at = barge_eos_at
        self.seen = []
        self.agent_speaking = False

    def _apply_rnnt_turn_taking(self, t, gen_text, is_blank, state):
        cur = gen_text[0, t]
        self.seen.append(cur)                         # the original logic sees the WRITTEN id
        if cur == T.BOS_ID:
            self.agent_speaking = True
        if self.barge_eos_at is not None and t == self.barge_eos_at and self.agent_speaking:
            gen_text[0, t] = T.EOS_ID
            self.agent_speaking = False

    def _apply_fc_state_machine(self, **kw):
        return None


class Tok:
    def __init__(self, v):
        self.v = v

    def decode(self, ids):
        return "".join(self.v.id_to_piece.get(i, " ?") for i in ids)


class FakeBrain:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.last_cache = object()

    def run(self, messages, decide_tool=True):
        self.calls.append([dict(m) for m in messages])
        ids = self.replies.pop(0)
        return {"ids": ids, "text": "x", "prefill_ms": 10.0, "ctx_len": 5, "tool_margin": -1.0, "func_top1": 12,
                "decode_ms_per_tok": 10.0, "tool": None}


class Log(list):
    def write(self, row):
        self.append(row)

    def close(self):
        pass


def rig(model_stream, asr, replies, barge_eos_at=None):
    v = FakeVocab()
    reply_ids = [[v.id(p) for p in r] for r in replies]
    tok = Tok(v)
    w = FakeWrapper(barge_eos_at)
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode, config=AdapterConfig(), tracer=tr)
    hook = M1Force(w, FakeBrain(reply_ids), tok, asr, ad, tr, Log(), Log(), lambda p: ("{}", 0.0))
    M1Force.brain_fp = staticmethod(lambda cache: "fp")
    un = hook.install()
    g = G(len(model_stream))
    for t, p in enumerate(model_stream):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    return hook, w, g, ad, v, tr


ASR = {"words": [{"k": 3, "word": "ciao"}, {"k": 5, "word": "vorrei"}], "nonblank": [1, 2, 3, 4, 5], "eou": [9]}


def test_forced_reply_goes_through_original_logic_and_is_emitted():
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [1500] * 6 + [T.PAD_ID] * 3 + [T.EOS_ID] + [T.PAD_ID] * 3
    hook, w, g, ad, v, tr = rig(model, ASR, [[" Certo", ",", " subito", "."]])
    emitted = g.d[0]
    reply = [v.id(p) for p in [" Certo", ",", " subito", "."]]
    # the original turn-taking saw exactly what was emitted at every frame (forcing happens before it)
    assert w.seen == emitted
    # reply ids appear once each, in order, after BOS; the model's own text never appears
    assert [x for x in emitted if x >= 1000] == reply and 1500 not in emitted
    # the brain got the causal user turn: only words closed before the BOS frame
    assert hook.brain.calls[0][-1] == {"role": "user", "content": "ciao vorrei"}
    assert ad.commands and "".join(c.text for c in ad.commands if hasattr(c, "text")) == "Certo, subito."
    assert all(c["same"] for c in hook.cache_checks)
    assert hook.messages[-1]["role"] == "assistant"


def test_reply_not_available_before_its_measured_time():
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [1500] * 8 + [T.EOS_ID]
    hook, w, g, *_ = rig(model, ASR, [[" Si", "."]])
    first = next(t for t, x in enumerate(g.d[0]) if x >= 1000)
    # prefill 10 ms + tokens at 10 ms: available at t = 10 + ceil(ms / 80) = 11 at the earliest, never at the BOS frame
    assert first >= 11


def test_duplex_barge_in_eos_cuts_forcing():
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [1500] * 10 + [T.PAD_ID] * 3
    hook, w, g, ad, v, tr = rig(model, ASR, [[" Uno", " due", " tre", " quattro", " cinque", " sei", "."]],
                                barge_eos_at=13)
    em = g.d[0]
    assert em[13] == T.EOS_ID
    assert all(x < 1000 for x in em[14:])                  # nothing forced after the cut
    assert any(r.get("source") == "duplex_rnnt" for r in tr.of("adapter.barge_in"))
    assert hook.messages[-1]["role"] == "assistant" and hook.messages[-1]["content"].strip() != ""


def test_observer_mode_never_writes():
    model = [T.PAD_ID] * 3 + [T.BOS_ID, 1500, 1501, T.PAD_ID, T.PAD_ID, T.EOS_ID]
    v = FakeVocab()
    w = FakeWrapper()
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=lambda ids: "w " * len(ids), tracer=tr)
    hook = M1Force(w, FakeBrain([]), Tok(v), {"words": [], "nonblank": [], "eou": []}, ad, tr, Log(), Log(),
                   lambda p: ("{}", 0.0), mode="observe")
    un = hook.install()
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    assert g.d[0] == model and not hook.mismatch
    assert "_apply_rnnt_turn_taking" not in w.__dict__


class SlowBrain(FakeBrain):
    def run(self, messages, decide_tool=True):
        r = super().run(messages, decide_tool)
        r["decode_ms_per_tok"] = 400.0                         # 5 frames per token: most ids still unreleased
        return r


def test_unreleased_ids_of_a_cut_turn_never_reach_the_next_turn():
    v = FakeVocab()
    first = [v.id(p) for p in [" uno", " due", " tre", " quattro", " cinque"]]
    second = [v.id(p) for p in [" sei", "."]]
    w = FakeWrapper(barge_eos_at=14)
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode, config=AdapterConfig(), tracer=tr)
    hook = M1Force(w, SlowBrain([first, second]), Tok(v), ASR, ad, tr, Log(), Log(), lambda p: ("{}", 0.0))
    M1Force.brain_fp = staticmethod(lambda cache: "fp")
    un = hook.install()
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [T.PAD_ID] * 9 + [T.BOS_ID] + [T.PAD_ID] * 30
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    after_second_bos = [x for x in g.d[0][20:] if x >= 1000]
    assert not set(after_second_bos) & set(first)
    assert any(e.get("why") == "drop_unreleased_reply_ids" for e in tr.of("force.turn_close"))


def test_bos_during_user_speech_is_suppressed_and_logged():
    asr = {"words": [{"k": 3, "word": "ciao"}, {"k": 12, "word": "vorrei"}], "nonblank": list(range(1, 14)),
           "eou": [23]}
    model = [T.PAD_ID] * 8 + [T.BOS_ID] + [1500] * 3 + [T.PAD_ID] * 13 + [T.BOS_ID] + [T.PAD_ID] * 12
    hook, w, g, ad, v, tr = rig(model, asr, [[" Si", "."]])
    assert g.d[0][8] == T.PAD_ID                               # BOS while the caller speaks: not taken
    assert hook.suppressed[0]["k"] == 8 and hook.suppressed[0]["why"] == "asr_speech_recent"
    assert tr.of("force.bos_suppressed")
    assert g.d[0][25] == T.BOS_ID                              # after the ASR end of utterance: taken
    assert hook.brain.calls[0][-1]["content"] == "ciao vorrei"


def test_no_empty_user_turn_after_the_opener():
    asr = {"words": [{"k": 3, "word": "ciao"}], "nonblank": [1, 2, 3], "eou": [13]}
    model = ([T.BOS_ID] + [T.PAD_ID] * 12 + [T.EOS_ID] + [T.PAD_ID] * 12 + [T.BOS_ID] + [T.PAD_ID] * 8 + [T.EOS_ID]
             + [T.PAD_ID] * 21 + [T.BOS_ID] + [T.PAD_ID] * 5)
    hook, w, g, ad, v, tr = rig(model, asr, [[" Benvenuto", "."], [" Bene", "."]])
    users = [m[-1]["content"] for m in hook.brain.calls]
    assert users == ["[call connected]", "ciao"]               # the third BOS (no new words) is not sent
    assert hook.skipped and hook.skipped[0]["why"] == "empty_user_turn"
    assert g.d[0][58] == T.EOS_ID                               # the empty turn is closed at the next frame


def test_shell_mode_never_writes_but_drives_the_adapter():
    v = FakeVocab()
    reply = [v.id(p) for p in [" Certo", "."]]
    w = FakeWrapper()
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode, config=AdapterConfig(), tracer=tr)
    hook = M1Force(w, FakeBrain([reply]), Tok(v), ASR, ad, tr, Log(), Log(), lambda p: ("{}", 0.0), mode="shell",
                   eos_hold=0)
    M1Force.brain_fp = staticmethod(lambda cache: "fp")
    un = hook.install()
    model = [T.PAD_ID] * 10 + [T.BOS_ID] + [1500] * 8 + [T.PAD_ID] * 6
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    assert g.d[0] == model                                         # the Thinker runs free
    assert "".join(c.text for c in ad.commands if hasattr(c, "text")) == "Certo."
    assert any(type(c).__name__ == "FinalizeSegment" for c in ad.commands)


def test_debounce_blocks_bos_right_after_a_cancel():
    asr = {"words": [{"k": 3, "word": "ciao"}], "nonblank": [1, 2, 3, 30, 31, 32, 33], "eou": [13]}
    v = FakeVocab()
    w = FakeWrapper()
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode, config=AdapterConfig(), tracer=tr)
    hook = M1Force(w, FakeBrain([[v.id(" a"), v.id(" b"), v.id(" c"), v.id(".")]] * 3), Tok(v), asr, ad, tr, Log(),
                   Log(), lambda p: ("{}", 0.0), debounce_frames=8)
    M1Force.brain_fp = staticmethod(lambda cache: "fp")
    un = hook.install()
    model = [T.PAD_ID] * 25 + [T.BOS_ID] + [T.PAD_ID] * 12 + [T.BOS_ID] * 3 + [T.PAD_ID] * 10
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    cancel_k = next(c.k for c in ad.commands if type(c).__name__ == "Cancel")   # caller speech 30..33 cut the reply
    assert any(x["why"] == "debounce_after_cancel" and x["k"] < cancel_k + 8 for x in hook.suppressed)


def test_no_new_words_gate_turns_a_bare_bos_into_pad():
    asr = {"words": [{"k": 3, "word": "ciao"}], "nonblank": [1, 2, 3], "eou": [13]}
    v = FakeVocab()
    w = FakeWrapper()
    tr = Tracer()
    ad = TalkerAdapter([FakeTalkerDriver(), PlaybackGate()], decode=v.decode, config=AdapterConfig(), tracer=tr)
    hook = M1Force(w, FakeBrain([[v.id(" Ok"), v.id(".")], [v.id(" Si"), v.id(".")]]), Tok(v), asr, ad, tr, Log(), Log(),
                   lambda p: ("{}", 0.0), no_new_words_gate=True, eos_hold=0)
    M1Force.brain_fp = staticmethod(lambda cache: "fp")
    un = hook.install()
    model = [T.BOS_ID] + [T.PAD_ID] * 25 + [T.BOS_ID] + [T.PAD_ID] * 15 + [T.BOS_ID] + [T.PAD_ID] * 5
    g = G(len(model))
    for t, p in enumerate(model):
        g[0, t] = p
        w._apply_rnnt_turn_taking(t, g, None, None)
    un()
    assert g.d[0][42] == T.PAD_ID and any(x["why"] == "no_new_caller_words" for x in hook.suppressed)
    assert not hook.skipped
