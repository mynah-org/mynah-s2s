"""Controller (CONTROL.md): one test per sequence diagram, plus race orderings and idempotency."""

from __future__ import annotations

import itertools

from frankenstein_s2s.controller import Controller, ControllerConfig, Intent, St
from frankenstein_s2s.tracing import Tracer


def I(kind, source, k, **kw):  # noqa: E743
    return Intent(kind, source, k, **kw)


def run(c, script, until):
    """script: {k: [intents]} -> list of (k, action kind, reason)."""
    out = []
    for k in range(until + 1):
        for it in script.get(k, []):
            c.post(it)
        out += [(a.k, a.kind, a.reason) for a in c.tick(k)]
    return out


def speech(k0, n):
    return {k: [I("USER_SPEECH", "asr", k)] for k in range(k0, k0 + n)}


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def kinds(acts):
    return [a[1] for a in acts]


def test_normal_turn():
    c = Controller()
    s = merge(speech(10, 5), {14: [I("WORDS", "asr", 14, payload="vorrei")], 17: [I("TAKE_TURN", "thinker", 17, payload="bos")],
                              19: [I("REPLY_TOKENS", "nano", 19, payload=[5, 6], request_id=1, epoch=0)],
                              24: [I("REPLY_DONE", "nano", 24, request_id=1, epoch=0)]})
    acts = run(c, s, 26)
    assert kinds(acts) == ["OPEN_TURN", "REQUEST_REPLY", "SPEAK", "CLOSE_TURN"]
    assert acts[0][0] == 17 and c.state is St.LISTENING


def test_tool_turn():
    c = Controller()
    s = merge(speech(1, 3), {3: [I("WORDS", "asr", 3)], 6: [I("TAKE_TURN", "thinker", 6, payload="tool_call_start")],
                             8: [I("TOOL_CALL", "nano", 8, request_id=1, epoch=0, call_id="c1", payload={"name": "find"})],
                             20: [I("TOOL_RESULT", "tool", 20, call_id="c1", epoch=0)],
                             22: [I("REPLY_TOKENS", "nano", 22, payload=[9], request_id=1, epoch=0)],
                             25: [I("REPLY_DONE", "nano", 25, request_id=1, epoch=0)]})
    assert kinds(run(c, s, 26)) == ["OPEN_TURN", "REQUEST_REPLY", "CALL_TOOL", "RESUME_AFTER_TOOL", "SPEAK",
                                    "CLOSE_TURN"]


def test_barge_in_during_speech_bumps_epoch_and_drops_stale_tokens():
    c = Controller()
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 6: [I("TAKE_TURN", "thinker", 6)],
                             7: [I("REPLY_TOKENS", "nano", 7, payload=[1], request_id=1, epoch=0)]},
              speech(10, 3), {13: [I("REPLY_TOKENS", "nano", 13, payload=[2], request_id=1, epoch=0)]})
    acts = run(c, s, 14)
    assert kinds(acts) == ["OPEN_TURN", "REQUEST_REPLY", "SPEAK", "HOLD", "HOLD", "CANCEL", "DROP_STALE"]
    assert c.epoch == 1 and acts[5][0] == 12


def test_barge_in_during_tool_wait_drops_the_late_result():
    c = Controller()
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)],
                             6: [I("TOOL_CALL", "nano", 6, request_id=1, epoch=0, call_id="c1")]},
              speech(8, 3), {15: [I("TOOL_RESULT", "tool", 15, call_id="c1", epoch=0)]})
    acts = run(c, s, 16)
    assert kinds(acts)[-2:] == ["CANCEL", "DROP_STALE"] and acts[-1][2] == "call_cancelled"


def test_backchannel_does_not_cancel():
    c = Controller()
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)],
                             6: [I("REPLY_TOKENS", "nano", 6, payload=[1], request_id=1, epoch=0)]},
              speech(9, 2), {14: [I("REPLY_DONE", "nano", 14, request_id=1, epoch=0)]})
    acts = run(c, s, 15)
    assert "CANCEL" not in kinds(acts) and kinds(acts)[-1] == "CLOSE_TURN"
    assert [a[2] for a in acts if a[1] == "HOLD"] == ["backchannel_candidate"] * 2


def test_long_reply_streams_in_chunks_without_state_changes():
    c = Controller()
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)]},
              {k: [I("REPLY_TOKENS", "nano", k, payload=[k], request_id=1, epoch=0)] for k in range(6, 30)},
              {31: [I("REPLY_DONE", "nano", 31, request_id=1, epoch=0)]})
    acts = run(c, s, 32)
    assert kinds(acts).count("SPEAK") == 24 and kinds(acts)[-1] == "CLOSE_TURN" and c.epoch == 0


def test_second_turn_gets_new_turn_and_request():
    c = Controller()
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)],
                             6: [I("REPLY_DONE", "nano", 6, request_id=1, epoch=0)]},
              speech(10, 2), {11: [I("WORDS", "asr", 11)], 14: [I("TAKE_TURN", "thinker", 14)]})
    acts = run(c, s, 15)
    opens = [a for a in c.actions if a.kind == "REQUEST_REPLY"]
    assert [(a.turn, a.data["request_id"]) for a in opens] == [(1, 1), (2, 2)]


def test_take_turn_while_asr_hears_speech_is_deferred_then_opened():
    c = Controller()
    s = merge(speech(1, 6), {3: [I("WORDS", "asr", 3)], 5: [I("TAKE_TURN", "thinker", 5)]})
    acts = run(c, s, 12)
    assert acts[0][1:] == ("DEFER", "asr_speech_recent")
    assert ("OPEN_TURN", "thinker_bos_deferred") in [(a[1], a[2]) for a in acts] and \
        next(a[0] for a in acts if a[1] == "OPEN_TURN") == 8          # last speech 6 + min_blank 2


def test_no_turn_without_new_caller_words_and_debounce_after_cancel():
    c = Controller(ControllerConfig(debounce=8))
    s = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)],
                             6: [I("CANCEL", "app", 6)], 8: [I("WORDS", "asr", 8)], 9: [I("TAKE_TURN", "thinker", 9)],
                             20: [I("TAKE_TURN", "thinker", 20)]})
    acts = run(c, s, 21)
    reasons = [(a[0], a[1], a[2]) for a in acts]
    assert (9, "HOLD", "debounce_after_cancel") in reasons and (20, "OPEN_TURN", "thinker_bos") in reasons
    c2 = Controller()
    run(c2, {1: [I("WORDS", "asr", 1)], 4: [I("TAKE_TURN", "thinker", 4)], 5: [I("REPLY_DONE", "nano", 5,
                                                                                  request_id=1, epoch=0)],
             9: [I("TAKE_TURN", "thinker", 9)]}, 10)
    assert c2.actions[-1].reason == "no_new_caller_words"


def test_reply_and_tool_timeouts():
    c = Controller(ControllerConfig(reply_timeout=5, tool_timeout=5))
    run(c, {1: [I("WORDS", "asr", 1)], 4: [I("TAKE_TURN", "thinker", 4)]}, 12)
    assert c.actions[-1].reason == "reply_timeout"
    c = Controller(ControllerConfig(tool_timeout=5))
    run(c, {1: [I("WORDS", "asr", 1)], 4: [I("TAKE_TURN", "thinker", 4)],
            5: [I("TOOL_CALL", "nano", 5, request_id=1, epoch=0, call_id="c")]}, 12)
    assert c.actions[-1].kind == "RESUME_AFTER_TOOL" and c.actions[-1].reason == "tool_timeout"


def test_race_orderings_give_the_same_decision():
    def build(order):
        c = Controller()
        pre = merge(speech(1, 2), {2: [I("WORDS", "asr", 2)], 5: [I("TAKE_TURN", "thinker", 5)],
                                   6: [I("TOOL_CALL", "nano", 6, request_id=1, epoch=0, call_id="c1")]})
        run(c, pre, 6)
        same_tick = [I("USER_SPEECH", "asr", 7), I("TOOL_RESULT", "tool", 7, call_id="c1", epoch=0),
                     I("TAKE_TURN", "thinker", 7), I("CANCEL", "app", 7)]
        for i in order:
            c.post(same_tick[i])
        return [(a.kind, a.reason) for a in c.tick(7)]
    results = {tuple(build(p)) for p in itertools.permutations(range(4))}
    assert len(results) == 1
    assert list(results)[0][0] == ("CANCEL", "app:cancel")        # cancel wins over the tool result


def test_race_speech_vs_take_turn_and_tokens_vs_barge_in():
    for order in itertools.permutations(range(2)):
        c = Controller()
        run(c, {1: [I("WORDS", "asr", 1)]}, 3)
        its = [I("TAKE_TURN", "thinker", 4), I("USER_SPEECH", "asr", 4)]
        for i in order:
            c.post(its[i])
        assert [(a.kind, a.reason) for a in c.tick(4)] == [("DEFER", "asr_speech_recent")]
    outs = set()
    for order in itertools.permutations(range(2)):
        c = Controller()
        run(c, merge({1: [I("WORDS", "asr", 1)], 4: [I("TAKE_TURN", "thinker", 4)]}, speech(8, 2)), 9)
        its = [I("REPLY_TOKENS", "nano", 10, payload=[1], request_id=1, epoch=0), I("USER_SPEECH", "asr", 10)]
        for i in order:
            c.post(its[i])
        outs.add(tuple((a.kind, a.reason) for a in c.tick(10)))
    assert outs == {(("CANCEL", "barge_in"), ("DROP_STALE", "old_epoch"))}


def test_idempotent_and_traced():
    tr = Tracer()
    c = Controller(tracer=tr)
    run(c, {1: [I("WORDS", "asr", 1)], 4: [I("TAKE_TURN", "thinker", 4)],
            5: [I("TOOL_CALL", "nano", 5, request_id=1, epoch=0, call_id="c1", uid="x"),
                I("TOOL_CALL", "nano", 5, request_id=1, epoch=0, call_id="c1", uid="x")]}, 6)
    assert [a.kind for a in c.actions].count("CALL_TOOL") == 1
    assert len(tr.of("ctl.intent")) == 4 and all("epoch" in e for e in tr.of("ctl.action"))
