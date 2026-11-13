"""Tool-call lifecycle (controller/ledger side, no model): every transition, late and duplicate results, timeouts,
cancellation on session close, epoch fence for user corrections, grounding barrier, facts only from valid results.
Fake clock for timestamps; backends are blocked on Events so the outcome never depends on scheduling."""

import json
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_session import CS, FakeEngine  # noqa: E402

from mynah_v2.agent.controller import Controller, FactsMemory  # noqa: E402
from mynah_v2.model import tokens as T  # noqa: E402
from mynah_v2.agent.tools import (CallStatus, InvalidTransition, ToolLedger, ToolRegistry,  # noqa: E402
                                  model_response_for)
from mynah_v2.runtime.session import Mode, Session  # noqa: E402
from mynah_v2.tracing.events import Tracer  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += int(ms * 1e6)


def blocking_tool():
    """Backend that returns only when released (or raises if told to)."""
    gate, started = threading.Event(), threading.Event()
    box = {"result": json.dumps({"ok": True}), "raise": None}

    def fn(args):
        started.set()
        gate.wait(5)
        if box["raise"]:
            raise box["raise"]
        return box["result"]
    return fn, gate, started, box


def make(timeout_s=None, tracer=None, clock=None):
    reg = ToolRegistry()
    fn, gate, started, box = blocking_tool()
    reg.register("slow", fn, "d", {"type": "object", "properties": {}})
    led = ToolLedger("s", tracer, **({"clock": clock} if clock else {}))
    h = reg.bind(led, lambda: 1, lambda: 10, timeout_s=timeout_s, intent_fn=lambda: "i1")["slow"]
    return led, h, gate, started, box


def run_async(h, args=None):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", h(args or {})))
    t.start()
    return t, out


def test_ids_and_full_happy_path_transitions_are_traced():
    tr = Tracer(None, session_id="s")
    led, h, gate, started, _ = make(timeout_s=2.0, tracer=tr)
    t, out = run_async(h, {"order_id": "A1"})
    assert started.wait(2)
    (c,) = led.calls.values()
    assert (c.call_id, c.turn_id, c.intent_id, c.model_step, c.status) == ("s-c1", 1, "i1", 10, CallStatus.RUNNING)
    gate.set()
    t.join(2)
    assert json.loads(out["r"]) == {"ok": True}
    assert [h_[1] for h_ in c.history] == ["pending", "running", "completed"]
    assert [e.name for e in tr.events] == ["tool.pending", "tool.running", "tool.completed"]


def test_timeout_returns_explicit_error_at_once_and_late_result_is_recorded_not_applied():
    tr = Tracer(None, session_id="s")
    led, h, gate, started, _ = make(timeout_s=0.05, tracer=tr)
    out = h({})                                     # returns after ~50 ms although the backend is still blocked
    (c,) = led.calls.values()
    assert c.status == CallStatus.TIMED_OUT
    assert json.loads(out) == {"error": "timeout", "message": "I'm sorry, that's taking longer than expected."}
    gate.set()                                      # backend finishes later
    for _ in range(100):
        if c.late:
            break
        threading.Event().wait(0.01)
    assert c.status == CallStatus.TIMED_OUT and c.result is None and [k for _, k in c.late] == ["late"]
    late = [e for e in tr.events if e.name == "tool.result.late"]
    assert late and late[0].fields["status"] == "timed_out" and late[0].fields["has_result"]
    mem = FactsMemory()
    assert mem.add_from_call(c) is False            # a timed-out call never becomes a fact


def test_backend_exception_is_failed_with_error_payload_in_both_modes():
    for timeout in (None, 1.0):
        led, h, gate, _, box = make(timeout_s=timeout)
        box["raise"] = RuntimeError("db down")
        gate.set()
        out = h({})
        (c,) = led.calls.values()
        assert c.status == CallStatus.FAILED and json.loads(out) == {"error": "db down"}


def test_double_delivery_is_ignored_and_traced_as_duplicate():
    clk = FakeClock()
    tr = Tracer(None, session_id="s")
    led = ToolLedger("s", tr, clock=clk)
    c = led.open("t", {}, 1)
    led.start(c.call_id)
    clk.advance_ms(30)
    assert led.deliver(c.call_id, "first") is True
    clk.advance_ms(20)
    assert led.deliver(c.call_id, "second") is False
    assert c.result == "first" and [k for _, k in c.late] == ["duplicate"]
    ev = [e for e in tr.events if e.name == "tool.result.late"][0]
    assert ev.fields["after_close_ms"] == 20.0
    assert [e for e in tr.events if e.name == "tool.completed"][0].fields["latency_ms"] == 30.0


def test_out_of_order_results_are_correlated_by_call_id():
    led = ToolLedger("s")
    a, b = led.open("t", {"id": "A"}, 1), led.open("t", {"id": "B"}, 1)
    led.start(a.call_id)
    led.start(b.call_id)
    led.deliver(b.call_id, "rb")                    # B answers first
    led.deliver(a.call_id, "ra")
    assert (a.result, b.result) == ("ra", "rb") and {a.status, b.status} == {CallStatus.COMPLETED}


def test_user_correction_during_call_makes_old_result_stale_and_new_call_valid():
    led = ToolLedger("s")
    mem = FactsMemory()
    led.on_close.append(mem.add_from_call)
    old = led.open("get_order_status", {"order_id": "XY0123"}, 1)
    led.start(old.call_id)
    led.bump_epoch()                                # user: "sorry, it is ABC456"
    assert old.status == CallStatus.STALE
    assert json.loads(model_response_for(old))["error"] == "superseded"
    assert led.deliver(old.call_id, json.dumps({"found": False})) is False    # late result of the old intent
    new = led.open("get_order_status", {"order_id": "ABC456"}, 1)
    led.start(new.call_id)
    led.deliver(new.call_id, json.dumps({"found": True, "status": "processing"}))
    facts = mem.all()
    assert [f.args for f in facts] == [{"order_id": "ABC456"}]


def test_strict_transitions_still_raise():
    led = ToolLedger("s")
    c = led.open("t", {}, 1)
    led.complete(c.call_id, "r")
    with pytest.raises(InvalidTransition):
        led.fail(c.call_id, "x")
    with pytest.raises(InvalidTransition):
        led.start(c.call_id)


class ReleasingEngine(FakeEngine):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.release_calls = []

    def release_fc(self, join_timeout_s=60.0):
        self.release_calls.append(join_timeout_s)
        active = self.fc_active()
        self.fc_from = self.fc_to = None
        return active


@pytest.mark.parametrize("kind,join", [("normal", 60.0), ("cancel", 60.0), ("force", 2.0)])
def test_session_close_cancels_open_calls_and_releases_fc(kind, join):
    tr = Tracer(None, session_id="s")
    eng = ReleasingEngine(fc_from=1, fc_to=100)
    led, h, gate, started, _ = make(timeout_s=5.0, tracer=tr)
    s = Session(eng, tr, Mode.ENHANCED, controller=Controller(None, ledger=led))
    s.open("p")
    t, out = run_async(h)                            # the pipeline's FC thread waiting on our handler
    assert started.wait(2)
    for _ in range(3):
        s.feed_chunk(torch.zeros(CS))
    s.close(kind)
    t.join(2)
    (c,) = led.calls.values()
    assert c.status == CallStatus.CANCELLED and c.error == f"session_closed:{kind}"
    assert json.loads(out["r"])["error"] == "cancelled"       # the waiting handler is unblocked at once
    assert eng.release_calls == [join] and eng.closed and not eng.fc_active()
    ev = [e for e in tr.events if e.name == "session.closed"][0].fields
    assert ev["kind"] == kind and ev["cancelled_calls"] == [c.call_id] and ev["fc_active_at_close"]
    gate.set()                                                # result after the session ended
    for _ in range(100):
        if c.late:
            break
        threading.Event().wait(0.01)
    assert [k for _, k in c.late] == ["late"] and c.result is None


def test_close_kind_is_validated():
    s = Session(FakeEngine(), Tracer(None), Mode.REFERENCE)
    s.open(None)
    with pytest.raises(ValueError):
        s.close("later")


def test_grounding_barrier_flags_agent_text_while_a_call_is_open():
    tr = Tracer(None, session_id="s")
    led = ToolLedger("s", tr)
    ctl = Controller(None, ledger=led)
    s = Session(FakeEngine(), tr, Mode.REFERENCE, controller=ctl)
    s.open("p")
    c = led.open("t", {}, 0)
    led.start(c.call_id)
    s.feed_chunk(torch.zeros(CS))                    # fake agent says "hi" on its first chunk
    v = [e for e in tr.events if e.name == "grounding.violation"]
    assert len(v) == 1 and v[0].fields["call_ids"] == [c.call_id] and ctl.grounding_violations == 1
    led.deliver(c.call_id, "r")
    s.close()


def test_no_violation_when_the_result_is_in_before_the_agent_speaks():
    tr = Tracer(None, session_id="s")
    led = ToolLedger("s", tr)
    ctl = Controller(None, ledger=led)
    s = Session(FakeEngine(), tr, Mode.REFERENCE, controller=ctl)
    s.open("p")
    c = led.open("t", {}, 0)
    led.start(c.call_id)
    led.deliver(c.call_id, "r")
    s.feed_chunk(torch.zeros(CS))
    assert ctl.grounding_violations == 0
    s.close()


def test_on_hold_messages_follow_reference_json_format():
    reg = ToolRegistry()
    reg.register("get_order_status", lambda a: "", "d", {}, ack_message="Let me check that order for you.")
    reg.register("other", lambda a: "", "d", {})
    assert reg.on_hold_messages() == {"get_order_status": ["Let me check that order for you."]}


def test_controller_facts_only_from_completed_calls_across_all_terminal_states():
    led = ToolLedger("s")
    ctl = Controller("base", ledger=led)
    outcomes = {}
    for st in ("completed", "failed", "timed_out", "cancelled", "stale"):
        c = led.open("t", {"k": st}, 1)
        led.start(c.call_id)
        if st == "completed":
            led.deliver(c.call_id, json.dumps({"v": 1}))
        elif st == "failed":
            led.deliver_error(c.call_id, "boom")
        elif st == "timed_out":
            led.time_out(c.call_id)
        elif st == "cancelled":
            led.cancel(c.call_id)
        else:
            led.epoch += 1
            led.deliver(c.call_id, json.dumps({"v": 2}))
        outcomes[st] = c.status.value
    assert outcomes == {k: k for k in outcomes}
    assert [f.args for f in ctl.memory.all()] == [{"k": "completed"}]


def test_result_is_consumed_only_at_eotr_and_linked_to_its_call():
    """A returned result is not 'consumed' until the model emits EOTR after the forced TOOL_RESPONSE."""
    tr = Tracer(None, session_id="s")
    led = ToolLedger("s", tr)
    ctl = Controller(None, ledger=led)
    eng = FakeEngine(fc_from=2, fc_to=5)
    s = Session(eng, tr, Mode.REFERENCE, controller=ctl)
    s.open("p")
    s.feed_chunk(torch.zeros(CS))
    c = led.open("t", {}, 0)
    led.start(c.call_id)
    led.deliver(c.call_id, "[]")
    assert ctl.consumed == {}                        # result returned, not consumed yet
    for _ in range(5):
        s.feed_chunk(torch.zeros(CS))
    ev = [e for e in tr.events if e.name == "tool.result.consumed"]
    assert len(ev) == 1 and ev[0].fields["call_id"] == c.call_id and c.call_id in ctl.consumed
    s.close()


def test_reference_mode_controller_is_observer_only():
    """REFERENCE + controller: bookkeeping happens, but the model input is unchanged (same chunks, no rehydration)."""
    from test_session import _chunks
    ref_in, ctl_in = FakeEngine(), FakeEngine()
    a = Session(ref_in, Tracer(None), Mode.REFERENCE)
    b = Session(ctl_in, Tracer(None), Mode.REFERENCE, controller=Controller("p", ledger=ToolLedger("s")))
    for s in (a, b):
        s.open("p")
        for c in _chunks(6):
            s.feed_chunk(c)
        s.close()
    assert ref_in.seen == ctl_in.seen and b.controller.turns()


def test_registry_warns_beyond_paper_tool_limit():
    reg = ToolRegistry()
    for i in range(5):
        reg.register(f"t{i}", lambda a: "", "d", {})
    with pytest.warns(UserWarning, match="at most 5"):
        reg.register("t5", lambda a: "", "d", {})


def test_grounding_audit_catches_answers_to_a_call_that_was_never_answered():
    """GPU finding (REFERENCE, tool without handler): the call stayed open for the model, nothing was injected, and
    the agent then stated an invented delivery date. The ledger never saw the call; the audit must still flag it."""
    tr = Tracer(None, session_id="s")
    ctl = Controller(None, ledger=ToolLedger("s", tr))
    class Talker(FakeEngine):
        def step(self, pcm, is_last=False):
            o = super().step(pcm, is_last)
            if self.k == 3:                              # the agent "answers" two chunks after the request
                o.text_delta, o.new_text_ids = "It arrives October 22.", [T.BOS_ID, 40]
            return o
    eng = Talker(fc_from=1, fc_to=10**9)                # phase-1 record (SOTC..EOTC), never answered
    s = Session(eng, tr, Mode.REFERENCE, controller=ctl)
    s.open("p")
    for _ in range(3):
        s.feed_chunk(torch.zeros(CS))
    assert ctl.requests_outstanding and ctl.grounding_violations == 2   # "hi" (same chunk as EOTC) + the answer
    v = [e for e in tr.events if e.name == "grounding.violation"][0].fields
    assert v["unanswered_requests"] and v.get("call_ids") is None
    s.close("cancel")
    assert [e for e in tr.events if e.name == "session.closed"][0].fields["unanswered_requests"] == 1


def test_consumption_in_the_same_chunk_precedes_that_chunks_text():
    """GPU false positive (s2): EOTR at step 169 and agent text at 172 arrived in one StepOutput; the text was
    audited before the consumption and flagged."""
    tr = Tracer(None, session_id="s")
    ctl = Controller(None, ledger=ToolLedger("s", tr))

    class SameChunk(FakeEngine):
        def step(self, pcm, is_last=False):
            o = super().step(pcm, is_last)
            if self.k == 3:                              # phase-2 record (EOTR) and agent text together
                o.text_delta, o.new_text_ids = "Your order", [T.BOS_ID, 40]
            return o
    eng = SameChunk(fc_from=1, fc_to=3)
    s = Session(eng, tr, Mode.REFERENCE, controller=ctl)
    s.open("p")
    for _ in range(4):
        s.feed_chunk(torch.zeros(CS))
    assert ctl.grounding_violations == 1                 # only the fake's own "hi" at chunk 1 (call open then)
    assert not ctl.requests_outstanding


def test_multi_call_executes_every_call_in_order_with_one_ledger_entry_each():
    """Reference executes only parsed[0] (P:790). ENHANCED: all calls, responses in call order (template [r1, r2]),
    unknown names answered in their slot, out-of-order completion does not reorder the response."""
    import time as _t
    tr = Tracer(None, session_id="s")
    led = ToolLedger("s", tr)
    reg = ToolRegistry()
    reg.register("slow", lambda a: (_t.sleep(0.2), json.dumps({"id": a["id"], "v": "slow"}))[1], "d", {})
    reg.register("fast", lambda a: json.dumps({"id": a["id"], "v": "fast"}), "d", {})
    bound = reg.bind(led, lambda: 1, lambda: None, timeout_s=2.0)
    ctl = Controller(None, ledger=led, tools=bound)
    out = ctl.execute_calls('<TOOLCALL>[{"name": "slow", "arguments": {"id": 1}}, '
                            '{"name": "fast", "arguments": {"id": 2}}, {"name": "nope", "arguments": {}}]</TOOLCALL>')
    items = json.loads("[" + out + "]")
    assert [i.get("v") for i in items[:2]] == ["slow", "fast"] and items[2]["error"] == "unknown_tool"
    st = sorted((c.name, c.status.value) for c in led.calls.values())
    assert st == [("fast", "completed"), ("nope", "failed"), ("slow", "completed")]
    assert any(e.name == "tool.multi_call" for e in tr.events)


def test_multi_call_slot_times_out_without_blocking_the_others():
    led = ToolLedger("s")
    reg = ToolRegistry()
    gate = threading.Event()
    reg.register("stuck", lambda a: (gate.wait(5), "{}")[1], "d", {})
    reg.register("ok", lambda a: json.dumps({"ok": 1}), "d", {})
    ctl = Controller(None, ledger=led, tools=reg.bind(led, lambda: 1, lambda: None, timeout_s=0.1), call_timeout_s=1.0)
    items = json.loads("[" + ctl.execute_calls('[{"name": "stuck", "arguments": {}}, {"name": "ok", "arguments": {}}]') + "]")
    assert items[0]["error"] == "timeout" and items[1] == {"ok": 1}
    gate.set()


def test_session_wires_multi_call_policy_only_in_enhanced_with_tools():
    from mynah_v2.runtime.session import EnhancedOptions
    eng = FakeEngine()
    eng.multi_call_handler = "unset"
    Session(eng, Tracer(None), Mode.REFERENCE, controller=Controller(None, tools={"x": lambda a: ""})).open(None)
    assert eng.multi_call_handler is None
    ctl = Controller(None, tools={"x": lambda a: ""})
    Session(eng, Tracer(None), Mode.ENHANCED, EnhancedOptions(), controller=ctl).open(None)
    assert eng.multi_call_handler == ctl.execute_calls


def test_adapter_routes_call_lists_to_the_policy_and_single_calls_to_the_reference():
    from mynah_v2.model.reference import ReferenceEngine

    class P:
        def _execute_tool_call(self, call_text):
            return "first-only"
    eng = object.__new__(ReferenceEngine)
    eng.pipeline = P()
    eng.unknown_tool_handler, eng.multi_call_handler = None, None
    eng.unknown_tool_events, eng.multi_call_events = [], []
    eng._wrap_executor()
    two = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
    assert eng.pipeline._execute_tool_call(two) == "first-only"            # REFERENCE: NVIDIA's behaviour
    eng.multi_call_handler = lambda t: "all"
    assert eng.pipeline._execute_tool_call(two) == "all"
    assert eng.pipeline._execute_tool_call('[{"name": "a", "arguments": {}}]') == "first-only"
    assert eng.multi_call_events and eng.multi_call_events[0]["n"] == 2
