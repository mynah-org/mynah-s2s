"""CPU tests of the experimental per-session call registry + admission control, on the call sequences the live runs
actually produced, plus concurrency, timeouts, side effects and cleanup.

Sequences (calls, arguments, backend results, caller speech in between) are copied from the traces:
- engine FX1 with the (PAD, EOTR) resume: m02 11x get_order_status(qgq999) found=false; t14 9x (WXYY615); t13
  XYZ123 found, then the corrected DF246 found=false 7x; m01 two random-number calls, a caller turn apart;
  no caller speech between the repeated calls;
- EN m07 official: (PAD, EOTR) find_customer(Mario Rossi, 20135) verified C1002 -> prepare_return(C1002);
- EN m07 r3 (lead 7.5 s, (PAD, PAD)): find_customer without postcode, caller says the postcode,
  prepare_return(customer_id="Mario Rossi"), find_customer(Mario Rossi, 20135), prepare_return(C1002)."""
import json
import threading
import time

from mynah_v2.agent.call_guard import (PRODUCTIVE, TIMEOUT, UNPRODUCTIVE, CallGuard, default_outcome,
                                       guarded_executor)
from mynah_v2.agent.tools import ToolLedger, ToolRegistry

NF = lambda oid: json.dumps({"order_id": oid.upper(), "found": False})  # noqa: E731 - engine orders_tool shape
FOUND_XYZ = json.dumps({"order_id": "XYZ123", "found": True, "status": "shipped", "delivery_date": "November 12, 2026",
                        "tracking_code": "7319"})
VERIFIED = json.dumps({"status": "verified", "customer_id": "C1002", "first_name": "Mario",
                       "latest_order": {"item": "Lysa bookcase", "status": "delivered", "date": "2026-11-05"}})
PENDING_P001 = json.dumps({"status": "pending_confirmation", "pending_id": "P001", "item": "Lysa bookcase"})
VERIF_REQ = json.dumps({"status": "verification_required", "matching_names": 1, "next_step": "Ask for the postcode."})
TENANT = {"find_customer": {"properties": {"full_name": {"type": "string", "x-source": "caller"},
                                           "postcode": {"type": "string", "x-source": "caller"}}},
          "prepare_return": {"properties": {"item": {"type": "string"},
                                            "customer_id": {"type": "string", "x-source": "result"}}},
          "commit_return": {"properties": {"pending_id": {"type": "string", "x-source": "result"}}},
          "escalate_to_human": {"properties": {"reason": {"type": "string"}}}}
SIDE = {"commit_return": "pending_id", "escalate_to_human": None}


def ct(name, args):
    return json.dumps([{"name": name, "arguments": args}])


def run(guard, seq):
    """seq: ("user", text) | (tool, args, backend_result). Returns the decisions in call order (rule or "run")."""
    out = []
    for s in seq:
        if s[0] == "user":
            guard.on_user_text(s[1])
            continue
        d = guard.admit(s[0], s[1])
        out.append(d.rule or "run")
        if d.allow:
            guard.complete(d, s[2])
        else:
            assert json.loads(d.response)["status"] == "not_executed" and json.loads(d.response)["next_step"]
    return out


def test_outcome_classifier():
    assert default_outcome(NF("qgq999")) == UNPRODUCTIVE
    assert default_outcome(VERIF_REQ) == UNPRODUCTIVE
    assert default_outcome(json.dumps({"error": "unknown_tool"})) == UNPRODUCTIVE
    assert default_outcome(json.dumps({"error": "timeout", "message": "x"})) == TIMEOUT
    assert default_outcome(FOUND_XYZ) == PRODUCTIVE and default_outcome(VERIFIED) == PRODUCTIVE
    assert default_outcome(PENDING_P001) == PRODUCTIVE
    assert default_outcome(json.dumps({"result": 1, "min": 1, "max": 10})) == PRODUCTIVE


# ------------------------------------------------------------------ measured sessions
def test_m02_loop_backend_runs_once_and_next_turn_is_served():
    g = CallGuard(session_id="m02")
    a = {"order_id": "qgq999"}
    seq = [("user", "where is my order? Q, Q, nine, nine, nine.")] + [("get_order_status", a, NF("qgq999"))] * 11
    seq += [("user", "give me a random number"), ("generate_random_number", {"min": 1, "max": 10}, '{"result": 10}')]
    got = run(g, seq)
    assert got[:3] == ["run", "repeat", "repeat"]
    assert set(got[3:11]) == {"budget"}
    assert got[11] == "run"
    assert sum(r == "run" for r in got[:11]) == 1        # the backend saw the order lookup once, not 11 times


def test_t14_repeat_reuses_the_earlier_result():
    g = CallGuard()
    g.on_user_text("check order W x y six one five.")
    a = {"order_id": "WXYY615"}
    d = g.admit("get_order_status", a)
    g.complete(d, NF("WXYY615"))
    r = g.admit("get_order_status", a)
    body = json.loads(r.response)
    assert r.rule == "repeat" and body["previous_result"] == {"order_id": "WXYY615", "found": False}
    assert "Do not call the tool again" in body["next_step"]


def test_t13_corrected_id_runs_then_its_repeat_is_refused():
    g = CallGuard()
    got = run(g, [("user", "where is my order X Y Z one two three"), ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ),
                  ("user", "sorry, I meant D F G two four six"),
                  ("get_order_status", {"order_id": "DF246"}, NF("DF246")),
                  ("get_order_status", {"order_id": "DF246"}, NF("DF246"))])
    assert got == ["run", "run", "repeat"]


def test_m01_two_calls_a_caller_turn_apart_both_run():
    g = CallGuard()
    got = run(g, [("user", "random number one to ten"), ("generate_random_number", {"min": 1, "max": 10}, '{"result": 1}'),
                  ("user", "another one, one to five"), ("generate_random_number", {"min": 1, "max": 5}, '{"result": 1}')])
    assert got == ["run", "run"]


def test_m07_chain_runs_without_new_speech():
    g = CallGuard(TENANT, side_effects=SIDE)
    got = run(g, [("user", "I want to return the bookcase I bought, Mario Rossi, postcode two zero one three five"),
                  ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED),
                  ("prepare_return", {"item": "Lysa bookcase", "customer_id": "C1002"}, PENDING_P001)])
    assert got == ["run", "run"]
    assert g.log[0]["caller_slots_unchecked"] == ["full_name", "postcode"]


def test_r3_invented_customer_id_is_refused_and_the_chain_still_completes():
    g = CallGuard(TENANT, side_effects=SIDE)
    got = run(g, [("user", "I want to return the bookcase I bought. Mario Rosie."),
                  ("find_customer", {"full_name": "Mario Rossi"}, VERIF_REQ),
                  ("user", "postcode two zero one three five."),
                  ("prepare_return", {"item": "bookcase", "customer_id": "Mario Rossi"}, PENDING_P001),
                  ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED),
                  ("prepare_return", {"item": "Lysa bookcase", "customer_id": "C1002"}, PENDING_P001)])
    assert got == ["run", "ungrounded", "run", "run"]


def test_id_from_a_negative_result_is_not_grounded():
    g = CallGuard(TENANT)
    got = run(g, [("user", "return, Lindvik, two zero one three five"),
                  ("find_customer", {"full_name": "Lindvik", "postcode": "20135"},
                   json.dumps({"status": "not_found", "customer_id": "C9999"})),
                  ("prepare_return", {"item": "x", "customer_id": "C9999"}, PENDING_P001)])
    assert got == ["run", "ungrounded"]


# ------------------------------------------------------------------ retries
def test_retry_with_new_arguments_after_not_found_is_not_blocked_by_similarity():
    g = CallGuard()
    got = run(g, [("user", "order Q Q nine nine nine"), ("get_order_status", {"order_id": "qgq999"}, NF("qgq999")),
                  ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999"))])
    assert got == ["run", "run"]


def test_two_negative_results_without_new_input_spend_the_budget():
    g = CallGuard()
    got = run(g, [("user", "order Q Q nine nine nine"), ("get_order_status", {"order_id": "qgq999"}, NF("qgq999")),
                  ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999")),
                  ("get_order_status", {"order_id": "QGQ990"}, NF("QGQ990"))])
    assert got == ["run", "run", "budget"]


def test_same_call_after_new_speech_is_a_legitimate_retry():
    g = CallGuard()
    got = run(g, [("user", "order Q Q Q nine nine nine"), ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999")),
                  ("user", "yes, Q Q Q nine nine nine, please check again"),
                  ("get_order_status", {"order_id": "qqq-999"}, NF("QQQ999"))])
    assert got == ["run", "run"]
    g.on_user_text("  ...  ")                                    # punctuation / silence is not new input
    assert g.admit("get_order_status", {"order_id": "QQQ999"}).rule == "repeat"


def test_productive_reopen_without_new_input_is_refused():
    g = CallGuard()
    got = run(g, [("user", "X Y Z one two three"), ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ),
                  ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ)])
    assert got == ["run", "repeat"]


# ------------------------------------------------------------------ side effects
def test_side_effect_without_idempotency_key_never_reruns_even_after_new_speech():
    g = CallGuard(TENANT, side_effects=SIDE)
    got = run(g, [("user", "I want a person"), ("escalate_to_human", {"reason": "return"}, '{"status": "transferring"}'),
                  ("user", "hello? a person please"), ("escalate_to_human", {"reason": "return"}, "{}")])
    assert got == ["run", "side_effect_done"]


def test_side_effect_with_idempotency_key_goes_to_the_backend():
    g = CallGuard(TENANT, side_effects=SIDE)
    got = run(g, [("user", "return, Mario Rossi, two zero one three five"),
                  ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED),
                  ("prepare_return", {"item": "Lysa bookcase", "customer_id": "C1002"}, PENDING_P001),
                  ("user", "yes"),
                  ("commit_return", {"pending_id": "P001"}, '{"status": "submitted"}'),
                  ("commit_return", {"pending_id": "P001"}, '{"status": "already_submitted"}'),
                  ("commit_return", {"pending_id": "P777"}, "{}")])
    assert got == ["run", "run", "run", "run", "ungrounded"]   # the backend deduplicates by pending_id


def test_timed_out_side_effect_is_never_retried_automatically():
    g = CallGuard(TENANT, side_effects=SIDE)
    g.on_user_text("a person please")
    d = g.admit("escalate_to_human", {"reason": "x"})
    g.complete(d, json.dumps({"error": "timeout", "message": "slow"}))
    g.on_user_text("hello?")
    r = g.admit("escalate_to_human", {"reason": "x"})
    assert r.rule == "side_effect_unknown" and "unknown" in json.loads(r.response)["next_step"]


# ------------------------------------------------------------------ single-flight, timeouts, concurrency
def test_timeout_with_backend_still_running_blocks_a_second_run_until_the_late_result():
    """ENHANCED bind(timeout_s): the handler returns the timeout error while the backend keeps running."""
    led = ToolLedger("s1")
    reg = ToolRegistry()
    release, runs = threading.Event(), []

    def slow(args):
        runs.append(args)
        release.wait(5)
        return NF(args["order_id"])
    reg.register("get_order_status", slow, "d", {"type": "object", "properties": {"order_id": {"type": "string"}}})
    h = reg.bind(led, lambda: 1, lambda: None, timeout_s=0.05)["get_order_status"]
    g = CallGuard(session_id="s1", ledger=led)
    ex = guarded_executor(lambda text: h(json.loads(text)[0]["arguments"]), g)
    g.on_user_text("order Q Q nine")
    first = json.loads(ex(ct("get_order_status", {"order_id": "QQ9"})))
    assert first["error"] == "timeout"
    again = json.loads(ex(ct("get_order_status", {"order_id": "QQ9"})))
    assert again["status"] == "not_executed" and "still running" in again["reason"]
    other = json.loads(ex(ct("get_order_status", {"order_id": "AB1"})))
    assert "still running in this session" in other["reason"]   # single-flight counts the running backend
    release.set()
    for _ in range(100):
        if led.calls[g.reg.entries()[0].call_id].late:
            break
        time.sleep(0.01)
    g.on_user_text("can you try again")                       # the refusals spent this turn's budget
    third = json.loads(ex(ct("get_order_status", {"order_id": "QQ9"})))
    assert third["found"] is False and len(runs) == 2        # a retry after the backend returned runs normally


def test_single_flight_within_a_session():
    g = CallGuard(max_active=1)
    g.on_user_text("x")
    d = g.admit("search_catalog", {"query": "sofa"})
    assert d.allow
    busy = g.admit("search_catalog", {"query": "bed"})
    assert busy.rule == "busy"
    g.complete(d, '{"results": []}')
    g.on_user_text("y")
    assert g.admit("search_catalog", {"query": "bed"}).allow


def test_four_sessions_run_four_calls_at_once():
    """B=4: one guard per session, nothing shared; four backends must be in flight together."""
    inside, peak, lock, gate = [0], [0], threading.Lock(), threading.Barrier(4, timeout=5)

    def backend(text):
        with lock:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        gate.wait()                                   # passes only if all four are inside at the same time
        with lock:
            inside[0] -= 1
        return '{"found": true}'
    guards = [CallGuard(session_id=f"s{i}") for i in range(4)]
    exs = [guarded_executor(backend, g) for g in guards]
    res = [None] * 4

    def worker(i):
        guards[i].on_user_text("order")
        res[i] = exs[i](ct("get_order_status", {"order_id": "XYZ123"}))
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert peak[0] == 4 and all(json.loads(r) == {"found": True} for r in res)


def test_guarded_executor_keeps_the_fc_contract():
    """Always a TOOL_RESPONSE for a parsed call (never None for a refusal: the model must not be left waiting); the
    backend runs only for admitted calls; unparsable text goes to the existing policy; a crash frees the entry."""
    ran = []

    def backend(call_text):
        ran.append(call_text)
        return NF("qgq999")
    g = CallGuard()
    ex = guarded_executor(backend, g)
    c = ct("get_order_status", {"order_id": "qgq999"})
    assert json.loads(ex(c))["found"] is False
    r = ex(c)
    assert r is not None and json.loads(r)["status"] == "not_executed" and len(ran) == 1
    assert ex("not json") == NF("qgq999") and len(ran) == 2
    multi = json.dumps([{"name": "get_order_status", "arguments": {"order_id": "qgq999"}},
                        {"name": "get_order_status", "arguments": {"order_id": "ABC456"}}])
    g3 = CallGuard()
    g3.on_user_text("x")
    g3.complete(g3.admit("get_order_status", {"order_id": "qgq999"}), NF("qgq999"))
    parts = json.loads("[" + guarded_executor(lambda t: '{"found": true}', g3)(multi) + "]")
    assert parts[0]["status"] == "not_executed" and parts[1] == {"found": True}

    def boom(t):
        raise RuntimeError("backend down")
    g2 = CallGuard()
    g2.on_user_text("x")
    assert "backend down" in json.loads(guarded_executor(boom, g2)(c))["error"]
    assert g2._active() == 0


# ------------------------------------------------------------------ memory bound, TTL, close
def test_ttl_and_lru_bound_free_entries():
    now = [0]
    g = CallGuard(ttl_s=10, max_entries=4, clock=lambda: now[0])
    for i in range(10):
        g.on_user_text(f"order {i}")
        d = g.admit("get_order_status", {"order_id": f"A{i}"})
        g.complete(d, FOUND_XYZ)
    assert len(g.reg.entries()) == 4                          # bounded
    g.on_user_text("again")
    now[0] = int(11e9)
    g.admit("get_order_status", {"order_id": "B1"})
    assert [e.args for e in g.reg.entries()] == [{"order_id": "B1"}]   # older than the TTL: gone


def test_issued_ids_are_bounded():
    g = CallGuard(max_issued=8)
    for i in range(20):
        g.on_user_text(f"lookup {i}")
        g.complete(g.admit("find_customer", {"n": i}), json.dumps({"status": "verified", "customer_id": f"C{i}"}))
    assert len(g.issued) == 8 and "c19" in g.issued and "c0" not in g.issued


def test_close_frees_everything_and_refuses_late_calls():
    g = CallGuard(TENANT)
    run(g, [("user", "Mario Rossi two zero one three five"),
            ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED)])
    g.close()
    assert g.reg.entries() == [] and not g.issued
    d = g.admit("prepare_return", {"item": "x", "customer_id": "C1002"})
    assert d.rule == "closed" and json.loads(d.response)["status"] == "not_executed"


def test_install_wraps_and_restores_the_engine_executor():
    from types import SimpleNamespace

    from mynah_v2.agent.call_guard import install, summary
    from mynah_v2.tracing.events import Tracer
    ran = []
    pipe = SimpleNamespace(_execute_tool_call=lambda t: (ran.append(t), NF("qgq999"))[1])
    orig = pipe._execute_tool_call
    eng = SimpleNamespace(pipeline=pipe)
    tr = Tracer(None, session_id="s")
    g = CallGuard(session_id="s")
    un = install(eng, g, tracer=tr)
    assert pipe._execute_tool_call is not orig
    g.on_user_text("order Q Q nine nine nine")
    c = ct("get_order_status", {"order_id": "qgq999"})
    pipe._execute_tool_call(c)
    refused = json.loads(pipe._execute_tool_call(c))
    assert refused["status"] == "not_executed" and len(ran) == 1
    ev = [e for e in tr.events if e.name == "tool.guard"]
    assert [e.fields["allow"] for e in ev] == [True, False] and ev[1].fields["rule"] == "repeat"
    s = summary(g)
    assert s["admitted"] == 1 and s["refused"] == {"repeat": 1}
    un()
    assert pipe._execute_tool_call is orig and g.closed


# ------------------------------------------------------------------ ASR shadow (observe only), ASR text from the traces
def _shadow_guard(**kw):
    from mynah_v2.agent.call_guard import AsrShadow
    return CallGuard(asr_shadow=AsrShadow(intent_tools={"transfer": "escalate_to_human"}), **kw)


def test_asr_shadow_slot_checks_on_measured_sessions():
    cases = [  # (caller ASR as traced, call args, expected per-slot verdict)        live truth
        ("you check order Wxy six one five.", {"order_id": "WXYY615"}, "mismatch"),   # spoken WXY615: call wrong
        ("you check order Wxy six one five.", {"order_id": "WXY615"}, "match"),
        ("my name is Marco Ricci,postcode two zero one two one.", {"postcode": "20012-001"}, "mismatch"),  # call wrong
        ("my name is Mario Rossi and my postcode is one hundred one two one.", {"postcode": "10121"}, "mismatch"),
        ("I want to return the bookcase eyebought.Mario Rosie, postcode two zero one three five.",
         {"postcode": "20135"}, "match"),
    ]
    for asr, args, want in cases:
        g = _shadow_guard()
        g.on_user_text(asr, model_step=100)
        d = g.admit("get_order_status", args)
        assert d.allow                                              # shadow never refuses
        assert list(g.log[-1]["asr_slots"].values()) == [want], (asr, args)
    g = _shadow_guard()
    g.on_user_text("Mario Rossi")
    g.admit("find_customer", {"full_name": "Mario Rossi"})
    assert g.log[-1]["asr_slots"] == {}                             # names are not id-like: not checked


def test_asr_shadow_transfer_intent_fulfilled_negated_and_absent():
    from mynah_v2.agent.call_guard import summary
    g = _shadow_guard()
    g.on_user_text("I'd like to speak to an operator, please.", model_step=60)
    r = summary(g)["asr_shadow"]
    assert [i["intent"] for i in r["unfulfilled"]] == ["transfer"]          # m04: promised, never called
    g.admit("escalate_to_human", {"reason": "operator"})
    assert summary(g)["asr_shadow"]["unfulfilled"] == []
    n = _shadow_guard()
    n.on_user_text("I don't want to talk to an operator, just check my order.")
    i = summary(n)["asr_shadow"]["intents"]
    assert i and i[0]["negated"] and i[0]["fulfilled"] is None and not summary(n)["asr_shadow"]["unfulfilled"]
    for text in ("I want to know where my order is, but I don't remember the order number",
                 "I was going to ask about an order, but never mind. Give me a random number between one and five."):
        o = _shadow_guard()
        o.on_user_text(text)
        assert summary(o)["asr_shadow"]["intents"] == []


def test_asr_shadow_never_changes_a_decision():
    seq = [("user", "where is my order? Q, Q, nine, nine, nine.")] + [("get_order_status", {"order_id": "qgq999"},
                                                                       NF("qgq999"))] * 5
    assert run(CallGuard(), seq) == run(_shadow_guard(), seq)


def test_compact_responses_same_decisions_shorter_text():
    from mynah_v2.agent.call_guard import COMPACT
    seq = [("user", "where is my order? Q, Q, nine, nine, nine.")] + [("get_order_status", {"order_id": "qgq999"},
                                                                       NF("qgq999"))] * 6
    full, comp = CallGuard(), CallGuard(response_style="compact")
    rules = []
    for g in (full, comp):
        out = []
        for s_ in seq:
            if s_[0] == "user":
                g.on_user_text(s_[1])
                continue
            d = g.admit(s_[0], s_[1])
            out.append((d.rule, d.response))
            if d.allow:
                g.complete(d, s_[2])
        rules.append(out)
    assert [r for r, _ in rules[0]] == [r for r, _ in rules[1]]           # identical admission
    for (rule, f), (_, c) in zip(*rules):
        if rule:
            assert len(c) < 70 and len(c) < len(f) / 2 and json.loads(c)["next_step"]
    assert json.loads(rules[1][1][1]) == {"status": "already_done", "next_step": "Tell the caller."}
    assert set(COMPACT) >= {"repeat", "in_flight", "busy", "side_effect_done", "side_effect_unknown", "ungrounded",
                            "budget", "closed"}
    try:
        CallGuard(response_style="tiny")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_spoken_spans_whole_runs_only():
    from mynah_v2.agent.call_guard import spoken_spans
    assert "XYZ123" in spoken_spans("Hi, where is my order X Y Z one two three?")
    assert "WXY615" in spoken_spans("you check order Wxy six one five.")
    assert "20135" in spoken_spans("I'm Mario Rosie, postcode two zero one three five.")
    assert "99999999" in spoken_spans("I'm Anna, post code nine nine nine nine nine nine nine nine.")
    assert "9999" not in spoken_spans("post code nine nine nine nine nine nine nine nine.")
    g = _shadow_guard()
    g.on_user_text("morning. I'm anniverspost code nine nine nine nine nine nine nine nine.has my order shipped")
    g.admit("find_customer", {"postcode": "9999"})
    assert g.log[-1]["asr_slots"] == {"postcode": "mismatch"}            # m11: the old substring rule said match


def test_transfer_intent_labelled_sentences():
    """Synthetic labelled set (negations and look-alikes): transfer / negated / none."""
    from mynah_v2.agent.call_guard import summary
    cases = {
        "I'd like to speak to an operator, please.": "transfer",
        "Can you transfer me to a real person?": "transfer",
        "Put me through to someone who can help.": "transfer",
        "I want to talk to a human.": "transfer",
        "Connect me with a representative.": "transfer",
        "I don't want to speak to an operator, just tell me the status.": "negated",
        "No need to transfer me to anyone, I'm fine.": "none",
        "Please do not connect me to an agent.": "negated",
        "Never mind the operator, where is my order?": "none",
        "My order number is X Y Z one two three.": "none",
        "The courier said someone would call me.": "none",
        "Is a human going to deliver it or a drone?": "none",
    }
    for text, want in cases.items():
        g = _shadow_guard()
        g.on_user_text(text)
        intents = summary(g)["asr_shadow"]["intents"]
        got = "none" if not intents else ("negated" if intents[0]["negated"] else "transfer")
        assert got == want or (want == "none" and got == "negated"), (text, got)


def test_echo_refusal_restates_the_earlier_outcome():
    from mynah_v2.agent.call_guard import ECHO_NOTE
    g = CallGuard(response_style="echo")
    g.on_user_text("check order W x y six one five.")
    a = {"order_id": "WXYY615"}
    g.complete(g.admit("get_order_status", a), NF("WXYY615"))
    r = g.admit("get_order_status", a)
    assert r.rule == "repeat"
    assert json.loads(r.response) == {"order_id": "WXYY615", "found": False, "note": ECHO_NOTE}   # outcome kept
    assert len(r.response) < 100
    full = CallGuard()
    full.on_user_text("x")
    full.complete(full.admit("get_order_status", a), NF("WXYY615"))
    assert full.admit("get_order_status", a).rule == r.rule                                   # same admission
    for _ in range(3):                                   # past the budget: the compact budget text, still a response
        r = g.admit("get_order_status", a)
    assert json.loads(r.response)["next_step"]


def test_attach_session_guard_wraps_bound_tools_and_detach_clears():
    from types import SimpleNamespace

    from mynah_v2.agent.call_guard import attach_session_guard, detach_session_guard
    from mynah_v2.model.session_tools import SessionToolBinding
    ran = []
    reg = {"builtin": lambda a: "b"}
    binding = SessionToolBinding(reg)
    binding.bind({"get_order_status": lambda a: (ran.append(a), NF("qgq999"))[1]})
    eng = SimpleNamespace(session_tools=binding, stream_id=7)
    g = CallGuard(session_id="s")
    attach_session_guard(eng, g)
    from mynah_v2.agent.call_guard import session_guard
    assert session_guard(eng) is g and eng._v2_resume_by_stream == {7: "pad_eotr"}
    assert not hasattr(reg["builtin"], "__v2_call_guard__")           # only the session's own tools are guarded
    g.on_user_text("order Q Q nine")
    assert json.loads(reg["get_order_status"]({"order_id": "qgq999"}))["found"] is False
    assert json.loads(reg["get_order_status"]({"order_id": "qgq999"}))["status"] == "not_executed"
    assert len(ran) == 1
    attach_session_guard(eng, g)                                       # idempotent: not wrapped twice
    assert reg["get_order_status"].__wrapped__.__wrapped__ is not None
    s = detach_session_guard(eng)
    assert s["admitted"] == 1 and s["refused"] == {"repeat": 1}
    assert session_guard(eng) is None and eng._v2_resume_by_stream == {} and eng._v2_guard_summaries[7] == s
    binding.unbind()
    assert set(reg) == {"builtin"}


def test_session_attaches_the_guard_by_default_in_enhanced_only():
    from types import SimpleNamespace

    import torch

    from mynah_v2.model.session_tools import SessionToolBinding
    from mynah_v2.runtime.session import EnhancedOptions, Mode, Session
    from mynah_v2.tracing.events import Tracer

    class Eng:
        chunk_samples = 2560

        def __init__(self):
            self.pipeline = SimpleNamespace(tool_registry={})
            self.session_tools = SessionToolBinding(self.pipeline.tool_registry)
            self.session_tools.bind({"t": lambda a: "{}"})
            self.stream_id = 0
            self.opened = 0

        def set_session_guard(self, guard, resume="pad_eotr", tracer=None):
            from mynah_v2.agent.call_guard import attach_session_guard
            attach_session_guard(self, guard, resume=resume, tracer=tracer)

        def open(self, p):
            self.opened += 1

        def close(self):
            from mynah_v2.agent.call_guard import detach_session_guard
            detach_session_guard(self)

        def fc_active(self):
            return False

        def step(self, x, is_last=False):
            raise AssertionError

    e = Eng()
    s = Session(e, Tracer(None, session_id="x"), Mode.ENHANCED, EnhancedOptions())
    s.open(None)
    assert s.guard is not None and e._v2_resume_by_stream == {0: "pad_eotr"}
    s.close("cancel")
    assert s.guard_summary is not None and e._v2_resume_by_stream == {}
    e2 = Eng()
    r = Session(e2, Tracer(None, session_id="y"), Mode.REFERENCE, None)
    r.open(None)
    assert r.guard is None and not getattr(e2, "_v2_resume_by_stream", None)   # REFERENCE stays NVIDIA's behaviour
    e3 = Eng()
    o = Session(e3, Tracer(None, session_id="z"), Mode.ENHANCED, EnhancedOptions(call_guard=False))
    o.open(None)
    assert o.guard is None


# ------------------------------------------------------------------ resume "by_outcome" (EXPERIMENT)
def test_resume_by_outcome_follows_the_last_call():
    """(PAD, EOTR) after a positive result (a chain may follow), (PAD, PAD) after a negative or refused one or when the
    caller spoke after the call; (PAD, EOTR) while a call is in flight or before any call."""
    g = CallGuard()
    assert g.resume_contract() == "pad_eotr"                        # no call yet
    g.on_user_text("find me, Mario Rossi, 20135")
    d = g.admit("find_customer", {"full_name": "Mario Rossi"})
    assert g.resume_contract() == "pad_eotr"                        # in flight (phase-1 hand-back)
    g.complete(d, json.dumps({"status": "found", "customer_id": "C1002"}))
    assert g.resume_contract() == "pad_eotr"                        # positive: the chain stays possible
    d = g.admit("prepare_return", {"customer_id": "C1002"})
    g.complete(d, json.dumps({"status": "not_found"}))
    assert g.resume_contract() == "pad_pad"                         # negative
    r = g.admit("prepare_return", {"customer_id": "C1002"})
    assert not r.allow and g.resume_contract() == "pad_pad"         # refused repeat
    g.on_user_text("ok try order X Y Z")
    d = g.admit("get_order_status", {"order_id": "XYZ"})
    g.complete(d, json.dumps({"order_id": "XYZ", "found": True}))
    g.on_user_text("and also")                                      # the caller spoke after the call
    assert g.resume_contract() == "pad_pad"
    assert [x["contract"] for x in g.resume_log] == ["pad_eotr"] * 3 + ["pad_pad"] * 3
    from mynah_v2.agent.call_guard import summary
    assert summary(g)["resume"][-1]["contract"] == "pad_pad"
