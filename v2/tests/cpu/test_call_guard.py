"""CPU tests of the experimental pre-execution call guard, on the call sequences the live runs actually produced.

Sequences (calls, arguments, backend results, caller speech in between) are copied from the traces:
- engine FX1 with the (PAD, EOTR) resume: m02 11x get_order_status(qgq999) found=false; t14 9x (WXYY615); t13
  XYZ123 found, then the corrected DF246 found=false 7x; m01 two random-number calls, a caller turn apart;
  no caller speech between the repeated calls;
- EN m07 official: (PAD, EOTR) find_customer(Mario Rossi, 20135) verified C1002 -> prepare_return(C1002);
- EN m07 r3 (lead 7.5 s, (PAD, PAD)): find_customer without postcode, caller says the postcode,
  prepare_return(customer_id="Mario Rossi"), find_customer(Mario Rossi, 20135), prepare_return(C1002)."""
import json

from mynah_v2.agent.call_guard import PRODUCTIVE, UNPRODUCTIVE, CallGuard, default_outcome, guarded_executor

NF = lambda oid: json.dumps({"order_id": oid.upper(), "found": False})  # noqa: E731 - engine orders_tool shape
FOUND_XYZ = json.dumps({"order_id": "XYZ123", "found": True, "status": "shipped", "delivery_date": "November 12, 2026",
                        "tracking_code": "7319"})
VERIFIED = json.dumps({"status": "verified", "customer_id": "C1002", "first_name": "Mario",
                       "latest_order": {"item": "Lysa bookcase", "status": "delivered", "date": "2026-11-05"}})
PENDING = json.dumps({"status": "pending_confirmation", "pending_id": "P001", "item": "Lysa bookcase"})
VERIF_REQ = json.dumps({"status": "verification_required", "matching_names": 1, "next_step": "Ask for the postcode."})
TENANT = {"find_customer": {"properties": {"full_name": {"type": "string", "x-source": "caller"},
                                           "postcode": {"type": "string", "x-source": "caller"}}},
          "prepare_return": {"properties": {"item": {"type": "string"},
                                            "customer_id": {"type": "string", "x-source": "result"}}},
          "commit_return": {"properties": {"pending_id": {"type": "string", "x-source": "result"}}}}


def run(guard, seq):
    """seq: ("user", text) | (tool, args, backend_result). Returns the decisions in call order (rule or "run")."""
    out = []
    for s in seq:
        if s[0] == "user":
            guard.on_user_text(s[1])
            continue
        d = guard.check(s[0], s[1])
        out.append(d.rule or "run")
        if d.allow:
            guard.record(s[0], s[1], s[2])
        else:
            assert json.loads(d.response)["status"] == "not_executed"
    return out


def test_outcome_classifier():
    assert default_outcome(NF("qgq999")) == UNPRODUCTIVE
    assert default_outcome(VERIF_REQ) == UNPRODUCTIVE
    assert default_outcome(json.dumps({"error": "timeout", "message": "x"})) == UNPRODUCTIVE
    assert default_outcome(FOUND_XYZ) == PRODUCTIVE and default_outcome(VERIFIED) == PRODUCTIVE
    assert default_outcome(PENDING) == PRODUCTIVE
    assert default_outcome(json.dumps({"result": 1, "min": 1, "max": 10})) == PRODUCTIVE


def test_m02_loop_stops_at_first_repeat_and_next_turn_runs():
    g = CallGuard()
    a = {"order_id": "qgq999"}
    seq = [("user", "where is my order? Q, Q, nine, nine, nine.")] + [("get_order_status", a, NF("qgq999"))] * 11
    seq += [("user", "give me a random number"), ("generate_random_number", {"min": 1, "max": 10}, '{"result": 10}')]
    got = run(g, seq)
    assert got[:3] == ["run", "repeat", "repeat"]
    assert set(got[3:11]) == {"budget"}                  # past the budget nothing runs, whatever the model asks
    assert got[11] == "run"                              # the caller's next question is served normally
    assert sum(r == "run" for r in got[:11]) == 1        # the backend saw the order lookup once, not 11 times


def test_t14_repeat_carries_the_earlier_result():
    g = CallGuard()
    g.on_user_text("check order W x y six one five.")
    a = {"order_id": "WXYY615"}
    assert g.check("get_order_status", a).allow
    g.record("get_order_status", a, NF("WXYY615"))
    d = g.check("get_order_status", a)
    body = json.loads(d.response)
    assert d.rule == "repeat" and body["previous_result"] == {"order_id": "WXYY615", "found": False}
    assert "Do not call the tool again" in body["next_step"]


def test_t13_corrected_id_is_a_legitimate_new_call_then_its_repeat_is_blocked():
    g = CallGuard()
    got = run(g, [("user", "where is my order X Y Z one two three"), ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ),
                  ("user", "sorry, I meant D F G two four six"),
                  ("get_order_status", {"order_id": "DF246"}, NF("DF246")),
                  ("get_order_status", {"order_id": "DF246"}, NF("DF246"))])
    assert got == ["run", "run", "repeat"]


def test_retry_with_new_arguments_after_not_found_is_not_blocked_by_similarity():
    g = CallGuard()
    got = run(g, [("user", "order Q Q nine nine nine"), ("get_order_status", {"order_id": "qgq999"}, NF("qgq999")),
                  ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999"))])
    assert got == ["run", "run"]


def test_two_negative_results_without_new_input_exhaust_the_budget():
    g = CallGuard()
    got = run(g, [("user", "order Q Q nine nine nine"), ("get_order_status", {"order_id": "qgq999"}, NF("qgq999")),
                  ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999")),
                  ("get_order_status", {"order_id": "QGQ990"}, NF("QGQ990"))])
    assert got == ["run", "run", "budget"]


def test_same_call_after_the_caller_spoke_again_is_a_legitimate_retry():
    g = CallGuard()
    got = run(g, [("user", "order Q Q Q nine nine nine"), ("get_order_status", {"order_id": "QQQ999"}, NF("QQQ999")),
                  ("user", "yes, Q Q Q nine nine nine, please check again"),
                  ("get_order_status", {"order_id": "qqq-999"}, NF("QQQ999"))])
    assert got == ["run", "run"]


def test_repeat_of_a_productive_call_without_new_input_is_a_reopen():
    g = CallGuard()
    got = run(g, [("user", "X Y Z one two three"), ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ),
                  ("get_order_status", {"order_id": "XYZ123"}, FOUND_XYZ)])
    assert got == ["run", "repeat"]


def test_m01_two_calls_a_caller_turn_apart_both_run():
    g = CallGuard()
    got = run(g, [("user", "random number one to ten"), ("generate_random_number", {"min": 1, "max": 10}, '{"result": 1}'),
                  ("user", "another one, one to five"), ("generate_random_number", {"min": 1, "max": 5}, '{"result": 1}')])
    assert got == ["run", "run"]


def test_m07_chain_runs():
    g = CallGuard(TENANT)
    got = run(g, [("user", "I want to return the bookcase I bought, Mario Rossi, postcode two zero one three five"),
                  ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED),
                  ("prepare_return", {"item": "Lysa bookcase", "customer_id": "C1002"}, PENDING)])
    assert got == ["run", "run"]                         # the chained call needs NO new caller speech
    assert g.log[0]["caller_slots_unchecked"] == ["full_name", "postcode"]


def test_r3_invented_customer_id_is_rejected_and_the_chain_still_completes():
    g = CallGuard(TENANT)
    got = run(g, [("user", "I want to return the bookcase I bought. Mario Rosie."),
                  ("find_customer", {"full_name": "Mario Rossi"}, VERIF_REQ),
                  ("user", "postcode two zero one three five."),
                  ("prepare_return", {"item": "bookcase", "customer_id": "Mario Rossi"}, PENDING),
                  ("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}, VERIFIED),
                  ("prepare_return", {"item": "Lysa bookcase", "customer_id": "C1002"}, PENDING)])
    assert got == ["run", "ungrounded", "run", "run"]


def test_id_from_a_negative_result_is_not_grounded():
    g = CallGuard(TENANT)
    got = run(g, [("user", "return, Lindvik, two zero one three five"),
                  ("find_customer", {"full_name": "Lindvik", "postcode": "20135"},
                   json.dumps({"status": "not_found", "customer_id": "C9999"})),
                  ("prepare_return", {"item": "x", "customer_id": "C9999"}, PENDING)])
    assert got == ["run", "ungrounded"]


def test_guarded_executor_keeps_the_fc_contract():
    """The executor always returns a TOOL_RESPONSE for a parsed call (never None for a rejection: the model must not
    be left waiting); the backend runs only for allowed calls; unparsable text goes to the existing policy."""
    ran = []

    def backend(call_text):
        ran.append(call_text)
        return NF("qgq999")
    g = CallGuard()
    ex = guarded_executor(backend, g)
    ct = json.dumps([{"name": "get_order_status", "arguments": {"order_id": "qgq999"}}])
    assert json.loads(ex(ct))["found"] is False
    r = ex(ct)
    assert r is not None and json.loads(r)["status"] == "not_executed" and len(ran) == 1
    assert ex("not json") == NF("qgq999") and len(ran) == 2       # passthrough, guard not consulted
    multi = json.dumps([{"name": "get_order_status", "arguments": {"order_id": "qgq999"}},
                        {"name": "get_order_status", "arguments": {"order_id": "ABC456"}}])
    g2 = CallGuard()
    g2.on_user_text("x")
    g2.record("get_order_status", {"order_id": "qgq999"}, NF("qgq999"))
    parts = json.loads("[" + guarded_executor(lambda t: '{"found": true}', g2)(multi) + "]")
    assert parts[0]["status"] == "not_executed" and parts[1] == {"found": True}
