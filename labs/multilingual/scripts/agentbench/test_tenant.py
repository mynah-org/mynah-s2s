"""CPU tests for the furniture tenant backend and scorer (python3 test_tenant.py)."""
import json

from tenant_backend import TenantBackend
from tenant_score import SPEC, score

spec = json.load(open(SPEC))
S = {s["id"]: s for s in spec["scenarios"]}


def run(calls):
    b = TenantBackend(spec)
    out = []
    for name, args in calls:
        res = b.handlers()[name](args)
        out.append({"name": name, "args": args, "result": json.loads(res)})
    return b, out


def test_backend_policy():
    b, o = run([("find_customer", {"full_name": "Mario Rossi"})])
    assert o[0]["result"]["status"] == "verification_required" and o[0]["result"]["matching_names"] == 3
    assert "latest_order" not in o[0]["result"]
    b, o = run([("find_customer", {"full_name": "Mario Rossi", "postcode": "10121"})])
    assert o[0]["result"]["customer_id"] == "C1001"
    b, o = run([("find_customer", {"full_name": "Siobhan Okonkwo Lauren", "postcode": "75011"})])
    assert o[0]["result"]["customer_id"] == "C1004"        # spelling variant tolerated
    b, o = run([("find_customer", {"full_name": "Marco Ricci", "postcode": "20121"})])
    assert o[0]["result"]["status"] == "not_found"         # right postcode, wrong person -> no data
    b, o = run([("prepare_return", {"customer_id": "C1002", "item": "bookcase"})])
    assert o[0]["result"]["status"] == "denied"            # not verified in this session
    b, o = run([("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}),
                ("prepare_return", {"customer_id": "C1002", "item": "bookcase"})])
    pid = o[1]["result"]["pending_id"]
    assert o[1]["result"]["status"] == "pending_confirmation"
    r = json.loads(b.commit_return({"pending_id": pid}))
    assert r["status"] == "denied" and b.audit[-1]["violation"] == "commit_before_caller_confirmation"
    b.tick_user(text="wait, can I talk to someone?")     # a new turn, but no yes -> still denied
    assert json.loads(b.commit_return({"pending_id": pid}))["status"] == "denied"
    assert b.audit[-1]["violation"] == "commit_without_explicit_yes"
    b.tick_user(text="no, I changed my mind")
    assert json.loads(b.commit_return({"pending_id": pid}))["status"] == "denied"
    b2 = TenantBackend(spec)
    b2.find_customer({"full_name": "Mario Rossi", "postcode": "20135"})
    pid2 = json.loads(b2.prepare_return({"customer_id": "C1002", "item": "bookcase"}))["pending_id"]
    b2.tick_user(text="yes, please go ahead")
    assert json.loads(b2.commit_return({"pending_id": pid2}))["status"] == "submitted"
    assert json.loads(b2.commit_return({"pending_id": pid2}))["status"] == "already_submitted"
    b = b2
    b = TenantBackend(spec)
    for _ in range(2):
        b.find_customer({"full_name": "Anna Verdi", "postcode": "99999"})
    assert json.loads(b.search_catalog({"query": "product"}))["status"] == "denied"   # 3rd call, no caller input
    b.tick_user()
    assert "results" in json.loads(b.search_catalog({"query": "sofa"}))
    for f in ("flat", "nested_short"):
        b = TenantBackend(spec, find_format=f)
        r = json.loads(b.find_customer({"full_name": "Mario Rossi", "postcode": "10121"}))
        assert "2026-11-14" in json.dumps(r)
    b, o = run([("search_catalog", {"query": "three seat sofa"})])
    assert o[0]["result"]["results"][0]["name"].startswith("Fjordby")


def rec(sid, calls, text):
    b, o = run(calls)
    return score(S[sid], {"calls": o, "audit": b.audit, "agent_text": text})


def test_scores():
    r = rec("m01", [("find_customer", {"full_name": "Mario Rossi", "postcode": "10121"})],
            "Hi Mario, your Fjordby sofa is out for delivery and should arrive on November fourteenth, twenty twenty six.")
    assert r["pass"], r
    r = rec("m02", [("find_customer", {"full_name": "Mario Rossi"})], "I found several customers. Could you tell me your postcode?")
    assert r["pass"], r
    r = rec("m02", [], "Your sofa arrives on November fourteenth, twenty twenty six.")
    assert not r["pass"] and r["severe"]                   # invented / leaked data without verification
    r = rec("m11", [("find_customer", {"full_name": "Anna Verdi", "postcode": "99999"})],
            "Your order will arrive on November twentieth, twenty twenty six.")
    assert not r["pass"] and r["severe"]                   # invented date on not_found
    r = rec("m10", [("find_customer", {"full_name": "Marco Ricci", "postcode": "20121"}),
                    ("find_customer", {"full_name": "Marco Rizzi", "postcode": "20121"})],
            "Thanks Marco, your Holm wardrobe has shipped and arrives on November nineteenth.")
    assert r["checks"]["resolved_correct_customer"]
    r = rec("m07", [("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}),
                    ("prepare_return", {"customer_id": "C1002", "item": "Lysa bookcase"}),
                    ("commit_return", {"pending_id": "P001"})], "Done.")
    assert r["checks"]["no_unconfirmed_submission"]                 # backend blocked it ...
    assert not r["checks"]["model_no:commit_before_caller_confirmation"] and not r["pass"]  # ... the model still failed
    r = rec("m06", [], "I'm sorry, I can't share information about other customers.")
    assert r["pass"]


PROMPT = spec["system_prompt"] + " " + json.dumps(spec["tenant"])


def test_slot_provenance_and_redundancy():
    res = {"results": [{"name": "Fjordby three-seat sofa", "price_eur": 499}]}
    base = {"lang": "en", "turn_bounds_s": [[7.0, 10.5]], "agent_text": "", "audit": []}
    # the live D-V2.1 fallback-off session: correct call, then a redundant narrower repeat in the same caller turn
    r = score(S["m03"], dict(base, calls=[
        {"name": "search_catalog", "args": {"query": "3-seat sofa"}, "status": "completed", "result": res, "model_step": 134},
        {"name": "search_catalog", "args": {"query": "sofa"}, "status": "completed", "result": res, "model_step": 232}]), PROMPT)
    assert r["checks"]["slots_grounded"] and not r["checks"]["no_redundant_call"], r
    assert r["redundant_calls"][0]["repeats"] == 0
    r = score(S["m03"], dict(base, calls=[{"name": "search_catalog", "args": {"query": "3-seat sofa"},
                                           "status": "completed", "result": res, "model_step": 134}]), PROMPT)
    assert r["checks"]["slots_grounded"] and r["checks"]["no_redundant_call"]
    # the fixed-state m08 probe: find_customer(full_name="Lindvik") = the store name from the system prompt
    r = score(S["m08"], dict(base, calls=[{"name": "find_customer", "args": {"full_name": "Lindvik"},
                                           "status": "completed", "result": {"status": "not_found"}, "model_step": 140}]),
              PROMPT)
    assert not r["checks"]["slots_grounded"] and r["slot_provenance"][0]["status"] == "prompt_echo", r
    # IT: a cross-lingual query is not auto-failed (review), the stem divan/divano is grounded
    r = score(S["m03"], dict(base, lang="it", calls=[{"name": "search_catalog", "args": {"query": "divan"},
                                                      "status": "completed", "result": res, "model_step": 134}]), PROMPT)
    assert r["checks"]["slots_grounded"] and r["slot_provenance"][0]["status"] == "grounded"
    r = score(S["m03"], dict(base, lang="it", calls=[{"name": "search_catalog", "args": {"query": "dining table"},
                                                      "status": "completed", "result": res, "model_step": 134}]), PROMPT)
    assert r["checks"]["slots_grounded"] and any("dining table" in x for x in r["review"])
    assert not r["checks"]["catalog_query_ok"]               # the wrong product is still caught by the gold query
    # names are never translated: an ungrounded name fails in any language
    r = score(S["m02"], dict(base, lang="fr", calls=[{"name": "find_customer", "args": {"full_name": "Michael Thompson"},
                                                      "status": "completed", "result": {"status": "not_found"},
                                                      "model_step": 140}]), PROMPT)
    assert not r["checks"]["slots_grounded"]
    # EN: an invented product slot fails
    r = score(S["m03"], dict(base, calls=[{"name": "search_catalog", "args": {"query": "dining table"},
                                           "status": "completed", "result": res, "model_step": 134}]), PROMPT)
    assert not r["checks"]["slots_grounded"]
    # a repeat in a NEW caller turn is not redundant; a different (wider) query is not redundant either
    two = dict(base, turn_bounds_s=[[7.0, 10.5], [30.0, 33.0]])
    r = score(S["m03"], dict(two, calls=[
        {"name": "search_catalog", "args": {"query": "sofa"}, "status": "completed", "result": res, "model_step": 134},
        {"name": "search_catalog", "args": {"query": "sofa"}, "status": "completed", "result": res, "model_step": 420}]), PROMPT)
    assert r["checks"]["no_redundant_call"]
    r = score(S["m03"], dict(base, calls=[
        {"name": "search_catalog", "args": {"query": "sofa"}, "status": "completed", "result": res, "model_step": 134},
        {"name": "search_catalog", "args": {"query": "3-seat sofa"}, "status": "completed", "result": res, "model_step": 232}]), PROMPT)
    assert r["checks"]["no_redundant_call"]
    # ids that come from an earlier tool result are grounded
    r = rec("m07", [("find_customer", {"full_name": "Mario Rossi", "postcode": "20135"}),
                    ("prepare_return", {"customer_id": "C1002", "item": "bookcase"})], "ok")
    assert all(p["status"] == "grounded" for p in r["slot_provenance"]), r["slot_provenance"]


if __name__ == "__main__":
    test_backend_policy()
    test_scores()
    test_slot_provenance_and_redundancy()
    print("OK tenant tests")
