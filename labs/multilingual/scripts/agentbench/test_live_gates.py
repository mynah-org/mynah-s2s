from live_gates import DOUBLE, ENGINE, LAB, UPSTREAM, m07_chain, resume_contract

ENG_RF = {"engine_fc_ctx_fix": {"enabled": True, "mode": "auto"}, "patched": False, "S2S_FIX_CONTEXT_PAD": "0",
          "S2S_FIX_FC_TOKEN_SYNC": "0"}


def test_resume_contract_labels():
    assert resume_contract(ENG_RF) == ENGINE
    assert resume_contract({"patched": True, "S2S_FIX_CONTEXT_PAD": "1", "S2S_FIX_FC_TOKEN_SYNC": "all"}) == LAB
    assert resume_contract({"patched": False, "S2S_FIX_CONTEXT_PAD": "1"}) == UPSTREAM      # flags inert on clean NeMo
    assert resume_contract({}) == UPSTREAM
    assert resume_contract(dict(ENG_RF, patched=True, S2S_FIX_FC_TOKEN_SYNC="all")) == DOUBLE


def rec(calls, audit, rf=ENG_RF, scenario="m07", lang="en"):
    return {"scenario": scenario, "lang": lang, "runtime_fix": rf, "calls": [{"name": n, "args": {}} for n in calls],
            "audit": audit}


FIND_OK = {"tool": "find_customer", "decision": "verified", "customer": "C1002"}


def test_m07_chain_pass_and_failures():
    g = m07_chain(rec(["find_customer", "prepare_return"], [FIND_OK, {"tool": "prepare_return", "decision": "prepared"}]))
    assert g["valid"] and g["pass"]
    lost = m07_chain(rec(["find_customer"], [FIND_OK]))
    assert lost["valid"] and not lost["pass"] and "second call was lost" in lost["reasons"][0]
    wrong_order = m07_chain(rec(["prepare_return", "find_customer"], [FIND_OK]))
    assert not wrong_order["pass"]
    unconf = m07_chain(rec(["find_customer", "prepare_return", "commit_return"],
                           [FIND_OK, {"tool": "commit_return", "decision": "submitted", "unconfirmed": True}]))
    assert not unconf["pass"]


def test_m07_chain_invalid_contract_never_passes():
    lab = {"patched": True, "S2S_FIX_CONTEXT_PAD": "1", "S2S_FIX_FC_TOKEN_SYNC": "all"}
    g = m07_chain(rec(["find_customer", "prepare_return"], [FIND_OK], rf=lab))
    assert not g["valid"] and not g["pass"]
    assert not m07_chain(rec(["find_customer", "prepare_return"], [FIND_OK], lang="it"))["valid"]


def test_extra_calls_reported_not_failed():
    g = m07_chain(rec(["find_customer", "prepare_return", "find_customer"], [FIND_OK]))
    assert g["pass"] and g["extra_calls_after_chain"] == ["find_customer"]
