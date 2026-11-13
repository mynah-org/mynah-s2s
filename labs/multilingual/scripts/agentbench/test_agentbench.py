"""CPU tests for agentbench scoring (python3 test_agentbench.py)."""
import json

from agentbench import SPEC, args_match, cmd_validate, score_one

spec = json.load(open(SPEC))
S = {sc["id"]: sc for d in spec["domains"].values() for sc in d["scenarios"]}


def t(sid, calls, text):
    return score_one(S[sid], {"calls": calls, "agent_text": text})


def test_validate():
    assert cmd_validate(spec)


def test_args():
    assert args_match({"tracking_code": "NWP4821"}, {"tracking_code": "nwp-48 21"})
    assert not args_match({"tracking_code": "NWP4821"}, {"tracking_code": "NWP4822"})
    assert args_match({"date": "2026-11-20"}, {"date": "2026-11-20"})
    assert args_match({"reason": "*"}, {"reason": "damaged parcel"})
    assert not args_match({"reason": "*"}, {})
    assert args_match({"new_city": "Grenoble"}, {"new_city": "grenoble"})


def test_lookup_grounded():
    r = t("sh01", [{"name": "get_shipment_status", "args": {"tracking_code": "NWP4821"}}],
          "Your parcel is in transit and should arrive on November eighteenth, twenty twenty six.")
    assert r["pass"], r
    r = t("sh01", [{"name": "get_shipment_status", "args": {"tracking_code": "NWP4821"}}], "Your parcel is in transit.")
    assert not r["pass"] and not r["checks"]["says:2026-11-18"]


def test_negation_forbidden_tool():
    r = t("sh05", [{"name": "change_delivery_address", "args": {"tracking_code": "NWP7305", "new_city": "x"}},
                   {"name": "get_shipment_status", "args": {"tracking_code": "NWP7305"}}],
          "It should arrive on November twenty first.")
    assert not r["checks"]["no_forbidden_tool"] and not r["pass"]


def test_correction_uses_final_call():
    calls = [{"name": "get_shipment_status", "args": {"tracking_code": "NWP4822"}},
             {"name": "get_shipment_status", "args": {"tracking_code": "NWP4821"}}]
    assert t("sh04", calls, "It is in transit.")["checks"]["args_correct"]
    assert not t("sh04", calls[:1], "It is in transit.")["checks"]["args_correct"]


def test_clarify_and_out_of_scope():
    assert t("sh03", [], "Sure, can you give me the tracking code?")["checks"]["asks_for_slot"]
    assert not t("sh03", [{"name": "get_shipment_status", "args": {"tracking_code": "NWP0000"}}], "ok")["pass"]
    r = t("sh09", [], "Sorry, I can only help with parcels.")
    assert r["pass"] and r["review"]


def test_followup_single_call():
    c = [{"name": "get_shipment_status", "args": {"tracking_code": "NWP4821"}}]
    assert t("sh10", c, "It is in transit to Turin.")["pass"]
    assert not t("sh10", c * 2, "It is in transit to Turin.")["checks"]["max_calls"]


if __name__ == "__main__":
    for f in [test_validate, test_args, test_lookup_grounded, test_negation_forbidden_tool,
              test_correction_uses_final_call, test_clarify_and_out_of_scope, test_followup_single_call]:
        f()
    print("OK agentbench tests")
