"""D-V3 behavioural gates G1 (EN regression) and G2 (IT/FR) from live agentbench sessions (CPU).

Pre-registered in .work/multilingual/dv3-run.md. Per session (needs summary.json, observe.jsonl):
  opener      : a SPONTANEOUS agent BOS (Thinker token BOS before the RNNT turn logic, not forced) before the caller
                starts.
  talk_over   : agent BOS onsets (spontaneous or forced) inside the caller's turn, [u_start + 2, u_end). Continuing an
                utterance begun before the caller (opener spill-over) is not an onset.
  first_action: the scenario's correct first action, judged on the FIRST call only (post-tool rows are reported, never
                gated):
                  m02  find_customer with a grounded full_name and no invented postcode, or no call (asks)
                  m03  search_catalog whose query names the product (sofa/couch/divano/canape; translation allowed)
                  m07  find_customer with full_name and postcode both grounded
                  m08  no call with an invented / prompt-echo slot (asking first is the correct action)
G1 (EN, 4 sessions): opener 4/4, talk_over 0 sessions, first_action pass on m03, m07, m08.
G2 (IT/FR, 8 sessions): opener and no talk-over 8/8, first_action pass >= 4/8.
Usage: dv3_gates.py <session_dir> ... [--out f.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BOS = 1


def onsets(forcing: dict[int, dict]) -> list[tuple[int, str]]:
    return [(t, "spontaneous" if r["pre"] == BOS else "forced") for t, r in sorted(forcing.items()) if r["post"] == BOS]


def first_action(scn: str, lang: str, first_call: dict | None, prov: list[dict]) -> tuple[bool, str]:
    st = {p["arg"]: p["status"] for p in prov}
    if scn == "m02":
        if first_call is None:
            return True, "no call (asks)"
        ok = first_call["name"] == "find_customer" and st.get("full_name") == "grounded" and "postcode" not in st
        return ok, f"{first_call['name']} {first_call['args']} {st}"
    if scn == "m03":
        if first_call is None or first_call["name"] != "search_catalog":
            return False, f"first call {first_call and first_call['name']}"
        q = str(first_call["args"].get("query", "")).lower()
        return any(k in q for k in ("sofa", "couch", "divano", "canape", "canapé")), f"query={q!r}"
    if scn == "m07":
        ok = bool(first_call) and first_call["name"] == "find_customer" and \
            st.get("full_name") == "grounded" and st.get("postcode") == "grounded"
        return ok, f"{first_call and first_call['name']} {first_call and first_call['args']} {st}"
    if scn == "m08":
        bad = [a for a, s in st.items() if s in ("prompt_echo", "ungrounded")]
        return first_call is None or not bad, "no call" if first_call is None else f"{first_call['args']} {st}"
    return False, "unknown scenario"


def session(sd: Path) -> dict:
    from tenant_score import SPEC, slot_provenance
    summ = json.load(open(sd / "summary.json"))
    rows = [json.loads(x) for x in open(sd / "observe.jsonl")]
    forcing = {}
    for r in rows:
        if r.get("kind") == "forcing" and r.get("fn") == "_apply_rnnt_turn_taking":
            forcing.setdefault(r["t"], r)
    b0, b1 = summ["turn_bounds_s"][0]
    u0, u1 = int(math.floor(b0 / 0.08 + 1e-9)), int(math.ceil(b1 / 0.08 - 1e-9))
    ons = onsets(forcing)
    spec = json.load(open(SPEC))
    sc = {s["id"]: s for s in spec["scenarios"]}[summ["scenario"]]
    calls = summ.get("calls", [])
    fc = calls[0] if calls else None
    prov = []
    if fc:
        rec = dict(summ, calls=[fc])
        prov = [p for p in slot_provenance(sc, rec, spec.get("system_prompt", "") + " " + json.dumps(spec.get("tenant", {})))]
    ok, why = first_action(summ["scenario"], summ.get("lang", "en"), fc, prov)
    return {"session": sd.name, "scenario": summ["scenario"], "lang": summ.get("lang"), "variant": summ.get("variant"),
            "runtime_fix": {k: (summ.get("runtime_fix") or {}).get(k) for k in ("S2S_FIX_CONTEXT_PAD", "S2S_FIX_FC_TOKEN_SYNC")},
            "user_frames": [u0, u1],
            "opener": any(t < u0 and how == "spontaneous" for t, how in ons),
            "talk_over": [(t, how) for t, how in ons if u0 + 2 <= t < u1],
            "first_action_ok": ok, "first_action": why,
            "slot_provenance": [(p["arg"], p["value"], p["status"]) for p in prov],
            "post_tool_calls": [(c["name"], c["args"]) for c in calls[1:]],
            "agent_text": (summ.get("agent_text") or "")[:300]}


def gates(rows: list[dict]) -> dict:
    en = [r for r in rows if r["lang"] == "en"]
    ml = [r for r in rows if r["lang"] in ("it", "fr")]
    by = {r["scenario"]: r for r in en}
    g1 = {"opener": f"{sum(r['opener'] for r in en)}/{len(en)}",
          "talk_over_sessions": sum(bool(r["talk_over"]) for r in en),
          "first_action": {s: by[s]["first_action_ok"] if s in by else None for s in ("m03", "m07", "m08")}}
    g1["pass"] = bool(en) and all(r["opener"] for r in en) and g1["talk_over_sessions"] == 0 and \
        all(v is True for v in g1["first_action"].values())
    n_ok = sum(r["first_action_ok"] for r in ml)
    g2 = {"opener_and_listen": f"{sum(r['opener'] and not r['talk_over'] for r in ml)}/{len(ml)}",
          "first_action": f"{n_ok}/{len(ml)}"}
    g2["pass"] = bool(ml) and all(r["opener"] and not r["talk_over"] for r in ml) and n_ok >= 4
    return {"G1": g1, "G2": g2, "PASS": g1["pass"] and g2["pass"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [session(Path(d)) for d in a.session_dir]
    rep = {"sessions": rows, "gates": gates(rows)}
    for r in rows:
        print(json.dumps({k: r[k] for k in ("session", "opener", "talk_over", "first_action_ok", "first_action")},
                         ensure_ascii=False))
    print(json.dumps(rep["gates"]))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
