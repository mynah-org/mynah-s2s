"""T2 factorial across RESULT TYPES: the six resume contracts on several post-tool states, one engine load.

Owner GO 2026-11-12 (cap 5 GPU-min): the question is not which contract minimises SOTC but when a post-result SOTC is
a legitimate continuation and when it is an invented call. Per scenario the state (token history, call, tool result,
audio) is IDENTICAL across contracts and arms; only the perception front-end (the arm) and the token pair consumed at
the first main-loop frame after the FC (the contract) change.

Scenarios (history = the recorded consumed text ids of the named trace, teacher-forced for every arm):
  m03_empty  A's EN m03 history and call search_catalog("3-seat sofa"); the backend's REAL empty-result string
             (the handler output for a non-matching query), so the result type is the only change vs job 277.
  m08_vr     the EN m08 guided session (A's opener, then D-V2.1; job 269, fix-p2-on): call find_customer(full_name=
             "Lindvik") at 147, already an invented slot (the caller never says a name). The handler's
             verification_required result. Measures the POST-RESULT behaviour only, not the first wrong call.
  m07_chain  A's EN m07 history and call find_customer("Mario Rossi", "20135") -> verified; live A then chained
             prepare_return at 294 under (0, 0). The legitimate-chain control: a contract that kills this SOTC costs
             a real capability.

Contracts (text, function) at the first resume frame: 0/0 upstream, 12/12, 12/0, 0/22, 12/22, and 12/22s (EOTR
consumed in one inserted silent step with PAD text, the training layout, then the model's own outputs).

Pre-registered validity checks (recorded; a failing check invalidates only that scenario's numbers):
  V0 phase lengths: len(call) == trace `function.arguments.done steps`; len(resp)+1 == `tool.result.injected steps`
     and converted chars == its `chars` (m08, m07; m03_empty has no live reference);
  V1 A on its own trace (m03_empty, m07): text argmax == recorded on 0..s, SOTC at s, call argmax == recorded call;
  H5 live replication: m08 D-V2.1 under 12/22 (= live F1+F2) and m07 A under 0/0 (= live flags OFF) emit SOTC at the
     FIRST resume frame (live observe logits: m08 t=225 SOTC-PAD +1.25, m07 t=293 +4.0; the second call's
     function.arguments.done model_step is reported one frame later, 226 / 294, which job 278 wrongly pre-registered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parent.parent / "multilingual"))   # labpaths (MYNAH_HOME / MYNAH_LAB_HOME)
from labpaths import LAB_HOME, MYNAH_HOME  # noqa: E402
from t1_probe import BOS, EOS, EOTC, PAD, SOTC, Thinker, fc_call, frame_tokens, logp_rows, trace_session  # noqa: E402

EOTR = 22
CONTRACTS = "0/0,12/12,12/0,0/22,12/22,12/22s"
# first-cell wall estimates for the budget guard (job 278: m08 41 s = 225 forced steps + 5 re-call decodes, ~85 ms/step)
FIRST_CELL_S = {"m08_vr": 42.0, "m07_chain": 55.0, "m03_empty": 40.0, "m02_nf": 35.0, "t14_nf": 32.0,
                "t13_found": 35.0}
FX1 = MYNAH_HOME / "results/v2/fcfix/FX1-fix"
RES = LAB_HOME / "results/agentbench"
SCEN = {
    "m03_empty": {"clip": "m03_t0", "trace": RES / "m1-parity/vc_en_m03/trace.jsonl", "trace_arm": "A",
                  "tool": "search_catalog", "args": {"query": "3-seat sofa"}, "result_args": {"query": "zzz qqq"},
                  "expect_result": '{"results": [], "next_step": "Ask the caller which product they mean."}',
                  "h5": None},
    "m08_vr": {"clip": "m08_t0", "trace": RES / "fix-p2-on/v21_en_m08_guided/trace.jsonl", "trace_arm": None,
               "tool": "find_customer", "args": {"full_name": "Lindvik"}, "result_args": None,
               "expect_result": None, "h5": ("D-V2.1", "12/22", "find_customer", 0)},
    "m07_chain": {"clip": "m07_t0", "audio_root": "tts_m0_en", "trace": RES / "rb-en/vc_en_m07/trace.jsonl", "trace_arm": "A",
                  "tool": "find_customer", "args": {"full_name": "Mario Rossi", "postcode": "20135"},
                  "result_args": None, "expect_result": None, "h5": ("A", "0/0", "prepare_return", 0)},
    # engine FX1-fix m02 (official, ENHANCED, (PAD, EOTR) resume, NVIDIA default system message + get_order_status +
    # generate_random_number): get_order_status(qgq999) -> found=false, then the IDENTICAL call re-opened at the first
    # resume frame (11x live). Model frame t <-> audio frame t (no lead). During the FC the incoming chunks were
    # digital silence (rms 0) and were not buffered; the main loop resumed on silence, so the post-call audio equals
    # the clip's own silence after the request (frames s+1.. of the clip).
    "m02_nf": {"wav": MYNAH_HOME / "data/agent/fcmulti_v1/m02.wav", "lead_s": 0.0, "tail_s": 0.0, "max_s": 24.0,
               "trace": FX1 / "m02/trace.jsonl", "trace_arm": "A", "family": "engine",
               "tool": "get_order_status", "args": {"order_id": "qgq999"}, "result_args": None,
               "orders_db": FX1 / "orders.sqlite", "expect_result": '{"order_id": "QGQ999", "found": false}',
               "prompt_chars": 3366, "prompt_tokens": 789, "h5": ("A", "12/22", "get_order_status", 0),
               "live_outcome": "identical_recall"},
    # engine FX1-fix t14: get_order_status(WXYY615) -> found=false, identical call re-opened at the first resume frame
    # (9x live). The caller's follow-up was spoken DURING the FC: ENHANCED buffered it and fed the engine injected
    # silence (Session.feed_chunk), so the post-call audio is silence (post_fc_zeros), not the clip.
    "t14_nf": {"wav": MYNAH_HOME / "data/agent/corpus_v1/t14.wav", "lead_s": 0.0, "post_fc_zeros": True,
               "trace": FX1 / "t14/trace.jsonl", "trace_arm": "A", "family": "engine",
               "tool": "get_order_status", "args": {"order_id": "WXYY615"}, "result_args": None,
               "orders_db": FX1 / "orders.sqlite", "expect_result": '{"order_id": "WXYY615", "found": false}',
               "h5": None, "live_outcome": "identical_recall"},
    # engine FX1-fix t13, FIRST call: get_order_status(XYZ123) -> found (productive); live: the model REPLIED (no
    # re-call). The caller's correction was spoken during the FC (buffered, injected silence fed): post_fc_zeros.
    # Control: b2 must not change a productive reply. (t13's DF246 loop is its SECOND call: not reachable here.)
    "t13_found": {"wav": MYNAH_HOME / "data/agent/corpus_v1/t13.wav", "lead_s": 0.0, "post_fc_zeros": True,
                  "trace": FX1 / "t13/trace.jsonl", "trace_arm": "A", "family": "engine",
                  "tool": "get_order_status", "args": {"order_id": "XYZ123"}, "result_args": None,
                  "orders_db": FX1 / "orders.sqlite", "expect_result": None, "h5": None, "live_outcome": "reply"},
}
# live_outcome = what the official model did live at this state, for the baseline-faithfulness gate. None = no live
# reference for arm A: m08 teacher-forces D's live history (replay reference, job 278 A 12/22: +0.69 ->
# find_customer(Lindvik, 1234)); m03_empty is a counterfactual (empty result substituted on A's m03 state).
for _k in ("m03_empty", "m08_vr"):
    SCEN[_k]["live_outcome"] = None
SCEN["m07_chain"]["live_outcome"] = "chain"


# ----------------------------------------------------------------------------------------------- P1 / P3 variants
# (owner 2026-11-14, screen before any live A/B). A cell is (scenario, arm, variant):
#   base: tenant prompt + the backend's own result string (= job 278);
#   p1:   P1_RULE appended to the tenant system message: the two rules of NVIDIA's own default system message
#         (offline_voicechat_fc_infer.DEFAULT_SYSTEM_MESSAGE, the most in-distribution wording we have). The prompt is
#         prefilled from stream open and the whole history replayed in every cell (th.prefill() per cell, one Thinker
#         per prompt): never a snapshot reuse;
#   p3:   a negative / empty result without a string "status" gains one ("not_found" / "no_results"); every original
#         field is kept. Strings only: the wrapper's number conversion turns a JSON bool into the string "False"
#         (live FX1 m02: {"found": "False"}), so no bool / digit is added. Its tokens, the phase-2 length and the
#         first free frame are recomputed; the state before the result is unchanged (prefix sha recorded);
#   p1p3: both;
#   b2:   decoding probe, no prompt / result change: SOTC_BIAS added to the function SOTC logit on the free frames
#         after the result only (the post-result window), raw and biased margins both recorded. NeMo has no such knob
#         (fc_sotc_boost is read and logged, never applied); this measures whether one would separate the cases.
# The teacher-forced history was recorded under the original prompt: for p1 the cell measures the post-result
# decision on that history, not whether the model would have reached it (own_text_mismatch_0_s reports that).
VARIANTS = ("base", "p1", "p3", "p1p3", "b2", "typed", "app", "appnt")
# explicit application contract for tools (owner 2026-11-14, configurability workstream B), appended to the tenant
# system message or to NVIDIA's default message; "appnt" adds the text "/no_think" (a plain prompt string here: the
# VoiceChat checkpoint has no reasoning mode, this only tests whether the string changes behaviour)
APP_CONTRACT = ("Rules for tools: make at most one tool call at a time and wait for its result. You may call another "
                "tool when it needs the result of the previous one, for example an id the previous tool returned. "
                "Never repeat a call with the same arguments unless the caller gives new information. Never invent "
                "ids or required details; if one is missing or unclear, ask the caller. If a tool finds nothing, tell "
                "the caller and ask what to do next. Only help with the tasks your tools support; politely decline "
                "anything else.")
P1_RULE = ("Tool-call arguments must be values the user spoke. If a required argument is missing, ask the user; never "
           "guess. If a tool call fails or returns an error, do not retry the tool call for the same request.")
SOTC_BIAS = -2.0
NEGATIVE_STATUS = {"not_found", "no_results", "verification_required", "denied"}


def variant_prompt(system_prompt: str, variant: str) -> str:
    if variant in ("p1", "p1p3"):
        return system_prompt + " " + P1_RULE
    if variant in ("app", "appnt"):
        return system_prompt + "\n\n" + APP_CONTRACT + (" /no_think" if variant == "appnt" else "")
    return system_prompt


def is_negative(r) -> bool:
    return isinstance(r, dict) and ("error" in r or r.get("found") is False or r.get("results") == []
                                    or str(r.get("status", "")).lower() in NEGATIVE_STATUS)


def structure_result(result: str) -> str:
    """P3 (minimal): a negative / empty result with no string status gains one; every original field is kept.
    Results that already carry a status, and productive results, are returned byte-identical."""
    try:
        r = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return result
    if not is_negative(r) or isinstance(r.get("status"), str):
        return result
    return json.dumps(dict(r, status="no_results" if r.get("results") == [] else "not_found"))


def result_variant(variant: str) -> str:
    return "p3" if variant in ("p3", "p1p3") else "typed" if variant == "typed" else "orig"


def typed_convert(wrapped: str, conv) -> str:
    """The wrapper's TOOL_RESPONSE number conversion (_convert_tool_response_nums_to_text: same tag regex, json.dumps
    of the converted JSON) with ONE difference: JSON bools stay bools. The wrapper's _convert_nums_in_json treats a
    bool as an int, inflect fails and it falls back to str(): {"found": false} reaches the model as "False" (bug,
    live FX1 m02). ``conv`` = the wrapper's _convert_nums_in_json, applied to every non-bool leaf."""
    import re

    def walk(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return conv(x)

    def rep(m):
        try:
            return m.group(1) + json.dumps(walk(json.loads(m.group(2)))) + m.group(3)
        except (json.JSONDecodeError, TypeError):
            return m.group(0)
    return re.compile(r'(<TOOL_RESPONSE>)(.*?)(</TOOL_RESPONSE>)', re.DOTALL).sub(rep, wrapped)


def gate(sc: str, cell: dict, call_text: str, contract: str = "12/22") -> dict:
    """Pre-registered per-cell decision (owner 2026-11-14). m02: the baseline must re-open the IDENTICAL call at the
    first resume frame (live behaviour); a variant passes if it replies with no identical re-call in the window.
    m07: every cell must keep the legitimate chained prepare_return(C1002)."""
    o = cell["contracts"][contract]["outcome"]
    txt = o.get("new_call_text") or ""
    same = o.get("first") == "new_call" and "".join(txt.split()) == "".join(call_text.split())
    if SCEN.get(sc, {}).get("live_outcome") in ("identical_recall", "reply") or sc == "m02_nf":
        return {"identical_recall": same, "reply": o.get("first") == "reply", "recall_at": o.get("new_call_sotc"),
                "other_call": o.get("first") == "new_call" and not same, "variant_pass": o.get("first") == "reply"}
    if sc == "m07_chain":
        keep = o.get("first") == "new_call" and "prepare_return" in txt and "C1002" in txt
        return {"chain_kept": keep, "variant_pass": keep}
    return {}


def prompt_variant(variant: str) -> str:
    return "p1" if variant in ("p1", "p1p3") else variant if variant in ("app", "appnt") else "orig"


def parse_plan(plan: str, scenarios, arms, variants) -> list[tuple[str, str, str]]:
    """'scenario:arm[:variant],...' (variant default base) or every scenario x arm x variant, scenario-major."""
    if plan:
        out = []
        for c in plan.split(","):
            p = c.split(":")
            out.append((p[0], p[1], p[2] if len(p) > 2 else "base"))
        return out
    return [(sc, arm, v) for sc in scenarios for arm in arms for v in variants]


# ----------------------------------------------------------------------------------------------- pure helpers
def build_schedule(hist: list[int], s: int, call: list[int], resp: list[int]) -> list[dict]:
    """Same layout as t2_posttool.schedule (positions 0..s main, s+1.. FC phase 1, then phase 2 forced)."""
    from t2_posttool import schedule
    return schedule(hist, s, call, resp)


def first_free(s: int, call: list[int], resp: list[int]) -> int:
    """First main-loop frame after the FC: s + phase-1 steps + phase-2 steps (len(resp) + 1) + 1."""
    return s + len(call) + len(resp) + 2


def parse_pair(pr: str) -> tuple[int, int, bool]:
    silent = pr.endswith("s")
    ta, fa = (int(x) for x in pr.rstrip("s").split("/"))
    return ta, fa, silent


def trace_lengths(sess: list[dict]) -> dict:
    """Phase lengths and result chars of the FIRST FC in a trace."""
    d = {}
    for e in sess:
        if e["ev"] == "function.arguments.done" and "call_steps" not in d:
            d["call_steps"] = e.get("steps")
        elif e["ev"] == "tool.result.injected" and "result_steps" not in d:
            d["result_steps"], d["result_chars"] = e.get("steps"), e.get("chars")
        elif e["ev"] == "tool.result.consumed" and "consumed" not in d:
            d["consumed"] = e.get("model_step")
    return d


def margin(lp: torch.Tensor, k: int) -> float:
    return round(float(lp[k] - torch.cat([lp[:k], lp[k + 1:]]).max()), 3)


def frame_row(pos: int, tl, fl) -> dict:
    lt, lf = logp_rows(tl), logp_rows(fl)
    return {"pos": pos, "text": int(lt.argmax()), "func": int(lf.argmax()), "sotc_margin": margin(lf, SOTC),
            "sotc_minus_pad": round(float(lf[SOTC] - lf[PAD]), 3), "bos_margin": margin(lt, BOS),
            "lp_SOTC": round(float(lf[SOTC]), 3), "lp_func_PAD": round(float(lf[PAD]), 3),
            "func_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lf.topk(2))],
            "text_top2": [[int(i), round(float(v), 3)] for v, i in zip(*lt.topk(2))]}


def summarise(rows: list[dict]) -> dict:
    bos = next((r["pos"] for r in rows if r["text"] == BOS), None)
    sotc = next((r["pos"] for r in rows if r["func"] == SOTC), None)
    first = None if bos is None and sotc is None else ("reply" if sotc is None or (bos is not None and bos < sotc)
                                                       else "new_call")
    ids = [r["text"] for r in rows if bos is not None and r["pos"] >= bos and r["text"] not in (PAD, BOS, EOS)]
    return {"first": first, "reply_bos": bos, "new_call_sotc": sotc, "reply_ids": ids,
            "max_sotc_margin": max((r["sotc_margin"] for r in rows), default=None)}


# ----------------------------------------------------------------------------------------------- GPU
def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(LAB_HOME / "data/agentbench"))
    ap.add_argument("--scenarios", default="m08_vr,m03_empty,m07_chain")
    ap.add_argument("--arms", default="A,D-V2.1,C-ridge")
    ap.add_argument("--contracts", default=CONTRACTS)
    ap.add_argument("--n-free", type=int, default=40)   # a late re-call must stay visible (16 was too short)
    ap.add_argument("--call-cap", type=int, default=48)
    ap.add_argument("--tail-s", type=float, default=4.0)
    ap.add_argument("--lead-s", type=float, default=7.0)
    ap.add_argument("--budget-s", type=float, default=265.0)
    ap.add_argument("--stop-on-unfaithful", type=int, default=1, help="1: stop the run at an unfaithful baseline; "
                    "0: mark it (its cells are not usable) and continue with the other states")
    ap.add_argument("--variants", default="base", help="comma list of " + "/".join(VARIANTS))
    ap.add_argument("--plan", default="", help="ordered 'scenario:arm[:variant],...' (priority order under the budget "
                                               "guard); default: every scenario x arm x variant, scenario-major")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    if (out / "t2f.json").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "multilingual"))
    from e2e_qa import apply_variant, parse_variant
    from m0_runner import load_16k, render_prompt
    from perception_swap import FrontEnd, load_nemo_dir
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import iter_chunks
    rep = {"scenarios": {}, "cells": [], "skipped": [], "timing_s": {}}

    def save():
        rep["elapsed_s"] = round(time.monotonic() - T0, 1)
        (out / "t2f.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    torch.manual_seed(1234)
    t = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}), llm_cache=True)
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):     # decode_audio=False harness fix (decision D7)
        return _orig_log(frames, torch.zeros(1, 0) if audio_wave is None else audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    rep["timing_s"]["engine_load"] = round(time.monotonic() - t, 1)
    spec = json.load(open(HERE / "tenant_furniture.json"))
    reg, be0 = ToolRegistry(), TenantBackend(spec)
    for tl in spec["tools"]:
        reg.register(tl["name"], be0.handlers()[tl["name"]], tl["description"], tl["parameters"])
    variants = a.variants.split(",")
    assert all(v in VARIANTS for v in variants), variants
    plan = parse_plan(a.plan, a.scenarios.split(","), a.arms.split(","), variants)
    fam = lambda sc: SCEN[sc].get("family", "tenant")       # noqa: E731
    assert not any(fam(sc) == "engine" and prompt_variant(v) == "p1" for sc, _, v in plan), \
        "the engine prompt IS NVIDIA's default message (P1 rules already in it): no p1 on engine scenarios"
    import offline_voicechat_fc_infer as fcmod
    from mynah_v2.agent.tools import ORDERS_SCHEMA, orders_tool
    reg_e = ToolRegistry()          # = run_stream --tools orders,random (FX1): same description, schema, builtin tool
    reg_e.register("get_order_status", lambda a_: "", "Get the status, delivery date and tracking code of a customer "
                   "order by its order id.", ORDERS_SCHEMA, ack_message="Let me check that order for you.")
    reg_e.schemas += fcmod.DEFAULT_TOOLS
    prompts, ths = {}, {}
    for f_, pv in sorted({(fam(sc), prompt_variant(v)) for sc, _, v in plan} | {("tenant", "orig")}):
        pvar = pv if pv in ("p1", "app", "appnt") else "base"
        prompts[(f_, pv)] = (render_prompt(reg.schemas, variant_prompt(spec["system_prompt"], pvar)) if f_ == "tenant"
                             else render_prompt(reg_e.schemas, variant_prompt(fcmod.DEFAULT_SYSTEM_MESSAGE, pvar)))
        ths[(f_, pv)] = Thinker(eng, prompts[(f_, pv)])   # own prefill embeddings; cache rebuilt per cell (prefill())
    assert len(prompts[("tenant", "orig")]) == 2738, len(prompts[("tenant", "orig")])
    rep["prompt_text_tail"] = {f"{f_}:{pv}": p[-700:] for (f_, pv), p in prompts.items()}
    if ("engine", "orig") in prompts:                      # = the live FX1 prompt (session.opened / NeMo log)
        assert len(prompts[("engine", "orig")]) == 3366 and ths[("engine", "orig")].P == 789, (
            len(prompts[("engine", "orig")]), ths[("engine", "orig")].P)
    rep["prompts"] = {f"{f_}:{pv}": {"chars": len(p), "sha": hashlib.sha256(p.encode()).hexdigest()[:12],
                                     "tokens": ths[(f_, pv)].P} for (f_, pv), p in prompts.items()}
    th = ths[("tenant", "orig")]
    w = eng.wrapper

    # ---- scenario states (CPU-side: tokens, schedule, checks)
    states = {}
    need_states = {(sc_, result_variant(v)) for sc_, _, v in plan} | {(sc_, "orig") for sc_, _, _ in plan}
    for sc, rv in sorted(need_states, key=lambda x: (x[0], x[1] != "orig")):
        cfg = SCEN[sc]
        sess = trace_session(str(cfg["trace"]))
        rec, (s, fids) = frame_tokens(sess), fc_call(sess)
        assert fids[0] == SOTC and fids[-1] == EOTC, sc
        call = fids[1:]
        if fam(sc) == "engine":
            result = orders_tool(cfg["orders_db"])(cfg["result_args"] or cfg["args"])   # FX1's own db, read only
        else:
            be = TenantBackend(spec)                  # fresh backend: no call budget carried between scenarios
            result = be.handlers()[cfg["tool"]](cfg["result_args"] or cfg["args"])
        if cfg["expect_result"] is not None:
            assert result == cfg["expect_result"], (sc, result)
        if rv == "p3":
            result = structure_result(result)
        wrapped = f"<TOOL_RESPONSE>[{result}]</TOOL_RESPONSE>"
        conv = w._convert_tool_response_nums_to_text(wrapped) if w._fc_convert_num_to_text else wrapped
        if rv == "typed":                     # bool kept as a JSON bool (bug fix probe, NOT p3)
            conv = typed_convert(wrapped, w._convert_nums_in_json)
            resp = list(w.tokenizer.text_to_ids(conv))
        else:
            resp = list(w._build_fc_response_tokens(wrapped))
        tlen = trace_lengths(sess)
        call_text = ths[(fam(sc), "orig")].decode(call[:-1])
        v0 = {"call_steps": [len(call), tlen.get("call_steps")], "result_steps": [len(resp) + 1, tlen.get("result_steps")],
              "result_chars": [len(conv), tlen.get("result_chars")],
              "call_has_args": all(str(v) in call_text for v in cfg["args"].values())}
        v0["pass"] = (v0["call_steps"][0] == v0["call_steps"][1] and v0["call_has_args"] and
                      (sc == "m03_empty" or (v0["result_steps"][0] == v0["result_steps"][1] and
                                             v0["result_chars"][0] == v0["result_chars"][1])))
        hist = [rec.get(t, PAD) for t in range(s + 1)]
        sched = build_schedule(hist, s, call, resp)
        ff = first_free(s, call, resp)
        if sc != "m03_empty":
            v0["first_free_vs_trace"] = [ff, (tlen.get("consumed") or -2) + 1]
            v0["pass"] = v0["pass"] and ff == (tlen.get("consumed") or -2) + 1
        if rv != "orig":                       # lengths differ from the trace by design: check against the orig state
            o = states[(sc, "orig")] if (sc, "orig") in states else None
            v0 = {"call_steps_equal_orig": o is not None and call == o["call"], "hist_equal_orig":
                  o is not None and hist == o["hist"], "extra_result_tokens": len(resp) - (len(o["resp"]) if o else 0),
                  "unchanged_result": o is not None and resp == o["resp"]}
            v0["pass"] = v0["call_steps_equal_orig"] and v0["hist_equal_orig"]
        prefix_sha = hashlib.sha256(json.dumps([[r["agent_in"], r["func_in"], r.get("forced_func"), r["audio"][0]]
                                                for r in sched if r["kind"] in ("main", "fc1")]).encode()).hexdigest()[:12]
        if rv != "orig":
            v0["prefix_equal_orig"] = o is not None and prefix_sha == o["prefix_sha"]
            v0["pass"] = v0["pass"] and v0["prefix_equal_orig"]
        states[(sc, rv)] = {"prefix_sha": prefix_sha, "s": s, "call": call, "resp": resp, "hist": hist, "rec": rec,
                            "sched": sched, "ff": ff,
                      "sha": hashlib.sha256(json.dumps([[r["agent_in"], r["func_in"], r.get("forced_func"),
                                                         r["audio"][0]] for r in sched]).encode()).hexdigest()[:12]}
        rep["scenarios"][f"{sc}:{rv}"] = {"s": s, "call_text": call_text, "result": result, "converted": conv,
                                "n_resp": len(resp), "first_free": ff, "V0": v0, "schedule_sha": states[(sc, rv)]["sha"],
                                "trace": str(cfg["trace"])}
        print("SCEN", sc, rv, json.dumps(rep["scenarios"][f"{sc}:{rv}"]["V0"]), call_text[:80], flush=True)
    save()

    # p3 on a productive result is byte-identical to base (structure_result leaves it): run it once, as base
    dup = [(sc, arm, v) for sc, arm, v in plan if result_variant(v) == "p3" and states[(sc, "p3")]["resp"] ==
           states[(sc, "orig")]["resp"] and (sc, arm, "p1" if v == "p1p3" else "base") in plan]
    rep["identical_by_construction"] = [list(p) for p in dup]
    plan = [p for p in plan if p not in dup]
    rep["plan"] = [list(p) for p in plan]
    ARMS = {"A": "vc",
            "C-ridge": "enc=ml;att=56;rnnt=ml;ln=orig;map={R}/maps/ridge_ml_out.pt",
            "D-V2.1": "enc=ml;att=56;rnnt=ml;ln={R}/v21/distill/v21/encoder_ln_best.pt;adapter={R}/v21/distill/v21/adapter_best.pt"}
    R = str(LAB_HOME / "results")
    fe = FrontEnd(eng)
    fe.snapshot()
    ckpt = {"en": load_nemo_dir(LAB_HOME / "ckpt/asr-en"),
            "ml": load_nemo_dir(LAB_HOME / "ckpt/asr-ml")}
    def clip_path(sc):   # the audio root of the LIVE session each trace comes from (m07: tts_m0_en, others tts_m1)
        if "wav" in SCEN[sc]:
            return SCEN[sc]["wav"]
        root = Path(a.data_root) / SCEN[sc].get("audio_root", "tts_m1")
        man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(root / "manifest.jsonl"))}
        return root / man[(SCEN[sc]["clip"], "en", "vivian")]["path"]
    feats, sils = {}, {}

    def capture(arm):
        """Apply the arm's front-end once and capture its user-audio embeddings for every scenario clip (the Thinker
        steps afterwards use only these tensors and the arm's FC silence embedding)."""
        t = time.monotonic()
        apply_variant(fe, parse_variant(ARMS[arm].format(R=R)), ckpt)
        sils[arm] = th.silence().detach().clone()
        orig_if, orig_fu = w.model_llm_interface, w.fusion_module.forward

        def stub(*args, **kw):
            pad = torch.full((1,), PAD, dtype=torch.long, device=th.dev)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad,
                    "cache": kw.get("cache")}
        for sc in sorted({sc_ for sc_, arm_, _ in plan if arm_ == arm}):
            x = load_16k(clip_path(sc))
            if SCEN[sc].get("max_s"):
                x = x[:int(SCEN[sc]["max_s"] * 16000)]
            lead, tail_s = SCEN[sc].get("lead_s", a.lead_s), SCEN[sc].get("tail_s", a.tail_s)
            if SCEN[sc].get("post_fc_zeros"):     # what the live engine got after the call frame: injected silence
                x = x[:(states[(sc, "orig")]["s"] + 1) * 1280]
                tail_s = (a.n_free + 12) * 0.08
            wav = torch.cat([torch.zeros(int(lead * 16000)), x, torch.zeros(int(tail_s * 16000))])
            fx = []

            def fuse(*args, **kw):
                ua = kw.get("user_audio_embeds")
                if ua is not None and ua.dim() == 3 and ua.shape[1] == 1:
                    fx.append(ua[0, 0].detach().clone())
                return orig_fu(*args, **kw)
            w.model_llm_interface, w.fusion_module.forward = stub, fuse
            try:
                eng.open(None)
                chunks = list(iter_chunks(wav, eng.chunk_samples))
                for i, c in enumerate(chunks):
                    eng.step(c, is_last=(i == len(chunks) - 1))
                eng.close()
            finally:
                w.model_llm_interface, w.fusion_module.forward = orig_if, orig_fu
            feats[(arm, sc)] = torch.stack(fx)
            need = states[(sc, "orig")]["s"] + 1 + a.n_free + 2
            assert feats[(arm, sc)].shape[0] >= need, (arm, sc, feats[(arm, sc)].shape, need)
        rep["timing_s"].setdefault("capture", []).append([arm, round(time.monotonic() - t, 1)])

    def decode_call(th, cache, sil):
        prev, ids = SOTC, []
        for _ in range(a.call_cap):
            _, fl, cache = th.step(cache, sil, PAD, prev)
            prev = int(fl.argmax())
            ids.append(prev)
            if prev == EOTC:
                break
        return ids, th.decode([i for i in ids if i != EOTC])

    def run_cell(arm, sc, v):
        t = time.monotonic()
        th = ths[(fam(sc), prompt_variant(v))]
        st, f, sil = states[(sc, result_variant(v))], feats[(arm, sc)], sils[arm]
        s, sched = st["s"], st["sched"]
        cache = th.prefill()
        main_pred, fc1, agree, nat, lp_eotr = [], [], [], None, None
        for r in sched:
            audio = f[r["audio"][1]] if r["audio"][0] == "feat" else sil
            tl, fl, cache = th.step(cache, audio, r["agent_in"], r["func_in"])
            if r["kind"] == "main":
                main_pred.append((int(tl.argmax()), int(fl.argmax())))
            elif r["kind"] == "fc1":
                fc1.append(int(fl.argmax()))
            elif r["forced_func"] is not None:
                agree.append(int(fl.argmax()) == r["forced_func"])
            else:
                lf = logp_rows(fl)
                nat, lp_eotr = int(lf.argmax()), round(float(lf[EOTR]), 3)
        hm = [t_ for t_ in range(s + 1) if main_pred[t_][0] != st["rec"].get(t_, PAD)]
        res = {"arm": arm, "scenario": sc, "variant": v, "prompt": rep["prompts"][f"{fam(sc)}:{prompt_variant(v)}"],
               "n_resp": len(st["resp"]), "schedule_sha": st["sha"], "nat_last": nat, "lp_EOTR_last": lp_eotr,
               "own_sotc_at_s": main_pred[s][1] == SOTC, "own_func_pad_before_s": all(p[1] == PAD for p in main_pred[:s]),
               "own_text_mismatch_0_s": len(hm), "fc1_match": [sum(p == c for p, c in zip(fc1, st["call"])), len(st["call"])],
               "fc2_forced_agree": [sum(agree), len(agree)], "contracts": {}}
        if SCEN[sc]["trace_arm"] == arm and v == "base":
            res["V1"] = {"text_mismatch": hm[:20], "sotc_at_s": main_pred[s][1] == SOTC,
                         "call_match": fc1 == st["call"], "pass": not hm and main_pred[s][1] == SOTC and fc1 == st["call"]}
        tail = f[s + 1:s + 1 + a.n_free]        # continuation: the live main loop resumes on frame s+1 of the audio
        for pr in a.contracts.split(","):
            biased = pr.endswith("b")               # "12/22b" = the same contract + the b2 bias, same history/cache
            ta, fa, silent = parse_pair(pr.rstrip("b"))
            bc, pre = th.clone(cache), None
            if silent:
                tl0, fl0, bc = th.step(bc, sil, ta, fa)
                pre = frame_row(-1, tl0, fl0)
                ta, fa = pre["text"], pre["func"]
            rows = []
            # job 278: when the silent EOTR step itself emits SOTC the FC starts there; do not feed the call tokens
            # through real-audio main frames (decode the call on FC phase-1 inputs instead)
            for k in range(0 if (pre is not None and pre["func"] == SOTC) else a.n_free):
                tl, fl, bc = th.step(bc, tail[k], ta, fa)
                raw = None
                if v == "b2" or biased:               # decoding probe: post-result window only
                    raw = frame_row(st["ff"] + k, tl, fl)["sotc_minus_pad"]
                    fl = fl.clone()
                    fl[SOTC] += SOTC_BIAS
                rows.append(frame_row(st["ff"] + k, tl, fl))
                if raw is not None:
                    rows[-1]["raw_sotc_minus_pad"] = raw
                ta, fa = rows[-1]["text"], rows[-1]["func"]
                if fa == SOTC:
                    break
            o = summarise(rows) if rows else {"first": None, "reply_bos": None, "new_call_sotc": None, "reply_ids": [],
                                              "max_sotc_margin": None}
            if pre is not None and pre["func"] == SOTC:
                o["first"], o["new_call_sotc"] = "new_call", "silent_step"
            if o["first"] == "new_call":
                o["new_call_ids"], o["new_call_text"] = decode_call(th, bc, sil)
            o["reply_text"] = th.decode(o.pop("reply_ids"))
            res["contracts"][pr] = {"silent_step": pre, "first_frame": rows[0] if rows else pre, "frames": rows[:6],
                                    "outcome": o}
        res["s"] = round(time.monotonic() - t, 1)
        return res

    costs = {}
    for i, (sc, arm, v) in enumerate(plan):
        need_cap = arm not in sils
        n_clips = len({sc_ for sc_, arm_, _ in plan if arm_ == arm})
        est = costs.get(sc, FIRST_CELL_S.get(sc, 45.0)) * 1.1 + (8.0 * n_clips if need_cap else 0.0)
        if time.monotonic() - T0 + est > a.budget_s:
            rep["skipped"] = [list(p) for p in plan[i:]]
            print("BUDGET: skipping", rep["skipped"], flush=True)
            break
        if need_cap:
            capture(arm)
        r = run_cell(arm, sc, v)
        costs[sc] = max(costs.get(sc, 0.0), r["s"])
        h5 = SCEN[sc]["h5"]
        if h5 and h5[0] == arm and v == "base":
            c = r["contracts"].get(h5[1], {})
            o = c.get("outcome", {})
            r["H5"] = {"contract": h5[1], "new_call_sotc": o.get("new_call_sotc"), "expect_sotc": states[(sc, "orig")]["ff"] + SCEN[sc]["h5"][3],
                       "new_call_text": o.get("new_call_text"), "expect_tool": h5[2],
                       "pass": o.get("new_call_sotc") == states[(sc, "orig")]["ff"] + SCEN[sc]["h5"][3] and h5[2] in (o.get("new_call_text") or "")}
        rep["cells"].append(r)
        stop = None
        lo = SCEN[sc].get("live_outcome")
        if "12/22" in r["contracts"] and (lo or sc in ("m02_nf", "m07_chain")):
            g = r["gate"] = gate(sc, r, rep["scenarios"][f"{sc}:orig"]["call_text"])
            if "12/22b" in r["contracts"]:
                r["gate_b2"] = gate(sc, r, rep["scenarios"][f"{sc}:orig"]["call_text"], "12/22b")
            if v == "base":            # the baseline must reproduce the live behaviour, or its cells are not usable
                ff = states[(sc, "orig")]["ff"]
                ok = bool(rep["scenarios"][f"{sc}:orig"]["V0"]["pass"]) and bool(r.get("V1", {}).get("pass"))
                ok = ok and (g["identical_recall"] and g["recall_at"] in (ff, ff + 1) if lo == "identical_recall"
                             else g["reply"] if lo == "reply" else g["chain_kept"])
                r["baseline_faithful"] = ok
                if not ok and a.stop_on_unfaithful:
                    stop = f"{sc} baseline does not reproduce the live behaviour"
            elif sc == "m07_chain" and not g["chain_kept"]:
                stop = f"m07 lost prepare_return(C1002) under {v}"
        if stop:
            rep["stop"] = stop
            rep["skipped"] = [list(p_) for p_ in plan[i + 1:]]
            save()
            print("STOP:", stop, flush=True)
            break
        save()
        print(f"{sc:10} {arm:8} {v:5} V1={r.get('V1', {}).get('pass')} H5={r.get('H5', {}).get('pass')} ({r['s']}s)", flush=True)
        for pr, c in r["contracts"].items():
            o = c["outcome"]
            print(f"   {pr:7} m0={c['first_frame']['sotc_margin']:+.2f} first={o['first']} sotc={o['new_call_sotc']} "
                  f"call={o.get('new_call_text', '')[:70]!r} reply={o['reply_text'][:70]!r}", flush=True)
    fe.restore()
    save()
    print("DONE", round(time.monotonic() - T0, 1), flush=True)


if __name__ == "__main__":
    main()
