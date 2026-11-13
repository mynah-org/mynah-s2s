"""Post-FC fix validation (CPU): per session, the token inputs the main loop really consumes at every FC resume, plus
calls, scorer v2, latency, internal text and OmniASR text.

Resume inputs: the first main-loop frame after an FC is r = (tool.result.consumed model_step) + 1 for a completed
two-phase FC. That frame embeds context.gen_text[r-1] and context.gen_function_text[r-1] (wrapper:2398-2406). The V2
trace's `thinker.step` events report exactly those context buffers (reference.py:325, `ctx.gen_text[0, lo:fi]`), so
the ids at r-1 are the consumed ones. Expected with F2: (PAD 12, EOTR 22); with the defect: (0, 0).
Usage: fix_eval.py <session_dir> [...] [--a-trace T] [--asr asr_omni3b.json ...] [--out f.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "multilingual"))

PAD, BOS, EOTR = 12, 1, 22


def buffers_from_trace(trace):
    """position -> id from thinker.step events, aligned at the START of the step (model_step - steps).

    After an FC the bulk step reports text ids for all `steps` positions (FC span + the 2 main frames) but function
    ids only for the FC span (96 vs 98 in EN m03), so aligning both lists at the end (first version) shifted the
    function channel by 2; start alignment is correct for both."""
    txt, fun = {}, {}
    for e in trace:
        if e["ev"] != "thinker.step" or not e.get("steps"):
            continue
        start = e["model_step"] - e["steps"]
        for key, d in (("text_ids", txt), ("func_ids", fun)):
            for i, t in enumerate(e.get(key) or []):
                d[start + i] = t
    return txt, fun


def resume_points(trace):
    out = []
    for e in trace:
        if e["ev"] == "tool.result.consumed" and e.get("model_step") is not None:
            out.append(e["model_step"] + 1)
    return out


def evaluate(sd: Path, a_trace, asr_sessions) -> dict:
    from guided_eval import evaluate as gev
    trace = [json.loads(x) for x in open(sd / "trace.jsonl")]
    summ = json.load(open(sd / "summary.json"))
    txt, fun = buffers_from_trace(trace)
    lg = {}
    if (sd / "observe.jsonl").exists():
        for x in map(json.loads, open(sd / "observe.jsonl")):
            if x.get("kind") == "logits" and x.get("t") is not None:
                lg.setdefault(x["t"], x)
    res = []
    for r in resume_points(trace):
        span = [p for p in range(r - 64, r) if p in txt]
        res.append({"resume_frame": r, "consumed": [txt.get(r - 1), fun.get(r - 1)],
                    "ok_PAD_EOTR": [txt.get(r - 1), fun.get(r - 1)] == [PAD, EOTR],
                    "zeros_in_fc_span_text": sum(1 for p in span if txt.get(p) == 0),
                    "next_action": None if r not in lg else {
                        "pred_text": lg[r]["pred_text"], "pred_func": lg[r]["pred_func"],
                        "sotc_minus_pad": round(lg[r]["func"]["SOTC"] - lg[r]["func"]["PAD"], 2)}})
    g = gev(sd, Path(a_trace) if a_trace and summ.get("opening") == "guided" else None, None, None, None)
    asr = " ".join(seg["text"] for s in asr_sessions                    # match the WAV path: same scenario names
                   if Path(s.get("wav", "")).parent.resolve() == sd.resolve() for seg in s.get("segments", []))
    return {"session": sd.name, "runtime_fix": summ.get("runtime_fix"), "opening": summ.get("opening"),
            "resume_inputs": res, "calls": [(c["name"], c["args"], c.get("model_step")) for c in summ.get("calls", [])],
            "score_pass": g["score_pass"], "score_failed": [k for k, v in g["score_checks"].items() if not v],
            "slot_provenance": [(p["arg"], p["value"], p["status"]) for p in g["slot_provenance"]],
            "redundant_calls": len(g["redundant_calls"]), "S1": g["S1"], "S1_pass": g["S1_pass"],
            "latency": g["latency"], "agent_text": summ.get("agent_text"), "asr_text": asr, "wall_s": summ.get("wall_s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="+")
    ap.add_argument("--a-trace", default=None)
    ap.add_argument("--asr", action="append", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    asr = [s for f in a.asr for s in json.load(open(f)).get("sessions", [])]
    rows = [evaluate(Path(d), a.a_trace, asr) for d in a.session_dir]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False)[:1500])
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
