"""Quick functional + timing comparison of a screening run against a baseline run (same items, same settings).

Model-text evidence only (no ASR): per item, text identity and word error rate vs the baseline, tool calls (name +
normalized args: emitted / missing / invented), greeting step (first agent turn), agent-turn onset steps, run-on /
self-asked questions (score_corpus.turn_stats), FC window wall time per call (function.started → tool.result.consumed),
step p50/p95, RTF, peak memory. Prints one JSON line per item + an aggregate; writes <run>/screen_vs_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_corpus import score_item, turn_stats  # noqa: E402

CHECK_KEYS = ("function_ok", "args_ok", "status_model", "date_model", "tracking_model", "forbidden_called",
              "answer_model", "format_ok", "asked_model", "must_not_hit", "grounding_violations", "n_calls")


TURN_TOL = 25            # model steps (2 s): agent turns starting this close to a base turn are the same turn

def outcome(base_ok, cand_ok) -> str:
    """Absolute success vs regression w.r.t. the BF16 baseline on the same GPU."""
    if cand_ok:
        return "pass" if base_ok else "fixed"
    return "regression" if base_ok else "intrinsic_fail"



def words(t: str) -> list[str]:
    t = re.sub(r"</?s>", " ", t or "").lower()
    return re.findall(r"[a-z0-9']+", t)


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return round(d[-1] / len(ref), 3)


def calls(s) -> list[tuple]:
    return [(c.get("name"), json.dumps({k: str(v).strip().upper() for k, v in (c.get("args") or {}).items()},
                                       sort_keys=True)) for c in s.get("tool_calls", [])]


def trace_info(p: Path) -> dict:
    ev = [json.loads(line) for line in p.open()] if p.exists() else []
    agent = [e["model_step"] for e in ev if e["ev"] == "turn.started" and e.get("who") == "agent"]
    fs = [e["t_ns"] for e in ev if e["ev"] == "function.started"]
    fc = [e["t_ns"] for e in ev if e["ev"] == "tool.result.consumed"]
    return {"greeting_step": agent[0] if agent else None, "agent_turn_steps": agent,
            "fc_window_ms": [round((b - a) / 1e6) for a, b in zip(fs, fc)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus-dir", action="append", default=[],
                    help="item sidecar dir or spec file (eval/*.json); enables per-item functional scoring of both runs")
    a = ap.parse_args()
    run, base = Path(a.run), Path(a.baseline)
    specs = {}
    for d in map(Path, a.corpus_dir):              # a sidecar dir, or a spec file (eval/*.json with an items list)
        if d.is_file():
            specs.update({it["id"]: it for it in json.loads(d.read_text())["items"]})
        else:
            specs.update({p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.stem not in ("manifest", "orders")})
    rows = []
    for d in sorted(p.parent for p in run.glob("*/summary.json")):
        i = d.name
        s = json.load(open(d / "summary.json"))
        bp = base / i / "summary.json"
        b = json.load(open(bp)) if bp.exists() else None
        r = {"id": i, "step_ms_p50": s.get("step_ms_p50"), "step_ms_p95": s.get("step_ms_p95"), "rtf": s.get("rtf"),
             "max_mem_gb": s.get("max_mem_gb"), **trace_info(d / "trace.jsonl"), **turn_stats(s)}
        if b is not None:
            bc, sc = calls(b), calls(s)
            r.update(text_identical=s.get("text") == b.get("text"),
                     wer_vs_base=wer(words(b.get("text")), words(s.get("text"))),
                     calls=sc, calls_missing=[c for c in bc if c not in sc], calls_invented=[c for c in sc if c not in bc],
                     base_step_ms_p50=b.get("step_ms_p50"),
                     base_greeting_step=trace_info(base / i / "trace.jsonl")["greeting_step"],
                     base_question_turns=turn_stats(b)["question_turns"],
                     base_continuation_turns=turn_stats(b)["continuation_turns"])
            # turn-aligned check (meaningful on multi-turn items, where turn_stats counts every legitimate later
            # answer as a "continuation" and every answer ending with "?" as a "question turn"): agent turns whose
            # start step has no counterpart in the base within TURN_TOL steps = new autonomous turns / missing turns
            bs, rs = trace_info(base / i / "trace.jsonl")["agent_turn_steps"], r["agent_turn_steps"]
            r["turns_new"] = [x for x in rs if not any(abs(x - y) <= TURN_TOL for y in bs)]
            r["turns_missing"] = [y for y in bs if not any(abs(x - y) <= TURN_TOL for x in rs)]
        if b is not None and i in specs and "expect" in specs[i]:
            sc, sb = score_item(specs[i], s, None), score_item(specs[i], b, None)
            r["ok"], r["base_ok"] = bool(sc["ok_model"]), bool(sb["ok_model"])
            r["outcome"] = outcome(r["base_ok"], r["ok"])
            r["checks_changed"] = {k: [sb.get(k), sc.get(k)] for k in CHECK_KEYS if sb.get(k) != sc.get(k)}
            # turn-taking is a check too: greeting step and number of agent turns
            if r["greeting_step"] != r["base_greeting_step"]:
                r["checks_changed"]["greeting_step"] = [r["base_greeting_step"], r["greeting_step"]]
            r["lexical_only"] = (not r["text_identical"]) and not r["checks_changed"] \
                and len(r["agent_turn_steps"]) == len(trace_info(base / i / "trace.jsonl")["agent_turn_steps"])
        rows.append(r)
        print(json.dumps(r), flush=True)
    cmp = [r for r in rows if "text_identical" in r]
    agg = {"n": len(rows), "text_identical": sum(r["text_identical"] for r in cmp),
           "wer_mean": round(sum(r["wer_vs_base"] for r in cmp) / max(1, len(cmp)), 3),
           "calls_missing": sum(len(r["calls_missing"]) for r in cmp),
           "calls_invented": sum(len(r["calls_invented"]) for r in cmp),
           "greeting_shifted": sum(r["greeting_step"] != r["base_greeting_step"] for r in cmp),
           "extra_question_turns": sum(max(0, r["question_turns"] - r["base_question_turns"]) for r in cmp),
           "extra_continuation_turns": sum(max(0, r["continuation_turns"] - r["base_continuation_turns"]) for r in cmp),
           "turns_new": sum(len(r["turns_new"]) for r in cmp), "turns_missing": sum(len(r["turns_missing"]) for r in cmp),
           "step_ms_p50_mean": round(sum(r["step_ms_p50"] for r in rows) / max(1, len(rows)), 1),
           "base_step_ms_p50_mean": round(sum(r["base_step_ms_p50"] for r in cmp) / max(1, len(cmp)), 1) if cmp else None,
           "max_mem_gb": max((r["max_mem_gb"] or 0) for r in rows) if rows else None}
    sc = [r for r in rows if "outcome" in r]
    if sc:
        agg["functional"] = {k: sum(r["outcome"] == k for r in sc) for k in ("pass", "fixed", "regression", "intrinsic_fail")}
        agg["functional"]["lexical_only_diffs"] = sum(r["lexical_only"] for r in sc)
        agg["regressions"] = [r["id"] for r in sc if r["outcome"] == "regression"]
    print("AGG " + json.dumps(agg), flush=True)
    (run / f"screen_vs_{a.tag}.json").write_text(json.dumps({"baseline": str(base), "rows": rows, "agg": agg}, indent=1))


if __name__ == "__main__":
    main()
