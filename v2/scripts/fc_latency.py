"""FC latency breakdown + call-repetition check over a run_stream output dir (one or two runs, e.g. fix OFF vs ON).

Per FC cycle (from summary.json fc_timeline + tool_calls, trace.jsonl):
  T_FC = phase-1 decode (call tokens; FC thread, incl. on-hold TTS) + gap (tool execution + hand-off)
         + phase-2 decode (forced result tokens → EOTR) + resume (FC end → first agent text of the answer)
  plus FC tokens, ms/token per phase, the tool's own latency (ledger), and per item: number of calls, calls repeated
  with identical (name, args), answer text. Single-session runs have no scheduler queue; the two-session gate reports
  the coordinator queue separately (rendezvous_stats.fc_by_sid).
Usage: fc_latency.py RUN [RUN_B]   (prints one ITEM line per item and CALL lines per cycle; JSON to stdout last)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def first_text_after(trace: Path, t_ns: int):
    if not trace.exists() or t_ns is None:
        return None
    for line in trace.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("ev") == "text.generated" and e.get("t_ns", 0) > t_ns and re.search(r"[A-Za-z]", e.get("text") or ""):
            return round((e["t_ns"] - t_ns) / 1e6, 1)
    return None


def analyze(run: Path) -> dict:
    items = {}
    for f in sorted(run.glob("*/summary.json")):
        s = json.loads(f.read_text())
        calls = [(c["name"], json.dumps(c["args"], sort_keys=True)) for c in s.get("tool_calls", [])]
        rep = sum(1 for i in range(1, len(calls)) if calls[i] in calls[:i])
        cyc = []
        tools = s.get("tool_calls", [])
        for i, c in enumerate(s.get("fc_timeline", {}).get("calls", [])):
            p1, p2 = c.get("phase1_steps"), c.get("phase2_steps")
            d = {"start_frame": c["start_frame"], "p1_steps": p1, "p1_ms": c.get("phase1_ms"),
                 "p1_ms_per_tok": round(c["phase1_ms"] / p1, 1) if c.get("phase1_ms") and p1 else None,
                 "gap_ms": c.get("gap_ms"), "tool_ms": tools[i]["latency_ms"] if i < len(tools) else None,
                 "p2_steps": p2, "p2_ms": c.get("phase2_ms"),
                 "p2_ms_per_tok": round(c["phase2_ms"] / p2, 1) if c.get("phase2_ms") and p2 else None,
                 "resume_ms": first_text_after(f.parent / "trace.jsonl", c.get("t_end_ns")),
                 "violations": c.get("violations")}
            cyc.append(d)
        items[f.parent.name] = {"calls": [(n, json.loads(a)) for n, a in calls], "repeated_calls": rep,
                                "fc_cycles": cyc, "text": (s.get("text") or "")[-300:],
                                "step_ms_p50": s.get("step_ms_p50"), "rtf": s.get("rtf")}
    return items


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    res = {str(r): analyze(r) for r in runs}
    for r, items in res.items():
        for k, it in items.items():
            print(f"ITEM {Path(r).name} {k} calls={json.dumps(it['calls'])} repeated={it['repeated_calls']} "
                  f"text={it['text'][-160:]!r}")
            for c in it["fc_cycles"]:
                print(f"  CALL f{c['start_frame']} p1 {c['p1_steps']} tok {c['p1_ms']} ms ({c['p1_ms_per_tok']} ms/tok) "
                      f"| gap {c['gap_ms']} ms (tool {c['tool_ms']}) | p2 {c['p2_steps']} tok {c['p2_ms']} ms "
                      f"({c['p2_ms_per_tok']} ms/tok) | resume {c['resume_ms']} ms | viol {c['violations']}")
    print("JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
