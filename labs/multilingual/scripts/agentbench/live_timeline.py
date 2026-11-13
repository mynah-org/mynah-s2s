"""Per-session timeline from observe.jsonl (+ summary.json) of a live run (CPU).

For each session:
  * BOS/EOS events per frame, labelled SPONTANEOUS (the Thinker's own token, `pre`) or FORCED (written by the RNNT
    turn-taking: pre != post), and suppressed model tokens (pre = BOS/EOS, post = something else);
  * RNNT/turn state transitions (agent_speaking, first_turn, speech_confirmed) with frames;
  * logP(BOS) in the lead-in silence (max and frame), logP(SOTC) peak after the caller's start, function top-k there;
  * tool calls with arguments, first call frame, final agent text.
Frames are 80 ms model steps. Usage: live_timeline.py <run_dir> [--out timeline.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAD, BOS, EOS, SOTC = 12, 1, 2, 20
NAMES = {BOS: "BOS", EOS: "EOS"}


def analyse(rows: list[dict], summary: dict) -> dict:
    lead_frames = int(round(summary.get("lead_s", 7.0) / 0.08))
    bounds = summary.get("turn_bounds_s") or [[summary.get("lead_s", 7.0), None]]
    user_end = int(round(bounds[0][1] / 0.08)) if bounds and bounds[0][1] else None
    forcing = [r for r in rows if r.get("kind") == "forcing" and r.get("fn") == "_apply_rnnt_turn_taking"]
    other_forcing = [r for r in rows if r.get("kind") == "forcing" and r.get("fn") != "_apply_rnnt_turn_taking"
                     and r.get("forced")]
    logits = [r for r in rows if r.get("kind") == "logits" and r.get("t") is not None]
    events = []
    for r in forcing:
        pre, post, t = r["pre"], r["post"], r["t"]
        if pre in NAMES and post == pre:
            events.append({"t": t, "s": round(t * 0.08, 2), "token": NAMES[pre], "how": "spontaneous"})
        elif post in NAMES and pre != post:
            events.append({"t": t, "s": round(t * 0.08, 2), "token": NAMES[post], "how": "forced", "replaced": pre,
                           "rnnt": r.get("rnnt")})
        elif pre in NAMES and post != pre:
            events.append({"t": t, "s": round(t * 0.08, 2), "token": NAMES[pre], "how": "suppressed", "post": post})
    for r in other_forcing:
        events.append({"t": r["t"], "s": round(r["t"] * 0.08, 2), "token": NAMES.get(r["post"], r["post"]),
                       "how": "forced(asr-channel)"})
    events.sort(key=lambda e: e["t"])
    trans, last = [], None
    for r in forcing:
        st = r.get("rnnt") or {}
        key = (st.get("agent_speaking"), st.get("first_turn"), st.get("speech_confirmed"))
        if key != last:
            trans.append({"t": r["t"], "agent_speaking": key[0], "first_turn": key[1], "speech_confirmed": key[2],
                          "blank_count": st.get("blank_count"), "nonblank_consec": st.get("nonblank_consec")})
            last = key
    by_t = {}
    for r in logits:            # main-loop rows; FC fast-path rows share t values later in time, keep the first
        by_t.setdefault(r["t"], r)
    lead = [r for t, r in by_t.items() if t < lead_frames]
    bos_lead = max(lead, key=lambda r: r["text"]["BOS"]) if lead else None
    after = [r for t, r in by_t.items() if t >= lead_frames and "func" in r]
    sotc = max(after, key=lambda r: r["func"]["SOTC"]) if after else None
    return {
        "lead_frames": lead_frames, "user_end_frame": user_end,
        "first_agent_bos": next(({"t": e["t"], "how": e["how"]} for e in events if e["token"] == "BOS"
                                 and e["how"] != "suppressed"), None),
        "events": events, "state_transitions": trans,
        "lead_bos_logp_max": None if not bos_lead else {"t": bos_lead["t"], "logp": round(bos_lead["text"]["BOS"], 3)},
        "sotc_peak": None if not sotc else {"t": sotc["t"], "rel_user_end": None if user_end is None else sotc["t"] - user_end,
                                            "logp": round(sotc["func"]["SOTC"], 3), "func_top": sotc.get("func_top")},
        "calls": [{"name": c["name"], "args": c["args"], "model_step": c.get("model_step")} for c in summary.get("calls", [])],
        "first_call_s": summary.get("first_call_s"), "premature_call": summary.get("premature_call"),
        "agent_text": summary.get("agent_text"), "user_rnnt": summary.get("user_rnnt"),
        "wall_s": summary.get("wall_s"), "fallback": summary.get("first_turn_fallback"),
        "fallback_frames": summary.get("first_turn_fallback_frames"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for d in sorted(Path(a.run_dir).iterdir()):
        if not (d / "observe.jsonl").exists() or not (d / "summary.json").exists():
            continue
        rows = [json.loads(line) for line in open(d / "observe.jsonl")]
        res[d.name] = analyse(rows, json.load(open(d / "summary.json")))
        r = res[d.name]
        print(f"== {d.name}  fallback={r['fallback']}  wall={r['wall_s']}s  first BOS={r['first_agent_bos']}  "
              f"lead logP(BOS) max={r['lead_bos_logp_max']}  SOTC peak={r['sotc_peak'] and {k: r['sotc_peak'][k] for k in ('t', 'rel_user_end', 'logp')}}")
        for e in r["events"][:12]:
            print("   ", e["s"], e["token"], e["how"], {k: e[k] for k in ("replaced", "post") if k in e})
        print("    calls:", r["calls"], "| text:", (r["agent_text"] or "")[:220])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
