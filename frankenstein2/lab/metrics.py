"""Session measurements from a run's records (model-free, unit-tested):

  timeline   user utterances [{"label", "start_k", "end_k", "barge"?}] (80 ms frames = steps)
  events     the session trace (frankenstein2.tracing lines)
  segs       playout segments [{"epoch", "turn", "seg", "lane", "holding", "first_step", "released", "text"}]
  wall       wall time (s, monotonic) at the START of each step, t0 = wall time of input sample 0
Clocks: "stream" = the session (audio) clock, step k = k * 80 ms of input (what a caller hears relative to their own
speech when the loop keeps real time); "wall" adds the loop's lag. Audio released at step s is heard from step
s + client_delay.
"""

from __future__ import annotations

FRAME_S = 0.08


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[int(p * (len(xs) - 1))], 3) if xs else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "p50": q(xs, .5), "p95": q(xs, .95), "max": round(max(xs), 3) if xs else None,
            "mean": round(sum(xs) / len(xs), 3) if xs else None}


def analyse(timeline, events, segs, wall, t0, client_delay_steps=1, tools_expected=None):
    ev = lambda name: [e for e in events if e["ev"] == name]  # noqa: E731
    ears_ev = ev("ears.event")
    endpoints = [e for e in ears_ev if e["kind"] == "ENDPOINT"]
    barges = [e for e in ears_ev if e["kind"] == "BARGE"]
    bcs = [e for e in ears_ev if e["kind"] == "BACKCHANNEL"]
    opens = [e for e in ev("ctl.action") if e["kind"] == "OPEN_TURN"]
    flushes = ev("audio.flush")
    agent_turns: dict = {}
    for s in segs:
        if s["lane"] != "agent" or s["first_step"] is None:
            continue
        r = agent_turns.setdefault((s["epoch"], s["turn"]), {"holding": None, "reply": None, "last": None})
        kind = "holding" if s["holding"] else "reply"
        if r[kind] is None or s["first_step"] < r[kind]:
            r[kind] = s["first_step"]
        end = s["first_step"] + (s["released"] + 1919) // 1920
        r["last"] = max(r["last"] or 0, end)
    per_utt = []
    for i, u in enumerate(timeline):
        e = u["end_k"]
        nxt = timeline[i + 1]["start_k"] if i + 1 < len(timeline) else 10 ** 9
        ep = next((x for x in endpoints if u["start_k"] <= x["model_step"] < nxt), None)
        early = [x["model_step"] for x in endpoints if u["start_k"] <= x["model_step"] < e]
        op = next((x for x in opens if x["model_step"] >= u["start_k"] and x["model_step"] < nxt), None)
        row = {"label": u["label"], "end_k": e, "early_endpoints": early,
               "endpoint_s": round((ep["model_step"] - e) * FRAME_S, 3) if ep else None,
               "turn_opened_s": round((op["model_step"] - e) * FRAME_S, 3) if op else None,
               "turn_id": op["turn_id"] if op else None}
        if op is not None:
            cand = [(k, v) for k, v in agent_turns.items() if k[1] == op["turn_id"]]
            if cand:
                v = cand[0][1]
                for kind in ("holding", "reply"):
                    if v[kind] is not None:
                        heard = v[kind] + client_delay_steps
                        row[f"first_{kind}_s"] = round((heard - (e + 1)) * FRAME_S, 3)
                        if heard < len(wall) and e + 1 < len(wall):
                            row[f"first_{kind}_wall_s"] = round(wall[heard] - wall[e + 1], 3)
        per_utt.append(row)
    # barge-ins: user onset -> playback stop (the flush step), for utterances that started over agent audio
    barge_rows = []
    for f in flushes:
        if f.get("reason") != "barge_in":
            continue
        k = f["model_step"]
        u = max((u for u in timeline if u["start_k"] <= k), key=lambda u: u["start_k"], default=None)
        b = max((b for b in barges if b["model_step"] <= k), key=lambda b: b["model_step"], default=None)
        barge_rows.append({"utt": u["label"] if u else None, "flush_step": k,
                           "onset_to_stop_s": round((k - u["start_k"]) * FRAME_S, 3) if u else None,
                           "dropped_s": round(f.get("dropped_samples", 0) / 24000, 2),
                           "onset_s_est": b.get("onset_s") if b else None})
    false_barges = []
    for b in barges:
        k = b["model_step"]
        active = any(u["start_k"] - 2 <= k and k <= u["end_k"] + int(2.5 / FRAME_S) for u in timeline)
        if not active:
            false_barges.append(k)
    lag = [round(wall[k] - (t0 + k * FRAME_S), 3) for k in range(len(wall))]
    tools = [{"name": e["name"], "args": e.get("args"), "args_raw": e.get("args_raw"), "turn_id": e.get("turn_id")}
             for e in ev("tool.completed")]
    out = {"utterances": per_utt, "barge_ins": barge_rows, "false_barges": false_barges,
           "backchannels": [b["model_step"] for b in bcs], "endpoints": [e["model_step"] for e in endpoints],
           "tools": tools, "loop_lag_s": {"end": lag[-1] if lag else None, **stats(lag)},
           "underruns": len(ev("audio.underrun")), "stale_audio": len(ev("audio.stale")),
           "ears_hop_ms": stats([e.get("ms") for e in ev("ears.hop")]),
           "ears_lag_steps": stats([e.get("lag_steps") for e in ev("ears.hop")])}
    if tools_expected is not None:
        out["tools_check"] = check_tools(tools, tools_expected)
    order = [r["turn_id"] for r in per_utt if r.get("turn_id") is not None]
    out["spoken_order_ok"] = order == sorted(order)
    return out


def check_tools(tools, expected):
    """expected: [{"name", "args": {key: substring}}]; each expected call must match a distinct actual call."""
    used, rows = set(), []
    for x in expected:
        hit = None
        for i, t in enumerate(tools):
            if i in used or t["name"] != x["name"]:
                continue
            a = t.get("args") or {}
            if all(str(v).lower() in str(a.get(k, "")).lower() for k, v in x.get("args", {}).items()):
                hit = i
                break
        if hit is not None:
            used.add(hit)
        rows.append({"expected": x, "ok": hit is not None, "got": tools[hit] if hit is not None else None})
    return {"ok": sum(r["ok"] for r in rows), "n": len(rows), "rows": rows}


def cer(ref: str, hyp: str) -> float:
    import re
    norm = lambda s: re.sub(r"[^\w]+", " ", s.lower()).strip()  # noqa: E731
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return round(d[len(h)] / len(r), 3)
