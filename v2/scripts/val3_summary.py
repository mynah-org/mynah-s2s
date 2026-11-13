"""VAL3 summary (CPU): FC budget runs of the multi-session gate (pair_gate.json, pair_trace.json, diag.json).

Per run: round wall p50/p95/max + misses; per session lag (start_delay: max, last-quarter mean, slope over k by least
squares, slope over the second half); FC cycles (p1/gap/p2/total, start/end vs realtime arrival); FC queue (max depth,
depth slope, in-round served hist, between-round served, forced); FC queue wait per FC session (fairness); FC work
per TICK window = [round k start, round k+1 start) (Thinker step_one GPU + eartts/fc + codec/fc GPU, steps); calls.
Usage: val3_summary.py RUN_DIR [...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def slope(ys):
    n = len(ys)
    if n < 2:
        return None
    mx, my = (n - 1) / 2, sum(ys) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return round(sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den, 2)


def fc_windows(run: Path):
    p = run / "diag.json"
    if not p.exists():
        return None
    dg = json.loads(p.read_text())
    rounds, tl = dg["rounds"], dg.get("timeline") or []
    starts = [r["ts_ms"] for r in rounds] + [float("inf")]
    out = []
    j = 0
    tl = sorted(tl, key=lambda e: e["t0_ms"])
    for i, r in enumerate(rounds):
        a, b = starts[i], starts[i + 1]
        w = defaultdict(float)
        while j < len(tl) and tl[j]["t0_ms"] < a:
            j += 1
        jj = j
        while jj < len(tl) and tl[jj]["t0_ms"] < b:
            e = tl[jj]
            key = None
            if e["comp"] == "thinker_step_one":
                key = "think1"
            elif e["tag"] == "fc" and e["comp"] in ("eartts", "codec"):
                key = e["comp"] + "_fc"
            if key:
                w[key + "_gpu"] += e["gpu_ms"]
                w[key + "_n"] += 1
            jj += 1
        w["fc_gpu"] = w["think1_gpu"] + w["eartts_fc_gpu"] + w["codec_fc_gpu"]
        out.append({"k": r["k"], **{k: round(v, 1) for k, v in w.items()}})
    return out


def summarize(run: Path) -> dict:
    g = json.loads((run / "pair_gate.json").read_text())
    tr = json.loads((run / "pair_trace.json").read_text())
    rounds = tr["rounds_ms"]
    res = {"run": run.name, "fc_budget": g.get("fc_budget"), "eartts_rng": g.get("eartts_rng"),
           "rounds": {"n": len(rounds), "p50": pct(rounds, .5), "p95": pct(rounds, .95), "max": round(max(rounds), 1),
                      "misses": g.get("pair_deadline_misses"), "busy_s": round(sum(rounds) / 1e3, 1),
                      "thr": g.get("pair_throughput_session_chunks_per_busy_s")}}
    lag = {}
    for sid, rows in tr["e2e"].items():
        sd = [x["start_delay_ms"] for x in rows]
        n = len(sd)
        lag[sid] = {"max_ms": round(max(sd), 1), "last_q_mean_ms": round(sum(sd[-(n // 4):]) / max(1, n // 4), 1),
                    "slope_ms_per_round": slope(sd), "slope_2nd_half": slope(sd[n // 2:]),
                    "end_ms": round(sd[-1], 1)}
    res["lag"] = lag
    fr = [x for x in tr.get("fc_rounds") or [] if x]
    q = [x["q_max"] for x in fr]
    res["fc_queue"] = {"q_max": max(q, default=0), "q_slope": slope(q), "q_end_last": fr[-1]["q_end"] if fr else None,
                       "served_hist": (g.get("fc_budget") or {}).get("served_in_round_hist"),
                       "served_in_rounds": sum(x["served"] for x in fr),
                       "served_between": sum(x["between_before"] for x in fr),
                       "forced": sum(x["forced"] for x in fr),
                       "rounds_deferring": sum(1 for x in fr if x["deferred_polls"])}
    fcs = (g.get("rendezvous_stats") or {}).get("fc_by_sid") or {}
    res["fc_wait_by_sid"] = {sid: {k: v.get(k) for k in ("n", "queue_ms_p50", "queue_ms_p95", "queue_ms_max",
                                                         "queue_ms_sum", "exec_ms_sum")} for sid, v in fcs.items()}
    ses = {}
    for sid, s in g["sessions"].items():
        ses[sid] = {"item": s["item"], "calls_identical": s.get("tool_calls_identical"),
                    "calls_pair": [(t["name"], t["args"]) for t in s.get("tools_pair") or []],
                    "ok_solo": s.get("ok_solo"), "ok_pair": s.get("ok_pair"), "leak": s.get("leak_in_pair"),
                    "errors": bool(s.get("errors")), "text_ids_identical": s.get("text_ids_identical"),
                    "nonfinite": (s.get("audio_pair") or {}).get("nonfinite"),
                    "rms_solo_pair": ((s.get("audio_solo") or {}).get("rms"), (s.get("audio_pair") or {}).get("rms")),
                    "fc_solo": s.get("fc_cycles_solo"), "fc_pair": s.get("fc_cycles_pair")}
    res["sessions"] = ses
    w = fc_windows(run)
    if w:
        busy = [x for x in w if x.get("fc_gpu", 0) > 0]
        res["fc_work_per_tick_window"] = {
            "windows_with_fc": len(busy), "fc_gpu_p50": pct([x["fc_gpu"] for x in busy], .5),
            "fc_gpu_p95": pct([x["fc_gpu"] for x in busy], .95), "fc_gpu_max": pct([x["fc_gpu"] for x in busy], 1.0),
            "think1_n_max": max((x.get("think1_n", 0) for x in w), default=0),
            "eartts_fc_n_max": max((x.get("eartts_fc_n", 0) for x in w), default=0),
            "codec_fc_n_max": max((x.get("codec_fc_n", 0) for x in w), default=0),
            "fc_gpu_total": round(sum(x["fc_gpu"] for x in w), 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    allr = [summarize(Path(r)) for r in a.runs]
    for r in allr:
        print("=" * 10, r["run"], "budget", (r["fc_budget"] or {}).get("budget"), "rng", r["eartts_rng"])
        print("ROUNDS", json.dumps(r["rounds"]))
        print("LAG", json.dumps(r["lag"]))
        print("FCQ", json.dumps(r["fc_queue"]))
        print("FCWAIT", json.dumps(r["fc_wait_by_sid"]))
        print("FCWORK", json.dumps(r.get("fc_work_per_tick_window")))
        for sid, s in r["sessions"].items():
            print("SESS", sid, json.dumps(s))
    if a.json:
        Path(a.json).write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
