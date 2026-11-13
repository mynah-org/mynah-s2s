"""VAL2 analysis (CPU) of multi-session gate runs made with V2_DIAG=1.

Per run dir (pair_gate.json, pair_trace.json, diag.json, codes.npz, wav/<sid>_<item>_{solo,pair}/output.wav):
  AUDIO  per session, solo vs pair: text tokens identical?, RVQ codes equality (all frames, first differing frame,
         first chunk = first 2 frames, per-codebook agreement), waveform corr, RMS per 80 ms frame in the greeting
         window, near-silent frames inside speech (RMS < 10% of the session's speech median) solo vs pair.
  ROUNDS per-round breakdown: rounds slower than --burst-ms vs the others: host / GPU ms per component and tag
         (main / fc) overlapping the round window, torch.cuda.synchronize count + wait per thread kind, FC state.
Usage: val2_analyze.py RUN_DIR [RUN_DIR ...] [--sessions c] [--burst-ms 400] [--rounds 60-80] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME = 1764            # output samples per 80 ms frame (22.05 kHz)


def wav(p):
    import soundfile as sf
    x, _ = sf.read(p)
    return x.astype(np.float64)


def frame_rms(x):
    n = len(x) // FRAME
    return np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1))


def audio_section(run: Path, sessions):
    gate = json.loads((run / "pair_gate.json").read_text())
    z = np.load(run / "codes.npz") if (run / "codes.npz").exists() else None
    out = {}
    for sid, s in gate["sessions"].items():
        if sessions and sid not in sessions:
            continue
        r = {"item": s.get("item"), "text_ids_identical": s.get("text_ids_identical"),
             "first_text_div_frame": s.get("first_text_div_frame")}
        if z is not None and f"solo__{sid}" in z and f"pair__{sid}" in z:
            a, b = z[f"solo__{sid}"], z[f"pair__{sid}"]
            n = min(len(a), len(b))
            eq = (a[:n] == b[:n])
            row = eq.all(1)
            r["codes"] = {"frames_solo": len(a), "frames_pair": len(b), "frames_equal": int(row.sum()), "n": n,
                          "first_diff_frame": int(np.argmin(row)) if not row.all() else None,
                          "first_chunk_equal": bool(row[:2].all()) if n >= 2 else None,
                          "first_25_frames_equal": int(row[:25].sum()),
                          "per_codebook_agree": [round(float(v), 3) for v in eq.mean(0)]}
        d = run / "wav"
        ws = sorted(d.glob(f"{sid}_*_solo/output.wav")), sorted(d.glob(f"{sid}_*_pair/output.wav"))
        if ws[0] and ws[1]:
            xs, xp = wav(ws[0][0]), wav(ws[1][0])
            n = min(len(xs), len(xp))
            r["wave_corr"] = round(float(np.corrcoef(xs[:n], xp[:n])[0, 1]), 4) if n > 1 else None
            r["wave_max_abs_diff"] = round(float(np.abs(xs[:n] - xp[:n]).max()), 5)
            rs, rp = frame_rms(xs), frame_rms(xp)
            r["greeting_rms_1.5_4s"] = {"solo": [round(v, 4) for v in rs[18:50:2]],
                                        "pair": [round(v, 4) for v in rp[18:50:2]]}
            for tag, rr in (("solo", rs), ("pair", rp)):
                sp = rr[rr > 0.005]
                med = float(np.median(sp)) if len(sp) else 0.0
                # near-silent frames strictly inside speech runs (speech before and after within 3 frames)
                holes = [i for i in range(3, len(rr) - 3) if rr[i] < 0.1 * med
                         and rr[i - 3:i].max() > 0.5 * med and rr[i + 1:i + 4].max() > 0.5 * med]
                r[f"holes_{tag}"] = {"speech_median_rms": round(med, 4), "n": len(holes),
                                     "at_s": [round(i * FRAME / 22050, 2) for i in holes[:12]]}
        out[sid] = r
    return out


def rounds_section(run: Path, burst_ms: float, window):
    dg = json.loads((run / "diag.json").read_text())
    tr = json.loads((run / "pair_trace.json").read_text())
    rounds, tl = dg["rounds"], dg.get("timeline") or []
    t0 = dg["pair_t0_ms"]
    syncs = [(ph, th, t - t0, w) for ph, th, t, w in dg["syncs"] if ph == "pair"]
    fc_at = defaultdict(list)
    for sid, rows in tr["e2e"].items():
        for x in rows:
            if x["fc"]:
                fc_at[x["k"]].append(sid)

    def kind(th):
        return "fc" if "_fc_async" in th else ("worker" if th.startswith("rz-") else "coord/main")

    per = []
    for r in rounds:
        a, b = r["ts_ms"], r["ts_ms"] + r["dt_ms"]
        comp = defaultdict(lambda: [0.0, 0.0, 0])
        for e in tl:
            if a <= e["t0_ms"] < b:
                c = comp[f'{e["comp"]}/{e["tag"]}']
                c[0] += e["host_ms"]; c[1] += e["gpu_ms"]; c[2] += 1
        sy = defaultdict(lambda: [0, 0.0])
        for _, th, t, w in syncs:
            if a <= t < b:
                s = sy[kind(th)]; s[0] += 1; s[1] += w
        per.append({"k": r["k"], "dt_ms": r["dt_ms"], "fc_sids": sorted(fc_at.get(r["k"], [])),
                    "comp": {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in sorted(comp.items())},
                    "sync": {k: [v[0], round(v[1], 1)] for k, v in sorted(sy.items())}})

    def agg(rows):
        tot = defaultdict(lambda: [0.0, 0.0, 0])
        st = defaultdict(lambda: [0, 0.0])
        for p in rows:
            for k, v in p["comp"].items():
                t = tot[k]; t[0] += v[0]; t[1] += v[1]; t[2] += v[2]
            for k, v in p["sync"].items():
                s = st[k]; s[0] += v[0]; s[1] += v[1]
        n = max(1, len(rows))
        return {"rounds": len(rows), "dt_ms_mean": round(sum(p["dt_ms"] for p in rows) / n, 1),
                "comp_per_round_host_gpu_calls": {k: [round(v[0] / n, 1), round(v[1] / n, 1), round(v[2] / n, 2)]
                                                  for k, v in sorted(tot.items())},
                "sync_per_round_n_ms": {k: [round(v[0] / n, 2), round(v[1] / n, 1)] for k, v in sorted(st.items())}}

    steady = [p for p in per if p["k"] >= 20]
    burst = [p for p in steady if p["dt_ms"] > burst_ms]
    calm_fc = [p for p in steady if p["dt_ms"] <= burst_ms and p["fc_sids"]]
    calm = [p for p in steady if p["dt_ms"] <= burst_ms and not p["fc_sids"]]
    res = {"n_rounds": len(per), "burst_ms": burst_ms, "burst_ks": [p["k"] for p in burst],
           "burst": agg(burst), "calm_with_fc": agg(calm_fc), "calm_no_fc": agg(calm)}
    if window:
        res["window"] = [p for p in per if window[0] <= p["k"] <= window[1]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--burst-ms", type=float, default=400.0)
    ap.add_argument("--rounds", default="", help="k range to dump per round, e.g. 60-80")
    ap.add_argument("--no-rounds", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    sess = [s for s in a.sessions.split(",") if s]
    win = tuple(int(x) for x in a.rounds.split("-")) if a.rounds else None
    allr = {}
    for r in a.runs:
        run = Path(r)
        d = {"audio": audio_section(run, sess)}
        if not a.no_rounds and (run / "diag.json").exists():
            d["rounds"] = rounds_section(run, a.burst_ms, win)
        allr[run.name] = d
        print("=" * 20, run.name)
        for sid, v in d["audio"].items():
            print("AUDIO", run.name, sid, json.dumps(v))
        if "rounds" in d:
            rr = d["rounds"]
            print("ROUNDS", run.name, "n", rr["n_rounds"], "burst ks", rr["burst_ks"])
            for g in ("burst", "calm_with_fc", "calm_no_fc"):
                print("ROUNDS", run.name, g, json.dumps(rr[g]))
            for p in rr.get("window", []):
                print("ROUND", run.name, json.dumps(p))
    if a.json:
        Path(a.json).write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
