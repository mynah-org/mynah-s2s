"""Where a multi-session round spends its time: per-component host / GPU spans from the gate's diag.json.

Needs a gate run with V2_DIAG=1 and --breakdown (default in the release preset). Prints, for a few rounds, every
component call in start order with its offset in the round, host time and GPU span, then the per-component medians
over the steady rounds and the round's GPU-idle estimate (round wall time minus the sum of GPU spans). Large gaps
between the end of one GPU span and the next start are host work with the GPU idle (see V2_HANDOFF_TRACE=1 for the
per-session hand-off times and V2_CPROFILE=FIRST-LAST for a host profile of a round window).

usage: python v2/scripts/tools/round_layout.py RUN_DIR [FIRST_ROUND] [N_ROUNDS]
"""
import json
import statistics as st
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = sys.argv[1].rstrip("/")
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    d = json.load(open(run + "/diag.json"))
    rounds, tl = d["rounds"], d["timeline"]
    for r in rounds[first:first + n]:
        a, b = r["ts_ms"], r["ts_ms"] + r["dt_ms"]
        print(f"== round {r['k']}  wall {r['dt_ms']:.1f} ms")
        for e in sorted((t for t in tl if a <= t["t0_ms"] < b), key=lambda t: t["t0_ms"]):
            print(f"  +{e['t0_ms'] - a:7.1f}  {e['comp']:20s} {e['thread']:12s} host {e['host_ms']:6.1f}  gpu {e['gpu_ms']:6.1f}")
    steady = rounds[20:-20] or rounds
    per, idle = {}, []
    for r in steady:
        a, b = r["ts_ms"], r["ts_ms"] + r["dt_ms"]
        ev = [t for t in tl if a <= t["t0_ms"] < b]
        for e in ev:
            per.setdefault(e["comp"], []).append(e["gpu_ms"])
        idle.append(r["dt_ms"] - sum(e["gpu_ms"] for e in ev))
    print(f"== steady rounds: {len(steady)}  wall p50 {st.median(r['dt_ms'] for r in steady):.1f} ms  "
          f"GPU-idle p50 {st.median(idle):.1f} ms")
    for c, v in sorted(per.items(), key=lambda kv: -st.median(kv[1])):
        print(f"  {c:20s} gpu p50 {st.median(v):6.1f} ms  calls {len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
