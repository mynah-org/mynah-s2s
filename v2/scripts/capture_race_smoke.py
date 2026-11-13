"""Deterministic CUDA-graph capture race smoke (no model): what another thread's CUDA calls do to an OPEN capture,
whether a failed capture poisons the process, and whether capture_guard prevents / recovers it.

Each case runs in its own subprocess (a broken CUDA context must not leak into the next case). Thread A opens a
capture (``capture_error_mode="thread_local"``, an RNG op inside, like EarTTS) and holds it open on a host event;
thread B then issues ONE call; A closes the capture. Afterwards the main thread probes an eager RNG op and the replay
of a graph captured BEFORE (with RNG: the EarTTS-graph analogue), then calls
capture_guard.reset_generator_capture_state() and probes again.
Cases: sync (torch.cuda.synchronize: perception_cache.py:542), rng (eager torch.rand), replay (RNG graph replay:
eartts_graph.py:273), capture2 (a second torch.cuda.graph enter: its __enter__ syncs), d2h (.cpu() on the legacy
stream: codec), guard_barrier (barrier policy: A defers while B is busy, captures once B is done), guard_recover
(sync case with the guard's failure path: key disabled + generator reset → probes must pass).
Writes one JSON line per case to --out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

CASES = ["sync", "rng", "replay", "capture2", "d2h", "guard_barrier", "guard_recover"]


def err(e):
    return None if e is None else f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def try_(fn):
    try:
        fn()
        torch.cuda.current_stream().synchronize()
        return None
    except Exception as e:  # noqa: BLE001
        return e


def run_case(case: str) -> dict:
    global torch
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from mynah_v2.model.capture_guard import GUARD, reset_generator_capture_state
    dev = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(1024, 1024, device=dev)
    # a graph with RNG captured while quiet (the EarTTS full-step graph analogue)
    g0, s0 = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        y0 = torch.rand(1024, device=dev)
    torch.cuda.current_stream().wait_stream(s0)
    with torch.cuda.graph(g0, capture_error_mode="thread_local"):
        y0 = torch.rand(1024, device=dev) + x[0]
    g0.replay()
    torch.cuda.synchronize()
    res = {"case": case, "torch": torch.__version__}
    started, proceed = threading.Event(), threading.Event()
    box = {}

    def body():                                            # the capture body: RNG + GEMMs, held open on an event
        y = x @ x
        z = torch.rand_like(y)
        started.set()
        proceed.wait(10)
        return (y @ x) + z

    def capturer(guarded=False, key="k"):
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                body()
            box["capture"] = "ok"
            box["graph"] = g
        except Exception as e:  # noqa: BLE001
            box["capture"] = err(e)
            started.set()
            if guarded:
                box["guard_recovered"] = GUARD.on_failure("smoke", key, e)

    def other(fn):
        started.wait(10)
        time.sleep(0.05)
        box["other"] = err(try_(fn))
        proceed.set()

    ops = {"sync": torch.cuda.synchronize,
           "rng": lambda: torch.rand(4, device=dev),
           "replay": g0.replay,
           "capture2": lambda: torch.cuda.graph(torch.cuda.CUDAGraph()).__enter__(),
           "d2h": lambda: (x[:4, :4] * 2).cpu()}
    if case in ops or case == "guard_recover":
        if case == "guard_recover":
            GUARD.configure("barrier")
        ta = threading.Thread(target=capturer, kwargs={"guarded": case == "guard_recover"})
        tb = threading.Thread(target=other, args=(ops["sync" if case == "guard_recover" else case],))
        ta.start()
        tb.start()
        ta.join(30)
        tb.join(30)
        res["capture_in_A"] = box.get("capture")
        res["call_in_B"] = box.get("other")
        if case == "guard_recover":
            res["guard_recovered"] = box.get("guard_recovered")
            res["guard_disabled_key"] = not GUARD.may_capture("smoke", "k")
            res["guard_stats"] = GUARD.report()
    elif case == "guard_barrier":
        GUARD.configure("barrier")
        busy = threading.Event()
        GUARD.add_provider(prov := (lambda me: "worker_running:B" if busy.is_set() else None))  # noqa: F841
        stop = threading.Event()
        b_errors = []

        def worker_b():                                    # B: syncs + RNG continuously while "running"
            busy.set()
            while not stop.is_set():
                e = try_(lambda: (torch.rand(256, device=dev).sum().item(), torch.cuda.synchronize()))
                if e is not None:
                    b_errors.append(err(e))
            busy.clear()
        tb = threading.Thread(target=worker_b)
        tb.start()
        time.sleep(0.2)
        decisions = []
        for attempt in range(2):
            if GUARD.may_capture("smoke", "k"):
                proceed.set()
                capturer()
                decisions.append("captured:" + str(box.get("capture")))
            else:
                (x @ x).sum().item()                       # eager fallback for this step
                decisions.append("eager")
            if attempt == 0:
                stop.set()
                tb.join(10)
        res["decisions"] = decisions
        res["errors_in_B"] = b_errors[:3]
        res["guard_stats"] = GUARD.report()
    # ---- after the case: is the process poisoned?
    res["probe_rng_after"] = err(try_(lambda: torch.rand(4, device=dev)))
    res["probe_g0_replay_after"] = err(try_(g0.replay))
    res["poisoned"] = res["probe_rng_after"] is not None or res["probe_g0_replay_after"] is not None
    if res["poisoned"]:
        res["reset"] = err(try_(reset_generator_capture_state))
        res["probe_rng_after_reset"] = err(try_(lambda: torch.rand(4, device=dev)))
        res["probe_g0_replay_after_reset"] = err(try_(g0.replay))
        res["new_capture_after_reset"] = err(try_(lambda: _new_capture(torch, dev)))
    return res


def _new_capture(torch, dev):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        torch.rand(16, device=dev)
    g.replay()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.case:
        print("CASE " + json.dumps(run_case(a.case), default=str), flush=True)
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for c in CASES:
            p = subprocess.run([sys.executable, __file__, "--case", c], capture_output=True, text=True, timeout=180)
            line = next((l[5:] for l in p.stdout.splitlines() if l.startswith("CASE ")), None)
            rec = json.loads(line) if line else {"case": c, "rc": p.returncode, "stderr": p.stderr[-800:]}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
