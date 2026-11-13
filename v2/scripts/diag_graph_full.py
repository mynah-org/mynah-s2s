"""Diagnose a failing eartts_graph_full capture: full traceback, live threads, graph stats (short input)."""

import os
import sys
import threading
import traceback
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def main():
    os.environ.setdefault("V2_GRAPH_SYNC_DEBUG", "warn")      # first host-synchronizing op in the step raises with a stack
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": False}))
    for n in ("mamba_host_pos", "no_timer_sync"):
        perf.enable(n, eng)
    info = perf.enable("eartts_graph_full", eng)
    eng.open("You are a helpful assistant.")
    x = torch.cat([load_16k(sys.argv[1]), torch.zeros(16000 * 5)])
    try:
        for i, c in enumerate(iter_chunks(x, eng.chunk_samples)):
            eng.step(c)
            if i == 80:
                break
        print("OK", info, flush=True)
        for st in info["stats"].get("sync_sites", []):
            print("SYNC_SITE\n" + st, flush=True)
    except Exception:  # noqa: BLE001
        print("TRACEBACK\n" + traceback.format_exc(), flush=True)
        print("THREADS", [(t.name, t.daemon) for t in threading.enumerate()], flush=True)
        for t in threading.enumerate():
            fr = sys._current_frames().get(t.ident)
            if fr is not None and t is not threading.current_thread():
                print("STACK", t.name, "".join(traceback.format_stack(fr)[-6:]), flush=True)
        print("STATS", {k: v for k, v in info["stats"].items() if k != "sync_sites"}, flush=True)
        for st in info["stats"].get("sync_sites", []):
            print("SYNC_SITE\n" + st, flush=True)
        raise


if __name__ == "__main__":
    main()
