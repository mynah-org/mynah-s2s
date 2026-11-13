"""Diagnose CUDA-graph experiments on a short input: enable --exp (in order), run --chunks chunks, print graph stats,
every host-synchronizing site seen during warm-ups (V2_GRAPH_SYNC_DEBUG=warn) and, on failure, the full traceback.
    diag_graph.py WAV OUT --exp mamba_host_pos,no_timer_sync,thinker_graph [--chunks 80]"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--chunks", type=int, default=80)
    a = ap.parse_args()
    os.environ.setdefault("V2_GRAPH_SYNC_DEBUG", "warn")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": False}))
    infos = {n: perf.enable(n, eng) for n in a.exp.split(",")}
    eng.open("You are a helpful assistant.")
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 5)])
    ok = True
    try:
        for i, c in enumerate(iter_chunks(x, eng.chunk_samples)):
            eng.step(c)
            if i + 1 >= a.chunks:
                break
    except Exception:  # noqa: BLE001
        ok = False
        print("TRACEBACK\n" + traceback.format_exc(), flush=True)
    for n, info in infos.items():
        st = info.get("stats") if isinstance(info, dict) else None
        if st is not None:
            print("STATS", n, json.dumps({k: v for k, v in st.items() if k != "sync_sites"}), flush=True)
            for site in st.get("sync_sites", []):
                print("SYNC_SITE", n, "\n" + site, flush=True)
    print("RESULT", "OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
