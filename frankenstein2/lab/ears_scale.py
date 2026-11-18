"""How many concurrent sessions the ears keep in real time (GPU): N sessions (different languages in turn: it, en,
fr) fed the session tracks hop by hop; per round of N hops the wall time sequential (one hop after the other) and
batched (hop_batch: one window encode for all sessions), against the 320 ms hop budget; CUDA memory per session.

usage: python ears_scale.py --bundle DIR --asr-dir DIR --out out.json [--ns 1,2,4,8,10] [--hops 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="1,2,4,8,10")
    ap.add_argument("--hops", type=int, default=40)
    ap.add_argument("--skip", type=int, default=30, help="hops fed first (window filled to 8 s+)")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.rules import Thresholds
    from frankenstein2.ears.server import PROMPTS
    from frankenstein2.ears.stream import HOP, EarsSession, hop_batch
    from frankenstein2.ears.turntaker import load_bundle
    torch.set_grad_enabled(False)
    model, proc, tt, _ = load_bundle(a.bundle, device="cuda")
    tracks = {l: sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0] for l in ("it", "en", "fr")}
    langs = ["it", "en", "fr"]
    res = {"base_cuda_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3)}
    for n in [int(x) for x in a.ns.split(",")]:
        row = {}
        for mode in ("sequential", "batched"):
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated()
            ss = [EarsSession(model, proc, tt, "cuda", PROMPTS[langs[i % 3]], Thresholds()) for i in range(n)]
            z = np.zeros(HOP, np.float32)
            times = []
            for h in range(a.skip + a.hops):
                items = []
                for i, s in enumerate(ss):
                    x = tracks[langs[i % 3]]
                    off = (h * HOP + i * 7 * HOP) % (len(x) - HOP)
                    items.append((s, x[off:off + HOP], z, "listen", False))
                t0 = time.monotonic()
                if mode == "batched":
                    hop_batch(items)
                else:
                    for s, u, ag, md, r in items:
                        s.hop(u, ag, md, r)
                torch.cuda.synchronize()
                if h >= a.skip:
                    times.append((time.monotonic() - t0) * 1e3)
            xs = sorted(times)
            row[mode] = {"round_ms_p50": round(xs[len(xs) // 2], 1), "round_ms_p95": round(xs[int(.95 * (len(xs) - 1))], 1),
                         "realtime": xs[int(.95 * (len(xs) - 1))] < 320.0,
                         "cuda_per_session_mb": round((torch.cuda.memory_allocated() - m0) / 2 ** 20 / n, 1),
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
            del ss
            torch.cuda.empty_cache()
        res[str(n)] = row
        print("N", n, json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
