"""The ears' full-window encode: CUDA graph replay vs eager (identical outputs expected), and its time per round of
N sessions (2N rows: mic + agent) for fp32 / tf32 / bf16 window precision.

usage: python ears_graph_check.py --bundle DIR --asr-dir DIR --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.own import WINDOW, OwnEars
    torch.set_grad_enabled(False)
    x = np.concatenate([sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0] for l in ("it", "en", "fr")])
    res = {}
    for prec in ("fp32", "tf32", "bf16"):
        e = OwnEars(a.bundle, device="cuda", slots=16, window_precision=prec)
        t0 = time.monotonic()
        n = e.warm_window_graphs(32)
        res[prec] = {"graphs": n, "capture_s": round(time.monotonic() - t0, 1)}
        for N in (1, 4, 10, 16):
            rows = torch.from_numpy(np.stack([x[i * 16000: i * 16000 + WINDOW] for i in range(2 * N)])).cuda()
            ref = e._encode_window_eager(rows)
            got = e._encode_window(rows)
            diff = float((ref - got).abs().max())
            ts = {}
            for name, fn in (("eager", e._encode_window_eager), ("graph", e._encode_window)):
                for _ in range(2):
                    fn(rows)
                torch.cuda.synchronize()
                t = time.monotonic()
                for _ in range(10):
                    fn(rows)
                torch.cuda.synchronize()
                ts[name] = round((time.monotonic() - t) * 100, 2)          # ms per call
            res[prec][f"N{N}"] = {"maxdiff_graph_vs_eager": diff, "ms": ts}
            print(prec, N, json.dumps(res[prec][f"N{N}"]), flush=True)
        del e
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
