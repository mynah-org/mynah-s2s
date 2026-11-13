"""P0 performance: attribute every host<->device round trip of the native step to its call site.

Process-local wrappers (audit process only, nothing upstream edited) on the calls that block the host on the GPU:
``Tensor.item / tolist / cpu / numpy / __bool__ / __int__ / __float__ / __index__`` (on CUDA tensors) and
``torch.cuda.synchronize``. For each call we record the first stack frame outside torch/our wrapper (file:line +
function) and the host time spent inside the call (= time the CPU waited for the GPU queue to drain, plus the copy).
Reported per 160 ms chunk over a steady window: count, blocked ms, and the source line, grouped by call site.
"""

from __future__ import annotations

import argparse
import collections
import json
import linecache
import sys
import time
import traceback
from pathlib import Path

import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k, render_prompt  # noqa: E402

STATS: dict = collections.defaultdict(lambda: [0, 0.0])
ACTIVE = {"on": False}
TORCH_DIR = str(Path(torch.__file__).parent)


def site() -> str:
    for fr in reversed(traceback.extract_stack()[:-2]):   # drop site() and the wrapper only
        f = fr.filename
        if f.startswith(TORCH_DIR) or f.endswith("sync_audit.py"):
            continue
        short = "/".join(f.split("/")[-2:])
        return f"{short}:{fr.lineno} {fr.name} | {(linecache.getline(f, fr.lineno) or '').strip()[:110]}"
    return "?"


def wrap_method(name: str):
    orig = getattr(torch.Tensor, name)

    def w(self, *a, **k):
        if not ACTIVE["on"] or not self.is_cuda:
            return orig(self, *a, **k)
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        s = STATS[(name, site())]
        s[0] += 1
        s[1] += (time.perf_counter() - t0) * 1000
        return r
    setattr(torch.Tensor, name, w)


def wrap_sync():
    orig = torch.cuda.synchronize

    def w(*a, **k):
        if not ACTIVE["on"]:
            return orig(*a, **k)
        t0 = time.perf_counter()
        r = orig(*a, **k)
        s = STATS[("cuda.synchronize", site())]
        s[0] += 1
        s[1] += (time.perf_counter() - t0) * 1000
        return r
    torch.cuda.synchronize = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--fc-prompt", action="store_true")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)          # immutable run directory
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    for n in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__", "__float__", "__index__"):
        wrap_method(n)
    wrap_sync()
    if a.fc_prompt:
        import offline_voicechat_fc_infer as fcmod
        prompt = render_prompt(fcmod.DEFAULT_TOOLS)
    else:
        prompt = "You are a helpful assistant."
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 30)])
    chunks = list(iter_chunks(x, eng.chunk_samples))[: a.warmup + a.chunks]
    eng.open(prompt)
    for c in chunks[: a.warmup]:
        eng.step(c)
    ACTIVE["on"] = True
    walls = []
    for c in chunks[a.warmup:]:
        t0 = time.perf_counter()
        eng.step(c)
        walls.append((time.perf_counter() - t0) * 1000)
    ACTIVE["on"] = False
    eng.close()
    n = len(walls)
    rows = sorted(({"kind": k[0], "site": k[1], "per_chunk": round(v[0] / n, 2), "blocked_ms_per_chunk": round(v[1] / n, 3)}
                   for k, v in STATS.items()), key=lambda r: -r["blocked_ms_per_chunk"])
    tot = {"chunks": n, "wall_ms_p50": sorted(walls)[n // 2],
           "calls_per_chunk": round(sum(r["per_chunk"] for r in rows), 1),
           "blocked_ms_per_chunk": round(sum(r["blocked_ms_per_chunk"] for r in rows), 2)}
    (out / "sync_audit.json").write_text(json.dumps({"total": tot, "sites": rows}, indent=1))
    print(json.dumps(tot), flush=True)
    for r in rows[:45]:
        print(f'{r["per_chunk"]:7.2f}/ch {r["blocked_ms_per_chunk"]:8.3f} ms  {r["kind"]:16s} {r["site"]}', flush=True)


if __name__ == "__main__":
    main()
