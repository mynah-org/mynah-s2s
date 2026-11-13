"""Per-component cost of one 160 ms chunk on the native engine (P1 profiling, before any optimisation).

Sources, all observational (no upstream edit):
  * NVIDIA's own per-component timers in the wrapper (``Time taken for perception / stt_model / tts_model /
    audio_codec / one step``), each preceded by ``torch.cuda.synchronize()`` in NVIDIA's code — the syncs are part of
    the reference path and are therefore included in the totals;
  * CUDA events around the NemotronH backbone ``llm.forward`` (splits the Thinker backbone from the rest of
    ``stt_model``: embeddings, heads, RNN-T, turn-taking logic);
  * wall time of ``ReferenceEngine.step`` (everything: framing, Python glue, state bookkeeping, copies to CPU).
Residual = step wall - (perception + stt_model + tts_model + audio_codec) = glue/CPU/sync overhead.

Phases reported separately: warmup (first ``--warmup`` chunks), steady (rest), first vs last quarter (growth), and FC
fast-path chunks (``fc_active``) when the input contains a tool call.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics as S
import sys
import time
from pathlib import Path

import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k, render_prompt  # noqa: E402

RX = re.compile(r"Time taken for (perception|stt_model|tts_model|audio_codec|one step): ([0-9.]+)s")


class Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.cur: dict[str, float] = {}

    def emit(self, rec):
        m = RX.search(rec.getMessage())
        if m:
            k = m.group(1).replace(" ", "_")
            self.cur[k] = self.cur.get(k, 0.0) + float(m.group(2)) * 1000.0

    def take(self) -> dict:
        d, self.cur = self.cur, {}
        return d


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "p50": round(S.median(xs), 2), "p95": round(xs[int(0.95 * (len(xs) - 1))], 2),
            "mean": round(sum(xs) / len(xs), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pad-silence", type=float, default=0.0)
    ap.add_argument("--no-llm-cache", action="store_true")
    ap.add_argument("--fc-prompt", action="store_true", help="NVIDIA FC tools prompt (to profile the FC fast path)")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="P1 experiment: torch.backends.cudnn.benchmark=True (codec convolutions); not a REFERENCE setting")
    a = ap.parse_args()
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")), llm_cache=not a.no_llm_cache)
    grab = Grab()
    for name in ("nemo_logger", "NeMo", ""):
        logging.getLogger(name).addHandler(grab)
    llm = eng.wrapper.model.stt_model.llm
    inner = llm.forward
    llm_ms: list[float] = []

    def timed_llm(*args, **kw):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        r = inner(*args, **kw)
        e.record()
        e.synchronize()
        llm_ms.append(s.elapsed_time(e))
        return r
    timed_llm.__wrapped_v2__ = getattr(inner, "__wrapped_v2__", inner)
    llm.forward = timed_llm

    if a.fc_prompt:
        import offline_voicechat_fc_infer as fcmod
        prompt = render_prompt(fcmod.DEFAULT_TOOLS)
    else:
        prompt = "You are a helpful assistant."
    report = {"llm_cache": eng.llm_cache_info, "inputs": []}
    for w in a.wavs:
        x = load_16k(w)
        if a.pad_silence:
            x = torch.cat([x, torch.zeros(int(a.pad_silence * 16000))])
        chunks = list(iter_chunks(x, eng.chunk_samples))
        eng.open(prompt)
        grab.take()
        rows = []
        torch.cuda.reset_peak_memory_stats()
        for i, c in enumerate(chunks):
            llm_ms.clear()
            t0 = time.monotonic_ns()
            o = eng.step(c, is_last=(i == len(chunks) - 1))
            wall = (time.monotonic_ns() - t0) / 1e6
            comp = grab.take()
            known = sum(comp.get(k, 0.0) for k in ("perception", "stt_model", "tts_model", "audio_codec"))
            rows.append({"i": i, "wall": wall, "fc": bool(o.fc_active), **comp,
                         "llm_backbone": sum(llm_ms) if llm_ms else None, "llm_calls": len(llm_ms),
                         "stt_rest": (comp.get("stt_model", 0.0) - sum(llm_ms)) if llm_ms and "stt_model" in comp else None,
                         "residual": wall - known if comp else None})
        eng.close()
        n = len(rows)
        q = max(1, n // 4)
        normal = [r for r in rows if not r["fc"]]

        def phase(rs):
            return {k: stats([r.get(k) for r in rs]) for k in
                    ("wall", "one_step", "perception", "stt_model", "llm_backbone", "stt_rest", "tts_model",
                     "audio_codec", "residual")}
        rep = {"wav": w, "chunks": n, "fc_chunks": n - len(normal),
               "warmup": phase(normal[:a.warmup]), "steady": phase(normal[a.warmup:]),
               "first_quarter": phase(normal[a.warmup:a.warmup + q]), "last_quarter": phase(normal[-q:]),
               "fc": phase([r for r in rows if r["fc"]]),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "rtf_steady_p50": None}
        st = rep["steady"]["wall"]
        if st:
            rep["rtf_steady_p50"] = round(st["p50"] / (1000 * eng.chunk_samples / 16000), 3)
        report["inputs"].append(rep)
        (out / f"{Path(w).stem}_rows.json").write_text(json.dumps(rows))
        print(json.dumps({"wav": Path(w).name, "chunks": n,
                          "steady_p50": {k: (v or {}).get("p50") for k, v in rep["steady"].items()},
                          "rtf": rep["rtf_steady_p50"], "fc_wall": rep["fc"]["wall"]}), flush=True)
    (out / "profile.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
