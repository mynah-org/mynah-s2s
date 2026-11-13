"""Short GPU check of RendezvousEngine(batch_codec=True) on the real model: N EN no-tool sessions, unpaced rounds,
run twice in one process (same MultiSessionEngine + BatchedThinker): per-session codec decode (off) then batched (on).
Per session: text / function ids identical (codec output never feeds the model) and agent audio max |diff| (expected
~1e-6, not 0: batch-n kernels). Writes <out>/codec_batch_check.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.model.session_state import MultiSessionEngine
from mynah_v2.model.thinker_batch import BatchedThinker
from mynah_v2.runtime.rendezvous import RendezvousEngine
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def run(eng, m, bt, wavs, batch_codec, tag, n_chunks):
    rz = RendezvousEngine(m, bt, batch_codec=batch_codec)
    sids = {f"{tag}{i}": w for i, w in enumerate(wavs)}
    for i, sid in enumerate(sids):
        random.seed(1234 + i)
        np.random.seed(1234 + i)
        rz.open(sid, "You are a helpful assistant.", seed=1234 + i)
    ch = {sid: list(iter_chunks(w, eng.chunk_samples))[:n_chunks] for sid, w in sids.items()}
    res = {sid: {"text": [], "func": [], "audio": []} for sid in sids}
    walls = []
    for k in range(n_chunks):
        t0 = time.perf_counter()
        out = rz.step_jobs({sid: c[k] for sid, c in ch.items()})
        walls.append((time.perf_counter() - t0) * 1e3)
        for sid, o in out.items():
            if isinstance(o, BaseException):
                raise o
            res[sid]["text"] += list(o.new_text_ids)
            res[sid]["func"] += list(o.new_func_ids)
            res[sid]["audio"].append(o.audio.float().reshape(-1))
    stats = {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
             for k, v in rz.stats.get("codec", {}).items()}
    rz.close_all()
    for r in res.values():
        r["audio"] = torch.cat(r["audio"])
    walls.sort()
    return list(res.values()), stats, {"p50": round(walls[len(walls) // 2], 1), "p95": round(walls[int(.95 * (len(walls) - 1))], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--items", default="corpus_v1/q01,corpus_v1/q05")
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    for n in ("mamba_host_pos", "no_timer_sync"):
        perf.enable(n, eng)
    wavs = [load_16k(Path(a.audio_root) / f"{i}.wav") for i in a.items.split(",")]
    m = MultiSessionEngine(eng)
    bt = BatchedThinker(eng, max_rows=len(wavs), lmax=4096, graphs=False)
    off, s_off, w_off = run(eng, m, bt, wavs, False, "off", a.chunks)
    on, s_on, w_on = run(eng, m, bt, wavs, True, "on", a.chunks)
    per = []
    for i, (x, y) in enumerate(zip(off, on)):
        n = min(len(x["audio"]), len(y["audio"]))
        d = (x["audio"][:n] - y["audio"][:n]).abs()
        per.append({"item": a.items.split(",")[i], "text_identical": x["text"] == y["text"],
                    "func_identical": x["func"] == y["func"], "n_text": len(x["text"]),
                    "audio_len": [len(x["audio"]), len(y["audio"])], "audio_max_abs_diff": float(d.max()) if n else None,
                    "audio_rms": float(x["audio"].pow(2).mean().sqrt()) if n else None,
                    "agent_tokens_non_pad": sum(t not in (0, 1, 2, 12) for t in x["text"])})
    rep = {"chunks": a.chunks, "items": a.items, "codec_stats_on": s_on, "codec_stats_off": s_off,
           "round_ms_off": w_off, "round_ms_on": w_on, "sessions": per}
    (out / "codec_batch_check.json").write_text(json.dumps(rep, indent=1))
    print("CODEC_BATCH_CHECK", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
