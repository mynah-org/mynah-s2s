"""G1: the Flash talker standalone (no engine, no Nano, no ASR on the GPU), driven through the same process protocol
as in the loop (e1/flash_server.py via e1/flash_client.FlashProc), on the texts the G0 session actually spoke.

Per variant (``generate`` = the package's HF generate for the code predictor, ``fast`` = the plain loop), one talker
process: every text is sent as start + final with ALL text available, one segment at a time (the next one starts
when the previous is done). Measured: talker step ms (backbone + code predictor), code predictor ms, codec decode ms
per chunk, prefill ms, codec frames per second of generation, first-chunk latency, and the codes of every segment
(fast must equal generate). Also checks the talker statistics arrive at close (the G0 loss, fixed in 3ec365e).

  python e1/g1_talker.py --g0-summary RUN/g0_it/summary.json --out NEW_DIR --flash-python ... --flash-harness ...
         --flash-root ... --flash-voice ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stats(xs):
    xs = sorted(xs)
    q = lambda p: round(xs[int(p * (len(xs) - 1))], 2) if xs else None   # noqa: E731
    return {"n": len(xs), "p50": q(.5), "p95": q(.95), "mean": round(sum(xs) / len(xs), 2) if xs else None}


def run_variant(a, texts, extra, log):  # pragma: no cover - GPU
    from e1.flash_client import FlashProc
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    p = FlashProc([a.flash_python, str(Path(__file__).resolve().parent / "flash_server.py"), "--harness",
                   a.flash_harness, "--root", a.flash_root, "--voice", a.flash_voice, "--lang", "it", *extra], env, log)
    p.start()
    ready = p.wait_ready(240)
    segs = []
    for i, text in enumerate(texts):
        t_send = time.monotonic_ns()
        p.submit_raw({"op": "start", "epoch": 0, "turn": i + 1, "seg": 0, "text": text})
        p.submit_raw({"op": "final", "epoch": 0, "turn": i + 1, "seg": 0})
        first, frames, done = None, 0, None
        t_end = time.monotonic() + 120
        while done is None and time.monotonic() < t_end:
            for ev in p.poll():
                if ev.get("ev") == "audio" and ev["turn"] == i + 1:
                    first = first or ev["t_ns"]
                    frames = ev["b"]
                elif ev.get("ev") == "done" and ev["turn"] == i + 1:
                    done = ev
            time.sleep(0.005)
        if done is None:
            segs.append({"text": text, "error": "timeout"})
            break
        gen_s = (done["t_ns"] - t_send) / 1e9
        segs.append({"text": text, "frames": done["frames"], "ended": done["ended"], "gen_s": round(gen_s, 3),
                     "fps": round(done["frames"] / gen_s, 2) if gen_s > 0 else None,
                     "first_chunk_ms": round((first - t_send) / 1e6, 1) if first else None, "codes": done["codes"]})
    rest = p.close()
    st = p.stats or {}
    frames = sum(s.get("frames", 0) for s in segs)
    gen = sum(s.get("gen_s", 0) for s in segs)
    return {"ready": ready, "stats_received": p.stats is not None, "close_events": len(rest),
            "step_ms": _stats(st.get("step_ms", [])), "cp_ms": _stats(st.get("cp_ms", [])),
            "decode_ms": _stats(st.get("decode_ms", [])), "prefill_ms": _stats(st.get("prefill_ms", [])),
            "frames": frames, "gen_s": round(gen, 2), "fps": round(frames / gen, 2) if gen else None,
            "first_chunk_ms": _stats([s["first_chunk_ms"] for s in segs if s.get("first_chunk_ms")]), "segments": segs}


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flash-python", required=True)
    ap.add_argument("--flash-harness", required=True)
    ap.add_argument("--flash-root", required=True)
    ap.add_argument("--flash-voice", required=True)
    ap.add_argument("--variants", default="generate,fast")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    texts = [t["text"] for t in json.load(open(a.g0_summary))["spoken_turns"] if t["text"].strip()]
    extra = {"generate": ["--profile"], "fast": ["--fast-cp", "--profile"], "fast_noprof": ["--fast-cp"],
             "generate_noprof": []}
    res = {"texts": texts}
    for v in a.variants.split(","):
        t0 = time.monotonic()
        res[v] = run_variant(a, texts, extra[v], str(out / f"flash_{v}.log"))
        r = res[v]
        print("G1", v, json.dumps({k: r[k] for k in r if k not in ("segments", "ready")}), round(time.monotonic() - t0),
              "s", flush=True)
    if "generate" in res and "fast" in res:
        a_, b_ = res["generate"]["segments"], res["fast"]["segments"]
        same = [x.get("codes") == y.get("codes") for x, y in zip(a_, b_)]
        res["codes_identical"] = {"segments": len(same), "identical": sum(same)}
        print("G1 CODES", json.dumps(res["codes_identical"]), flush=True)
    (out / "g1.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
