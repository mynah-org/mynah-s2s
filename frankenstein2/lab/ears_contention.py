"""Where do the ears' milliseconds go when the LLM decodes next to them? N sessions (it/en/fr tracks on the mic, half
of them with a speaking agent channel) hopped together, back to back; per phase (window encode, GRU, streaming ASR)
GPU-synchronised times, alone and with a background thread running batched Qwen decode on its own CUDA stream (as
the one-process engine's LLM thread), with the eager greedy RNN-T loop and with its CUDA graph. Also checks that both
greedy paths give the same pieces.

usage: python ears_contention.py --bundle DIR --asr-dir DIR --qwen DIR --out out.json [--n 10] [--hops 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--qwen-mode", default="int8")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--hops", type=int, default=40)
    ap.add_argument("--skip", type=int, default=26)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2 import gpu
    from frankenstein2.ears.own import HOP, OwnEars
    from frankenstein2.llm.qwen_own.engine import QwenOwn
    torch.set_grad_enabled(False)
    tracks = {l: sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0] for l in ("it", "en", "fr")}
    langs = ["it", "en", "fr"]
    ears = OwnEars(a.bundle, device="cuda", slots=a.n + 2, window_precision="bf16")
    t0 = time.monotonic()
    ng = ears.asr.warm_greedy_graphs()
    print("greedy graphs", ng, round(time.monotonic() - t0, 1), "s", flush=True)
    q = QwenOwn(a.qwen, mode=a.qwen_mode, slots=a.n + 2, max_len=4096)
    sess = []
    for i in range(a.n):
        s = q.open()
        q.prefill(s, q.chat_ids([{"role": "user", "content": "Tell me a long story about a furniture store."}]))
        sess.append(s)
    q.warm_graphs(4096)
    q.stream = torch.cuda.Stream()
    stop = threading.Event()
    llm_steps = [0]

    def llm_loop():
        with torch.inference_mode(), torch.cuda.stream(q.stream):
            toks = [1] * a.n
            while not stop.is_set():
                for s in sess:                                   # keep the sessions short of the cache end
                    if len(q.sess[s]["ids"]) > 3500:
                        q.sess[s]["ids"] = q.sess[s]["ids"][:200]
                logits = q.decode_batch(sess, toks)
                toks = gpu.host(logits.argmax(-1))
                llm_steps[0] += 1

    def run(tag, graphs, with_llm, polite=True, cap_s=120.0):
        gpu.POLITE = polite
        ears.greedy_graphs = graphs
        for sid in list(ears.sess):
            ears.close(sid)
        for i in range(a.n):
            ears.open(i, langs[i % 3])
        ears.prof = None
        th = None
        if with_llm:
            stop.clear()
            th = threading.Thread(target=llm_loop, daemon=True)
            th.start()
        z = np.zeros(HOP, np.float32)
        rounds, pieces = [], {i: [] for i in range(a.n)}
        st0 = llm_steps[0]
        tl0 = time.monotonic()
        t_start = time.monotonic()
        for h in range(a.skip + a.hops):
            if time.monotonic() - t_start > cap_s:
                print("CAP", tag, "hops done", h, flush=True)
                break
            if h == a.skip:
                ears.prof = {}
                st0, tl0 = llm_steps[0], time.monotonic()
            items = []
            for i in range(a.n):
                x = tracks[langs[i % 3]]
                off = (h * HOP + i * 7 * HOP) % (len(x) - HOP)
                y = tracks[langs[(i + 1) % 3]]
                agent = y[off:off + HOP] if i % 2 else z
                items.append((i, x[off:off + HOP], agent, "listen", False))
            tr = time.monotonic()
            res = ears.hop(items)
            if h >= a.skip:
                rounds.append((time.monotonic() - tr) * 1e3)
            for i, r in enumerate(res):
                pieces[i] += [p["id"] for p in r["pieces"]]
        llm_rate = (llm_steps[0] - st0) / (time.monotonic() - tl0)
        if th is not None:
            stop.set()
            th.join()
        pc = lambda v: {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}  # noqa: E731
        row = {k: pc(v) for k, v in (ears.prof or {}).items() if v}
        row["round_ms"] = pc(rounds) if rounds else None
        row["llm_steps_per_s"] = round(llm_rate, 1) if with_llm else None
        print("EARS", tag, json.dumps(row), flush=True)
        return row, pieces

    res = {}
    res["alone_eager"], p_e = run("alone_blocking", False, False, polite=False)
    res["alone_polite"], p_g = run("alone_polite", False, False, polite=True)
    res["pieces_equal"] = p_e == p_g
    res["llm_polite"], p_p = run("llm_polite", False, True, polite=True)
    res["pieces_equal_under_llm"] = p_p == p_e
    res["llm_blocking"], _ = run("llm_blocking", False, True, polite=False, cap_s=60.0)
    gpu.POLITE = True
    t = time.monotonic()
    n = 0
    with torch.cuda.stream(q.stream):
        while time.monotonic() - t < 5:
            q.decode_batch(sess, [1] * a.n)
            torch.cuda.synchronize()
            n += 1
    res["llm_alone_steps_per_s"] = round(n / (time.monotonic() - t), 1)
    print("LLM alone steps/s", res["llm_alone_steps_per_s"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
