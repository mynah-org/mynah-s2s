"""Per-component timings of the one-process engine's models at N concurrent sessions, in one process (the way the
scheduler runs them): OwnEars hop (N sessions, real speech), PocketBatch step (N rows), Qwen decode_batch (N slots).
Memory (VRAM, peak RSS) after each load.
usage: python runtime_bench.py --ears-bundle DIR --asr-dir DIR --pocket-weights DIR --pocket-tokenizers DIR
       --voice ref.wav --qwen-model DIR [--qwen-mode int8] [--ns 1,4,10] --out out.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--ears-bundle", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--pocket-weights", required=True)
    ap.add_argument("--pocket-tokenizers", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--qwen-model", default="")
    ap.add_argument("--qwen-mode", default="int8")
    ap.add_argument("--ns", default="1,4,10")
    ap.add_argument("--hops", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.own import OwnEars
    from frankenstein2.tts.pocket_own import PocketBatch, PocketOwn
    ns = [int(x) for x in a.ns.split(",")]
    res = {"device": torch.cuda.get_device_name(0), "mem": {}}

    def mem(tag):
        torch.cuda.synchronize()
        res["mem"][tag] = {"alloc_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3),
                           "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                           "rss_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
        print("MEM", tag, res["mem"][tag], flush=True)

    def pc(x):
        return {"p50": round(float(np.percentile(x, 50)), 1), "p95": round(float(np.percentile(x, 95)), 1),
                "max": round(float(np.max(x)), 1)}
    with torch.inference_mode():
        mem("start")
        ears = OwnEars(a.ears_bundle, device="cuda", slots=max(ns))
        mem("ears")
        tracks = {}
        for lang in ("it", "en", "fr"):
            x, _ = sf.read(Path(a.asr_dir) / f"track_{lang}.wav", dtype="float32")
            tracks[lang] = x
        res["ears"] = {}
        for n in ns:
            sids = [f"s{n}_{i}" for i in range(n)]
            langs = [("it", "en", "fr")[i % 3] for i in range(n)]
            for s, l in zip(sids, langs):
                ears.open(s, l)
            ms, pieces = [], 0
            z = np.zeros(5120, np.float32)
            for h in range(a.hops):
                items = [(s, tracks[l][(h + 3 * i) * 5120:(h + 3 * i + 1) * 5120], z, "listen", False)
                         for i, (s, l) in enumerate(zip(sids, langs))]
                torch.cuda.synchronize()
                t0 = time.monotonic()
                out = ears.hop(items)
                torch.cuda.synchronize()
                if h >= 4:
                    ms.append((time.monotonic() - t0) * 1e3)
                pieces += sum(len(o["pieces"]) for o in out)
            res["ears"][n] = {"hop_ms": pc(ms), "pieces": pieces, "realtime_320ms": float(np.percentile(ms, 95)) < 320}
            print("EARS", n, res["ears"][n], flush=True)
            for s in sids:
                ears.close(s)
        mem("ears_run")
        snap = Path(a.pocket_weights)
        tk = sorted(Path(a.pocket_tokenizers).glob("*/languages/italian_24l/tokenizer.json"))[0]
        pk = PocketOwn(snap / "languages/italian_24l/model.safetensors", tk, lang="it", device="cuda")
        mem("pocket_it")
        voice = pk.voice_from_file(a.voice)
        mem("pocket_voice")
        res["pocket"] = {}
        text = "Buongiorno, il suo ordine arriverà giovedì e il corriere la chiamerà prima della consegna."
        (prep, after), = pk.chunks(text)[:1]
        for n in ns:
            b = PocketBatch(pk, slots=n, max_len=512)
            t0 = time.monotonic()
            for i in range(n):
                b.start(i, voice, prep, after, gen=torch.Generator(device="cuda").manual_seed(i))
            torch.cuda.synchronize()
            t_pref = (time.monotonic() - t0) * 1e3 / n
            ms = []
            live = list(range(n))
            for st in range(40):
                torch.cuda.synchronize()
                t0 = time.monotonic()
                r = b.step(live)
                torch.cuda.synchronize()
                ms.append((time.monotonic() - t0) * 1e3)
                live = [k for k in live if r.get(k) is not None]
                if not live:
                    break
            res["pocket"][n] = {"step_ms": pc(ms[2:]), "prefill_ms_per_row": round(t_pref, 1),
                                "realtime_80ms": float(np.percentile(ms[2:], 95)) < 80}
            print("POCKET", n, res["pocket"][n], flush=True)
            for i in range(n):
                b.stop(i)
            del b
        mem("pocket_run")
        if a.qwen_model:
            from frankenstein2.llm.qwen_own.engine import QwenOwn
            q = QwenOwn(a.qwen_model, device="cuda", mode=a.qwen_mode, slots=max(ns), max_len=4096)
            mem("qwen")
            res["qwen"] = {"footprint": q.footprint()}
            msgs = [{"role": "system", "content": "You are a helpful shop assistant. Answer briefly."},
                    {"role": "user", "content": "Hello, I would like to know when my order arrives."}]
            ids = q.chat_ids(msgs)
            for n in ns:
                ss = [q.open() for _ in range(n)]
                t0 = time.monotonic()
                toks = []
                for s in ss:
                    q.prefill_begin(s, ids)
                    out = None
                    while out is None:
                        out = q.prefill_step(s, 256)
                    toks.append(int(out.argmax()))
                torch.cuda.synchronize()
                t_pref = (time.monotonic() - t0) * 1e3 / n
                ms = []
                for st in range(40):
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    lg = q.decode_batch(ss, toks)
                    toks = lg.argmax(-1).tolist()
                    torch.cuda.synchronize()
                    ms.append((time.monotonic() - t0) * 1e3)
                res["qwen"][n] = {"decode_ms": pc(ms[3:]), "prefill_ms_per_prompt": round(t_pref, 1),
                                  "prompt_tokens": len(ids), "tok_s_per_session": round(1e3 / np.percentile(ms[3:], 50), 1)}
                print("QWEN", n, res["qwen"][n], flush=True)
                for s in ss:
                    q.close(s)
            mem("qwen_run")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
