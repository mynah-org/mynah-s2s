"""Batched Pocket (PocketBatch: one FlowLM step for every row, per-row KV slots and offsets) against the single-row
path (PocketOwn.stream), same per-row noise generators: rows joining at different steps, different texts.
usage: python pocket_batch_check.py --hf-home DIR --ref-wav aria.wav [--device cpu] --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TEXTS = ["Buongiorno, il suo ordine arriverà giovedì.", "Abbiamo il divano Fjordby a tre posti.",
         "Mi dispiace, posso aiutarla solo con gli ordini."]


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["HF_HOME"] = a.hf_home
    import numpy as np
    import torch
    import wave
    from frankenstein2.tts.pocket_own import PocketBatch, PocketOwn, resample_poly
    snap = Path(a.hf_home) / "hub/models--kyutai--pocket-tts/snapshots/3e82814a68665eec246ff649b14c71331f955c06"
    tok = sorted((Path(a.hf_home) / "hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots").glob(
        "*/languages/italian_24l/tokenizer.json"))[0]
    m = PocketOwn(snap / "languages/italian_24l/model.safetensors", tok, lang="it", device=a.device)
    w = wave.open(a.ref_wav)
    pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    wav = torch.from_numpy(resample_poly(pcm, 24000, w.getframerate()).astype(np.float32))[None]
    torch.manual_seed(0)
    voice = m.voice_state(wav)
    gen = lambda i: torch.Generator(device=a.device).manual_seed(100 + i)  # noqa: E731
    t0 = time.monotonic()
    single = [torch.cat(list(m.stream(voice, t, gen=gen(i)))).cpu().numpy() for i, t in enumerate(TEXTS)]
    t_single = time.monotonic() - t0
    b = PocketBatch(m, slots=4, max_len=512)
    outs = {i: [] for i in range(3)}
    join = {0: 0, 1: 0, 2: 5}
    t0 = time.monotonic()
    step = 0
    started, live = set(), set()
    while True:
        for i, js in join.items():
            if js == step and i not in started:
                (prep, after), = m.chunks(TEXTS[i])
                b.start(i, voice, prep, after, gen=gen(i))
                started.add(i)
                live.add(i)
        if not live:
            break
        res = b.step(sorted(live))
        for k, v in res.items():
            if v is None:
                live.discard(k)
                b.stop(k)
            else:
                outs[k].append(v.cpu().numpy())
        step += 1
    t_batch = time.monotonic() - t0
    rows = []
    for i in range(3):
        x = np.concatenate(outs[i])
        n = min(len(x), len(single[i]))
        rows.append({"len_single": len(single[i]), "len_batch": len(x),
                     "maxabs": float(np.abs(x[:n] - single[i][:n]).max()) if n else None})
    res = {"rows": rows, "s_single": round(t_single, 2), "s_batch": round(t_batch, 2)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
