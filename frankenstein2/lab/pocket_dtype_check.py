"""Our Pocket in fp32 vs bf16 (FlowLM transformer weights, activations, KV; in-place batched attention): batched step
time at 1 and 10 rows, content by our own ASR on the same texts and seeds, listening files for both.

usage: python pocket_dtype_check.py --hf-home DIR --ref-wav aria.wav --bundle EARS_BUNDLE --out DIR [--langs it,en,fr]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TEXTS = {
    "it": ["Buongiorno, il suo ordine arriverà giovedì quattordici novembre.", "Abbiamo il divano Fjordby a tre posti.",
           "Mi dispiace, posso aiutarla solo con gli ordini di mobili."],
    "en": ["Good morning, your order will arrive on Thursday, November fourteenth.",
           "We have the Fjordby three-seat sofa.", "Sorry, I can only help with furniture orders."],
    "fr": ["Bonjour, votre commande arrivera jeudi quatorze novembre.", "Nous avons le canapé Fjordby trois places.",
           "Désolé, je ne peux vous aider que pour les commandes de meubles."],
}
NAMES = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l"}


def gen_batch(m, voice, texts, rows, seed0, PocketBatch, torch):  # pragma: no cover - model run
    """rows sessions (texts cycled) stepped together -> (audio per row, step ms list)."""
    b = PocketBatch(m, slots=max(rows, 1), max_len=768)
    gen = lambda i: torch.Generator(device=m.dev).manual_seed(seed0 + i)  # noqa: E731
    outs = {i: [] for i in range(rows)}
    for i in range(rows):
        (prep, after), = m.chunks(texts[i % len(texts)])
        b.start(i, voice, prep, after, gen=gen(i))
    live, ms = set(range(rows)), []
    while live:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        res = b.step(sorted(live))
        torch.cuda.synchronize()
        ms.append((time.monotonic() - t0) * 1e3)
        for k, v in res.items():
            if v is None:
                live.discard(k)
                b.stop(k)
            else:
                outs[k].append(v.float().cpu().numpy())
    return outs, ms


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--langs", default="it,en,fr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["HF_HOME"] = a.hf_home
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from frankenstein2.ears.own import OwnEars
    from frankenstein2.tts.pocket_own import PocketBatch, PocketOwn
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ears = OwnEars(a.bundle, device="cuda", slots=2)
    hub = Path(a.hf_home) / "hub"
    res = {}
    for lang in a.langs.split(","):
        w = sorted(hub.glob(f"models--kyutai--pocket-tts/snapshots/*/languages/{NAMES[lang]}/model.safetensors"))[0]
        tk = sorted(hub.glob(f"models--kyutai--pocket-tts-without-voice-cloning/snapshots/*/languages/{NAMES[lang]}/"
                             "tokenizer.json"))[0]
        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            m = PocketOwn(w, tk, lang=lang, device="cuda", dtype=dt)
            torch.manual_seed(0)
            voice = m.voice_from_file(a.ref_wav)
            outs, ms1 = gen_batch(m, voice, TEXTS[lang], 3, 100, PocketBatch, torch)
            _, ms10 = gen_batch(m, voice, TEXTS[lang], 10, 200, PocketBatch, torch)
            row = {"step_ms_b3_p50": round(float(np.median(ms1[2:])), 2), "step_ms_b10_p50": round(float(np.median(ms10[2:])), 2),
                   "texts": []}
            for i in range(3):
                x = np.concatenate(outs[i]) if outs[i] else np.zeros(1, np.float32)
                sf.write(out / f"{lang}_{name}_{i}.wav", x, 24000, subtype="PCM_16")
                hyp = ears.transcribe(resample_poly(x, 2, 3).astype(np.float32), lang)
                row["texts"].append({"ref": TEXTS[lang][i], "hyp": hyp, "s": round(len(x) / 24000, 2)})
            res[f"{lang}/{name}"] = row
            print(lang, name, json.dumps(row, ensure_ascii=False), flush=True)
            del m
            torch.cuda.empty_cache()
    (out / "pocket_dtype.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    html = ["<html><meta charset=utf-8><body><h3>Pocket fp32 vs bf16 (our runtime)</h3><table border=1>"]
    for lang in a.langs.split(","):
        for i in range(3):
            html.append(f"<tr><td>{lang} {i}</td>" + "".join(
                f"<td>{n}<br><audio controls src='{lang}_{n}_{i}.wav'></audio><br>"
                f"{res[f'{lang}/{n}']['texts'][i]['hyp']}</td>" for n in ("fp32", "bf16")) + "</tr>")
    html.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
