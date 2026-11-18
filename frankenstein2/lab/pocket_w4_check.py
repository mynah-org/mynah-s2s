"""Our Pocket with weight-only quantised FlowLM linears (pocket_swap's mode (b) format + fused kernel) and the batched
Mimi decode: per variant (bf16 plain, w4, w8) the batched step time at 3 and 10 rows, FlowLM weight bytes, content by
our own ASR, and listening files; for bf16 also the per-row Mimi decode, to compare audio and step time.

usage: python pocket_w4_check.py --hf-home DIR --ref-wav aria.wav --bundle EARS_BUNDLE --out DIR [--langs it,en,fr]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.lab.pocket_dtype_check import NAMES, TEXTS, gen_batch  # noqa: E402

VARIANTS = (("bf16", "", True), ("bf16_mimi_rows", "", False), ("w4", "w4", True), ("w8", "w8", True))


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
    from frankenstein2.tts import pocket_own
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
        models = {}
        for name, wq, mimi_batch in VARIANTS:
            pocket_own.MIMI_BATCH = mimi_batch
            if wq not in models:
                torch.cuda.empty_cache()
                m0 = torch.cuda.memory_allocated()
                models[wq] = PocketOwn(w, tk, lang=lang, device="cuda", dtype=torch.bfloat16, wq=wq)
                models[wq]._alloc = torch.cuda.memory_allocated() - m0
            m = models[wq]
            torch.manual_seed(0)
            voice = m.voice_from_file(a.ref_wav)
            outs, ms3 = gen_batch(m, voice, TEXTS[lang], 3, 100, PocketBatch, torch)
            _, ms10 = gen_batch(m, voice, TEXTS[lang], 10, 200, PocketBatch, torch)
            row = {"step_ms_b3_p50": round(float(np.median(ms3[2:])), 2),
                   "step_ms_b10_p50": round(float(np.median(ms10[2:])), 2),
                   "alloc_gb": round(m._alloc / 2 ** 30, 3), "bytes": m.nbytes(), "texts": []}
            for i in range(3):
                x = np.concatenate(outs[i]) if outs[i] else np.zeros(1, np.float32)
                sf.write(out / f"{lang}_{name}_{i}.wav", x, 24000, subtype="PCM_16")
                hyp = ears.transcribe(resample_poly(x, 2, 3).astype(np.float32), lang)
                row["texts"].append({"ref": TEXTS[lang][i], "hyp": hyp, "s": round(len(x) / 24000, 2)})
            if name == "bf16_mimi_rows":                    # same latents (same seeds): batched vs per-row Mimi
                diffs = []
                for i in range(3):
                    x0, _ = sf.read(out / f"{lang}_bf16_{i}.wav", dtype="float32")
                    x1, _ = sf.read(out / f"{lang}_bf16_mimi_rows_{i}.wav", dtype="float32")
                    n = min(len(x0), len(x1))
                    e = float(np.sqrt(np.mean((x0[:n] - x1[:n]) ** 2)) / (np.sqrt(np.mean(x1[:n] ** 2)) + 1e-9))
                    diffs.append({"len_eq": len(x0) == len(x1), "rel_rms_diff": round(e, 5)})
                row["mimi_batch_vs_rows"] = diffs
            res[f"{lang}/{name}"] = row
            print("PCHK", lang, name, json.dumps(row, ensure_ascii=False), flush=True)
        del models
        torch.cuda.empty_cache()
    (out / "pocket_w4.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    names = [v[0] for v in VARIANTS]
    html = ["<html><meta charset=utf-8><body><h3>Pocket (our runtime): bf16 / w4 / w8, batched Mimi</h3><table border=1>"]
    for lang in a.langs.split(","):
        for i in range(3):
            html.append(f"<tr><td>{lang} {i}</td>" + "".join(
                f"<td>{n}<br><audio controls src='{lang}_{n}_{i}.wav'></audio><br>"
                f"{res[f'{lang}/{n}']['texts'][i]['hyp']}</td>" for n in names) + "</tr>")
    html.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
