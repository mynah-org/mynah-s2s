"""Is a voice cloned from cached Mimi latents the same as one cloned from scratch, and how much does the talker's
own sampling move the ears' score? (GPU, the real models)

    python frankenstein2/lab/voice_cache_check.py <engine args> --langs it,cs --repeats 4 --out DIR

Per language: the prompted FlowLM KV cache from the cached latents vs from the WAV (max abs difference over the
filled part), then the suite's question spoken ``--repeats`` times and transcribed by the ears (character ratio per
take). Prints one ``VOICE`` line per language.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frankenstein"))


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from frankenstein2.ears.client import f32b64
    from frankenstein2.lab.lang_suite import QUESTION, char_ratio
    from frankenstein2.server import build as B
    from frankenstein2.tts.pocket_own import SR, end_on_pause
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--langs", default="it,cs")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    a.pocket_preload = ""
    ears = B.start_ears(a, out)
    talker = B.start_pocket(a, out)
    sched = a._single["sched"]
    vwav = next(iter(B.voices(a).values()))
    for lang in [x for x in a.langs.split(",") if x]:
        def job():
            L = talker.langs.ensure(talker.names(lang, None), 300, talker._in_use)
            cached = L.model.voice_from_file(vwav)              # latents from the cache (another language's encode)
            timing = dict(getattr(L.model, "voice_timing", {}))
            x, sr = sf.read(vwav, dtype="float32")
            x = x if x.ndim == 1 else x[:, 0]
            x = resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
            fresh = L.model.voice_state(torch.from_numpy(x)[None])  # from the WAV, no cache
            n = min(cached.off, fresh.off)
            diff = (cached.c[:, :, :n].float() - fresh.c[:, :, :n].float()).abs().max().item()
            takes = []
            for _ in range(a.repeats):
                with torch.no_grad():
                    takes.append(torch.cat(list(L.model.stream(cached, QUESTION[lang]))).float().cpu().numpy())
            return {"off": [cached.off, fresh.off], "max_abs_diff": diff, "timing": timing}, takes
        r, takes = sched.call(job, timeout=900)
        r["lang"], r["ratios"], r["heard"] = lang, [], []
        for i, y in enumerate(takes):
            sf.write(out / f"{lang}_{i}.wav", y, 24000, subtype="PCM_16")
            t = ears.request({"op": "transcribe", "id": f"{lang}{i}", "lang": lang,
                              "pcm": f32b64(resample_poly(y, 2, 3).astype(np.float32))}, "transcript", 120)
            r["ratios"].append(char_ratio(t["text"], QUESTION[lang]))
            r["heard"].append(t["text"][:120])
        print("VOICE", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
