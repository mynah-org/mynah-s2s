"""Our Pocket runtime (tts/pocket_own.py) against the pocket-tts package (test reference only): voice prompt state,
generated latents and audio with the same seeds, per language (it / fr / en 24-layer cloning checkpoints).
usage: python pocket_check.py --hf-home DIR --ref-wav aria.wav [--device cpu] [--langs it,fr,en] --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TEXT = {"it": "Buongiorno, il suo ordine arriverà giovedì. Posso aiutarla in altro?",
        "fr": "Bonjour, votre commande arrivera jeudi. Puis-je vous aider ?",
        "en": "Hello, your order will arrive on Thursday. Can I help with anything else?"}
LANG = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l"}


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--langs", default="it,fr,en")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["HF_HOME"] = a.hf_home
    import numpy as np
    import torch
    import wave
    torch.backends.mkldnn.enabled = False
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import download_if_necessary
    from pocket_tts.utils.config import load_config
    from pocket_tts.models.tts_model import CONFIGS_DIR
    from frankenstein2.tts.pocket_own import PocketOwn, resample_poly
    res = {}
    w = wave.open(a.ref_wav)
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    for lang in a.langs.split(","):
        r = {}
        ref = TTSModel.load_model(language=LANG[lang]).to(a.device)
        cfg = load_config(str(CONFIGS_DIR / f"{LANG[lang]}.yaml"))
        wpath = download_if_necessary(cfg.weights_path)
        tpath = download_if_necessary(cfg.flow_lm.lookup_table.tokenizer_path)
        own = PocketOwn(wpath, tpath, lang=lang, device=a.device)
        # resampling of the prompt (ours vs scipy)
        from scipy.signal import resample_poly as rp
        g = np.gcd(sr, 24000)
        x_ref = rp(pcm, 24000 // g, sr // g)
        x_own = resample_poly(pcm, 24000, sr)
        r["resample_maxabs"] = float(np.abs(x_ref - x_own).max())
        wav24 = torch.from_numpy(x_own.astype(np.float32))[None]
        torch.manual_seed(0)
        st_ref = ref.get_state_for_audio_prompt(wav24.clone())
        torch.manual_seed(0)
        st_own = own.voice_state(wav24.clone())
        kref = [v["cache"] for k, v in st_ref.items() if "cache" in v]
        n = st_own.off
        r["voice_len"] = [int(list(st_ref.values())[0]["offset"].view(-1)[0]), n]
        r["voice_kv_maxabs"] = max(float((kref[i][:, 0, :n] - st_own.c[i, :, :n]).abs().max()) for i in range(len(kref)))
        torch.manual_seed(1)
        t0 = time.monotonic()
        a_ref = ref.generate_audio(st_ref, TEXT[lang]).float().cpu().numpy()
        t1 = time.monotonic()
        torch.manual_seed(1)
        a_own = torch.cat(list(own.stream(st_own, TEXT[lang]))).float().cpu().numpy()
        t2 = time.monotonic()
        m = min(len(a_ref), len(a_own))
        r.update(len_ref=len(a_ref), len_own=len(a_own), audio_maxabs=float(np.abs(a_ref[:m] - a_own[:m]).max()) if m else None,
                 audio_rms_ref=float(np.sqrt((a_ref ** 2).mean())), s_ref=round(t1 - t0, 2), s_own=round(t2 - t1, 2))
        import soundfile as sf
        od = Path(a.out).parent / (Path(a.out).stem + "_wav")
        od.mkdir(exist_ok=True)
        sf.write(od / f"{lang}_ref.wav", a_ref, 24000)
        sf.write(od / f"{lang}_own.wav", a_own, 24000)
        res[lang] = r
        print(lang, json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
