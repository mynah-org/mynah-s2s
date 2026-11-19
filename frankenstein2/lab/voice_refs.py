"""Reference voices for cloning, for the languages whose talker has no predefined voice (the community Pocket models).
Each reference is trimmed to <= 10 s, mono 24 kHz, peak-normalised; OUT/sources.json says where each comes from and
under which licence (frankenstein2/lab/voice_audition.py then picks the ones the ASR understands best).

  OUT/<lang>/<name>.wav       the voices the community models' authors publish as voice prompts (same licence)
  OUT/_donated/<name>.wav     --donations: the voices volunteers donated to Kyutai for speech synthesis (CC0,
                              kyutai/tts-voices voice-donations), mostly English speakers: candidates for every language
  OUT/<lang>/fleurs_*.wav     --fleurs: native speakers from Google's FLEURS (CC-BY-4.0). LAB COMPARISON ONLY: those
                              speakers recorded a read-speech corpus, not a voice for synthesis, so a clone of them is
                              never published (samples, defaults)

    python frankenstein2/lab/voice_refs.py --out DIR [--langs ko,tr,ja,...] [--donations] [--fleurs]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from pathlib import Path

# voices published with the community models (frankenstein2/scripts/build_packages.py SOURCES), same licence
COMMUNITY = {
    "ja": ("takuma104/pocket-tts-ja-cc", "CC-BY-4.0", [f"voices/{g}_0{i}.wav" for g in ("female", "male")
                                                     for i in range(1, 6)]),
    "cs": ("vvolhejn/pocket-tts-czech", "CC-BY-4.0", [f"voices/cs_{x}.wav" for x in (
        "f_nytrova", "f_vahalova", "m_eybert", "m_rykala", "m_zaloudik", "m_zenisek")]),
    "et": ("cbentes/pocket-tts-estonian", "CC-BY-SA-4.0", [f"voices/et_{x}.wav" for x in (
        "f_reporter", "f_rohemae", "f_sobak", "m_kelmsaar", "m_kilumets", "m_maiberg")]),
    "hi": ("saryps-labs/pocket-tts-hindi", "CC-BY-4.0", [f"samples/voice_0{x}.wav" for x in (
        "1_f", "2_f", "3_f", "6_m")]),
    "pl": ("shefowl/pocket-tts-polish-6l", "CC-BY-4.0", ["reference.wav"]),
}
# FLEURS where the community model ships no voice, or only one
FLEURS = {"ko": "ko_kr", "tr": "tr_tr", "ru": "ru_ru", "pl": "pl_pl", "hi": "hi_in"}


def prep(x, sr, max_s=10.0):
    """Mono 24 kHz, leading/trailing silence cut, at most max_s, peak 0.9."""
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 24000:
        g = gcd(sr, 24000)
        x = resample_poly(x, 24000 // g, sr // g).astype(np.float32)
    a = np.abs(x)
    v = np.where(a > 0.02 * (a.max() or 1))[0]
    if len(v):
        x = x[max(0, v[0] - 2400): v[-1] + 4800]
    x = x[: int(max_s * 24000)]
    return (x / (np.abs(x).max() or 1) * 0.9).astype(np.float32)


def main():  # pragma: no cover - downloads
    import io
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=",".join(sorted(set(COMMUNITY) | set(FLEURS))))
    ap.add_argument("--fleurs", action="store_true", help="also FLEURS clips (lab comparison only, see above)")
    ap.add_argument("--fleurs-per-gender", type=int, default=3)
    ap.add_argument("--donations", action="store_true", help="Kyutai's donated voices (CC0) into OUT/_donated")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src_path = out / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.exists() else {}
    for lang in [x for x in a.langs.split(",") if x]:
        d = out / lang
        d.mkdir(exist_ok=True)
        if lang in COMMUNITY:
            repo, lic, files = COMMUNITY[lang]
            for f in files:
                try:
                    p = hf_hub_download(repo, f)
                except Exception as e:  # noqa: BLE001
                    print("REF_MISSING", lang, repo, f, type(e).__name__, flush=True)
                    continue
                x, sr = sf.read(p, dtype="float32")
                name = Path(f).stem
                sf.write(d / f"{name}.wav", prep(x, sr), 24000, subtype="PCM_16")
                sources[f"{lang}/{name}"] = {"source": f"https://huggingface.co/{repo}/blob/main/{f}",
                                             "licence": lic}
                print("REF", lang, name, repo, flush=True)
        if a.fleurs and lang in FLEURS:
            code = FLEURS[lang]
            tsv = hf_hub_download("google/fleurs", f"data/{code}/dev.tsv", repo_type="dataset")
            rows = list(csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
            # id, file, raw text, normalised text, phonemes, samples (16 kHz), gender
            ok = [r for r in rows if len(r) >= 7 and 6 * 16000 <= int(r[5]) <= 14 * 16000]
            rnd = random.Random(0)
            pick, seen = [], set()
            for g in ("FEMALE", "MALE"):
                cand = [r for r in ok if r[6].upper() == g]
                rnd.shuffle(cand)
                for r in cand:                         # one clip per speaker id (FLEURS ids are per sentence;
                    if r[0] in seen:                   # different sentences are mostly different speakers)
                        continue
                    seen.add(r[0])
                    pick.append((g[0].lower(), r))
                    if sum(1 for x, _ in pick if x == g[0].lower()) >= a.fleurs_per_gender:
                        break
            want = {r[1]: (g, r) for g, r in pick}
            tar = hf_hub_download("google/fleurs", f"data/{code}/audio/dev.tar.gz", repo_type="dataset")
            n = 0
            with tarfile.open(tar) as t:
                for m in t:
                    base = Path(m.name).name
                    if base not in want:
                        continue
                    g, r = want[base]
                    x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
                    name = f"fleurs_{g}_{Path(base).stem}"
                    sf.write(d / f"{name}.wav", prep(x, sr), 24000, subtype="PCM_16")
                    sources[f"{lang}/{name}"] = {"source": f"https://huggingface.co/datasets/google/fleurs "
                                                           f"(data/{code}/audio/dev.tar.gz: {base})",
                                                 "licence": "CC-BY-4.0", "text": r[2]}
                    n += 1
                    print("REF", lang, name, "fleurs", flush=True)
            if n < len(want):
                print("REF_SHORT", lang, n, "of", len(want), flush=True)
    if a.donations:
        from huggingface_hub import list_repo_files
        d = out / "_donated"
        d.mkdir(exist_ok=True)
        files = [f for f in list_repo_files("kyutai/tts-voices") if f.startswith("voice-donations/")
                 and f.endswith(".wav") and not f.endswith("_enhanced.wav")]
        for f in files:
            name = Path(f).stem
            if (d / f"{name}.wav").exists():
                continue
            try:
                x, sr = sf.read(hf_hub_download("kyutai/tts-voices", f), dtype="float32")
            except Exception as e:  # noqa: BLE001
                print("REF_MISSING", f, type(e).__name__, flush=True)
                continue
            sf.write(d / f"{name}.wav", prep(x, sr), 24000, subtype="PCM_16")
            sources[f"_donated/{name}"] = {"source": f"https://huggingface.co/kyutai/tts-voices/blob/main/{f}",
                                           "licence": "CC0-1.0"}
        print("REF donated", len(files), flush=True)
    src_path.write_text(json.dumps(sources, indent=1, ensure_ascii=False))
    print("REFS_DONE", len(sources), flush=True)


if __name__ == "__main__":
    main()
