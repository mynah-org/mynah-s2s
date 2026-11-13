"""Audio-side quality gate for experiments that change the TTS numerics (e.g. EarTTS BF16), CPU only.

Inputs: a perf_ab run directory (baseline.wav / experiment.wav + *_text.json) and an OmniASR run over the two WAVs
(asr_eval.py layout: <asr_dir>/{fp32,bf16}/output.wav + asr_omni3b.json). For each side:
  * hallucination: word error rate of the ASR against the model's internal text, and the share of ASR words that
    never occur in the internal text;
  * repetition: share of repeated word 3-grams in the ASR;
  * EOS / stray audio: seconds of voiced audio outside every agent turn window [BOS, EOS + 1.5 s lag];
  * speaker proxies: median F0 and F0 IQR over voiced frames (librosa pyin), mean spectral centroid — proxies only,
    not a speaker-verification score;
  * quality proxies: clipping ratio, peak, RMS, mean spectral flatness.
Prints both sides and the deltas; PASS/FAIL thresholds are applied by the reader, documented in .work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

BOS, EOS, PAD = 1, 2, 12
SR = 22050


def words(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    return re.findall(r"[a-z0-9']+", s)


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def turns(ids: list[int], frame_s: float, lag_s: float = 1.5):
    out, start = [], None
    for k, t in enumerate(ids):
        if t == BOS and start is None:
            start = k
        elif t == EOS and start is not None:
            out.append((start * frame_s, k * frame_s + lag_s))
            start = None
    if start is not None:
        out.append((start * frame_s, len(ids) * frame_s + lag_s))
    return out


def side(wav: Path, text: dict, asr_text: str) -> dict:
    import librosa
    x, sr = sf.read(wav, dtype="float32")
    assert sr == SR
    hop = int(0.08 * SR)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop + 1, hop)])
    voiced = rms > 0.003
    win = turns(text["text_ids"], text["frame_s"])
    stray = sum(1 for k, v in enumerate(voiced) if v and not any(a <= k * 0.08 < b for a, b in win)) * 0.08
    ref, hyp = words(text["text"]), words(asr_text)
    tri = [tuple(hyp[i:i + 3]) for i in range(max(0, len(hyp) - 2))]
    rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
    vx = np.concatenate([x[k * hop:(k + 1) * hop] for k in range(len(voiced)) if voiced[k]]) if voiced.any() else x[:SR]
    f0, vflag, _ = librosa.pyin(vx, fmin=60, fmax=400, sr=SR, frame_length=2048)
    f0 = f0[~np.isnan(f0)]
    return {"asr": asr_text, "wer_vs_internal_text": round(wer(ref, hyp), 3),
            "asr_words_not_in_text": round(sum(w not in set(ref) for w in hyp) / max(1, len(hyp)), 3),
            "repeated_trigram_share": round(rep, 3), "stray_voiced_s": round(stray, 2),
            "voiced_s": round(float(voiced.sum()) * 0.08, 2),
            "f0_median_hz": round(float(np.median(f0)), 1) if f0.size else None,
            "f0_iqr_hz": round(float(np.subtract(*np.percentile(f0, [75, 25]))), 1) if f0.size else None,
            "spectral_centroid_hz": round(float(librosa.feature.spectral_centroid(y=vx, sr=SR).mean()), 1),
            "spectral_flatness": round(float(librosa.feature.spectral_flatness(y=vx).mean()), 4),
            "clip_ratio": float(np.mean(np.abs(x) > 0.99)), "peak": round(float(np.abs(x).max()), 3),
            "rms_voiced": round(float(np.sqrt(np.mean(vx ** 2))), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True, help="perf_ab run dir")
    ap.add_argument("--asr", required=True, help="dir with {base,exp}/output.wav and asr_omni3b.json")
    ap.add_argument("--names", default="fp32,bf16", help="sub-dir names for baseline,experiment in --asr")
    a = ap.parse_args()
    ab, asr = Path(a.ab), Path(a.asr)
    rec = json.loads((asr / "asr_omni3b.json").read_text())

    def asr_of(name):                       # asr_eval.py layout: {"sessions": [{"scenario", "segments": [...]}]}
        for it in rec.get("sessions", []):
            if it.get("scenario") == name:
                return " ".join(seg.get("text", "") for seg in it.get("segments", []))
        return ""
    b_name, e_name = a.names.split(",")
    res = {}
    for tag, nm in (("baseline", b_name), ("experiment", e_name)):
        res[tag] = side(ab / f"{tag}.wav", json.loads((ab / f"{tag}_text.json").read_text()), asr_of(nm))
    res["delta"] = {k: (round(res["experiment"][k] - res["baseline"][k], 4)
                        if isinstance(res["baseline"][k], (int, float)) and isinstance(res["experiment"][k], (int, float))
                        else None) for k in res["baseline"] if k != "asr"}
    (asr / "tts_quality.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
