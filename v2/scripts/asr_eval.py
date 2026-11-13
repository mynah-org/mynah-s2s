#!/usr/bin/env python3
"""Evaluation ASR of the audio the agent actually spoke (OmniASR CTC 3B v2, official pipeline, CUDA).

This is evidence (2) of three that are kept separate: (1) the model's internal text and tool calls (summary.json),
(2) the ASR of the spoken audio (this script), (3) the tool ground truth (SQLite rows / corpus expectations).
scripts/score_corpus.py and scripts/tts_quality.py read its output; they never merge the three.

Backend: the open-source Omnilingual ASR model omniASR_CTC_3B_v2 (Meta, Apache-2.0 weights) through the official
`omnilingual_asr` ASRInferencePipeline (bf16 on CUDA, greedy CTC). Install it in its OWN environment (it pins its own
torch / fairseq2), e.g.
    uv venv $MYNAH_HOME/.venv-asr && $MYNAH_HOME/.venv-asr/bin/python -m pip install omnilingual-asr
    $MYNAH_HOME/.venv-asr/bin/python v2/scripts/asr_eval.py <run_dir> [<run_dir> ...]
The first run downloads the checkpoint into the fairseq2 cache (FAIRSEQ2_CACHE_DIR). If the backend cannot load, the
script exits non-zero: there is no silent fallback to another ASR.

Input layout: <run_dir>/<item>/output.wav (mono agent audio, as written by run_stream.py / rendezvous_pair.py), or
<run_dir>/<item>/session.wav (stereo, agent on the right channel). Each file is cut into speech segments (energy,
20 ms frames, silence >= 0.6 s, segments <= 25 s) and each segment is transcribed. Outputs per run dir:
asr_<tag>.txt ("== <item>" headers, "SEG start-end text" lines) and asr_<tag>.json (segments + model card, dtype,
library versions, script sha256). The WAVs are never modified.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch, width, sr, raw = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype={2: "<i2", 4: "<i4"}[width]).astype(np.float32) / float(1 << (width * 8 - 1))
    return (x.reshape(-1, ch) if ch > 1 else x[:, None]), sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def speech_segments(x: np.ndarray, sr: int, min_sil: float = 0.6, max_len: float = 25.0, thr_db: float = -45.0):
    hop = int(0.02 * sr)
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    segs, start, sil = [], None, 0
    for i, v in enumerate(db > thr_db):
        if v:
            start = i if start is None else start
            sil = 0
        elif start is not None:
            sil += 1
            if sil * 0.02 >= min_sil or (i - start) * 0.02 >= max_len:
                segs.append((start, i - sil + 1))
                start, sil = None, 0
    if start is not None:
        segs.append((start, n))
    return [(a * hop, b * hop) for a, b in segs if (b - a) * 0.02 >= 0.3]


def load_omni(card: str, dtype: str):
    try:
        import fairseq2
        import omnilingual_asr
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except Exception as e:  # noqa: BLE001
        sys.exit(f"OMNIASR_UNAVAILABLE: {type(e).__name__}: {e} (install omnilingual-asr in its own venv; no fallback)")
    if not torch.cuda.is_available():
        sys.exit("OMNIASR_UNAVAILABLE: CUDA not available in this environment (no CPU fallback)")
    pipe = ASRInferencePipeline(model_card=card, device="cuda", dtype=getattr(torch, dtype))
    meta = {"backend": "omnilingual_asr", "model_card": card, "dtype": dtype, "device": "cuda",
            "decoding": "greedy CTC (official ASRInferencePipeline)",
            "fairseq2_cache_dir": os.environ.get("FAIRSEQ2_CACHE_DIR", ""),
            "omnilingual_asr": getattr(omnilingual_asr, "__version__", "unknown"), "fairseq2": fairseq2.__version__,
            "torch": torch.__version__}

    def transcribe(paths: list[Path]) -> list[str]:
        return [str(t) for t in pipe.transcribe([str(p) for p in paths], batch_size=len(paths))]
    return transcribe, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run dirs containing <item>/output.wav")
    ap.add_argument("--card", default="omniASR_CTC_3B_v2")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-tag", default="omni3b", help="output file tag: asr_<tag>.txt / .json")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    transcribe, meta = load_omni(a.card, a.dtype)
    meta["script"] = {"name": Path(__file__).name, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    print("ASR", json.dumps(meta), flush=True)
    for run in a.runs:
        lines, rows, t0 = [], [], time.time()
        wavs = sorted(glob.glob(f"{run}/*/session.wav")) + sorted(glob.glob(f"{run}/*/output.wav"))
        if not wavs:
            sys.exit(f"no <item>/output.wav or <item>/session.wav under {run}")
        pending = []
        with tempfile.TemporaryDirectory() as td:
            for f in map(Path, wavs):
                x, sr = read_wav(f)
                ch = x[:, 1] if f.name == "session.wav" and x.shape[1] > 1 else x[:, 0]
                segs = speech_segments(ch, sr)
                seg_paths = []
                for k, (s0, s1) in enumerate(segs):
                    p = Path(td) / f"{f.parent.name}_{k:03d}.wav"
                    write_wav(p, ch[s0:s1], sr)
                    seg_paths.append(p)
                pending.append((f, sr, segs, seg_paths))
            allp = [p for *_, sp in pending for p in sp]
            texts = []
            for i in range(0, len(allp), max(1, a.batch)):
                texts += transcribe(allp[i:i + a.batch])
            k = 0
            for f, sr, segs, sp in pending:
                tx = texts[k:k + len(sp)]
                k += len(sp)
                lines.append(f"== {f.parent.parent.name}/{f.parent.name}  ({f.name}, agent channel, {sr} Hz)")
                lines += [f"SEG {s0 / sr:6.2f}s-{s1 / sr:6.2f}s  {t}" for (s0, s1), t in zip(segs, tx)]
                rows.append({"scenario": f.parent.name, "wav": str(f), "sample_rate": sr,
                             "segments": [{"start_s": round(s0 / sr, 2), "end_s": round(s1 / sr, 2), "text": t}
                                          for (s0, s1), t in zip(segs, tx)]})
        Path(run, f"asr_{a.out_tag}.txt").write_text("\n".join(lines) + "\n")
        Path(run, f"asr_{a.out_tag}.json").write_text(json.dumps({"meta": meta, "seconds": round(time.time() - t0, 1),
                                                                 "sessions": rows}, indent=1, ensure_ascii=False))
        print("\n".join(lines), flush=True)
        print(f"ASR_DONE {run} items={len(rows)} tag={a.out_tag}", flush=True)


if __name__ == "__main__":
    main()
