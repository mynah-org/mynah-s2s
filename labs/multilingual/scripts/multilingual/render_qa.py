"""Render the QA corpus (qa_corpus.json) to 16 kHz mono WAVs with Qwen3-TTS 0.6B CustomVoice (GPU, data prep).

Default: two preset user voices (aiden, serena), native language id per clip. Owner note (2026-11-11): among the
CustomVoice presets only Ryan pronounces IT/FR/ES/DE acceptably; the others sound Chinese-accented in non-English.
Use `--voices ryan` for non-English evaluation audio (pronunciation quality is measured with OmniASR in asr_qa.py). Leading/trailing digital silence trimmed to 50 ms; the evaluation harness appends its own trailing silence.
Output: <out>/<lang>/<voice>/<id>.wav + <out>/manifest.jsonl (text, duration, sha256, TTS revision, seed).
Limitation (recorded in the manifest): synthetic, non-native preset voices -> accent; FLEURS real speech is the
complementary check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO, REV = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "85e237c12c027371202489a0ec509ded67b5e4b5"
LANG = {"en": "English", "it": "Italian", "fr": "French", "de": "German", "es": "Spanish"}


def resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(x)[None], sr, to)[0].numpy()


def trim(x: np.ndarray, sr: int, thr_db: float = -50.0, keep_s: float = 0.05) -> np.ndarray:
    hop = int(0.01 * sr)
    n = len(x) // hop
    if n == 0:
        return x
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    act = np.where(e > thr_db)[0]
    if not len(act):
        return x
    k = int(keep_s * sr)
    return x[max(0, act[0] * hop - k): min(len(x), (act[-1] + 1) * hop + k)]


def render_plan(m, a):
    import time
    out = Path(a.out)
    if (out / "manifest.jsonl").exists():
        raise SystemExit(f"{out}/manifest.jsonl exists")
    out.mkdir(parents=True, exist_ok=True)  # job 163 failed: the partial manifest was opened before the dir existed
    plan = [json.loads(l) for l in open(a.plan)]
    plan.sort(key=lambda r: (r["lang"], len(r["text"])))  # similar lengths per batch -> less padding waste
    rows, t0 = [], time.monotonic()
    part = out / "manifest.partial.jsonl"
    done = set()
    if part.exists():  # resume after an interruption (same plan, same seeds)
        for l in open(part):
            r = json.loads(l)
            rows.append(r)
            done.add((r["id"], r["lang"], r["voice"]))
    todo = [r for r in plan if (r["id"], r["lang"], r["voice"]) not in done]
    with open(part, "a") as fp:
        for i in range(0, len(todo), a.batch):
            b = todo[i:i + a.batch]
            seed = int(hashlib.sha256(("|".join(f"{r['id']}{r['lang']}{r['voice']}" for r in b)
                                       + (f"|salt{a.seed_salt}" if a.seed_salt else "")).encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            wavs, sr = m.generate_custom_voice(text=[r["text"] for r in b], speaker=[r["voice"] for r in b],
                                               language=[LANG[r["lang"]] for r in b])
            for r, w in zip(b, wavs):
                x = trim(np.asarray(w, dtype=np.float32).reshape(-1), sr)
                x = resample(x, sr, 16000)
                x = x / max(1e-6, float(np.abs(x).max())) * 0.7
                p = out / r["lang"] / r["voice"] / f"{r['id']}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, x, 16000, subtype="PCM_16")
                row = {**r, "path": str(p.relative_to(out)), "dur": round(len(x) / 16000, 3),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "tts": f"{REPO}@{REV}",
                       "seed": seed, "batch": len(b), "synthetic": True,
                       "chars_per_s": round(len(r["text"]) / max(0.05, len(x) / 16000), 1)}
                rows.append(row)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            print(f"{len(rows)}/{len(plan)} {time.monotonic() - t0:.0f}s | {b[0]['lang']} {b[0]['text'][:60]}",
                  flush=True)
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "clips", f"{time.monotonic() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="en,it,fr,de,es")
    ap.add_argument("--voices", default="aiden,serena")
    ap.add_argument("--plan", default=None, help="corpus V2 plan jsonl {id, split, cat, lang, voice, text}: one clip "
                    "per row, batched (--batch); per-batch seed; manifest rows keep every plan field")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed-salt", type=int, default=0, help="re-render with a different TTS seed (corpus v2 retries)")
    ap.add_argument("--train-jsonl", default=None,
                    help="training items {id, en, it, ...}: one clip per item and language, voice rotated per item+lang")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    snap = snapshot_download(REPO, revision=REV, local_files_only=True)
    m = Qwen3TTSModel.from_pretrained(snap, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    sup = None
    for fn in ("get_supported_speakers", "get_supported_spks"):
        if hasattr(m, fn):
            sup = [s.lower() for s in getattr(m, fn)()]
    print("supported speakers:", sup, flush=True)
    voices = a.voices.split(",")
    if sup:
        bad = [v for v in voices if v not in sup]
        if bad:
            raise SystemExit(f"voices {bad} not in {sup}")
    if a.plan:
        return render_plan(m, a)
    if a.train_jsonl:
        items = [json.loads(l) for l in open(a.train_jsonl)]
        for it in items:
            it.update({"split": "train", "cat": it.get("topic", ""), "text": {l: it[l] for l in a.langs.split(",")}})
    else:
        items = json.load(open(a.corpus))["items"]
    out = Path(a.out)
    rows = []
    langs = a.langs.split(",")
    for li, lang in enumerate(langs):
        for vi, v in enumerate(voices):
            (out / lang / v).mkdir(parents=True, exist_ok=True)
            for ii, it in enumerate(items):
                if a.train_jsonl and (ii + li) % len(voices) != vi:
                    continue
                text = it["text"][lang]
                seed = int(hashlib.sha256(f"{v}|{lang}|{text}".encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                wavs, sr = m.generate_custom_voice(text=text, speaker=v, language=LANG[lang])
                x = trim(np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr)
                x = resample(x, sr, 16000)
                x = x / max(1e-6, float(np.abs(x).max())) * 0.7
                p = out / lang / v / f"{it['id']}.wav"
                sf.write(p, x, 16000, subtype="PCM_16")
                rows.append({"id": it["id"], "split": it["split"], "cat": it["cat"], "lang": lang, "voice": v,
                             "text": text, "path": str(p.relative_to(out)), "dur": round(len(x) / 16000, 3),
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "tts": f"{REPO}@{REV}",
                             "seed": seed, "synthetic": True})
                print(lang, v, it["id"], rows[-1]["dur"], text, flush=True)
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "clips")


if __name__ == "__main__":
    main()
