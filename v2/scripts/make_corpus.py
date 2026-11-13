"""Render the fixed V2 regression corpus (eval/corpus_v1.json) to user-side WAVs (24 kHz mono PCM16) + JSON sidecars.

Synthetic caller only (Qwen3-TTS 0.6B CustomVoice, pinned revision, preset speakers, fixed seed per utterance), same
timeline convention as scripts/agent/make_scenarios.py: the first offset is absolute, later offsets are measured from
the end of the previous utterance; tail_s of silence after the last one. Writes <out>/<id>.wav, <out>/<id>.json,
<out>/orders.json, <out>/manifest.json (sha256 of every WAV). Refuses an existing output directory.

Runs with the V2 interpreter and PYTHONPATH=~/mynah-s2s/cache/pylib-tts (qwen-tts 0.1.1 + transformers 4.57.3,
isolated from the engine's packages).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).parents[1] / "eval/corpus_v1.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.corpus).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    snap = snapshot_download(spec["tts"]["model"], revision=spec["tts"]["revision"], local_files_only=True)
    m = Qwen3TTSModel.from_pretrained(snap, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    manifest = {"corpus": spec["name"], "corpus_sha256": hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest(),
                "tts": spec["tts"], "items": {}}
    for it in spec["items"]:
        segs, meta, prev_end = [], [], None
        for u in it["utterances"]:
            seed = int(hashlib.sha256(f'{spec["tts"]["seed"]}{it["speaker"]}{u["text"]}'.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            wavs, sr = m.generate_custom_voice(text=u["text"], speaker=it["speaker"], language="English")
            x = np.asarray(wavs[0], np.float32)
            assert sr == SR, sr
            start = u["offset"] if prev_end is None else round(prev_end + u["offset"], 3)
            prev_end = start + len(x) / SR
            segs.append((start, x))
            meta.append({"start": start, "end": round(prev_end, 3), "text": u["text"], "seed": seed})
        total = max(s + len(x) / SR for s, x in segs) + it["tail_s"]
        y = np.zeros(int(total * SR), np.float32)
        for s, x in segs:
            i = int(s * SR)
            y[i:i + len(x)] += x
        y = np.clip(y, -1, 1)
        p = out / f'{it["id"]}.wav'
        sf.write(p, y, SR, subtype="PCM_16")
        (out / f'{it["id"]}.json').write_text(json.dumps(
            {**{k: it[k] for k in ("id", "category", "speaker", "expect")}, "duration": round(total, 2),
             "utterances": meta, "synthetic": True}, indent=1))
        manifest["items"][it["id"]] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "duration": round(total, 2)}
        print(it["id"], round(total, 1), "s", [round(mm["end"] - mm["start"], 2) for mm in meta], flush=True)
    (out / "orders.json").write_text(json.dumps(spec["orders"], indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
