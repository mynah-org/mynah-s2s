"""Pronunciation / intelligibility check of rendered QA (or training) clips: OmniASR CTC 3B v2 (OSS, official
`omnilingual_asr` pipeline, GPU) transcribes every clip of one or more manifest.jsonl files; CER and WER against the
rendered text, per (manifest, voice, lang). Checkpoint resolved offline from FAIRSEQ2_CACHE_DIR (the official card URL,
file pre-placed under its sha1-of-URL directory; sha256 recorded in ckpt/omniasr/*.sha256).

Normalisation for scoring (both sides): NFC, lower-case, punctuation removed, whitespace collapsed; accents KEPT
(they are part of pronunciation). Output <out>/rows.jsonl, <out>/summary.json, <out>/asr_<tag>.json (provenance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

CARD = "omniASR_CTC_3B_v2"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s']|_", " ", s).replace("'", " ")
    return " ".join(s.split())


def ed(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--voices", default="", help="comma list filter (empty = all)")
    ap.add_argument("--splits", default="", help="comma list filter (empty = all)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-sec", type=float, default=39.0, help="OmniASR CTC caps input at 40 s; longer clips (e.g. a "
                                                                   "runaway TTS generation) are recorded as skipped")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    out.mkdir(parents=True)
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    clips = []
    for m in a.manifests:
        root = Path(m).parent
        for l in open(m):
            r = json.loads(l)
            if a.voices and r["voice"] not in a.voices.split(","):
                continue
            if a.splits and r["split"] not in a.splits.split(","):
                continue
            clips.append({**r, "manifest": m, "abs": str(root / r["path"])})
    skipped = [c for c in clips if c.get("dur", 0) > a.max_sec]
    clips = [c for c in clips if c.get("dur", 0) <= a.max_sec]
    with open(out / "skipped.jsonl", "w") as f:
        for c in skipped:
            f.write(json.dumps({"id": c["id"], "lang": c["lang"], "voice": c["voice"], "dur": c.get("dur"),
                                "reason": "too_long_for_asr"}, ensure_ascii=False) + "\n")
    print(f"{len(clips)} clips ({len(skipped)} skipped > {a.max_sec}s)", flush=True)
    t0 = time.monotonic()
    pipe = ASRInferencePipeline(model_card=CARD, device="cuda", dtype=torch.bfloat16)
    print(f"model loaded {time.monotonic() - t0:.0f}s", flush=True)
    t1 = time.monotonic()
    hyps = pipe.transcribe([c["abs"] for c in clips], batch_size=a.batch)
    dt = time.monotonic() - t1
    agg = {}
    with open(out / "rows.jsonl", "w") as f:
        for c, h in zip(clips, hyps):
            ref, hyp = norm(c["text"]), norm(h)
            ce, cn = ed(ref.replace(" ", ""), hyp.replace(" ", "")), max(1, len(ref.replace(" ", "")))
            we, wn = ed(ref.split(), hyp.split()), max(1, len(ref.split()))
            row = {"manifest": c["manifest"], "split": c["split"], "voice": c["voice"], "lang": c["lang"], "id": c["id"],
                   "ref": c["text"], "hyp": h, "cer": ce / cn, "wer": we / wn}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            k = f"{Path(c['manifest']).parent.name}|{c['voice']}|{c['lang']}"
            g = agg.setdefault(k, [0, 0, 0, 0, 0])
            g[0] += ce; g[1] += cn; g[2] += we; g[3] += wn; g[4] += 1
    summ = {k: {"cer": round(g[0] / g[1], 4), "wer": round(g[2] / g[3], 4), "n": g[4]} for k, g in sorted(agg.items())}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / f"asr_{a.tag}.json").write_text(json.dumps({
        "model_card": CARD, "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B-v2.pt (offline cache)",
        "dtype": "bfloat16", "decoding": "CTC greedy (pipeline default)", "batch": a.batch,
        "omnilingual_asr": version("omnilingual-asr"), "fairseq2": version("fairseq2"), "torch": torch.__version__,
        "script_sha256": sha, "manifests": a.manifests, "n_clips": len(clips), "transcribe_s": round(dt, 1),
        "normalisation": "NFC lower, punctuation removed, accents kept"}, indent=1))
    for k, v in summ.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
