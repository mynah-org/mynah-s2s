"""D1 pilot, step 2 (GPU, omni venv): word timings by CTC forced alignment of the KNOWN reference text on OmniASR
CTC 3B v2 emissions (same model card and preprocessing as asr_qa.py). Only clips whose strict ASR verification passed
are in the sample, so the reference text is what is spoken.

Target sequence = the clip's words (d1_sample `words`), each encoded on its own with the OmniASR tokenizer (special
tokens dropped) and concatenated; `torchaudio.functional.forced_align` gives one emission frame per target token;
a word spans [first frame of its first token, last frame of its last token + 1]. The CTC blank is DETECTED as the most
frequent argmax over the frames seen so far (CTC emits blank on most frames; checked against the vocab, logged).
Writes {key: {"frame_s", "words": [[start_s, end_s], ...], "score"}}.
Usage: d1_align.py --sample d1_sample.json --data <tts root> --out d1_align.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

CARD = "omniASR_CTC_3B_v2"


def main():  # pragma: no cover - GPU
    import torch
    import torchaudio.functional as AF
    from fairseq2.nn.batch_layout import BatchLayout
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu (pilot rerun: GPU budget kept for the extraction)")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.monotonic()
    clips = json.load(open(a.sample))["clips"]
    pipe = ASRInferencePipeline(model_card=CARD, device=a.device,
                                dtype=torch.bfloat16 if a.device == "cuda" else torch.float32)
    vi = pipe.tokenizer.vocab_info
    special = {x for x in (getattr(vi, "bos_idx", None), getattr(vi, "eos_idx", None), getattr(vi, "pad_idx", None))
               if x is not None}
    print("vocab", vi, "load", round(time.monotonic() - t0, 1), flush=True)
    out, argmax_count = {}, Counter()
    for b in range(0, len(clips), a.batch):
        chunk = clips[b:b + a.batch]
        paths = [str(Path(a.data) / c["path"]) for c in chunk]
        wavs = list(pipe._build_audio_wavform_pipeline(paths).and_return())
        batch = pipe._create_batch_simple([(w, None) for w in wavs])
        lay = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens, device=batch.source_seqs.device)
        with torch.no_grad():
            logits, bl = pipe.model(batch.source_seqs, lay)
        lp = torch.log_softmax(logits.float(), -1)
        for k in range(len(chunk)):
            argmax_count.update(lp[k, :int(bl.seq_lens[k])].argmax(-1).tolist())
        bl_id = argmax_count.most_common(1)[0][0]
        for k, c in enumerate(chunk):
            n = int(bl.seq_lens[k])
            em = lp[k, :n]
            ids, owner = [], []
            for wi, w in enumerate(c["words"]):
                t = [int(x) for x in pipe.token_encoder(w).tolist() if int(x) not in special]
                if bl_id in t:
                    raise SystemExit(f"detected blank {bl_id} is a character of {w!r}")
                ids += t
                owner += [wi] * len(t)
            if not ids or len(ids) > n:
                out[f"{c['id']}|{c['lang']}"] = {"error": f"targets {len(ids)} frames {n}"}
                continue
            ali, sc = AF.forced_align(em.unsqueeze(0).cpu(), torch.tensor([ids], dtype=torch.int32), blank=bl_id)
            ali = ali[0].tolist()
            # token index per emitted frame: walk the alignment, a new token starts when the label changes from blank
            # or repeats after a blank
            tok_frames, ti, prev = [[] for _ in ids], -1, bl_id
            for f, lab in enumerate(ali):
                if lab != bl_id and (lab != prev or prev == bl_id):
                    ti += 1
                if lab != bl_id and 0 <= ti < len(ids):
                    tok_frames[ti].append(f)
                prev = lab
            fs = c["dur"] / n                       # seconds per emission frame (clip duration / frames)
            wspan = []
            for wi in range(len(c["words"])):
                fr = [f for t, o in enumerate(owner) if o == wi for f in tok_frames[t]]
                wspan.append([round(min(fr) * fs, 3), round((max(fr) + 1) * fs, 3)] if fr else None)
            out[f"{c['id']}|{c['lang']}"] = {"frame_s": round(fs, 5), "words": wspan,
                                             "score": round(float(sc[0].mean()), 3), "blank": bl_id}
        print(f"aligned {b + len(chunk)}/{len(clips)} {time.monotonic() - t0:.0f}s", flush=True)
        Path(a.out + ".partial").write_text(json.dumps({"card": CARD, "alignments": out}, ensure_ascii=False))
    Path(a.out).write_text(json.dumps({"card": CARD, "blank": argmax_count.most_common(1)[0][0], "argmax_top": argmax_count.most_common(3),
                                       "alignments": out}, ensure_ascii=False))
    print("done", len(out), round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
