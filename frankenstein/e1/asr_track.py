"""E1 step 1 (CPU only): build the fixed user track per language and run the multilingual streaming ASR on it.

ASR = nvidia/nemotron-3.5-asr-streaming-0.6b (donor of the multilingual lab), loaded and decoded with the lab's own
code paths, unchanged:
  encoder  labs/multilingual/scripts/multilingual/encoders.py SpeechEncoder("ml", att=[56, 0]) (cache-aware
           FastConformer, offline forward with chunked-limited attention = what the streaming cache reproduces)
  decoder  the greedy RNNT of a0_rnnt_readout.py (joint(prompt_kernel(cat[enc, onehot(lang)]), pred), <= 10 symbols
           per frame), here with the frame index of every emitted piece recorded.
The language prompt is the session locale (it 15, fr 8, en 0). Encoder frame i = 80 ms of track audio = model frame k.

Outputs (in --out, a NEW directory): track_<lang>.wav (stays on the GPU host), asr_<lang>.json with the pieces per
frame, the closed words (causal: a word exists from its close frame), ASR non-blank frames, EOU / BOU frames, the
timeline and the per-turn transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e1.script import LANGS, VOICE, TURNS, LEAD_S, timeline  # noqa: E402
from frankenstein_s2s.usertext import TurnTaker, UserTextScheduler  # noqa: E402

LANG_IDX = {"auto": 101, "en": 0, "it": 15, "fr": 8}
TAG_PREFIX = "<"


def main():  # pragma: no cover - CPU model run on the GPU host
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    import yaml
    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True, help="labs/multilingual/scripts/multilingual")
    ap.add_argument("--audio-root", required=True, help="agentbench ab1 root (manifest.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--close-after", type=int, default=3)
    ap.add_argument("--eou-frames", type=int, default=10)
    ap.add_argument("--bou-frames", type=int, default=3)
    ap.add_argument("--att-right", type=int, default=3, help="right context frames: 0, 1, 3, 6, 13")
    ap.add_argument("--noise-rms", type=float, default=3e-4,
                    help="seeded white-noise floor added to the whole track (about -70 dBFS): with exact digital "
                         "silence the zero-lookahead encoder dropped whole utterances (CPU probe, E1 prep)")
    ap.add_argument("--lang-mode", default="session", choices=["session", "auto"])
    ap.add_argument("--reset", default="tag", choices=["none", "tag", "eou"],
                    help="reset the RNNT predictor state after a language tag (tag) or after a tag or an end of "
                         "utterance (eou): one decoder state over a whole session drops words after the first tag")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, a.lab_scripts)
    from encoders import ASR_DIRS, SpeechEncoder
    from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
    torch.set_num_threads(8)
    t0 = time.monotonic()
    root = Path(a.audio_root)
    man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(root / "manifest.jsonl"))}
    d = ASR_DIRS["ml"]
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    dec = RNNTDecoder.from_config_dict(OmegaConf.create(cfg["decoder"]))
    jnt = RNNTJoint.from_config_dict(OmegaConf.create(cfg["joint"]))
    dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
    jnt.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("joint.")})
    nd = cfg["model_defaults"]["enc_hidden"]
    pk = torch.nn.Sequential(torch.nn.Linear(nd + 128, 2 * nd), torch.nn.ReLU(), torch.nn.Linear(2 * nd, nd))
    pk.load_state_dict({k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
    dec.eval(), jnt.eval(), pk.eval()
    vocab = list(jnt.vocabulary)
    blank = len(vocab)
    enc = SpeechEncoder("ml", att=[56, a.att_right], device="cpu")

    @torch.no_grad()
    def greedy(e: torch.Tensor, lang_idx: int) -> tuple:
        oh = torch.zeros(e.shape[0], 128)
        oh[:, lang_idx] = 1
        f = pk(torch.cat([e, oh], -1))

        def fresh():
            return dec.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)
        g, st = fresh()
        per, resets, blank_run, spoke = {}, [], 0, False
        for t in range(f.shape[0]):
            emitted = False
            for _ in range(10):
                k = int(jnt.joint(f[t:t + 1][None], g)[0, 0, 0].argmax())
                if k == blank:
                    break
                emitted = True
                per.setdefault(t, []).append(vocab[k])
                g, st = dec.predict(torch.tensor([[k]]), st, add_sos=False, batch_size=1)
                if a.reset in ("tag", "eou") and vocab[k].strip().startswith("<") and vocab[k].strip().endswith(">"):
                    g, st = fresh()
                    resets.append([t, "tag"])
                    spoke = False
                    break
            blank_run = 0 if emitted else blank_run + 1
            spoke = spoke or emitted
            if a.reset == "eou" and spoke and blank_run >= a.eou_frames:
                g, st = fresh()
                resets.append([t, "eou"])
                spoke = False
        return per, resets

    def load_16k(p):
        w, sr = sf.read(p, dtype="float32", always_2d=True)
        w = torch.from_numpy(w.T.copy()).mean(0)
        return AF.resample(w[None], sr, 16000)[0] if sr != 16000 else w

    inputs = []
    for lang in a.langs.split(","):
        clips, durs = {}, {}
        for _, iid, _, _ in TURNS:
            m = man[(iid, lang, VOICE[lang])]
            p = root / m["path"]
            clips[iid] = load_16k(p)
            durs[iid] = clips[iid].numel() / 16000
            inputs.append({"lang": lang, "id": iid, "path": m["path"], "sha256": m["sha256"], "text": m["text"],
                           "voice": VOICE[lang], "dur_s": round(durs[iid], 3)})
        tl = timeline(durs)
        parts = [torch.zeros(int(LEAD_S * 16000))]
        for (_, iid, gap, _) in TURNS:
            parts += [clips[iid], torch.zeros(int(gap * 16000))]
        wav = torch.cat(parts)
        if a.noise_rms > 0:
            gen = torch.Generator().manual_seed(0)
            wav = wav + torch.randn(wav.numel(), generator=gen) * a.noise_rms
        sf.write(out / f"track_{lang}.wav", wav.numpy(), 16000)
        e = enc(wav)["out"]
        per0, resets = greedy(e, LANG_IDX["auto" if a.lang_mode == "auto" else lang])
        # causality: a piece decoded at encoder frame t used audio up to frame t + right context
        per = {t + a.att_right: v for t, v in per0.items()}
        words, sched_words = [], []
        sched = UserTextScheduler(lambda w: [1000], close_after=a.close_after,
                                  on_close=lambda k, w, ids: sched_words.append({"k": k, "word": w}))
        tt = TurnTaker(eou_frames=a.eou_frames, bou_frames=a.bou_frames)
        nonblank, eou, bou, raw_nonblank = [], [], [], []
        raw_set = {k for k, v in per.items() if any(not (p.startswith(TAG_PREFIX) and p.endswith(">")) for p in v)}
        F = int(e.shape[0]) + a.att_right
        for k in range(F):
            pieces = per.get(k, [])
            real = [p for p in pieces if not (p.startswith(TAG_PREFIX) and p.endswith(">"))]
            sched.on_asr(k, pieces)
            sched.step(k)
            if real:
                raw_nonblank.append(k)
            # activity = a piece in the last 3 frames (the RNNT emits sparsely: 1 piece every 1-3 frames in speech)
            nb = any(j in raw_set for j in (k, k - 1, k - 2)) if real or raw_nonblank else False
            if nb:
                nonblank.append(k)
            for ev in tt.step(k, nb):
                (eou if ev == "eou" else bou).append(k)
                if ev == "eou":
                    sched.flush(k)
        sched.flush(F)
        words = sched_words
        by_turn = {}
        for turn in tl:
            lo = turn["start_k"]
            nxt = [x["start_k"] for x in tl if x["start_k"] > lo]
            hi = min(nxt) if nxt else F + 1
            by_turn[turn["label"]] = " ".join(w["word"] for w in words if lo <= w["k"] < hi)
        res = {"lang": lang, "lang_idx": LANG_IDX["auto" if a.lang_mode == "auto" else lang],
               "att": [56, a.att_right], "reset": a.reset, "resets": resets, "noise_rms": a.noise_rms,
               "emission_offset_frames": a.att_right, "frames": F, "frame_s": 0.08,
               "close_after": a.close_after, "eou_frames": a.eou_frames, "bou_frames": a.bou_frames,
               "pieces": {str(k): v for k, v in per.items()}, "words": words, "nonblank": nonblank,
               "raw_nonblank": raw_nonblank, "eou": eou,
               "bou": bou, "timeline": tl, "transcript_by_turn": by_turn,
               "track_sha256": hashlib.sha256(np.ascontiguousarray(wav.numpy()).tobytes()).hexdigest()}
        (out / f"asr_{lang}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(lang, F, "frames", round(time.monotonic() - t0, 1), "s |", json.dumps(by_turn, ensure_ascii=False),
              "| eou", eou, flush=True)
    (out / "inputs.json").write_text(json.dumps({"items": inputs, "asr_dir": str(d)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
