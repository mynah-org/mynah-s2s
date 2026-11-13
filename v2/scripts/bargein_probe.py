"""Targeted turn-taking probe (paper §4 endpointing fallback, App. B RNN-T barge-in; FDB 1.0 reports 448 ms smooth
turn-taking and 480 ms user-interruption latency on H100/vLLM). REFERENCE driver, deterministic decoding, cache ON.

Run 1 (baseline): base WAV + trailing silence.
  * endpointing: for every user utterance (energy VAD on the input, 80 ms frames), latency from its end to the next
    agent BOS (text channel) and to the first agent audio above -50 dBFS.
Run 2 (interruption): the same input with a user utterance mixed in ``--offset`` seconds after the start of the
agent's last answer in run 1 (outside any tool call).
  * barge-in: first agent EOS after the interruption onset (text channel), time until the agent audio falls silent,
    whether the agent starts a new turn after the interruption ends, and what it says.
Times are on the input timeline (frame k = k × 80 ms), not wall clock: they measure the model + turn-taking logic, not
GB10 speed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mynah_v2.model import tokens as T
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402

FRAME_S = 0.08
OUT_SR = 22050


def drive(eng, wav, prompt):
    eng.open(prompt)
    chunks = list(iter_chunks(wav, eng.chunk_samples))
    audio = []
    for i, c in enumerate(chunks):
        o = eng.step(c, is_last=(i == len(chunks) - 1))
        audio.append(o.audio.float().cpu().reshape(-1))
    gt, ga, n, gf = eng.token_tensors()
    eng.close()
    return gt[0, :n].tolist(), torch.cat(audio).numpy()


def speech_segments(x: np.ndarray, sr: int, thr=0.003, min_gap_s=0.4):
    hop = int(FRAME_S * sr)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop + 1, hop)])
    on = rms > thr
    segs, start, gap = [], None, 0
    for k, v in enumerate(on):
        if v:
            if start is None:
                start = k
            gap = 0
        elif start is not None:
            gap += 1
            if gap * FRAME_S >= min_gap_s:
                segs.append((start, k - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, len(on)))
    return [(a * FRAME_S, b * FRAME_S) for a, b in segs], rms


def first_after(ids, frame, tok):
    return next((k for k in range(max(0, frame), len(ids)) if ids[k] == tok), None)


def agent_audio_onset(audio, t0_s, thr=0.003):
    hop = int(FRAME_S * OUT_SR)
    for k in range(int(t0_s / FRAME_S), len(audio) // hop):
        if np.sqrt(np.mean(audio[k * hop:(k + 1) * hop] ** 2)) > thr:
            return k * FRAME_S
    return None


def agent_audio_silent_from(audio, t0_s, thr=0.003, hold_s=0.48):
    hop = int(FRAME_S * OUT_SR)
    need, run = int(hold_s / FRAME_S), 0
    for k in range(int(t0_s / FRAME_S), len(audio) // hop):
        if np.sqrt(np.mean(audio[k * hop:(k + 1) * hop] ** 2)) <= thr:
            run += 1
            if run >= need:
                return (k - need + 1) * FRAME_S
        else:
            run = 0
    return None


def text_between(tok, ids, a, b):
    return tok.ids_to_text([i for i in ids[a:b] if i not in (T.PAD_ID, T.BOS_ID, T.EOS_ID)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--interrupt", action="append", required=True,
                    help="WAV:start,end of an interrupting utterance (repeatable)")
    ap.add_argument("--offsets", default="2.0", help="comma list: seconds after the agent's last BOS in run 1")
    ap.add_argument("--pad", type=float, default=25.0)
    ap.add_argument("--prompt", default="You are a helpful assistant.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    tok = eng.tokenizer
    base = torch.cat([load_16k(a.base), torch.zeros(int(a.pad * 16000))])
    ids1, aud1 = drive(eng, base, a.prompt)
    segs, _ = speech_segments(base.numpy(), 16000)
    endpoint = []
    for s, e in segs:
        fb = first_after(ids1, int(round(e / FRAME_S)), T.BOS_ID)
        on = agent_audio_onset(aud1, e)
        endpoint.append({"user_seg": [round(s, 2), round(e, 2)],
                         "bos_latency_s": None if fb is None else round(fb * FRAME_S - e, 2),
                         "audio_latency_s": None if on is None else round(on - e, 2)})
    bos_frames = [k for k, t in enumerate(ids1) if t == T.BOS_ID]
    if not bos_frames:
        raise SystemExit("baseline: the agent never started a turn")
    runs = []
    for spec in a.interrupt:
        wpath, seg = spec.rsplit(":", 1)
        s0, s1 = (float(v) for v in seg.split(","))
        utt = load_16k(wpath)[int(s0 * 16000):int(s1 * 16000)]
        for off in (float(o) for o in a.offsets.split(",")):
            t_int = bos_frames[-1] * FRAME_S + off
            mixed = base.clone()
            i0 = int(t_int * 16000)
            mixed[i0:i0 + utt.numel()] += utt[:max(0, mixed.numel() - i0)]
            ids2, aud2 = drive(eng, mixed, a.prompt)
            t_end = t_int + (s1 - s0)
            f_int, f_end = int(round(t_int / FRAME_S)), int(round(t_end / FRAME_S))
            eos = first_after(ids2, f_int, T.EOS_ID)
            speaking = any(t != T.PAD_ID for t in ids2[bos_frames[-1]:f_int])
            nb = first_after(ids2, f_end, T.BOS_ID)
            silent = agent_audio_silent_from(aud2, t_int)
            r = {"interrupt": f"{Path(wpath).stem}:{seg}", "offset_s": off, "t_start_s": round(t_int, 2),
                 "agent_was_speaking": speaking,
                 "eos_after_onset_s": None if eos is None else round(eos * FRAME_S - t_int, 2),
                 "audio_silent_after_onset_s": None if silent is None else round(silent - t_int, 2),
                 "new_turn_after_end_s": None if nb is None else round(nb * FRAME_S - t_end, 2),
                 "agent_text_after": text_between(tok, ids2, nb, nb + 120) if nb is not None else ""}
            runs.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            sf.write(out / f"agent_{Path(wpath).stem}_{off:.1f}.wav", aud2, OUT_SR)
    res = {"endpointing": endpoint, "interrupts": runs,
           "thresholds_cfg": {k: eng.cfg.s2s.get(k) for k in ("rnnt_eou_frames", "rnnt_bou_frames", "asr_eou_first_turn")}}
    sf.write(out / "agent_baseline.wav", aud1, OUT_SR)
    (out / "bargein.json").write_text(json.dumps({**res, "ids_baseline": ids1}))
    print(json.dumps({"endpointing": endpoint}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
