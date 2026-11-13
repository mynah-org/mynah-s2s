"""REFERENCE-mode driver parity: NVIDIA's own driver (StreamingS2SPipeline.run) vs the V2 Session driver,
same process, same engine instance, same WAV, same system prompt. Compares the per-frame token channels
(agent text, user-ASR/RNN-T channel, function), the decoded texts, and the agent waveform (EarTTS + codec output:
run() writes ``<output_dir>/wav/<stem>.wav``; V2 concatenates the per-step audio). Valid for inputs without a tool call: with FC,
run() pre-loads the whole file and uses the blocking real-audio FC path, while V2 (like NVIDIA's Triton backend)
uses the live non-blocking path (streaming_s2s_pipeline.py:940-985 vs 1590+), so they are not comparable by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import Mode, Session, iter_chunks
from mynah_v2.tracing.events import Tracer

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def reseed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="You are a helpful assistant.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--pad-silence", type=float, default=0.0,
                    help="append this many seconds of silence to each input (written to <out>/inputs/)")
    ap.add_argument("--no-llm-cache", action="store_true", help="both drivers on NVIDIA's shipped no-cache Thinker")
    ap.add_argument("--seed", type=int, default=None,
                    help="re-seed torch (CPU+CUDA) before EACH driver: EarTTS samples (noise/top-p/guidance), so the "
                         "waveform is only comparable with the same RNG stream")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="P1 experiment: torch.backends.cudnn.benchmark=True (codec convolutions); not a REFERENCE setting")
    a = ap.parse_args()
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": bool(a.deterministic)})
    eng = ReferenceEngine(cfg, llm_cache=not a.no_llm_cache)
    print("LLM_CACHE", eng.llm_cache_info, flush=True)
    tok = eng.tokenizer
    res = []
    wavs = list(a.wavs)
    if a.pad_silence > 0:   # same padded file for both drivers: long-silence behaviour of the reference itself
        import soundfile as sf
        (out / "inputs").mkdir(exist_ok=True)
        padded = []
        for w in wavs:
            x = load_16k(w).numpy()
            p_ = out / "inputs" / f"{Path(w).stem}_pad{int(a.pad_silence)}s.wav"
            sf.write(p_, np.concatenate([x, np.zeros(int(a.pad_silence * 16000), dtype=x.dtype)]), 16000)
            padded.append(str(p_))
        wavs = padded
    for w in wavs:
        # A) NVIDIA driver. run() closes the session at the end; capture stream 0's saved token tensors first.
        cap = {}
        orig_close = eng.pipeline.close_session

        def close_capture():
            cap["tt"] = eng.pipeline.get_or_create_state(0).get_token_tensors()
            eng.pipeline.close_session = orig_close
            return orig_close()
        eng.pipeline.close_session = close_capture
        if a.seed is not None:
            reseed(a.seed)
        po = eng.pipeline.run([w], options=[S2SRequestOptions(system_prompt=a.prompt)])
        gt, ga, nn, gf = cap["tt"]
        nv_ids_t, nv_ids_f = gt[0, :nn].tolist(), (gf[0, :nn].tolist() if gf is not None else [])
        nv_ids_a = ga[0, :nn].tolist() if ga is not None else []
        nv_text = tok.ids_to_tokens(nv_ids_t)
        nv_fc = tok.ids_to_tokens(nv_ids_f)
        # B) V2 driver, same chunking (0.16 s, last chunk zero-padded), no tail, last chunk flagged is_last
        tr = Tracer(None, session_id=Path(w).stem)
        sess = Session(eng, tr, Mode.REFERENCE)
        if a.seed is not None:
            reseed(a.seed)
        sess.open(a.prompt)
        ids_t, ids_f, audio, asr_txt = [], [], [], []
        chunks = list(iter_chunks(load_16k(w), eng.chunk_samples))
        for i, c in enumerate(chunks):
            o = eng.step(c, is_last=(i == len(chunks) - 1))
            ids_t += o.new_text_ids
            ids_f += o.new_func_ids
            audio.append(o.audio.float().cpu().reshape(-1))
            asr_txt.append(o.asr_delta)
        import soundfile as sf
        sf.write(out / f"{Path(w).stem}_v2.wav", torch.cat(audio).numpy() if audio else np.zeros(0), 22050)
        tt = eng.token_tensors()
        if tt is not None:  # saved on is_last: prefer the authoritative final tensors
            gt, ga, n, gf = tt
            ids_t = gt[0, :n].tolist()
            ids_f = gf[0, :n].tolist() if gf is not None else ids_f
            ids_a = ga[0, :n].tolist() if ga is not None else []
        else:
            ids_a = []
        sess.closed = True
        eng.close()
        v2_text = tok.ids_to_tokens(ids_t) if hasattr(tok, "ids_to_tokens") else [str(i) for i in ids_t]
        v2_fc = tok.ids_to_tokens(ids_f) if hasattr(tok, "ids_to_tokens") else [str(i) for i in ids_f]
        n = min(len(nv_text), len(v2_text))
        first_div = next((i for i in range(n) if nv_text[i] != v2_text[i]), None)
        agree = sum(nv_text[i] == v2_text[i] for i in range(n)) / max(n, 1)
        v2_audio = torch.cat(audio).numpy() if audio else np.zeros(0, np.float32)
        nv_wav = out / "_nemo" / "wav" / f"{Path(w).stem}.wav"
        aud = {"nvidia_wav": str(nv_wav), "found": nv_wav.exists()}
        if nv_wav.exists():
            nv_audio, _sr = sf.read(nv_wav, dtype="float32")
            m = min(len(nv_audio), len(v2_audio))
            d = np.abs(nv_audio[:m] - v2_audio[:m]) if m else np.zeros(1)
            aud.update({"nvidia_samples": len(nv_audio), "v2_samples": len(v2_audio),
                        "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                        "nvidia_rms": float(np.sqrt(np.mean(nv_audio[:m] ** 2))) if m else 0.0,
                        "identical": bool(len(nv_audio) == len(v2_audio) and float(d.max()) == 0.0)})
        r = {"wav": w, "ids_identical": nv_ids_t == ids_t and nv_ids_f[:len(ids_f)] == ids_f[:len(nv_ids_f)],
             "nvidia_frames": len(nv_text), "v2_frames": len(v2_text), "text_agreement": round(agree, 4),
             "first_divergence_frame": first_div, "fc_identical": nv_fc[:n] == v2_fc[:n],
             "nvidia_text": po.texts[0] if po.texts else "", "deterministic": a.deterministic,
             "llm_cache": eng.llm_cache_info is not None, "seed": a.seed,
             "asr_ids_identical": nv_ids_a == ids_a, "asr_text_nvidia": po.asr_texts[0] if po.asr_texts else "",
             "asr_text_v2": "".join(asr_txt), "audio": aud,
             "nvidia_raw_head": " ".join(nv_text[:60]), "v2_raw_head": " ".join(v2_text[:60])}
        res.append(r)
        print(json.dumps({k: r[k] for k in ("wav", "ids_identical", "nvidia_frames", "v2_frames", "text_agreement",
                                             "first_divergence_frame", "fc_identical", "asr_ids_identical", "audio")}), flush=True)
        (out / f"{Path(w).stem}.json").write_text(json.dumps(
            {**r, "nvidia_raw": nv_text, "v2_raw": v2_text, "nvidia_fc_raw": nv_fc, "v2_fc_raw": v2_fc}, indent=1))
    (out / "parity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
