"""Codec (EarTTS VAE) streaming-decode numerics + GPU cost, on the real checkpoint codec (GPU).

Decides three candidates of the codec path (mynah_v2.model.codec_async):
  A. codec_nosync (cached ISTFT envelope) vs upstream spec_to_wav: decoded audio must be bit-identical;
  B. FC "2 codes per decode": streaming decode 1+1 (what the FC thread does) vs 2-at-once with the cache (main path);
  C. batched cross-session decode: B sessions x 2 codes, individually (each its own cache) vs one [B, 2, Q] call with a
     batched cache.
Each reports max |diff| of the concatenated audio over the whole sequence (and bit_identical). Codes: random valid ids
(the codec is a deterministic conv net: numerics do not depend on the codes being speech) plus the checkpoint's
silence code. Timing (GPU span via events, host ms) is indicative only on GB10.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mynah_v2.model import codec_async as ca
from mynah_v2.model.reference import ReferenceEngine, build_cfg


def decode_stream(codec, codes, step, Cache, cache=None):
    """codes [B, N, Q] -> audio [B, 1, S], decoding `step` codes per call with one streaming cache."""
    from nemo.collections.speechlm2.parts.precision import fp32_precision
    cache = Cache() if cache is None else cache
    outs = []
    with fp32_precision(), torch.no_grad():
        for i in range(0, codes.shape[1], step):
            c = codes[:, i:i + step]
            ln = torch.full((c.shape[0],), c.shape[1], dtype=torch.long, device=c.device)
            outs.append(codec.decode(c, ln, cache=cache)[0])
    return torch.cat(outs, dim=-1)


def diff(a, b):
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "bit_identical": False}
    d = (a - b).abs()
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "bit_identical": bool(torch.equal(a, b)),
            "ref_rms": float(a.pow(2).mean().sqrt())}


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    host = (time.perf_counter() - t0) * 1e3 / reps
    torch.cuda.synchronize()
    return {"gpu_ms": round(e0.elapsed_time(e1) / reps, 3), "host_ms": round(host, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="codes per sequence (even)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--nondeterministic", action="store_true", help="s2s.deterministic=false (serving kernels)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": not a.nondeterministic}))
    import nemo.collections.speechlm2.modules.ear_tts_vae_codec as cmod
    Cache = cmod.CausalConv1dCache
    tts = eng.wrapper.model.tts_model
    codec = tts.audio_codec
    cfg = codec.config
    Q, V = int(cfg.num_quantizers), int(cfg.codebook_size)
    dev = next(codec.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(1234)
    codes = torch.randint(0, V, (a.batch, a.n, Q), generator=g).to(dev)
    sil = getattr(tts, "codec_silence_tokens", None)
    if sil is not None:                                    # some silence frames, as in real streams
        codes[:, : a.n // 4] = sil.view(1, 1, -1).to(dev)
    rep = {"codec": {"Q": Q, "V": V, "dtype": str(next(codec.parameters()).dtype), "wav_to_token_ratio":
                     int(cfg.wav_to_token_ratio)}, "n_codes": a.n, "batch": a.batch,
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark}

    one = codes[:1]
    base2 = decode_stream(codec, one, 2, Cache)
    base1 = decode_stream(codec, one, 1, Cache)
    rep["repeat_2_vs_2"] = diff(base2, decode_stream(codec, one, 2, Cache))
    # A. codec_nosync
    ca.nosync_enable(eng)
    try:
        rep["A_nosync_step2"] = diff(base2, decode_stream(codec, one, 2, Cache))
        rep["A_nosync_step1"] = diff(base1, decode_stream(codec, one, 1, Cache))
    finally:
        ca.nosync_disable(eng)
    # B. 1+1 vs 2-at-once
    rep["B_1plus1_vs_2"] = diff(base1, base2)
    # C. batched sessions vs individual (2 codes per call)
    indiv = torch.cat([decode_stream(codec, codes[b:b + 1], 2, Cache) for b in range(a.batch)], dim=0)
    batched = decode_stream(codec, codes, 2, Cache)
    rep["C_batched_vs_individual"] = diff(indiv, batched)
    rep["C_per_session_max"] = [float((indiv[b] - batched[b]).abs().max()) for b in range(a.batch)]

    # D. stateless windowed re-decode with causal run-up (llama.cpp-style: no codec cache; decode codes[i-R : i+2]
    # from zero state, keep the last 2 codes' samples) vs the full stateless decode and vs the cached stream.
    from nemo.collections.speechlm2.parts.precision import fp32_precision
    r = int(cfg.wav_to_token_ratio)

    def dec_nocache(c):
        with fp32_precision(), torch.no_grad():
            return codec.decode(c, torch.full((c.shape[0],), c.shape[1], dtype=torch.long, device=dev))[0]
    full = dec_nocache(one)
    lag = None
    best = None
    for s in range(0, 2 * r):                               # cached stream = full decode delayed by `lag` samples
        m = min(full.shape[-1] - s, base2.shape[-1]) - s
        if m <= r:
            break
        e = float((base2[..., s:s + m] - full[..., :m]).abs().max())
        if best is None or e < best:
            best, lag = e, s
    rep["D_cached_vs_full_stateless"] = {"lag_samples": lag, "max_abs_diff_aligned": best,
                                         "len": [int(full.shape[-1]), int(base2.shape[-1])]}
    for R in (2, 4, 8):
        mx_full, mx_tail_ok = 0.0, 0.0
        for i in range(R, a.n - 2, 2):
            w = dec_nocache(one[:, i - R:i + 2])[..., -2 * r:]
            ref = full[..., i * r:(i + 2) * r]
            d = (w - ref).abs()
            mx_full = max(mx_full, float(d.max()))
            mx_tail_ok = max(mx_tail_ok, float(d[..., : 2 * r - lag].max()) if lag else float(d.max()))
        rep[f"D_runup{R}_vs_full_stateless"] = {"max_abs_diff": mx_full,
                                                 "max_abs_diff_excl_last_lag_samples": mx_tail_ok}

    # timing (indicative, GB10): per decode call, warm caches
    from nemo.collections.speechlm2.parts.precision import fp32_precision

    def call(c, cache):
        def f():
            with fp32_precision(), torch.no_grad():
                codec.decode(c, torch.full((c.shape[0],), c.shape[1], dtype=torch.long, device=dev), cache=cache)
        return f
    caches = [Cache() for _ in range(a.batch)]
    for b in range(a.batch):
        decode_stream(codec, codes[b:b + 1, :8], 2, Cache, cache=caches[b])
    bc = Cache()
    decode_stream(codec, codes[:, :8], 2, Cache, cache=bc)
    t = {}
    for tag, en in (("orig", False), ("nosync", True)):
        if en:
            ca.nosync_enable(eng)
        try:
            t[f"{tag}_1code_x2"] = timed(lambda: (call(codes[:1, :1], caches[0])(), call(codes[:1, 1:2], caches[0])()),
                                         a.reps)
            t[f"{tag}_2codes_x1"] = timed(call(codes[:1, :2], caches[0]), a.reps)
            t[f"{tag}_B{a.batch}_individual_2codes"] = timed(
                lambda: [call(codes[b:b + 1, :2], caches[b])() for b in range(a.batch)], a.reps)
            t[f"{tag}_B{a.batch}_batched_2codes"] = timed(call(codes[:, :2], bc), a.reps)
            t[f"{tag}_B{a.batch}_batched_stateless_runup8"] = timed(lambda: dec_nocache(codes[:, :10]), a.reps)
        finally:
            if en:
                ca.nosync_disable(eng)
    rep["timing_per_iter"] = t
    (out / "codec_numerics.json").write_text(json.dumps(rep, indent=1))
    print("CODEC_NUMERICS", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
