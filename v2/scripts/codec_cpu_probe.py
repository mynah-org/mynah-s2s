"""Probe real EarTTS codec CPU execution without changing the serving path.

This is deliberately an isolated experiment: it deep-copies the loaded codec
to CPU, decodes the same code stream on GPU and CPU, and reports numerical
parity plus warm timing. It is not the persistent worker yet; the result tells
us whether the CPU path is worth implementing behind an opt-in adapter.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg


def decode_stream(codec, codes, cache_cls, step=2):
    cache = cache_cls()
    outs = []
    with torch.no_grad():
        for i in range(0, codes.shape[1], step):
            chunk = codes[:, i:i + step]
            lengths = torch.full((chunk.shape[0],), chunk.shape[1], dtype=torch.long, device=codes.device)
            outs.append(codec.decode(chunk, lengths, cache=cache)[0])
    return torch.cat(outs, dim=-1)


def timed(fn, reps):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rewrite", action="store_true", help="replace safe depthwise/pointwise Conv1d paths on CPU")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    import nemo.collections.speechlm2.modules.ear_tts_vae_codec as cmod

    gpu_codec = eng.wrapper.model.tts_model.audio_codec.eval()
    cache_cls = cmod.CausalConv1dCache
    cfg = gpu_codec.config
    q, vocab = int(cfg.num_quantizers), int(cfg.codebook_size)
    device = next(gpu_codec.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(1234)
    codes_gpu = torch.randint(0, vocab, (1, a.n, q), generator=gen).to(device)

    result = {"n": a.n, "q": q, "vocab": vocab, "gpu_device": str(device), "cpu_copy": "pending"}
    gpu = decode_stream(gpu_codec, codes_gpu, cache_cls)
    result["gpu_ms"] = round(timed(lambda: decode_stream(gpu_codec, codes_gpu, cache_cls), a.reps), 3)

    try:
        cpu_codec = copy.deepcopy(gpu_codec).to("cpu").eval()
    except Exception as exc:  # keep the probe useful if NeMo has non-copyable state
        result["cpu_copy"] = "failed"
        result["cpu_error"] = f"{type(exc).__name__}: {exc}"
        (out / "codec_cpu_probe.json").write_text(json.dumps(result, indent=1))
        print("CODEC_CPU_PROBE", json.dumps(result), flush=True)
        return

    result["cpu_copy"] = "ok"
    if a.rewrite:
        from mynah_v2.model.codec_cpu import rewrite_codec
        result["rewrite"] = rewrite_codec(cpu_codec)
    codes_cpu = codes_gpu.detach().cpu()
    cpu = decode_stream(cpu_codec, codes_cpu, cache_cls)
    d = (gpu.detach().cpu().float() - cpu.float()).abs()
    result["cpu_ms"] = round(timed(lambda: decode_stream(cpu_codec, codes_cpu, cache_cls), a.reps), 3)
    result["shape"] = [list(gpu.shape), list(cpu.shape)]
    result["max_abs_diff"] = float(d.max())
    result["mean_abs_diff"] = float(d.mean())
    result["gpu_rms"] = float(gpu.float().pow(2).mean().sqrt())
    result["cpu_rms"] = float(cpu.float().pow(2).mean().sqrt())
    result["parity_ok"] = bool(gpu.shape == cpu.shape and d.max() < 1e-4)
    (out / "codec_cpu_probe.json").write_text(json.dumps(result, indent=1))
    print("CODEC_CPU_PROBE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
