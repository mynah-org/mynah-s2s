"""GPU-busy vs host time per component on a steady window (P1: overhead-bound or bandwidth-bound?).

Labels (instance wrappers, no upstream edit): ``v2.llm_backbone`` (NemotronH forward), ``v2.tts_step``
(EarTTS ``infer_codes_one_step``), ``v2.codec_decode`` (``audio_codec.decode``), ``v2.perception``
(perception cache manager step). torch.profiler records CPU and CUDA activity for ``--chunks`` chunks after
``--warmup`` chunks; for each label we report host wall (CPU total), summed GPU kernel time and the kernel count, per
chunk. GPU time << host time ⇒ launch/overhead-bound (CUDA graphs / compile help); GPU time ≈ host time ⇒ device-bound
(bytes or FLOPs must go down).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def labelled(fn, name):
    @functools.wraps(fn)
    def w(*a, **k):
        with record_function(name):
            return fn(*a, **k)
    if hasattr(fn, "__wrapped_v2__"):
        w.__wrapped_v2__ = fn.__wrapped_v2__
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--exp", default="", help="comma list of mynah_v2.model.perf experiments to enable first")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--detail", default="",
                    help="comma list of module paths under the VoiceChat model (e.g. stt_model.lm_head, "
                         "tts_model.tts_model.backbone): label their forward, report param bytes per call and the "
                         "effective bandwidth = bytes x calls / GPU kernel time (missing paths are reported, not fatal)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)          # immutable run directory
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    from mynah_v2.model import perf
    exp_info = [perf.enable(n, eng) for n in filter(None, a.exp.split(","))]
    m = eng.wrapper.model
    llm = m.stt_model.llm
    llm.forward = labelled(llm.forward, "v2.llm_backbone")
    m.tts_model.infer_codes_one_step = labelled(m.tts_model.infer_codes_one_step, "v2.tts_step")
    m.tts_model.audio_codec.decode = labelled(m.tts_model.audio_codec.decode, "v2.codec_decode")
    from mynah_v2.model.perf import _find_eartts
    ear = _find_eartts(eng)
    ear.generate_step = labelled(ear.generate_step, "v2.tts_gen_step")     # MoG refinement (static shapes)
    if hasattr(ear, "mog_head"):
        ear.mog_head.infer = labelled(ear.mog_head.infer, "v2.tts_mog_infer")
    pcm = getattr(eng.wrapper, "perception_cache_mgr", None)
    if pcm is not None:
        pcm.step = labelled(pcm.step, "v2.perception")
    perc = m.stt_model.perception                  # full-buffer path (exact profile; perception_graph wraps this)
    perc.forward = labelled(perc.forward, "v2.perception_full")
    detail = {}
    for path in filter(None, a.detail.split(",")):
        mod = m
        try:
            for part in path.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        except (AttributeError, IndexError, TypeError):
            detail[path] = {"missing": True}
            continue
        lab = "mod:" + path
        nbytes = sum(p.numel() * p.element_size() for p in mod.parameters())
        dtypes = sorted({str(p.dtype) for p in mod.parameters()})
        mod.forward = labelled(mod.forward, lab)
        detail[path] = {"label": lab, "param_GB": round(nbytes / 1e9, 4), "dtypes": dtypes}
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 30)])
    chunks = list(iter_chunks(x, eng.chunk_samples))[: a.warmup + a.chunks]
    eng.open("You are a helpful assistant.")
    for c in chunks[: a.warmup]:
        eng.step(c)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for c in chunks[a.warmup:]:
            with record_function("v2.chunk"):
                eng.step(c)
    eng.close()
    # GPU time attributed to a label = kernels launched while the label was on the host stack
    evs = prof.events()
    labels = ["v2.chunk", "v2.llm_backbone", "v2.tts_step", "v2.tts_gen_step", "v2.tts_mog_infer", "v2.codec_decode",
              "v2.perception", "v2.perception_full"] + [d["label"] for d in detail.values() if "label" in d]
    res = {}
    labset = set(labels)
    for lab in labels:
        spans = [e for e in evs if e.name == lab]
        host = sum(e.time_range.elapsed_us() for e in spans) / 1000.0
        gpu, nk, gpu_ex, nk_ex = 0.0, 0, 0.0, 0
        for e in spans:
            stack = [(e, True)]                    # (event, still exclusive to this label?)
            while stack:
                cur, excl = stack.pop()
                for k in cur.kernels:
                    gpu += k.duration / 1000.0
                    nk += 1
                    if excl:
                        gpu_ex += k.duration / 1000.0
                        nk_ex += 1
                for ch in cur.cpu_children:
                    stack.append((ch, excl and ch.name not in labset))
        n = max(1, a.chunks)
        res[lab] = {"calls_per_chunk": round(len(spans) / n, 2), "host_ms_per_chunk": round(host / n, 2),
                    "gpu_kernel_ms_per_chunk": round(gpu / n, 2), "kernels_per_chunk": round(nk / n, 1),
                    "gpu_exclusive_ms_per_chunk": round(gpu_ex / n, 2), "kernels_exclusive_per_chunk": round(nk_ex / n, 1),
                    "gpu_busy_ratio": round(gpu / host, 3) if host else None}
    res["_note"] = ("gpu_kernel_ms = inclusive (kernels under nested labels counted in every enclosing label: do not "
                    "sum across labels); gpu_exclusive_ms = minus kernels under nested labelled children; host_ms is "
                    "profiler-inflated wall; eff_GBps uses module PARAMETER bytes x calls (an estimate of weight "
                    "traffic, not measured DRAM bytes; > peak means the module does not read all its params per call "
                    "or hits L2). Work launched from other threads (perception worker) is not attributed to labels.")
    # host<->device synchronisation points per chunk (explicit syncs, scalar reads, D2H copies)
    sync = {}
    for e in prof.key_averages():
        nm = e.key
        if any(k in nm for k in ("Synchronize", "Memcpy", "_local_scalar_dense", "aten::item")):
            sync[nm] = round(e.count / max(1, a.chunks), 1)
    res["sync_per_chunk"] = sync
    for path, d in detail.items():
        if "label" in d:
            r = res[d["label"]]
            gb = d["param_GB"] * r["calls_per_chunk"]
            d.update(r, weight_GB_per_chunk=round(gb, 3),
                     eff_GBps=round(gb / r["gpu_kernel_ms_per_chunk"] * 1000, 1) if r["gpu_kernel_ms_per_chunk"] else None)
    res["detail"] = detail
    res["experiments"] = {"names": a.exp, "info": exp_info, "cudnn_benchmark": a.cudnn_benchmark}
    top = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    (out / "kernels_top.txt").write_text(top)
    (out / "kernels.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
