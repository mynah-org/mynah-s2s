"""Opt-in kernel-level profiling of multi-session rounds (V2_TORCH_PROFILE="k0-k1"; diagnostics only).

* ``label_eartts(tts_model)``: wraps the forward of EarTTS sub-modules with ``torch.profiler.record_function`` labels
  (backbone linears by role q/k/v/o/gate/up/down, attention core, MLP glue, norms, backbone remainder, other direct
  children of the TTS model such as the MoG head / embeddings / codec). Kernels are attributed to the INNERMOST label
  enclosing the CPU op that launched them, so "tts.bb.attn" = attention minus its projections (SDPA, RoPE, KV
  read/write, casts), "tts.bb.mlp" = activation/multiply glue, etc. Kernels replayed inside a CUDA graph cannot be
  attributed to labels (they hang off the graph launch): profile EarTTS eager (``--eartts-batch eager``).
* ``summarize(prof, out)``: kernel table per label (count, total ms, share), kernel names with a dtype guess, memcpy
  and synchronizing runtime calls, written as JSON + text; the chrome trace is exported next to it.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import torch

ROLE = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _wrap(mod, label):
    orig = mod.forward

    def fwd(*a, **kw):
        with torch.profiler.record_function(label):
            return orig(*a, **kw)
    fwd._v2_label = label                       # lets later instance overrides (e.g. fused norms) keep the label
    mod.forward = fwd


def label_eartts(tts) -> dict:
    n = defaultdict(int)
    for name, mod in tts.named_children():              # outer TTS model: codec, prompt latents, EarTTS module
        _wrap(mod, f"tts.{name}")
        n[f"tts.{name}"] += 1
    inner = next((m for m in tts.modules() if isinstance(getattr(m, "backbone", None), torch.nn.Module)), None)
    bb = getattr(inner, "backbone", None)
    if inner is not None:
        for name, mod in inner.named_children():         # EarTTS module: embeddings, MoG head, ... (not the backbone)
            if mod is bb:
                continue
            _wrap(mod, f"tts.ear.{name}")
            n[f"tts.ear.{name}"] += 1
    if bb is not None:
        for name, mod in bb.named_modules():
            if mod is bb:
                continue
            leaf, cls = name.rsplit(".", 1)[-1], type(mod).__name__
            if isinstance(mod, torch.nn.Linear) and leaf in ROLE:
                lab = f"tts.bb.{leaf}"
            elif "Attention" in cls:
                lab = "tts.bb.attn"
            elif "MLP" in cls:
                lab = "tts.bb.mlp"
            elif "Norm" in cls:
                lab = "tts.bb.norm"
            elif "Embedding" in cls:
                lab = "tts.bb.embed"
            else:
                continue
            _wrap(mod, lab)
            n[lab] += 1
        _wrap(bb, "tts.bb")
        n["tts.bb"] += 1
    return dict(n)


def _dtype_guess(k: str) -> str:
    s = k.lower()
    for pat, d in (("bf16", "bf16"), ("bfloat16", "bf16"), ("f16", "fp16"), ("half", "fp16"), ("tf32", "tf32"),
                   ("i8", "int8"), ("s8", "int8")):
        if pat in s:
            return d
    if re.search(r"(^|_)sgemm|gemv2t|gemvx|gemv_kernel|f32|float", s):
        return "fp32?"
    return "?"


def _kind(k: str) -> str:
    s = k.lower()
    if "gemm" in s or "xmma" in s or "cutlass" in s or "matmul" in s:
        return "gemm"
    if "gemv" in s:
        return "gemv"
    if "flash" in s or "attention" in s or "fmha" in s or "sdpa" in s:
        return "attention"
    if "memcpy" in s or "memset" in s:
        return "memcpy"
    if "reduce" in s or "norm" in s:
        return "reduce/norm"
    if "elementwise" in s or "vectorized" in s or "copy" in s or "cast" in s:
        return "elementwise/copy"
    return "other"


def summarize(prof, out, rounds: int) -> dict:
    evs = prof.events()
    by_label = defaultdict(lambda: {"n": 0, "us": 0.0})
    by_label_kernel = defaultdict(lambda: {"n": 0, "us": 0.0})
    by_kernel = defaultdict(lambda: {"n": 0, "us": 0.0})
    sync = defaultdict(lambda: {"n": 0, "us": 0.0})
    total = 0.0

    def label_of(e):
        p = e
        while p is not None:
            nm = p.name
            if nm.startswith("tts.") or nm.startswith("comp:"):
                return nm
            p = p.cpu_parent
        return "(unlabeled)"

    for e in evs:
        nm = e.name
        if any(s in nm for s in ("Synchronize", "cudaMemcpy", "cudaStreamWaitEvent", "cudaEventSynchronize")):
            sync[nm]["n"] += 1
            sync[nm]["us"] += e.cpu_time_total
        ks = getattr(e, "kernels", None) or []
        if not ks:
            continue
        lab = label_of(e)
        for k in ks:
            d = k.duration
            total += d
            by_label[lab]["n"] += 1
            by_label[lab]["us"] += d
            by_kernel[k.name]["n"] += 1
            by_kernel[k.name]["us"] += d
            by_label_kernel[(lab, k.name)]["n"] += 1
            by_label_kernel[(lab, k.name)]["us"] += d
    # every CUDA kernel, including those replayed inside CUDA graphs (not linked to CPU ops): totals by kind/dtype
    allk = defaultdict(lambda: {"n": 0, "us": 0.0})
    try:
        for ke in prof.profiler.kineto_results.events():
            nm = ke.name()                       # skip GPU-projected user annotations (our record_function labels)
            if "CUDA" in str(ke.device_type()) and not nm.startswith(("Memset", "cuda", "comp:", "tts.", "ProfilerStep")):
                allk[nm]["n"] += 1
                allk[nm]["us"] += ke.duration_ns() / 1e3
    except Exception as e:  # noqa: BLE001
        allk = {"_error": {"n": 0, "us": 0.0, "err": str(e)}}
    all_total = sum(v["us"] for v in allk.values())
    r = max(1, rounds)
    res = {"rounds": rounds, "all_kernels_ms_per_round": round(all_total / 1e3 / r, 3),
           "unlinked_kernels_ms_per_round": round((all_total - total) / 1e3 / r, 3), "kernel_ms_total": round(total / 1e3, 2), "kernel_ms_per_round": round(total / 1e3 / r, 3),
           "by_label": {k: {"calls_per_round": round(v["n"] / r, 1), "ms_per_round": round(v["us"] / 1e3 / r, 3),
                            "share": round(v["us"] / max(total, 1e-9), 4)}
                        for k, v in sorted(by_label.items(), key=lambda kv: -kv[1]["us"])},
           "top_kernels": [{"kernel": k[:160], "kind": _kind(k), "dtype": _dtype_guess(k), "calls_per_round":
                            round(v["n"] / r, 1), "ms_per_round": round(v["us"] / 1e3 / r, 3),
                            "share": round(v["us"] / max(total, 1e-9), 4)}
                           for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1]["us"])[:60]],
           "by_label_top_kernels": {},
           "sync_and_copies_host": {k: {"n_per_round": round(v["n"] / r, 2), "host_ms_per_round": round(v["us"] / 1e3 / r, 3)}
                                    for k, v in sorted(sync.items(), key=lambda kv: -kv[1]["us"])}}
    per = defaultdict(list)
    for (lab, k), v in by_label_kernel.items():
        per[lab].append((v["us"], v["n"], k))
    for lab, lst in per.items():
        lst.sort(reverse=True)
        res["by_label_top_kernels"][lab] = [{"kernel": k[:140], "kind": _kind(k), "dtype": _dtype_guess(k),
                                             "calls_per_round": round(n / r, 1), "ms_per_round": round(us / 1e3 / r, 3)}
                                            for us, n, k in lst[:8]]
    akinds = defaultdict(float)
    for k, v in allk.items():
        akinds[(_kind(k), _dtype_guess(k))] += v["us"]
    res["all_by_kind_dtype"] = {f"{a}/{b}": {"ms_per_round": round(us / 1e3 / r, 3), "share": round(us / max(all_total, 1e-9), 4)}
                                for (a, b), us in sorted(akinds.items(), key=lambda kv: -kv[1])}
    res["all_top_kernels"] = [{"kernel": k[:160], "kind": _kind(k), "dtype": _dtype_guess(k), "calls_per_round":
                               round(v["n"] / r, 1), "ms_per_round": round(v["us"] / 1e3 / r, 3),
                               "share": round(v["us"] / max(all_total, 1e-9), 4)}
                              for k, v in sorted(allk.items(), key=lambda kv: -kv[1]["us"])[:40]]
    kinds = defaultdict(float)
    for k, v in by_kernel.items():
        kinds[(_kind(k), _dtype_guess(k))] += v["us"]
    res["by_kind_dtype"] = {f"{a}/{b}": {"ms_per_round": round(us / 1e3 / r, 3), "share": round(us / max(total, 1e-9), 4)}
                            for (a, b), us in sorted(kinds.items(), key=lambda kv: -kv[1])}
    (out / "torch_profile.json").write_text(json.dumps(res, indent=1))
    lines = [f"kernel ms/round linked {res['kernel_ms_per_round']} all {res['all_kernels_ms_per_round']} "
             f"(unlinked = graph replays etc. {res['unlinked_kernels_ms_per_round']}) over {rounds} rounds"]
    lines += [f"ALLKIND {k:28s} {v['ms_per_round']:8.3f} ms/round {100 * v['share']:5.1f}%" for k, v in res["all_by_kind_dtype"].items()]
    lines += [f"ALLKERNEL {x['ms_per_round']:8.3f} {100 * x['share']:5.1f}% {x['kind']:16s} {x['dtype']:6s} {x['calls_per_round']:6.1f}/r {x['kernel']}"
              for x in res["all_top_kernels"][:25]]
    lines += [f"LABEL {k:28s} {v['ms_per_round']:8.3f} ms/round {100 * v['share']:5.1f}% {v['calls_per_round']:7.1f} kern/round"
              for k, v in res["by_label"].items()]
    lines += [f"KIND {k:28s} {v['ms_per_round']:8.3f} ms/round {100 * v['share']:5.1f}%" for k, v in res["by_kind_dtype"].items()]
    lines += [f"KERNEL {x['ms_per_round']:8.3f} {100 * x['share']:5.1f}% {x['kind']:16s} {x['dtype']:6s} {x['calls_per_round']:6.1f}/r {x['kernel']}"
              for x in res["top_kernels"][:30]]
    lines += [f"SYNC {k} {v}" for k, v in res["sync_and_copies_host"].items()]
    (out / "torch_profile.txt").write_text("\n".join(lines) + "\n")
    import os
    if os.environ.get("V2_TORCH_TRACE") == "1":                 # chrome trace is ~0.7 GB for 20 B=4 rounds: opt-in
        try:
            prof.export_chrome_trace(str(out / "torch_trace.json"))
        except Exception as e:  # noqa: BLE001
            res["trace_error"] = str(e)
    return res
