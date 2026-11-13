"""VRAM audit of the VoiceChat V2 engine (what is resident, what grows per session), for the B=2 → B=4 memory plan.

Model: M_total(N) = M_shared + N · M_session + M_runtime(N). This script measures, on ONE GPU process:
  1. resident weights by sub-component and dtype (unique storages, so tied/shared tensors count once):
     Thinker (llm: MLP / Mamba / attention / norms / embeddings), heads, perception, other STT, EarTTS (backbone,
     embeddings incl. RVQ, MoG head, other), codec, other; plus tensors held OUTSIDE the module tree by our runtime
     (e.g. `_v2_*_saved` A/B copies, graph static buffers) found by scanning engine attributes;
  2. allocator state per phase (allocated / reserved / free / peak): after load, after each experiment, after a
     B=1 session of --chunks chunks (sampled every 10 chunks → per-chunk growth = session state slope), after close
     (residual = what a closed session leaves behind).
Optional experiments (--exp, perf names incl. `wor:w8:mlp` resident) are applied after the baseline audit, and the
weight audit is repeated, so "GiB actually freed" is measured, not computed. Writes <out>/memory_audit.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import weakref
from collections import defaultdict
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402

GB = 2**30


def gib(x):
    return round(x / GB, 3)


def mem(tag):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"phase": tag, "allocated_GiB": gib(torch.cuda.memory_allocated()),
            "reserved_GiB": gib(torch.cuda.memory_reserved()), "device_free_GiB": gib(free), "total_GiB": gib(total),
            "peak_allocated_GiB": gib(torch.cuda.max_memory_allocated())}


def classify(path: str) -> str:
    p = path
    if ".stt_model.llm." in p or p.startswith("stt_model.llm."):
        if ".mixer." in p:
            if any(k in p for k in (".up_proj", ".down_proj")):
                return "thinker.mlp"
            if any(k in p for k in (".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                return "thinker.attention"
            if any(k in p for k in (".in_proj", ".out_proj", ".conv1d", "A_log", ".D", ".dt_bias", ".norm.")):
                return "thinker.mamba"
            return "thinker.mixer_other"
        if "embed" in p:
            return "thinker.embeddings"
        if "norm" in p:
            return "thinker.norms"
        return "thinker.other"
    if "stt_model.lm_head" in p or "stt_model.function_head" in p or "stt_model.asr_head" in p:
        return "heads"
    if "stt_model.embed" in p:
        return "stt.embeddings"
    if "stt_model.perception" in p:
        return "perception"
    if "stt_model." in p:
        return "stt.other"
    if "audio_codec" in p:
        return "codec"
    if "tts_model" in p:
        if "backbone" in p:
            return "eartts.backbone"
        if "rvq" in p or "embed" in p:
            return "eartts.embeddings_rvq"
        if "mog" in p:
            return "eartts.mog_head"
        return "eartts.other"
    return "other"


def weight_audit(model) -> dict:
    seen, by = set(), defaultdict(lambda: defaultdict(int))
    for kind, it in (("param", model.named_parameters(remove_duplicate=False)),
                     ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, t in it:
            if t is None or not t.is_cuda or t.numel() == 0:
                continue
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            if key in seen:
                continue
            seen.add(key)
            by[classify(name)][str(t.dtype)] += t.untyped_storage().nbytes()
    out = {k: {d: gib(b) for d, b in v.items()} | {"total": gib(sum(v.values()))} for k, v in sorted(by.items())}
    out["_sum_GiB"] = round(sum(v["total"] for v in out.values() if isinstance(v, dict)), 3)
    return out, seen


def outside_tensors(engine, seen) -> dict:
    """CUDA tensors reachable from engine attributes but NOT part of the module tree (A/B copies, static buffers)."""
    res, visited = defaultdict(int), set()

    def walk(obj, path, depth):
        if depth > 4 or id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.numel():
                key = (obj.untyped_storage().data_ptr(), obj.untyped_storage().nbytes())
                if key not in seen:
                    seen.add(key)
                    res[path.split(".")[0]] += obj.untyped_storage().nbytes()
            return
        if isinstance(obj, torch.nn.Module):
            for n, p in obj.named_parameters(recurse=True):
                walk(p, f"{path}.{n}", depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj[:64]):
                walk(x, f"{path}", depth + 1)
        elif isinstance(obj, dict):
            for k, x in list(obj.items())[:64]:
                walk(x, f"{path}", depth + 1)
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for k, x in list(vars(obj).items()):
                walk(x, f"{path}.{k}" if path else k, depth + 1)

    for k, v in vars(engine).items():
        if k.startswith("_v2") or k in ("_v2_thinker_graph", "_v2_full_graph"):
            walk(v, k, 0)
    return {k: gib(b) for k, b in sorted(res.items(), key=lambda kv: -kv[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc-ctx-fix", default="auto", choices=["auto", "on", "off"],
                    help="FC hand-back fix (selective): auto = ON for the official VoiceChat checkpoint only; off = rollback")
    ap.add_argument("--profile", default="optimized-graph", help="perf profile applied before the baseline audit")
    ap.add_argument("--exp", default="", help="comma list of experiments applied AFTER the baseline audit")
    ap.add_argument("--chunks", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {"phases": []}
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}), fc_ctx_fix=a.fc_ctx_fix)
    res["phases"].append(mem("after_load"))
    torch.cuda.reset_peak_memory_stats()
    if a.profile:
        res["profile"] = str(perf.apply_profile(a.profile, eng))
    res["phases"].append(mem("after_profile"))
    model = eng.wrapper.model
    w0, seen = weight_audit(model)
    res["weights_baseline"] = w0
    res["outside_module_tree_baseline"] = outside_tensors(eng, set(seen))
    if a.exp:
        names = a.exp.split(",")
        # resident quantization must come BEFORE any graph capture over the swapped modules (else the graph keeps
        # pointers to / captures over the BF16 originals)
        wor = [i for i, n in enumerate(names) if n.startswith("wor:")]
        graphs = [i for i, n in enumerate(names) if "graph" in n]
        if wor and graphs and min(graphs) < max(wor):
            raise SystemExit(f"order error: {names} — put wor:* before graph experiments")
        # weakrefs to every CUDA param, by group: after the swap + gc, dead refs = storages really released
        refs = defaultdict(list)
        for name, t in model.named_parameters(remove_duplicate=True):
            if t.is_cuda:
                refs[classify(name)].append((weakref.ref(t), t.untyped_storage().nbytes()))
        alloc0 = torch.cuda.memory_allocated()
        res["exp"] = {n: str(perf.enable(n, eng)) for n in names}
        gc.collect()
        torch.cuda.empty_cache()
        res["phases"].append(mem("after_exp"))
        w1, seen1 = weight_audit(model)
        res["weights_after_exp"] = w1
        res["outside_module_tree_after_exp"] = outside_tensors(eng, set(seen1))
        rel = {}
        for g, lst in refs.items():
            dead = [b for r, b in lst if r() is None]
            if dead:
                rel[g] = {"params_released": len(dead), "of": len(lst), "GiB_released": gib(sum(dead))}
        res["exp_storage_release"] = rel
        res["exp_allocated_delta_GiB"] = gib(torch.cuda.memory_allocated() - alloc0)   # negative = really freed
        res["exp_weights_delta_GiB"] = round(w1["_sum_GiB"] - w0["_sum_GiB"], 3)       # module-tree bytes delta
    # one B=1 session: per-chunk growth
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 60)])
    chunks = list(iter_chunks(x, eng.chunk_samples))[: a.chunks]
    torch.cuda.reset_peak_memory_stats()
    res["phases"].append(mem("before_open"))
    eng.open("You are a helpful assistant.")
    res["phases"].append(mem("after_open"))
    res["session_open_cost_GiB"] = round(res["phases"][-1]["allocated_GiB"] - res["phases"][-2]["allocated_GiB"], 3)
    samples = []
    with torch.inference_mode():
        for k, c in enumerate(chunks):
            eng.step(c)
            if k % 10 == 0 or k == len(chunks) - 1:
                samples.append({"chunk": k, **{kk: v for kk, v in mem("").items() if kk != "phase"}})
    res["session_samples"] = samples
    if len(samples) >= 3:
        a0, a1 = samples[1], samples[-1]                       # skip chunk 0 (first-step allocations / captures)
        res["session_growth_MiB_per_chunk"] = round((a1["allocated_GiB"] - a0["allocated_GiB"]) * 1024
                                                    / max(1, a1["chunk"] - a0["chunk"]), 2)
    eng.close()
    gc.collect()
    res["phases"].append(mem("after_close"))
    torch.cuda.empty_cache()
    res["phases"].append(mem("after_close_empty_cache"))
    res["gpu"] = torch.cuda.get_device_name()
    (out / "memory_audit.json").write_text(json.dumps(res, indent=1))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "session_samples"}), flush=True)


if __name__ == "__main__":
    main()
