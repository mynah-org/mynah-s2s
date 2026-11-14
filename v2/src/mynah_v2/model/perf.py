# SPDX-License-Identifier: MIT AND Apache-2.0
# Original code: MIT (see LICENSE). Portions derived from Apache-2.0 sources and modified, under the Apache License 2.0
# (LICENSES/Apache-2.0.txt): the causal-mask construction in mamba_host_pos, from NemotronH modeling_nemotron_h.py
# (Copyright 2024 HuggingFace Inc. team; Copyright (c) 2025 NVIDIA CORPORATION), and _make_generate_step, from NeMo
# Speech ear_tts_model.py (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES). See NOTICE.md.
"""EXPERIMENTAL performance patches for the native engine — never used by REFERENCE, each one isolated and reversible.

Every experiment is ``enable(name, engine) -> info`` / ``disable(name, engine)`` and must keep the BF16 math: the gate
is ``scripts/perf_ab.py`` (same process, deterministic, same RNG seed: token identity on every channel + waveform).
Evidence for each target: ``scripts/sync_audit.py`` (I1, GB10): 163 blocking host<->device calls per 160 ms chunk,
227 ms of host time blocked, of which
  * 54/chunk (125.7 ms) — NemotronH Mamba mixer ``cuda_kernels_forward``: ``cache_position[0] > 0`` on a CUDA tensor
    (modeling_nemotron_h.py:375), i.e. one host sync per Mamba layer per step; plus ``_update_mamba_mask`` (l.1466)
    and the causal-mask size ``cache_position[-1] + 1`` (l.1433) once per step;
  * 7/chunk wrapper timers ``torch.cuda.synchronize()`` around perception / stt / tts / codec / step (W, only used to
    log "Time taken for …");
  * 36/chunk EarTTS stop/eos checks (``generate_step``) — not addressed here yet.
"""

from __future__ import annotations

import re
import sys
import types
import os
from typing import Callable

import torch

EXPERIMENTS: dict[str, tuple[Callable, Callable]] = {}


def experiment(name):
    def deco(fns):
        EXPERIMENTS[name] = fns
        return fns
    return deco


def _parametric(name: str):
    """``wo:<fmt>:<selector>``: weight-only format on a precision.select selector, '+' standing for ',' (the --exp lists
    are comma-separated), e.g. ``wo:f8:all``, ``wo:w8:all+heads``, ``wo:w8:mlp+mamba``, ``wo:f8:all+!attention``."""
    if name.startswith(("wo:", "wor:")):
        kind, fmt, sel = name.split(":", 2)
        tag = "wo_" + re.sub(r"\W", "_", f"{fmt}_{sel}")
        sel = sel.replace("+", ",")
        if kind == "wor":                    # RESIDENT: the BF16 originals are released (deployment memory mode)
            def _no_restore(e):
                raise NotImplementedError(f"{name}: resident mode freed the BF16 originals; reload the engine")
            return (lambda e: _wo_swap(e, fmt, sel, tag, keep=False), _no_restore)
        return (lambda e: _wo_swap(e, fmt, sel, tag), lambda e: _wo_restore(e, tag))
    return EXPERIMENTS[name]


def enable(name: str, engine) -> dict:
    return _parametric(name)[0](engine)


def disable(name: str, engine) -> None:
    _parametric(name)[1](engine)


# Named, validated combinations (never applied to REFERENCE by the CLIs). "optimized-exact" = the experiments that are
# bit-identical to the baseline (tokens + seeded waveform, N1 2026-11-11: p50 304 -> 291 ms with cudnn benchmark).
PROFILES: dict[str, tuple[str, ...]] = {
    "optimized-exact": ("mamba_host_pos", "no_timer_sync", "eartts_graph_genstep"),
}


def apply_profile(name: str, engine) -> dict:
    return {n: enable(n, engine) for n in PROFILES[name]}


def remove_profile(name: str, engine) -> None:
    for n in reversed(PROFILES[name]):
        disable(n, engine)


# ------------------------------------------------------------------------------------------------ mamba_host_pos
def _host_pos(llm):
    return getattr(llm, "_v2_host_pos", None)        # (pos, t) set by llm_cache.position_injecting_forward


def _mamba_enable(engine) -> dict:
    llm = engine.wrapper.model.stt_model.llm
    if engine.llm_cache_info is None:
        raise RuntimeError("mamba_host_pos needs the incremental Thinker (llm_cache)")
    n = 0
    for layer in llm.layers:
        if getattr(layer, "block_type", None) != "mamba" or hasattr(layer.mixer.forward, "__v2_hostpos__"):
            continue
        orig = layer.mixer.forward

        def fwd(hidden_states, *a, _orig=orig, **kw):
            hp = _host_pos(llm)
            if hp is not None and kw.get("cache_position") is not None:
                # the CUDA mixer path only uses cache_position for `cache_position[0] > 0`: a host tensor with the
                # same values answers it without a device sync
                kw["cache_position"] = torch.arange(hp[0], hp[0] + hp[1])
            return _orig(hidden_states, *a, **kw)
        fwd.__v2_hostpos__ = orig
        layer.mixer.forward = fwd
        n += 1
    orig_mm, orig_cm = llm._update_mamba_mask, llm._update_causal_mask

    def mamba_mask(attention_mask, cache_position):
        hp = _host_pos(llm)
        if hp is not None and hp[0] > 0:
            return None                                       # = original branch `cache_position[0] > 0`
        return orig_mm(attention_mask, cache_position)

    def causal_mask(attention_mask, input_tensor, cache_position):
        hp = _host_pos(llm)
        if hp is None or attention_mask is not None or llm.config._attn_implementation == "flash_attention_2":
            return orig_cm(attention_mask, input_tensor, cache_position)
        # original code (modeling_nemotron_h.py:1424-1457) with the size taken from the host position
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = hp[0] + hp[1]
        cm = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            cm = torch.triu(cm, diagonal=1)
        cm *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        return cm[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    llm._update_mamba_mask, llm._update_causal_mask = mamba_mask, causal_mask
    llm._v2_hostpos_orig = (orig_mm, orig_cm)
    return {"mamba_layers_patched": n}


def _mamba_disable(engine) -> None:
    llm = engine.wrapper.model.stt_model.llm
    for layer in llm.layers:
        f = getattr(layer.mixer, "forward", None)
        if f is not None and hasattr(f, "__v2_hostpos__"):
            layer.mixer.forward = f.__v2_hostpos__
    if hasattr(llm, "_v2_hostpos_orig"):
        llm._update_mamba_mask, llm._update_causal_mask = llm._v2_hostpos_orig
        del llm._v2_hostpos_orig


experiment("mamba_host_pos")((_mamba_enable, _mamba_disable))


# ------------------------------------------------------------------------------------------------ no_timer_sync
class _CudaNoSync(types.ModuleType):
    def __init__(self, real):
        super().__init__("torch.cuda")
        self._real = real

    def synchronize(self, *a, **k):          # timer-only syncs in the wrapper: the default stream keeps the order
        return None

    def __getattr__(self, name):
        return getattr(self._real, name)


class _TorchNoSync(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = _CudaNoSync(torch.cuda)

    def __getattr__(self, name):
        return getattr(torch, name)


_WRAPPER_MOD = "nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper"


def _nosync_enable(engine) -> dict:
    mod = sys.modules[_WRAPPER_MOD]
    if not isinstance(mod.torch, _TorchNoSync):
        mod._v2_real_torch = mod.torch
        mod.torch = _TorchNoSync()
    return {"module": _WRAPPER_MOD, "note": "torch.cuda.synchronize() is a no-op inside the wrapper module only"}


def _nosync_disable(engine) -> None:
    mod = sys.modules[_WRAPPER_MOD]
    if hasattr(mod, "_v2_real_torch"):
        mod.torch = mod._v2_real_torch
        del mod._v2_real_torch


experiment("no_timer_sync")((_nosync_enable, _nosync_disable))


# perception_cache.PerceptionCacheManager.step has its own two timer-only ``torch.cuda.synchronize()`` (pinned 097dfe9,
# perception_cache.py:542 per sub-step and :557 before "Time taken for encoder"), not covered by no_timer_sync (other
# module). A device-wide sync makes the calling thread wait for ALL queued GPU work — in multi-session, the other
# sessions' work too — so perception shows host time == GPU time (round 18 B=4: 45 ms/round). Everything after it runs
# on the same stream, so dropping these two syncs keeps the order (token-neutral by construction; verified by A/B).
# Graph capture (setup, :244) happens at engine init, before this is enabled.
_PERC_MOD = "nemo.collections.speechlm2.inference.model_wrappers.perception_cache"


def _perc_nosync_enable(engine) -> dict:
    mod = sys.modules[_PERC_MOD]
    if not isinstance(mod.torch, _TorchNoSync):
        mod._v2_real_torch = mod.torch
        mod.torch = _TorchNoSync()
    return {"module": _PERC_MOD, "note": "torch.cuda.synchronize() is a no-op inside perception_cache only"}


def _perc_nosync_disable(engine) -> None:
    mod = sys.modules[_PERC_MOD]
    if hasattr(mod, "_v2_real_torch"):
        mod.torch = mod._v2_real_torch
        del mod._v2_real_torch


experiment("perception_nosync")((_perc_nosync_enable, _perc_nosync_disable))


# ------------------------------------------------------------------------------------------------ eartts_host_schedule
# EarTTS `generate_step` (ear_tts_model.py:1548-1679) blocks the host 3-4 times per refinement iteration:
#   `torch.all(eos_flag) and not ignore_eos_flag_stop` (the wrapper passes ignore_eos_flag_stop=True, but torch.all is
#   evaluated first), `torch.all(k == 0)` and `k[0].item()` twice per iteration. The unmasking schedule `ks` is a pure
#   function of (num_iter, exponent, num_quantizers): we compute it ONCE with the original device ops, keep it on the
#   host, and run the identical loop with host ints. Same kernels, same RNG calls in the same order.
def _find_eartts(engine):
    for m in engine.wrapper.model.tts_model.modules():
        if "generate_step" in type(m).__dict__ and type(m).__module__.endswith("ear_tts_model"):
            return m
    raise RuntimeError("EarTTS module with generate_step not found")


def _rng_randn_like(t):
    """torch.randn_like unless --eartts-rng session is enabled (model/eartts_rng.py: per-session counter noise)."""
    from mynah_v2.model import eartts_rng
    return eartts_rng.randn_like(t)


def _make_generate_step(mod_ns, cache: dict):
    F = mod_ns.F
    get_masking_rate, depthsum_encoding_step = mod_ns.get_masking_rate, mod_ns.depthsum_encoding_step
    TopP, TopK, fp32_precision = mod_ns.TopPLogitsWarper, mod_ns.TopKLogitsWarper, mod_ns.fp32_precision

    def schedule(self, num_iter, exponent, device):
        key = (num_iter, float(exponent), self.config.num_quantizers, str(device))
        if key not in cache:
            rates = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(-1)
            num_maskings = torch.ceil(get_masking_rate(rates, exponent=exponent) * self.config.num_quantizers).long()
            ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])
            cache[key] = [int(v) for v in ks[:, 0].tolist()]          # one-time host copy (exact integers)
        return cache[key]

    def generate_step(self, hidden_states, num_iter, guidance_scale=None, top_p_or_k=None, noise_scale=None,
                      exponent=None, eos_threshold=None, ignore_eos_flag_stop=False):
        if guidance_scale is not None:
            if not isinstance(guidance_scale, list):
                guidance_scale = [guidance_scale] * (1 + num_iter)
            assert len(guidance_scale) == 1 + num_iter
        if top_p_or_k is not None:
            if not isinstance(top_p_or_k, list):
                top_p_or_k = [top_p_or_k] * (1 + num_iter)
            assert len(top_p_or_k) == 1 + num_iter
        if noise_scale is not None:
            if not isinstance(noise_scale, list):
                noise_scale = [noise_scale] * num_iter
            assert len(noise_scale) == num_iter
        if exponent is None:
            exponent = self.config.exponent
        if guidance_scale is not None:
            hidden_states, uncond_hidden_states = hidden_states.chunk(2, dim=0)
        else:
            uncond_hidden_states = hidden_states[:0, :0, :0]
        b, t, _ = hidden_states.size()
        d = self.config.num_quantizers
        device = hidden_states.device
        if not self.config.disable_eos_prediction:
            if guidance_scale is not None:
                lm_logits = self.lm_head(hidden_states + guidance_scale[0] * (hidden_states - uncond_hidden_states))
            else:
                lm_logits = self.lm_head(hidden_states)
            if top_p_or_k is not None:
                lm_logits = (TopP(top_p_or_k[0])(None, lm_logits.view(-1, lm_logits.size(-1))).view_as(lm_logits)
                             if isinstance(top_p_or_k[0], float)
                             else TopK(top_p_or_k[0])(None, lm_logits.view(-1, lm_logits.size(-1))).view_as(lm_logits))
            with fp32_precision():
                lm_logits = F.log_softmax(lm_logits, -1)
            eos_flag = lm_logits[..., -1] > eos_threshold if eos_threshold is not None else lm_logits.argmax(-1) == 1
            if not ignore_eos_flag_stop and torch.all(eos_flag):      # host read only when it can matter
                return None, lm_logits, eos_flag
        else:
            lm_logits = None
            eos_flag = None
        code = torch.zeros((b, t, d), dtype=torch.long, device=device) + self.config.codebook_size
        cnt = 0
        for i, k in enumerate(schedule(self, num_iter, exponent, device)):
            if k == 0:
                continue
            guidance_scale_i = guidance_scale[i] if guidance_scale is not None else 0.0
            top_p_or_k_i = top_p_or_k[i] if top_p_or_k is not None else 1.0
            noise_scale_i = noise_scale[i] if noise_scale is not None else 1.0
            mog_input_embeds = self.embed_code(self.depthsum_embedding(code))
            if self.config.random_target_masking:
                mog_input_embeds += self.embed_target_mask(cnt + k - 1)
            if guidance_scale_i > 0.0:
                mog_input_embeds = torch.cat([mog_input_embeds + hidden_states, mog_input_embeds + uncond_hidden_states], 0)
            else:
                mog_input_embeds += hidden_states
            mog_mu, mog_logs = self.mog_head.infer(mog_input_embeds, guidance_scale=guidance_scale_i,
                                                   top_p_or_k=top_p_or_k_i)
            z = mog_mu + torch.exp(mog_logs) * _rng_randn_like(mog_mu) * noise_scale_i   # default: torch.randn_like
            code = getattr(self, "_v2_depthsum_step", depthsum_encoding_step)(self.rvq_embs, z, code, cnt, k)
            cnt += k
        return code, lm_logits, eos_flag
    return generate_step


def _eartts_enable(engine) -> dict:
    m = _find_eartts(engine)
    if "generate_step" not in m.__dict__:
        ns = sys.modules[type(m).__module__]
        cache: dict = {}
        m.generate_step = types.MethodType(_make_generate_step(ns, cache), m)
        m._v2_sched_cache = cache
    return {"module": type(m).__name__}


def _eartts_disable(engine) -> None:
    m = _find_eartts(engine)
    if "generate_step" in m.__dict__:
        del m.generate_step


experiment("eartts_host_schedule")((_eartts_enable, _eartts_disable))


# ------------------------------------------------------------------------------------------------ eartts_graph_genstep
# K1 (GB10): an EarTTS step is ~3500 kernel launches; the MoG refinement (`generate_step`, static shapes: CFG batch 2,
# one frame) is ~1220 of them and is host-launch-bound. Capture it once per argument signature in a CUDA graph and
# replay it. Builds on the host schedule (no host reads inside, which capture forbids). RNG: torch.randn_like inside
# the graph uses the default CUDA generator through PyTorch's graph-safe philox offsets, so replays draw the same
# numbers as eager calls in the same order (verified by the perf_ab waveform check, not assumed).
def _graph_enable(engine) -> dict:
    m = _find_eartts(engine)
    if "generate_step" not in m.__dict__:
        _eartts_enable(engine)
    eager = m.generate_step
    graphs: dict = {}
    stats = {"captures": 0, "replays": 0, "eager": 0}

    def key_of(hs, num_iter, kw):
        return (tuple(hs.shape), hs.dtype, num_iter, tuple(sorted((k, repr(v)) for k, v in kw.items())))

    def generate_step(hidden_states, num_iter, **kw):
        if kw.get("ignore_eos_flag_stop") is not True or not hidden_states.is_cuda:
            stats["eager"] += 1
            return eager(hidden_states, num_iter, **kw)
        key = key_of(hidden_states, num_iter, kw)
        g = graphs.get(key)
        if g is None:
            static_in = hidden_states.clone()
            # warm up on a side stream (required before capture); it consumes RNG like the eager call it replaces
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out_warm = eager(static_in, num_iter, **kw)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            # thread_local: another thread's CUDA calls (the main thread's codec decode / .cpu() while the FC thread captures a
            # new bucket) must not invalidate this capture or raise there (G1 crash 2026-11-11, t01 3rd call, bucket 1024).
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                static_out = eager(static_in, num_iter, **kw)
            graphs[key] = g = (graph, static_in, static_out)
            stats["captures"] += 1
            return out_warm                                # this call's result = the eager warm-up
        graph, static_in, static_out = g
        static_in.copy_(hidden_states)
        graph.replay()
        stats["replays"] += 1
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in static_out)

    m.generate_step = generate_step
    m._v2_graph_stats = stats
    m._v2_graph_eager = eager
    return {"module": type(m).__name__, "stats": stats}


def _graph_disable(engine) -> None:
    m = _find_eartts(engine)
    if hasattr(m, "_v2_graph_eager"):
        m.generate_step = m._v2_graph_eager
        del m._v2_graph_eager
    _eartts_disable(engine)


experiment("eartts_graph_genstep")((_graph_enable, _graph_disable))


# ------------------------------------------------------------------------------------------------ eartts_bf16_backbone
# "Fewer bytes": EarTTS reads ~3.1 GB of FP32 backbone weights per step (Gemma3, 778M). NeMo-Speech.cpp ships the
# EarTTS Gemma3 in BF16; NVIDIA's yaml keeps it FP32 ("bfloat16 causes hallucinations"). This experiment puts ONLY the
# backbone in BF16 (inputs cast in, hidden states cast back to FP32; MoG head, lm_head, embeddings and codec stay FP32)
# so the claim can be measured (waveform will differ: the gate is ASR + listening, not parity).
def _cast_cache(pkv, dtype):
    """Return a cache whose float tensors are in ``dtype`` WITHOUT touching ``pkv`` (the EarTTS prompt/initial state
    is shared across sessions; converting it in place made the FP32 baseline unrecoverable — J5)."""
    import copy
    need = False
    for attr in ("key_cache", "value_cache"):
        lst = getattr(pkv, attr, None)
        if isinstance(lst, list) and any(isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype
                                         for t in lst):
            need = True
    layers = getattr(pkv, "layers", None)
    if isinstance(layers, list) and any(isinstance(getattr(l, "keys", None), torch.Tensor)
                                        and getattr(l, "keys").dtype != dtype for l in layers):
        need = True
    if not need:
        return pkv
    new = copy.copy(pkv)
    for attr in ("key_cache", "value_cache"):
        lst = getattr(pkv, attr, None)
        if isinstance(lst, list):
            setattr(new, attr, [t.to(dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                                for t in lst])
    if isinstance(layers, list):
        new_layers = []
        for lay in layers:
            nl = copy.copy(lay)
            for attr in ("keys", "values"):
                t = getattr(lay, attr, None)
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    setattr(nl, attr, t.to(dtype))
            new_layers.append(nl)
        new.layers = new_layers
    return new


def _bf16_enable(engine) -> dict:
    m = _find_eartts(engine)
    bb = m.backbone
    if hasattr(bb, "_v2_fp32_state"):
        return {"already": True}
    bb._v2_fp32_state = {k: v.detach().clone() for k, v in bb.state_dict().items()}
    bb.to(torch.bfloat16)
    orig = bb.forward

    def fwd(*a, **kw):
        if kw.get("inputs_embeds") is not None:
            kw["inputs_embeds"] = kw["inputs_embeds"].to(torch.bfloat16)
        if kw.get("past_key_values") is not None:
            kw["past_key_values"] = _cast_cache(kw["past_key_values"], torch.bfloat16)
        out = orig(*a, **kw)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            out.last_hidden_state = out.last_hidden_state.float()
        return out
    fwd.__v2_bf16__ = orig
    bb.forward = fwd
    return {"backbone_params_M": round(sum(p.numel() for p in bb.parameters()) / 1e6, 1)}


def _bf16_disable(engine) -> None:
    m = _find_eartts(engine)
    bb = m.backbone
    if hasattr(bb.forward, "__v2_bf16__"):
        bb.forward = bb.forward.__v2_bf16__
    if hasattr(bb, "_v2_fp32_state"):
        bb.to(torch.float32)
        bb.load_state_dict(bb._v2_fp32_state)            # exact FP32 weights back
        del bb._v2_fp32_state


experiment("eartts_bf16_backbone")((_bf16_enable, _bf16_disable))


# ------------------------------------------------------------------------------------------------ weight-only formats
# NOT exact (the format changes the model): adoption only through the closed-loop corpus gate. The original BF16
# modules are kept on the engine so disable() restores them exactly (A/B in one process).
def _wo_swap(engine, fmt: str, selector: str, tag: str, keep: bool = True) -> dict:
    from mynah_v2.model.precision import select, thinker_linears
    from mynah_v2.model.wo_gemv import WOLinear
    stt = engine.wrapper.model.stt_model
    lins = thinker_linears(stt.llm, stt)
    chosen = select(lins, selector, len(stt.llm.layers))
    saved, before, after = [], 0, 0
    for li, bt, name, parent in lins:
        if (li, name) not in chosen:
            continue
        lin = getattr(parent, name)
        q = WOLinear(lin, fmt)
        before += lin.weight.numel() * lin.weight.element_size()
        after += q.nbytes()
        setattr(parent, name, q)
        if keep:                             # A/B mode: originals kept so disable() restores them exactly
            saved.append((parent, name, lin))
        del lin
    setattr(engine, f"_v2_{tag}_saved", saved)
    freed = None
    if not keep:
        import gc
        gc.collect()
        a0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        freed = {"allocated_GB_after": round(a0 / 2**30, 2)}
    return {"format": fmt, "selector": selector, "linears": len(chosen), "resident": not keep,
            "bf16_GB": round(before / 2**30, 2), "packed_GB": round(after / 2**30, 2), "after_release": freed}


def _wo_restore(engine, tag: str) -> None:
    for parent, name, lin in getattr(engine, f"_v2_{tag}_saved", []):
        setattr(parent, name, lin)
    setattr(engine, f"_v2_{tag}_saved", [])


experiment("thinker_w8")((lambda e: _wo_swap(e, "w8", "all", "thinker_w8"), lambda e: _wo_restore(e, "thinker_w8")))
experiment("heads_w8")((lambda e: _wo_swap(e, "w8", "heads", "heads_w8"), lambda e: _wo_restore(e, "heads_w8")))


# ------------------------------------------------------------------------------------------------ eartts_graph_full
# Whole EarTTS step in one CUDA graph per KV bucket (mynah_v2.model.eartts_graph). Replaces eartts_graph_genstep (the
# MoG-only graph cannot nest inside it). NOT exact by construction (masked padded attention window): measured.
def _full_enable(engine, lmax: int = 4096, slots: int = 2) -> dict:
    from mynah_v2.model.eartts_graph import FullStepGraph
    m = _find_eartts(engine)
    if hasattr(m, "_v2_graph_eager"):
        raise RuntimeError("disable eartts_graph_genstep before eartts_graph_full")
    if "generate_step" not in m.__dict__:
        _eartts_enable(engine)                       # host schedule: no device reads inside the MoG refinement
    duplex = engine.wrapper.model.tts_model
    fsg = FullStepGraph(duplex, lmax=lmax, slots=slots)
    duplex.infer_codes_one_step = fsg.step
    engine._v2_full_graph = fsg
    return {"lmax": fsg.lmax, "kv_slot_GB": round(2 * fsg.shape[0] * 2 * fsg.shape[1] * fsg.lmax * fsg.shape[2] * 4 / 2**30, 2),
            "stats": fsg.stats}


def _full_disable(engine) -> None:
    fsg = getattr(engine, "_v2_full_graph", None)
    if fsg is not None:
        engine.wrapper.model.tts_model.infer_codes_one_step = fsg.eager_step
        fsg.remove()
        del engine._v2_full_graph
    _eartts_disable(engine)


experiment("eartts_graph_full")((_full_enable, _full_disable))
# NeMo non-blocking FC hand-back fix (PAD id + FC-written tokens copied back into the context): see fc_ctx_fix.py
def _fcfix_enable(engine) -> dict:
    from mynah_v2.model import fc_ctx_fix
    return fc_ctx_fix.enable(engine)


def _fcfix_disable(engine) -> None:
    from mynah_v2.model import fc_ctx_fix
    fc_ctx_fix.disable(engine)


experiment("fc_ctx_fix")((_fcfix_enable, _fcfix_disable))
# diagnostic variants (which part of the hand-back changes the resume step): pad fill only / one channel copied
# fc_ctx_fix_sel = the candidate for the OFFICIAL checkpoint: real PAD + text/ASR copied back, function channel PAD over
# the FC span (EOTR never fed back at resume; resume reads (PAD, PAD)). Not for other checkpoints (e.g. the lab's D-V2.1).
_FCFIX_VARIANTS = {"fc_ctx_fix_pad_only": (True, (), False),
                   "fc_ctx_fix_func_only": (False, ("gen_function_text",), False),
                   "fc_ctx_fix_text_only": (False, ("gen_text", "gen_asr_text"), False),
                   "fc_ctx_fix_sel": (True, ("gen_text", "gen_asr_text"), True)}
for _n, (_pf, _ch, _fp) in _FCFIX_VARIANTS.items():
    experiment(_n)(((lambda pf, ch, fp: lambda e: __import__("mynah_v2.model.fc_ctx_fix", fromlist=["x"]).enable(
        e, pad_fill=pf, channels=ch, func_pad=fp))(_pf, _ch, _fp), _fcfix_disable))


def _fcprof_enable(engine) -> dict:
    from mynah_v2.model import fc_profile
    return fc_profile.enable(engine)


def _fcprof_disable(engine) -> None:
    from mynah_v2.model import fc_profile
    fc_profile.disable(engine)


experiment("fc_profile")((_fcprof_enable, _fcprof_disable))       # diagnostic, token-neutral (CUDA events, FC thread)
# multi-session: 2 slots per session (main loop + FC fast path); same graphs/numerics, only the slot pool is larger
experiment("eartts_graph_full_s4")((lambda e: _full_enable(e, slots=4), _full_disable))
PROFILES["optimized-graph"] = ("mamba_host_pos", "no_timer_sync", "eartts_graph_full", "eartts_fused_rmsnorm")
PROFILES["optimized-graph-s4"] = ("mamba_host_pos", "no_timer_sync", "eartts_graph_full_s4", "eartts_fused_rmsnorm")
experiment("eartts_graph_full_s8")((lambda e: _full_enable(e, slots=8), _full_disable))
PROFILES["optimized-graph-s8"] = ("mamba_host_pos", "no_timer_sync", "eartts_graph_full_s8", "eartts_fused_rmsnorm")


# ------------------------------------------------------------------------------------------------ thinker_graph
# Thinker decode step (backbone + heads) as one CUDA graph per attention-KV bucket (mynah_v2.model.thinker_graph).
# Needs mamba_host_pos (enable it first). NOT exact by construction (masked padded attention window): measured.
def _tg_enable(engine, lmax: int = 16384) -> dict:
    from mynah_v2.model.thinker_graph import ThinkerGraph
    tg = ThinkerGraph(engine, lmax=lmax)
    engine._v2_thinker_graph = tg
    return {"lmax": lmax, "attention_layers": tg.attn, "stats": tg.stats}


def _tg_disable(engine) -> None:
    tg = getattr(engine, "_v2_thinker_graph", None)
    if tg is not None:
        tg.remove()
        del engine._v2_thinker_graph


experiment("thinker_graph")((_tg_enable, _tg_disable))


# ------------------------------------------------------------------------------------------------ perception_graph
# Full-buffer perception (exact path, no perception cache) as one CUDA graph for the full 71-frame window
# (mynah_v2.model.perception_graph). Same module, same input, same kernels → expected bit-identical: measured, not assumed.
def _pg_enable(engine) -> dict:
    from mynah_v2.model.perception_graph import DEFAULT_BUFFER_FRAMES, FRAME_SAMPLES, PerceptionGraph
    w = engine.wrapper
    if getattr(w, "use_perception_cache", False):
        raise RuntimeError("perception_graph covers the full-buffer path only (perception cache is on)")
    frames = int(w.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_FRAMES))
    pg = PerceptionGraph(w.model.stt_model.perception, frames * FRAME_SAMPLES)
    engine._v2_perception_graph = pg
    return {"full_len": pg.full_len, "stats": pg.stats}


def _pg_disable(engine) -> None:
    pg = getattr(engine, "_v2_perception_graph", None)
    if pg is not None:
        pg.remove()
        del engine._v2_perception_graph


experiment("perception_graph")((_pg_enable, _pg_disable))


# ------------------------------------------------------------------------------------------------ codec_nosync / codec_fc_d2h
# Codec-path host syncs (mynah_v2.model.codec_async): ``codec_nosync`` caches the ISTFT window envelope so the per-decode
# ``assert (window_envelope > 1e-11).all()`` (ear_tts_vae_codec.py:248, the device-wide sync behind codec host == GPU
# time) runs once per frame count; same ops, same tensor -> bit-identical by construction (verified on GPU anyway).
# ``codec_fc_d2h``: FC-thread decode output ``.cpu()`` -> pinned async copy + event, waited on first data read.
def fuse_rmsnorms(root, bf16_native: bool = False, types=("Gemma3RMSNorm",)) -> int:
    """Gemma3RMSNorm.forward = (x.float() * rsqrt(mean(x^2) + eps)) * (1 + w.float()), cast back to x's dtype:
    ~7 elementwise/reduce kernels per call. Replace it (instance attribute) with ONE torch rms_norm over x.float() with
    the precomputed FP32 weight (1 + w): same formula and precision, fewer kernels (reduction order may differ ~1 ulp).
    Default (exact): every dtype goes through the FP32 math (cast in, fused norm with FP32 (1 + w), cast out) = the
    reference formula; for BF16 inputs that is 3 kernels instead of ~9. bf16_native=True: rms_norm directly on BF16
    with a BF16 (1 + w) (FP32 accumulation inside, no casts) — faster, but rounds (1 + w) and the normalized value
    separately (double rounding, ≤ 2 BF16 ulp).
    The precomputed weight is taken NOW (in its FP32 value): apply before any in-place dtype change of the norms."""
    n = 0
    for mod in root.modules():
        if type(mod).__name__ not in types or "_v2_w1" in mod.__dict__:
            continue
        prev = mod.__dict__.get("forward")                   # an existing instance override (profiler label wrapper)
        label = getattr(prev, "_v2_label", None)
        if prev is not None and label is None:
            continue                                         # unknown override: leave it alone
        w1 = 1.0 + mod.weight.detach().float()
        w1_bf16 = w1.to(torch.bfloat16)                     # --eartts-dtype bf16: native BF16 path, no cast kernels
        mod._v2_w1 = w1

        def fwd(x, _w1=w1, _w1b=w1_bf16, _eps=float(mod.eps), _d=(int(mod.weight.shape[0]),), _native=bf16_native):
            if _native and x.dtype == torch.bfloat16:      # rms_norm accumulates in FP32 internally; output BF16
                return torch.nn.functional.rms_norm(x, _d, _w1b, _eps)
            return torch.nn.functional.rms_norm(x.float(), _d, _w1, _eps).type_as(x)
        if label is not None:
            fused = fwd

            def fwd(x, _f=fused, _lab=label):
                with torch.profiler.record_function(_lab):
                    return _f(x)
            fwd._v2_label = label
        mod._v2_prev_forward = prev
        mod.forward = fwd
        n += 1
    return n


def unfuse_rmsnorms(root, types=("Gemma3RMSNorm",)) -> None:
    for mod in root.modules():
        if type(mod).__name__ in types and "_v2_w1" in mod.__dict__:
            prev = mod.__dict__.pop("_v2_prev_forward", None)
            if prev is not None:
                mod.forward = prev
            else:
                del mod.forward
            mod.__dict__.pop("_v2_w1", None)


def _ernorm_enable(engine) -> dict:
    import os
    if os.environ.get("V2_FUSED_RMSNORM", "1") == "0":   # fallback: keep the original Gemma3RMSNorm forward
        return {"norms": 0, "disabled_by": "V2_FUSED_RMSNORM=0"}
    return {"norms": fuse_rmsnorms(_find_eartts(engine).backbone), "fused_kernel": hasattr(torch, "_fused_rms_norm")}


def _ernorm_disable(engine) -> None:
    unfuse_rmsnorms(_find_eartts(engine).backbone)


experiment("eartts_fused_rmsnorm")((_ernorm_enable, _ernorm_disable))


# NeMo's EarTTS MoG head uses its own RMSNorm class with the same formula (x.float() * rsqrt(mean(x^2)+eps) * (1+w),
# cast back): 3 MLP layers + 1 final norm, run at each of the 8 unmasking iterations of every EarTTS step.
def _mognorm_enable(engine) -> dict:
    return {"norms": fuse_rmsnorms(_find_eartts(engine).mog_head, types=("RMSNorm",))}


def _mognorm_disable(engine) -> None:
    unfuse_rmsnorms(_find_eartts(engine).mog_head, types=("RMSNorm",))


experiment("eartts_fused_rmsnorm_mog")((_mognorm_enable, _mognorm_disable))


# --- eartts_rvq_cache: hoist constant RVQ-table work out of the per-iteration MoG loop (bit-exact) -------------------
# Upstream NeMo ear_tts_model.py recomputes, at every one of the 8 unmasking iterations of every EarTTS step:
#   depthsum_embedding:      F.pad(self.rvq_embs, [0, 0, 0, 1])  -> a full copy of the [31, 1024, 512] FP32 table
#   depthsum_encoding_step:  embs[i].pow(2).sum(-1) for each quantizer i -> constant per-code squared norms
# Both depend only on the weights. Computing them once with the same ops gives the same values, so the per-step
# arithmetic (and its order) is unchanged. Recomputed if rvq_embs is replaced (set_rvq_embs) via a version key.

def _rvq_key(m):
    e = m.rvq_embs
    return (e.data_ptr(), e._version, tuple(e.shape), e.dtype)


def _rvq_consts(m):
    key = _rvq_key(m)
    c = m.__dict__.get("_v2_rvq_consts")
    if c is None or c[0] != key:
        with torch.no_grad():
            padded = torch.nn.functional.pad(m.rvq_embs, [0, 0, 0, 1])
            sq = [m.rvq_embs[i].pow(2).sum(-1) for i in range(m.rvq_embs.shape[0])]
        c = (key, padded, sq)
        m.__dict__["_v2_rvq_consts"] = c
    return c


def _depthsum_embedding_cached(self, code):
    F = torch.nn.functional
    b, t, d = code.size()
    _, v, h = self.rvq_embs.size()
    ret = torch.zeros((b, t, h), device=code.device)
    embs = _rvq_consts(self)[1]
    for i in range(d):
        ret = ret + F.embedding(code[..., i], embs[i])
    return ret


def _make_depthsum_step_cached(m):
    F = torch.nn.functional

    def step(embs, r, code, depth_str=0, k=72):
        sq = _rvq_consts(m)[2]
        for i in range(depth_str, depth_str + k):
            idx_sel = (sq[i] - 2 * (r.unsqueeze(-2) @ embs[i].transpose(-1, -2)).squeeze(-2)).argmin(-1)
            emb_i = F.embedding(idx_sel, embs[i])
            r = r - emb_i
            code[..., i] = idx_sel
        return code
    return step


def _rvq_enable(engine) -> dict:
    m = _find_eartts(engine)
    _rvq_consts(m)
    m.depthsum_embedding = types.MethodType(_depthsum_embedding_cached, m)
    m._v2_depthsum_step = _make_depthsum_step_cached(m)
    e = m.rvq_embs
    return {"rvq_embs": list(e.shape), "dtype": str(e.dtype), "padded_MB": round(e.numel() * e.element_size() / 2**20, 1)}


def _rvq_disable(engine) -> None:
    m = _find_eartts(engine)
    for k in ("depthsum_embedding", "_v2_depthsum_step", "_v2_rvq_consts"):
        m.__dict__.pop(k, None)


experiment("eartts_rvq_cache")((_rvq_enable, _rvq_disable))


# ``eartts_rvq_q4`` only replaces the hot depth-sum gather. The residual
# argmin encoder still consumes the original BF16 table, because changing its
# distances can change later code selections; a fused quantized encoder is a
# separate correctness gate.
def _rvq_q4_enable(engine) -> dict:
    from mynah_v2.model.rvq_q4 import RVQInt4Embedding, RVQInt4Encoder

    m = _find_eartts(engine)
    if hasattr(m, "_v2_rvq_q4"):
        return {"rvq_q4": "already_enabled"}
    q = RVQInt4Embedding(m.rvq_embs)
    original = m.__dict__.get("depthsum_embedding")
    use_encoder = os.environ.get("MYNAH_INT4_RVQ_ENCODE", "0") == "1"

    def depthsum(self, code):
        return self._v2_rvq_q4(code)

    m._v2_rvq_q4 = q
    m._v2_rvq_q4_original = original
    encoder = None
    if use_encoder:
        original_step = m.__dict__.get("_v2_depthsum_step")
        m._v2_rvq_q4_original_step = original_step
        encoder = RVQInt4Encoder(q)
        m._v2_rvq_q4_encoder = encoder

        def encode_step(self, embs, residual, code, depth_str=0, k=72):
            return self._v2_rvq_q4_encoder(residual, code, depth_str, k)

        m._v2_depthsum_step = types.MethodType(encode_step, m)
    m.depthsum_embedding = types.MethodType(depthsum, m)
    torch.cuda.empty_cache()
    return {"rvq_q4": True, "shape": list(m.rvq_embs.shape), "bytes": q.nbytes(),
            "encode_table": "Q4_fused" if encoder is not None else "BF16",
            "encoder_bytes": encoder.nbytes() if encoder is not None else 0}


def _rvq_q4_disable(engine) -> None:
    m = _find_eartts(engine)
    original = m.__dict__.pop("_v2_rvq_q4_original", None)
    original_step = m.__dict__.pop("_v2_rvq_q4_original_step", None)
    # These are assigned nn.Modules and therefore live in ``_modules``;
    # removing only from __dict__ leaves a stale registered child and makes a
    # subsequent enable/disable cycle retain the old GPU allocation.
    m._modules.pop("_v2_rvq_q4", None)
    m._modules.pop("_v2_rvq_q4_encoder", None)
    if original is not None:
        m.depthsum_embedding = original
    else:
        m.__dict__.pop("depthsum_embedding", None)
    if original_step is not None:
        m._v2_depthsum_step = original_step
    else:
        m.__dict__.pop("_v2_depthsum_step", None)


experiment("eartts_rvq_q4")((_rvq_q4_enable, _rvq_q4_disable))


# --- codec_depthwise_q4: fused small-K depthwise convolution -----------------
# The causal cache supplies the left context, so ConvNeXt's depthwise layer
# itself is a plain [B,C,T] -> [B,C,T-K+1] convolution.  These filters are too
# narrow for tinygemm; the dedicated kernel dequantizes each nibble in the
# K-tap dot product.  Keep this opt-in until waveform parity/RTF is measured on
# the target GPU: it is a bandwidth reduction, not a promise of a faster
# launch for every codec chunk size.
def _codec_depthwise_q4_enable(engine) -> dict:
    from mynah_v2.model.depthwise_q4 import DepthwiseConv1dQ4

    codec = engine.wrapper.model.tts_model.audio_codec
    if hasattr(codec, "_v2_depthwise_q4_originals"):
        return {"codec_depthwise_q4": "already_enabled"}
    originals = []
    replaced = 0
    bytes_q4 = 0
    for path, block in list(codec.named_modules()):
        dw = getattr(block, "dwconv", None)
        if not isinstance(dw, torch.nn.Conv1d):
            continue
        if not (dw.in_channels == dw.out_channels == dw.groups and dw.kernel_size[0] <= 15
                and dw.stride == (1,) and dw.dilation == (1,) and dw.padding == (0,)):
            continue
        q4 = DepthwiseConv1dQ4(dw)
        parent_path, child = path.rsplit(".", 1) if "." in path else ("", path)
        parent = codec.get_submodule(parent_path) if parent_path else codec
        originals.append((parent, child, dw))
        setattr(parent, child, q4)
        replaced += 1
        bytes_q4 += q4.nbytes() + (dw.bias.numel() * dw.bias.element_size() if dw.bias is not None else 0)
    if not replaced:
        raise RuntimeError("no supported codec ConvNeXt depthwise convolutions found")
    codec._v2_depthwise_q4_originals = originals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"codec_depthwise_q4": True, "layers": replaced, "q4_bytes": bytes_q4}


def _codec_depthwise_q4_disable(engine) -> None:
    codec = engine.wrapper.model.tts_model.audio_codec
    originals = getattr(codec, "_v2_depthwise_q4_originals", None)
    if originals is None:
        return
    for parent, child, original in originals:
        setattr(parent, child, original)
    del codec._v2_depthwise_q4_originals


experiment("codec_depthwise_q4")((_codec_depthwise_q4_enable, _codec_depthwise_q4_disable))


# --- rnnt_decoder_q4: fused inference-only RNNT prediction LSTM -------------
def _find_rnnt_decoder(engine):
    stt = engine.wrapper.model.stt_model
    for name in ("rnnt_decoder", "_rnnt_decoder"):
        decoder = getattr(stt, name, None)
        if decoder is not None and hasattr(decoder, "prediction"):
            return decoder
    for decoder in stt.modules():
        if type(decoder).__name__ == "RNNTDecoder" and hasattr(decoder, "prediction"):
            return decoder
    raise RuntimeError("RNNTDecoder module not found")


def _rnnt_q4_enable(engine) -> dict:
    from mynah_v2.model.rnnt_q4 import Int4LSTM

    decoder = _find_rnnt_decoder(engine)
    rnn = decoder.prediction["dec_rnn"]
    if isinstance(rnn, Int4LSTM):
        return {"rnnt_decoder_q4": "already_enabled"}
    q4 = Int4LSTM(rnn)
    decoder.prediction["dec_rnn"] = q4
    decoder._v2_rnnt_q4_original = rnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rnnt_decoder_q4": True, "layers": q4.num_layers, "bytes": q4.nbytes(),
            "fallback": "BF16 LSTM when disabled"}


def _rnnt_q4_disable(engine) -> None:
    decoder = _find_rnnt_decoder(engine)
    original = getattr(decoder, "_v2_rnnt_q4_original", None)
    if original is not None:
        decoder.prediction["dec_rnn"] = original
        del decoder._v2_rnnt_q4_original


experiment("rnnt_decoder_q4")((_rnnt_q4_enable, _rnnt_q4_disable))
experiment("eartts_fused_rmsnorm_bf16native")((lambda e: {"norms": fuse_rmsnorms(_find_eartts(e).backbone, bf16_native=True)},
                                               _ernorm_disable))


# --- eartts_mog_q4: selected-mixture INT4 projection -----------------------------------------------
# ``MoGHead.infer`` indexes proj_mus.weight directly instead of calling the
# Linear module. A normal Linear replacement would therefore be dead weight;
# the optional kernel preserves the selected-matrix contract and dequantizes
# nibbles inside the selected GEMV. It is kept out of the default exact profile
# until the GB10 round-trip confirms both quality and latency.
def _mog_q4_enable(engine) -> dict:
    from mynah_v2.model.mog_q4 import SelectedMoGQ4

    m = _find_eartts(engine)
    head = m.mog_head
    if not hasattr(head, "proj_mus") or not hasattr(head, "num_predictions"):
        raise RuntimeError("EarTTS MoG head not found")
    if isinstance(head.proj_mus, SelectedMoGQ4):
        return {"mog_q4": "already_enabled"}
    if not isinstance(head.proj_mus, torch.nn.Linear):
        raise RuntimeError("MoG proj_mus is not an ordinary Linear")
    original = head.proj_mus
    proj_state = (original.in_features, original.out_features, original.weight.detach().cpu(),
                  original.bias.detach().cpu() if original.bias is not None else None,
                  original.weight.device, original.weight.dtype)
    head.proj_mus = SelectedMoGQ4(original, head.num_predictions, head.low_rank or head.out_size)
    low_q = None
    low_state = None
    use_low_q4 = os.environ.get("MYNAH_INT4_MOG_LOW", "0") == "1"
    if getattr(head, "low_rank", None) and use_low_q4:
        low = head.low_mat
        low_state = (low.detach().cpu(), low.device, low.dtype)
        low_q = SelectedMoGQ4.from_tensor(low.detach(), group=128)
        # The serving infer path below no longer reads the BF16 parameter. Keep
        # its A/B copy on host memory so the quantized experiment really frees
        # the large GPU matrix instead of retaining it through a closure.
        head.low_mat = torch.nn.Parameter(torch.empty(0, device=low.device, dtype=low.dtype), requires_grad=False)
    old_infer = head.infer
    ns = type(head).infer.__globals__
    F = ns["F"]
    TopP, TopK = ns["TopPLogitsWarper"], ns["TopKLogitsWarper"]
    gumbel_like = ns["gumbel_like"]
    batch_matmul = ns["batch_matmul"]

    def infer(self, x, guidance_scale: float = 0.0, top_p_or_k: float | int = 1.0):
        b, t, _ = x.size()
        n, d = self.num_predictions, self.low_rank or self.out_size
        x = self.mlp_stack(x)
        if guidance_scale > 0:
            b //= 2
            x_cond, x_uncond = x.chunk(2, dim=0)
            x = x_cond + guidance_scale * (x_cond - x_uncond)
        logits = self.proj_logits(x)
        if top_p_or_k is not None:
            logits = (TopP(top_p_or_k)(None, logits.view(-1, n)).view_as(logits)
                      if isinstance(top_p_or_k, float)
                      else TopK(top_p_or_k)(None, logits.view(-1, n)).view_as(logits))
        mixture_indices = (F.log_softmax(logits, dim=-1) + gumbel_like(logits)).argmax(-1)
        mu = self.proj_mus.selected(x.view(b * t, -1), mixture_indices.view(b * t)).view(b, t, d)
        if self.low_rank:
            if self._v2_mog_q4_low is not None:
                mu = self._v2_mog_q4_low.selected(mu.view(b * t, -1), mixture_indices.view(b * t)).view(
                    b, t, self.out_size
                )
            else:
                mu = batch_matmul(
                    mu.view(b * t, -1), self.low_mat.detach().view(n, self.out_size, -1),
                    mixture_indices.view(b * t), BLOCK_SIZE_DIN=d, BLOCK_SIZE_DOUT=self.out_size,
                ).view(b, t, self.out_size)
            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((b, t, d), device=x.device)
        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mu * torch.exp(logs) + mu_res, logs

    # Keep the bound method for reversible A/B runs. ``old_infer`` is retained
    # as an object attribute so disabling does not depend on class mutation.
    head.__dict__["_v2_mog_q4_original"] = old_infer
    head.__dict__["_v2_mog_q4_original_proj_state"] = proj_state
    head.__dict__["_v2_mog_q4_original_low_state"] = low_state
    head._v2_mog_q4_low = low_q
    head.infer = types.MethodType(infer, head)
    torch.cuda.empty_cache()
    return {"mog_q4": True, "group": head.proj_mus.group, "bytes": head.proj_mus.nbytes() + (low_q.nbytes() if low_q else 0),
            "num_predictions": head.num_predictions, "out_size": head.low_rank or head.out_size,
            "low_mat_q4": use_low_q4}


def _mog_q4_disable(engine) -> None:
    m = _find_eartts(engine)
    head = m.mog_head
    old = head.__dict__.pop("_v2_mog_q4_original", None)
    proj_state = head.__dict__.pop("_v2_mog_q4_original_proj_state", None)
    low_state = head.__dict__.pop("_v2_mog_q4_original_low_state", None)
    head.__dict__.pop("_v2_mog_q4_low", None)
    if proj_state is not None:
        in_features, out_features, weight, bias, device, dtype = proj_state
        lin = torch.nn.Linear(in_features, out_features, bias=bias is not None, device=device, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(weight.to(device=device, dtype=dtype))
            if bias is not None:
                lin.bias.copy_(bias.to(device=device, dtype=dtype))
        head.proj_mus = lin
    if low_state is not None:
        weight, device, dtype = low_state
        head.low_mat = torch.nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
    if old is not None:
        head.infer = old


experiment("eartts_mog_q4")((_mog_q4_enable, _mog_q4_disable))


def _codec_nosync_enable(engine) -> dict:
    from mynah_v2.model.codec_async import nosync_enable
    return nosync_enable(engine)


def _codec_nosync_disable(engine) -> None:
    from mynah_v2.model.codec_async import nosync_disable
    nosync_disable(engine)


def _codec_fcd2h_enable(engine) -> dict:
    from mynah_v2.model.codec_async import fc_d2h_enable
    return fc_d2h_enable(engine)


def _codec_fcd2h_disable(engine) -> None:
    from mynah_v2.model.codec_async import fc_d2h_disable
    fc_d2h_disable(engine)


experiment("codec_nosync")((_codec_nosync_enable, _codec_nosync_disable))
experiment("codec_fc_d2h")((_codec_fcd2h_enable, _codec_fcd2h_disable))


# ``codec_fc_2codes``: the FC thread decodes its codes in pairs (2 per codec call, like the main path) instead of one
# per EarTTS step; every code decoded once and in order (streaming cache aligned), a pending code decoded when the FC
# phase returns. Audio-only, ~1e-6 vs 1+1 (not bit-identical). codec_async.make_fc2_decode / make_fc2_steps.
def _codec_fc2_enable(engine) -> dict:
    from mynah_v2.model.codec_async import fc2_enable
    return fc2_enable(engine)


def _codec_fc2_disable(engine) -> None:
    from mynah_v2.model.codec_async import fc2_disable
    fc2_disable(engine)


experiment("codec_fc_2codes")((_codec_fc2_enable, _codec_fc2_disable))
