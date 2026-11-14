"""Incremental (cached) Thinker for the NVIDIA native engine — V2 extension, OFF by default.

Why: the reference native path disables the LLM cache for Nemotron
(``inference/streaming/state/s2s_context_manager.py:65-76``: "Nemotron requires NemotronHHybridDynamicCache which is
not yet supported"), so every 80 ms step re-runs the Thinker over the whole history (system prompt + all frames).
Measured on GB10 (WebSocket demo, 2.7 min session): ``stt_model`` 237 -> 488 ms per step, growing with the session.
NVIDIA's production engine is vLLM; the paper's latency numbers are not reachable on the native path as shipped.

What is missing upstream is small, and everything else is already wired for "native cache mode":
  * the context manager must create a NemotronH ``HybridMambaAttentionDynamicCache`` (attention KV + Mamba conv/SSM
    state) instead of ``None``;
  * ``DuplexSTTModel.forward`` passes the cache as ``cfg.cache_key`` (default ``past_key_values``; NemotronH wants
    ``cache_params``) and reads it back from ``out.cache_params``;
  * NemotronHBlock never passes the cache to its attention mixer, so attention must get it explicitly;
  * NemotronH needs ``cache_position``: without it every call is treated as position 0 (the Mamba mixers re-initialise
    their conv state). We count positions on the cache object.
The pipeline's prefill (``_prefill_system_prompt`` native-cache branch), per-step call and FC fast path already
handle a non-None ``context.dynamic_cache``.

No upstream file is edited: the cache_key is set on the live config, ``llm.forward`` is wrapped on the instance and
the context-manager class gets a process-local hook. Correctness gate: ``scripts/cache_parity.py`` (tokens with vs
without cache on the same input, deterministic decoding).
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import torch


def _hybrid_cache_cls(llm) -> type:
    """NemotronH's ``HybridMambaAttentionDynamicCache`` (remote code, Nano-9B-v2 snapshot 6533e8de) is not usable as
    shipped for incremental inference — which is presumably why the NVIDIA native path disables it:
      * ``conv_kernel_size`` is read by the Mamba mixer (prefill padding) but never set;
      * ``update_conv_state`` / ``update_ssm_state`` were copied from Mamba2 and call ``.device`` on what are Python
        lists here (per-layer tensors).
    This subclass fixes exactly those three points; everything else is the original class."""
    mod = sys.modules[type(llm).__module__]
    base = getattr(mod, "HybridMambaAttentionDynamicCache", None)
    if base is None:
        raise RuntimeError(f"{type(llm).__module__} has no HybridMambaAttentionDynamicCache")

    class V2HybridMambaAttentionCache(base):
        def __init__(self, config, batch_size, dtype=torch.float16, device=None, ssm_dtype=None):
            super().__init__(config, batch_size, dtype=dtype, device=device)
            self.conv_kernel_size = config.conv_kernel
            # SSM recurrent state precision (vLLM exposes the same knob, mamba_ssm_cache_dtype, for Nemotron-H)
            self.ssm_dtype = ssm_dtype
            if ssm_dtype is not None:
                self.ssm_states = [t.to(ssm_dtype) if t.numel() else t for t in self.ssm_states]

        def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False):
            cur = self.conv_states[layer_idx]
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(device=cur.device, dtype=cur.dtype)
            else:
                cur = cur.roll(shifts=-1, dims=-1)
                cur[:, :, -1] = new_conv_state[:, 0, :].to(device=cur.device, dtype=cur.dtype)
                self.conv_states[layer_idx] = cur
            return self.conv_states[layer_idx]

        def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
            t = new_ssm_state.to(self.ssm_states[layer_idx].device)
            if self.ssm_dtype is not None:
                t = t.to(self.ssm_dtype)
            self.ssm_states[layer_idx] = t
            return self.ssm_states[layer_idx]

    return V2HybridMambaAttentionCache


def cache_passing_attention(orig: Callable[..., Any], holder) -> Callable[..., Any]:
    """NemotronHBlock calls its attention mixer as ``mixer(hidden_states, cache_position=...)`` — **without**
    ``past_key_value`` (modeling_nemotron_h.py, NemotronHBlock.forward), so with a cache every decode step would
    attend to itself only. Pass the live cache (set on ``holder`` by the backbone wrapper) explicitly."""

    def forward(hidden_states, *args, **kw):
        c = getattr(holder, "_v2_active_cache", None)
        if c is not None and kw.get("past_key_value") is None:
            kw["past_key_value"] = c
        return orig(hidden_states, *args, **kw)

    forward.__wrapped_v2__ = orig  # type: ignore[attr-defined]
    return forward


def position_injecting_forward(orig: Callable[..., Any], holder=None) -> Callable[..., Any]:
    """Wrap a NemotronH backbone ``forward``: when a hybrid cache is passed without ``cache_position``, continue from
    the cached length (and expose the cache to the attention wrappers via ``holder``). Decode steps after the prefill
    must be single-frame (the Mamba decode kernel is 1 token)."""

    def forward(*args, **kw):
        cache = kw.get("cache_params")
        if cache is not None and kw.get("cache_position") is None:
            emb = kw.get("inputs_embeds")
            if emb is None:
                raise ValueError("cached NemotronH call needs inputs_embeds")
            t = int(emb.shape[1])
            # Own counter: NemotronH get_seq_length() reports 1 on a fresh cache (empty tensors are (B, 0), it reads
            # shape[-2]); HF generate() always passes cache_position, so upstream never hits it.
            pos = int(getattr(cache, "_v2_pos", 0))
            if pos > 0 and t != 1:
                if getattr(cache, "_v2_continuation", False) and holder is not None:
                    # EXPERIMENT fc_phase2_prefill: T known positions consumed in one call (same final state as T
                    # decode steps; see model/fc_phase2_prefill.py). Only when the caller marked the cache for it.
                    from mynah_v2.model.fc_phase2_prefill import continuation_forward
                    import sys as _sys
                    h = continuation_forward(holder, emb, cache, chunk=getattr(cache, "_v2_continuation_chunk", 16))
                    out_cls = getattr(_sys.modules[type(holder).__module__], "NemotronHOutput", None)
                    return out_cls(last_hidden_state=h, cache_params=cache) if out_cls is not None else \
                        {"last_hidden_state": h, "cache_params": cache}
                raise ValueError(f"cached decode expects 1 frame per call after the prefill, got {t} at position {pos}")
            kw["cache_position"] = torch.arange(pos, pos + t, device=emb.device)
            if holder is not None:
                holder._v2_active_cache = cache
                holder._v2_host_pos = (pos, t)   # host-side position (perf experiments avoid device syncs with it)
            try:
                out = orig(*args, **kw)
            finally:
                if holder is not None:
                    holder._v2_active_cache = None
                    holder._v2_host_pos = None
            cache._v2_pos = pos + t
            return out
        return orig(*args, **kw)

    forward.__wrapped_v2__ = orig  # type: ignore[attr-defined]
    return forward


def enable_hybrid_llm_cache(engine, ssm_dtype: torch.dtype | None = torch.float32) -> dict:
    """Turn on incremental Thinker inference for ``ReferenceEngine`` ``engine`` (call before ``open``).
    Returns a small description for traces/summaries."""
    from omegaconf import open_dict

    from nemo.collections.speechlm2.inference.streaming.state import s2s_context_manager as scm

    stt = engine.wrapper.model.stt_model
    llm = stt.llm
    cls = _hybrid_cache_cls(llm)
    p = next(llm.parameters())
    dtype, device = p.dtype, p.device

    with open_dict(stt.cfg):
        stt.cfg.cache_key = "cache_params"
    if not hasattr(llm.forward, "__wrapped_v2__"):
        llm.forward = position_injecting_forward(llm.forward, holder=llm)
    n_attn = 0
    for layer in getattr(llm, "layers", []):
        if getattr(layer, "block_type", None) == "attention":
            n_attn += 1
            if not hasattr(layer.mixer.forward, "__wrapped_v2__"):
                layer.mixer.forward = cache_passing_attention(layer.mixer.forward, llm)
    if n_attn == 0:
        raise RuntimeError("no attention blocks found on the NemotronH backbone (layout changed?)")

    def factory():
        return cls(llm.config, 1, dtype=dtype, device=device, ssm_dtype=ssm_dtype)

    mgr_cls = scm.S2SContextManager
    if not getattr(mgr_cls, "_v2_hooked", False):
        orig_init, orig_create = mgr_cls.__init__, mgr_cls._create_context

        def __init__(self, *a, **kw):
            orig_init(self, *a, **kw)
            if mgr_cls._v2_cache_factory is not None:
                self.cache_disabled = False

        def _create_context(self):
            ctx = orig_create(self)
            if mgr_cls._v2_cache_factory is not None:
                ctx.dynamic_cache = mgr_cls._v2_cache_factory()
            return ctx

        mgr_cls.__init__, mgr_cls._create_context = __init__, _create_context
        mgr_cls._v2_hooked = True
    mgr_cls._v2_cache_factory = factory
    # the pipeline may already hold a context manager built before the hook
    cm = getattr(engine.pipeline, "context_manager", None)
    if cm is not None:
        cm.cache_disabled = False
    return {"llm_cache": cls.__name__, "dtype": str(dtype), "ssm_dtype": str(ssm_dtype), "cache_key": "cache_params",
            "attention_blocks_patched": n_attn}


def disable_hybrid_llm_cache(engine) -> None:
    """Back to the reference behaviour (no cache, full-history recompute) for the next session."""
    from nemo.collections.speechlm2.inference.streaming.state import s2s_context_manager as scm
    from omegaconf import open_dict

    scm.S2SContextManager._v2_cache_factory = None
    stt = engine.wrapper.model.stt_model
    llm = stt.llm
    if hasattr(llm.forward, "__wrapped_v2__"):
        llm.forward = llm.forward.__wrapped_v2__
    for layer in getattr(llm, "layers", []):
        if hasattr(getattr(layer, "mixer", None), "forward") and hasattr(layer.mixer.forward, "__wrapped_v2__"):
            layer.mixer.forward = layer.mixer.forward.__wrapped_v2__
    with open_dict(stt.cfg):
        stt.cfg.pop("cache_key", None)
    cm = getattr(engine.pipeline, "context_manager", None)
    if cm is not None:
        cm.cache_disabled = True
