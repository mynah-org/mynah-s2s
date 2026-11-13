"""EXPERIMENTAL: the Thinker decode step (NemotronH backbone + text/function/ASR heads) as one CUDA graph per KV bucket.

Why: after W8 the Thinker is host-bound (K2: backbone 136.5 GPU-ms vs 181.5 host-ms per chunk in BF16; W8 halves the
GPU part, not the ~450 launches + HF Python per step).

What makes a decode step capturable (modeling_nemotron_h.py, snapshot 6533e8de):
  * Mamba mixers: decode updates conv/SSM state IN PLACE (causal_conv1d_update, selective_state_update); the only
    data-dependent branch, `cache_position[0] > 0`, is answered on the host by ``mamba_host_pos`` (required). The
    per-session state tensors are copied once into persistent buffers and the cache's lists point at them, so the
    captured graphs stay valid across sessions.
  * Attention (4 layers, no rotary): the cache ``update`` concatenates (new shapes every step). Here it writes in
    place into static K/V [L, B, H, Lmax, D] at a device position tensor and returns the first ``Lb`` positions
    (power-of-two bucket ≥ t+1); ``_update_causal_mask`` returns a static additive mask (0 valid, dtype-min empty) —
    the same values the original decode mask has over the valid keys.
  * The position only reaches the mask and the cache, so the host keeps it (cache._v2_pos += 1 after each replay).
Prefill (T > 1), batch > 1, no cache, t ≥ Lmax → the upstream path (eager). A capture error is fatal (raised).
Not exact by construction (masked padded attention window) → parity measured, not assumed.
"""

from __future__ import annotations

import threading
import time

import torch

from mynah_v2.model.capture_guard import GUARD
from mynah_v2.model.eartts_graph import bucket_for, sync_watch


class ThinkerGraph:
    def __init__(self, engine, lmax: int = 16384):
        self.stt = engine.wrapper.model.stt_model
        self.llm = self.stt.llm
        if not hasattr(self.llm, "_v2_hostpos_orig"):
            raise RuntimeError("thinker_graph needs mamba_host_pos")
        self.attn = [i for i, l in enumerate(self.llm.layers) if getattr(l, "block_type", None) == "attention"]
        self.mamba = [i for i, l in enumerate(self.llm.layers) if getattr(l, "block_type", None) == "mamba"]
        m0 = self.llm.layers[self.attn[0]].mixer
        self.kvh, self.hd = m0.num_key_value_heads, m0.head_dim
        p = next(self.llm.parameters())
        self.dtype, self.device = p.dtype, p.device
        self.lmax = lmax
        self.K = self.V = None                    # [n_attn, B, kvh, lmax, hd]
        self.mask = None                          # [B, 1, 1, lmax]
        self.conv = self.ssm = None               # persistent Mamba state buffers (lists, per mamba layer)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.bucket = 64
        self.current = None                       # the converted cache object
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.lock = threading.Lock()
        self.stats = {"captures": 0, "replays": 0, "eager": 0, "eager_reasons": {}, "converted": 0,
                      "bucket_transitions": 0, "steps_by_bucket": {}, "max_t": 0, "lmax": lmax, "capture_ms": 0.0}
        self._orig_forward = self.stt.forward
        self._orig_cm = self.llm._update_causal_mask
        self.llm._update_causal_mask = self._causal_mask
        self.stt.forward = self._forward
        impl = getattr(self.llm.config, "_attn_implementation", None)
        if impl not in ("eager", "sdpa"):                 # FA2 wants a 2-D padding mask, not the 4-D additive one
            raise RuntimeError(f"thinker_graph: unsupported attention implementation {impl!r}")
        self._tl = threading.local()                      # graph mode is per THREAD: a prefill in the main thread
        self.graph_mode = False                           # must not see the capture state of the FC thread
        self._neg_control: set = set()                    # TESTS ONLY: re-inject a known bug ("drop_mask" |
        #                                                   "orig_block") to prove the parity tests catch it
        # NemotronHBlock.forward runs its whole body inside `with torch.cuda.stream(torch.cuda.default_stream(dev))`
        # (modeling_nemotron_h.py:767, "avoid NaN issues when using multiple GPUs"). Under capture that moves every
        # block's kernels OFF the capturing stream: they ran once eagerly during capture and never entered the graph,
        # so replays only re-ran norm_f + heads on stale hidden states (T1b/T2b: <unk> output, 124 ms/chunk below the
        # BF16 bandwidth floor). In graph mode (warm-up + capture) the blocks run the same body on the current stream.
        self._orig_block_fwd = {}
        for i, layer in enumerate(self.llm.layers):
            if type(layer).__name__ == "NemotronHBlock":
                self._orig_block_fwd[i] = layer.forward
                layer.forward = self._stream_neutral(layer, layer.forward)
        if len(self._orig_block_fwd) != len(self.llm.layers):
            raise RuntimeError("thinker_graph: unexpected backbone block type (stream-neutral patch not applied)")

    @property
    def graph_mode(self) -> bool:
        return getattr(self._tl, "on", False)

    @graph_mode.setter
    def graph_mode(self, v: bool) -> None:
        self._tl.on = bool(v)

    # ------------------------------------------------------------------------------------------------ patches
    def _stream_neutral(self, blk, orig):
        """Same body as NemotronHBlock.forward (snapshot 6533e8de, lines 760-791) without the stream switch; used
        only in graph mode, the upstream forward otherwise (eager numerics untouched)."""
        def forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
            if not self.graph_mode or "orig_block" in self._neg_control:
                return orig(hidden_states, cache_params=cache_params, cache_position=cache_position,
                            attention_mask=attention_mask)
            if "drop_mask" in self._neg_control:
                attention_mask = None
            residual = hidden_states
            hidden_states = blk.norm(hidden_states.to(dtype=blk.norm.weight.dtype))
            if blk.residual_in_fp32:
                residual = residual.to(torch.float32)
            if blk.block_type == "mamba":
                hidden_states = blk.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position)
            elif blk.block_type == "attention":
                # upstream does NOT pass attention_mask to the attention mixer (fine in eager: the cache holds exactly
                # t+1 keys). Here K/V are bucket-sized views with zero padding → without the additive mask the padded
                # keys score 0 and dilute the softmax ((t+1)/bucket). Pass the static mask (graph mode only).
                hidden_states = blk.mixer(hidden_states, attention_mask=attention_mask,
                                          cache_position=cache_position)[0]
            elif blk.block_type == "mlp":
                hidden_states = blk.mixer(hidden_states)
            else:
                raise ValueError(f"Invalid block_type: {blk.block_type}")
            return residual + hidden_states
        forward.__wrapped_v2__ = orig
        return forward

    def _causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.graph_mode and attention_mask is None:
            return self.mask[..., :self.bucket]
        return self._orig_cm(attention_mask, input_tensor, cache_position)

    def _update(self, key_states, value_states, layer_idx, *a, **kw):
        j = self.attn.index(layer_idx)
        self.K[j].index_copy_(2, self.pos, key_states.to(self.K.dtype))
        self.V[j].index_copy_(2, self.pos, value_states.to(self.V.dtype))
        return self.K[j][:, :, :self.bucket], self.V[j][:, :, :self.bucket]

    def _why(self, r):
        self.stats["eager"] += 1
        self.stats["eager_reasons"][r] = self.stats["eager_reasons"].get(r, 0) + 1

    # ------------------------------------------------------------------------------------------------ cache conversion
    def _convert(self, c):
        """Move a session's hybrid cache onto the persistent buffers (once per session, after the prefill)."""
        b = c.key_cache[self.attn[0]].shape[0]
        t0 = int(getattr(c, "_v2_pos", 0))
        if self.K is None or self.K.shape[1] != b:
            if self.graphs:
                raise RuntimeError("thinker_graph: batch size changed after capture")
            self.K = torch.zeros(len(self.attn), b, self.kvh, self.lmax, self.hd, dtype=self.dtype, device=self.device)
            self.V = torch.zeros_like(self.K)
            self.mask = torch.full((b, 1, 1, self.lmax), torch.finfo(self.dtype).min, dtype=self.dtype,
                                   device=self.device)
            self.conv = [torch.empty_like(c.conv_states[i]) for i in self.mamba]
            self.ssm = [torch.empty_like(c.ssm_states[i]) for i in self.mamba]
        self.K.zero_()
        self.V.zero_()
        self.mask.fill_(torch.finfo(self.dtype).min)
        for j, li in enumerate(self.attn):
            k, v = c.key_cache[li], c.value_cache[li]
            if k.shape[-2] != t0:
                raise RuntimeError(f"attention cache length {k.shape[-2]} != position {t0}")
            self.K[j, :, :, :t0] = k
            self.V[j, :, :, :t0] = v
        self.mask[..., :t0] = 0.0
        for j, li in enumerate(self.mamba):
            if c.conv_states[li].shape != self.conv[j].shape or c.ssm_states[li].dtype != self.ssm[j].dtype:
                raise RuntimeError("thinker_graph: Mamba state layout changed")
            self.conv[j].copy_(c.conv_states[li])
            self.ssm[j].copy_(c.ssm_states[li])
            c.conv_states[li] = self.conv[j]           # decode kernels update these in place from now on
            c.ssm_states[li] = self.ssm[j]
        for li in self.attn:                            # free the concatenated tensors, keep the layout markers
            c.key_cache[li] = c.key_cache[li][:, :, :0]
            c.value_cache[li] = c.value_cache[li][:, :, :0]
        c.update = self._update
        c._v2_graph = self
        self.current = c
        self.stats["converted"] += 1

    def _unconvert(self, c, t):
        """Back to the upstream cache (t ≥ Lmax): concatenated K/V of the valid prefix, unpatched update."""
        for j, li in enumerate(self.attn):
            c.key_cache[li] = self.K[j, :, :, :t].clone()
            c.value_cache[li] = self.V[j, :, :, :t].clone()
        for j, li in enumerate(self.mamba):
            c.conv_states[li] = self.conv[j].clone()
            c.ssm_states[li] = self.ssm[j].clone()
        del c.update
        c._v2_graph = None
        if self.current is c:
            self.current = None

    # ------------------------------------------------------------------------------------------------ step
    def _forward(self, input_embeds, cache=None, **kw):
        if (cache is None or input_embeds.dim() != 3 or input_embeds.shape[1] != 1 or not input_embeds.is_cuda
                or kw):
            self._why("not_decode" if cache is not None else "no_cache")
            return self._orig_forward(input_embeds, cache=cache, **kw)
        with self.lock:
            t = int(getattr(cache, "_v2_pos", 0))
            if getattr(cache, "_v2_past_lmax", False) or t + 1 > self.lmax:
                # outgrew the static window: back to the upstream cache ONCE, then eager for the rest of the session
                # (without the flag every later step would convert (zero 2 x lmax K/V) and unconvert again)
                if getattr(cache, "_v2_graph", None) is self:
                    self._unconvert(cache, t)
                cache._v2_past_lmax = True
                self._why("reached_lmax")
                return self._orig_forward(input_embeds, cache=cache)
            if getattr(cache, "_v2_graph", None) is not self:
                if self.current is not None and self.current is not cache:
                    self._unconvert(self.current, int(self.current._v2_pos))   # a new session takes the buffers
                self._convert(cache)
            prev = self.bucket
            self.pos.fill_(t)
            self.mask[..., t] = 0.0
            self.bucket = bucket_for(t + 1, self.lmax)
            if self.stats["replays"] + self.stats["captures"] and self.bucket != prev:
                self.stats["bucket_transitions"] += 1
            sb = self.stats["steps_by_bucket"]
            sb[str(self.bucket)] = sb.get(str(self.bucket), 0) + 1
            self.stats["max_t"] = max(self.stats["max_t"], t + 1)
            key = (self.bucket, input_embeds.dtype, tuple(input_embeds.shape))
            if key not in self.graphs:
                if not GUARD.may_capture("thinker_graph", key):  # barrier policy: eager on the static buffers
                    self._why("capture_deferred")
                    self.graph_mode = True
                    try:
                        out = self._orig_forward(input_embeds, cache=cache)
                    finally:
                        self.graph_mode = False
                    cache._v2_pos = t + 1
                    return out | {"cache": cache}
                return self._capture(key, input_embeds, cache, t)
            graph, static_in, static_out = self.graphs[key]
            static_in.copy_(input_embeds)
            graph.replay()
            cache._v2_pos = t + 1
            self.stats["replays"] += 1
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in static_out.items()} | {"cache": cache}

    def _capture(self, key, input_embeds, cache, t):
        with GUARD.capturing():
            return self._capture_locked(key, input_embeds, cache, t)

    def _capture_locked(self, key, input_embeds, cache, t):
        t0 = time.perf_counter()
        with torch.inference_mode(False):                   # normal tensor: replayed from no-grad and FC threads
            static_in = input_embeds.clone()
        self.graph_mode = True
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), sync_watch(self.stats):   # warm-up = this step's real result
                out_warm = self._orig_forward(static_in, cache=cache)
            torch.cuda.current_stream().wait_stream(s)
            cache._v2_pos = t                               # the capture call must see the same position
            graph = torch.cuda.CUDAGraph()
            prev_stream = torch.cuda.current_stream()
            try:
                # thread_local: another thread's CUDA calls (the main thread's codec decode / .cpu() while the FC thread captures a
                # new bucket) must not invalidate this capture or raise there (G1 crash 2026-11-11, t01 3rd call, bucket 1024).
                with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                    static_out = self._orig_forward(static_in, cache=cache)
            except Exception as e:
                self.stats.setdefault("first_capture_error", f"{type(e).__name__}: {e}")
                if GUARD.on_failure("thinker_graph", key, e, stream=prev_stream):    # barrier policy: generator reset, key → eager
                    cache._v2_pos = t + 1
                    return ({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out_warm.items()}
                            | {"cache": cache})
                raise RuntimeError(f"thinker_graph capture failed (bucket {self.bucket}): {type(e).__name__}: {e}") from e
        finally:
            self.graph_mode = False
        static_out = {k: v for k, v in static_out.items() if k != "cache"}
        self.graphs[key] = (graph, static_in, static_out)
        cache._v2_pos = t + 1
        self.stats["captures"] += 1
        self.stats["capture_ms"] += (time.perf_counter() - t0) * 1000
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out_warm.items()} | {"cache": cache}

    def remove(self):
        self.stt.forward = self._orig_forward
        self.llm._update_causal_mask = self._orig_cm
        for i, f in self._orig_block_fwd.items():
            self.llm.layers[i].forward = f
        if self.current is not None:
            self._unconvert(self.current, int(self.current._v2_pos))
        self.graphs.clear()
