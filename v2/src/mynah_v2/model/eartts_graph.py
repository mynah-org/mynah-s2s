"""EXPERIMENTAL: the whole EarTTS step (embeddings, conditioning, Gemma3 backbone, MoG refinement) in one CUDA graph.

Why (K2, 2026-11-11): the EarTTS step is host-launch-bound — backbone 31.7 GPU-ms vs 87.8 host-ms per chunk, ~4000
kernels; only the MoG refinement was graphed so far (``eartts_graph_genstep``). The backbone cannot be graphed with
HF's ``DynamicCache``: every step ``torch.cat``s the whole KV (new shapes, O(t) copy — also a cost that grows with the
session: 516 KB per frame for 28 layers × 16 heads × 72 × FP32 × CFG batch 2).

How:
  * ``StaticKV`` replaces the cache: K/V buffers [L, B, H, Lmax, D] written in place (``index_copy_`` at a device
    position tensor), an additive mask [B, 1, 1, Lmax] (0 = valid, -inf = empty) and a host-side length. The
    backbone attends over the first ``Lb`` positions, Lb = power-of-two bucket ≥ t+1 (min 64), so each bucket is one
    static shape → one graph; masked slots contribute exactly 0 probability.
  * The backbone gets the mask mapping (Gemma3 accepts a prepared dict), ``cache_position`` and ``position_ids``
    explicitly, so no host reads happen inside the step (EarTTS host schedule is required for the MoG part).
  * One graph per (pool, bucket, input signature), shared memory pool; inputs copied into static buffers; outputs
    cloned. RNG: PyTorch graph-safe philox, same scheme as ``eartts_graph_genstep``.
  * Buffers live in a small pool (default 2 slots: main loop + FC fast path); a slot is freed when its StaticKV is
    garbage-collected. No free slot, t ≥ Lmax, or any capture error → the step runs eager (upstream path) and the
    reason is counted; a StaticKV that reaches Lmax is converted back to a ``DynamicCache`` for the rest of the session.
Numerics: attention over a masked padded window is mathematically identical but not bit-identical to attention over
exactly t keys (different reduction length) → this is NOT an exact experiment; parity is measured, not assumed.
"""

from __future__ import annotations

import contextlib
import os
import threading
import weakref

import torch

from mynah_v2.model import eartts_rng
from mynah_v2.model.capture_guard import (BUDGET, GUARD, buckets, free_gib, is_oom, precapture_buckets,
                                          soft_cleanup)

NEG = float("-inf")


@contextlib.contextmanager
def sync_watch(stats: dict):
    """V2_GRAPH_SYNC_DEBUG=1|error: the first host-synchronizing op raises (with its stack); =warn: every sync site in
    the block is recorded in stats["sync_sites"] (deduplicated stacks). Unset: no-op."""
    mode = os.environ.get("V2_GRAPH_SYNC_DEBUG")
    if not mode:
        yield
        return
    import traceback
    import warnings
    sites = stats.setdefault("sync_sites", [])
    prev_show = warnings.showwarning

    def show(msg, cat, fn, ln, file=None, line=None):
        if "synchroniz" in str(msg):
            st = "".join(traceback.format_stack()[-14:-1])
            if st not in sites:
                sites.append(st)
        else:
            prev_show(msg, cat, fn, ln, file, line)
    warnings.showwarning = show
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn" if mode == "warn" else "error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = prev_show


def bucket_for(n: int, lmax: int, lo: int = 64) -> int:
    b = lo
    while b < n:
        b *= 2
    return min(b, lmax)


class _Slot:
    def __init__(self, n_layers, b, h, lmax, d, dtype, device, mask_dtype=None):
        with torch.inference_mode(False):      # normal tensors: written in place from no-grad and inference-mode threads
            self.k = torch.zeros(n_layers, b, h, lmax, d, dtype=dtype, device=device)
            self.v = torch.zeros_like(self.k)
            self.mask = torch.full((b, 1, 1, lmax), NEG, dtype=mask_dtype or dtype, device=device)
            self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.busy = False

    @classmethod
    def view_of(cls, fsg, i: int) -> "_Slot":
        """Slot i of fsg's pooled buffers: views [L, 2, H, Lmax, D] / [2, 1, 1, Lmax] (same layout as a plain slot)."""
        s = cls.__new__(cls)
        s.k, s.v = fsg.pool_k[:, :, i], fsg.pool_v[:, :, i]
        s.mask = fsg.pool_mask[:, i]
        with torch.inference_mode(False):
            s.pos = torch.zeros(1, dtype=torch.long, device=fsg.device)
        s.busy = False
        return s


class StaticKV:
    """Duck-typed HF cache: Gemma3 attention only calls ``update``; ``get_seq_length`` for completeness."""

    def __init__(self, slot: _Slot, idx: int, lmax: int):
        self.slot, self.idx, self.lmax = slot, idx, lmax
        self.t = 0
        self.bucket = 64
        slot.busy = True
        self._fin = weakref.finalize(self, _release, slot)

    @classmethod
    def from_dynamic(cls, dyn, slot: _Slot, idx: int, lmax: int) -> "StaticKV":
        kv = cls(slot, idx, lmax)
        ks, vs = _dyn_tensors(dyn)
        t0 = ks[0].shape[-2]
        slot.k.zero_()
        slot.v.zero_()
        slot.mask.fill_(NEG)
        for li, (k, v) in enumerate(zip(ks, vs)):
            slot.k[li, :, :, :t0] = k
            slot.v[li, :, :, :t0] = v
        slot.mask[..., :t0] = 0.0
        kv.t = t0
        return kv

    def prepare(self):
        """Host side, before the step: position t becomes valid, choose the bucket (no device reads)."""
        self.slot.pos.fill_(self.t)
        self.slot.mask[..., self.t] = 0.0
        self.bucket = bucket_for(self.t + 1, self.lmax)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        s = self.slot
        s.k[layer_idx].index_copy_(2, s.pos, key_states.to(s.k.dtype))
        s.v[layer_idx].index_copy_(2, s.pos, value_states.to(s.v.dtype))
        k, v = s.k[layer_idx][:, :, :self.bucket], s.v[layer_idx][:, :, :self.bucket]
        if k.dtype != key_states.dtype:                       # --eartts-dtype bf16w / bf16kv: KV dtype != compute
            k, v = k.to(key_states.dtype), v.to(key_states.dtype)
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.t

    def masks(self):
        m = self.slot.mask[..., :self.bucket]
        return {"full_attention": m, "sliding_attention": m}

    def to_dynamic(self, config):
        from transformers import DynamicCache
        dyn = DynamicCache(config=config)
        for li in range(self.slot.k.shape[0]):
            dyn.update(self.slot.k[li, :, :, :self.t].to(self.slot.mask.dtype, copy=True),
                       self.slot.v[li, :, :, :self.t].to(self.slot.mask.dtype, copy=True), li)
        return dyn


def cast_cache(pkv, dtype):
    """A cache whose float K/V tensors are in ``dtype``, WITHOUT touching ``pkv`` (the EarTTS initial/prompt cache may
    be shared: converting it in place made the FP32 baseline unrecoverable, J5). Same object if nothing to convert."""
    import copy
    ks, vs = _dyn_tensors(pkv) if (hasattr(pkv, "layers") or hasattr(pkv, "key_cache")) else ([], [])
    if not any(isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype for t in (*ks, *vs)):
        return pkv
    new = copy.copy(pkv)
    if hasattr(pkv, "layers"):
        layers = []
        for lay in pkv.layers:
            nl = copy.copy(lay)
            for attr in ("keys", "values"):
                t = getattr(lay, attr, None)
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    setattr(nl, attr, t.to(dtype))
            if isinstance(getattr(nl, "dtype", None), torch.dtype):
                nl.dtype = dtype
            layers.append(nl)
        new.layers = layers
    else:
        new.key_cache = [t.to(dtype) if t.is_floating_point() else t for t in pkv.key_cache]
        new.value_cache = [t.to(dtype) if t.is_floating_point() else t for t in pkv.value_cache]
    return new


def _release(slot):
    slot.busy = False


def _dyn_tensors(dyn):
    if hasattr(dyn, "layers"):                                  # transformers >= 4.56 Cache API
        return [l.keys for l in dyn.layers], [l.values for l in dyn.layers]
    return list(dyn.key_cache), list(dyn.value_cache)


class FullStepGraph:
    def __init__(self, duplex_tts, lmax: int = 4096, slots: int = 2):
        self.dt = duplex_tts                                   # DuplexEARTTS (has infer_codes_one_step, tts_model)
        self.m = duplex_tts.tts_model                          # RVQEARTTSModel
        self.bb = self.m.backbone
        cfg = self.bb.config
        if getattr(cfg, "_attn_implementation", "eager") not in ("eager", "sdpa"):
            raise RuntimeError(f"unsupported attention implementation {cfg._attn_implementation}")
        self.lmax = min(lmax, int(getattr(cfg, "sliding_window", None) or lmax))
        p = next(self.bb.parameters())
        self.shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim)
        self.dtype, self.device = p.dtype, p.device
        self.n_slots = slots
        self.slots: list[_Slot | None] = [None] * slots
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.lock = threading.Lock()
        self.stats = {"captures": 0, "replays": 0, "eager": 0, "converted": 0, "to_dynamic": 0,
                      "eager_reasons": {}, "capture_ms": 0.0, "bucket_transitions": 0, "steps_by_bucket": {},
                      "max_t": 0, "lmax": self.lmax}
        self._sig = None                                       # (arg shapes/dtypes, kw, KV batch) of the first real step
        self._t0 = None                                        # smallest KV length at conversion (precapture order)
        self._kv_of_slot: dict = {}                            # slot idx -> weakref(StaticKV) (eartts_batch idle rows)
        self.pool_k = None
        self.bb_dtype = None                                   # None: backbone in its loaded dtype (FP32, default)
        self.kv_dtype = self.dtype                             # static K/V slots (self.dtype: compute / masks)
        with torch.inference_mode(False):
            self.rng_b1 = torch.zeros(1, 2, dtype=torch.long, device=self.device)   # (seed, step), eartts_rng
        self.eager_step = duplex_tts.infer_codes_one_step
        self.bb_forward = self.bb.forward
        self.active: StaticKV | None = None
        self.bb.forward = self._bb_forward
        # upstream _prepare_conditioning branches on `subword_mask.any()` (a host read: forbidden inside a capture).
        # The step reads it ONCE outside the graph, the value is part of the graph key, and the patched function uses
        # that host value → the branch taken is exactly the eager one.
        self.mask_any: bool | None = None
        # CharAwareSubwordEncoder maps subword ids to characters through Python lists (.cpu().tolist(), per-token
        # lengths): not capturable. Its output depends only on the token id, so it is computed eagerly outside the
        # graph (same CFG-doubled input as upstream, cached per id) and fed to the graph through a static buffer.
        self._sub_cache: dict = {}
        self.sub_static: torch.Tensor | None = None
        self._prep_orig = getattr(self.m, "_prepare_conditioning", None)
        if self._prep_orig is not None:
            self.m._prepare_conditioning = self._prepare_conditioning

    def _prepare_conditioning(self, context_hidden_state, subword_ids, subword_mask, uncond_dec_flag):
        if self.mask_any is None:
            return self._prep_orig(context_hidden_state, subword_ids, subword_mask, uncond_dec_flag)
        m = self.m
        cond = torch.zeros((1, 1, m.hidden_size), device=uncond_dec_flag.device)
        if m.embed_context is not None and context_hidden_state is not None:
            cond = cond + m.embed_context(context_hidden_state)
        if self.mask_any == "batch":                          # eartts_batch: per-row subword embedding, already
            if m.embed_subword is not None and subword_ids is not None:     # zeroed on rows whose mask is False
                cond = cond + self.batch_sub                   # (x + 0 == x: same value as skipping the branch)
            return torch.where(uncond_dec_flag, m.null_emb, cond)
        if m.embed_subword is not None and subword_ids is not None:
            if subword_mask is None and context_hidden_state is not None:
                raise RuntimeError("graph path expects an explicit subword_mask")
            if subword_mask is not None and self.mask_any:
                cond = cond + self.sub_static
        return torch.where(uncond_dec_flag, m.null_emb, cond)

    # -- backbone: inject masks/positions when the active cache is ours
    def _bb_forward(self, *a, **kw):
        # eartts_rng: every noise draw of a step happens AFTER the backbone (MoG infer / generate_step), so the draw
        # counter restarts here: a capture's warm-up and the captured call (two forwards in one rows() context) use
        # the same draw indices as an eager step (ER2 smoke: without it B=1 replays drew other noise than eager).
        eartts_rng.CTX.draw = 0
        kv = kw.get("past_key_values")
        if isinstance(kv, StaticKV):
            kw["attention_mask"] = kv.masks()
            kw["cache_position"] = kv.slot.pos
            kw["position_ids"] = kv.slot.pos.view(1, 1)
        elif getattr(kv, "_v2_batch_kv", False):             # eartts_batch.BatchKV: per-row positions / masks
            kw.update(kv.inject())
        if self.bb_dtype is None:
            return self.bb_forward(*a, **kw)
        # --eartts-dtype bf16: cast at the backbone boundary (inside any graph being captured): inputs in, hidden
        # states back to the caller's dtype; MoG / lm_head / embeddings / CFG combination stay FP32 outside.
        x = kw.get("inputs_embeds")
        out_dtype = x.dtype if x is not None else torch.float32
        if x is not None and x.dtype != self.bb_dtype:
            kw["inputs_embeds"] = x.to(self.bb_dtype)
        if kv is not None and not isinstance(kv, StaticKV) and not getattr(kv, "_v2_batch_kv", False):
            kw["past_key_values"] = cast_cache(kv, self.bb_dtype)       # NEW cache object; the source is untouched
        out = self.bb_forward(*a, **kw)
        if getattr(out, "last_hidden_state", None) is not None and out.last_hidden_state.dtype != out_dtype:
            out.last_hidden_state = out.last_hidden_state.to(out_dtype)
        return out

    def set_backbone_dtype(self, dtype) -> dict:
        """--eartts-dtype bf16: backbone weights AND static KV in ``dtype`` (see set_dtypes)."""
        return self.set_dtypes(weights=dtype, kv=dtype)

    MODES = {"fp32": (None, None), "bf16": ("bf16", "bf16"), "bf16w": ("bf16", None), "bf16kv": (None, "bf16")}

    def set_mode(self, mode: str) -> dict | None:
        """--eartts-dtype fp32 | bf16 (weights + KV) | bf16w (weights only, KV FP32) | bf16kv (weights FP32, KV)."""
        w, k = self.MODES[mode]
        if w is None and k is None:
            return None
        return self.set_dtypes(weights=torch.bfloat16 if w else None, kv=torch.bfloat16 if k else None)

    def set_dtypes(self, weights=None, kv=None) -> dict:
        """weights: backbone parameters (Gemma3 layers + norms) converted IN PLACE with no copy kept (the FP32 storage
        is released: not reversible within the process — reload the engine for FP32); RoPE buffers stay FP32,
        positions stay integers; inputs/hidden states cast at the backbone boundary. kv: dtype of the static K/V slots
        (B=1 and pooled); masks follow the COMPUTE dtype (SDPA wants mask dtype == query dtype); when KV and compute
        dtypes differ, the attention window is cast to the compute dtype on read. Allowed while no slot is busy; drops
        the pool (an eartts_batch built before must be rebuilt) and every graph."""
        if any(s is not None and s.busy for s in self.slots):
            raise RuntimeError("set_dtypes: a slot is in use (call it before sessions open)")
        cuda = self.device.type == "cuda"
        a0 = torch.cuda.memory_allocated() if cuda else 0
        params = list(self.bb.parameters())
        old_w, old_kv = params[0].dtype, self.kv_dtype
        n = sum(p.numel() for p in params)
        if weights is not None:
            with torch.no_grad():
                for p in params:
                    if p.is_floating_point() and p.dtype != weights:
                        p.data = p.data.to(weights)
            self.bb_dtype = weights
            self.dtype = weights                                   # compute dtype: masks
        if kv is not None:
            self.kv_dtype = kv
        self.slots = [None] * self.n_slots
        self.pool_k = self.pool_v = self.pool_mask = None
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()
        if cuda:
            torch.cuda.empty_cache()
        nl, h, d = self.shape
        esz = lambda dt: torch.finfo(dt).bits // 8                 # noqa: E731
        slot = lambda kd, md: 2 * nl * 2 * h * self.lmax * d * esz(kd) + 2 * self.lmax * esz(md)   # noqa: E731
        gib = lambda b: round(b / 2**30, 3)                        # noqa: E731
        new_w = params[0].dtype
        info = {"weights": f"{old_w} -> {new_w}", "kv": f"{old_kv} -> {self.kv_dtype}",
                "from": str(old_w), "to": str(new_w), "backbone_params_M": round(n / 1e6, 1),
                "weights_GiB_before": gib(n * esz(old_w)), "weights_GiB_after": gib(n * esz(new_w)),
                "allocated_delta_GiB": gib((torch.cuda.memory_allocated() - a0)) if cuda else None,
                "slot_GiB_per_session_before": gib(slot(old_kv, old_w)),
                "slot_GiB_per_session_after": gib(slot(self.kv_dtype, self.dtype)),
                "lmax": self.lmax, "attn_implementation": getattr(self.bb.config, "_attn_implementation", None)}
        for b in (4, 8):
            info[f"B{b}_weights_plus_slots_GiB_before"] = gib(n * esz(old_w) + b * slot(old_kv, old_w))
            info[f"B{b}_weights_plus_slots_GiB_after"] = gib(n * esz(new_w) + b * slot(self.kv_dtype, self.dtype))
        self.stats["backbone_dtype"] = info
        return info

    # -- pooled slots (eartts_batch): every slot is a view of ONE [L, 2, S, H, Lmax, D] buffer, so the batched step
    # sees all sessions' CFG rows as one contiguous [L, 2S, ...] cache (cond rows 0..S-1, uncond rows S..2S-1: the
    # order of upstream's CFG torch.cat). A single session's slot view is [L, 2, H, Lmax, D], as before.
    def enable_pool(self, batch: int = 2) -> None:
        if getattr(self, "pool_k", None) is not None:
            return
        if any(s is not None and s.busy for s in self.slots):
            raise RuntimeError("eartts pool: a slot is in use (enable before sessions open)")
        if batch != 2:
            raise RuntimeError("eartts pool supports the CFG batch of 2 only")
        nl, h, d = self.shape
        S = self.n_slots
        self.slots = [None] * S                               # old per-slot buffers (solo phase) released
        # B=1 graphs referenced the old buffers: dropped, re-captured on the pool. Their memory pool is NOT reused:
        # capturing into a pool whose graphs were all destroyed asserts in torch 2.10 (CUDACachingAllocator.cpp:2731
        # "use_count > 0", Ada round 31) and poisons the generator → fresh pool handle.
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode(False):
            self.pool_k = torch.zeros(nl, 2, S, h, self.lmax, d, dtype=self.kv_dtype, device=self.device)
            self.pool_v = torch.zeros_like(self.pool_k)
            self.pool_mask = torch.full((2, S, 1, 1, self.lmax), NEG, dtype=self.dtype, device=self.device)
        for i in range(S):
            self.slots[i] = _Slot.view_of(self, i)
        self.stats["pooled_slots"] = S

    def live_t(self, i: int):
        """Current length of the StaticKV holding slot i (None if the slot is free)."""
        r = self._kv_of_slot.get(i)
        kv = r() if r is not None else None
        return None if kv is None or not kv.slot.busy else kv.t

    def _why(self, r):
        self.stats["eager"] += 1
        self.stats["eager_reasons"][r] = self.stats["eager_reasons"].get(r, 0) + 1

    def _slot(self, batch):
        for i in range(self.n_slots):
            s = self.slots[i]
            if s is None:
                n, h, d = self.shape
                self.slots[i] = s = _Slot(n, batch, h, self.lmax, d, self.kv_dtype, self.device, self.dtype)
            if not s.busy and s.k.shape[1] == batch:
                return i, s
        return None, None

    def _convert(self, kv, args, kw):
        """DynamicCache → StaticKV in a free slot (a session's first step). Returns the StaticKV or the eager reason.
        Caller holds self.lock."""
        if kv is None:
            return "no_cache"
        batch = _dyn_tensors(kv)[0][0].shape[0]
        idx, slot = self._slot(batch)
        if slot is None:
            return "no_free_slot"
        if _dyn_tensors(kv)[0][0].shape[-2] + 1 > self.lmax:
            return "too_long"
        kv = StaticKV.from_dynamic(kv, slot, idx, self.lmax)
        self._kv_of_slot[idx] = weakref.ref(kv)
        kv._v2_rng_seed, kv._v2_rng_step = eartts_rng.session_seed(), 0   # per-session noise stream (if enabled)
        self.stats["converted"] += 1
        if self._sig is None:                                 # learned call signature for precapture()
            self._sig = (tuple((tuple(x.shape), x.dtype) for x in args), dict(kw), batch)
        self._t0 = kv.t if self._t0 is None else min(self._t0, kv.t)   # session start position (KV length)
        return kv

    @staticmethod
    def _on_device(x) -> bool:                       # CPU tests replace it
        return x.is_cuda

    def convert_for_batch(self, d: dict) -> bool:
        """eartts_batch: convert a first-step call's DynamicCache in place (d["past_key_values"]) so it can join the
        batched step instead of going B=1. False if this call must stay on the per-session path."""
        kv = d.get("past_key_values")
        x = d.get("current_subword_id")
        if isinstance(kv, StaticKV) or kv is None or x is None or x.numel() != 1 or not self._on_device(x) \
                or d.get("request_id") is not None or not d.get("ignore_eos_flag_stop", True):
            return False
        args = tuple(d[a] for a in ("current_subword_id", "prev_subword_id", "current_subword_mask",
                                    "prev_audio_tokens"))
        kw = {k: d.get(k, dv) for k, dv in (("guidance_enabled", True), ("generation_config", None),
                                            ("ignore_eos_flag_stop", True))}
        with self.lock:
            conv = self._convert(kv, args, kw)
        if isinstance(conv, str):
            return False
        d["past_key_values"] = conv
        return True

    def step(self, current_subword_id, prev_subword_id, current_subword_mask, prev_audio_tokens, past_key_values,
             guidance_enabled=True, generation_config=None, ignore_eos_flag_stop=True, request_id=None):
        args = (current_subword_id, prev_subword_id, current_subword_mask, prev_audio_tokens)
        kw = dict(guidance_enabled=guidance_enabled, generation_config=generation_config,
                  ignore_eos_flag_stop=ignore_eos_flag_stop)
        with self.lock:
            kv = past_key_values
            if request_id is not None or not ignore_eos_flag_stop or not self._on_device(current_subword_id):
                self._why("unsupported_args")
                return self._eager(args, kv, kw, request_id)
            if current_subword_id.numel() != 1 and not isinstance(kv, StaticKV):
                self._why("batch_gt_1")
                return self._eager(args, kv, kw, request_id)
            if not isinstance(kv, StaticKV):
                conv = self._convert(kv, args, kw)
                if isinstance(conv, str):
                    self._why(conv)
                    return self._eager(args, kv, kw, request_id)
                kv = conv
            if kv.t + 1 > self.lmax:                         # session outgrew the static window: eager from now on
                dyn = kv.to_dynamic(self.bb.config)
                self.stats["to_dynamic"] += 1
                self._why("reached_lmax")
                return self._eager(args, dyn, kw, request_id)
            # the one host read per step, outside the graph: token id + mask flag
            if current_subword_id.numel() != 1:
                raise RuntimeError("eartts_graph_full: batch > 1 on a static cache (unsupported)")
            tid, mflag = torch.stack([current_subword_id.reshape(-1)[0],
                                      current_subword_mask.reshape(-1)[0].long()]).tolist()
            self.mask_any = bool(mflag)
            if self.mask_any and getattr(self.m, "embed_subword", None) is not None:
                self._load_subword(tid, current_subword_id, current_subword_mask, guidance_enabled)
            prev_bucket = kv.bucket if kv.t > 0 else None
            kv.prepare()
            if prev_bucket is not None and kv.bucket != prev_bucket:
                self.stats["bucket_transitions"] += 1
            sb = self.stats["steps_by_bucket"]
            sb[str(kv.bucket)] = sb.get(str(kv.bucket), 0) + 1
            self.stats["max_t"] = max(self.stats["max_t"], kv.t + 1)
            key = (kv.idx, kv.bucket, self.mask_any, tuple((tuple(x.shape), x.dtype) for x in args),
                   tuple(sorted((k, repr(v)) for k, v in kw.items())))
            self._rng_fill(kv)
            with eartts_rng.rows(self.rng_b1):              # per-session noise (--eartts-rng session), else no-op
                out = self._dispatch(key, args, kv, kw)
            kv._v2_rng_step = getattr(kv, "_v2_rng_step", 0) + 1
            return out

    def _rng_fill(self, kv) -> None:
        if eartts_rng.STATE["enabled"]:
            self.rng_b1.copy_(torch.tensor([[getattr(kv, "_v2_rng_seed", 0), getattr(kv, "_v2_rng_step", 0)]]))

    def _dispatch(self, key, args, kv, kw):
        if key not in self.graphs:
            if not GUARD.may_capture("eartts", key):          # barrier policy: not quiet now → eager on the static
                self._why("capture_deferred")                 # cache (= the graph's warm-up numerics), retry later
                self.stats["capture_deferred"] = self.stats.get("capture_deferred", 0) + 1
                codes, _ = self.eager_step(*args, kv, **kw)
                kv.t += 1
                return codes, kv
            self.stats["lazy_captures"] = self.stats.get("lazy_captures", 0) + 1
            return self._capture(key, args, kv, kw)
        g = self.graphs[key]
        graph, static_in, static_out = g
        for s_, x in zip(static_in, args):
            s_.copy_(x)
        graph.replay()
        kv.t += 1
        self.stats["replays"] += 1
        return static_out.clone(), kv

    def sub_embed(self, tid, ids, mask, guidance):
        """CFG-doubled subword embedding of one token id ([2 or 1, 1, H]), computed eagerly once per id."""
        key = (tid, bool(guidance))
        e = self._sub_cache.get(key)
        if e is None:
            reps = 2 if guidance else 1
            with torch.no_grad():
                e = self.m.embed_subword(torch.cat([ids] * reps, 0), torch.cat([mask] * reps, 0)).detach().clone()
            self._sub_cache[key] = e
            self.stats["subword_cache_misses"] = self.stats.get("subword_cache_misses", 0) + 1
        return e

    def _load_subword(self, tid, ids, mask, guidance):
        e = self.sub_embed(tid, ids, mask, guidance)
        if self.sub_static is None or self.sub_static.shape != e.shape:
            if self.sub_static is not None and self.graphs:     # a None buffer is referenced by no graph
                raise RuntimeError("subword embedding shape changed after capture")
            with torch.inference_mode(False):                   # normal tensor: written from both modes
                self.sub_static = torch.empty_like(e)
        self.sub_static.copy_(e)

    def _eager(self, args, kv, kw, request_id):
        self.mask_any = None                                  # upstream conditioning, untouched
        out = self.eager_step(*args, kv, **kw, **({"request_id": request_id} if request_id is not None else {}))
        return out

    def _capture(self, key, args, kv, kw):
        with GUARD.capturing():
            return self._capture_locked(key, args, kv, kw)

    def _capture_locked(self, key, args, kv, kw):
        import time
        t0 = time.perf_counter()
        with torch.inference_mode(False):
            # NORMAL tensors: the main loop (no_grad) and the FC thread (inference_mode) replay the same keys; an
            # inference tensor captured by the FC thread would make a later main-loop replay's copy_ raise
            static_in = tuple(x.clone() for x in args)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), sync_watch(self.stats):  # warm-up = this call's real result (writes KV at t)
            codes_warm, _ = self.eager_step(*static_in, kv, **kw)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        prev_stream = torch.cuda.current_stream()
        try:
            # thread_local: another thread's CUDA calls (the main thread's codec decode / .cpu() while the FC thread captures a
            # new bucket) must not invalidate this capture or raise there (G1 crash 2026-11-11, t01 3rd call, bucket 1024).
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                static_out, _ = self.eager_step(*static_in, kv, **kw)
        except Exception as e:
            # A failed capture leaves PyTorch's CUDA RNG in "capturing" state for the whole process (any later random
            # op raises "Offset increment outside graph capture"), so an eager fallback is NOT safe: fail fast.
            self.stats.setdefault("first_capture_error", f"{type(e).__name__}: {e}")
            if GUARD.on_failure("eartts", key, e, stream=prev_stream):           # barrier policy: generator reset, key → eager for good
                self.stats["capture_failed_recovered"] = self.stats.get("capture_failed_recovered", 0) + 1
                kv.t += 1
                return codes_warm, kv
            raise RuntimeError(f"eartts_graph_full capture failed (bucket {kv.bucket}): {type(e).__name__}: {e}") from e
        self.graphs[key] = (graph, static_in, static_out)
        self.stats["captures"] += 1
        self.stats["capture_ms"] += (time.perf_counter() - t0) * 1000
        kv.t += 1
        return codes_warm, kv

    # ------------------------------------------------------------------------------------------------ precapture
    def precapture_keys(self, n_slots: int | None = None) -> list[tuple[int, int, bool]]:
        """(slot, bucket, mask_any), bucket-major: the bucket of the first converted session (learned t0) and upward,
        then the lower ones (capture_guard.precapture_buckets, BUDGET.min/max_bucket); every slot < n_slots, both
        conditioning branches."""
        n = self.n_slots if n_slots is None else min(n_slots, self.n_slots)
        bl = precapture_buckets(self.lmax, self._t0, BUDGET.min_bucket, BUDGET.max_bucket)
        return [(i, b, m) for b in bl for i in range(n) for m in (True, False)]

    def _precapture_args(self, subword_id: int):
        dev = self.device
        if self._sig is not None:
            sig, kw, batch = self._sig
            args = tuple(torch.zeros(shape, dtype=dt, device=dev) for shape, dt in sig)
        else:
            c = int(self.dt.codec_silence_tokens.numel())
            ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
            args = (ids, ids.clone(), torch.ones(1, 1, dtype=torch.bool, device=dev),
                    torch.zeros(1, 1, c, dtype=torch.long, device=dev))
            kw = dict(guidance_enabled=True, generation_config=self.dt._get_generation_config(guidance_enabled=True),
                      ignore_eos_flag_stop=True)
            batch = 2 if kw["guidance_enabled"] else 1
        args = (torch.full_like(args[0], subword_id), torch.full_like(args[1], subword_id), *args[2:])
        return args, kw, batch

    def precapture(self, n_slots: int | None = None, subword_id: int = 0) -> dict:
        """Capture every enumerable key of slots [0, n_slots) at a QUIET point (before sessions run concurrently), so
        no capture happens later in a session or FC thread. Uses the call signature of the first real step if one
        was seen (``_sig``), else the upstream one (wrapper :2534-2548: ids/mask [1,1], codes [1,1,C], CFG batch 2,
        guidance on, ``_get_generation_config(True)``). Slots must be free; buffers are reset afterwards. Allocates
        the slots (KV ~2 GiB each at lmax 4096): pass n_slots = the number of sessions."""
        import time
        why = GUARD.quiet_reason()
        if why is not None:
            GUARD.note_precapture("eartts", 0, 0.0, skipped=why)
            return {"skipped": why}
        args, kw, batch = self._precapture_args(subword_id)
        t0 = time.perf_counter()
        a0 = torch.cuda.memory_allocated()
        n_new, stop, slots_mem = 0, None, 0
        all_b = buckets(self.lmax)
        keys = self.precapture_keys(n_slots)
        with self.lock, torch.no_grad():                          # the main loop's mode (wrapper: no_grad)
            try:
                for i, b, m in keys:
                    try:
                        s = self.slots[i]
                        if s is None:                             # slots are needed by the sessions anyway
                            nl, h, d = self.shape
                            m0 = torch.cuda.memory_allocated()
                            self.slots[i] = s = _Slot(nl, batch, h, self.lmax, d, self.kv_dtype, self.device, self.dtype)
                            slots_mem += torch.cuda.memory_allocated() - m0
                        if s.busy or s.k.shape[1] != batch:
                            raise RuntimeError(f"precapture: slot {i} busy or KV batch {s.k.shape[1]} != {batch}")
                        self.mask_any = m
                        margs = (args[0], args[1], torch.full_like(args[2], m), args[3])
                        key = (i, b, m, tuple((tuple(x.shape), x.dtype) for x in margs),
                               tuple(sorted((k_, repr(v)) for k_, v in kw.items())))
                        if key in self.graphs:
                            continue
                        if free_gib() < BUDGET.min_free_gib:
                            stop = f"budget: free {free_gib():.2f} GiB < min_free_gib {BUDGET.min_free_gib}"
                            break
                        kv = StaticKV(s, i, self.lmax)
                        try:
                            k = all_b.index(b)
                            kv.t = all_b[k - 1] if k > 0 else 0       # → bucket_for(t + 1) == b
                            if m and getattr(self.m, "embed_subword", None) is not None:
                                self._load_subword(subword_id, margs[0], margs[2], kw.get("guidance_enabled", True))
                            kv.prepare()
                            assert kv.bucket == b, (kv.bucket, b)
                            self._rng_fill(kv)
                            with eartts_rng.rows(self.rng_b1):        # per-row noise buffer bound in the graph
                                self._capture(key, margs, kv, kw)
                        finally:
                            kv._fin()                             # release the slot NOW (not at a later gc)
                            s.k.zero_()
                            s.v.zero_()
                            s.mask.fill_(NEG)
                            s.pos.zero_()
                    except Exception as e:  # noqa: BLE001
                        if not is_oom(e):
                            raise
                        stop = f"oom at slot {i} bucket {b}: {str(e).splitlines()[0][:160]}"
                        break
                    if key not in self.graphs:                    # failed + recovered by the guard: retried lazily
                        GUARD.disabled.discard(("eartts", key))
                        stop = f"capture failed at slot {i} bucket {b}: {self.stats.get('first_capture_error')}"
                        break
                    n_new += 1
            finally:
                self.mask_any = None
        if stop is not None:
            soft_cleanup()
        ms = (time.perf_counter() - t0) * 1e3
        self.stats["precaptured"] = self.stats.get("precaptured", 0) + n_new
        out = {"precaptured": n_new, "planned": len(keys), "stopped": stop, "ms": round(ms, 1), "slots": n_slots,
               "bucket_order": precapture_buckets(self.lmax, self._t0, BUDGET.min_bucket, BUDGET.max_bucket),
               "t0": self._t0, "sig": "learned" if self._sig is not None else "upstream-default",
               "slots_alloc_GiB": round(slots_mem / 2**30, 3),
               "alloc_delta_GiB_incl_slots": round((torch.cuda.memory_allocated() - a0) / 2**30, 3)}
        self.stats["precapture"] = out
        GUARD.note_precapture("eartts", n_new, ms, stopped=stop)
        return out

    def remove(self):
        self.bb.forward = self.bb_forward
        if self._prep_orig is not None:
            self.m._prepare_conditioning = self._prep_orig
        self.graphs.clear()
