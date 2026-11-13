# SPDX-License-Identifier: MIT AND Apache-2.0
# Original code: MIT (see LICENSE). The spec_to_wav ops in make_cached_spec_to_wav are derived from NeMo Speech
# ear_tts_vae_codec.py (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, Apache License 2.0, LICENSES/Apache-2.0.txt)
# and modified (cached window envelope). See NOTICE.md.
"""Codec-path host syncs (EarTTS VAE codec decode) — opt-in experiments, registered in ``mynah_v2.model.perf``.

Where the codec blocks the host (pinned NeMo 097dfe9):
  * ``modules/ear_tts_vae_codec.py:248`` ``spec_to_wav``: ``assert (window_envelope > 1e-11).all()`` — a CUDA bool read
    on EVERY ``RVQVAEModel.decode`` call (main path wrapper :2663, FC path :1558, reminder :2270). This is the sync that
    makes CompTimer's codec host time == GPU time: it drains the whole default stream (all sessions' queued work).
    The envelope depends only on (T, n_fft, hop, win, device): ``codec_nosync`` computes it once per key with the
    original ops (assert included, one sync per new key) and reuses the identical tensor -> same division, same bits.
  * FC path wrapper :1561 ``_tts_decoded_audio.detach().cpu()`` (and reminder :2273): synchronous D2H per EarTTS step.
    Those chunks are only appended to a list / put on a queue; the host first touches them in the pipeline
    (``torch.cat(...).to(device)``, streaming_s2s_pipeline.py:1022/1315/2094). ``codec_fc_async_d2h`` makes that
    ``.cpu()`` an async copy into pinned memory + CUDA event; the returned CPU tensor (``PendingHost``) waits on the event
    the first time any torch op reads its DATA (metadata such as ``.shape`` does not wait).
  * wrapper :2678 ``torch.cuda.synchronize()`` after the main-path decode: already a no-op under ``no_timer_sync``.
  * main path: decoded audio stays on device (state.update_state, s2s_state.py:142); the host copy is V2's own
    ``reference.py`` StepOutput (one per session-step; real consumption). The next sync after the codec in
    ``infer_one_step`` is the token ``.tolist()`` (wrapper :2698), so removing the codec sync moves, not removes, the
    wait there (all threads share the default stream). Only the number of stream drains per round decreases.
"""

from __future__ import annotations

import sys
import threading

import torch
import torch.nn.functional as F

_CODEC_MOD = "nemo.collections.speechlm2.modules.ear_tts_vae_codec"


# ------------------------------------------------------------------------------------------------ codec_nosync
def make_cached_spec_to_wav(orig, cache: dict, lock: threading.Lock):
    """``spec_to_wav`` (ear_tts_vae_codec.py:166-251, copied op for op) with the window envelope cached per key."""

    def spec_to_wav(spec, n_fft, hop_length, win_length, window_fn=torch.hann_window, constrain_value_range=False):
        with torch.autocast(device_type=spec.device.type, enabled=False):
            if window_fn != torch.hann_window:
                raise ValueError(f"`window_fn` should be 'torch.hann_window', but got '{window_fn}'.")
            pad = (win_length - hop_length) // 2
            T = spec.size(-1)
            window = window_fn(win_length, device=spec.device)
            ifft = torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward")
            window_unsqz = window.unsqueeze(-1)
            if constrain_value_range:
                ifft = torch.where(ifft >= 0, torch.minimum(ifft, window_unsqz), torch.maximum(ifft, -window_unsqz))
            ifft = ifft * window_unsqz
            output_size = (T - 1) * hop_length + win_length
            wav = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, win_length),
                         stride=(1, hop_length))[..., 0, 0, pad:-pad]
            key = (T, n_fft, hop_length, win_length, str(spec.device))
            env = cache.get(key)
            if env is None:
                window_sq = window.square().expand(T, -1).transpose(0, 1)
                env = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, win_length),
                             stride=(1, hop_length)).squeeze()[pad:-pad]
                # the original check, evaluated once per key (host sync here only); also orders the tensor before
                # any other thread/stream reads it from the cache
                assert (env > 1e-11).all(), "Window envelope has zero values, cannot normalize."
                with lock:
                    env = cache.setdefault(key, env)
            return wav / env

    spec_to_wav.__v2_orig__ = orig
    return spec_to_wav


def nosync_enable(engine) -> dict:
    mod = sys.modules.get(_CODEC_MOD) or __import__(_CODEC_MOD, fromlist=["spec_to_wav"])
    if not hasattr(mod.spec_to_wav, "__v2_orig__"):
        mod.spec_to_wav = make_cached_spec_to_wav(mod.spec_to_wav, {}, threading.Lock())
    return {"module": _CODEC_MOD, "note": "window envelope cached per (T, n_fft, hop, win, device); assert once per key"}


def nosync_disable(engine) -> None:
    mod = sys.modules.get(_CODEC_MOD)
    if mod is not None and hasattr(mod.spec_to_wav, "__v2_orig__"):
        mod.spec_to_wav = mod.spec_to_wav.__v2_orig__


# ------------------------------------------------------------------------------------------------ deferred D2H
def _metadata_funcs():
    T = torch.Tensor
    return {T.shape.__get__, T.size, T.dim, T.ndim.__get__, T.numel, T.dtype.__get__, T.device.__get__,
            T.is_cuda.__get__, T.__len__, T.is_floating_point, T.requires_grad.__get__}


_META = _metadata_funcs()


class PendingHost(torch.Tensor):
    """A CPU tensor whose content is being written by an async device->host copy. The first torch op that reads its
    data waits on the copy's event (``event.synchronize()``: only that copy, never a device-wide sync), then the op runs
    on the plain tensor and returns plain tensors. Metadata ops never wait. The device source is kept alive until the
    wait (the caching host allocator also records the copy's stream use on the pinned block, so a block is never
    reused before its copy completes even if this tensor is dropped unread)."""

    @staticmethod
    def wrap(host: torch.Tensor, event, keepalive=None) -> "PendingHost":
        t = host.as_subclass(PendingHost)
        t._v2_event, t._v2_keep = event, keepalive
        return t

    def wait(self) -> None:
        ev = self.__dict__.get("_v2_event")
        if ev is not None:
            ev.synchronize()
            self._v2_event, self._v2_keep = None, None

    @property
    def ready(self) -> bool:
        return self.__dict__.get("_v2_event") is None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in _META:
            _visit(args, _wait_one)
            _visit(kwargs, _wait_one)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def _wait_one(x):
    if isinstance(x, PendingHost):
        x.wait()


def _visit(obj, fn):
    if isinstance(obj, (list, tuple)):
        for o in obj:
            _visit(o, fn)
    elif isinstance(obj, dict):
        for o in obj.values():
            _visit(o, fn)
    else:
        fn(obj)


def to_host_async(src: torch.Tensor) -> PendingHost:
    """Enqueue ``src`` (CUDA) -> fresh pinned host buffer on the current stream; never blocks the host."""
    src = src.as_subclass(torch.Tensor).detach()
    host = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
    host.copy_(src, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return PendingHost.wrap(host, ev, keepalive=src)


class DeferredD2H(torch.Tensor):
    """Device tensor marker returned by the FC-thread codec decode: ``.cpu()`` becomes ``to_host_async``. Every other
    op behaves as on a plain tensor (``detach`` keeps the marker, which is all the FC code does before ``.cpu()``)."""

    def cpu(self, *a, **kw):
        if a or kw or self.device.type != "cuda":
            return self.as_subclass(torch.Tensor).cpu(*a, **kw)
        return to_host_async(self)


def is_fc_thread() -> bool:
    # upstream FC threads: threading.Thread(target=_fc_async_thread_fn) -> "Thread-N (_fc_async_thread_fn)"
    return "_fc_async" in threading.current_thread().name


def make_fc_decode(orig, predicate=is_fc_thread):
    def decode(*a, **kw):
        out = orig(*a, **kw)
        if not predicate():
            return out
        wav, wav_len = out
        return (wav.as_subclass(DeferredD2H) if isinstance(wav, torch.Tensor) else wav), wav_len

    decode.__v2_fcd2h__ = orig
    return decode


def fc_d2h_enable(engine) -> dict:
    codec = engine.wrapper.model.tts_model.audio_codec
    # instance attribute on top of whatever is installed (CompTimer may wrap it again later)
    if not hasattr(codec.decode, "__v2_fcd2h__"):
        codec._v2_fcd2h_had = "decode" in codec.__dict__
        codec.decode = make_fc_decode(codec.decode)
    return {"note": "FC-thread codec output: .cpu() -> pinned async copy + event (PendingHost)"}


def fc_d2h_disable(engine) -> None:
    codec = engine.wrapper.model.tts_model.audio_codec
    f = codec.__dict__.get("decode")
    if f is not None and hasattr(f, "__v2_fcd2h__"):
        if codec.__dict__.pop("_v2_fcd2h_had", False):
            codec.decode = f.__v2_fcd2h__
        else:
            del codec.decode


# ------------------------------------------------------------------------------------------------ codec_fc_2codes
# FC path (wrapper :1541-1564): one codec decode per EarTTS step (1 code) keeps the streaming cache aligned. Here the
# codes are decoded in PAIRS (2 per call, like the main path): on the 1st code of a pair the decode wrapper keeps the
# code (per FC phase, thread-local) and returns an empty CPU chunk; on the 2nd it decodes both with the cache. Every
# code is decoded exactly once, in order, so the cache stays aligned; a code still pending when ``_run_fc_async_steps``
# returns (odd step count, abort, exception) is decoded right then and its audio appended to the returned chunk list
# and the output queue (same order as upstream). Not bit-identical (2-at-once vs 1+1: max |diff| 1.4e-6, CA1/CN1);
# audio-only (codec output never feeds the model). The reminder path (:2270) runs outside the scope: unchanged.
_FC2 = threading.local()


def _fc2_scope():
    return getattr(_FC2, "scope", None)


def make_fc2_decode(orig):
    def decode(*a, **kw):
        sc = _fc2_scope()
        if sc is None:
            return orig(*a, **kw)
        from mynah_v2.model.codec_batch import decode_kwargs
        d = decode_kwargs(a, kw)
        if (d is None or d["cache"] is None or d["flush"] or not isinstance(d["code"], torch.Tensor)
                or d["code"].dim() != 3 or tuple(d["code"].shape[:2]) != (1, 1)
                or (sc["pending"] is not None and sc["pending"]["cache"] is not d["cache"])):
            # not pairable (never seen in the FC loop): decode the held code first, then this call, audio in order
            w0 = _decode_pending(orig, sc)
            out = orig(*a, **kw)
            if w0 is None:
                return out
            return torch.cat([w0.to(out[0].device), out[0]], dim=-1), None
        pend = sc["pending"]
        if pend is None:
            # own copy: the code may live in a static (CUDA-graph) output buffer overwritten by the next EarTTS step
            d["code"] = d["code"].clone()
            if isinstance(d["code_len"], torch.Tensor):
                d["code_len"] = d["code_len"].clone()
            sc["pending"] = d
            sc["held"] += 1
            empty = torch.zeros((1, 1, 0), dtype=torch.float32)          # CPU: .detach().cpu() is a no-op
            return empty, (None if d["code_len"] is None else d["code_len"] * 0)
        sc["pending"] = None
        sc["pairs"] += 1
        code = torch.cat([pend["code"], d["code"]], dim=1)
        ln = None if d["code_len"] is None else pend["code_len"].reshape(-1) + d["code_len"].reshape(-1)
        return orig(code, ln, constrain_value_range=d["constrain_value_range"], cache=d["cache"], flush=False)

    decode.__v2_fc2__ = orig
    return decode


def _fp32_ctx():
    try:
        from nemo.collections.speechlm2.parts.precision import fp32_precision
        return fp32_precision()
    except ImportError:                                  # CPU tests without NeMo
        import contextlib
        return contextlib.nullcontext()


def _decode_pending(orig, sc):
    """Decode a held code alone (the cache stays aligned); returns its audio (or None if nothing is held)."""
    pend = sc["pending"]
    if pend is None:
        return None
    sc["pending"] = None
    sc["flushed"] += 1
    with _fp32_ctx(), torch.no_grad():
        wav, _ = orig(pend["code"], pend["code_len"], constrain_value_range=pend["constrain_value_range"],
                      cache=pend["cache"], flush=False)
    return wav


def _flush_into_scope(orig, sc):
    """End of an FC phase: the held code's audio goes after everything the phase produced."""
    wav = _decode_pending(orig, sc)
    if wav is not None:
        sc["late_audio"].append(wav.detach().cpu())


def make_fc2_steps(orig_steps, decode_inner, stats):
    def run(*a, **kw):
        if not getattr(run, "__v2_fc2_on__", True):
            return orig_steps(*a, **kw)
        outer = _fc2_scope()
        sc = {"pending": None, "held": 0, "pairs": 0, "flushed": 0, "late_audio": []}
        _FC2.scope = sc
        res = None
        try:
            res = orig_steps(*a, **kw)
        finally:
            _FC2.scope = outer
            _flush_into_scope(decode_inner(), sc)
            for k in ("held", "pairs", "flushed"):
                stats[k] = stats.get(k, 0) + sc[k]
            stats["calls"] = stats.get("calls", 0) + 1
            q = kw.get("tts_audio_output_queue")
            for x in sc["late_audio"]:
                if q is not None:
                    q.put(x)
                if isinstance(res, tuple) and len(res) >= 4 and isinstance(res[3], list):
                    res[3].append(x)
        return res

    run.__v2_fc2__ = orig_steps
    return run


def fc2_enable(engine) -> dict:
    w = engine.wrapper
    codec = w.model.tts_model.audio_codec
    stats = getattr(engine, "_v2_fc2_stats", None)
    if stats is None:
        stats = engine._v2_fc2_stats = {}
    if not hasattr(codec.decode, "__v2_fc2__"):
        codec._v2_fc2_had = "decode" in codec.__dict__
        codec.decode = make_fc2_decode(codec.decode)
    dec_inner = codec.__dict__["decode"].__v2_fc2__
    if getattr(w, "_v2_fc2_steps", None) is None:
        cur = w._run_fc_async_steps
        instrumented = hasattr(engine, "_instrument_fc") and hasattr(cur, "__wrapped_v2__")
        # like fc_ctx_fix: wrap the call UNDER the engine's FC recorder and rebuild the recorder on top
        # (recorded → fc2 → [fc_fixed →] upstream), so MultiSessionEngine can still re-route it
        inner = cur.__wrapped_v2__ if instrumented else cur
        f = make_fc2_steps(inner, lambda: dec_inner, stats)
        w._v2_fc2_steps = f
        w._v2_fc2_had = "_run_fc_async_steps" in w.__dict__
        w._run_fc_async_steps = f
        if instrumented:
            engine._instrument_fc()
    w._v2_fc2_steps.__v2_fc2_on__ = True
    return {"note": "FC codec decodes 2 codes per call (a pending code is decoded when the FC phase returns)",
            "stats": stats}


def fc2_disable(engine) -> None:
    w = engine.wrapper
    codec = w.model.tts_model.audio_codec
    f = getattr(w, "_v2_fc2_steps", None)
    if f is not None:
        cur = w.__dict__.get("_run_fc_async_steps")
        if cur is f:                                         # top of the chain (no recorder)
            if w.__dict__.pop("_v2_fc2_had", False):
                w._run_fc_async_steps = f.__v2_fc2__
            else:
                del w._run_fc_async_steps
            del w._v2_fc2_steps
        elif getattr(cur, "__wrapped_v2__", None) is f and hasattr(engine, "_instrument_fc"):
            w._run_fc_async_steps = f.__v2_fc2__             # directly under the recorder: unwrap, re-instrument
            engine._instrument_fc()
            w.__dict__.pop("_v2_fc2_had", None)
            del w._v2_fc2_steps
        else:                                                # buried under a later wrapper: becomes a pass-through
            f.__v2_fc2_on__ = False
    f = codec.__dict__.get("decode")
    if f is not None and hasattr(f, "__v2_fc2__"):
        if codec.__dict__.pop("_v2_fc2_had", False):
            codec.decode = f.__v2_fc2__
        else:
            del codec.decode
