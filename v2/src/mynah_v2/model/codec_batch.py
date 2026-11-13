"""Batched streaming codec decode across sessions (opt-in, used by RendezvousEngine(batch_codec=True)).

Every main-path session decodes its new codes once per chunk (wrapper :2663, ``audio_codec.decode(codes [1, k, Q],
len [1], cache=<its CausalConv1dCache>)``). The codec is a causal conv net whose streaming state is a dict
layer -> tensor [1, C, pad] per session (ear_tts_vae_codec.py:440-500), so n requests with the same k and the same
cache layout are one call on codes [n, k, Q] with a fresh cache holding each layer's tensors concatenated on dim 0;
afterwards every session's cache gets its row back (``[i:i+1]``). Same kernels at batch n instead of 1: NOT
bit-identical (measured on the real codec, GB10: max |diff| 3.5e-6 on audio rms 0.05, CA1/CA4); audio-only — codec
output never feeds the model (EarTTS consumes codes), so tokens are unaffected by construction.
Requests that cannot be stacked (flush, no cache, batch != 1, different k / layout / modes) run alone, unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

_PARAMS = ("code", "code_len", "constrain_value_range", "cache", "flush")
_DEFAULTS = {"code_len": None, "constrain_value_range": True, "cache": None, "flush": False}


def decode_kwargs(args: tuple, kw: dict) -> Optional[dict]:
    """RVQVAEModel.decode(code, code_len=None, constrain_value_range=True, cache=None, flush=False) -> kwargs dict,
    or None for an unexpected signature."""
    if len(args) > len(_PARAMS) or any(k not in _PARAMS for k in kw):
        return None
    d = dict(_DEFAULTS)
    for name, v in zip(_PARAMS, args):
        if name in kw:
            return None
        d[name] = v
    d.update(kw)
    return d if "code" in d else None


def stack_key(d: dict) -> Optional[tuple]:
    """Requests with equal keys can share one batched call; None = must run alone."""
    code, cache = d["code"], d["cache"]
    if cache is None or d["flush"] or not isinstance(code, torch.Tensor) or code.dim() != 3 or code.shape[0] != 1:
        return None
    if d["code_len"] is not None and (not isinstance(d["code_len"], torch.Tensor) or d["code_len"].numel() != 1):
        return None
    layout = []
    for k, t in sorted(cache.cache.items(), key=lambda kv: str(kv[0])):
        if t.dim() != 3 or t.shape[0] != 1:
            return None
        layout.append((str(k), tuple(t.shape[1:]), t.dtype, t.device))
    return (tuple(code.shape[1:]), code.dtype, code.device, bool(d["constrain_value_range"]),
            d["code_len"] is None, type(cache), tuple(layout))


def batched_decode(inner: Callable, reqs: list[dict]) -> list[Any]:
    """One ``inner`` decode for n >= 2 stackable requests (same stack_key); returns per-request (wav, wav_len) and
    writes every session's cache rows back."""
    caches = [d["cache"] for d in reqs]
    bc = type(caches[0])()
    for k in caches[0].cache:
        bc.cache[k] = torch.cat([c.cache[k] for c in caches], dim=0)
    code = torch.cat([d["code"] for d in reqs], dim=0)
    lens = None if reqs[0]["code_len"] is None else torch.cat([d["code_len"].reshape(1) for d in reqs], dim=0)
    wav, wav_len = inner(code, lens, constrain_value_range=reqs[0]["constrain_value_range"], cache=bc, flush=False)
    for i, c in enumerate(caches):
        c.cache.clear()
        c.cache.update({k: t[i:i + 1] for k, t in bc.cache.items()})
    return [(wav[i:i + 1], None if wav_len is None else wav_len[i:i + 1]) for i in range(len(reqs))]
