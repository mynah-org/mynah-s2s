"""The streaming ASR of several sessions in one forward: one cache-aware encoder call for every session's 320 ms
chunk, one batched greedy RNN-T loop, per-row language prompt, per-session caches.

Each session keeps its own encoder caches (attention K/V per layer, causal-conv states) and RNN-T predictor state.
For a step, the sessions' caches are concatenated along the batch axis, the encoder runs once on the stacked mel
chunks, and the updated caches are split back. Sessions can be stacked when their caches have the same shape and
mask: once the attention window is full (after `sliding_window` frames, ~5.7 s) every session qualifies, whatever
its absolute position (relative positions, chunk-aligned windows); before that, sessions are grouped by their
exact length (sessions that started at the same hop share a group). Checked against per-session streaming
(lab/ears_batch_check.py): same transcripts, last layer within float tolerance.

The greedy RNN-T loop runs for all rows at once with no host sync per step (a row that emits blank or hits
`max_symbols_per_step` advances its frame; the loop ends when every row has consumed its frames).
"""

from __future__ import annotations

import copy

import numpy as np
import torch


def _group_key(pair) -> tuple:
    if pair.chunks == 0:
        return ("first",)
    win = pair.enc.config.sliding_window
    n = pair.pkv.layers[0].cumulative_length if pair.pkv.layers and pair.pkv.layers[0].is_initialized else 0
    return ("full",) if n >= win else ("len", n)


def _merge_kv(pairs):
    from transformers import DynamicCache
    out = DynamicCache(config=pairs[0].enc.config)
    for li, layer in enumerate(out.layers):
        src = [p.pkv.layers[li] for p in pairs]
        if not src[0].is_initialized:
            continue
        layer.lazy_initialization(src[0].keys, src[0].values)
        layer.keys = torch.cat([s.keys for s in src], 0)
        layer.values = torch.cat([s.values for s in src], 0)
        layer.cumulative_length = max(s.cumulative_length for s in src)
    return out


def _merge_pc(pairs):
    from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import \
        NemotronAsrStreamingEncoderCausalConvPaddingCache as PC
    out = PC()
    for key, layer in pairs[0].pc.layers.items():
        m = copy.copy(layer)
        m.cache = torch.cat([p.pc.layers[key].cache for p in pairs], 0)
        out.layers[key] = m
    return out


def _split_pc(batched, pairs):
    for key, layer in batched.layers.items():
        for i, p in enumerate(pairs):
            d = copy.copy(layer)
            d.cache = layer.cache[i:i + 1].clone()
            p.pc.layers[key] = d


@torch.no_grad()
def stream_chunks(pairs: list):
    """One chunk for each of ``pairs`` (StreamChannelPair, user row; all ready) -> list of last-layer (1, T, 1024)."""
    out = [None] * len(pairs)
    groups: dict = {}
    for i, p in enumerate(pairs):
        groups.setdefault(_group_key(p), []).append(i)
    for key, idx in groups.items():
        ps = [pairs[i] for i in idx]
        if key == ("first",) or len(ps) == 1:
            for i, p in zip(idx, ps):
                out[i] = p.chunk()[1][:1]
            continue
        p0 = ps[0]
        lo = [p.mel_done * p.hop_len - p.n_fft // 2 for p in ps]
        xs = [p.buf[0, l - p.buf0: l - p.buf0 + p.mel_per * p.hop_len + p.win] for p, l in zip(ps, lo)]
        f = p0.fe(xs, sampling_rate=16000, center=False, return_tensors="pt")
        feats = f["input_features"][:, :p0.mel_per].to(p0.dev)
        cum = [p.pkv.layers[0].cumulative_length for p in ps]
        pkv, pc = _merge_kv(ps), _merge_pc(ps)
        o = p0.enc(input_features=feats, past_key_values=pkv, padding_cache=pc, use_cache=True,
                   num_lookahead_tokens=p0.k, output_attention_mask=False)
        T = o.last_hidden_state.shape[1]
        for li, layer in enumerate(o.past_key_values.layers):
            for j, p in enumerate(ps):
                dst = p.pkv.layers[li]
                dst.keys = layer.keys[j:j + 1]
                dst.values = layer.values[j:j + 1]
                dst.cumulative_length = cum[j] + T
        _split_pc(o.padding_cache, ps)
        for j, (i, p) in enumerate(zip(idx, ps)):
            out[i] = o.last_hidden_state[j:j + 1]
            p.mel_done += p.mel_per
            p.chunks += 1
            keep_from = p.mel_done * p.hop_len - p.n_fft // 2
            if keep_from > p.buf0:
                p.buf = p.buf[:, keep_from - p.buf0:]
                p.buf0 = keep_from
    return out


@torch.no_grad()
def rnnt_batch(rnnts: list, lasts: list) -> list:
    """Greedy RNN-T for several sessions' frames at once. rnnts: RnntStream per row (own predictor state and
    language prompt); lasts: (1, T, 1024) last layer per row (equal T). -> [[(frame offset, token id)]] per row."""
    r0 = rnnts[0]
    m, dev = r0.m, r0.dev
    B = len(rnnts)
    last = torch.cat(lasts, 0)
    T = last.shape[1]
    oh = torch.nn.functional.one_hot(torch.tensor([r.prompt_id for r in rnnts], device=dev),
                                     num_classes=r0.num_prompts).to(last.dtype)[:, None, :].expand(-1, T, -1)
    enc = m.encoder_projector(m.prompt_projector(torch.cat([last, oh], -1)))          # (B, T, H)
    d = torch.cat([r.d for r in rnnts], 0)                                               # (B, H)
    h = torch.cat([r.hc[0] for r in rnnts], 1)
    c = torch.cat([r.hc[1] for r in rnnts], 1)
    dec = m.decoder
    ar = torch.arange(B, device=dev)
    idx = torch.zeros(B, dtype=torch.long, device=dev)
    sym = torch.zeros(B, dtype=torch.long, device=dev)
    toks, frames, valid = [], [], []
    blank, max_sym = r0.blank, r0.max_sym
    for step in range(T * (max_sym + 1)):
        done = idx >= T
        e = enc[ar, idx.clamp(max=T - 1)]
        tok = m.joint(decoder_hidden_states=d, encoder_hidden_states=e).argmax(-1)
        is_blank = (tok == blank) | done
        emit = ~is_blank
        toks.append(tok)
        frames.append(idx.clone())
        valid.append(emit)
        sym = torch.where(is_blank, torch.zeros_like(sym), sym + 1)
        force = sym >= max_sym
        idx = idx + (is_blank | force).long()
        sym = torch.where(force, torch.zeros_like(sym), sym)
        o, (h2, c2) = dec.lstm(dec.embedding(tok[:, None]), (h, c))
        d2 = dec.decoder_projector(o)[:, 0]
        u = emit[:, None]
        d = torch.where(u, d2, d)
        h = torch.where(u[None], h2, h)
        c = torch.where(u[None], c2, c)
        if (step + 1) % 4 == 0 and bool((idx >= T).all()):
            break
    for i, r in enumerate(rnnts):
        r.d, r.hc = d[i:i + 1], (h[:, i:i + 1].contiguous(), c[:, i:i + 1].contiguous())
    tk = torch.stack(toks, 1).tolist()
    fr = torch.stack(frames, 1).tolist()
    va = torch.stack(valid, 1).tolist()
    return [[(fr[i][s], tk[i][s]) for s in range(len(tk[i])) if va[i][s]] for i in range(B)]
