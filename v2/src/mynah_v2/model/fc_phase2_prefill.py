"""EXPERIMENT (opt-in, perf experiment "fc_phase2_prefill"): FC phase 2 as ONE Thinker continuation forward.

Phase 2 of NeMo's non-blocking FC forces the N `<TOOL_RESPONSE>` tokens one per step and lets the model predict the
token after them (EOTR). Each forced step runs a full Thinker forward whose outputs are all discarded (text/ASR forced to
PAD, function prediction overwritten by the forced token; wrapper ~:1576-1593). Only the last, natural step's function
argmax is used. The inputs of all N+1 steps are known in advance (agent PAD, the silence embedding, the function token
of the previous step), so the N+1 positions can be consumed in one forward with the same final state.

Two pieces:
- ``continuation_forward(llm, embeds, cache)``: NemotronH over T > 1 new positions on an existing hybrid cache,
  equivalent to T single-frame decode steps. The large GEMMs (Mamba in/out projections, MLPs, attention projections)
  run once on T rows; the Mamba recurrence (conv state + SSM state) is advanced token by token with the SAME kernels the
  decode step uses (``causal_conv1d_update``, ``selective_state_update``); attention appends T keys and uses an explicit
  bottom-right causal mask (upstream's ``is_causal`` is top-left, wrong for a continuation; no rotary in NemotronH).
  Not bit-exact by construction (GEMM M=T vs M=1, attention backend with a mask): measured, not assumed.
- a deferred Thinker interface installed only for the duration of a phase-2 ``_run_fc_async_steps`` call, only on the
  calling thread: while forced tokens are queued (``fc_state["forced_function_tokens"]`` non-empty) it records the
  fused input and returns PAD predictions without touching the cache; at the natural step it runs one continuation
  forward over all recorded inputs + the current one and returns the real function prediction. The upstream loop, its
  buffers, ``fc_state`` and return value are untouched. An early exit (abort / quit) flushes the recorded inputs so the
  cache advances by exactly the steps the loop counted.

Scope of this prototype: B=1, the upstream hybrid cache (falls back to the sequential path when the cache is converted
by ``thinker_graph`` or when no NemotronH kernels are available), any number of positions per chunk (W8's batched
kernel covers ≤ 16 rows: ``chunk`` splits the continuation). Off by default.
"""

from __future__ import annotations

import sys
import threading

import torch


# ------------------------------------------------------------------------------------------------ continuation forward
def _mamba_continue(m, x: torch.Tensor, cache, mod) -> torch.Tensor:
    proj = m.in_proj(x)                                             # [1, T, P]: one GEMM for the T positions
    T = x.shape[1]
    d_mlp = (proj.shape[-1] - 2 * m.intermediate_size - 2 * m.n_groups * m.ssm_state_size - m.num_heads) // 2
    gs = m.n_groups * m.ssm_state_size
    A = -torch.exp(m.A_log.float())
    A = A[:, None, ...][:, :, None].expand(-1, m.head_dim, m.ssm_state_size).to(dtype=torch.float32)
    dt_bias = m.dt_bias[:, None, ...].expand(-1, m.head_dim)
    D = m.D[:, None, ...].expand(-1, m.head_dim)
    ys, gates = [], []
    for t in range(T):                                              # = the decode branch, one position at a time
        _, _, gate, xbc, dt = proj[:, t].split([d_mlp, d_mlp, m.intermediate_size, m.conv_dim, m.num_heads], dim=-1)
        xbc = mod.causal_conv1d_update(xbc, cache.conv_states[m.layer_idx], m.conv1d.weight.squeeze(1), m.conv1d.bias,
                                       m.activation)
        hs, B, C = torch.split(xbc, [m.intermediate_size, gs, gs], dim=-1)
        y = mod.selective_state_update(
            cache.ssm_states[m.layer_idx], hs.view(1, m.num_heads, m.head_dim),
            dt[:, :, None].expand(-1, -1, m.head_dim), A, B.view(1, m.n_groups, -1), C.view(1, m.n_groups, -1), D,
            z=None, dt_bias=dt_bias, dt_softplus=True)
        ys.append(y.view(1, m.num_heads * m.head_dim))
        gates.append(gate)
    y = m.norm(torch.cat(ys, 0), torch.cat(gates, 0))               # gated RMSNorm per position ([T, D] rows)
    return m.out_proj(y)[None]                                      # [1, T, H]: one GEMM


def _attention_continue(a, x: torch.Tensor, cache, mod) -> torch.Tensor:
    bsz, T, _ = x.shape
    q = a.q_proj(x).view(bsz, T, a.num_heads, a.head_dim).transpose(1, 2)
    k = a.k_proj(x).view(bsz, T, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    v = a.v_proj(x).view(bsz, T, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    k, v = cache.update(k, v, a.layer_idx)                          # appends: [B, kvh, P + T, hd]
    k = mod.repeat_kv(k, a.num_key_value_groups)
    v = mod.repeat_kv(v, a.num_key_value_groups)
    L = k.shape[-2]
    # query i (absolute position L - T + i) sees keys 0 .. L - T + i: bottom-right causal
    mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(diagonal=L - T)
    o = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                                         attn_mask=mask)
    o = o.transpose(1, 2).contiguous().view(bsz, T, a.num_heads * a.head_dim)
    return a.o_proj(o)


def continuation_forward(llm, embeds: torch.Tensor, cache, chunk: int | None = None) -> torch.Tensor:
    """NemotronH backbone over ``embeds`` [1, T, H] continuing ``cache`` (position ``cache._v2_pos``). Updates the cache
    in place (conv / SSM states, appended K/V, ``_v2_pos += T``) and returns the final hidden states [1, T, H]
    (after ``norm_f``), i.e. what T single-frame decode steps would have produced."""
    if embeds.shape[0] != 1:
        raise ValueError("continuation_forward is B=1")
    if chunk is not None and embeds.shape[1] > chunk:
        outs = [continuation_forward(llm, embeds[:, i:i + chunk], cache) for i in range(0, embeds.shape[1], chunk)]
        return torch.cat(outs, 1)
    mod = sys.modules[type(llm).__module__]
    pos = int(getattr(cache, "_v2_pos", 0))
    if pos <= 0:
        raise ValueError("continuation_forward continues a prefilled cache (position > 0)")
    h = embeds
    for blk in llm.layers:
        residual = h
        y = blk.norm(h.to(dtype=blk.norm.weight.dtype))
        if getattr(blk, "residual_in_fp32", False):
            residual = residual.to(torch.float32)
        if blk.block_type == "mamba":
            y = _mamba_continue(blk.mixer, y, cache, mod)
        elif blk.block_type == "attention":
            y = _attention_continue(blk.mixer, y, cache, mod)
        elif blk.block_type == "mlp":
            y = blk.mixer(y)
        else:
            raise ValueError(f"unknown NemotronH block type {blk.block_type!r}")
        h = residual + y
    h = llm.norm_f(h)
    cache._v2_pos = pos + embeds.shape[1]
    return h


# ------------------------------------------------------------------------------------------------ deferred interface
class DeferredPhase2:
    """Stands in for ``wrapper.model_llm_interface`` during one phase-2 call on one thread (see module doc)."""

    def __init__(self, inner, stt, fc_state: dict, owner: int, stats: dict, chunk: int | None = None):
        self.inner, self.stt, self.fc_state, self.owner, self.stats, self.chunk = inner, stt, fc_state, owner, stats, chunk
        self.pending: list[torch.Tensor] = []
        self.cache = None

    def __getattr__(self, name):                                   # everything else: the real interface
        return getattr(self.inner, name)

    def _pad(self, like: torch.Tensor) -> torch.Tensor:
        return torch.full((like.shape[0],), self.stt.text_pad_id, dtype=torch.long, device=like.device)

    def flush(self) -> torch.Tensor | None:
        if not self.pending:
            return None
        x = torch.cat(self.pending, 1)
        self.pending = []
        with torch.no_grad():
            h = continuation_forward(self.stt.llm, x, self.cache, chunk=self.chunk)
        self.stats["prefill_calls"] = self.stats.get("prefill_calls", 0) + 1
        self.stats["prefill_positions"] = self.stats.get("prefill_positions", 0) + int(x.shape[1])
        return h

    def __call__(self, input_embeds, cache=None, **kw):
        if threading.get_ident() != self.owner or cache is None or input_embeds.shape[1] != 1:
            return self.inner(input_embeds, cache=cache, **kw)
        if self.fc_state.get("forced_function_tokens"):               # a forced step: its outputs are discarded
            self.cache = cache
            self.pending.append(input_embeds.detach())
            pad = self._pad(input_embeds)
            return {"predicted_token": pad, "asr_predicted_token": pad, "function_predicted_token": pad, "cache": cache}
        if not self.pending:                                        # nothing deferred: the normal path, unchanged
            return self.inner(input_embeds, cache=cache, **kw)
        self.cache = cache
        self.pending.append(input_embeds.detach())
        h = self.flush()
        with torch.no_grad():
            fl = self.stt.function_head(h[:, -1:])
        pad = self._pad(input_embeds)
        return {"predicted_token": pad, "asr_predicted_token": pad,
                "function_predicted_token": fl[:, -1].argmax(dim=-1), "cache": cache}


def _wrap(inner_steps, engine, stats: dict, chunk: int | None):
    w = engine.wrapper

    def run(*a, **kw):
        fs = kw.get("fc_state") or {}
        cache = kw.get("dynamic_cache")
        if (not getattr(run, "__v2_on__", True) or not kw.get("tool_response_text") or fs.get("active")
                or cache is None or getattr(cache, "_v2_graph", None) is not None
                or "causal_conv1d_update" not in vars(sys.modules[type(w.model.stt_model.llm).__module__])):
            stats["passthrough"] = stats.get("passthrough", 0) + 1
            return inner_steps(*a, **kw)
        real = w.model_llm_interface
        d = DeferredPhase2(real, w.model.stt_model, fs, threading.get_ident(), stats, chunk)
        w.model_llm_interface = d
        try:
            out = inner_steps(*a, **kw)
        finally:
            w.model_llm_interface = real
            if d.pending:                                           # early exit: keep the cache = counted steps
                d.flush()
                stats["flushed_on_exit"] = stats.get("flushed_on_exit", 0) + 1
        stats["phase2_calls"] = stats.get("phase2_calls", 0) + 1
        return out
    run.__v2_p2pre__ = inner_steps
    return run


def enable(engine, chunk: int | None = None) -> dict:
    w = engine.wrapper
    stats = getattr(engine, "_v2_p2pre_stats", None)
    if stats is None:
        stats = engine._v2_p2pre_stats = {}
    if getattr(w, "_v2_p2pre", None) is None:
        cur = w._run_fc_async_steps
        instrumented = hasattr(engine, "_instrument_fc") and hasattr(cur, "__wrapped_v2__")
        inner = cur.__wrapped_v2__ if instrumented else cur
        f = _wrap(inner, engine, stats, chunk)
        w._v2_p2pre = f
        w._run_fc_async_steps = f
        if instrumented:
            engine._instrument_fc()
    w._v2_p2pre.__v2_on__ = True
    return {"note": "FC phase 2 as one Thinker continuation forward (deferred forced steps)", "chunk": chunk,
            "stats": stats}


def disable(engine) -> None:
    f = getattr(engine.wrapper, "_v2_p2pre", None)
    if f is not None:
        f.__v2_on__ = False
