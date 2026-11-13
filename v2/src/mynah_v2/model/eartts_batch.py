"""EXPERIMENTAL (opt-in): ONE EarTTS step for several sessions (rendezvous kind "eartts").

Why: at B=4 the main-path EarTTS is ~94 GPU-ms per round, 4 sessions x 2 steps x (CFG batch 2) run one after the
other; each step is host-launch-bound (graph per session) — one step over all sessions' rows amortizes it.

How (on top of eartts_graph.FullStepGraph with pooled slots, ``fsg.enable_pool()``):
  * every slot is a view of ONE K/V buffer [L, 2, S, H, Lmax, D]; viewed as [L, 2S, ...] the rows are exactly
    upstream's CFG order (torch.cat([x] * 2): cond rows 0..S-1, uncond rows S..2S-1), so the batched step runs the
    UNCHANGED upstream infer_codes_one_step over S "sessions" (all slots; non-participating rows are dummies);
  * per-row positions: K/V written with a flat ``index_copy_`` at (row, head, pos[row]); position_ids [2S, 1] (RoPE);
    mask [2S, 1, 1, Lb] with the shared bucket Lb = bucket of the longest participant; a row's own position is
    unmasked before the step (a participant's next key; for a dummy row its future position, overwritten by that
    session's own next step before it is ever read — the row always has one finite key, never NaN);
  * conditioning: per-row subword embedding (cached per token id, zero on rows whose mask is False: x + 0 == x);
  * one CUDA graph per (bucket, call signature) over all S rows, captured only through the capture guard (barrier /
    precapture()); otherwise eager.
Not exact vs per-session: batched GEMMs (different reduction tiling) and the RNG — the MoG sampling draws ONE random
tensor for all rows from the generator that is current on the coordinator, so per-session sampled codes are NOT the
B=1 ones (distributionally equivalent; EarTTS is already a non-exact experiment).
"""

from __future__ import annotations

import time

import torch

from mynah_v2.model.capture_guard import GUARD
from mynah_v2.model import eartts_rng
from mynah_v2.model.eartts_graph import StaticKV, bucket_for, NEG


class BatchKV:
    """Duck-typed HF cache over the pooled slots (all S sessions x CFG 2 rows)."""

    _v2_batch_kv = True

    def __init__(self, fsg, s_pos: torch.Tensor):
        nl, h, d = fsg.shape
        S, T = fsg.n_slots, fsg.lmax
        self.h, self.d, self.T, self.S = h, d, T, S
        self.K = fsg.pool_k.view(nl, 2 * S, h, T, d)
        self.V = fsg.pool_v.view(nl, 2 * S, h, T, d)
        self.mask = fsg.pool_mask.view(2 * S, 1, 1, T)
        self.s_pos = s_pos
        dev = s_pos.device
        with torch.inference_mode(False):
            r = torch.arange(2 * S, device=dev)
            self.base = ((r[:, None] * h + torch.arange(h, device=dev)[None, :]) * T)    # [2S, H]
            self.rows = r
            self.pos2 = torch.zeros(2 * S, 1, dtype=torch.long, device=dev)
            self.idx = torch.zeros(2 * S * h, dtype=torch.long, device=dev)
        self.bucket = 64

    def compute(self) -> None:
        """Device-side (inside the graph): per-row positions, flat K/V indices, unmask each row's own position."""
        p2 = torch.cat([self.s_pos, self.s_pos])
        self.pos2.copy_(p2[:, None])
        self.idx.copy_((self.base + p2[:, None]).reshape(-1))
        self.mask.view(2 * self.S, self.T).index_put_((self.rows, p2), torch.zeros((), dtype=self.mask.dtype,
                                                                                    device=self.mask.device))

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.K[layer_idx].view(-1, self.d).index_copy_(0, self.idx,
                                                       key_states.reshape(-1, self.d).to(self.K.dtype))
        self.V[layer_idx].view(-1, self.d).index_copy_(0, self.idx,
                                                       value_states.reshape(-1, self.d).to(self.V.dtype))
        k, v = self.K[layer_idx][:, :, :self.bucket], self.V[layer_idx][:, :, :self.bucket]
        if k.dtype != key_states.dtype:                      # --eartts-dtype bf16w / bf16kv
            k, v = k.to(key_states.dtype), v.to(key_states.dtype)
        return k, v

    def get_seq_length(self, layer_idx=0):
        return 0

    def inject(self) -> dict:
        m = self.mask[..., :self.bucket]
        return {"attention_mask": {"full_attention": m, "sliding_attention": m},
                "cache_position": self.pos2[:1, 0], "position_ids": self.pos2}


ARGS = ("current_subword_id", "prev_subword_id", "current_subword_mask", "prev_audio_tokens")


def step_kwargs(args, kw) -> dict | None:
    """Normalize an infer_codes_one_step call to keyword arguments (None: unexpected signature → pass through)."""
    names = ARGS + ("past_key_values", "guidance_enabled", "generation_config", "ignore_eos_flag_stop", "request_id")
    if len(args) > len(names):
        return None
    d = dict(zip(names, args))
    for k, v in kw.items():
        if k not in names or k in d:
            return None
        d[k] = v
    return d if all(k in d for k in ARGS + ("past_key_values",)) else None


class BatchedEarTTS:
    def __init__(self, fsg, graphs: bool = True):
        self.f = fsg
        fsg.enable_pool()
        self.S = fsg.n_slots
        self.use_graphs = graphs
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self._static_sig = None
        self.stats = {"runs": 0, "rows_hist": {}, "captures": 0, "replays": 0, "eager": 0, "capture_deferred": 0,
                      "capture_failed_recovered": 0, "precaptured": 0, "capture_ms": 0.0, "steps_by_bucket": {}}

    # ------------------------------------------------------------------------------------------------ eligibility
    def group_key(self, d: dict):
        """None if this call cannot join a batched step (it then runs the per-session path), else its stack key."""
        f, kv = self.f, d.get("past_key_values")
        if f.pool_k is None or not isinstance(kv, StaticKV) or f.slots[kv.idx] is not kv.slot:
            return None
        if d.get("request_id") is not None or not d.get("ignore_eos_flag_stop", True) \
                or not d.get("guidance_enabled", True):
            return None
        x = d["current_subword_id"]
        if x.numel() != 1 or kv.t + 1 > f.lmax:
            return None
        return (tuple((tuple(d[a].shape), d[a].dtype, d[a].device.type) for a in ARGS),
                repr(d.get("generation_config")))

    @staticmethod
    def _graphable(x) -> bool:                       # CPU tests replace it
        return x.is_cuda

    # ------------------------------------------------------------------------------------------------ static inputs
    def _ensure_static(self, sig, sub_shape, sub_dtype):
        if self._static_sig == (sig, sub_shape, sub_dtype):
            return
        if self.graphs:
            raise RuntimeError("eartts_batch: call signature changed after capture")
        S, dev = self.S, self.f.device
        with torch.inference_mode(False):
            self.s_in = tuple(torch.zeros((S,) + tuple(shape[1:]), dtype=dt, device=dev) for shape, dt, _ in sig)
            self.s_pos = torch.zeros(S, dtype=torch.long, device=dev)
            self.s_rng = torch.zeros(S, 2, dtype=torch.long, device=dev)   # per-row (seed, step): eartts_rng
            self.f.batch_sub = torch.zeros((2 * S,) + tuple(sub_shape[1:]), dtype=sub_dtype, device=dev)
            self.ridx = torch.zeros(0, dtype=torch.long, device=dev)
        self.kv = BatchKV(self.f, self.s_pos)
        self._static_sig = (sig, sub_shape, sub_dtype)

    def _sub_shape(self, d):
        """Shape/dtype of one CFG-doubled subword embedding, without running the char encoder (it cannot embed an
        arbitrary id: empty char lists raise upstream): [2, 1, hidden] in the embedding module's dtype."""
        es = getattr(self.f.m, "embed_subword", None)
        p = next(es.parameters(), None) if isinstance(es, torch.nn.Module) else None
        return (2, 1, int(self.f.m.hidden_size)), (p.dtype if p is not None else self.f.dtype)

    # ------------------------------------------------------------------------------------------------ the step
    def run(self, kws: list[dict]) -> list[tuple]:
        """One batched EarTTS step for the given per-session calls (same group_key, distinct slots).
        Returns [(codes [1, 1, C], kv)] in the input order, kv.t advanced like the per-session path."""
        f = self.f
        with f.lock:
            kvs = [d["past_key_values"] for d in kws]
            rows = [kv.idx for kv in kvs]
            if len(set(rows)) != len(rows):
                raise RuntimeError("eartts_batch: two calls on the same slot")
            sig = tuple((tuple(d.shape), d.dtype, d.device.type) for d in (kws[0][a] for a in ARGS))
            sub_shape, sub_dtype = self._sub_shape(kws[0])
            self._ensure_static(sig, sub_shape, sub_dtype)
            kw = {k: kws[0][k] for k in ("guidance_enabled", "generation_config", "ignore_eos_flag_stop") if k in kws[0]}
            # the one host read: token ids + mask flags of all participants
            flat = torch.stack([v for d in kws for v in (d["current_subword_id"].reshape(-1)[0],
                                                         d["current_subword_mask"].reshape(-1)[0].long())]).tolist()
            dev = f.device
            ridx = torch.tensor(rows, dtype=torch.long, device=dev)
            for s_, a in zip(self.s_in, ARGS):
                s_.zero_()
                s_.index_copy_(0, ridx, torch.cat([d[a] for d in kws], 0).to(s_.dtype))
            # per-row subword embedding (cond row r, uncond row S + r); zero where the mask is False / dummy rows
            f.batch_sub.zero_()
            if getattr(f.m, "embed_subword", None) is not None:
                sel = [(r, f.sub_embed(int(flat[2 * j]), d["current_subword_id"], d["current_subword_mask"], True))
                       for j, (r, d) in enumerate(zip(rows, kws)) if flat[2 * j + 1]]
                if sel:
                    sr = torch.tensor([r for r, _ in sel] + [self.S + r for r, _ in sel], dtype=torch.long, device=dev)
                    ev = torch.cat([torch.stack([e[0] for _, e in sel]), torch.stack([e[1] for _, e in sel])], 0)
                    f.batch_sub.index_copy_(0, sr, ev.to(f.batch_sub.dtype))
            # positions: participants at their next position; every other slot at its own next position (dummy row)
            pos = [0] * self.S
            for i in range(self.S):
                t = f.live_t(i)
                pos[i] = min(t, f.lmax - 1) if t is not None else 0
            for kv in kvs:
                pos[kv.idx] = kv.t
                kv.bucket = bucket_for(kv.t + 1, f.lmax)
            self.s_pos.copy_(torch.tensor(pos, dtype=torch.long))
            if eartts_rng.STATE["enabled"]:               # each row: its own session's (seed, step); dummies (0, 0)
                rng = [[0, 0] for _ in range(self.S)]
                for kv in kvs:
                    rng[kv.idx] = [getattr(kv, "_v2_rng_seed", 0), getattr(kv, "_v2_rng_step", 0)]
                self.s_rng.copy_(torch.tensor(rng, dtype=torch.long))
            bucket = max(kv.bucket for kv in kvs)
            self.kv.bucket = bucket
            st = self.stats
            st["runs"] += 1
            st["rows_hist"][len(kws)] = st["rows_hist"].get(len(kws), 0) + 1
            st["steps_by_bucket"][str(bucket)] = st["steps_by_bucket"].get(str(bucket), 0) + 1
            key = (bucket, sig, repr(kw.get("generation_config")))
            codes = self._dispatch(key, kw)
            out = []
            for kv, r in zip(kvs, rows):                  # (the per-session path re-fills slot.pos in prepare())
                kv.t += 1
                kv._v2_rng_step = getattr(kv, "_v2_rng_step", 0) + 1
                out.append((codes[r:r + 1].clone(), kv))
            return out

    def _forward(self, kw):
        f = self.f
        self.kv.compute()
        f.mask_any = "batch"
        try:
            with eartts_rng.rows(self.s_rng):             # per-session noise rows (no-op unless enabled)
                codes, _ = f.eager_step(*self.s_in, self.kv, **kw)
        finally:
            f.mask_any = None
        return codes

    def _dispatch(self, key, kw):
        if not (self.use_graphs and self._graphable(self.s_in[0])):
            self.stats["eager"] += 1
            return self._forward(kw)
        g = self.graphs.get(key)
        if g is not None:
            g[0].replay()
            self.stats["replays"] += 1
            return g[1]
        if not GUARD.may_capture("eartts_batch", key):
            self.stats["capture_deferred"] += 1
            self.stats["eager"] += 1
            return self._forward(kw)
        return self._capture(key, kw)

    def _capture(self, key, kw):
        with GUARD.capturing():
            t0 = time.perf_counter()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                   # warm-up = this step's real result
                warm = self._forward(kw)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            prev_stream = torch.cuda.current_stream()
            try:
                with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                    static_out = self._forward(kw)
            except Exception as e:  # noqa: BLE001
                self.stats.setdefault("first_capture_error", f"{type(e).__name__}: {e}")
                if GUARD.on_failure("eartts_batch", key, e, stream=prev_stream):
                    self.stats["capture_failed_recovered"] += 1
                    return warm
                raise RuntimeError(f"eartts_batch capture failed (bucket {key[0]}): {type(e).__name__}: {e}") from e
            self.graphs[key] = (g, static_out)
            self.stats["captures"] += 1
            self.stats["capture_ms"] += (time.perf_counter() - t0) * 1e3
            return warm

    # ------------------------------------------------------------------------------------------------ precapture
    def precapture(self, buckets_list=None) -> dict:
        """Capture the batched graph for every bucket at a QUIET point before sessions open (all rows dummies; slots
        reset afterwards). Signature: the first real per-session step's (solo phase) else the upstream default."""
        from mynah_v2.model.capture_guard import buckets, is_oom, soft_cleanup
        f = self.f
        why = GUARD.quiet_reason()
        if why is not None:
            GUARD.note_precapture("eartts_batch", 0, 0.0, skipped=why)
            return {"skipped": why}
        if any(s.busy for s in f.slots):
            raise RuntimeError("eartts_batch precapture: a slot is in use")
        args, kw, _ = f._precapture_args(0)
        d = dict(zip(ARGS, args))
        sig = tuple((tuple(d[a].shape), d[a].dtype, d[a].device.type) for a in ARGS)
        sub_shape, sub_dtype = self._sub_shape(d)
        bl = buckets_list or buckets(f.lmax)
        t0 = time.perf_counter()
        n, stop = 0, None
        with f.lock, torch.no_grad():
            self._ensure_static(sig, sub_shape, sub_dtype)
            for s_ in self.s_in:
                s_.zero_()
            f.batch_sub.zero_()
            self.s_pos.zero_()
            try:
                for b in bl:
                    key = (b, sig, repr(kw.get("generation_config")))
                    if key in self.graphs:
                        continue
                    self.kv.bucket = b
                    try:
                        self._capture(key, kw)
                    except Exception as e:  # noqa: BLE001
                        if not is_oom(e):
                            raise
                        stop = f"oom at bucket {b}: {str(e).splitlines()[0][:160]}"
                        break
                    if key not in self.graphs:
                        GUARD.disabled.discard(("eartts_batch", key))
                        stop = f"capture failed at bucket {b}"
                        break
                    n += 1
            finally:                                      # dummy writes at position 0 of every (free) slot: reset
                f.pool_k.zero_()
                f.pool_v.zero_()
                f.pool_mask.fill_(NEG)
        if stop:
            soft_cleanup()
        ms = (time.perf_counter() - t0) * 1e3
        self.stats["precaptured"] += n
        GUARD.note_precapture("eartts_batch", n, ms, stopped=stop)
        return {"precaptured": n, "buckets": bl, "ms": round(ms, 1), "stopped": stop,
                "sig": "learned" if f._sig is not None else "upstream-default"}
