"""EXPERIMENTAL: batched Thinker decode — one forward with a real batch dimension over R session rows.

Implements ``runtime.batch_contract.ThinkerBatchBackend`` (shared weights, independent per-session recurrent state).
Generalises the milestone-1 prototype (scripts/thinker_b2.py) on top of ``ThinkerGraph`` (subclassed, not modified):

  * per-row state in backend-owned static buffers: attention K/V [n_attn, R, kvh, lmax, hd], additive mask
    [R, 1, 1, lmax] (0 valid, dtype-min empty), device position ``posv`` [R] + host position, Mamba conv/SSM [R, ...]
    per Mamba layer (the upstream decode kernels are batched and update the state in place);
  * K/V written at a PER-ROW position (flat ``index_copy_`` at row*kvh*lmax + head*lmax + pos), PER-ROW mask; NemotronH
    attention has no RoPE, so positions only live in K/V + mask. One shared bucket = bucket_for(max pos + 1, lmax);
    shorter rows are covered by their mask (I5);
  * Mamba ``cache_position[0] > 0`` is answered on the host (``mamba_host_pos``, required) from the combined cache's
    ``_v2_pos`` (any value > 0: every attached row is past its prefill).

Inactive rows — compaction, not torch.where. The rows stepped together always occupy the leading physical rows
[0, n); the forward runs on prefix VIEWS (``K[j][:n]``, ``conv[j][:n]``), so rows >= n are neither read nor written
(I1). Joining appends at row n_attached; leaving moves the last attached row into the hole (valid K/V prefix + Mamba
rows + mask row); a step over a subset swaps participants to the front. Trade-off: the alternative (run all R rows and
restore inactive rows with torch.where) pays, EVERY step, the full Mamba state read+write of every inactive row
(9B: ~ n_mamba x nheads x hd x dstate x 4 B ≈ 100+ MB per row) plus a backup copy of it, plus the K/V/attention work
over the shared bucket; compaction pays a one-off row copy only when the set of stepped sessions changes (join/leave/
pause), and nothing for idle rows. Cost: graphs are keyed by (n_rows, bucket) — at most max_rows x n_buckets graphs.

CUDA graphs (optional): one per (n_rows, bucket, dtype, H), private memory pool, ``capture_error_mode="thread_local"``.
They read/write only backend-owned buffers (static input, K/V, mask, posv, kv_idx, Mamba rows) — slot reuse never
leaves a dangling per-session pointer (I4). Mask/position bookkeeping (mask scatter at posv, kv_idx, posv += 1) runs
outside the graph.

Rows reaching lmax: before a step, a session whose next position would not fit is detached to its own B=1 cache
(``_v2_past_lmax``), its frame runs through the upstream eager path, and the cache is handed back in ``evicted``; the
other rows are not touched. B=1 path: by default (``b1_graph=False``) sessions that are NOT attached keep the
upstream eager forward exactly; ``b1_graph=True`` also routes them through ThinkerGraph's B=1 CUDA graph. Not exact by
construction (masked padded attention window, batched GEMMs) → parity is measured (tests/gpu/test_thinker_batch_gpu.py).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping

import torch

from mynah_v2.model.capture_guard import BUDGET, GUARD, free_gib, is_oom, precapture_buckets, soft_cleanup
from mynah_v2.model.eartts_graph import bucket_for, sync_watch
from mynah_v2.model.thinker_graph import ThinkerGraph
from mynah_v2.runtime.batch_contract import BackendMemory, SessionId, ThinkerOut

HEADS = ("text_logits", "function_logits", "asr_logits")


# ------------------------------------------------------------------------------------------------ host bookkeeping
class RowTable:
    """Pure host bookkeeping: session id <-> physical row, host positions. Attached rows are always compact
    [0, n_attached); ``arrange`` brings a subset to the front. No tensors (CPU-tested)."""

    def __init__(self, max_rows: int):
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        self.max_rows = max_rows
        self.row_of: dict[SessionId, int] = {}
        self.sid_at: list[SessionId | None] = [None] * max_rows
        self.pos: dict[SessionId, int] = {}

    @property
    def n(self) -> int:
        return len(self.row_of)

    def attach(self, sid: SessionId, pos: int) -> int:
        if sid in self.row_of:
            raise KeyError(f"session {sid!r} already attached")
        if self.n >= self.max_rows:
            raise RuntimeError(f"batch full ({self.max_rows} rows)")
        if pos < 1:
            raise ValueError(f"attach needs a prefilled cache (position {pos})")
        r = self.n
        self.row_of[sid], self.sid_at[r], self.pos[sid] = r, sid, int(pos)
        return r

    def release(self, sid: SessionId) -> tuple[int, int | None]:
        """Free ``sid``'s row. Returns (row, src): when src is not None the last attached row ``src`` must be moved
        into ``row`` (already reflected in the table)."""
        r = self.row_of.pop(sid)
        del self.pos[sid]
        last = self.n                                    # index of the old last attached row
        self.sid_at[r] = None
        if r == last:
            return r, None
        moved = self.sid_at[last]
        self.sid_at[r], self.sid_at[last] = moved, None
        self.row_of[moved] = r
        return r, last

    def swap(self, a: int, b: int) -> None:
        sa, sb = self.sid_at[a], self.sid_at[b]
        self.sid_at[a], self.sid_at[b] = sb, sa
        if sb is not None:
            self.row_of[sb] = a
        if sa is not None:
            self.row_of[sa] = b

    def arrange(self, sids) -> list[tuple[int, int]]:
        """Swap rows so that ``sids`` occupy rows [0, len(sids)). Returns the swaps (applied to the table) that the
        tensor buffers must replay in order."""
        want = set(sids)
        for s in want:
            if s not in self.row_of:
                raise KeyError(f"session {s!r} not attached")
        n = len(want)
        outside = sorted(self.row_of[s] for s in want if self.row_of[s] >= n)
        holes = [r for r in range(n) if self.sid_at[r] not in want]
        assert len(outside) == len(holes)
        swaps = list(zip(holes, outside))
        for a, b in swaps:
            self.swap(a, b)
        return swaps

    def order(self, n: int) -> list[SessionId]:
        return list(self.sid_at[:n])

    def advance(self, sids) -> None:
        for s in sids:
            self.pos[s] += 1

    def bucket(self, sids, lmax: int) -> int:
        return bucket_for(max(self.pos[s] for s in sids) + 1, lmax)


# ------------------------------------------------------------------------------------------------ backend
class BatchedThinker(ThinkerGraph):
    """``ThinkerBatchBackend`` over ``max_rows`` rows. See the module docstring."""

    def __init__(self, engine, max_rows: int = 2, lmax: int = 16384, graphs: bool = True, b1_graph: bool = False,
                 precapture: bool = False):
        stt = engine.wrapper.model.stt_model
        if b1_graph:
            # Round 8 (Ada, 2026-11-12): the B=1-graph routing of non-attached sessions did not replay (stats 0) — not
            # verified, and not needed by the vertical path (non-attached sessions stay upstream eager). Refused
            # explicitly instead of shipping an unverified branch.
            raise NotImplementedError("BatchedThinker(b1_graph=True) is disabled (unverified on GPU)")
        if isinstance(getattr(stt.forward, "__self__", None), ThinkerGraph):
            raise RuntimeError("a ThinkerGraph is already installed (disable the thinker_graph experiment first; "
                               "it cannot be stacked with BatchedThinker)")
        super().__init__(engine, lmax=lmax)
        if not b1_graph:
            self.stt.forward = self._eager_forward        # non-attached sessions: upstream eager, as without us
        self.max_rows = max_rows
        self.use_graphs = graphs
        # precapture=True (capture_guard): at the first attach (buffers just allocated, no session state in them yet,
        # coordinator thread = quiet point) capture every (n_rows <= max_rows, bucket <= lmax) key, so no lazy capture
        # happens while sessions run concurrently. Frame dtype/H: the last prefill seen (else model dtype/hidden size).
        self.precapture_on = precapture
        self._frame_sig = None
        self.table = RowTable(max_rows)
        self._attn_j = {li: j for j, li in enumerate(self.attn)}
        self.bK = self.bV = self.bmask = None
        self.bconv = self.bssm = None
        self.posv = self.kv_base = self.kv_idx = None
        self.bbucket = 64
        self._template = None
        self._ccs: dict[int, Any] = {}
        self._caches: dict[SessionId, Any] = {}          # sid -> the caller's cache object (refilled on detach)
        self._mamba_layout: dict[tuple, tuple] = {}      # (sid, li) -> (shape, stride) of the attached B=1 state
        self.bgraphs: dict = {}
        self.bpool = torch.cuda.graph_pool_handle()
        self._pool_delta = 0
        self.evicted: dict[SessionId, Any] = {}         # sessions pushed out at lmax: caller continues them eagerly
        self._closed = False
        self.bstats = {"attach": 0, "detach": 0, "moves": 0, "swaps": 0, "steps": 0, "eager_steps": 0,
                       "captures": 0, "replays": 0, "evictions": 0, "capture_ms": 0.0, "steps_by_n": {},
                       "capture_deferred": 0, "capture_failed_recovered": 0, "precaptured": 0,
                       "lazy_captures": 0,
                       "steps_by_bucket": {}, "max_rows": max_rows, "lmax": lmax}

    # ------------------------------------------------------------------------------------------------ mode / patches
    def _batch_mode(self, on: bool) -> None:
        self._tl.batch = bool(on)
        self.graph_mode = on

    def _in_batch(self) -> bool:
        return getattr(self._tl, "batch", False)

    def _causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.graph_mode and self._in_batch() and attention_mask is None:
            n = input_tensor.shape[0]
            m = self.bmask[:n, :, :, :self.bbucket]
            if "shared_mask" in self._neg_control:          # TESTS ONLY: one mask (union) for all rows
                m = m.amax(0, keepdim=True).expand(n, -1, -1, -1)
            return m
        return super()._causal_mask(attention_mask, input_tensor, cache_position)

    def _guard(self, cache):
        if cache is not None and getattr(cache, "_v2_batch", None) is self:
            raise RuntimeError("this session cache is attached to BatchedThinker: use step()/step_one() or detach")

    def _forward(self, input_embeds, cache=None, **kw):     # b1_graph=True: ThinkerGraph B=1 path + guard
        self._guard(cache)
        return super()._forward(input_embeds, cache=cache, **kw)

    def _eager_forward(self, input_embeds, cache=None, **kw):
        self._guard(cache)
        if getattr(input_embeds, "dim", lambda: 0)() == 3:
            self._frame_sig = (input_embeds.dtype, int(input_embeds.shape[-1]))
        return self._orig_forward(input_embeds, cache=cache, **kw)

    def _update_rows(self, key_states, value_states, layer_idx, *a, **kw):
        j = self._attn_j[layer_idx]
        n = key_states.shape[0]
        if key_states.shape[2] != 1:
            raise RuntimeError("batched Thinker: decode is one frame per row")
        idx = self.kv_idx[:n * self.kvh]
        self.bK[j].view(-1, self.hd).index_copy_(0, idx, key_states.reshape(n * self.kvh, self.hd).to(self.bK.dtype))
        self.bV[j].view(-1, self.hd).index_copy_(0, idx, value_states.reshape(n * self.kvh, self.hd).to(self.bV.dtype))
        return self.bK[j][:n, :, :self.bbucket], self.bV[j][:n, :, :self.bbucket]

    # ------------------------------------------------------------------------------------------------ buffers
    def _alloc(self, c):
        R = self.max_rows
        dev = self.device
        self.bK = torch.zeros(len(self.attn), R, self.kvh, self.lmax, self.hd, dtype=self.dtype, device=dev)
        self.bV = torch.zeros_like(self.bK)
        self.bmask = torch.full((R, 1, 1, self.lmax), torch.finfo(self.dtype).min, dtype=self.dtype, device=dev)
        self.bconv = [torch.zeros((R,) + tuple(c.conv_states[li].shape[1:]), dtype=c.conv_states[li].dtype, device=dev)
                      for li in self.mamba]
        self.bssm = [torch.zeros((R,) + tuple(c.ssm_states[li].shape[1:]), dtype=c.ssm_states[li].dtype, device=dev)
                     for li in self.mamba]
        self.posv = torch.zeros(R, dtype=torch.long, device=dev)
        rows = torch.arange(R, device=dev)[:, None] * self.kvh + torch.arange(self.kvh, device=dev)[None, :]
        self.kv_base = rows * self.lmax                    # [R, kvh] flat offset of (row, head, pos 0)
        self.kv_idx = torch.zeros(R * self.kvh, dtype=torch.long, device=dev)
        tpl = copy.copy(c)
        for k in [k for k in vars(tpl) if k.startswith("_v2_")]:
            delattr(tpl, k)
        nl = len(c.key_cache)
        tpl.key_cache = [torch.zeros(1, 0, device=dev) for _ in range(nl)]
        tpl.value_cache = [torch.zeros(1, 0, device=dev) for _ in range(nl)]
        tpl.conv_states = [torch.zeros(1, 0, device=dev) for _ in range(nl)]
        tpl.ssm_states = [torch.zeros(1, 0, device=dev) for _ in range(nl)]
        self._template = tpl
        if self.precapture_on and self.use_graphs and torch.device(dev).type == "cuda":
            self._precapture(start_pos=int(getattr(c, "_v2_pos", 0)))

    def _precapture_keys(self, dtype, h, start_pos=None) -> list:
        """Bucket-major: the first session's bucket, then upward, then the lower ones (capture_guard.precapture_buckets,
        BUDGET.min/max_bucket); per bucket every row count, largest first (the full batch is the hot key)."""
        bl = precapture_buckets(self.lmax, start_pos, BUDGET.min_bucket, BUDGET.max_bucket)
        return [(n, b, dtype, h) for b in bl for n in range(self.max_rows, 0, -1)]

    def _precapture(self, start_pos=None) -> None:
        """Capture the enumerable keys on the freshly allocated (empty) buffers, then reset them. Runs only when the
        capture guard reports the calling thread as the only GPU-issuing one (else skipped: lazy captures remain).
        FAIL SOFT: stops when free memory would drop below BUDGET.min_free_gib or a warm-up/capture OOMs; the remaining
        keys stay lazy (barrier-guarded) and the run continues."""
        why = GUARD.quiet_reason()
        if why is not None:
            GUARD.note_precapture("thinker_batch", 0, 0.0, skipped=why)
            return
        dtype, h = self._frame_sig or (self.dtype, int(self.llm.config.hidden_size))
        cuda = self.device.type == "cuda"
        t0 = time.perf_counter()
        a0 = torch.cuda.memory_allocated() if cuda else 0
        n_new, stop = 0, None
        saved_bucket = self.bbucket
        keys = self._precapture_keys(dtype, h, start_pos)
        try:
            for key in keys:
                if key in self.bgraphs:
                    continue
                if cuda and free_gib() < BUDGET.min_free_gib:
                    stop = f"budget: free {free_gib():.2f} GiB < min_free_gib {BUDGET.min_free_gib}"
                    break
                n, b = key[0], key[1]
                try:
                    x = torch.zeros(n, 1, h, dtype=dtype, device=self.device)
                    self.kv_idx[:n * self.kvh].copy_(self.kv_base[:n].reshape(-1))
                    self.bbucket = b
                    self._capture(key, x, self._cache_view(n), max(1, b // 2))
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        raise
                    stop = f"oom at {key[:2]}: {str(e).splitlines()[0][:160]}"
                    break
                if key not in self.bgraphs:                  # failed + recovered by the guard: no permanent disable,
                    GUARD.disabled.discard(("thinker_batch", key))   # the key is retried lazily (barrier-guarded)
                    stop = f"capture failed at {key[:2]}: {self.bstats.get('first_capture_error')}"
                    break
                n_new += 1
        finally:
            self.bbucket = saved_bucket
            # graphs read/write these buffers by address only: reset the warm-ups' garbage (attach rewrites rows)
            self.bK.zero_()
            self.bV.zero_()
            self.bmask.fill_(torch.finfo(self.dtype).min)
            self.posv.zero_()
            self.kv_idx.zero_()
            for t in (*self.bconv, *self.bssm):
                t.zero_()
        if stop is not None and cuda:
            soft_cleanup()
        self.bstats["precaptured"] += n_new
        self.bstats["precapture_sig"] = f"{dtype} H={h}"
        self.bstats["precapture_keys"] = {"planned": len(keys), "captured": n_new, "stopped": stop,
                                          "first": [list(k[:2]) for k in keys[:3]]}
        if cuda:                                           # graph pools + static in/out (allocated, not reserved:
            self.bstats["precapture_alloc_delta"] = torch.cuda.memory_allocated() - a0   # graph() empties the cache)
            self.bstats["precapture_pool_bytes"] = self._pool_bytes()
        GUARD.note_precapture("thinker_batch", n_new, (time.perf_counter() - t0) * 1e3, stopped=stop)

    def _cache_view(self, n: int):
        """Combined cache for the leading n rows (views of the static buffers; one object per n, kept for graphs)."""
        cc = self._ccs.get(n)
        if cc is None:
            t = self._template
            cc = copy.copy(t)
            cc.key_cache, cc.value_cache = list(t.key_cache), list(t.value_cache)
            cc.conv_states, cc.ssm_states = list(t.conv_states), list(t.ssm_states)
            for j, li in enumerate(self.mamba):
                cc.conv_states[li] = self.bconv[j][:n]
                cc.ssm_states[li] = self.bssm[j][:n]
            cc.update = self._update_rows
            self._ccs[n] = cc
        return cc

    def _copy_row(self, src: int, dst: int, t: int) -> None:
        self.bK[:, dst, :, :t] = self.bK[:, src, :, :t]
        self.bV[:, dst, :, :t] = self.bV[:, src, :, :t]
        self.bmask[dst] = self.bmask[src]
        self.posv[dst] = self.posv[src]
        for j in range(len(self.mamba)):
            self.bconv[j][dst] = self.bconv[j][src]
            self.bssm[j][dst] = self.bssm[j][src]

    def _swap_rows(self, a: int, b: int, ta: int, tb: int) -> None:
        """Swap rows a (valid length ta) and b (tb): K/V valid prefixes only, whole mask/Mamba rows."""
        ka, va = self.bK[:, a, :, :ta].clone(), self.bV[:, a, :, :ta].clone()
        self.bK[:, a, :, :tb] = self.bK[:, b, :, :tb]
        self.bV[:, a, :, :tb] = self.bV[:, b, :, :tb]
        self.bK[:, b, :, :ta] = ka
        self.bV[:, b, :, :ta] = va
        ab = torch.tensor([a, b], device=self.device)
        ba = torch.tensor([b, a], device=self.device)
        for t in [self.bmask, self.posv, *self.bconv, *self.bssm]:
            t.index_copy_(0, ab, t.index_select(0, ba))

    # ------------------------------------------------------------------------------------------------ contract API
    def _check_open(self):
        if self._closed:
            raise RuntimeError("BatchedThinker is closed")

    def attach(self, sid: SessionId, cache: Any) -> int:
        with self.lock:
            self._check_open()
            return self._attach(sid, cache)

    def _attach(self, sid, c) -> int:
        if getattr(c, "_v2_batch", None) is not None:
            raise RuntimeError("cache already attached to a batch backend")
        t = int(getattr(c, "_v2_pos", 0))
        if getattr(c, "_v2_past_lmax", False) or t + 1 > self.lmax:
            raise ValueError(f"session at position {t} does not fit lmax={self.lmax}: keep it eager")
        if sid in self.table.row_of:
            raise KeyError(f"session {sid!r} already attached")
        if self.table.n >= self.max_rows:
            raise RuntimeError(f"batch full ({self.max_rows} rows)")
        g = getattr(c, "_v2_graph", None)
        if g is self:
            self._unconvert(c, t)                          # was on our B=1 graph buffers: back to a plain cache
        elif g is not None:
            raise RuntimeError("cache is held by another ThinkerGraph")
        for li in self.attn:
            k, v = c.key_cache[li], c.value_cache[li]
            if tuple(k.shape) != (1, self.kvh, t, self.hd) or k.shape != v.shape:
                raise RuntimeError(f"attention cache layer {li}: shape {tuple(k.shape)} != (1, {self.kvh}, {t}, "
                                   f"{self.hd}) (B=1 cache after prefill expected)")
            if k.dtype != self.dtype:
                raise RuntimeError(f"K/V dtype {k.dtype} != model dtype {self.dtype} (round trip would not be exact)")
        if self.bK is None:
            self._alloc(c)
        for j, li in enumerate(self.mamba):
            for st, buf in ((c.conv_states[li], self.bconv[j]), (c.ssm_states[li], self.bssm[j])):
                if st.shape[0] != 1 or tuple(st.shape[1:]) != tuple(buf.shape[1:]) or st.dtype != buf.dtype:
                    raise RuntimeError(f"Mamba layer {li}: state {tuple(st.shape)} {st.dtype} does not match the "
                                       f"backend rows {tuple(buf.shape[1:])} {buf.dtype}")
        r = self.table.attach(sid, t)
        for j, li in enumerate(self.attn):
            self.bK[j, r, :, :t] = c.key_cache[li][0]
            self.bV[j, r, :, :t] = c.value_cache[li][0]
            c.key_cache[li] = c.key_cache[li].new_empty((1, self.kvh, 0, self.hd))   # release the session copy
            c.value_cache[li] = c.value_cache[li].new_empty((1, self.kvh, 0, self.hd))
        self.bmask[r].fill_(torch.finfo(self.dtype).min)
        self.bmask[r, ..., :t] = 0.0
        self.posv[r] = t
        for j, li in enumerate(self.mamba):
            cs, ss = c.conv_states[li], c.ssm_states[li]
            self._mamba_layout[(sid, li)] = ((tuple(cs.shape), cs.stride()), (tuple(ss.shape), ss.stride()))
            self.bconv[j][r] = cs[0]
            self.bssm[j][r] = ss[0]
            c.conv_states[li] = cs.new_empty((1, 0))
            c.ssm_states[li] = ss.new_empty((1, 0))
        c._v2_batch = self
        self._caches[sid] = c
        self.bstats["attach"] += 1
        return r

    def detach(self, sid: SessionId) -> Any:
        with self.lock:
            self._check_open()
            return self._detach(sid)

    def _detach(self, sid):
        if sid not in self.table.row_of:
            raise KeyError(f"session {sid!r} not attached")
        c = self._caches.pop(sid)
        r, t = self.table.row_of[sid], self.table.pos[sid]
        for j, li in enumerate(self.attn):
            c.key_cache[li] = self.bK[j, r:r + 1, :, :t].clone(memory_format=torch.contiguous_format)
            c.value_cache[li] = self.bV[j, r:r + 1, :, :t].clone(memory_format=torch.contiguous_format)
        for j, li in enumerate(self.mamba):
            (cshape, cstride), (sshape, sstride) = self._mamba_layout.pop((sid, li))
            cs = torch.empty_strided(cshape, cstride, dtype=self.bconv[j].dtype, device=self.device)
            ss = torch.empty_strided(sshape, sstride, dtype=self.bssm[j].dtype, device=self.device)
            cs.copy_(self.bconv[j][r:r + 1])
            ss.copy_(self.bssm[j][r:r + 1])
            c.conv_states[li], c.ssm_states[li] = cs, ss
        c._v2_pos = t
        del c._v2_batch
        _, src = self.table.release(sid)
        if src is not None:
            self._copy_row(src, r, self.table.pos[self.table.sid_at[r]])
            self.bstats["moves"] += 1
            free = src
        else:
            free = r
        self.bmask[free].fill_(torch.finfo(self.dtype).min)
        self.bstats["detach"] += 1
        return c

    def positions(self) -> Mapping[SessionId, int]:
        with self.lock:
            return dict(self.table.pos)

    def step_one(self, sid: SessionId, frame: Any) -> ThinkerOut:
        """B=1 through the batched path (n_rows=1): same numerics as ThinkerGraph's B=1 masked-window step."""
        return self.step({sid: frame})[sid]

    def step(self, frames: Mapping[SessionId, Any]) -> dict[SessionId, ThinkerOut]:
        with self.lock:
            self._check_open()
            return self._step(frames)

    def _step(self, frames):
        out: dict[SessionId, ThinkerOut] = {}
        for sid, x in frames.items():
            if sid not in self.table.row_of:
                raise KeyError(f"session {sid!r} not attached")
            if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1:
                raise ValueError(f"frame for {sid!r} must be [1, 1, H], got {tuple(x.shape)}")
        # rows that would pass lmax: back to their own cache, this frame eager (other rows untouched)
        for sid in [s for s in frames if self.table.pos[s] + 1 > self.lmax]:
            c = self._detach(sid)
            c._v2_past_lmax = True
            o = self._orig_forward(frames[sid], cache=c)
            out[sid] = _thinker_out(o, None)
            self.evicted[sid] = c
            self.bstats["evictions"] += 1
        parts = [s for s in frames if s not in out]
        if not parts:
            return out
        for a, b in self.table.arrange(parts):
            # table already swapped: row a now holds the sid that was at b
            self._swap_rows(a, b, self.table.pos[self.table.sid_at[b]], self.table.pos[self.table.sid_at[a]])
            self.bstats["swaps"] += 1
        n = len(parts)
        order = self.table.order(n)
        x = torch.cat([frames[s] for s in order], 0)
        # ---- bookkeeping outside the graph: mask at posv, flat K/V indices, shared bucket
        pv = self.posv[:n]
        self.bmask[:n].view(n, self.lmax).scatter_(1, pv[:, None], 0.0)
        if "shared_pos" in self._neg_control:            # TESTS ONLY: every row writes at the max position
            pv = pv.max().expand(n)
        self.kv_idx[:n * self.kvh].copy_((self.kv_base[:n] + pv[:, None]).reshape(-1))
        self.bbucket = self.table.bucket(order, self.lmax)
        hp = max(self.table.pos[s] for s in order)       # > 0: Mamba decode branch (host answer)
        cc = self._cache_view(n)
        if self.use_graphs and self._graphable(x):
            res = self._graph_step(n, x, cc, hp)
        else:
            res = self._eager_batched(x, cc, hp)
        self.posv[:n] += 1
        self.table.advance(order)
        st = self.bstats
        st["steps"] += 1
        st["steps_by_n"][str(n)] = st["steps_by_n"].get(str(n), 0) + 1
        st["steps_by_bucket"][str(self.bbucket)] = st["steps_by_bucket"].get(str(self.bbucket), 0) + 1
        for r, s in enumerate(order):
            out[s] = _thinker_out(res, r)
        return out

    @staticmethod
    def _graphable(x) -> bool:                       # CPU tests replace it to drive the capture policy on fakes
        return x.is_cuda

    def _eager_batched(self, x, cc, hp):
        """The batched forward without a graph (same static buffers/mask/bucket as the graph's warm-up)."""
        self._batch_mode(True)
        try:
            cc._v2_pos = hp
            res = self._orig_forward(x, cache=cc)
        finally:
            self._batch_mode(False)
        self.bstats["eager_steps"] += 1
        return res

    def _graph_step(self, n, x, cc, hp):
        key = (n, self.bbucket, x.dtype, x.shape[-1])
        entry = self.bgraphs.get(key)
        if entry is None:
            if not GUARD.may_capture("thinker_batch", key):      # barrier policy: not quiet now → eager, retry later
                self.bstats["capture_deferred"] += 1
                return self._eager_batched(x, cc, hp)
            self.bstats["lazy_captures"] += 1
            return self._capture(key, x, cc, hp)
        g, sin, sout = entry
        sin.copy_(x)
        g.replay()
        self.bstats["replays"] += 1
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in sout.items()}

    def _capture(self, key, x, cc, hp):
        with GUARD.capturing():
            return self._capture_locked(key, x, cc, hp)

    def _capture_locked(self, key, x, cc, hp):
        t0 = time.perf_counter()
        with torch.inference_mode(False):
            # a NORMAL tensor: replays copy into it from no-grad (session) and inference-mode (FC) callers alike; an
            # inference tensor here would make a later replay outside inference mode raise
            static_in = x.clone()
        self._batch_mode(True)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), sync_watch(self.bstats):   # warm-up = this step's real result
                cc._v2_pos = hp
                out_warm = self._orig_forward(static_in, cache=cc)
            torch.cuda.current_stream().wait_stream(s)
            cc._v2_pos = hp
            g = torch.cuda.CUDAGraph()
            prev_stream = torch.cuda.current_stream()
            r0 = torch.cuda.memory_reserved()
            try:
                with torch.cuda.graph(g, pool=self.bpool, capture_error_mode="thread_local"):
                    static_out = self._orig_forward(static_in, cache=cc)
            except Exception as e:
                self.bstats.setdefault("first_capture_error", f"{type(e).__name__}: {e}")
                if GUARD.on_failure("thinker_batch", key, e, stream=prev_stream):     # barrier policy: generator reset, key → eager
                    self.bstats["capture_failed_recovered"] += 1
                    return {k: v for k, v in out_warm.items() if k != "cache"}
                raise RuntimeError(f"thinker_batch capture failed {key[:2]}: {type(e).__name__}: {e}") from e
            self._pool_delta += torch.cuda.memory_reserved() - r0
        finally:
            self._batch_mode(False)
        self.bgraphs[key] = (g, static_in, {k: v for k, v in static_out.items() if k != "cache"})
        self.bstats["captures"] += 1
        self.bstats["capture_ms"] += (time.perf_counter() - t0) * 1000
        return {k: v for k, v in out_warm.items() if k != "cache"}

    # ------------------------------------------------------------------------------------------------ memory / close
    def memory(self) -> BackendMemory:
        with self.lock:
            esz = torch.finfo(self.dtype).bits // 8
            kv = 2 * len(self.attn) * self.kvh * self.lmax * self.hd * esz
            mask = self.lmax * esz
            extra = {"kv_bytes_per_slot": kv, "mask_bytes_per_slot": mask, "n_attn": len(self.attn),
                     "kvh": self.kvh, "hd": self.hd, "lmax": self.lmax, "max_rows": self.max_rows,
                     "graphs": len(self.bgraphs), "pool_reserved_delta_at_capture": self._pool_delta}
            static = 0
            mamba = None
            if self.bK is not None:
                mamba = sum(b[0].numel() * b.element_size() for b in (*self.bconv, *self.bssm))
                static = sum(t.numel() * t.element_size() for t in
                             (self.bK, self.bV, self.bmask, self.posv, self.kv_base, self.kv_idx, *self.bconv,
                              *self.bssm))
                static += sum(g[1].numel() * g[1].element_size() for g in self.bgraphs.values())
            extra["mamba_bytes_per_slot"] = mamba                # None until the first attach (layout from a cache)
            pool, how = self._pool_bytes()
            extra["graph_pool_method"] = how
            return BackendMemory(bytes_per_slot=kv + mask + (mamba or 0), graph_pool_bytes=pool,
                                 static_buffers_bytes=static, extra=extra)

    def _pool_bytes(self) -> tuple[int, str]:
        if not self.bgraphs:
            return 0, "no graphs"
        try:
            pid = tuple(self.bpool)
            segs = torch.cuda.memory_snapshot()
            hit = [s for s in segs if tuple(s.get("segment_pool_id", ())) == pid]
            if hit:
                return int(sum(s["total_size"] for s in hit)), "memory_snapshot segment_pool_id"
        except Exception:                                    # noqa: BLE001
            pass
        return int(self._pool_delta), "memory_reserved delta during capture"

    def close(self) -> None:
        """Teardown: attached sessions are written back into their own cache objects (continue eagerly), graphs and
        static buffers are released, the model patches are removed. Later calls raise."""
        with self.lock:
            if self._closed:
                return
            for sid in list(self.table.row_of):
                self._detach(sid)
            self.bgraphs.clear()
            self._ccs.clear()
            self._template = None
            self.bK = self.bV = self.bmask = self.bconv = self.bssm = None
            self.posv = self.kv_base = self.kv_idx = None
            self._closed = True
            super().remove()

    def remove(self) -> None:
        self.close()

    def snapshot(self, sid: SessionId) -> dict:
        """Clones of one session's logical state (tests/debug): K/V valid prefix, Mamba rows, mask row, positions."""
        with self.lock:
            r, t = self.table.row_of[sid], self.table.pos[sid]
            return {"K": self.bK[:, r, :, :t].clone(), "V": self.bV[:, r, :, :t].clone(),
                    "mask": self.bmask[r].clone(), "posv": int(self.posv[r]), "pos": t,
                    "conv": [b[r].clone() for b in self.bconv], "ssm": [b[r].clone() for b in self.bssm]}


def _thinker_out(o: dict, row: int | None) -> ThinkerOut:
    def pick(h):
        v = o.get(h)
        if not isinstance(v, torch.Tensor):
            return v
        return v if row is None else v[row:row + 1]
    return ThinkerOut(*(pick(h) for h in HEADS))
