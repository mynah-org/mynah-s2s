"""EXPERIMENTAL batched cache-aware perception for multi-session rounds (opt-in, RendezvousEngine(batch_perception=...)).

Upstream ``PerceptionCacheManager.step`` (nemo .../model_wrappers/perception_cache.py @097dfe9) encodes ONE session per
call: mel preprocessor on the session's full sliding audio buffer → per sub-step slice (first vs subsequent chunk,
pre-encode cache) → FastConformer ``cache_aware_stream_step`` (B=1 CUDA graph) → modality adapter → proj. In a
rendezvous round N sessions call it back to back (~11 ms each on the Ada, serialized on the device).

``BatchedPerception.run(calls)`` executes the same math for n calls at once:
  * mel: ONE preprocessor call on the stacked buffers when every buffer has the same length (steady state; warm-up
    buffers of different fill levels fall back to one call per row) — or always per row (``batch_mel=False``);
  * per sub-step, rows are grouped by (first-chunk graph kind, mel length, drop_extra_pre_encoded); each group runs one
    eager encoder + adapter + proj step with the per-session caches concatenated along their batch dimension (found
    from ``get_initial_cache_state(1)`` vs ``(2)``), then outputs and caches are split back per row;
  * the slicing is the upstream one, line for line (perception_cache.py:425-483);
  * no host read of device values; the upstream device-wide ``torch.cuda.synchronize()`` after every sub-step and at
    the end (perception_cache.py:542/557) IS reproduced by default (``sync=True``): dropping them (perception_nosync)
    was measured NOT token-neutral on the Ada (round 20, q12 diverged; baseline runs 5/5 identical), so the sync
    semantics are kept until proven otherwise — one sync per batched round instead of one per session.
A single call (n == 1) is NOT handled here: the rendezvous runs the installed upstream ``step`` (B=1 graph) for it, so
a lone session stays byte-identical.

CUDA graphs per batch size (``graphs=True`` + ``precapture(max_rows)``). Ada round 21: the EAGER batched B=4 step cost
~95 ms/round (launch-bound) vs 4 x ~11 ms for the B=1 graph replays → eager batching is a regression. ``precapture``
captures, at a QUIET point (before any session thread exists: RendezvousEngine construction), one graph per row count
n = 2..max_rows for the steady-state key (subsequent chunk: mel length chunk + pre-encode cache, drop_extra), with
static stacked inputs (mel, lengths, caches) and static outputs, exactly like upstream's B=1 capture
(perception_cache.py:199-318). There is NO lazy capture: any other key (first chunk of a new session, n above
max_rows, a row count whose capture was skipped) runs eager. Replay = copy rows into the static inputs → replay →
clone each row's outputs (the next replay overwrites them). Measured: eager B=1 == upstream B=1 graph bit for bit
(PB1, Ada); batched rows vs B=1 differ by ~3e-3 max abs (other GEMM M) yet 4/4 items token-identical (PB1 T).
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _second(x):
    return x[1] if isinstance(x, (list, tuple)) else x


class BatchedPerception:
    def __init__(self, mgr, batch_mel: bool = True, sync: bool = True, graphs: bool = False, capture=None):
        self.mgr = mgr
        self.batch_mel = bool(batch_mel)
        self.sync = bool(sync)
        self.use_graphs = bool(graphs)
        self.graphs: dict[tuple, "_StepGraph"] = {}
        self._capture = capture or _cuda_capture         # injectable (CPU tests: static-buffer emulation)
        perception = mgr.model.stt_model.perception
        self.perception, self.encoder = perception, perception.encoder
        self.bdims = self._batch_dims()
        self.stats = {"calls": 0, "rows": 0, "hist": {}, "groups": 0, "mel_batched": 0, "mel_per_row": 0,
                      "graph_groups": 0, "eager_groups": 0, "eager_keys": {}, "captured": [], "capture_ms": 0.0}

    def _batch_dims(self) -> tuple:
        """Batch dimension of each cache tensor (cache_last_channel, cache_last_time, cache_last_channel_len)."""
        one = self.encoder.get_initial_cache_state(batch_size=1)
        two = self.encoder.get_initial_cache_state(batch_size=2)
        dims = []
        for a, b in zip(one, two):
            if a is None:
                dims.append(None)
                continue
            d = [i for i, (x, y) in enumerate(zip(a.shape, b.shape)) if x != y]
            if len(d) != 1 or a.shape[d[0]] != 1 or b.shape[d[0]] != 2:
                raise RuntimeError(f"cannot find the batch dim of a perception cache: {tuple(a.shape)} vs {tuple(b.shape)}")
            dims.append(d[0])
        return tuple(dims)

    # ------------------------------------------------------------------------------------------------ mel
    def _mel(self, audios: list) -> list:
        pre = self.mgr.preprocessor
        dev = self.mgr.device
        lens = {int(a.shape[1]) for a in audios}
        if self.batch_mel and len(audios) > 1 and len(lens) == 1:
            x = torch.cat([a.to(torch.float32) for a in audios], dim=0)
            ln = torch.full((len(audios),), lens.pop(), dtype=torch.long, device=dev)
            mel, _ = pre(input_signal=x, length=ln)
            self.stats["mel_batched"] += 1
            return [mel[i:i + 1] for i in range(len(audios))]
        out = []
        for a in audios:
            ln = torch.tensor([a.shape[1]], dtype=torch.long, device=dev)
            mel, _ = pre(input_signal=a.to(torch.float32), length=ln)
            out.append(mel)
            self.stats["mel_per_row"] += 1
        return out

    # ------------------------------------------------------------------------------------------------ slicing
    def _chunk(self, ps: torch.Tensor, sub_frame_idx: int, sub_step: int, num_sub_steps: int):
        """mel chunk of one row for one sub-step + (use_first_graph, drop_extra_pre_encoded): perception_cache.py:429-483."""
        cfg, m = self.mgr.streaming_cfg, self.mgr
        chunk_size_first, chunk_size = _first(cfg.chunk_size), _second(cfg.chunk_size)
        shift_size_first, shift_size = _first(cfg.shift_size), _second(cfg.shift_size)
        pre_first, pre = _first(cfg.pre_encode_cache_size), _second(cfg.pre_encode_cache_size)
        base = self.encoder._cfg.att_context_size[1] + 1
        is_first = sub_frame_idx == 0
        if is_first:
            if m.pad_and_drop_preencoded:
                cur_chunk, cur_pre, drop = chunk_size, pre, cfg.drop_extra_pre_encoded
            else:
                cur_chunk, cur_pre, drop = chunk_size_first, pre_first, 0
            mel = ps[:, :, :cur_chunk]
            if cur_pre > 0:
                z = torch.zeros((ps.size(0), m.input_features, cur_pre), device=ps.device, dtype=ps.dtype)
                mel = torch.cat([z, mel], dim=-1)
        else:
            cur_chunk, cur_pre, drop = chunk_size, pre, cfg.drop_extra_pre_encoded
            mel_T = ps.shape[-1]
            step_number = sub_frame_idx // base
            start = shift_size_first + (step_number - 1) * shift_size
            end = start + cur_chunk
            offset = chunk_size - shift_size_first
            if end > mel_T - offset:
                end = mel_T - offset - (num_sub_steps - 1 - sub_step) * shift_size
                start = end - cur_chunk
            main = ps[:, :, start:end]
            cache_mel = ps[:, :, max(0, start - cur_pre):start]
            if cache_mel.shape[-1] < cur_pre:
                z = torch.zeros((cache_mel.size(0), cache_mel.size(1), cur_pre - cache_mel.shape[-1]),
                                device=ps.device, dtype=cache_mel.dtype)
                cache_mel = torch.cat([z, cache_mel], dim=-1)
            mel = torch.cat([cache_mel, main], dim=-1)
        use_first_graph = is_first and not m.pad_and_drop_preencoded
        return mel, (use_first_graph, int(mel.shape[-1]), int(drop))

    # ------------------------------------------------------------------------------------------------ encode
    def _cat(self, caches: list, j: int):
        d = self.bdims[j]
        xs = [c[j] for c in caches]
        if d is None or any(x is None for x in xs):
            if any(x is not None for x in xs):
                raise RuntimeError("perception batch: some rows have a cache tensor, others None")
            return None
        return torch.cat(xs, dim=d)

    def _split(self, x, j: int, n: int):
        if x is None:
            return [None] * n
        d = self.bdims[j]
        # always a fresh tensor: graph outputs are static buffers overwritten by the next replay
        return [x.narrow(d, i, 1).clone(memory_format=torch.contiguous_format) for i in range(n)]

    def _math(self, mel, lens, c_ch, c_t, c_len, drop: int):
        """encoder step + modality adapter + proj on stacked rows (the body of upstream's graphs / eager branch)."""
        enc, enc_len, c_ch, c_t, c_len = self.encoder.cache_aware_stream_step(
            processed_signal=mel, processed_signal_length=lens, cache_last_channel=c_ch, cache_last_time=c_t,
            cache_last_channel_len=c_len, keep_all_outputs=True, drop_extra_pre_encoded=drop)
        adapted, _ = self.perception.modality_adapter(audio_signal=enc, length=enc_len)
        chunk = self.perception.proj(adapted.transpose(1, 2))
        return chunk, enc, c_ch, c_t, c_len

    def _rows(self, n, chunk, enc, c_ch, c_t, c_len, clone: bool):
        raw = enc.transpose(1, 2)
        cut = (lambda t, i: t[i:i + 1].clone()) if clone else (lambda t, i: t[i:i + 1])
        new = list(zip(self._split(c_ch, 0, n), self._split(c_t, 1, n), self._split(c_len, 2, n)))
        return [cut(chunk, i) for i in range(n)], [cut(raw, i) for i in range(n)], new

    def _encode(self, mel, caches: list, drop: int, key: tuple):
        n = mel.shape[0]
        g = self.graphs.get((n,) + key) if self.use_graphs else None
        if g is not None:
            self.stats["graph_groups"] += 1
            g.mel.copy_(mel)
            for j in range(3):
                if g.caches[j] is not None:
                    g.caches[j].copy_(self._cat(caches, j))
            g.replay()
            return self._rows(n, *g.out, clone=True)
        self.stats["eager_groups"] += 1
        if self.use_graphs:
            k = str((n,) + key)
            self.stats["eager_keys"][k] = self.stats["eager_keys"].get(k, 0) + 1
        lens = torch.full((n,), mel.shape[-1], dtype=torch.long, device=mel.device)
        out = self._math(mel, lens, self._cat(caches, 0), self._cat(caches, 1), self._cat(caches, 2), drop)
        return self._rows(n, *out, clone=False)

    # ------------------------------------------------------------------------------------------------ graphs
    def steady_key(self) -> tuple:
        """(use_first_graph, mel_len, drop) of every subsequent-chunk sub-step (and of the first chunk when
        pad_and_drop_preencoded): the only key that is pre-captured."""
        cfg = self.mgr.streaming_cfg
        return (False, int(_second(cfg.chunk_size)) + int(_second(cfg.pre_encode_cache_size)),
                int(cfg.drop_extra_pre_encoded))

    def precapture(self, max_rows: int) -> dict:
        """Capture one graph per row count 2..max_rows for the steady key. Call ONLY at a quiet point (no other thread
        issuing GPU work: a capture is invalidated by another thread's device sync / RNG use, capture_guard.py)."""
        import time
        if not self.use_graphs:
            return {}
        key = self.steady_key()
        t0 = time.perf_counter()
        with torch.no_grad():
            for n in range(2, int(max_rows) + 1):
                if (n,) + key not in self.graphs:
                    self.graphs[(n,) + key] = _StepGraph(self, n, key)
                    self.stats["captured"].append(str((n,) + key))
        self.stats["capture_ms"] = round(self.stats["capture_ms"] + (time.perf_counter() - t0) * 1e3, 1)
        return {"keys": list(self.stats["captured"]), "ms": self.stats["capture_ms"]}

    def _sync(self):
        if self.sync and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            self.stats["syncs"] = self.stats.get("syncs", 0) + 1

    def run(self, calls: list[dict]) -> list[tuple]:
        """``calls``: kwargs of ``PerceptionCacheManager.step`` (audio_input, frame_idx, num_frames_per_chunk,
        perception_cache), all with the same num_frames_per_chunk. Returns one (encoded, new_cache, raw_emb) per call,
        in order, like ``step``."""
        n = len(calls)
        nfs = {int(c["num_frames_per_chunk"]) for c in calls}
        if len(nfs) != 1:
            raise ValueError(f"perception batch needs one num_frames_per_chunk, got {sorted(nfs)}")
        nf = nfs.pop()
        base = self.encoder._cfg.att_context_size[1] + 1
        if nf % base != 0:
            raise ValueError(f"num_frames_per_chunk must be a multiple of (lookahead + 1) = {base}. Got {nf}")
        num_sub = nf // base
        mels = self._mel([c["audio_input"] for c in calls])
        caches = [(c["perception_cache"].cache_last_channel, c["perception_cache"].cache_last_time,
                   c["perception_cache"].cache_last_channel_len) for c in calls]
        enc_rows: list[list] = [[] for _ in range(n)]
        raw_rows: list[list] = [[] for _ in range(n)]
        for sub in range(num_sub):
            groups: dict[tuple, list] = {}
            chunks = {}
            for i, c in enumerate(calls):
                mel, key = self._chunk(mels[i], int(c["frame_idx"]) + sub * base, sub, num_sub)
                chunks[i] = mel
                groups.setdefault(key, []).append(i)
            for key, rows in groups.items():
                drop = key[2]
                self.stats["groups"] += 1
                encs, raws, new = self._encode(torch.cat([chunks[i] for i in rows], dim=0),
                                               [caches[i] for i in rows], drop, key)
                for k, i in enumerate(rows):
                    enc_rows[i].append(encs[k])
                    raw_rows[i].append(raws[k])
                    caches[i] = new[k]
            self._sync()                                   # perception_cache.py:542 (per sub-step)
        cls = type(calls[0]["perception_cache"])
        out = []
        for i in range(n):
            enc = torch.cat(enc_rows[i], dim=1) if len(enc_rows[i]) > 1 else enc_rows[i][0]
            raw = torch.cat(raw_rows[i], dim=1) if len(raw_rows[i]) > 1 else raw_rows[i][0]
            out.append((enc, cls(cache_last_channel=caches[i][0], cache_last_time=caches[i][1],
                                 cache_last_channel_len=caches[i][2]), raw))
        self._sync()                                       # perception_cache.py:557 (after the cat)
        self.stats["calls"] += 1
        self.stats["rows"] += n
        self.stats["hist"][n] = self.stats["hist"].get(n, 0) + 1
        return out


def step_kwargs(args: tuple, kw: dict) -> Optional[dict]:
    """Normalize a ``PerceptionCacheManager.step`` call to kwargs (None if the signature is not the expected one)."""
    names = ("audio_input", "frame_idx", "num_frames_per_chunk", "perception_cache")
    if len(args) > len(names) or set(kw) - set(names):
        return None
    d: dict[str, Any] = dict(zip(names, args))
    for k, v in kw.items():
        if k in d:
            return None
        d[k] = v
    return d if set(d) == set(names) else None


class _StepGraph:
    """One captured batched step for a fixed (n, use_first, mel_len, drop): static inputs → graph → static outputs."""

    def __init__(self, bp: BatchedPerception, n: int, key: tuple):
        _, mel_len, drop = key
        dev = bp.mgr.device
        self.mel = torch.zeros((n, bp.mgr.input_features, mel_len), dtype=torch.float32, device=dev)
        self.lens = torch.full((n,), mel_len, dtype=torch.long, device=dev)
        self.caches = [c.clone() if c is not None else None
                       for c in bp.encoder.get_initial_cache_state(batch_size=n)]
        self.replay, self.out = bp._capture(lambda: bp._math(self.mel, self.lens, *self.caches, drop))


def _cuda_capture(fn):
    """Warm-up (3 eager runs, side stream) then capture on the default capture path, like upstream's B=1 graphs."""
    cur = torch.cuda.current_stream()
    s = torch.cuda.Stream()
    s.wait_stream(cur)
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    cur.wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g.replay, out
