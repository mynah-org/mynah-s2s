"""EXPERIMENTAL multi-session engine with a REAL batched Thinker forward: cooperative rendezvous (owner GO, option A).

Why cooperative: NVIDIA's ``infer_one_step`` is monolithic (perception → Thinker → sampling/turn-taking/FC → EarTTS →
codec) and its wrapper/pipeline keep per-session state on shared objects. Instead of making that upstream code
thread-safe, each session runs its unchanged ``ReferenceEngine.step`` in its OWN worker thread, but only ONE session
thread executes wrapper code at any time (a baton held by the coordinator):

  round(jobs):  for each ready session s:  switch s in (MultiSessionEngine/SessionStateStore swap incl. RNG) →
                run s until it PARKS at its Thinker call or finishes the job;
                while some sessions are parked:  run ONE Thinker execution for all of them on the coordinator thread
                (attached DECODE frames → backend.step(B=n) / step_one; everything else B=1), then resume each parked
                session (switch in → deliver its result → run until it parks again or finishes).

Guarantees (each tested in tests/cpu/test_rendezvous.py):
  * the Thinker runs ONLY on the coordinator thread (sessions and FC threads only submit requests);
  * perception / EarTTS / codec of different sessions never run concurrently (one baton); FC background threads keep
    their upstream concurrency (same as single-session mode) and their Thinker requests are served B=1 by the
    coordinator, also between rounds (``pump``);
  * explicit Thinker OWNERSHIP per session: while a session's FC thread owns its Thinker (FC phase), a Thinker request
    from that session's main thread raises (no double advance) — not a timing-dependent flag;
  * no waiting for absent sessions: a round only contains sessions that have a job; every hand-off has a timeout; a
    failing/hanging session is failed and removed, the others continue;
  * the B=1 single-session path (ReferenceEngine without this module) is untouched.
Not done here (by design): refactors of infer_one_step.

Opt-in EarTTS batching (``batch_eartts=True``, model/eartts_batch.py): session threads also PARK at
``tts_model.infer_codes_one_step`` (main path); the coordinator serves them after the Thinker and before the codec,
n >= 2 calls on static KV slots as ONE BatchedEarTTS.run (pooled slots, per-row positions/masks, graphs per bucket
pre-captured at construction or captured only through the capture guard), anything else (a session's first step =
DynamicCache conversion, past lmax) alone through the installed per-session step. FC threads, reminders and the
coordinator pass through. Not exact (batched GEMMs; one RNG draw for all rows from the coordinator's generator).

Opt-in perception batching (``batch_perception=True``, model/perception_batch.py): session threads also PARK at
``perception_cache_mgr.step``; the coordinator serves all perception requests parked in the round before any Thinker
request (so the sessions reach their first Thinker frame together), n >= 2 as ONE BatchedPerception.run, a lone request
through the installed upstream step (B=1 graph, byte-identical). Calls from other threads (FC threads, coordinator)
pass through unchanged. Default off: without it this module never touches the perception manager.
``perception_graphs=True``: the batched step replays CUDA graphs pre-captured at construction for 2..backend.max_rows
rows (steady-state chunk key); other keys run eager.

Opt-in codec batching (``batch_codec=True``, model/codec_batch.py): session threads also PARK at
``tts_model.audio_codec.decode`` (main path, one call per chunk with the session's streaming cache); the coordinator
serves codec requests only when no session of the round is parked at perception or the Thinker (so every session that
reaches its codec this round is in the batch), n >= 2 stackable requests as ONE decode with the caches concatenated on
dim 0 (audio-only, not bit-identical: ~1e-6), a lone / non-stackable request through the installed decode (unchanged).
FC threads, the reminder and the coordinator pass through. Default off: the codec is never touched.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from mynah_v2.runtime.batch_contract import Phase, ThinkerOut


def _autocast_state() -> tuple:
    """(enabled, dtype) of the calling thread's CUDA autocast (thread-local: the coordinator re-enters it)."""
    try:
        return bool(torch.is_autocast_enabled("cuda")), torch.get_autocast_dtype("cuda")
    except TypeError:                                   # older torch
        return bool(torch.is_autocast_enabled()), torch.get_autocast_gpu_dtype()


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _gpu_exclusive(fn):
    """Coordinator GPU work (serving a stage, switching a session in, handing out the baton) runs under the capture
    guard's lock, which every CUDA-graph capture holds from before capture_begin to after capture_end: so no
    coordinator-issued GPU work and no newly resumed session can overlap an open capture."""
    def wrapped(self, *a, **kw):
        with self._guard.lock:
            return fn(self, *a, **kw)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


class RendezvousError(RuntimeError):
    pass


class OwnershipError(RendezvousError):
    pass


@dataclass
class _Req:
    sid: str
    x: Any
    cache: Any
    kw: dict
    thread: threading.Thread
    from_fc: bool
    kind: str = "thinker"              # thinker | perception | codec
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    rnnt: Any = None                   # RNN-T overlap: (frame, state) captured at park time


def _rnnt_inputs_from_caller():
    """(rnnt_frame, rnnt_state) of the infer_one_step frame that is calling the Thinker, computed exactly as NVIDIA's
    code does right after it (W: rnnt_enc[:, emb_idx, :]); None when not available (no RNN-T, other caller)."""
    import sys
    f = sys._getframe(2)
    while f is not None and f.f_code.co_name != "infer_one_step":
        f = f.f_back
    if f is None:
        return None
    L = f.f_locals
    state, asr_emb = L.get("rnnt_partial_hypotheses"), L.get("asr_emb")
    w = L.get("self")
    if state is None or asr_emb is None or w is None or not isinstance(state, dict):
        return None
    try:
        pc = L.get("perception_cache")
        emb_idx = L["frame_offset"] if (w.use_perception_cache and pc is not None and pc.is_initialized()) \
            else L["current_frame_index"]
        emb_idx = min(emb_idx, asr_emb.shape[1] - 1)
    except (KeyError, AttributeError):
        return None
    return asr_emb[:, emb_idx, :], state


def _record_stream_all(x, stream) -> None:
    if isinstance(x, torch.Tensor):
        x.record_stream(stream)
    elif isinstance(x, dict):
        for v in x.values():
            _record_stream_all(v, stream)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _record_stream_all(v, stream)


class _Worker:
    """One thread per session; runs jobs handed over by the coordinator, parks at Thinker calls."""

    def __init__(self, rz: "RendezvousEngine", sid: str):
        self.rz, self.sid = rz, sid
        self.state = "idle"            # idle | running | parked | done | failed | stopped
        self.job: Optional[Callable[[], Any]] = None
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.req: Optional[_Req] = None
        self.go = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"rz-{sid}", daemon=True)
        self.thread.start()

    def _loop(self):
        self.rz.multi.store.bind_thread(self.sid)          # routed fields of this thread resolve to this session
        while True:
            self.go.wait()
            self.go.clear()
            job = self.job
            if job is None:
                break
            try:
                r = job()
                with self.rz.cv:
                    self.result, self.state = r, "done"
                    self.rz.cv.notify_all()
            except BaseException as e:  # noqa: BLE001 - reported to the coordinator, never swallowed
                with self.rz.cv:
                    self.error, self.state = e, "failed"
                    self.rz.cv.notify_all()
        with self.rz.cv:
            self.state = "stopped"
            self.rz.cv.notify_all()


class RendezvousEngine:
    """N sessions on one ReferenceEngine (through MultiSessionEngine) with the Thinker batched by a backend
    implementing batch_contract.ThinkerBatchBackend. Driver-thread API (open/run_round/pump/close/close_all)."""

    def __init__(self, multi, backend, thinker_owner=None, step_timeout_s: float = 60.0,
                 batch_perception: bool = False, perception_batch_mel: bool = True,
                 perception_graphs: bool = False, batch_codec: bool = False, batch_eartts: bool = False,
                 eartts_graphs: bool = True, eartts_precapture: bool = False,
                 fc_budget: Optional[int] = None, fc_force_after_s: float = 1.0, fc_merge: bool = True,
                 rnnt_overlap: Optional[bool] = None):
        self.multi, self.backend = multi, backend
        # FC budget (VAL3): max FC-thread Thinker requests served INSIDE one round, summed over every pump() of the
        # round; None = unlimited (legacy). Outside rounds (idle pump between rounds, open/close) service is unlimited.
        # Anti-deadlock escape: if the coordinator has waited fc_force_after_s for a session that does not park while
        # FC requests are queued, one more request is served anyway (counted as forced).
        self.fc_budget, self.fc_force_after_s = fc_budget, fc_force_after_s
        self._round_fc: Optional[dict] = None
        self._between_fc = 0
        self.fc_rounds: list[dict] = []
        # FC merge (default ON since L40S A/B 2026-11-13; fc_merge=False = legacy fallback): when the round runs its batched Thinker step, the FC-thread decode requests already queued
        # for ATTACHED sessions join the same backend.step as extra rows (their session's main thread cannot use the
        # row while its FC thread owns the Thinker), instead of B=1 step_one calls served by pump().
        self.fc_merge = fc_merge
        self.engine = multi.engine
        self.owner = thinker_owner if thinker_owner is not None else self.engine.wrapper.model.stt_model
        self.timeout = step_timeout_s
        self.cv = threading.Condition()
        self.coord = threading.current_thread()
        self.workers: dict[str, _Worker] = {}
        self.fc_q: "queue.Queue[_Req]" = queue.Queue()
        self.thinker_owner: dict[str, threading.Thread] = {}     # explicit ownership of each session's Thinker
        self.phase: dict[str, Phase] = {}
        self.failed: dict[str, BaseException] = {}
        self.stats = {"rounds": 0, "batched_calls": 0, "batch_hist": {}, "b1_calls": 0, "fc_b1_calls": 0,
                      "attach": 0, "detach": 0, "reattach_after_b1": 0}
        self.calls: list[dict] = []                   # audit: one entry per Thinker execution
        self.job_log: list[dict] = []                 # one entry per session job: start/end, frames by route, waits
        # opt-in (V2_HANDOFF_TRACE=1): one record per baton hand-off = host time a session thread runs between two
        # parks (from its resume to its next park / job end), keyed by the phase it resumed from and parked at
        import os as _os
        self.handoffs: Optional[list] = [] if _os.environ.get("V2_HANDOFF_TRACE") == "1" else None
        self._inner = self.owner.forward              # whatever is installed (backend's B=1 / upstream)
        self._had_forward = "forward" in self.owner.__dict__
        self._intercept = self._make_interceptor()
        self.owner.forward = self._intercept
        self._wrap_fc()
        # RNN-T overlap (release preset; None = env V2_RNNT_OVERLAP=1): run each parked session's RNN-T decode step
        # (user-ASR / turn-taking input) on a side stream WHILE the batched Thinker runs (see _rnnt_prefetch); the
        # wrapper's own call then takes the result. Bit-identical outputs; needs CUDA (off otherwise).
        if rnnt_overlap is None:
            rnnt_overlap = _os.environ.get("V2_RNNT_OVERLAP") == "1"
        self.rnnt_overlap = bool(rnnt_overlap) and torch.cuda.is_available()
        self._rnnt_pre: dict = {}
        self.rnnt_stats = {"prefetched": 0, "used": 0, "miss": 0, "skipped": 0}
        if self.rnnt_overlap:
            self._wrap_rnnt()
        self.pbatch = None
        if batch_perception:
            self._wrap_perception(perception_batch_mel, perception_graphs)
        self._codec = None
        if batch_codec:
            self._wrap_codec()
        self.ebatch = None
        self._tts_seed: dict = {}
        if batch_eartts:
            self._wrap_eartts(eartts_graphs, eartts_precapture)
        self._serving = False                         # coordinator executing a Thinker request (capture barrier)
        from mynah_v2.model.capture_guard import GUARD
        self._guard = GUARD
        GUARD.add_provider(self.quiet_for)

    # ------------------------------------------------------------------------------------------------ capture barrier
    def quiet_for(self, me: threading.Thread) -> Optional[str]:
        """capture_guard provider: None when ``me`` is the only thread of this engine that may issue GPU work right
        now, else the reason. Captures are allowed only from the coordinator or the session worker holding the baton:
        then no other worker is running, the coordinator is not serving, no session is in an FC phase (its FC thread
        and the upstream perception/reminder threads run free) and no FC Thinker request is queued. None of these
        can start while ``me`` captures (only the coordinator hands out the baton; FC threads are only spawned by the
        running worker itself). Helper threads are also scanned by name by the guard."""
        worker_threads = {w.thread for w in self.workers.values()}
        if me is not self.coord and me not in worker_threads:
            return f"non_session_thread:{me.name}"
        if me is not self.coord and self._serving:
            return "coordinator_serving"
        for sid, w in list(self.workers.items()):
            if w.thread is not me and w.state == "running":
                return f"worker_running:{sid}"
        for sid, t in list(self.thinker_owner.items()):
            if t is not None:
                return f"fc_phase:{sid}"
        if not self.fc_q.empty():
            return "fc_request_queued"
        return None

    # ------------------------------------------------------------------------------------------------ hooks
    def _make_interceptor(self):
        def forward(input_embeds, cache=None, **kw):
            t = threading.current_thread()
            if t is self.coord:
                return self._inner(input_embeds, cache=cache, **kw)
            sid, from_fc = self._sid_of(t)
            own = self.thinker_owner.get(sid)
            if own is not None and own is not t:
                raise OwnershipError(f"Thinker of {sid!r} is owned by {own.name} ({self.phase.get(sid)}); "
                                     f"request from {t.name} refused (would advance it twice)")
            req = _Req(sid, input_embeds, cache, kw, t, from_fc)
            req.t_park = time.perf_counter()
            if self.rnnt_overlap and not from_fc:
                req.rnnt = _rnnt_inputs_from_caller()
            # grad / inference mode are THREAD-LOCAL in torch: the coordinator must run the Thinker in the mode of the
            # requesting thread (the wrapper runs without grad). Otherwise every forward on the coordinator builds and
            # keeps an autograd graph (round 9b, Ada: +~120 MB per chunk → OOM at chunk 42).
            req.grad, req.inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
            if from_fc:
                self.fc_q.put(req)
                with self.cv:
                    self.cv.notify_all()
            else:
                w = self.workers[sid]
                with self.cv:
                    w.req, w.state = req, "parked"
                    self.cv.notify_all()
            # the COORDINATOR enforces hand-off timeouts; a parked thread may legitimately wait while the coordinator
            # times out OTHER sessions, so its own wait is only a zombie guard (must exceed any round's timeouts)
            park = max(60.0, 10 * self.timeout * max(1, len(self.workers)))
            if not req.done.wait(park):
                raise RendezvousError(f"Thinker request of {sid!r} not served within {park}s (coordinator gone?)")
            # worker: `done` is set by the coordinator only AFTER it switched this session back in (resume)
            if req.error is not None:
                raise req.error
            return req.result
        forward.__v2_rendezvous__ = True
        return forward

    # ------------------------------------------------------------------------------------------------ RNN-T overlap
    # NVIDIA's infer_one_step runs, per session and frame, AFTER the Thinker: forced turn-taking (Thinker output),
    # then _rnnt_step(asr_emb frame, rnnt_state) (user-side RNN-T greedy decode: joint + predictor, B=1, with host
    # syncs), then the RNN-T turn-taking. _rnnt_step reads only the perception frame and the RNN-T state left by the
    # previous frame (forced turn-taking only flips rnnt_state['forced_bos'] in place, which _rnnt_step carries by
    # reference), so the same call can run earlier: on a side stream while the batched Thinker occupies the GPU.
    # Same ops, same batch size 1, same inputs -> same values. The wrapper's call takes the prefetched result only
    # if it gets the very same state object and frame (and every carried state value is unchanged), else it runs
    # the original step.
    def _rnnt_prefetch(self, batch, pre_ev) -> None:
        w = self.engine.wrapper
        side = getattr(self, "_rnnt_stream", None)
        if side is None:
            side = self._rnnt_stream = torch.cuda.Stream()
        for r in batch:
            inp = getattr(r, "rnnt", None)
            if inp is None:
                self.rnnt_stats["skipped"] += 1
                continue
            frame, state = inp
            snap = {k: v for k, v in state.items()}
            with torch.cuda.stream(side):
                side.wait_event(pre_ev)
                with torch.no_grad():
                    res = self._rnnt_inner(frame, state)
                ev = torch.cuda.Event()
                ev.record(side)
            self._rnnt_pre[r.sid] = (state, frame.data_ptr(), tuple(frame.shape), snap, res, ev)
            self.rnnt_stats["prefetched"] += 1

    def _wrap_rnnt(self):
        w = self.engine.wrapper
        self._rnnt_inner = w._rnnt_step
        self._rnnt_had = "_rnnt_step" in w.__dict__
        workers = self.workers

        def step(encoder_frame, rnnt_state):
            t = threading.current_thread()
            sid = next((s for s, wk in workers.items() if wk.thread is t), None)
            pre = self._rnnt_pre.pop(sid, None) if sid is not None else None
            if pre is not None:
                state, ptr, shape, snap, res, ev = pre
                same = (rnnt_state is state and encoder_frame.data_ptr() == ptr and tuple(encoder_frame.shape) == shape
                        and rnnt_state.keys() == snap.keys() and all(rnnt_state[k] is v for k, v in snap.items()))
                if same:
                    main = torch.cuda.current_stream()
                    main.wait_event(ev)
                    new_state, is_blank = res
                    _record_stream_all(new_state, main)
                    is_blank.record_stream(main)
                    self.rnnt_stats["used"] += 1
                    return new_state, is_blank
                self.rnnt_stats["miss"] += 1
            return self._rnnt_inner(encoder_frame, rnnt_state)
        self._rnnt_wrapper = step
        w._rnnt_step = step

    def _wrap_fc(self):
        """FC background thread: takes Thinker ownership of its session for its whole duration (explicit phase)."""
        w = self.engine.wrapper
        inner = w._run_fc_async_steps
        sid_of_stream = self.multi._sid_of_stream

        def fc(*args, **kw):
            sid = sid_of_stream.get(kw.get("stream_id"))
            t = threading.current_thread()
            if sid is None or t is self.coord or (sid in self.workers and t is self.workers[sid].thread):
                return inner(*args, **kw)              # driver-side FC call: ownership unchanged
            with self.cv:
                prev = self.thinker_owner.get(sid)
                self.thinker_owner[sid], self.phase[sid] = t, Phase.FC_FORCED
            try:
                return inner(*args, **kw)
            finally:
                with self.cv:
                    self.thinker_owner[sid], self.phase[sid] = prev, Phase.DECODE
                    self.cv.notify_all()
        fc.__wrapped_v2__ = getattr(inner, "__wrapped_v2__", inner)
        self._fc_prev = inner
        w._run_fc_async_steps = fc

    def _wrap_perception(self, batch_mel: bool, graphs: bool = False):
        from mynah_v2.model.perception_batch import BatchedPerception, step_kwargs
        pcm = getattr(self.engine.wrapper, "perception_cache_mgr", None)
        if pcm is None:
            raise RendezvousError("batch_perception needs the wrapper's perception cache (use_perception_cache)")
        self.pbatch = BatchedPerception(pcm, batch_mel=batch_mel, graphs=graphs)   # run() looked up per call (CompTimer)
        # graphs per row count are captured HERE, before any session thread exists (quiet point; never lazily)
        self.perception_capture = self.pbatch.precapture(int(getattr(self.backend, "max_rows", 2))) if graphs else {}
        self._pcm, self._pcm_had = pcm, "step" in pcm.__dict__
        self._pcm_inner = pcm.step                     # whatever is installed (upstream / CompTimer wrapper)
        self.stats["perception"] = {"b1": 0, "batched": 0, "batch_hist": {}, "passthrough": 0}
        rz = self

        def step(*args, **kw):
            t = threading.current_thread()
            sid = next((s for s, w in rz.workers.items() if w.thread is t), None)
            d = step_kwargs(args, kw)
            if sid is None or d is None:               # coordinator, FC thread, unexpected signature: unchanged
                rz.stats["perception"]["passthrough"] += 1
                return rz._pcm_inner(*args, **kw)
            req = _Req(sid, None, None, d, t, False, kind="perception")
            req.t_park = time.perf_counter()
            req.grad, req.inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
            w = rz.workers[sid]
            with rz.cv:
                w.req, w.state = req, "parked"
                rz.cv.notify_all()
            park = max(60.0, 10 * rz.timeout * max(1, len(rz.workers)))
            if not req.done.wait(park):
                raise RendezvousError(f"perception request of {sid!r} not served within {park}s (coordinator gone?)")
            if req.error is not None:
                raise req.error
            return req.result
        step.__v2_rendezvous__ = True
        self._pcm_intercept = step
        pcm.step = step

    @_gpu_exclusive
    def _serve_perception(self, reqs: list[_Req]):
        """All perception requests of the round: one batched call per (grad, inference) mode and frames-per-chunk,
        a lone request through the installed B=1 step (upstream graph: byte-identical to the unbatched path)."""
        groups: dict[tuple, list[_Req]] = {}
        for r in reqs:
            groups.setdefault((r.grad, r.inf, int(r.kw["num_frames_per_chunk"])), []).append(r)
        ps = self.stats["perception"]
        for (grad, inf, _), rs in groups.items():
            t0 = time.perf_counter()
            with torch.inference_mode(inf), torch.set_grad_enabled(grad):
                if len(rs) == 1:
                    r = rs[0]
                    try:
                        r.result = self._pcm_inner(**r.kw)
                    except BaseException as e:  # noqa: BLE001 - delivered to the requesting thread
                        r.error = e
                    ps["b1"] += 1
                else:
                    try:
                        outs = self.pbatch.run([r.kw for r in rs])
                        for r, o in zip(rs, outs):
                            r.result = o
                    except BaseException as e:  # noqa: BLE001 - a batched failure fails every member
                        for r in rs:
                            r.error = e
                    ps["batched"] += 1
                    ps["batch_hist"][len(rs)] = ps["batch_hist"].get(len(rs), 0) + 1
            for r in rs:
                r.route, r.exec_ms = "perc", (time.perf_counter() - t0) * 1e3

    def _wrap_codec(self):
        from mynah_v2.model.codec_batch import decode_kwargs
        codec = self.engine.wrapper.model.tts_model.audio_codec
        self._codec, self._codec_had = codec, "decode" in codec.__dict__
        self._codec_inner = codec.decode               # whatever is installed (upstream / CompTimer / codec_fc_d2h)
        self.stats["codec"] = {"b1": 0, "batched": 0, "batch_hist": {}, "passthrough": 0, "unstackable": 0}
        rz = self

        def decode(*args, **kw):
            t = threading.current_thread()
            sid = next((s for s, w in rz.workers.items() if w.thread is t), None)
            d = decode_kwargs(args, kw)
            if sid is None or d is None or d["cache"] is None:   # coordinator, FC thread, fallback path: unchanged
                rz.stats["codec"]["passthrough"] += 1
                return rz._codec_inner(*args, **kw)
            req = _Req(sid, None, None, d, t, False, kind="codec")
            req.t_park = time.perf_counter()
            req.grad, req.inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
            req.amp = _autocast_state()
            w = rz.workers[sid]
            with rz.cv:
                w.req, w.state = req, "parked"
                rz.cv.notify_all()
            park = max(60.0, 10 * rz.timeout * max(1, len(rz.workers)))
            if not req.done.wait(park):
                raise RendezvousError(f"codec request of {sid!r} not served within {park}s (coordinator gone?)")
            if req.error is not None:
                raise req.error
            return req.result
        decode.__v2_rendezvous__ = True
        self._codec_intercept = decode
        codec.decode = decode

    @_gpu_exclusive
    def _serve_codec(self, reqs: list[_Req]):
        """All codec requests of the round: stackable ones (same codes/cache layout, same modes) as one batched decode,
        everything else alone through the installed decode."""
        from mynah_v2.model.codec_batch import batched_decode, stack_key
        cs = self.stats["codec"]
        groups: dict[tuple, list[_Req]] = {}
        alone: list[_Req] = []
        for r in reqs:
            k = stack_key(r.kw)
            if k is None:
                alone.append(r)
                cs["unstackable"] += 1
            else:
                groups.setdefault((r.grad, r.inf, r.amp, k), []).append(r)
        runs = [[r] for r in alone] + list(groups.values())
        for rs in runs:
            r0 = rs[0]
            t0 = time.perf_counter()
            en, dt = r0.amp
            with torch.inference_mode(r0.inf), torch.set_grad_enabled(r0.grad), \
                    torch.autocast("cuda", dtype=dt, enabled=True) if en else _nullctx():
                if len(rs) == 1:
                    try:
                        r0.result = self._codec_inner(**r0.kw)
                    except BaseException as e:  # noqa: BLE001 - delivered to the requesting thread
                        r0.error = e
                    cs["b1"] += 1
                else:
                    try:
                        for r, o in zip(rs, batched_decode(self._codec_inner, [r.kw for r in rs])):
                            r.result = o
                    except BaseException as e:  # noqa: BLE001 - a batched failure fails every member
                        for r in rs:
                            r.error = e
                    cs["batched"] += 1
                    cs["batch_hist"][len(rs)] = cs["batch_hist"].get(len(rs), 0) + 1
            for r in rs:
                r.route, r.exec_ms = "codec", (time.perf_counter() - t0) * 1e3

    def _wrap_eartts(self, graphs: bool, precapture: bool):
        from mynah_v2.model.eartts_batch import BatchedEarTTS, step_kwargs
        fsg = getattr(self.engine, "_v2_full_graph", None)
        if fsg is None:
            raise RendezvousError("batch_eartts needs the eartts_graph_full experiment (static KV slots)")
        self.ebatch = BatchedEarTTS(fsg, graphs=graphs)     # pools the slots now (before any session is open)
        # batched graphs are captured HERE (quiet point) when asked; otherwise lazily, only through the capture guard
        self.eartts_capture = self.ebatch.precapture() if (graphs and precapture) else {}
        duplex = self.engine.wrapper.model.tts_model
        self._duplex, self._tts_had = duplex, "infer_codes_one_step" in duplex.__dict__
        self._tts_inner = duplex.infer_codes_one_step   # whatever is installed (fsg.step / CompTimer wrapper)
        self.stats["eartts"] = {"b1": 0, "batched": 0, "batch_hist": {}, "passthrough": 0, "unbatchable": 0}
        rz = self

        def step(*args, **kw):
            t = threading.current_thread()
            sid = next((s for s, w in rz.workers.items() if w.thread is t), None)
            d = step_kwargs(args, kw)
            if sid is None or d is None:               # coordinator, FC thread, reminder, odd signature: unchanged
                rz.stats["eartts"]["passthrough"] += 1
                return rz._tts_inner(*args, **kw)
            req = _Req(sid, None, None, d, t, False, kind="eartts")
            req.t_park = time.perf_counter()
            req.grad, req.inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
            req.amp = _autocast_state()
            w = rz.workers[sid]
            with rz.cv:
                w.req, w.state = req, "parked"
                rz.cv.notify_all()
            park = max(60.0, 10 * rz.timeout * max(1, len(rz.workers)))
            if not req.done.wait(park):
                raise RendezvousError(f"eartts request of {sid!r} not served within {park}s (coordinator gone?)")
            if req.error is not None:
                raise req.error
            return req.result
        step.__v2_rendezvous__ = True
        self._tts_intercept = step
        duplex.infer_codes_one_step = step

    def _with_seed(self, r, fn, *a, **kw):
        """Run fn with the requesting session's EarTTS seed pending (a DynamicCache → StaticKV conversion inside
        takes it; eartts_rng)."""
        from mynah_v2.model import eartts_rng
        prev, eartts_rng.CTX.seed = eartts_rng.CTX.seed, self._tts_seed.get(r.sid)
        try:
            return fn(*a, **kw)
        finally:
            eartts_rng.CTX.seed = prev

    @_gpu_exclusive
    def _serve_eartts(self, reqs: list[_Req]):
        """All EarTTS requests of the round: n >= 2 batchable ones (static KV slot, same signature and modes) as ONE
        BatchedEarTTS.run; everything else (first step of a session = DynamicCache conversion, past lmax, ...) alone
        through the installed per-session step."""
        es = self.stats["eartts"]
        groups: dict[tuple, list[_Req]] = {}
        alone: list[_Req] = []
        conv = getattr(getattr(self.ebatch, "f", None), "convert_for_batch", None)
        for r in reqs:
            k = self.ebatch.group_key(r.kw)
            if k is None and conv is not None and len(reqs) > 1 and self._with_seed(r, conv, r.kw):   # first step →
                es["converted"] = es.get("converted", 0) + 1                      # slot now, batched below
                k = self.ebatch.group_key(r.kw)
            if k is None:
                alone.append(r)
                es["unbatchable"] += 1
            else:
                groups.setdefault((r.grad, r.inf, r.amp, k), []).append(r)
        for g in [g for g in groups.values() if len(g) == 1]:
            alone += g
        runs = [[r] for r in alone] + [g for g in groups.values() if len(g) > 1]
        for rs in runs:
            r0 = rs[0]
            t0 = time.perf_counter()
            en, dt = r0.amp
            with torch.inference_mode(r0.inf), torch.set_grad_enabled(r0.grad), \
                    torch.autocast("cuda", dtype=dt, enabled=True) if en else _nullctx():
                if len(rs) == 1:
                    try:
                        r0.result = self._with_seed(r0, self._tts_inner, **r0.kw)
                    except BaseException as e:  # noqa: BLE001 - delivered to the requesting thread
                        r0.error = e
                    es["b1"] += 1
                else:
                    try:
                        for r, o in zip(rs, self.ebatch.run([r.kw for r in rs])):
                            r.result = o
                    except BaseException as e:  # noqa: BLE001 - a batched failure fails every member
                        for r in rs:
                            r.error = e
                    es["batched"] += 1
                    es["batch_hist"][len(rs)] = es["batch_hist"].get(len(rs), 0) + 1
            for r in rs:
                r.route, r.exec_ms = "tts", (time.perf_counter() - t0) * 1e3

    def _sid_of(self, t):
        for sid, w in self.workers.items():
            if w.thread is t:
                return sid, False
        sid = getattr(self.multi.store._tl, "sid", None)        # FC threads are bound by MultiSessionEngine's router
        if sid is None:
            raise RendezvousError(f"Thinker call from unknown thread {t.name}")
        return sid, True

    # ------------------------------------------------------------------------------------------------ Thinker exec
    def _attached(self, sid):
        return sid in self.backend.positions()

    def _wrap_out(self, out: ThinkerOut, cache):
        d = {"text_logits": out.text_logits, "function_logits": out.function_logits, "cache": cache}
        if out.asr_logits is not None:
            d["asr_logits"] = out.asr_logits
        return d

    def _b1(self, req: _Req):
        """One B=1 Thinker execution on the coordinator thread (attached decode → step_one; anything else → the
        installed B=1 forward on the session's B=1 cache, detaching/re-attaching around it)."""
        decode = req.cache is not None and getattr(req.x, "dim", lambda: 0)() == 3 and req.x.shape[1] == 1 \
            and not req.kw
        if self._attached(req.sid):
            if decode:
                return self._wrap_out(self.backend.step_one(req.sid, req.x), req.cache)
            cache = self.backend.detach(req.sid)
            self.stats["detach"] += 1
            out = self._inner(req.x, cache=cache, **req.kw)
            self._maybe_attach(req.sid, out.get("cache", cache) if isinstance(out, dict) else cache)
            self.stats["reattach_after_b1"] += 1
            return out
        out = self._inner(req.x, cache=req.cache, **req.kw)
        if req.cache is not None:
            self._maybe_attach(req.sid, out.get("cache", req.cache) if isinstance(out, dict) else req.cache)
        return out

    def _maybe_attach(self, sid, cache):
        if self._attached(sid) or cache is None or int(getattr(cache, "_v2_pos", 0)) <= 0:
            return
        if getattr(cache, "_v2_past_lmax", False):
            return
        if len(self.backend.positions()) >= self.backend.max_rows:
            return                                             # stays B=1 (eager) until a row frees up
        self.backend.attach(sid, cache)
        self.stats["attach"] += 1

    @_gpu_exclusive
    def _serve(self, reqs: list[_Req]):
        """Execute a set of requests: attached single-frame decode requests in ONE batched forward, the rest B=1,
        each group under the grad/inference mode of its requesting threads."""
        modes = {(r.grad, r.inf) for r in reqs}
        if len(modes) > 1:                                  # never mix modes in one forward: serve per mode
            for m in modes:
                self._serve([r for r in reqs if (r.grad, r.inf) == m])
            return
        grad, inf = modes.pop()
        prev, self._serving = self._serving, True
        try:
            with torch.inference_mode(inf), torch.set_grad_enabled(grad):
                self._serve_same_mode(reqs)
        finally:
            self._serving = prev

    def _take_fc_for_merge(self) -> list:
        """Queued FC requests that can ride the round's batched Thinker step (fc_merge); others stay queued."""
        if not self.fc_merge:
            return []
        take, keep = [], []
        while True:
            try:
                r = self.fc_q.get_nowait()
            except queue.Empty:
                break
            ok = (self._attached(r.sid) and r.cache is not None and getattr(r.x, "dim", lambda: 0)() == 3
                  and r.x.shape[1] == 1 and not r.kw)
            (take if ok else keep).append(r)
        for r in keep:
            self.fc_q.put(r)
        if take:
            self.stats["fc_merged"] = self.stats.get("fc_merged", 0) + len(take)
            if self._round_fc is not None:
                self._round_fc["merged"] = self._round_fc.get("merged", 0) + len(take)
        return take

    def _serve_same_mode(self, reqs: list[_Req]):
        batch = [r for r in reqs if (self.fc_merge or not r.from_fc) and self._attached(r.sid) and r.cache is not None
                 and r.x.dim() == 3 and r.x.shape[1] == 1 and not r.kw]
        rest = [r for r in reqs if r not in batch]
        for r in reqs:
            r.route = "b2" if (len(batch) >= 2 and r in batch) else ("fc" if r.from_fc else "b1")
        if len(batch) >= 2:
            t0 = time.perf_counter()
            pre_ev = None
            if self.rnnt_overlap:
                pre_ev = torch.cuda.Event()
                pre_ev.record()                        # inputs of the RNN-T steps are ready before the Thinker
            try:
                outs = self.backend.step({r.sid: r.x for r in batch})
                if pre_ev is not None:
                    self._rnnt_prefetch(batch, pre_ev)
                for r in batch:
                    r.result = self._wrap_out(outs[r.sid], r.cache)
            except BaseException as e:  # noqa: BLE001 - a batched failure fails every member (not attributable)
                for r in batch:
                    r.error = e
            n = len(batch)
            for r in batch:
                r.exec_ms = (time.perf_counter() - t0) * 1e3
            self.stats["batched_calls"] += 1
            self.stats["batch_hist"][n] = self.stats["batch_hist"].get(n, 0) + 1
            self.calls.append({"sids": [r.sid for r in batch], "batched": True, "ms": (time.perf_counter() - t0) * 1e3})
        else:
            rest = batch + rest
        for r in rest:
            t0 = time.perf_counter()
            try:
                r.result = self._b1(r)
            except BaseException as e:  # noqa: BLE001 - delivered to the requesting thread
                r.error = e
            r.exec_ms = (time.perf_counter() - t0) * 1e3
            self.stats["fc_b1_calls" if r.from_fc else "b1_calls"] += 1
            self.calls.append({"sids": [r.sid], "batched": False, "fc": r.from_fc,
                               "ms": (time.perf_counter() - t0) * 1e3})
        for r in reqs:
            if r.from_fc:         # FC decode breakdown per session: queue wait (park → served start) vs execution
                fc = self.stats.setdefault("fc_by_sid", {}).setdefault(r.sid, {"n": 0, "queue_ms": [], "exec_ms": []})
                fc["n"] += 1
                fc["queue_ms"].append(round((time.perf_counter() - r.t_park) * 1e3 - r.exec_ms, 2))
                fc["exec_ms"].append(round(r.exec_ms, 2))
                r.done.set()      # FC threads run free; session threads are resumed one at a time by _run_until_parked

    def pump(self, until: Optional[float] = None, force: bool = False) -> None:
        """Serve FC-thread Thinker requests (B=1) until ``until`` (monotonic) or until the queue is empty if None.
        Inside a round the FC budget applies (``force`` serves one request past it)."""
        while True:
            rf = self._round_fc
            if rf is not None:
                rf["q_max"] = max(rf["q_max"], self.fc_q.qsize())
                # fc_merge: inside a round FC requests wait for the round's batched Thinker step (merge); budget: cap
                if self.fc_merge or (self.fc_budget is not None and rf["served"] >= self.fc_budget):
                    if not force or self.fc_q.empty():
                        if not self.fc_q.empty():
                            rf["deferred_polls"] += 1
                        return
                    force = False
                    rf["forced"] += 1
            elif until is not None and self.fc_budget is not None and time.monotonic() >= until:
                return                                 # budgeted: between rounds only idle time serves FC requests
            try:
                req = self.fc_q.get(timeout=max(0.0, until - time.monotonic()) if until else 0.0)
            except queue.Empty:
                return
            self._serve([req])
            if rf is not None:
                rf["served"] += 1
            else:
                self._between_fc += 1
            if until is None:
                continue

    # ------------------------------------------------------------------------------------------------ hand-off
    def _run_until_parked(self, sid: str, job=None) -> str:
        """Give the baton to ``sid`` (switch its state in, start a job or resume it) and wait until it parks or ends,
        serving FC requests meanwhile. Returns the worker state."""
        w = self.workers[sid]
        prev = "start" if job is not None else getattr(w.req, "kind", "thinker")
        th0 = time.perf_counter()
        self._handoff(sid, w, job)
        th1 = time.perf_counter()
        t_wait = time.monotonic()
        deadline = t_wait + self.timeout
        while True:
            self.pump(force=time.monotonic() - t_wait > self.fc_force_after_s)
            with self.cv:
                if w.state in ("parked", "done", "failed"):
                    if self.handoffs is not None:
                        nxt = getattr(w.req, "kind", "thinker") if w.state == "parked" else w.state
                        self.handoffs.append((sid, prev or "thinker", nxt or "thinker", round((th1 - th0) * 1e3, 3),
                                              round((time.perf_counter() - th1) * 1e3, 3)))
                    if w.state != "parked" and getattr(w, "log", None) is not None:
                        w.log["t_end"], w.log["state"] = time.perf_counter(), w.state
                        self.job_log.append(w.log)
                        w.log = None
                    return w.state
                left = deadline - time.monotonic()
                if left <= 0:
                    if getattr(w, "log", None) is not None:
                        w.log["t_end"], w.log["state"] = time.perf_counter(), "timeout"
                        self.job_log.append(w.log)
                        w.log = None
                    w.state = "failed"
                    w.error = RendezvousError(f"session {sid!r} did not park/finish within {self.timeout}s")
                    return "failed"
                self.cv.wait(min(left, 0.05))

    def _resume_order(self, sids: list[str]) -> list[str]:
        """Resume the currently swapped-in session first.

        A worker keeps its session state swapped in while it is parked.  The
        old fixed order therefore paid an unnecessary full capture/restore
        pair at the start of every stage: if B was active and the requests
        were ``[A, B]``, resuming A then B caused B→A→B swaps.  Keeping the
        active worker first reduces that to B→A while preserving the set of
        requests, their phase ordering, and the batch composition.
        """
        active = self.multi.store.active
        if active not in sids:
            return sids
        return [active, *[sid for sid in sids if sid != active]]

    def _resume_parked(self, sids: list[str], state: dict[str, str]) -> None:
        """Resume a phase's parked workers in swap-minimising order."""
        for sid in self._resume_order(sids):
            state[sid] = self._run_until_parked(sid)

    @_gpu_exclusive
    def _handoff(self, sid, w, job):
        """Switch ``sid`` in and give it the baton. Under the capture-guard lock: while any thread holds an open
        capture (capture_begin → capture_end), no session is switched in or resumed (not even after a timeout)."""
        self.multi._switch(sid)
        with self.cv:
            if job is not None:
                w.job, w.result, w.error, w.state = job, None, None, "running"
                w.log = {"sid": sid, "t_start": time.perf_counter(), "b2": 0, "b1": 0, "wait_ms": 0.0, "exec_ms": 0.0}
            else:
                w.state = "running"
        if job is None:
            r = w.req
            w.log[getattr(r, "route", "b1")] = w.log.get(getattr(r, "route", "b1"), 0) + 1
            pre = {"perception": "perc_", "codec": "codec_", "eartts": "tts_"}.get(r.kind, "")   # Thinker keys unchanged
            w.log[pre + "wait_ms"] = w.log.get(pre + "wait_ms", 0.0) + (time.perf_counter() - r.t_park) * 1e3
            w.log[pre + "exec_ms"] = w.log.get(pre + "exec_ms", 0.0) + getattr(r, "exec_ms", 0.0)
            w.req.done.set()                               # deliver the Thinker result = resume the parked thread
        else:
            w.go.set()                                     # start a new job

    def _fail(self, sid: str, err: BaseException):
        self.failed[sid] = err
        if self._attached(sid):
            try:
                self.backend.detach(sid)
            except BaseException:  # noqa: BLE001
                pass
        self.thinker_owner.pop(sid, None)

    def run_round(self, jobs: dict[str, Callable[[], Any]]) -> dict[str, Any]:
        """Run one job per session (e.g. ``lambda: multi.step(sid, chunk)``) with the Thinker batched across sessions.
        Returns {sid: result or exception}. Sessions without a job are not waited for."""
        self.stats["rounds"] += 1
        out: dict[str, Any] = {}
        outer = self._round_fc is None                 # open() runs a nested-free round of its own
        if outer:
            self._round_fc = {"served": 0, "forced": 0, "deferred_polls": 0, "q_start": self.fc_q.qsize(),
                              "q_max": self.fc_q.qsize(), "between_before": self._between_fc}
            self._between_fc = 0
        try:
            res = self._run_round(jobs, out)
            if outer and self.fc_merge and not self._round_fc.get("merged") and not self.fc_q.empty():
                rf = self._round_fc                    # no batched Thinker step took them: legacy B=1 service
                self._round_fc = None
                n0 = self._between_fc
                try:
                    self.pump()
                finally:
                    self._round_fc = rf
                    rf["fallback_b1"] = self._between_fc - n0
                    self._between_fc = n0
            return res
        finally:
            if outer:
                rf, self._round_fc = self._round_fc, None
                rf["q_end"] = self.fc_q.qsize()
                self.fc_rounds.append(rf)

    def _run_round(self, jobs, out):
        live = [s for s in jobs if s in self.workers and s not in self.failed]
        state = {s: self._run_until_parked(s, job=jobs[s]) for s in live}
        while True:
            parked = [s for s in live if state[s] == "parked"]
            if not parked:
                break
            perc = [s for s in parked if self.workers[s].req.kind == "perception"]
            if perc:                                       # perception first: Thinker-parked sessions keep waiting
                self._serve_perception([self.workers[s].req for s in perc])
                self._resume_parked(perc, state)
                continue
            think = [s for s in parked if self.workers[s].req.kind == "thinker"]
            if think:                                      # codec-parked sessions wait until no Thinker is pending
                self._serve([self.workers[s].req for s in think] + self._take_fc_for_merge())
                self._resume_parked(think, state)
                continue
            tts = [s for s in parked if self.workers[s].req.kind == "eartts"]
            if tts:                                        # EarTTS after the Thinker, before the codec
                self._serve_eartts([self.workers[s].req for s in tts])
                self._resume_parked(tts, state)
                continue
            self._serve_codec([self.workers[s].req for s in parked])
            self._resume_parked(parked, state)
        for s in live:
            w = self.workers[s]
            if state[s] == "failed":
                self._fail(s, w.error)
                out[s] = w.error
            else:
                out[s] = w.result
        for s in jobs:
            if s not in live:
                out[s] = self.failed.get(s, RendezvousError(f"session {s!r} is not open"))
        return out

    # ------------------------------------------------------------------------------------------------ sessions
    def open(self, sid: str, system_prompt: Optional[str], tools: Optional[dict] = None, seed: Optional[int] = None,
             guard: bool | dict = True):
        if sid in self.workers:
            raise RendezvousError(f"session {sid!r} already open")
        # registering the session (store record, unique stream id) happens on the driver; the prefill runs in the
        # session's own worker thread so its Thinker call goes through the coordinator like every other call
        stream = self.multi._next_stream
        self.multi._next_stream += 1
        self.multi._check_eartts_slots()
        self.multi.store.open(sid, seed=seed, overrides={"engine.stream_id": stream})
        import zlib                                    # EarTTS per-session noise seed (eartts_rng, if enabled)
        self._tts_seed[sid] = int(seed) if seed is not None else zlib.crc32(sid.encode())
        self.multi._stream_of[sid], self.multi._sid_of_stream[stream] = stream, sid
        self.workers[sid] = _Worker(self, sid)
        self.phase[sid] = Phase.PREFILL

        def job():
            if tools:
                self.engine.bind_session_tools(tools)
                if guard:                                # default ON: the session's tools behind its own guard
                    from mynah_v2.agent.call_guard import CallGuard
                    self.engine.set_session_guard(CallGuard(session_id=str(sid),
                                                            **(guard if isinstance(guard, dict) else {})))
            self.engine.open(system_prompt)
        res = self.run_round({sid: job})[sid]
        if isinstance(res, BaseException):
            raise res
        self.phase[sid] = Phase.DECODE
        return stream

    def pause(self, sid: str) -> None:
        """Free the session's backend row (its state goes back into its B=1 cache); it re-attaches lazily at its next
        Thinker call. Used when a session stops being stepped for a while (slot reuse {A+B} → {B+C} → {A+C})."""
        if self._attached(sid):
            self.backend.detach(sid)
            self.stats["detach"] += 1

    def step_jobs(self, chunks: dict[str, Any], is_last: Optional[dict] = None) -> dict[str, Any]:
        eng = self.engine
        return self.run_round({s: (lambda c=c, last=bool((is_last or {}).get(s)): eng.step(c, is_last=last))
                               for s, c in chunks.items()})

    def close(self, sid: str) -> None:
        w = self.workers.get(sid)
        if w is None:
            raise RendezvousError(f"unknown session {sid!r}")
        try:
            if self._attached(sid):
                try:
                    self.backend.detach(sid)
                    self.stats["detach"] += 1
                except Exception as e:  # noqa: BLE001 - recorded; must not block the teardown of other sessions
                    self.stats.setdefault("detach_errors", []).append(f"{sid}: {type(e).__name__}: {e}")
            ev = getattr(self.backend, "evicted", None)       # past-lmax sessions continue B=1 on their own cache;
            if isinstance(ev, dict) and ev.pop(sid, None) is not None:   # drop the backend's reference at close
                self.stats["evicted_closed"] = self.stats.get("evicted_closed", 0) + 1
            if sid not in self.failed:
                self.run_round({sid: self.engine.close})
        finally:
            try:
                self.multi.store.drop(sid) if sid in self.multi.store.sids() else None
            finally:
                self.multi._stream_of.pop(sid, None)
                w.job = None
                w.go.set()
                w.thread.join(5)
                del self.workers[sid]
                self.phase[sid] = Phase.CLOSED
                self.thinker_owner.pop(sid, None)

    def close_all(self) -> None:
        try:
            for sid in list(self.workers):
                self.close(sid)
        finally:
            self._guard.remove_provider(self.quiet_for)
            if self.owner.__dict__.get("forward") is self._intercept:
                if self._had_forward:
                    self.owner.forward = self._inner
                else:
                    del self.owner.forward
            if self.rnnt_overlap and self.engine.wrapper.__dict__.get("_rnnt_step") is getattr(self, "_rnnt_wrapper", None):
                if self._rnnt_had:
                    self.engine.wrapper._rnnt_step = self._rnnt_inner
                else:
                    del self.engine.wrapper._rnnt_step
            if getattr(self.engine.wrapper, "_run_fc_async_steps", None) is not self._fc_prev:
                self.engine.wrapper._run_fc_async_steps = self._fc_prev
            if self.ebatch is not None and self._duplex.__dict__.get("infer_codes_one_step") is self._tts_intercept:
                if self._tts_had:
                    self._duplex.infer_codes_one_step = self._tts_inner
                else:
                    del self._duplex.infer_codes_one_step
            if self._codec is not None and self._codec.__dict__.get("decode") is self._codec_intercept:
                if self._codec_had:
                    self._codec.decode = self._codec_inner
                else:
                    del self._codec.decode
            if self.pbatch is not None and self._pcm.__dict__.get("step") is self._pcm_intercept:
                if self._pcm_had:
                    self._pcm.step = self._pcm_inner
                else:
                    del self._pcm.step
