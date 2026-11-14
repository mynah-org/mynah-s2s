"""Per-session state isolation for sharing ONE loaded ReferenceEngine between several sessions (EXPERIMENTAL, opt-in).

Implements ``runtime.batch_contract.SessionStateStore`` (capture / restore / drop) for every piece of mutable state
that is NOT Thinker recurrent state. The single-session path never touches this module.

Two mechanisms, chosen per field by WHO touches it:
  * swap   — fields read/written only by the driver (main) thread inside ``generate_step``: ``restore`` writes the
             session's objects onto the owner, ``capture`` reads them back and leaves a ``Poisoned`` sentinel, so any
             access with no session swapped in raises instead of silently using another session's object.
  * routed — fields ALSO touched by the FC background thread (``_fc_async_thread_fn`` → ``_run_fc_async_steps`` →
             ``_apply_rnnt_turn_taking`` / tool executor / ``get_state_fn``), which keeps running while the driver
             steps other sessions. A per-instance subclass of the owner gets a data descriptor per field that resolves
             to the CURRENT session's record: the thread-bound session (FC threads, bound by ``stream_id``) else the
             swapped-in one. Swapping such a field would make A's FC thread write into B's state.

Thinker recurrent state (hybrid cache: attention K/V, Mamba conv/SSM; ThinkerGraph buffers) is NOT copied here: it is
reached through the session's own context object (``context.dynamic_cache``, travelling with the swapped
``pipeline.context_manager``) or, when the batched backend owns it, recorded only as a reference (``thinker_ref``).

RNG (contract I8): each session has its own CPU / CUDA default-generator / Python ``random`` state, saved by
``capture`` and set by ``restore`` (first restore: ``manual_seed(seed)`` when a seed is given). CUDA graphs captured
on the default generator (eartts_graph_full, eartts_graph_genstep) do not freeze an offset: each replay reads the
generator's current (seed, philox offset) and advances it by the captured increment, exactly like the eager kernels,
so set_rng_state before a session's step makes its replays draw from the session's own stream. Caveats (GPU-verify):
  * get/set_rng_state must not run while any thread is CAPTURING (first use of a bucket, possibly in an FC thread):
    warm the graph buckets before interleaving;
  * the FC background thread consumes the default generator concurrently with the driver: during a live FC thread
    the RNG streams of the FC session and of the session being stepped are NOT isolated (process-global generator;
    a per-thread generator needs an upstream ``generator=`` argument).
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import torch

from mynah_v2.runtime.batch_contract import SessionId


class SessionStateError(RuntimeError):
    pass


class Poisoned:
    """Placeholder left on an owner while no session is swapped in. Any use raises."""
    __slots__ = ("key",)

    def __init__(self, key: str):
        object.__setattr__(self, "key", key)

    def _boom(self, *a, **k):
        raise SessionStateError(f"{object.__getattribute__(self, 'key')} used while no session is swapped in")

    def __getattr__(self, name):
        self._boom()

    def __repr__(self):
        return f"<Poisoned {object.__getattribute__(self, 'key')}>"

    __setattr__ = __bool__ = __len__ = __iter__ = __contains__ = __getitem__ = __setitem__ = __call__ = _boom
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __hash__ = __index__ = __int__ = __float__ = _boom
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _boom


@dataclass(frozen=True)
class FieldSpec:
    owner: str                                   # key in the store's ``owners``
    attr: str
    default: Callable[[SimpleNamespace], Any]    # fresh value for a new session; gets ns(owners, original, fields)
    routed: bool = False

    @property
    def key(self) -> str:
        return f"{self.owner}.{self.attr}"


@dataclass
class SessionRecord:
    sid: SessionId
    fields: dict
    seed: int | None = None
    rng: dict | None = None
    thinker_ref: Any = None                      # e.g. {"backend": "thinker_batch", "row": 3}; never a tensor copy
    epoch: int = 0                               # logical-session generation: unique per open(), never reused


def rng_get(cuda: bool) -> dict:
    st = {"cpu": torch.get_rng_state(), "py": random.getstate()}
    if cuda:
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def rng_set(st: dict) -> None:
    torch.set_rng_state(st["cpu"])
    random.setstate(st["py"])
    if "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"])


def rng_seed(seed: int) -> None:
    torch.manual_seed(seed)                      # CPU + every CUDA device
    random.seed(seed)


class _Routed:
    """Data descriptor: wins over the instance __dict__ and over nn.Module.__getattr__."""

    def __init__(self, store: "SessionStateStore", key: str):
        self.store, self.key = store, key

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.store._current_record().fields[self.key]

    def __set__(self, obj, value):
        self.store._current_record().fields[self.key] = value

    def __delete__(self, obj):
        raise SessionStateError(f"cannot delete routed field {self.key}")


class SessionStateStore:
    """Capture/restore of registered per-session fields (see module doc). Not thread-safe for open/restore/capture/drop:
    those are driver-thread operations; routed reads/writes from FC threads are."""

    def __init__(self, owners: dict[str, object], specs: list[FieldSpec], rng: bool = True,
                 cuda_rng: bool | None = None):
        keys = [s.key for s in specs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate field spec")
        missing = {s.owner for s in specs} - set(owners)
        if missing:
            raise SessionStateError(f"specs name unknown owners {sorted(missing)}")
        self.owners, self.specs = owners, list(specs)
        self.keys = frozenset(keys)
        self.rng = rng
        self.cuda_rng = torch.cuda.is_available() if cuda_rng is None else cuda_rng
        self._records: dict[SessionId, SessionRecord] = {}
        self._dropped: set[SessionId] = set()
        self._epoch = 0
        self._active: SessionId | None = None
        self._tl = threading.local()
        self._installed = False
        self._original: dict[str, Any] = {}      # swap: owner value at install; routed: instance __dict__ value
        self._classes: dict[str, type] = {}
        # hot path of every session switch (several per round per session): resolved once
        self._swap = [(self.owners[s.owner], s.attr, s.key) for s in self.specs if not s.routed]
        self._poisons = {key: Poisoned(key) for _, _, key in self._swap}

    # ---------------------------------------------------------------------------------------------------- install
    def install(self) -> None:
        """Validate every field exists, move routed fields behind descriptors, poison swap fields."""
        if self._installed:
            raise SessionStateError("already installed")
        for s in self.specs:
            o = self.owners[s.owner]
            if not s.routed and not hasattr(o, s.attr):
                raise SessionStateError(f"registered field {s.key} does not exist on its owner")
        for name, o in self.owners.items():
            routed = [s for s in self.specs if s.owner == name and s.routed]
            if not routed:
                continue
            cls = type(o)
            for s in routed:
                if s.attr in o.__dict__:
                    self._original[s.key] = o.__dict__.pop(s.attr)
            ns = {s.attr: _Routed(self, s.key) for s in routed}
            ns["__module__"] = cls.__module__
            ns["_v2_session_routed"] = True
            o.__class__ = type(cls.__name__, (cls,), ns)
            self._classes[name] = cls
        for s in self.specs:
            if not s.routed:
                self._original[s.key] = getattr(self.owners[s.owner], s.attr)
        self._installed = True
        self._poison()

    def uninstall(self) -> None:
        """Teardown: original classes and values back, every record forgotten (no references kept)."""
        if not self._installed:
            return
        for name, cls in self._classes.items():
            self.owners[name].__class__ = cls
        for s in self.specs:
            if s.key in self._original:
                o = self.owners[s.owner]
                if s.routed:
                    o.__dict__[s.attr] = self._original[s.key]
                else:
                    setattr(o, s.attr, self._original[s.key])
        self._classes.clear()
        self._original.clear()
        self._dropped.update(self._records)
        self._records.clear()
        self._active = None
        self._installed = False

    def _poison(self) -> None:
        for o, attr, key in self._swap:
            setattr(o, attr, self._poisons[key])

    # ---------------------------------------------------------------------------------------------------- sessions
    def open(self, sid: SessionId, seed: int | None = None, overrides: dict | None = None,
             thinker_ref: Any = None) -> SessionRecord:
        if not self._installed:
            raise SessionStateError("install() first")
        if sid in self._records:
            raise SessionStateError(f"session {sid!r} already open")
        overrides = dict(overrides or {})
        unknown = set(overrides) - self.keys
        if unknown:
            raise SessionStateError(f"unknown field(s) {sorted(unknown)}")
        fields: dict = {}
        self._epoch += 1
        ns = SimpleNamespace(owners=self.owners, original=self._original, fields=fields, sid=sid, store=self,
                             epoch=self._epoch)
        for s in self.specs:
            fields[s.key] = overrides.pop(s.key) if s.key in overrides else s.default(ns)
        self._dropped.discard(sid)
        rec = self._records[sid] = SessionRecord(sid, fields, seed=seed, thinker_ref=thinker_ref, epoch=self._epoch)
        return rec

    def capture(self, sid: SessionId) -> dict:
        """Swap ``sid`` OUT: its live swap-field objects go back into its record, owners get poisoned."""
        if self._active != sid:
            raise SessionStateError(f"capture({sid!r}) but the swapped-in session is {self._active!r}")
        rec = self._records[sid]
        fields = rec.fields
        for o, attr, key in self._swap:
            v = getattr(o, attr)
            if type(v) is Poisoned:
                raise SessionStateError(f"{key} was poisoned while {sid!r} was active")
            fields[key] = v
        if self.rng:
            rec.rng = rng_get(self.cuda_rng)
        self._poison()
        self._active = None
        return {"sid": sid, "fields": dict(rec.fields), "rng": rec.rng, "thinker_ref": rec.thinker_ref}

    def restore(self, sid: SessionId) -> None:
        """Swap ``sid`` IN. Raises for a dropped/unknown session or when another session is still swapped in."""
        if sid in self._dropped:
            raise SessionStateError(f"session {sid!r} was dropped")
        rec = self._records.get(sid)
        if rec is None:
            raise SessionStateError(f"unknown session {sid!r}")
        if self._active == sid:
            return
        if self._active is not None:
            raise SessionStateError(f"restore({sid!r}) while {self._active!r} is swapped in: capture it first")
        fields = rec.fields
        if fields.keys() != self.keys:
            raise SessionStateError(f"record of {sid!r} has fields {sorted(set(fields) ^ self.keys)} "
                                    "that do not match the registry")
        for o, attr, key in self._swap:
            setattr(o, attr, fields[key])
        if self.rng:
            if rec.rng is not None:
                rng_set(rec.rng)
            elif rec.seed is not None:
                rng_seed(rec.seed)
        self._active = sid

    def drop(self, sid: SessionId) -> None:
        if sid not in self._records:
            raise SessionStateError(f"unknown session {sid!r}")
        if self._active == sid:
            self._poison()
            self._active = None
        del self._records[sid]
        self._dropped.add(sid)

    # ---------------------------------------------------------------------------------------------------- access
    @property
    def active(self) -> SessionId | None:
        return self._active

    def sids(self) -> list[SessionId]:
        return list(self._records)

    def record(self, sid: SessionId) -> SessionRecord:
        if sid not in self._records:
            raise SessionStateError(f"unknown session {sid!r}")
        return self._records[sid]

    def get(self, sid: SessionId, key: str) -> Any:
        """Current value of one field of ``sid`` (live owner value when it is the swapped-in session)."""
        rec = self.record(sid)
        if key not in self.keys:
            raise SessionStateError(f"unknown field {key!r}")
        spec = next(s for s in self.specs if s.key == key)
        if self._active == sid and not spec.routed:
            return getattr(self.owners[spec.owner], spec.attr)
        return rec.fields[key]

    def set_thinker_ref(self, sid: SessionId, ref: Any) -> None:
        self.record(sid).thinker_ref = ref

    def bind_thread(self, sid: SessionId) -> None:
        """Routed fields read/written by THIS thread resolve to ``sid`` from now on (FC background threads)."""
        self.record(sid)
        self._tl.sid = sid

    def unbind_thread(self) -> None:
        self._tl.sid = None

    def _current_record(self) -> SessionRecord:
        sid = getattr(self._tl, "sid", None) or self._active
        if sid is None:
            raise SessionStateError("routed field used with no current session (none swapped in or bound)")
        rec = self._records.get(sid)
        if rec is None:
            raise SessionStateError(f"routed field used by dropped/unknown session {sid!r}")
        return rec


# ======================================================================================================== VoiceChat
def _const(v):
    return lambda ns: v


def _new_list(ns):
    return []


def _new_context_manager(ns):
    p, orig = ns.owners["pipeline"], ns.original["pipeline.context_manager"]
    return type(orig)(s2s_model=p.s2s_model, num_slots=p.batch_size, max_len=p.max_len)


def _new_bufferer(ns):
    p, orig = ns.owners["pipeline"], ns.original["pipeline.bufferer"]
    return type(orig)(sample_rate=p.input_sample_rate, buffer_size_in_secs=p.buffer_size_in_secs)


def _new_registry(ns):
    # base registry = builtins + constructor tools (SessionToolBinding.base); session tools are bound per session
    b = ns.original.get("engine.session_tools")
    return dict(b.base) if b is not None else dict(ns.original["pipeline.tool_registry"])


class StaleSessionError(RuntimeError):
    """A callback (tool execution) of a logical session that is closed or whose sid now names a new session."""


def _new_executor(ns):
    """Per-session tool executor. Upstream runs the tool in a fresh ThreadPoolExecutor worker (pipeline :1726-1727,
    when FC reminders are on): that thread is bound to no session, so routed fields (tool_registry, session_tools,
    tool handlers) would resolve to whichever session the driver has swapped in → session A's call executed with
    B's tools (round 11/13 PG-p1: A's get_order_status logged in B). The pipeline snapshots ``self._execute_tool_call``
    when it spawns the FC thread, in the session's own thread → it gets THIS closure, which binds the session's sid in
    whatever thread finally runs the tool and restores the previous binding afterwards."""
    orig, sid, store, epoch = ns.original["pipeline._execute_tool_call"], ns.sid, ns.store, ns.epoch

    def execute(call_text):
        rec = store._records.get(sid)
        if rec is None or rec.epoch != epoch:     # late callback of a closed session (sid closed, or reopened /
            raise StaleSessionError(               # reused for a new logical session): never act on the new one
                f"tool call of closed session {sid!r} (epoch {epoch}) refused"
                + ("" if rec is None else f"; {sid!r} is now epoch {rec.epoch}"))
        prev = getattr(store._tl, "sid", None)
        store.bind_thread(sid)
        try:
            return orig(call_text)
        finally:
            store._tl.sid = prev
    execute.__v2_session__ = (sid, epoch)
    return execute


def _new_tool_binding(ns):
    from mynah_v2.model.session_tools import SessionToolBinding
    return SessionToolBinding(ns.fields["pipeline.tool_registry"])


def reference_fields() -> list[FieldSpec]:
    """Every per-session field of ReferenceEngine + NVIDIA wrapper/pipeline (nemotron-labs-voicechat @ 097dfe9e).
    W = nemotron_voicechat_inference_wrapper.py, P = streaming_s2s_pipeline.py, I = s2s_pipeline_interface.py."""
    F = FieldSpec
    return [
        # --- wrapper turn-taking / TTS gating: mutated by the driver AND by the FC thread → routed
        F("wrapper", "_post_tc_bos_exempt", _const(False), routed=True),     # W:248, 2483, 3090-3091, 3265
        F("wrapper", "_redirect_tokens_queue", _new_list, routed=True),      # W:244, 3036-3037, 3208, 3256, 3264
        F("wrapper", "_agent_eos_just_fired", _const(False), routed=True),   # W:3043, 3168, 3206, 3242, 3257, 3263
        F("wrapper", "_agent_idle", _const(True), routed=True),              # W:554, 2093 (fallback; per-stream wins)
        F("wrapper", "_tts_in_turn_content", _const(0), routed=True),        # W:2592-2634 (getattr default 0)
        F("wrapper", "_tts_in_turn_pads", _const(0), routed=True),           # W:2592-2634
        F("wrapper", "_last_user_audio_emb", _const(None), routed=True),     # W:888, 2375; read by FC W:1266
        # --- pipeline: per-session objects. _state_pool is read by the FC thread (get_state_fn = bound method,
        #     P:1609) → routed; context manager / bufferer / prompt flag only by the driver → swapped.
        F("pipeline", "_state_pool", lambda ns: {}, routed=True),            # I:32, 53-64 (open_session clears ALL)
        F("pipeline", "context_manager", _new_context_manager),              # P:154, 888, 2221 (replaced per stream)
        F("pipeline", "bufferer", _new_bufferer),                            # P:140, 2331, 2749 (reset() = all ids)
        F("pipeline", "_stream_has_prompt", _const(False)),                  # P:895, 2261, 2302
        F("pipeline", "tool_registry", _new_registry, routed=True),          # P:480-488, 767 (executor in FC thread)
        F("pipeline", "_execute_tool_call", _new_executor, routed=True),     # P:1603 snapshot, 1726 pool thread, 1781
        # --- engine (ours, reference.py): cursors read by step() on the driver → swapped; tool policy read by the
        #     executor wrapper, which runs in the FC thread for two-phase calls → routed
        F("engine", "stream_id", _const(0)),                                 # overridden with a unique id per session
        F("engine", "_started", _const(False)),
        F("engine", "_opts", _const(None)),
        F("engine", "_pos", lambda ns: {"text": 0, "asr": 0, "func": 0}),
        F("engine", "_last_frame_idx", _const(0)),
        F("engine", "_rec_cursor", _const(0)),
        F("engine", "fc_records", _new_list),                                # appended by the routed FC recorder
        F("engine", "session_tools", _new_tool_binding, routed=True),
        F("engine", "unknown_tool_handler", _const(None), routed=True),
        F("engine", "multi_call_handler", _const(None), routed=True),
        F("engine", "unknown_tool_events", _new_list, routed=True),
        F("engine", "multi_call_events", _new_list, routed=True),
    ]


class MultiSessionEngine:
    """Opt-in: N sessions on one ReferenceEngine, stepped in any interleaved order (one step at a time).
    No scheduler, no batching: each ``step(sid, ...)`` swaps ``sid`` in (if needed) and runs the unchanged
    ``ReferenceEngine.step``. ``close_all()`` puts the engine back in single-session shape."""

    def __init__(self, engine, rng: bool = True, cuda_rng: bool | None = None):
        if getattr(engine.session_tools, "active", False):
            raise SessionStateError("engine has a bound single session: close it first")
        self.engine = engine
        self._driver = threading.current_thread()
        self._stream_of: dict[SessionId, int] = {}
        self._sid_of_stream: dict[int, SessionId] = {}
        self._next_stream = 1                    # 0 stays the single-session id; ids are never reused
        if not hasattr(engine, "_opts"):        # set by ReferenceEngine.open only; must exist to be registered
            engine._opts = None
        self.store = SessionStateStore({"engine": engine, "wrapper": engine.wrapper, "pipeline": engine.pipeline},
                                       reference_fields(), rng=rng, cuda_rng=cuda_rng)
        self.store.install()
        self._fc_prev = None
        self._llm_prev = None
        try:
            self._install_fc_router()
            self._install_thinker_lock()
        except BaseException:
            self.close_all()
            raise

    # ------------------------------------------------------------------------------------------------ hooks
    def _install_fc_router(self) -> None:
        """Replace the engine's FC recorder: route the call to the session that owns ``stream_id`` (bind the FC
        thread to it for the rest of its life, so the tool executor and get_state_fn resolve to it too) and append
        the FCRecord to THAT session's list."""
        from mynah_v2.model.reference import fc_record
        w, eng, store = self.engine.wrapper, self.engine, self.store
        cur = w._run_fc_async_steps
        inner = getattr(cur, "__wrapped_v2__", None)
        if inner is None:
            raise SessionStateError("wrapper._run_fc_async_steps is not the ReferenceEngine recorder")

        def routed(*args, **kw):
            stream = kw.get("stream_id")
            sid = self._sid_of_stream.get(stream)
            if sid is None:
                raise SessionStateError(f"FC call for unknown stream {stream!r}")
            if threading.current_thread() is not self._driver:
                store.bind_thread(sid)
            elif store.active != sid:
                raise SessionStateError(f"driver FC call for {sid!r} while {store.active!r} is swapped in")
            t0 = time.monotonic_ns()
            out = inner(*args, **kw)
            rec = fc_record(args, kw, out, t0)
            if rec is not None:
                with eng._fc_lock:
                    store.get(sid, "engine.fc_records").append(rec)
            return out

        routed.__wrapped_v2__ = inner
        self._fc_prev = cur
        w._run_fc_async_steps = routed

    def _install_thinker_lock(self) -> None:
        """llm_cache publishes the live cache/position on the backbone object (``_v2_active_cache``/``_v2_host_pos``)
        for the duration of a forward: two threads (A's FC thread, B's driver step) in the Thinker at once would
        read each other's cache. Serialize whole backbone forwards (outermost layer, removed by close_all)."""
        mod = getattr(getattr(getattr(self.engine.wrapper, "model", None), "stt_model", None), "llm", None)
        if mod is None or not hasattr(mod, "forward"):
            return
        had = "forward" in mod.__dict__
        orig = mod.forward
        lock = threading.RLock()

        def forward(*a, **kw):
            with lock:
                return orig(*a, **kw)
        forward.__v2_session_lock__ = orig
        self._llm_prev = (mod, had, orig, forward)
        mod.forward = forward

    # ------------------------------------------------------------------------------------------------ sessions
    def _check_eartts_slots(self) -> None:
        fsg = getattr(self.engine, "_v2_full_graph", None)
        # ONE slot per session: the FC thread uses the session's own EarTTS KV by reference (pipeline :1556-1561),
        # measured on the Ada (FCX single session with FC: 1 slot; 4 sessions with overlapping FC: 4 slots). A missing
        # slot would make steps fall back to eager (counted in fsg.stats["eager_reasons"]["no_free_slot"]).
        if fsg is not None and len(self.store.sids()) + 1 > fsg.n_slots:
            raise SessionStateError(f"eartts_graph_full has {fsg.n_slots} slots; {len(self.store.sids()) + 1} "
                                    "sessions need 1 each, else steps fall back to eager (parity break)")

    def _switch(self, sid: SessionId) -> None:
        a = self.store.active
        if a == sid:
            return
        if sid not in self.store.sids():
            self.store.restore(sid)              # raises (dropped / unknown) before anything is swapped out
        if a is not None:
            self.store.capture(a)
        self.store.restore(sid)

    def open(self, sid: SessionId, system_prompt: str | None, tools: dict | None = None, seed: int | None = None,
             fc_wait_s: float = 10.0, guard: bool | dict = True) -> int:
        self._check_eartts_slots()
        if sid in self.store.sids():
            raise SessionStateError(f"session {sid!r} already open")
        stream = self._next_stream
        self._next_stream += 1
        self.store.open(sid, seed=seed, overrides={"engine.stream_id": stream})
        self._stream_of[sid], self._sid_of_stream[stream] = stream, sid
        if self.store.active is not None:
            self.store.capture(self.store.active)
        self.store.restore(sid)
        if tools:
            self.engine.bind_session_tools(tools)
            if guard:                                    # default ON: every session's tools behind its own guard
                from mynah_v2.agent.call_guard import CallGuard
                self.engine.set_session_guard(CallGuard(session_id=str(sid), **(guard if isinstance(guard, dict) else {})))
        self.engine.open(system_prompt, fc_wait_s=fc_wait_s)
        return stream

    def step(self, sid: SessionId, pcm16k: torch.Tensor, is_last: bool = False):
        self._switch(sid)
        return self.engine.step(pcm16k, is_last=is_last)

    def close(self, sid: SessionId) -> None:
        self._switch(sid)
        try:
            self.engine.close()
        finally:
            self.store.drop(sid)
            # keep _sid_of_stream: a still-running FC thread of this stream must hit the dropped record and raise,
            # never resolve to another session (stream ids are not reused)
            self._stream_of.pop(sid, None)

    def close_all(self) -> None:
        try:
            for sid in list(self.store.sids()):
                self.close(sid)
        finally:
            if self._fc_prev is not None:
                self.engine.wrapper._run_fc_async_steps = self._fc_prev
                self._fc_prev = None
            if self._llm_prev is not None:
                mod, had, orig, fwd = self._llm_prev
                if mod.__dict__.get("forward") is fwd:
                    if had:
                        mod.forward = orig
                    else:
                        del mod.forward
                self._llm_prev = None
            self.store.uninstall()
