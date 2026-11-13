"""Shared contract of the experimental true-batching path (OWNED BY THE INTEGRATOR — other modules import, never edit).

Modules and responsibilities:
  model/thinker_batch.py      implements ThinkerBatchBackend: owns ALL Thinker recurrent state of attached sessions
                              (per-row K/V + mask + position, Mamba conv/SSM rows, CUDA graphs per (n_rows, bucket)).
  model/session_state.py      implements SessionStateStore: swaps NON-Thinker per-session state (wrapper/engine fields,
                              pipeline context slot, EarTTS/codec/perception caches, FC/tool state, RNG) in/out.
  runtime/batch_scheduler.py  decides WHEN and WITH WHOM a session's frame runs (ready frames, 160 ms deadlines, tool
                              pending, grouping by phase/bucket); never touches tensors except passing frames through.
  scripts/two_session_bench.py  drives sessions through the scheduler (pair) or alone (solo) and checks isolation.

Invariants (each must have a test in the owning module):
  I1  A slot's state changes ONLY when its sid is in the `frames` of a `step()` call (inactive/free rows untouched).
  I2  `attach(sid, cache)` then `detach(sid)` returns a B=1 cache equal to the input (exact round trip); a detached cache
      continued by the upstream eager path matches an uninterrupted solo run.
  I3  Slot reuse leaves no trace: after detach(A) and attach(C) on the same row, C's outputs equal C's solo run; the
      sequence {A+B} → {B+C} → {A+C} (re-attach) matches the solo runs of A, B, C.
  I4  CUDA graphs are keyed by (n_rows, bucket) and read/write ONLY backend-owned static buffers; replay after slot reuse,
      teardown (`close`) or a bucket change never reads stale per-session buffers.
  I5  Positions and masks are per row; rows of different lengths share one bucket = bucket_for(max active pos + 1).
  I6  The backend never decides deadlines/grouping; the scheduler never reads K/V, Mamba state or CUDA buffers — it only
      sees `ThinkerBatchBackend.positions()`, `phase` and host bookkeeping.
  I7  Phases other than DECODE (PREFILL multi-token, FC_FORCED injection, TOOL_PENDING waits) run B=1 through
      `step_one` (or are not stepped) until a batched variant is proven equal; they never block other sessions.
  I8  Per-session RNG: EarTTS/other sampling uses the session's own generator/state (swapped by SessionStateStore);
      batch composition must not change any session's audio when determinism is expected.
  I9  Memory is accounted, not assumed: the backend reports bytes per slot and graph-pool bytes; the harness reports
      weights / caches / graph pools / allocator reserve / load peak separately.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

SessionId = str


class Phase(enum.Enum):
    DECODE = "decode"              # one frame, batchable
    PREFILL = "prefill"            # multi-token prompt/context: B=1 only
    FC_FORCED = "fc_forced"        # forced tokens / tool-response injection: B=1 only until proven
    TOOL_PENDING = "tool_pending"  # waiting for a tool result: frames may be buffered, never block others
    CLOSED = "closed"


@dataclass(frozen=True)
class ThinkerOut:
    """Per-session Thinker outputs of one frame (each tensor [1, 1, V] or None when the head is disabled)."""
    text_logits: Any
    function_logits: Any = None
    asr_logits: Any = None


@dataclass
class SlotInfo:
    sid: SessionId
    row: int
    pos: int                       # host position = number of frames in the session's Thinker context
    active: bool = True


@dataclass
class BackendMemory:
    bytes_per_slot: int            # K/V rows + Mamba conv/SSM rows for one session at lmax
    graph_pool_bytes: int = 0
    static_buffers_bytes: int = 0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class ThinkerBatchBackend(Protocol):
    max_rows: int

    def attach(self, sid: SessionId, cache: Any) -> int: ...
    """Take ownership of a session's B=1 hybrid cache (after its prefill); returns the row. Raises when full."""

    def detach(self, sid: SessionId) -> Any: ...
    """Release the row; returns a B=1 cache the upstream eager path can continue (I2)."""

    def positions(self) -> Mapping[SessionId, int]: ...

    def step(self, frames: Mapping[SessionId, Any]) -> dict[SessionId, ThinkerOut]: ...
    """ONE forward with a real batch dimension over exactly these attached sessions (frames: [1, 1, H] each)."""

    def step_one(self, sid: SessionId, frame: Any) -> ThinkerOut: ...
    """B=1 forward for one attached session (same state, production-equivalent numerics)."""

    def memory(self) -> BackendMemory: ...

    def close(self) -> None: ...


@runtime_checkable
class SessionStateStore(Protocol):
    def capture(self, sid: SessionId) -> Any: ...
    def restore(self, sid: SessionId) -> None: ...      # swap the session's non-Thinker state in
    def drop(self, sid: SessionId) -> None: ...         # teardown; later restore(sid) must raise


@dataclass
class FrameRequest:
    """What the scheduler hands to the engine glue for one session frame."""
    sid: SessionId
    chunk_idx: int
    frame_idx: int                 # 0 or 1 within the 160 ms chunk
    phase: Phase
    arrival_s: float               # chunk arrival time (deadline = arrival_s + 0.160)
    payload: Optional[Any] = None  # embeds / audio as produced by the per-session perception step
