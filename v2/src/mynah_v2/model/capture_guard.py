"""EXPERIMENTAL (opt-in): process-wide policy for LAZY CUDA-graph captures under multi-session concurrency.

Why (Ada round 19, MS2 B=8): a capture is only safe if NO other thread issues certain CUDA calls while it is open.
``capture_error_mode="thread_local"`` only relaxes the checks for the CAPTURING thread's own unsafe calls; it does not
protect against (torch 2.10, VERIFIED by v2/scripts/capture_race_smoke.py on GB10):
  * another thread's device-wide sync (``torch.cuda.synchronize()`` — perception_cache.py:542/557, the wrapper's timers,
    and ``torch.cuda.graph.__enter__`` itself, i.e. a SECOND concurrent capture): it raises "operation not permitted
    when stream is capturing" in that thread and invalidates the open capture;
  * the default CUDA generator is process-wide: between capture_begin and capture_end it is in "capturing" state, so
    any RNG op / RNG graph replay in ANOTHER thread raises "Offset increment outside graph capture encountered
    unexpectedly";
  * a capture that fails leaves that generator in "capturing" state for good (capture_end raises before its
    epilogue) → every later RNG op or RNG-graph replay in every session fails ("poisoned process");
  * ``torch.cuda.graph`` shares one class-level capture stream between threads.

Policy (``GUARD.mode``):
  "lazy"     — default: capture when a key is first seen (legacy) — but only while no multi-session rendezvous is
               registered; with one (Ada round 33) lazy behaves like "barrier". A failed capture is recovered in
               every mode (stream restored, generator reset, key eager) instead of poisoning the process.
  "barrier"  — a capture may start only when the calling thread is the ONLY GPU-issuing thread: every registered
               provider (RendezvousEngine: no other worker running, coordinator idle, no FC phase, no queued FC
               request) and the helper-thread scan (upstream FC / live-perception threads alive) must say "quiet".
               Otherwise the step runs EAGER on the same static buffers (same numerics as the graph's warm-up, slower)
               and the capture is retried the next time the key comes up. Captures are serialized by ``GUARD.lock``.
               A failed capture is recovered (generator reset by a complete dummy capture, key disabled → eager)
               instead of poisoning the process.
Pre-capture (separate opt-in, enumerable keys only, done at a quiet point before sessions go concurrent):
``BatchedThinker(precapture=True)`` and ``FullStepGraph.precapture(n_slots)``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import threading
import time
import weakref
from typing import Callable, Optional

# Thread targets of the pinned upstream (NeMo @097dfe9) that issue GPU work concurrently with the session thread:
#   streaming_s2s_pipeline.py:1842 FC background thread (_fc_async_thread_fn: Thinker via the rendezvous, EarTTS steps,
#   reminders, codec decode); wrapper :1330 / :1347 perception threads started inside _run_fc_async_steps.
# Python >= 3.10 names an unnamed thread "Thread-N (<target>)".
GPU_HELPER_TARGETS = ("_fc_async_thread_fn", "_perception_live_worker", "_perception_background_worker")

MODES = ("lazy", "barrier")


def helper_threads(exclude: Optional[threading.Thread] = None) -> list[str]:
    return [t.name for t in threading.enumerate()
            if t is not exclude and t.is_alive() and any(p in t.name for p in GPU_HELPER_TARGETS)]


class CaptureGuard:
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or os.environ.get("V2_CAPTURE_POLICY", "lazy")
        if self.mode not in MODES:
            raise ValueError(f"V2_CAPTURE_POLICY={self.mode!r} (expected one of {MODES})")
        self.lock = threading.RLock()                    # serializes captures (shared capture stream, generator)
        self._providers: list[Callable[[threading.Thread], Optional[str]]] = []
        self.disabled: set = set()                       # (site, key) whose capture failed: eager for good
        self.stats = {"allowed": 0, "deferred": {}, "deferred_by_site": {}, "failed": 0, "recovered": 0,
                      "recover_errors": [], "disabled_keys": [], "precaptured": {}, "precapture_ms": {},
                      "precapture_skipped": {}}

    # ------------------------------------------------------------------------------------------------ configuration
    def configure(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"capture policy {mode!r} (expected one of {MODES})")
        self.mode = mode

    @property
    def barrier(self) -> bool:
        return self.mode == "barrier"

    def add_provider(self, fn: Callable[[threading.Thread], Optional[str]]) -> None:
        """Bound methods are held weakly (an engine that is dropped without close_all stops voting)."""
        self._providers.append(weakref.WeakMethod(fn) if hasattr(fn, "__self__") else (lambda f=fn: f))

    def remove_provider(self, fn) -> None:
        self._providers = [p for p in self._providers if p() is not None and p() != fn]

    # ------------------------------------------------------------------------------------------------ barrier
    def quiet_reason(self, me: Optional[threading.Thread] = None) -> Optional[str]:
        """None when ``me`` (default: the calling thread) is the only thread that may issue GPU work, else why not."""
        me = me or threading.current_thread()
        for ref in list(self._providers):
            p = ref()
            if p is None:
                continue
            r = p(me)
            if r:
                return r
        h = helper_threads(exclude=me)
        if h:
            return f"helper_thread:{h[0]}"
        return None

    def may_capture(self, site: str, key) -> bool:
        """False for a key whose capture failed (any mode). Lazy mode: otherwise True (legacy). Barrier mode: True
        only if all is quiet."""
        if (site, key) in self.disabled:
            self._defer(site, "disabled_after_failure")
            return False
        if not self.barrier and not any(r() is not None for r in self._providers):
            return True                                  # lazy, single-session (no rendezvous registered): legacy
        # barrier mode, or lazy mode while a multi-session rendezvous is registered (Ada round 33: a lazy B=1 EarTTS
        # capture from an FC thread broke other sessions' ops): capture only when quiet, else eager + retry
        r = self.quiet_reason()
        if r is not None:
            self._defer(site, r.split(":")[0])
            return False
        self.stats["allowed"] += 1
        return True

    def _defer(self, site: str, why: str) -> None:
        d = self.stats["deferred"]
        d[why] = d.get(why, 0) + 1
        s = self.stats["deferred_by_site"]
        s[site] = s.get(site, 0) + 1

    @contextlib.contextmanager
    def capturing(self):
        """Held around every capture: captures are serialized process-wide (torch.cuda.graph shares one capture
        stream between threads; a second __enter__ syncs the device and breaks the open capture)."""
        with self.lock:
            yield

    # ------------------------------------------------------------------------------------------------ failure
    def on_failure(self, site: str, key, exc: BaseException, stream=None) -> bool:
        """A capture raised (any mode, since Ada round 31: a lazy-mode failure used to poison every session). Restore
        the thread's stream (a failing capture_begin / capture_end leaves torch.cuda.graph's capture stream current),
        reset the generator, disable the key, return True: the caller returns its warm-up result and runs that key
        eagerly from now on. False only if the generator reset itself failed (caller raises: fail fast)."""
        if stream is not None:
            try:
                torch_ = __import__("torch")
                torch_.cuda.set_stream(stream)
            except Exception:  # noqa: BLE001
                pass
        self.stats["failed"] += 1
        self.disabled.add((site, key))
        self.stats["disabled_keys"].append(f"{site} {key!s:.120} {type(exc).__name__}: {str(exc)[:160]}")
        try:
            reset_generator_capture_state()
            self.stats["recovered"] += 1
        except Exception as e:  # noqa: BLE001 - reported; the caller then raises the original error
            self.stats["recover_errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
            return False
        return True

    def note_precapture(self, site: str, n: int, ms: float, skipped: Optional[str] = None,
                        stopped: Optional[str] = None) -> None:
        if skipped:
            self.stats["precapture_skipped"][site] = skipped
            return
        if stopped:
            self.stats.setdefault("precapture_stopped", {})[site] = stopped
        self.stats["precaptured"][site] = self.stats["precaptured"].get(site, 0) + n
        self.stats["precapture_ms"][site] = round(self.stats["precapture_ms"].get(site, 0.0) + ms, 1)

    def report(self) -> dict:
        return {"mode": self.mode} | {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v))
                                      for k, v in self.stats.items()}

    def reset_stats(self) -> None:
        providers, mode = self._providers, self.mode
        self.__init__(mode)
        self._providers = providers


def reset_generator_capture_state(device=None, attempts: int = 3) -> None:
    """Clear the default CUDA generator's "capturing" flag left by a failed capture: a COMPLETE dummy capture runs the
    generator's capture prologue + epilogue (same generator-state object as every existing graph). Verified by an RNG
    probe (consumes one philox increment). Raises if the probe still fails."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device() if device is None else device)
    last = None
    for _ in range(attempts):
        try:
            x = torch.zeros(1, device=dev)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                g.capture_begin(capture_error_mode="thread_local")
                try:
                    x.add_(1)
                finally:
                    g.capture_end()
            torch.cuda.current_stream(dev).wait_stream(s)
            del g
            torch.empty(1, device=dev).uniform_()          # probe: raises while the generator is still "capturing"
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.05)
    raise RuntimeError(f"CUDA generator still in capture state after {attempts} resets: {last}") from last


def buckets(lmax: int, lo: int = 64) -> list[int]:
    """Every value ``bucket_for(n, lmax, lo)`` can take for 1 <= n <= lmax (the graph key space per (rows, slot))."""
    out, b = [], lo
    while b < lmax:
        out.append(b)
        b *= 2
    out.append(lmax)
    return sorted(set(out))


def precapture_buckets(lmax: int, start_pos: Optional[int] = None, min_bucket: int = 0,
                       max_bucket: Optional[int] = None) -> list[int]:
    """Pre-capture order: the bucket of ``start_pos`` (first session's position) and the ones above it ascending (the
    buckets a running session reaches next), then the lower ones (shorter prompts) descending; restricted to
    [min_bucket, max_bucket]. A budget stop therefore keeps the most useful keys; the rest stay lazy (barrier-guarded)."""
    from mynah_v2.model.eartts_graph import bucket_for
    bl = [b for b in buckets(lmax) if b >= min_bucket and (max_bucket is None or b <= max_bucket)]
    if start_pos is None:
        return bl
    s = bucket_for(start_pos + 1, lmax)
    return [b for b in bl if b >= s] + sorted((b for b in bl if b < s), reverse=True)


@dataclasses.dataclass
class PrecaptureBudget:
    """Process-wide pre-capture limits (rendezvous_pair: --precapture-*). Pre-capture stops (fail soft: remaining keys
    lazy + barrier-guarded) when the device would keep less than ``min_free_gib`` free or a capture/warm-up OOMs."""
    min_bucket: int = 0
    max_bucket: Optional[int] = None
    min_free_gib: float = 6.0


def free_gib() -> float:
    """Device free memory + this process's reserved-but-unallocated cache (what a new allocation could still get)."""
    import torch
    free, _ = torch.cuda.mem_get_info()
    return (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 2**30


def soft_cleanup() -> Optional[str]:
    """After a pre-capture stop: release the allocator cache and, if an OOM hit INSIDE a capture (generator left
    "capturing": the RNG probe fails), reset the generator. Returns an error string if that reset failed."""
    import torch
    torch.cuda.empty_cache()
    try:
        torch.empty(1, device="cuda").uniform_()
        return None
    except Exception:  # noqa: BLE001 - poisoned: reset
        try:
            reset_generator_capture_state()
            return None
        except Exception as e:  # noqa: BLE001
            GUARD.stats["recover_errors"].append(f"soft_cleanup: {type(e).__name__}: {str(e)[:200]}")
            return str(e)


def is_oom(e: BaseException) -> bool:
    while e is not None:
        if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
            return True
        e = e.__cause__
    return False


GUARD = CaptureGuard()
BUDGET = PrecaptureBudget()
