"""Capture policy (model/capture_guard.py) on CPU fakes: barrier decisions, eager fallback parity, failure recovery
bookkeeping, pre-capture key enumeration, and the RendezvousEngine "only GPU-issuing thread" provider.
The CUDA side (real poisoning + generator reset) is covered by scripts/capture_race_smoke.py on a GB10 machine."""

import importlib
import sys
import threading
from pathlib import Path

import pytest
import torch

from mynah_v2.model import capture_guard as CG
from mynah_v2.model.capture_guard import GUARD, CaptureGuard, buckets
from mynah_v2.model.eartts_graph import bucket_for
from mynah_v2.model.thinker_batch import BatchedThinker

sys.path.insert(0, str(Path(__file__).parent))
TB = importlib.import_module("test_thinker_batch")
model = TB.model                                          # fixture (fake hybrid Thinker)
RZ = importlib.import_module("test_rendezvous")
fake_nemo = RZ.fake_nemo                                  # autouse fixture (nemo stand-ins)


@pytest.fixture(autouse=True)
def lazy_guard():
    saved = list(GUARD._providers)                       # engines left open by other test modules do not vote here
    GUARD._providers = []
    GUARD.configure("lazy")
    GUARD.reset_stats()
    GUARD.disabled.clear()
    yield
    GUARD.configure("lazy")
    GUARD.reset_stats()
    GUARD.disabled.clear()
    GUARD._providers = saved


# ------------------------------------------------------------------------------------------------ key spaces
def test_bucket_key_space_matches_bucket_for():
    for lmax in (32, 64, 100, 3000, 4096, 16384):
        seen = {bucket_for(n, lmax) for n in range(1, lmax + 1)}
        assert sorted(seen) == buckets(lmax), lmax
    assert buckets(4096) == [64, 128, 256, 512, 1024, 2048, 4096]
    # BatchedThinker keys (n_rows, bucket): 4 x 7 = 28 at max_rows 4, 8 x 7 = 56 at max_rows 8 (lmax 4096)
    assert len(buckets(4096)) * 4 == 28 and len(buckets(4096)) * 8 == 56


# ------------------------------------------------------------------------------------------------ guard decisions
def test_lazy_mode_never_defers_but_recovers_a_failed_capture(monkeypatch):
    g = CaptureGuard("lazy")
    assert g.may_capture("x", 1) is True                 # single-session lazy never defers
    assert g.stats["deferred"] == {}
    g.add_provider(busy := (lambda me: "busy"))
    assert g.may_capture("x", 1) is False                # a rendezvous is registered: lazy = barrier
    g.remove_provider(busy)
    monkeypatch.setattr(CG, "reset_generator_capture_state", lambda *a, **k: None)
    assert g.on_failure("x", 1, RuntimeError("boom")) is True    # recovered (Ada round 31: no more poisoning)
    assert g.may_capture("x", 1) is False and g.may_capture("x", 2) is True


def test_barrier_defers_when_any_provider_or_helper_thread_is_busy():
    g = CaptureGuard("barrier")
    state = {"why": "worker_running:b"}
    g.add_provider(lambda me: state["why"])
    assert g.may_capture("eartts", 1) is False
    state["why"] = None
    assert g.may_capture("eartts", 1) is True
    stop = threading.Event()

    def _perception_live_worker():                     # same target name as the upstream FC perception thread
        stop.wait(5)
    t = threading.Thread(target=_perception_live_worker, daemon=True)
    t.start()
    try:
        assert g.quiet_reason().startswith("helper_thread:")
        assert g.may_capture("eartts", 2) is False
    finally:
        stop.set()
        t.join(5)
    assert g.may_capture("eartts", 2) is True
    assert g.stats["deferred"] == {"worker_running": 1, "helper_thread": 1}
    assert g.stats["deferred_by_site"] == {"eartts": 2} and g.stats["allowed"] == 2


def test_failed_capture_is_recovered_and_the_key_runs_eager_for_good(monkeypatch):
    g = CaptureGuard("barrier")
    calls = []
    monkeypatch.setattr(CG, "reset_generator_capture_state", lambda *a, **k: calls.append(1))
    assert g.on_failure("thinker_batch", (2, 128), RuntimeError("operation not permitted when stream is capturing"))
    assert calls == [1] and g.stats["recovered"] == 1 and g.stats["failed"] == 1
    assert g.may_capture("thinker_batch", (2, 128)) is False     # disabled: no second attempt
    assert g.may_capture("thinker_batch", (2, 256)) is True
    assert g.stats["deferred"] == {"disabled_after_failure": 1}

    def broken(*a, **k):
        raise RuntimeError("still capturing")
    monkeypatch.setattr(CG, "reset_generator_capture_state", broken)
    assert g.on_failure("eartts", 7, RuntimeError("x")) is False  # unrecoverable: caller fails fast (raises)
    assert g.stats["recover_errors"]


def test_providers_are_weak_and_removable():
    g = CaptureGuard("barrier")

    class Eng:
        def quiet_for(self, me):
            return "busy"
    e = Eng()
    g.add_provider(e.quiet_for)
    assert g.quiet_reason() == "busy"
    g.remove_provider(e.quiet_for)
    assert g.quiet_reason() is None
    g.add_provider(e.quiet_for)
    del e
    assert g.quiet_reason() is None                      # dropped engine stops voting


# ------------------------------------------------------------------------------------------------ BatchedThinker
def _run_pair(bt, stt, n_steps=70):
    pa, sa = TB.frames(1, n_steps, 5)
    pb, sb = TB.frames(2, n_steps, 40)                  # positions cross 64 → bucket 128
    bt.attach("A", TB.prefilled(stt, pa))
    bt.attach("B", TB.prefilled(stt, pb))
    out = []
    for xa, xb in zip(sa, sb):
        o = bt.step({"A": xa, "B": xb})
        out.append(torch.cat([TB.logits(o["A"]), TB.logits(o["B"])], -1))
    return torch.cat(out)


def test_batched_deferred_capture_runs_eager_with_identical_results(model, monkeypatch):
    eng, stt = model
    ref_bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=False)
    ref = _run_pair(ref_bt, stt)
    ref_bt.close()
    GUARD.configure("barrier")
    busy = {"why": "fc_phase:c"}
    GUARD.add_provider(busy_fn := (lambda me: busy["why"]))  # noqa: F841 - kept alive (strong lambda ref)
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=True)
    monkeypatch.setattr(bt, "_graphable", lambda x: True)

    def no_capture(*a, **k):
        raise AssertionError("captured while not quiet")
    monkeypatch.setattr(bt, "_capture", no_capture)
    got = _run_pair(bt, stt)
    assert torch.equal(got, ref)                        # eager fallback = same static-buffer forward
    assert bt.bstats["capture_deferred"] == 70 and bt.bstats["eager_steps"] == 70
    assert GUARD.stats["deferred"] == {"fc_phase": 70}
    bt.close()
    GUARD.remove_provider(busy_fn)


def test_batched_quiet_capture_is_attempted_with_the_step_key(model, monkeypatch):
    eng, stt = model
    GUARD.configure("barrier")
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=True)
    monkeypatch.setattr(bt, "_graphable", lambda x: True)
    keys = []

    def fake_capture(key, x, cc, hp):                   # no graph stored: every step asks again
        keys.append(key)
        return {k: v for k, v in bt._eager_batched(x, cc, hp).items() if k != "cache"}
    monkeypatch.setattr(bt, "_capture", fake_capture)
    _run_pair(bt, stt, n_steps=30)
    assert {k[:2] for k in keys} == {(2, 64), (2, 128)} and len(keys) == 30
    assert bt.bstats["lazy_captures"] == 30 and GUARD.stats["allowed"] == 30
    bt.close()


def test_precapture_enumerates_every_rows_bucket_key_and_resets_buffers(model, monkeypatch):
    eng, stt = model
    bt = BatchedThinker(eng, max_rows=3, lmax=256, graphs=True, precapture=True)
    keys = []

    def fake_capture(key, x, cc, hp):
        assert x.shape == (key[0], 1, TB.H) and bt.bbucket == key[1] and hp > 0
        out = bt._eager_batched(x, cc, hp)               # writes garbage into the static rows, like a warm-up
        keys.append(key)
        bt.bgraphs[key] = (None, x, {})
        return out
    monkeypatch.setattr(bt, "_capture", fake_capture)
    pa, _ = TB.frames(1, 1, 5)
    c = TB.prefilled(stt, pa)
    bt._frame_sig = (pa.dtype, TB.H)
    bt._alloc(c)                                        # CPU: _alloc does not precapture (cuda only) → call it
    bt._precapture()
    want = {(n, b, pa.dtype, TB.H) for n in (1, 2, 3) for b in buckets(256)}
    assert set(keys) == want and len(keys) == 3 * 3      # buckets(256) = 64, 128, 256
    assert bt.bstats["precaptured"] == 9
    assert torch.count_nonzero(bt.bK) == 0 and torch.count_nonzero(bt.bV) == 0
    assert bool((bt.bmask == torch.finfo(bt.dtype).min).all())
    assert all(torch.count_nonzero(t) == 0 for t in (*bt.bconv, *bt.bssm))
    keys.clear()
    bt._precapture()                                    # idempotent: existing keys are skipped
    assert keys == []
    bt.close()


def test_precapture_bucket_order_and_limits():
    pb = CG.precapture_buckets
    assert pb(4096) == buckets(4096)
    assert pb(4096, start_pos=700) == [1024, 2048, 4096, 512, 256, 128, 64]   # first session's bucket, up, then down
    assert pb(4096, start_pos=700, min_bucket=512, max_bucket=2048) == [1024, 2048, 512]


def test_precapture_oom_fails_soft_keeps_the_hot_keys_and_resets_buffers(model, monkeypatch):
    eng, stt = model
    bt = BatchedThinker(eng, max_rows=3, lmax=256, graphs=True, precapture=True)
    keys = []

    def fake_capture(key, x, cc, hp):
        if len(keys) == 4:
            raise RuntimeError("thinker_batch capture failed") from torch.OutOfMemoryError("CUDA out of memory")
        bt._eager_batched(x, cc, hp)
        keys.append(key)
        bt.bgraphs[key] = (None, x, {})
    monkeypatch.setattr(bt, "_capture", fake_capture)
    pa, _ = TB.frames(1, 1, 70)                          # first session at position 70 → bucket 128 first
    c = TB.prefilled(stt, pa)
    bt._frame_sig = (pa.dtype, TB.H)
    bt._alloc(c)
    bt._precapture(start_pos=int(c._v2_pos))           # must not raise
    assert [k[:2] for k in keys] == [(3, 128), (2, 128), (1, 128), (3, 256)]
    st = bt.bstats["precapture_keys"]
    assert st["captured"] == 4 and st["planned"] == 9 and st["stopped"].startswith("oom at (2, 256)")
    assert torch.count_nonzero(bt.bK) == 0 and all(torch.count_nonzero(t) == 0 for t in (*bt.bconv, *bt.bssm))
    bt.attach("A", c)                                    # the run continues: attach + steps on the reset buffers
    bt.close()


def test_precapture_is_skipped_when_not_quiet(model, monkeypatch):
    eng, stt = model
    GUARD.add_provider(busy := (lambda me: "worker_running:a"))
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=True, precapture=True)
    monkeypatch.setattr(bt, "_capture", lambda *a: (_ for _ in ()).throw(AssertionError("captured")))
    pa, _ = TB.frames(1, 1, 5)
    bt._alloc(TB.prefilled(stt, pa))
    bt._precapture()
    assert GUARD.stats["precapture_skipped"] == {"thinker_batch": "worker_running:a"}
    GUARD.remove_provider(busy)
    bt.close()


# ------------------------------------------------------------------------------------------------ rendezvous provider
def test_rendezvous_provider_quiet_only_for_the_baton_holder_or_idle_coordinator():
    _, m, be, rz = RZ.rz_engine()
    RZ.open_(rz, "A")
    RZ.open_(rz, "B")
    seen = {}
    real_step = be.step

    def step(frames):                                   # coordinator, both workers parked at the rendezvous
        seen.setdefault("serve", GUARD.quiet_reason())
        return real_step(frames)
    be.step = step
    r = rz.step_jobs({"A": RZ.T.pcm("A", 0), "B": RZ.T.pcm("B", 0)})
    assert seen["serve"] is None
    # a worker holding the baton (others idle, coordinator waiting) is quiet; any other thread is not
    res = rz.run_round({"A": lambda: GUARD.quiet_reason()})
    assert res["A"] is None
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", GUARD.quiet_reason()))
    t.start()
    t.join(5)
    assert box["r"].startswith("non_session_thread")
    # an FC phase anywhere (its thread runs free) or another running worker blocks the capture
    rz.thinker_owner["B"] = t
    assert GUARD.quiet_reason() == "fc_phase:B"
    rz.thinker_owner["B"] = None
    rz.workers["B"].state = "running"
    assert GUARD.quiet_reason() == "worker_running:B"
    rz.workers["B"].state = "done"
    assert GUARD.quiet_reason() is None
    assert not isinstance(r["A"], BaseException)
    rz.close_all()
    assert all(p() is None or getattr(p(), "__self__", None) is not rz for p in GUARD._providers)
