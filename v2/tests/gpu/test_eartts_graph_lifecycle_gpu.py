"""Lifecycle of the full-step EarTTS CUDA graph (mynah_v2.model.eartts_graph) under the serving patterns that the
corpus run exercises: sessions reusing a slot, graphs captured in one thread and replayed in another (FC thread vs
main thread), first capture of a large bucket while another thread synchronizes, session reset mid-bucket, and the
CUDA RNG stream after many replays. Every session is compared with the eager DynamicCache path (parity), not only
counted. Tiny random Gemma3 backbone (same fakes as test_eartts_graph_gpu.py)."""

import threading

import pytest
import torch

from test_eartts_graph_gpu import FakeDuplex

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def session(d, step, steps, seed, start_id=5, prefill_len=5):
    """One session: prefill (deterministic per seed), then ``steps`` decode steps with RNG inside the step."""
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(2, prefill_len, 64, device="cuda", generator=g)
    pkv = d.tts_model.backbone(inputs_embeds=x, use_cache=True).past_key_values
    torch.manual_seed(seed)
    ids = torch.tensor([[start_id]], device="cuda")
    prev = torch.tensor([[1]], device="cuda")
    audio = torch.zeros(1, 1, 4, device="cuda")
    outs = []
    for i in range(steps):
        codes, pkv = step(ids, prev, torch.ones_like(ids), audio, pkv)
        outs.append(codes.clone())
        prev, ids = ids, (ids * 7 + i) % 64
        audio = codes[..., :4].detach()
    return torch.cat(outs), pkv


def close(a, b):
    return torch.allclose(a, b, atol=2e-5, rtol=1e-4), float((a - b).abs().max())


class SyncNoise:
    """Main-thread CUDA activity with host syncs (codec decode + .cpu() in the real pipeline) while a worker runs."""

    def __init__(self):
        self.stop, self.n, self.errors = threading.Event(), 0, []
        self.x = torch.randn(256, 256, device="cuda")

    def run_until(self, th):
        while th.is_alive():
            try:
                (self.x @ self.x).sum().cpu()
                self.n += 1
            except Exception as e:                       # noqa: BLE001
                self.errors.append(e)
                return


def in_worker(fn):
    box, errs = {}, []

    def w():
        try:
            with torch.inference_mode():
                box["r"] = fn()
        except Exception as e:                           # noqa: BLE001
            errs.append(e)
    th = threading.Thread(target=w)
    noise = SyncNoise()
    th.start()
    noise.run_until(th)
    th.join()
    assert not errs, errs
    assert not noise.errors, noise.errors
    assert noise.n > 0
    return box["r"]


@torch.inference_mode()
def test_sessions_reusing_one_slot_each_match_eager():
    """No residual KV/mask/position from the previous session: 3 sessions with different inputs on ONE slot."""
    import gc
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    specs = [(11, 5, 140), (12, 9, 70), (13, 3, 200)]          # seed, first id, steps (different buckets reached)
    refs = [session(d, d.infer_codes_one_step, n, s, i)[0] for s, i, n in specs]
    fsg = FullStepGraph(d, lmax=512, slots=1)
    for (s, i, n), ref in zip(specs, refs):
        got, pkv = session(d, fsg.step, n, s, i)
        ok, err = close(got, ref)
        assert ok, (s, err)
        del pkv
        gc.collect()
    fsg.remove()
    assert fsg.stats["converted"] == 3 and fsg.stats["eager"] == 0


@torch.inference_mode()
def test_graph_captured_in_worker_replayed_in_main_and_vice_versa():
    """G1 pattern: main thread captures the small buckets, the FC thread captures a NEW bucket while the main thread
    syncs, later the main thread replays that worker-captured graph (shared pool, shared slot)."""
    import gc
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref_a = session(d, d.infer_codes_one_step, 100, 21)[0]       # buckets 64, 128
    ref_b = session(d, d.infer_codes_one_step, 200, 22)[0]       # + 256 (new)
    ref_c = session(d, d.infer_codes_one_step, 200, 23)[0]       # replays 256 in main
    fsg = FullStepGraph(d, lmax=512, slots=1)
    got_a, pkv = session(d, fsg.step, 100, 21)
    del pkv
    gc.collect()
    caps_main = fsg.stats["captures"]
    got_b, pkv = in_worker(lambda: session(d, fsg.step, 200, 22))
    del pkv
    gc.collect()
    caps_worker = fsg.stats["captures"] - caps_main
    got_c, pkv = session(d, fsg.step, 200, 23)
    del pkv
    gc.collect()
    fsg.remove()
    for got, ref in ((got_a, ref_a), (got_b, ref_b), (got_c, ref_c)):
        ok, err = close(got, ref)
        assert ok, err
    assert caps_main == 2 and caps_worker == 1                    # 256 captured in the worker
    assert fsg.stats["captures"] == 3                             # session C only replayed


@torch.inference_mode()
def test_first_capture_of_bucket_1024_in_worker_while_main_syncs():
    """The literal G1 crash: first capture of the 1024 bucket inside the worker thread under concurrent syncs."""
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref = session(d, d.infer_codes_one_step, 560, 31)[0]          # 5 + 560 > 512 → bucket 1024
    fsg = FullStepGraph(d, lmax=2048)
    got, _ = in_worker(lambda: session(d, fsg.step, 560, 31))
    fsg.remove()
    ok, err = close(got, ref)
    assert ok, err
    assert fsg.stats["steps_by_bucket"].get("1024", 0) > 0 and fsg.stats["eager"] == 0


@torch.inference_mode()
def test_session_reset_mid_bucket_then_new_session_matches_eager():
    """A session abandoned mid-bucket (barge-in / close) leaves no state: the next session starts clean."""
    import gc
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref = session(d, d.infer_codes_one_step, 90, 42)[0]
    fsg = FullStepGraph(d, lmax=512, slots=1)
    _, pkv = session(d, fsg.step, 170, 41)                        # stops inside bucket 256
    del pkv
    gc.collect()
    got, pkv = session(d, fsg.step, 90, 42)
    fsg.remove()
    ok, err = close(got, ref)
    assert ok, err


@torch.inference_mode()
def test_rng_stream_after_replays_matches_eager():
    """Graph replays advance the CUDA generator exactly like the eager steps they replace: the next random draw after
    a session is identical (otherwise every later sampled step would silently diverge)."""
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    session(d, d.infer_codes_one_step, 150, 51)
    after_eager = torch.randn(8, device="cuda")
    fsg = FullStepGraph(d, lmax=512)
    session(d, fsg.step, 150, 51)
    after_graph = torch.randn(8, device="cuda")
    fsg.remove()
    assert fsg.stats["replays"] > 100
    assert torch.equal(after_eager, after_graph)
