"""Full-step EarTTS CUDA graph (mynah_v2.model.eartts_graph) on a tiny random Gemma3 backbone: graph replays vs the
eager DynamicCache path over many steps (bucket changes, prefill conversion, RNG inside the step, Lmax fallback)."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


class FakeRVQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Gemma3TextConfig, Gemma3TextModel
        cfg = Gemma3TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                               num_attention_heads=4, num_key_value_heads=4, head_dim=16, sliding_window=7500,
                               max_position_embeddings=8192, attn_implementation="sdpa")
        self.backbone = Gemma3TextModel(cfg)
        self.emb = torch.nn.Embedding(64, 64)
        self.head = torch.nn.Linear(64, 8)


class FakeDuplex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tts_model = FakeRVQ()

    def infer_codes_one_step(self, current_subword_id, prev_subword_id, current_subword_mask, prev_audio_tokens,
                             past_key_values, guidance_enabled=True, generation_config=None, ignore_eos_flag_stop=True):
        m = self.tts_model
        x = (m.emb(current_subword_id) + m.emb(prev_subword_id) + prev_audio_tokens.float().mean(-1, keepdim=True))
        x = torch.cat([x, x * 0.5], 0)                                    # CFG batch 2, like EarTTS
        out = m.backbone(inputs_embeds=x, past_key_values=past_key_values, use_cache=True)
        h = out.last_hidden_state[:1]
        codes = m.head(h) + 0.01 * torch.randn_like(m.head(h))             # RNG inside the step
        return codes, out.past_key_values


def prefill(d, n=5):
    x = torch.randn(2, n, 64, device="cuda", generator=torch.Generator("cuda").manual_seed(7))
    return d.tts_model.backbone(inputs_embeds=x, use_cache=True).past_key_values


def drive(d, step, steps, seed=3):
    torch.manual_seed(seed)
    pkv = prefill(d)
    ids = torch.tensor([[5]], device="cuda")
    prev = torch.tensor([[1]], device="cuda")
    audio = torch.zeros(1, 1, 4, device="cuda")
    outs = []
    for i in range(steps):
        codes, pkv = step(ids, prev, torch.ones_like(ids), audio, pkv)
        outs.append(codes.clone())
        prev, ids = ids, (ids * 7 + i) % 64
        audio = codes[..., :4].detach()
    return torch.cat(outs), pkv


@torch.inference_mode()
def test_graph_matches_eager_across_buckets_and_rng():
    from mynah_v2.model.eartts_graph import FullStepGraph, StaticKV
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref, _ = drive(d, d.infer_codes_one_step, 150)
    fsg = FullStepGraph(d, lmax=512)
    got, pkv = drive(d, fsg.step, 150)
    fsg.remove()
    assert isinstance(pkv, StaticKV) and pkv.t == 5 + 150
    assert fsg.stats["converted"] == 1 and fsg.stats["eager"] == 0
    assert fsg.stats["captures"] == 3                       # buckets 64, 128, 256
    assert fsg.stats["replays"] == 150 - 3
    assert torch.allclose(got, ref, atol=2e-5, rtol=1e-4), float((got - ref).abs().max())


@torch.inference_mode()
def test_lmax_fallback_to_dynamic_continues_correctly():
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref, _ = drive(d, d.infer_codes_one_step, 120)
    fsg = FullStepGraph(d, lmax=100)
    got, pkv = drive(d, fsg.step, 120)
    fsg.remove()
    assert fsg.stats["to_dynamic"] == 1 and fsg.stats["eager_reasons"].get("reached_lmax", 0) >= 1
    assert not hasattr(pkv, "slot")                         # back on the upstream cache
    assert torch.allclose(got, ref, atol=2e-5, rtol=1e-4), float((got - ref).abs().max())


@torch.inference_mode()
def test_slot_released_and_reused_across_sessions():
    import gc
    from mynah_v2.model.eartts_graph import FullStepGraph
    d = FakeDuplex().cuda().eval()
    fsg = FullStepGraph(d, lmax=256, slots=1)
    for _ in range(3):                                      # three sessions, one slot, graphs reused
        _, pkv = drive(d, fsg.step, 70)
        del pkv
        gc.collect()
    fsg.remove()
    assert fsg.stats["converted"] == 3 and fsg.stats["eager_reasons"].get("no_free_slot", 0) == 0
    assert fsg.stats["captures"] == 2                       # buckets 64, 128 captured once, replayed later


@torch.inference_mode()
def test_capture_in_worker_while_main_thread_syncs():
    """G1 crash shape: the FC thread captures a NEW bucket while the main thread keeps issuing synchronizing CUDA ops
    (codec decode + .cpu()). With global capture mode the main thread's op raised "operation not permitted when stream
    is capturing"; thread_local capture must let both proceed and keep parity with eager."""
    import threading
    from mynah_v2.model.eartts_graph import FullStepGraph
    torch.manual_seed(0)
    d = FakeDuplex().cuda().eval()
    ref, _ = drive(d, d.infer_codes_one_step, 150)
    fsg = FullStepGraph(d, lmax=512)
    box, stop, errors = {}, threading.Event(), []

    def worker():
        try:
            with torch.inference_mode():
                box["got"], _ = drive(d, fsg.step, 150)      # crosses buckets 64 -> 128 -> 256: three captures
        except Exception as e:                              # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    th = threading.Thread(target=worker)
    x = torch.randn(256, 256, device="cuda")
    th.start()
    n = 0
    while not stop.is_set():                                # main thread: sync-heavy work during the captures
        try:
            (x @ x).sum().cpu()
            n += 1
        except Exception as e:                              # noqa: BLE001
            errors.append(e)
            break
    th.join()
    fsg.remove()
    assert not errors, errors
    assert n > 0 and fsg.stats["captures"] == 3
    assert torch.allclose(box["got"], ref, atol=2e-5, rtol=1e-4), float((box["got"] - ref).abs().max())
