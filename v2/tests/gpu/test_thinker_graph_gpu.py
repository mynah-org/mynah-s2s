"""Thinker decode CUDA graph (mynah_v2.model.thinker_graph) on a TINY NemotronH built from the real remote code
(snapshot 6533e8de: same NemotronHBlock with its default-stream switch, same Mamba2 CUDA kernels, same hybrid cache),
with the real V2 patches (llm_cache position injection + attention cache passing, mamba_host_pos). Graph vs eager
parity per step (logits), across bucket transitions, session change and return, capture/replay in a worker thread
(the FC thread) while the main thread synchronizes, and the Lmax fallback. Seconds, no 11B load.

T1b/T2b failure this guards against: NemotronHBlock.forward runs inside `torch.cuda.stream(default_stream)`, which
moved every block off the capturing stream (graph replays only re-ran norm_f + heads; warm-up raced across streams)."""

import glob
import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu


def _module_dir():
    env = os.environ.get("V2_NEMOTRON_H_DIR")
    if env:
        return env
    for pat in (os.environ.get("MYNAH_HOME", "~/mynah-s2s") + "/cache/v2/hf-modules/transformers_modules/nvidia/NVIDIA-Nemotron-Nano-9B-v2/*",
                "~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-Nano-9B-v2/*"):
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        hits = [h for h in hits if os.path.isfile(os.path.join(h, "modeling_nemotron_h.py"))]
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="module")
def nh():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    d = _module_dir()
    if d is None:
        pytest.skip("NemotronH remote code not found (set V2_NEMOTRON_H_DIR)")
    try:
        import causal_conv1d  # noqa: F401
        import mamba_ssm  # noqa: F401
    except ImportError:
        pytest.skip("mamba_ssm / causal_conv1d kernels required (the real decode path)")
    name = "v2_tiny_nemotron_h"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(d, "__init__.py"),
                                                      submodule_search_locations=[d])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    cfgm = importlib.import_module(name + ".configuration_nemotron_h")
    mod = importlib.import_module(name + ".modeling_nemotron_h")
    assert mod.is_fast_path_available, "decode must use the in-place CUDA kernels"
    return cfgm, mod


class TinyStt(torch.nn.Module):
    """DuplexSTTModel.forward shape (duplex_stt_model.py:466): backbone with cache_params, text + function heads."""

    def __init__(self, llm, vocab):
        super().__init__()
        h = llm.config.hidden_size
        self.llm = llm
        self.lm_head = torch.nn.Linear(h, vocab, bias=False)
        self.function_head = torch.nn.Linear(h, vocab, bias=False)

    def forward(self, input_embeds, cache=None, **kw):
        kwargs = {"inputs_embeds": input_embeds, "return_dict": True, "use_cache": cache is not None}
        if cache is not None:
            kwargs["cache_params"] = cache
        out = self.llm(**kwargs)
        hs = out["last_hidden_state"]
        ans = {"text_logits": self.lm_head(hs), "function_logits": self.function_head(hs)}
        if cache is not None:
            ans["cache"] = out.cache_params
        return ans


def build(nh, seed=0):
    from mynah_v2.model import perf
    from mynah_v2.model.llm_cache import _hybrid_cache_cls, cache_passing_attention, position_injecting_forward
    cfgm, mod = nh
    torch.manual_seed(seed)
    cfg = cfgm.NemotronHConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=6,
                               hybrid_override_pattern="M*-M*-", num_attention_heads=4, head_dim=32,
                               num_key_value_heads=2, ssm_state_size=16, mamba_num_heads=8, mamba_n_groups=1,
                               mamba_head_dim=16, mamba_d_conv=4, mamba_chunk_size=16, max_position_embeddings=4096)
    cfg._attn_implementation = "eager"          # = the real model: NemotronHAttention (calls F.sdpa internally)
    llm = mod.NemotronHModel(cfg).cuda().eval()
    stt = TinyStt(llm, 128).cuda().eval()
    llm.forward = position_injecting_forward(llm.forward, holder=llm)            # as enable_hybrid_llm_cache
    for layer in llm.layers:
        if layer.block_type == "attention":
            layer.mixer.forward = cache_passing_attention(layer.mixer.forward, llm)
    eng = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=stt)), llm_cache_info={"tiny": 1})
    perf.enable("mamba_host_pos", eng)
    cls = _hybrid_cache_cls(llm)
    new_cache = lambda: cls(cfg, 1, dtype=torch.float32, device="cuda", ssm_dtype=torch.float32)  # noqa: E731
    return eng, stt, new_cache


def inputs(seed, n, prefill=6, h=128):
    g = torch.Generator("cuda").manual_seed(seed)
    return torch.randn(1, prefill, h, device="cuda", generator=g), [torch.randn(1, 1, h, device="cuda", generator=g)
                                                                   for _ in range(n)]


def run(stt, cache, pre, steps):
    outs = []
    if pre is not None:
        stt(pre, cache=cache)
    for e in steps:
        o = stt(e, cache=cache)
        outs.append(torch.cat([o["text_logits"][:, -1], o["function_logits"][:, -1]], -1).clone())
    return torch.cat(outs)


def close(a, b):
    return torch.allclose(a, b, atol=1e-4, rtol=1e-4), float((a - b).abs().max())


def assert_state_equal(c_ref, c_got, stt):
    """Mamba conv/SSM state and attention K/V after the session (logit parity alone can hide a drifting state)."""
    for i, layer in enumerate(stt.llm.layers):
        if layer.block_type == "mamba":
            for a, b in ((c_ref.conv_states[i], c_got.conv_states[i]), (c_ref.ssm_states[i], c_got.ssm_states[i])):
                assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (i, float((a - b).abs().max()))
        elif layer.block_type == "attention":
            for a, b in ((c_ref.key_cache[i], c_got.key_cache[i]), (c_ref.value_cache[i], c_got.value_cache[i])):
                assert a.shape == b.shape, (i, a.shape, b.shape)
                assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (i, float((a - b).abs().max()))


@torch.inference_mode()
def test_graph_matches_eager_across_buckets(nh):
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(1, 130)                                     # positions 6..135 → buckets 64, 128, 256
    cref = new_cache()
    ref = run(stt, cref, pre, steps)
    tg = ThinkerGraph(eng, lmax=1024)
    cg = new_cache()
    got = run(stt, cg, pre, steps)
    tg.remove()                                                     # unconverts: state back on the upstream cache
    ok, err = close(got, ref)
    assert ok, err
    assert_state_equal(cref, cg, stt)
    s = tg.stats
    assert s["captures"] == 3 and s["eager"] == 1 and s["replays"] == 127, s   # prefill eager, 3 captures
    assert s["bucket_transitions"] == 2


def mamba_attn_snapshot(stt, cache):
    """Clone of the recurrent state a session carries after a step (Mamba conv/SSM per layer, attention K/V)."""
    snap = {}
    for i, layer in enumerate(stt.llm.layers):
        if layer.block_type == "mamba":
            snap[i] = (cache.conv_states[i].clone(), cache.ssm_states[i].clone())
    return snap


@torch.inference_mode()
def test_state_after_every_step_incl_capture_steps(nh):
    """Owner gate: the state must match eager IMMEDIATELY after each step — in particular right after each capture
    (a double advance from warm-up + capture would show at that step, not be hidden by a later compensation).
    Mamba conv/SSM (the persistent graph buffers the cache lists point at) and the attention K/V prefix."""
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(3, 75)                                      # captures at buckets 64 (step 0) and 128 (step 58)
    cref = new_cache()
    stt(pre, cache=cref)
    ref_states, ref_kv = [], []
    attn = [i for i, l in enumerate(stt.llm.layers) if l.block_type == "attention"]
    for e in steps:
        stt(e, cache=cref)
        ref_states.append(mamba_attn_snapshot(stt, cref))
        ref_kv.append({i: (cref.key_cache[i].clone(), cref.value_cache[i].clone()) for i in attn})
    tg = ThinkerGraph(eng, lmax=1024)
    cg = new_cache()
    stt(pre, cache=cg)
    actions = []
    for k, e in enumerate(steps):
        before = dict(tg.stats)
        stt(e, cache=cg)
        actions.append("capture" if tg.stats["captures"] > before["captures"] else "replay")
        got = mamba_attn_snapshot(stt, cg)                         # cg lists point at tg's persistent buffers
        for i, (c_ref, s_ref) in ref_states[k].items():
            c_got, s_got = got[i]
            assert torch.allclose(c_got, c_ref, atol=1e-4, rtol=1e-4), (k, actions[-1], "conv", i)
            assert torch.allclose(s_got, s_ref, atol=1e-4, rtol=1e-4), (k, actions[-1], "ssm", i)
        t = int(cg._v2_pos)
        for j, i in enumerate(attn):
            k_ref, v_ref = ref_kv[k][i]
            assert k_ref.shape[-2] == t
            assert torch.allclose(tg.K[j, :, :, :t], k_ref, atol=1e-4, rtol=1e-4), (k, actions[-1], "K", i)
            assert torch.allclose(tg.V[j, :, :, :t], v_ref, atol=1e-4, rtol=1e-4), (k, actions[-1], "V", i)
    tg.remove()
    assert actions.count("capture") == 2 and actions[0] == "capture", actions


@torch.inference_mode()
def test_padded_kv_window_is_masked(nh):
    """The bucket is padded beyond t: whatever sits there must not change the result (upstream never passes
    attention_mask to the attention mixer — the graph path must). Poison the padding before every step."""
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(2, 80)
    ref = run(stt, new_cache(), pre, steps)
    tg = ThinkerGraph(eng, lmax=1024)
    cache = new_cache()
    stt(pre, cache=cache)
    outs = []
    g = torch.Generator("cuda").manual_seed(5)
    for e in steps:
        if tg.K is not None:
            t = int(cache._v2_pos)                                  # position about to be written
            tg.K[:, :, :, t:] = 5.0 + torch.randn(tg.K[:, :, :, t:].shape, device="cuda", generator=g)
            tg.V[:, :, :, t:] = torch.randn(tg.V[:, :, :, t:].shape, device="cuda", generator=g) * 7
        o = stt(e, cache=cache)
        outs.append(torch.cat([o["text_logits"][:, -1], o["function_logits"][:, -1]], -1).clone())
    tg.remove()
    ok, err = close(torch.cat(outs), ref)
    assert ok, err


@torch.inference_mode()
def test_session_change_and_return_match_eager(nh):
    """A 40 steps → B 50 steps → A 40 more: unconvert/convert must hand state back and forth exactly."""
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(11, 80)
    pb, sb = inputs(12, 50)
    ref_a = run(stt, new_cache(), pa, sa)
    ref_b = run(stt, new_cache(), pb, sb)
    tg = ThinkerGraph(eng, lmax=1024)
    ca, cb = new_cache(), new_cache()
    a1 = run(stt, ca, pa, sa[:40])
    b = run(stt, cb, pb, sb)
    a2 = run(stt, ca, None, sa[40:])
    tg.remove()
    for got, ref in ((torch.cat([a1, a2]), ref_a), (b, ref_b)):
        ok, err = close(got, ref)
        assert ok, err
    assert tg.stats["converted"] == 3                               # A, B, A again


@torch.inference_mode()
def test_capture_and_replay_in_worker_thread_while_main_syncs(nh):
    """FC-thread shape: the first decode steps (captures included) run in a worker while the main thread issues
    synchronizing CUDA work; then the main thread continues the same session on the worker-captured graphs."""
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(21, 100)
    ref = run(stt, new_cache(), pre, steps)
    tg = ThinkerGraph(eng, lmax=1024)
    cache = new_cache()
    box, errs = {}, []

    def worker():
        try:
            with torch.inference_mode():
                box["a"] = run(stt, cache, pre, steps[:70])
        except Exception as e:                                       # noqa: BLE001
            errs.append(e)
    th = threading.Thread(target=worker)
    x = torch.randn(256, 256, device="cuda")
    th.start()
    n = 0
    while th.is_alive():
        (x @ x).sum().cpu()
        n += 1
    th.join()
    assert not errs, errs
    b = run(stt, cache, None, steps[70:])
    tg.remove()
    ok, err = close(torch.cat([box["a"], b]), ref)
    assert ok, err
    assert n > 0 and tg.stats["captures"] == 2                      # 64 and 128 captured in the worker


@torch.inference_mode()
def test_lmax_fallback_continues_eagerly(nh):
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(31, 90)
    ref = run(stt, new_cache(), pre, steps)
    tg = ThinkerGraph(eng, lmax=64)
    got = run(stt, new_cache(), pre, steps)
    tg.remove()
    ok, err = close(got, ref)
    assert ok, err
    assert tg.stats["eager_reasons"].get("reached_lmax", 0) >= 1
    assert tg.stats["converted"] == 1                               # past lmax: no convert/unconvert per step


@torch.inference_mode()
def test_remove_restores_upstream_blocks(nh):
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, _ = build(nh)
    before = [l.forward for l in stt.llm.layers]
    tg = ThinkerGraph(eng, lmax=256)
    assert all(l.forward is not b for l, b in zip(stt.llm.layers, before))
    tg.remove()
    assert all(l.forward == b for l, b in zip(stt.llm.layers, before))


@pytest.mark.parametrize("bug", ["drop_mask", "orig_block"])
@torch.inference_mode()
def test_negative_controls_the_parity_gate_catches_both_known_bugs(nh, bug):
    """Re-inject each bug found on 2026-11-12 (unmasked padded K/V window; upstream block with its default-stream
    switch under capture) and require the SAME parity check used above to fail. A gate that stays green with the bug
    in would be worse than no gate."""
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(4, 70)
    ref = run(stt, new_cache(), pre, steps)
    tg = ThinkerGraph(eng, lmax=1024)
    tg._neg_control.add(bug)
    try:
        got = run(stt, new_cache(), pre, steps)
    except Exception:                                               # noqa: BLE001  (a crash also counts as caught)
        got = None
    tg.remove()
    if got is not None:
        ok, err = close(got, ref)
        assert not ok, f"parity gate did NOT catch {bug} (max diff {err})"


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(pytest.main([str(Path(__file__)), "-q", "-m", "gpu"]))
