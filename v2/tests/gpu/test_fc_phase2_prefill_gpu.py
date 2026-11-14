"""FC phase-2 continuation prefill (mynah_v2.model.fc_phase2_prefill) on a TINY NemotronH built from the real remote
code with the real CUDA kernels and V2 cache patches (same builder as test_thinker_graph_gpu). The continuation over T
positions must equal T single-frame decode steps: per-position logits, the Mamba conv/SSM states and the attention K/V,
and the next decode step after it. FP32 strict; BF16 reported. Seconds, no 11B load."""

import pytest
import torch

from test_thinker_graph_gpu import TinyStt, assert_state_equal, build, inputs, nh  # noqa: F401  (fixture + helpers)

pytestmark = pytest.mark.gpu


def _seq(stt, cache, steps):
    out = []
    for e in steps:
        o = stt(e, cache=cache)
        out.append(o["function_logits"][:, -1].clone())
    return torch.cat(out)


@torch.inference_mode()
@pytest.mark.parametrize("T", [1, 5, 34, 87])
def test_continuation_equals_sequential_decode_fp32(nh, T):
    from mynah_v2.model.fc_phase2_prefill import continuation_forward
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(3, T + 1)
    c_ref, c_got = new_cache(), new_cache()
    stt(pre, cache=c_ref)
    stt(pre, cache=c_got)
    ref = _seq(stt, c_ref, steps[:T])
    h = continuation_forward(stt.llm, torch.cat(steps[:T], 1), c_got)
    got = stt.function_head(h)[0]
    assert got.shape == ref.shape
    err = float((got - ref).abs().max())
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), err
    assert c_got._v2_pos == c_ref._v2_pos
    assert_state_equal(c_ref, c_got, stt)
    nxt_ref = stt(steps[T], cache=c_ref)["function_logits"][:, -1]          # the cache stays usable
    nxt_got = stt(steps[T], cache=c_got)["function_logits"][:, -1]
    assert torch.allclose(nxt_got, nxt_ref, atol=1e-4, rtol=1e-4), float((nxt_got - nxt_ref).abs().max())


@torch.inference_mode()
def test_chunked_equals_unchunked(nh):
    from mynah_v2.model.fc_phase2_prefill import continuation_forward
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(4, 34)
    a, b = new_cache(), new_cache()
    stt(pre, cache=a)
    stt(pre, cache=b)
    x = torch.cat(steps, 1)
    ha = continuation_forward(stt.llm, x, a)
    hb = continuation_forward(stt.llm, x, b, chunk=16)
    assert torch.allclose(ha, hb, atol=1e-4, rtol=1e-4), float((ha - hb).abs().max())
    assert_state_equal(a, b, stt)


@torch.inference_mode()
def test_bf16_drift_report(nh, capsys):
    """BF16 (the serving dtype): not exact by construction; report the drift and require the same argmax."""
    from mynah_v2.model.fc_phase2_prefill import continuation_forward
    eng, stt, new_cache = build(nh)
    stt.to(torch.bfloat16)
    cfg = stt.llm.config
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    cls = _hybrid_cache_cls(stt.llm)
    mk = lambda: cls(cfg, 1, dtype=torch.bfloat16, device="cuda", ssm_dtype=torch.float32)  # noqa: E731
    pre, steps = inputs(5, 34)
    pre, steps = pre.bfloat16(), [s.bfloat16() for s in steps]
    a, b = mk(), mk()
    stt(pre, cache=a)
    stt(pre, cache=b)
    ref = _seq(stt, a, steps).float()
    got = stt.function_head(continuation_forward(stt.llm, torch.cat(steps, 1), b))[0].float()
    d = (got - ref).abs()
    with capsys.disabled():
        print(f"\nBF16 continuation vs sequential: max |dlogit| {float(d.max()):.4f}, mean {float(d.mean()):.5f}, "
              f"argmax equal {int((got.argmax(-1) == ref.argmax(-1)).sum())}/{ref.shape[0]}")
    assert float(d.max()) < 0.5
    assert int((got.argmax(-1) == ref.argmax(-1)).sum()) >= ref.shape[0] - 1


@torch.inference_mode()
def test_deferred_interface_returns_the_natural_prediction(nh):
    """The deferred phase-2 interface on a simulated loop: forced steps return PAD and leave the cache; the natural
    step returns the same function token as the sequential loop and leaves the same state."""
    import threading
    from mynah_v2.model.fc_phase2_prefill import DeferredPhase2
    eng, stt, new_cache = build(nh)
    stt.text_pad_id = 12
    pre, steps = inputs(6, 34)

    def real(input_embeds, cache=None, **kw):
        o = stt(input_embeds, cache=cache)
        return {"predicted_token": o["text_logits"][:, -1].argmax(-1), "asr_predicted_token": o["text_logits"][:, -1].argmax(-1),
                "function_predicted_token": o["function_logits"][:, -1].argmax(-1), "cache": cache}
    c_ref, c_got = new_cache(), new_cache()
    stt(pre, cache=c_ref)
    stt(pre, cache=c_got)
    seq_tok = None
    for e in steps:
        seq_tok = real(e, cache=c_ref)["function_predicted_token"]
    fs = {"forced_function_tokens": list(range(33))}                # 33 forced + 1 natural = 34 steps
    d = DeferredPhase2(real, stt, fs, threading.get_ident(), {})
    tok = None
    for e in steps:
        ans = d(e, cache=c_got)
        if fs["forced_function_tokens"]:
            assert int(ans["function_predicted_token"]) == 12 and c_got._v2_pos == pre.shape[1]
            fs["forced_function_tokens"].pop(0)                     # the upstream state machine pops after the call
        else:
            tok = ans["function_predicted_token"]
    assert int(tok) == int(seq_tok)
    assert c_got._v2_pos == c_ref._v2_pos
    assert_state_equal(c_ref, c_got, stt)
