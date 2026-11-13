"""cache_position injection for the V2 incremental Thinker (no model needed)."""
import pytest
import torch

from mynah_v2.model.llm_cache import position_injecting_forward


class FakeCache:
    """Mimics NemotronH: get_seq_length() is wrong (1) on a fresh cache, so it must not be used."""
    def __init__(self):
        self.n = 0

    def get_seq_length(self):
        return self.n or 1


def _fwd():
    seen = []

    def orig(**kw):
        seen.append(kw.get("cache_position"))
        c = kw.get("cache_params")
        if c is not None:
            c.n += kw["inputs_embeds"].shape[1]
        return {"ok": True}
    return position_injecting_forward(orig), seen


def test_prefill_then_single_steps_get_increasing_positions():
    f, seen = _fwd()
    c = FakeCache()
    f(inputs_embeds=torch.zeros(1, 5, 8), cache_params=c)      # prompt prefill
    f(inputs_embeds=torch.zeros(1, 1, 8), cache_params=c)
    f(inputs_embeds=torch.zeros(1, 1, 8), cache_params=c)
    assert [s.tolist() for s in seen] == [[0, 1, 2, 3, 4], [5], [6]]


def test_without_cache_nothing_is_injected_and_explicit_positions_are_kept():
    f, seen = _fwd()
    f(inputs_embeds=torch.zeros(1, 3, 8))
    c = FakeCache()
    f(inputs_embeds=torch.zeros(1, 1, 8), cache_params=c, cache_position=torch.tensor([42]))
    assert seen[0] is None and seen[1].tolist() == [42]


def test_multi_frame_decode_after_prefill_is_refused():
    f, _ = _fwd()
    c = FakeCache()
    f(inputs_embeds=torch.zeros(1, 4, 8), cache_params=c)
    with pytest.raises(ValueError):
        f(inputs_embeds=torch.zeros(1, 2, 8), cache_params=c)


def test_attention_wrapper_receives_the_cache_only_inside_a_cached_call():
    from types import SimpleNamespace
    from mynah_v2.model.llm_cache import cache_passing_attention
    holder = SimpleNamespace()
    got = []
    attn = cache_passing_attention(lambda h, **kw: got.append(kw.get("past_key_value")), holder)
    c = FakeCache()

    def backbone(**kw):
        attn(kw["inputs_embeds"])                 # what NemotronHBlock does: no past_key_value
        if kw.get("cache_params") is not None:
            kw["cache_params"].n += kw["inputs_embeds"].shape[1]
    f = position_injecting_forward(backbone, holder=holder)
    f(inputs_embeds=torch.zeros(1, 3, 8), cache_params=c)
    f(inputs_embeds=torch.zeros(1, 3, 8))         # uncached call: nothing injected
    attn(torch.zeros(1, 1, 8))                    # outside any call
    assert got == [c, None, None]


def test_host_position_is_exposed_during_the_call_and_cleared_after():
    from mynah_v2.model.llm_cache import position_injecting_forward
    import torch

    class Holder:
        pass
    h = Holder()
    seen = []

    class Cache:
        pass
    c = Cache()

    def orig(**kw):
        seen.append(h._v2_host_pos)
        return None
    f = position_injecting_forward(orig, holder=h)
    f(inputs_embeds=torch.zeros(1, 3, 4), cache_params=c)
    f(inputs_embeds=torch.zeros(1, 1, 4), cache_params=c)
    assert seen == [(0, 3), (3, 1)] and h._v2_host_pos is None


def test_bf16_cache_cast_never_mutates_the_shared_cache():
    import torch
    from mynah_v2.model.perf import _cast_cache

    class C:
        pass
    c = C()
    c.key_cache = [torch.zeros(2, dtype=torch.float32)]
    c.value_cache = [torch.ones(2, dtype=torch.float32)]
    n = _cast_cache(c, torch.bfloat16)
    assert n is not c and n.key_cache[0].dtype == torch.bfloat16
    assert c.key_cache[0].dtype == torch.float32 and c.value_cache[0].dtype == torch.float32
    assert _cast_cache(n, torch.bfloat16) is n
