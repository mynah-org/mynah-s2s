"""BatchedThinker bookkeeping on CPU (no CUDA, no Mamba kernels): RowTable invariants, and the whole backend driven
over a FAKE hybrid model that has the same call structure as NemotronH (block type name, stream-neutral patch hooks,
cache.update for attention, in-place conv/SSM state updates in decode, `cache_position[0] > 0` branch, mask only
passed to the attention mixer in graph mode). The real numerics are covered by tests/gpu/test_thinker_batch_gpu.py;
these tests prove the row/position/mask logic (I1, I2, I3, I5, eviction, negative controls) independently of it."""

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mynah_v2.model.eartts_graph import bucket_for
from mynah_v2.model.thinker_batch import BatchedThinker, RowTable
from mynah_v2.runtime.batch_contract import BackendMemory, ThinkerBatchBackend, ThinkerOut

H, KVH, HD, C, KC, S = 8, 2, 4, 6, 3, 4
PATTERN = "M*M*"


# ------------------------------------------------------------------------------------------------ RowTable
def test_rowtable_attach_release_compacts():
    t = RowTable(3)
    assert [t.attach(s, p) for s, p in (("a", 5), ("b", 9), ("c", 2))] == [0, 1, 2]
    with pytest.raises(RuntimeError):
        t.attach("d", 3)
    with pytest.raises(KeyError):
        t.attach("a", 3)
    assert t.release("a") == (0, 2)                   # last row (c) moves into the hole
    assert t.row_of == {"b": 1, "c": 0} and t.sid_at == ["c", "b", None]
    assert t.release("b") == (1, None)
    assert t.attach("d", 7) == 1
    with pytest.raises(ValueError):
        t.attach("e", 0)                               # not prefilled


def test_rowtable_arrange_random_invariants():
    rng = random.Random(0)
    for _ in range(300):
        R = rng.randint(1, 6)
        t = RowTable(R)
        sids = [f"s{i}" for i in range(rng.randint(1, R))]
        for s in sids:
            t.attach(s, rng.randint(1, 100))
        for _ in range(5):
            part = rng.sample(sids, rng.randint(1, len(sids)))
            before = dict(t.row_of)
            swaps = t.arrange(part)
            assert sorted(t.row_of[s] for s in part) == list(range(len(part)))
            assert sorted(t.row_of.values()) == list(range(len(sids)))      # still compact, bijective
            assert all(t.sid_at[r] == s for s, r in t.row_of.items())
            rows = [r for ab in swaps for r in ab]
            assert len(rows) == len(set(rows))                               # disjoint swaps
            moved = {s for s in sids if t.row_of[s] != before[s]}
            assert moved <= set(t.sid_at[r] for r in rows)


def test_rowtable_shared_bucket_is_max_position():
    t = RowTable(4)
    t.attach("a", 10)
    t.attach("b", 63)
    t.attach("c", 300)
    assert t.bucket(["a", "b"], 1024) == 64
    assert t.bucket(["a", "b", "c"], 1024) == bucket_for(301, 1024) == 512
    t.advance(["b"])
    assert t.bucket(["a", "b"], 1024) == 128 and t.pos["a"] == 10


# ------------------------------------------------------------------------------------------------ fake hybrid model
class FakeCache:
    def __init__(self):
        self.key_cache = [torch.zeros(1, 0) for _ in PATTERN]
        self.value_cache = [torch.zeros(1, 0) for _ in PATTERN]
        self.conv_states = [torch.zeros(1, C, KC) if p == "M" else torch.zeros(1, 0) for p in PATTERN]
        self.ssm_states = [torch.zeros(1, C, S) if p == "M" else torch.zeros(1, 0) for p in PATTERN]

    def update(self, k, v, li, *a, **kw):
        if self.key_cache[li].shape[-1] == 0:
            self.key_cache[li], self.value_cache[li] = k, v
        else:
            self.key_cache[li] = torch.cat([self.key_cache[li], k], 2)
            self.value_cache[li] = torch.cat([self.value_cache[li], v], 2)
        return self.key_cache[li], self.value_cache[li]


class Attn(torch.nn.Module):
    def __init__(self, li, holder):
        super().__init__()
        self.layer_idx, self.holder = li, [holder]
        self.num_key_value_heads, self.head_dim = KVH, HD
        self.q, self.k, self.v = (torch.nn.Linear(H, KVH * HD) for _ in range(3))
        self.o = torch.nn.Linear(KVH * HD, H)

    def forward(self, h, attention_mask=None, cache_position=None):
        b, t, _ = h.shape
        sh = lambda x: x.view(b, t, KVH, HD).transpose(1, 2)   # noqa: E731
        q, k, v = sh(self.q(h)), sh(self.k(h)), sh(self.v(h))
        k, v = self.holder[0]._v2_active_cache.update(k, v, self.layer_idx)
        m = None if attention_mask is None else attention_mask[..., :k.shape[-2]]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=m is None and t > 1)
        return self.o(o.transpose(1, 2).reshape(b, t, KVH * HD)), None, None


class Mamba(torch.nn.Module):
    def __init__(self, li):
        super().__init__()
        self.layer_idx = li
        self.inp, self.out = torch.nn.Linear(H, C), torch.nn.Linear(C, H)
        self.w = torch.nn.Parameter(torch.randn(C, KC) * 0.5)
        self.bv, self.cv = torch.nn.Parameter(torch.randn(C, S)), torch.nn.Parameter(torch.randn(C, S))

    def _one(self, conv, ssm, p):                       # p [B, C]; in place, like the CUDA decode kernels
        conv.copy_(torch.cat([conv[..., 1:], p[..., None]], -1))
        y = torch.tanh((conv * self.w).sum(-1))
        ssm.mul_(0.9).add_(y[..., None] * self.bv)
        return (ssm * self.cv).sum(-1)

    def forward(self, h, cache_params=None, cache_position=None):
        p = self.inp(h)
        li = self.layer_idx
        if cache_position[0] > 0:
            return self.out(self._one(cache_params.conv_states[li], cache_params.ssm_states[li], p[:, 0]))[:, None]
        conv = torch.zeros(h.shape[0], C, KC)
        ssm = torch.zeros(h.shape[0], C, S)
        ys = [self._one(conv, ssm, p[:, i]) for i in range(h.shape[1])]
        cache_params.conv_states[li], cache_params.ssm_states[li] = conv, ssm
        return self.out(torch.stack(ys, 1))


class NemotronHBlock(torch.nn.Module):                  # same class name: ThinkerGraph patches it
    def __init__(self, li, kind, holder):
        super().__init__()
        self.block_type = {"M": "mamba", "*": "attention"}[kind]
        self.norm = torch.nn.LayerNorm(H)
        self.residual_in_fp32 = False
        self.mixer = Mamba(li) if kind == "M" else Attn(li, holder)

    def forward(self, h, cache_params=None, cache_position=None, attention_mask=None):
        r = h
        h = self.norm(h)
        if self.block_type == "mamba":
            h = self.mixer(h, cache_params=cache_params, cache_position=cache_position)
        else:
            h = self.mixer(h, cache_position=cache_position)[0]   # upstream: no mask to the mixer
        return r + h


class FakeLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.layers = torch.nn.ModuleList([NemotronHBlock(i, k, self) for i, k in enumerate(PATTERN)])
        self._v2_hostpos_orig = True
        self._v2_active_cache = None

    def _update_causal_mask(self, attention_mask, x, cache_position):
        tl = int(cache_position[-1]) + 1
        m = torch.zeros(x.shape[1], tl)
        if x.shape[1] > 1:
            m = torch.triu(torch.full((x.shape[1], tl), torch.finfo(x.dtype).min), diagonal=1 + tl - x.shape[1])
        return m[None, None].expand(x.shape[0], 1, -1, -1)

    def forward(self, x, cache):
        pos, t = int(cache._v2_pos), x.shape[1]
        cp = torch.arange(pos, pos + t)
        mask = self._update_causal_mask(None, x, cp)
        self._v2_active_cache = cache
        try:
            for blk in self.layers:
                x = blk(x, cache_params=cache, cache_position=cp,
                        attention_mask=mask if blk.block_type == "attention" else None)
        finally:
            self._v2_active_cache = None
        cache._v2_pos = pos + t
        return x


class FakeStt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = FakeLLM()
        self.lm_head = torch.nn.Linear(H, 16)
        self.function_head = torch.nn.Linear(H, 16)

    def forward(self, input_embeds, cache=None, **kw):
        h = self.llm(input_embeds, cache)
        return {"text_logits": self.lm_head(h), "function_logits": self.function_head(h), "cache": cache}


def new_cache():
    c = FakeCache()
    c._v2_pos = 0
    return c


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    torch.manual_seed(0)
    stt = FakeStt().double()
    torch.set_default_dtype(torch.float64)              # fake caches/state created in fp64 → tight parity
    yield SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=stt))), stt
    torch.set_default_dtype(torch.float32)


def frames(seed, n, prefill):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, prefill, H, generator=g), [torch.randn(1, 1, H, generator=g) for _ in range(n)]


def logits(o):
    if isinstance(o, ThinkerOut):
        return torch.cat([o.text_logits[:, -1], o.function_logits[:, -1]], -1)
    return torch.cat([o["text_logits"][:, -1], o["function_logits"][:, -1]], -1)


def solo(stt, pre, steps):
    c = new_cache()
    stt(pre, cache=c)
    return torch.cat([logits(stt(x, cache=c)) for x in steps]), c


def prefilled(stt, pre, steps=()):
    c = new_cache()
    stt(pre, cache=c)
    for x in steps:
        stt(x, cache=c)
    return c


def state_equal(a, b, exact=False):
    for li, p in enumerate(PATTERN):
        pairs = ((a.conv_states[li], b.conv_states[li]), (a.ssm_states[li], b.ssm_states[li])) if p == "M" else \
            ((a.key_cache[li], b.key_cache[li]), (a.value_cache[li], b.value_cache[li]))
        for x, y in pairs:
            assert x.shape == y.shape, (li, x.shape, y.shape)
            assert (torch.equal(x, y) if exact else torch.allclose(x, y, atol=1e-10)), (li, float((x - y).abs().max()))
    assert int(a._v2_pos) == int(b._v2_pos)


def close(a, b):
    return torch.allclose(a, b, atol=1e-10, rtol=1e-10), float((a - b).abs().max())


def poison(bt, g):
    """Garbage in every K/V position that is not a valid key of an attached row (incl. free rows)."""
    for r in range(bt.max_rows):
        sid = bt.table.sid_at[r]
        t = bt.table.pos[sid] if sid is not None else 0
        bt.bK[:, r, :, t:] = 5 + torch.randn(bt.bK[:, r, :, t:].shape, generator=g)
        bt.bV[:, r, :, t:] = 7 * torch.randn(bt.bV[:, r, :, t:].shape, generator=g)


# ------------------------------------------------------------------------------------------------ backend on the fake
def test_protocol_and_b2_parity_across_bucket(model):
    eng, stt = model
    pa, sa = frames(1, 70, 5)
    pb, sb = frames(2, 70, 40)                          # different lengths; positions cross 64 → bucket 128
    ra, ca_ref = solo(stt, pa, sa)
    rb, cb_ref = solo(stt, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=False)
    assert isinstance(bt, ThinkerBatchBackend)
    assert bt.attach("A", prefilled(stt, pa)) == 0 and bt.attach("B", prefilled(stt, pb)) == 1
    buckets, ga, gb = [], [], []
    for xa, xb in zip(sa, sb):
        o = bt.step({"A": xa, "B": xb})
        buckets.append(bt.bbucket)
        ga.append(logits(o["A"]))
        gb.append(logits(o["B"]))
    assert buckets[0] == 64 and buckets[-1] == 128      # I5: shared bucket from the max position
    assert bt.positions() == {"A": 75, "B": 110}
    for got, ref in ((ga, ra), (gb, rb)):
        ok, err = close(torch.cat(got), ref)
        assert ok, err
    state_equal(bt.detach("A"), ca_ref)
    state_equal(bt.detach("B"), cb_ref)
    bt.close()


def test_i2_round_trip_exact_and_eager_continuation(model):
    eng, stt = model
    pre, steps = frames(3, 40, 7)
    ref, cref = solo(stt, pre, steps)
    c = prefilled(stt, pre)
    outs = [logits(stt(x, cache=c)) for x in steps[:5]]
    snap = {k: [t.clone() for t in getattr(c, k)] for k in ("key_cache", "value_cache", "conv_states", "ssm_states")}
    bt = BatchedThinker(eng, max_rows=3, lmax=256, graphs=False)
    bt.attach("A", c)
    with pytest.raises(RuntimeError):
        stt(steps[5], cache=c)                          # attached cache is guarded
    back = bt.detach("A")
    assert back is c and c._v2_pos == 12 and not hasattr(c, "_v2_batch")
    for k, ts in snap.items():
        for a, b in zip(ts, getattr(c, k)):
            assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), k
    bt.attach("A", c)
    outs += [logits(bt.step_one("A", x)) for x in steps[5:25]]
    c = bt.detach("A")
    outs += [logits(stt(x, cache=c)) for x in steps[25:]]          # upstream eager continues it
    ok, err = close(torch.cat(outs), ref)
    assert ok, err
    state_equal(c, cref)
    bt.close()


def test_i1_free_and_paused_rows_untouched(model):
    eng, stt = model
    pa, sa = frames(4, 50, 6)
    pb, sb = frames(5, 50, 20)
    ra, _ = solo(stt, pa, sa)
    rb, _ = solo(stt, pb, sb[:10])
    bt = BatchedThinker(eng, max_rows=4, lmax=256, graphs=False)
    bt.attach("B", prefilled(stt, pb))                  # B at row 0, A at row 1: stepping A alone forces a swap
    bt.attach("A", prefilled(stt, pa))
    got_a, got_b = [], []
    for i, xa in enumerate(sa):
        free = (bt.bK[:, 2:].clone(), bt.bV[:, 2:].clone(), bt.bmask[2:].clone(),
                [b[2:].clone() for b in bt.bconv], [b[2:].clone() for b in bt.bssm])
        if i < 10:
            o = bt.step({"A": xa, "B": sb[i]})
            got_b.append(logits(o["B"]))
        else:
            before = bt.snapshot("B")
            o = bt.step({"A": xa})                      # B attached but paused
            after = bt.snapshot("B")
            for k in ("K", "V", "mask"):
                assert torch.equal(before[k], after[k]), (i, k)
            assert before["pos"] == after["pos"] == before["posv"] == after["posv"]
            assert all(torch.equal(x, y) for x, y in zip(before["conv"] + before["ssm"], after["conv"] + after["ssm"]))
        got_a.append(logits(o["A"]))
        assert torch.equal(free[0], bt.bK[:, 2:]) and torch.equal(free[1], bt.bV[:, 2:])
        assert torch.equal(free[2], bt.bmask[2:])
        assert all(torch.equal(x, y[2:]) for x, y in zip(free[3] + free[4], bt.bconv + bt.bssm))
    assert bt.bstats["swaps"] >= 1
    for got, ref in ((got_a, ra), (got_b, rb)):
        ok, err = close(torch.cat(got), ref)
        assert ok, err
    # B resumes after the pause exactly where it stopped
    cb = bt.detach("B")
    rb_full, _ = solo(stt, pb, sb[:20])
    tail = [logits(stt(x, cache=cb)) for x in sb[10:20]]
    ok, err = close(torch.cat(tail), rb_full[10:])
    assert ok, err
    bt.close()


def test_i3_slot_reuse_sequence_with_poisoned_padding(model):
    """{A+B} → detach A, attach C → {B+C} → detach B, re-attach A → {A+C}; each matches its solo run."""
    eng, stt = model
    (pa, sa), (pb, sb), (pc, sc) = frames(6, 60, 30), frames(7, 60, 4), frames(8, 60, 50)
    refs = {k: solo(stt, p, s)[0] for k, (p, s) in {"A": (pa, sa), "B": (pb, sb), "C": (pc, sc)}.items()}
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=False)
    g = torch.Generator().manual_seed(9)
    got = {"A": [], "B": [], "C": []}
    idx = {"A": 0, "B": 0, "C": 0}
    seqs = {"A": sa, "B": sb, "C": sc}
    caches = {}

    def run(sids, n):
        for _ in range(n):
            poison(bt, g)
            o = bt.step({s: seqs[s][idx[s]] for s in sids})
            for s in sids:
                got[s].append(logits(o[s]))
                idx[s] += 1

    bt.attach("A", prefilled(stt, pa))
    bt.attach("B", prefilled(stt, pb))
    run(["A", "B"], 15)
    caches["A"] = bt.detach("A")
    assert bt.attach("C", prefilled(stt, pc)) == 1      # reuses the freed row
    run(["B", "C"], 15)
    bt.detach("B")
    bt.attach("A", caches["A"])
    run(["A", "C"], 20)
    for s in "ABC":
        ok, err = close(torch.cat(got[s]), refs[s][:idx[s]])
        assert ok, (s, err)
    bt.close()


def test_lmax_eviction_does_not_disturb_other_rows(model):
    eng, stt = model
    pa, sa = frames(10, 80, 5)
    pb, sb = frames(11, 80, 40)
    ra, _ = solo(stt, pa, sa)
    rb, _ = solo(stt, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=64, graphs=False)
    bt.attach("A", prefilled(stt, pa))
    bt.attach("B", prefilled(stt, pb))
    ga, gb = [], []
    cb = None
    for xa, xb in zip(sa, sb):
        fr = {"A": xa} if "A" in bt.positions() else {}
        if cb is None:
            fr["B"] = xb
        o = bt.step(fr) if fr else {}
        if "B" in bt.evicted:
            cb = bt.evicted.pop("B")
            assert cb._v2_past_lmax
        elif cb is not None:
            o["B"] = stt(xb, cache=cb)                  # caller continues B eagerly
        if "A" in bt.evicted:
            ca = bt.evicted.pop("A")
            ga.append(logits(o["A"]))
            gb.append(logits(o["B"]))
            break
        ga.append(logits(o["A"]))
        gb.append(logits(o["B"]))
    assert bt.bstats["evictions"] == 2
    for got, ref in ((ga, ra), (gb, rb)):
        ok, err = close(torch.cat(got), ref[:len(got)])
        assert ok, err
    rest = [logits(stt(x, cache=ca)) for x in sa[len(ga):]]
    ok, err = close(torch.cat(ga + rest), ra)
    assert ok, err
    bt.close()


@pytest.mark.parametrize("bug", ["drop_mask", "shared_mask", "shared_pos"])
def test_negative_controls_break_parity(model, bug):
    eng, stt = model
    pa, sa = frames(12, 30, 5)
    pb, sb = frames(13, 30, 33)
    ra, _ = solo(stt, pa, sa)
    rb, _ = solo(stt, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=False)
    bt._neg_control.add(bug)
    bt.attach("A", prefilled(stt, pa))
    bt.attach("B", prefilled(stt, pb))
    out = [bt.step({"A": xa, "B": xb}) for xa, xb in zip(sa, sb)]
    bt.close()
    ok_a, _ = close(torch.cat([logits(o["A"]) for o in out]), ra)
    ok_b, _ = close(torch.cat([logits(o["B"]) for o in out]), rb)
    assert not (ok_a and ok_b), f"parity gate did not catch {bug}"


def test_close_restores_model_and_returns_state(model):
    eng, stt = model
    before = [l.forward for l in stt.llm.layers]
    fwd = stt.forward
    pa, sa = frames(14, 20, 5)
    ra, _ = solo(stt, pa, sa)
    bt = BatchedThinker(eng, max_rows=2, lmax=256, graphs=False)
    c = prefilled(stt, pa)
    bt.attach("A", c)
    got = [logits(bt.step({"A": x})["A"]) for x in sa[:8]]
    m = bt.memory()
    assert isinstance(m, BackendMemory)
    esz = 8                                              # fp64 fake
    n_attn, n_mamba = PATTERN.count("*"), PATTERN.count("M")
    assert m.bytes_per_slot == 2 * n_attn * KVH * 256 * HD * esz + 256 * esz + n_mamba * (C * KC + C * S) * esz
    assert m.graph_pool_bytes == 0
    bt.close()
    assert stt.forward == fwd and all(l.forward == b for l, b in zip(stt.llm.layers, before))
    with pytest.raises(RuntimeError):
        bt.step({"A": sa[8]})
    got += [logits(stt(x, cache=c)) for x in sa[8:]]    # close wrote A back into its own cache
    ok, err = close(torch.cat(got), ra)
    assert ok, err
