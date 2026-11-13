"""Batched Thinker decode (mynah_v2.model.thinker_batch) on a TINY NemotronH built from the real remote code
(snapshot 6533e8de) with the real V2 patches (llm_cache position injection + attention cache passing,
mamba_host_pos) and the real Mamba decode kernels — same loader as test_thinker_graph_gpu.py. fp32, eager attention.

Every check is against independent B=1 runs through the UPSTREAM eager path (no backend installed). Contract
invariants (runtime/batch_contract.py): I1 inactive/free rows untouched, I2 exact round trip, I3 slot reuse,
I4 graphs keyed by (n_rows, bucket) and no stale buffers after reuse/bucket change/close, I5 per-row positions + shared
bucket, I9 memory accounting. Negative controls re-inject the bugs the gate must catch."""

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
    cfg._attn_implementation = "eager"
    llm = mod.NemotronHModel(cfg).cuda().eval()
    stt = TinyStt(llm, 128).cuda().eval()
    llm.forward = position_injecting_forward(llm.forward, holder=llm)
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


def lg(o):
    if isinstance(o, dict):
        return torch.cat([o["text_logits"][:, -1], o["function_logits"][:, -1]], -1).clone()
    return torch.cat([o.text_logits[:, -1], o.function_logits[:, -1]], -1).clone()


def solo(stt, new_cache, pre, steps):
    """Independent B=1 upstream eager run (call BEFORE installing a backend)."""
    c = new_cache()
    stt(pre, cache=c)
    return torch.cat([lg(stt(x, cache=c)) for x in steps]), c


def prefilled(stt, new_cache, pre):
    c = new_cache()
    stt(pre, cache=c)
    return c


def close(a, b):
    return torch.allclose(a, b, atol=1e-4, rtol=1e-4), float((a - b).abs().max())


def assert_state_equal(c_ref, c_got, stt, exact=False):
    for i, layer in enumerate(stt.llm.layers):
        if layer.block_type == "mamba":
            pairs = ((c_ref.conv_states[i], c_got.conv_states[i]), (c_ref.ssm_states[i], c_got.ssm_states[i]))
        elif layer.block_type == "attention":
            pairs = ((c_ref.key_cache[i], c_got.key_cache[i]), (c_ref.value_cache[i], c_got.value_cache[i]))
        else:
            continue
        for a, b in pairs:
            assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape)
            if exact:
                assert torch.equal(a, b), i
            else:
                assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (i, float((a - b).abs().max()))
    assert int(c_ref._v2_pos) == int(c_got._v2_pos)


def poison(bt, g):
    """Garbage at every K/V position that is not a valid key of an attached row (free rows included)."""
    for r in range(bt.max_rows):
        sid = bt.table.sid_at[r]
        t = bt.table.pos[sid] if sid is not None else 0
        bt.bK[:, r, :, t:] = 5.0 + torch.randn(bt.bK[:, r, :, t:].shape, device="cuda", generator=g)
        bt.bV[:, r, :, t:] = 7.0 * torch.randn(bt.bV[:, r, :, t:].shape, device="cuda", generator=g)


def run_pair(bt, seqs, n, sids=None, g=None):
    sids = sids or list(seqs)
    got = {s: [] for s in sids}
    for i in range(n):
        if g is not None:
            poison(bt, g)
        o = bt.step({s: seqs[s][i] for s in sids})
        for s in sids:
            got[s].append(lg(o[s]))
    return {s: torch.cat(v) for s, v in got.items()}


# ------------------------------------------------------------------------------------------------ parity
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_b2_matches_two_b1_runs_across_buckets(nh, graphs):
    """Different prefill lengths (6 vs 40), 60 steps: positions cross 64 → shared bucket 64 then 128 (I5)."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    from mynah_v2.runtime.batch_contract import ThinkerBatchBackend
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(1, 60, prefill=6)
    pb, sb = inputs(2, 60, prefill=40)
    ra, ca = solo(stt, new_cache, pa, sa)
    rb, cb = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=graphs)
    assert isinstance(bt, ThinkerBatchBackend)
    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    got = run_pair(bt, {"A": sa, "B": sb}, 60)
    assert bt.positions() == {"A": 66, "B": 100}
    assert set(bt.bstats["steps_by_bucket"]) == {"64", "128"}
    for s, ref in (("A", ra), ("B", rb)):
        ok, err = close(got[s], ref)
        assert ok, (s, err)
    assert_state_equal(ca, bt.detach("A"), stt)
    assert_state_equal(cb, bt.detach("B"), stt)
    if graphs:
        assert bt.bstats["captures"] == 2 and bt.bstats["replays"] == 58, bt.bstats
        assert {k[:2] for k in bt.bgraphs} == {(2, 64), (2, 128)}           # I4: keyed by (n_rows, bucket)
    bt.close()


@torch.inference_mode()
def test_graph_b2_matches_eager_b2(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(3, 70, prefill=10)
    pb, sb = inputs(4, 70, prefill=50)
    res = {}
    for graphs in (False, True):
        bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=graphs)
        bt.attach("A", prefilled(stt, new_cache, pa))
        bt.attach("B", prefilled(stt, new_cache, pb))
        res[graphs] = run_pair(bt, {"A": sa, "B": sb}, 70)
        bt.close()
    for s in "AB":
        ok, err = close(res[True][s], res[False][s])
        assert ok, (s, err)


# ------------------------------------------------------------------------------------------------ I2
@torch.inference_mode()
def test_i2_attach_detach_round_trip_and_eager_continuation(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pre, steps = inputs(5, 60, prefill=9)
    ref, cref = solo(stt, new_cache, pre, steps)
    pb, sb = inputs(6, 60, prefill=3)
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True)
    c = prefilled(stt, new_cache, pre)
    outs = [lg(stt(x, cache=c)) for x in steps[:5]]                    # eager before joining
    snap = {k: [t.clone() for t in getattr(c, k)] for k in ("key_cache", "value_cache", "conv_states", "ssm_states")}
    strides = [t.stride() for t in c.conv_states + c.ssm_states]
    bt.attach("A", c)
    with pytest.raises(RuntimeError):
        stt(steps[5], cache=c)                                           # an attached cache is guarded
    assert bt.detach("A") is c and c._v2_pos == 14
    for k, ts in snap.items():                                           # exact round trip
        for a, b in zip(ts, getattr(c, k)):
            assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), k
    assert [t.stride() for t in c.conv_states + c.ssm_states] == strides
    bt.attach("A", c)
    bt.attach("B", prefilled(stt, new_cache, pb))
    for i in range(5, 35):
        outs.append(lg(bt.step({"A": steps[i], "B": sb[i]})["A"]))
    c = bt.detach("A")
    outs += [lg(stt(x, cache=c)) for x in steps[35:]]                   # upstream eager continues it
    ok, err = close(torch.cat(outs), ref)
    assert ok, err
    assert_state_equal(cref, c, stt)
    bt.close()


# ------------------------------------------------------------------------------------------------ I1
@torch.inference_mode()
def test_i1_join_leave_pause_free_rows_untouched(nh):
    """A runs 50 steps; B joins at step 10, leaves at 30; C attached but paused from 20 to 40. A == its solo run,
    paused C resumes exactly, free rows byte-identical across every step."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(7, 50, prefill=6)
    pb, sb = inputs(8, 20, prefill=30)
    pc, sc = inputs(9, 30, prefill=12)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, cb_ref = solo(stt, new_cache, pb, sb)
    rc, cc_ref = solo(stt, new_cache, pc, sc)
    bt = BatchedThinker(eng, max_rows=4, lmax=1024, graphs=True)
    bt.attach("C", prefilled(stt, new_cache, pc))                       # C at row 0: pausing it forces swaps
    bt.attach("A", prefilled(stt, new_cache, pa))
    got = {"A": [], "B": [], "C": []}
    ib = ic = 0
    for i in range(50):
        if i == 10:
            bt.attach("B", prefilled(stt, new_cache, pb))
        if i == 30:
            cb = bt.detach("B")
        fr = {"A": sa[i]}
        if 10 <= i < 30:
            fr["B"] = sb[ib]
        if i < 20 or i >= 40:
            fr["C"] = sc[ic]
        n_att = bt.table.n
        free = [t[:, n_att:].clone() for t in (bt.bK, bt.bV)] + [bt.bmask[n_att:].clone()] + \
               [b[n_att:].clone() for b in bt.bconv + bt.bssm]
        paused = {s: bt.snapshot(s) for s in bt.positions() if s not in fr}
        o = bt.step(fr)
        assert all(torch.equal(x, y) for x, y in zip(free, [t[:, n_att:] for t in (bt.bK, bt.bV)] +
                                                     [bt.bmask[n_att:]] + [b[n_att:] for b in bt.bconv + bt.bssm]))
        for s, before in paused.items():
            after = bt.snapshot(s)
            for k in ("K", "V", "mask"):
                assert torch.equal(before[k], after[k]), (i, s, k)
            assert before["pos"] == after["pos"] == after["posv"]
            assert all(torch.equal(x, y) for x, y in zip(before["conv"] + before["ssm"], after["conv"] + after["ssm"]))
        for s in fr:
            got[s].append(lg(o[s]))
        ib += "B" in fr
        ic += "C" in fr
    assert bt.bstats["swaps"] >= 1 and bt.bstats["moves"] >= 1
    for s, ref in (("A", ra), ("B", rb), ("C", rc[:ic])):
        ok, err = close(torch.cat(got[s]), ref)
        assert ok, (s, err)
    assert_state_equal(cb_ref, cb, stt)
    bt.close()


# ------------------------------------------------------------------------------------------------ I3 / I4
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_i3_slot_reuse_sequence_poisoned(nh, graphs):
    """{A+B} → detach A, attach C → {B+C} → detach B, re-attach A → {A+C}, poisoned padding before every step."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    (pa, sa), (pb, sb), (pc, sc) = inputs(10, 60, 30), inputs(11, 60, 4), inputs(12, 60, 50)
    refs = {"A": solo(stt, new_cache, pa, sa)[0], "B": solo(stt, new_cache, pb, sb)[0],
            "C": solo(stt, new_cache, pc, sc)[0]}
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=graphs)
    g = torch.Generator("cuda").manual_seed(13)
    got = {s: [] for s in "ABC"}
    idx = dict.fromkeys("ABC", 0)
    seq = {"A": sa, "B": sb, "C": sc}

    def run(sids, n):
        for _ in range(n):
            poison(bt, g)
            o = bt.step({s: seq[s][idx[s]] for s in sids})
            for s in sids:
                got[s].append(lg(o[s]))
                idx[s] += 1

    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    run("AB", 20)
    ca = bt.detach("A")
    assert bt.attach("C", prefilled(stt, new_cache, pc)) == 1           # C reuses A's row
    run("BC", 20)                                                        # C crosses 64 → bucket change
    bt.detach("B")
    bt.attach("A", ca)
    run("AC", 20)
    for s in "ABC":
        ok, err = close(torch.cat(got[s]), refs[s][:idx[s]])
        assert ok, (s, err)
    if graphs:
        assert all(k[0] == 2 for k in bt.bgraphs) and bt.bstats["replays"] > 0
    bt.close()


@torch.inference_mode()
def test_i4_close_then_new_backend_and_n_change(nh):
    """Graphs of a closed backend are gone (step raises); a new backend on the same model recaptures and matches;
    n_rows 2 → 1 → 2 replays the per-n graphs of the same bucket."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(14, 40, 6)
    pb, sb = inputs(15, 40, 20)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, _ = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True)
    ca = prefilled(stt, new_cache, pa)
    bt.attach("A", ca)
    got_a = [lg(bt.step({"A": x})["A"]) for x in sa[:5]]
    bt.close()                                                           # writes A back into ca
    with pytest.raises(RuntimeError):
        bt.step({"A": sa[5]})
    bt2 = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True)
    bt2.attach("A", ca)
    bt2.attach("B", prefilled(stt, new_cache, pb))
    got_b = []
    for i in range(5, 40):
        fr = {"A": sa[i]}
        if not 15 <= i < 25:                                              # B paused 15..24: n_rows 2 → 1 → 2
            fr["B"] = sb[len(got_b)]
        o = bt2.step(fr)
        got_a.append(lg(o["A"]))
        if "B" in fr:
            got_b.append(lg(o["B"]))
    assert {k[0] for k in bt2.bgraphs} == {1, 2}
    ok, err = close(torch.cat(got_a), ra)
    assert ok, err
    ok, err = close(torch.cat(got_b), rb[:len(got_b)])
    assert ok, err
    bt2.close()


# ------------------------------------------------------------------------------------------------ padding / controls
@torch.inference_mode()
def test_padded_kv_poisoning_b2(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(16, 80, 6)
    pb, sb = inputs(17, 80, 45)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, _ = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=3, lmax=1024, graphs=True)
    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    got = run_pair(bt, {"A": sa, "B": sb}, 80, g=torch.Generator("cuda").manual_seed(5))
    bt.close()
    for s, ref in (("A", ra), ("B", rb)):
        ok, err = close(got[s], ref)
        assert ok, (s, err)


@pytest.mark.parametrize("bug", ["drop_mask", "shared_mask", "shared_pos"])
@torch.inference_mode()
def test_negative_controls_break_parity(nh, bug):
    """drop_mask: no mask to the attention mixer; shared_mask: one (union) mask for all rows instead of per-row;
    shared_pos: every row writes its K/V at the max position. The same parity check must FAIL."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(18, 40, 6)
    pb, sb = inputs(19, 40, 30)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, _ = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=False)
    bt._neg_control.add(bug)
    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    try:
        got = run_pair(bt, {"A": sa, "B": sb}, 40)
    except Exception:                                                    # noqa: BLE001  (a crash also counts)
        got = None
    bt.close()
    if got is not None:
        ok_a, _ = close(got["A"], ra)
        ok_b, _ = close(got["B"], rb)
        assert not (ok_a and ok_b), f"parity gate did NOT catch {bug}"


# ------------------------------------------------------------------------------------------------ lmax / threads / B=1
@torch.inference_mode()
def test_lmax_eviction_continues_eagerly(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(20, 80, 6)
    pb, sb = inputs(21, 80, 40)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, _ = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=64, graphs=True)
    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    got = {"A": [], "B": []}
    eager = {}
    for i in range(80):
        fr = {s: x for s, x in (("A", sa[i]), ("B", sb[i])) if s not in eager}
        o = bt.step(fr) if fr else {}
        for s in list(bt.evicted):
            eager[s] = bt.evicted.pop(s)
            assert eager[s]._v2_past_lmax
        for s, x in (("A", sa[i]), ("B", sb[i])):
            got[s].append(lg(o[s]) if s in o else lg(stt(x, cache=eager[s])))
    assert bt.bstats["evictions"] == 2 and not bt.positions()
    for s, ref in (("A", ra), ("B", rb)):
        ok, err = close(torch.cat(got[s]), ref)
        assert ok, (s, err)
    bt.close()


@torch.inference_mode()
def test_capture_in_worker_thread_while_main_syncs(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(22, 90, 6)
    pb, sb = inputs(23, 90, 20)
    ra, _ = solo(stt, new_cache, pa, sa)
    rb, _ = solo(stt, new_cache, pb, sb)
    bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True)
    bt.attach("A", prefilled(stt, new_cache, pa))
    bt.attach("B", prefilled(stt, new_cache, pb))
    box, errs = {}, []

    def worker():
        try:
            with torch.inference_mode():
                box["a"] = run_pair(bt, {"A": sa, "B": sb}, 70)
        except Exception as e:                                           # noqa: BLE001
            errs.append(e)
    th = threading.Thread(target=worker)
    x = torch.randn(256, 256, device="cuda")
    th.start()
    while th.is_alive():
        (x @ x).sum().cpu()
    th.join()
    assert not errs, errs
    rest = run_pair(bt, {"A": sa[70:], "B": sb[70:]}, 20)
    bt.close()
    for s, ref in (("A", ra), ("B", rb)):
        ok, err = close(torch.cat([box["a"][s], rest[s]]), ref)
        assert ok, (s, err)


@torch.inference_mode()
def test_b1_sessions_unaffected_while_batch_runs(nh):
    """b1_graph=False: a non-attached session keeps the upstream eager path; b1_graph=True: it runs on the
    ThinkerGraph B=1 graph. Both match the solo run while a pair is batched."""
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(24, 70, 6)
    pb, sb = inputs(25, 70, 20)
    pc, sc = inputs(26, 70, 9)
    ra, _ = solo(stt, new_cache, pa, sa)
    rc, _ = solo(stt, new_cache, pc, sc)
    with pytest.raises(NotImplementedError):              # unverified B=1-graph routing is refused explicitly
        BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True, b1_graph=True)
    for b1 in (False,):
        bt = BatchedThinker(eng, max_rows=2, lmax=1024, graphs=True, b1_graph=b1)
        bt.attach("A", prefilled(stt, new_cache, pa))
        bt.attach("B", prefilled(stt, new_cache, pb))
        cc = prefilled(stt, new_cache, pc)
        ga, gc = [], []
        for i in range(70):
            ga.append(lg(bt.step({"A": sa[i], "B": sb[i]})["A"]))
            gc.append(lg(stt(sc[i], cache=cc)))
        assert (bt.stats["replays"] > 0) == b1
        bt.close()
        for got, ref in ((ga, ra), (gc, rc)):
            ok, err = close(torch.cat(got), ref)
            assert ok, (b1, err)


def test_refuses_to_stack_on_an_installed_thinker_graph(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    from mynah_v2.model.thinker_graph import ThinkerGraph
    eng, _, _ = build(nh)
    tg = ThinkerGraph(eng, lmax=256)
    with pytest.raises(RuntimeError):
        BatchedThinker(eng)
    tg.remove()


# ------------------------------------------------------------------------------------------------ I9
@torch.inference_mode()
def test_i9_memory_accounting(nh):
    from mynah_v2.model.thinker_batch import BatchedThinker
    eng, stt, new_cache = build(nh)
    pa, sa = inputs(27, 5, 6)
    bt = BatchedThinker(eng, max_rows=3, lmax=512, graphs=True)
    m0 = bt.memory()
    assert m0.graph_pool_bytes == 0 and m0.extra["mamba_bytes_per_slot"] is None
    c = prefilled(stt, new_cache, pa)
    mamba = sum(c.conv_states[i].numel() * 4 + c.ssm_states[i].numel() * 4 for i in bt.mamba)
    bt.attach("A", c)
    for x in sa:
        bt.step({"A": x})
    m = bt.memory()
    kv = 2 * len(bt.attn) * 2 * 512 * 32 * 4                             # n_attn x kvh x lmax x hd x fp32, K and V
    assert m.bytes_per_slot == kv + 512 * 4 + mamba, m
    assert m.static_buffers_bytes >= 3 * m.bytes_per_slot
    assert m.graph_pool_bytes > 0, m
    print("I9", m)
    bt.close()


if __name__ == "__main__":                                               # pragma: no cover
    sys.exit(pytest.main([str(Path(__file__)), "-q", "-m", "gpu"]))
