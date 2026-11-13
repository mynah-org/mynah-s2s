"""Batched EarTTS step (model/eartts_batch.py) on a CPU fake with the upstream call structure (CFG torch.cat doubling,
_prepare_conditioning, backbone with past_key_values.update + position_ids + mask dict, deterministic code head), and
the rendezvous "eartts" stage (serving order perception → Thinker → EarTTS → codec, FC / non-session threads pass
through). Reference = the plain upstream eager step on a concatenating cache per session."""

import importlib
import math
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mynah_v2.model import eartts_batch as EB
from mynah_v2.model import eartts_rng as ER
from mynah_v2.model.capture_guard import GUARD
from mynah_v2.model.eartts_batch import BatchedEarTTS, BatchKV
from mynah_v2.model.eartts_graph import FullStepGraph, StaticKV

sys.path.insert(0, str(Path(__file__).parent))
RZ = importlib.import_module("test_rendezvous")
fake_nemo = RZ.fake_nemo                                  # autouse fixture (nemo stand-ins)

L_, H_, D_, HD, C_, V_, NTOK = 2, 2, 4, 8, 3, 5, 20


# ------------------------------------------------------------------------------------------------ fake EarTTS
class RefCache:
    def __init__(self):
        self.key_cache, self.value_cache = [], []

    def update(self, k, v, li, cache_kwargs=None):
        if len(self.key_cache) <= li:
            self.key_cache.append(k)
            self.value_cache.append(v)
        else:
            self.key_cache[li] = torch.cat([self.key_cache[li], k], 2)
            self.value_cache[li] = torch.cat([self.value_cache[li], v], 2)
        return self.key_cache[li], self.value_cache[li]

    def get_seq_length(self, layer_idx=0):
        return 0 if not self.key_cache else self.key_cache[0].shape[2]


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=L_, num_key_value_heads=H_, head_dim=D_,
                                      _attn_implementation="eager", sliding_window=None)
        self.q = torch.nn.ModuleList(torch.nn.Linear(HD, H_ * D_) for _ in range(L_))
        self.k = torch.nn.ModuleList(torch.nn.Linear(HD, H_ * D_) for _ in range(L_))
        self.v = torch.nn.ModuleList(torch.nn.Linear(HD, H_ * D_) for _ in range(L_))
        self.o = torch.nn.ModuleList(torch.nn.Linear(H_ * D_, HD) for _ in range(L_))
        self.register_buffer("inv_freq", torch.arange(1, HD + 1, dtype=torch.float64) * 0.1, persistent=False)

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, cache_position=None,
                past_key_values=None, use_cache=True):
        x = inputs_embeds
        b = x.shape[0]
        if position_ids is None:
            position_ids = torch.full((b, 1), past_key_values.get_seq_length(), dtype=torch.long)
        x = x + torch.sin(position_ids[..., None].double() * self.inv_freq).to(x.dtype)   # RoPE-like, FP32+ math
        sh = lambda t: t.view(b, 1, H_, D_).transpose(1, 2)   # noqa: E731
        for li in range(L_):
            q, k, v = sh(self.q[li](x)), sh(self.k[li](x)), sh(self.v[li](x))
            K, Vv = past_key_values.update(k, v, li)
            s = q @ K.transpose(-1, -2) / math.sqrt(D_)
            if isinstance(attention_mask, dict):
                s = s + attention_mask["full_attention"]
            a = torch.softmax(s, -1) @ Vv
            x = x + self.o[li](a.transpose(1, 2).reshape(b, 1, H_ * D_))
        return SimpleNamespace(last_hidden_state=x, past_key_values=past_key_values)


class TTS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.hidden_size = HD
        self.embed_context = torch.nn.Linear(HD, HD)
        self.emb_sub = torch.nn.Embedding(NTOK, HD)
        self.null_emb = torch.nn.Parameter(torch.randn(HD))
        self.head = torch.nn.Linear(HD, C_ * V_)

    def embed_subword(self, ids, mask):
        return self.emb_sub(ids)

    def _prepare_conditioning(self, context_hidden_state, subword_ids, subword_mask, uncond_dec_flag):
        cond = torch.zeros((1, 1, self.hidden_size))
        cond = cond + self.embed_context(context_hidden_state)
        if subword_mask is not None and bool(subword_mask.any()):
            cond = cond + self.embed_subword(subword_ids, subword_mask)
        return torch.where(uncond_dec_flag, self.null_emb, cond)


class Duplex:
    def __init__(self, dtype=torch.float64):
        self.tts_model = TTS().to(dtype)
        self.embed_tok = torch.nn.Embedding(NTOK, HD).to(dtype)
        self.embed_code = torch.nn.Embedding(V_, HD).to(dtype)
        self.noise = False                              # True: stochastic sampling through the EarTTS noise hooks
        self.last_h = None

    def infer_codes_one_step(self, current_subword_id, prev_subword_id, current_subword_mask, prev_audio_tokens,
                             past_key_values, guidance_enabled=True, generation_config=None, ignore_eos_flag_stop=True,
                             request_id=None):
        m = self.tts_model
        ctx = self.embed_tok(prev_subword_id)
        x = self.embed_code(prev_audio_tokens).sum(-2)
        flag = torch.zeros(x.shape[0], 1, 1, dtype=torch.bool)
        ids, mask = current_subword_id, current_subword_mask
        if guidance_enabled:
            x, ctx, ids, mask = (torch.cat([t] * 2, 0) for t in (x, ctx, ids, mask))
            flag = torch.cat([flag, torch.ones_like(flag)], 0)
        cond = m._prepare_conditioning(ctx, ids, mask, flag)
        out = m.backbone(inputs_embeds=x + cond, attention_mask=None, position_ids=None,
                         past_key_values=past_key_values, use_cache=True)
        h = out.last_hidden_state
        hc, hu = h.chunk(2) if guidance_enabled else (h, h)
        self.last_h = hc.detach().float().clone()
        logits = m.head(hc + 1.5 * (hc - hu)).view(hc.shape[0], 1, C_, V_)
        if self.noise:                                  # Gumbel-max + Gaussian jitter, like MoG infer/generate_step
            logits = logits + ER.gumbel_like(logits) + 0.3 * ER.randn_like(logits)
        return logits.argmax(-1), out.past_key_values


@pytest.fixture
def tts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    d = Duplex()
    with torch.no_grad():                                # the wrapper runs the EarTTS without grad
        yield d
    torch.set_default_dtype(torch.float32)


def call(g, cur, prev, mask, codes):
    return dict(current_subword_id=torch.tensor([[cur]]), prev_subword_id=torch.tensor([[prev]]),
                current_subword_mask=torch.tensor([[mask]]), prev_audio_tokens=codes, guidance_enabled=True,
                generation_config={"num_iter": 3}, ignore_eos_flag_stop=True)


class Sess:
    """A session driven by the reference (upstream eager on a concatenating cache) and, after `fork`, by a StaticKV
    in a pooled slot (same history)."""

    def __init__(self, duplex, seed, prefill):
        self.d, self.g = duplex, torch.Generator().manual_seed(seed)
        self.ref, self.codes, self.prev = RefCache(), torch.zeros(1, 1, C_, dtype=torch.long), 1
        for _ in range(prefill):
            self.step_ref(self.next_in())

    def next_in(self):
        cur = int(torch.randint(2, NTOK, (1,), generator=self.g))
        mask = bool(torch.rand(1, generator=self.g) < 0.6)
        kw = call(self.g, cur, self.prev, mask, self.codes)
        self.prev = cur
        return kw

    def step_ref(self, kw):
        codes, self.ref = self.d.infer_codes_one_step(**kw, past_key_values=self.ref)   # (a cast may return a new one)
        self.codes = codes
        return codes

    def fork(self, fsg, i):
        dyn = RefCache()
        dyn.key_cache = [k.clone() for k in self.ref.key_cache]
        dyn.value_cache = [v.clone() for v in self.ref.value_cache]
        self.kv = StaticKV.from_dynamic(dyn, fsg.slots[i], i, fsg.lmax)
        fsg._kv_of_slot[i] = weakref.ref(self.kv)


def make(tts, slots=3, lmax=128):
    fsg = FullStepGraph(tts, lmax=lmax, slots=slots)   # (reference steps call the upstream method directly)
    be =BatchedEarTTS(fsg, graphs=False)
    return fsg, be


def test_pooled_slot_views_match_upstream_cfg_row_order(tts):
    fsg, be = make(tts)
    assert fsg.pool_k.shape == (L_, 2, 3, H_, 128, D_)
    s1 = fsg.slots[1]
    s1.k[0, 1, 0, 5] = 7.0                             # session 1, uncond row, head 0, position 5
    flat = fsg.pool_k.view(L_, 6, H_, 128, D_)
    assert flat[0, 3 + 1, 0, 5].eq(7.0).all()          # uncond rows follow all cond rows (torch.cat([x] * 2))
    assert s1.mask.shape == (2, 1, 1, 128)


def test_batched_step_matches_per_session_reference_across_buckets_and_masks(tts):
    fsg, be = make(tts, slots=3, lmax=128)
    A, B = Sess(tts, 1, 10), Sess(tts, 2, 50)          # B crosses the 64 bucket while A stays in it
    A.fork(fsg, 0)
    B.fork(fsg, 2)                                     # slot 1 stays free (dummy row)
    for _ in range(25):
        ka, kb = A.next_in(), B.next_in()
        ra, rb = A.step_ref(ka), B.step_ref(kb)
        (ca, kva), (cb, kvb) = be.run([dict(ka, past_key_values=A.kv), dict(kb, past_key_values=B.kv)])
        assert kva is A.kv and kvb is B.kv
        assert torch.equal(ca, ra) and torch.equal(cb, rb)
    assert A.kv.t == 35 and B.kv.t == 75
    assert be.stats["steps_by_bucket"] == {"64": 14, "128": 11} and be.stats["rows_hist"] == {2: 25}
    for s in (A, B):                                   # static K/V valid prefix == reference cache
        for li in range(L_):
            assert torch.allclose(s.kv.slot.k[li, :, :, :s.kv.t], s.ref.key_cache[li], atol=1e-12)


def test_idle_live_session_is_not_disturbed_and_continues_alone(tts):
    fsg, be = make(tts, slots=3)
    A, B, Cs = Sess(tts, 1, 10), Sess(tts, 2, 20), Sess(tts, 3, 30)
    for i, s in enumerate((A, B, Cs)):
        s.fork(fsg, i)
    for _ in range(6):                                  # C idle: its row is a dummy at its next position
        ka, kb = A.next_in(), B.next_in()
        A.step_ref(ka), B.step_ref(kb)
        be.run([dict(ka, past_key_values=A.kv), dict(kb, past_key_values=B.kv)])
    for _ in range(4):                                  # C resumes (alone and batched): still == its reference
        kc, ka = Cs.next_in(), A.next_in()
        rc, ra = Cs.step_ref(kc), A.step_ref(ka)
        (cc, _), (ca, _) = be.run([dict(kc, past_key_values=Cs.kv), dict(ka, past_key_values=A.kv)])
        assert torch.equal(cc, rc) and torch.equal(ca, ra)
    kc = Cs.next_in()
    assert torch.equal(be.run([dict(kc, past_key_values=Cs.kv)])[0][0], Cs.step_ref(kc))


def test_first_step_dynamic_cache_is_converted_and_joins_the_batch(tts, monkeypatch):
    fsg, be = make(tts, slots=3)
    monkeypatch.setattr(fsg, "_on_device", lambda x: True)
    A, B = Sess(tts, 1, 10), Sess(tts, 2, 30)
    kws = []
    for s in (A, B):                                   # first EarTTS call of each session: a DynamicCache
        dyn = RefCache()
        dyn.key_cache = [k.clone() for k in s.ref.key_cache]
        dyn.value_cache = [v.clone() for v in s.ref.value_cache]
        k = s.next_in()
        kws.append((s, k, dict(k, past_key_values=dyn)))
    for s, _, d in kws:
        assert be.group_key(d) is None and fsg.convert_for_batch(d) and be.group_key(d) is not None
        s.kv = d["past_key_values"]
    outs = be.run([d for _, _, d in kws])
    for (s, k, _), (c, kv) in zip(kws, outs):
        assert kv is s.kv and torch.equal(c, s.step_ref(k))
    assert fsg.stats["converted"] == 2 and {A.kv.idx, B.kv.idx} == {0, 1}


def test_group_key_rejects_what_the_batched_step_cannot_run(tts):
    fsg, be = make(tts)
    A = Sess(tts, 1, 5)
    A.fork(fsg, 0)
    k = A.next_in()
    assert be.group_key(dict(k, past_key_values=A.kv)) is not None
    assert be.group_key(dict(k, past_key_values=A.ref)) is None                 # DynamicCache: first step, B=1
    assert be.group_key(dict(k, past_key_values=A.kv, request_id="x")) is None
    assert be.group_key(dict(k, past_key_values=A.kv, guidance_enabled=False)) is None
    A.kv.t = fsg.lmax
    assert be.group_key(dict(k, past_key_values=A.kv)) is None                  # past lmax: per-session path
    assert EB.step_kwargs((1, 2, 3, 4, 5), {"guidance_enabled": True})["past_key_values"] == 5
    assert EB.step_kwargs((1,), {"bogus": 1}) is None


def test_batched_capture_goes_through_the_guard(tts, monkeypatch):
    fsg, be = make(tts)
    be.use_graphs = True
    monkeypatch.setattr(be, "_graphable", lambda x: True)
    A, B = Sess(tts, 1, 10), Sess(tts, 2, 12)
    A.fork(fsg, 0)
    B.fork(fsg, 1)
    GUARD.configure("barrier")
    saved = list(GUARD._providers)
    GUARD._providers = []
    GUARD.add_provider(busy := (lambda me: "fc_phase:x"))
    try:
        monkeypatch.setattr(be, "_capture", lambda *a: (_ for _ in ()).throw(AssertionError("captured")))
        ka, kb = A.next_in(), B.next_in()
        ra, rb = A.step_ref(ka), B.step_ref(kb)
        (ca, _), (cb, _) = be.run([dict(ka, past_key_values=A.kv), dict(kb, past_key_values=B.kv)])
        assert torch.equal(ca, ra) and torch.equal(cb, rb)
        assert be.stats["capture_deferred"] == 1 and be.stats["eager"] == 1
    finally:
        GUARD.remove_provider(busy)
        GUARD._providers = saved
        GUARD.configure("lazy")
        GUARD.reset_stats()


# ------------------------------------------------------------------------------------------------ rendezvous stage
class FakeKV:
    def __init__(self):
        self.s = []


TTS_LOG: list = []


def fake_tts_step(current_subword_id, prev_subword_id, current_subword_mask, prev_audio_tokens, past_key_values,
                  **kw):
    TTS_LOG.append(("b1", threading.current_thread().name))
    kv = past_key_values
    kv.s.append(float(current_subword_id.reshape(-1)[0]))
    return torch.tensor([[[int(sum(kv.s) * 3 + len(kv.s)) % 7]]]), kv


class FakeEBatch:
    def __init__(self, fsg, graphs=True):
        self.calls = []

    def precapture(self):
        return {}

    def group_key(self, d):
        return ("k",) if isinstance(d.get("past_key_values"), FakeKV) else None

    def run(self, kws):
        self.calls.append(len(kws))
        TTS_LOG.append(("batched", len(kws)))
        out = []
        for d in kws:
            kv = d["past_key_values"]
            kv.s.append(float(d["current_subword_id"].reshape(-1)[0]))
            out.append((torch.tensor([[[int(sum(kv.s) * 3 + len(kv.s)) % 7]]]), kv))
        return out


class TPipeline(RZ.Pipeline):
    def generate_step(self, frames):
        f, w = frames[0], self.s2s_model
        pre = f.is_first and f.samples.numel() == 0
        super().generate_step(frames)
        if pre:
            ctx, _ = self.context_manager.get_context([f.stream_id])
            ctx.tts = FakeKV()
            return
        ctx, _ = self.context_manager.get_context([f.stream_id])
        codes = []
        for k in range(2):                                # after the Thinker frames: one EarTTS step per frame
            c, ctx.tts = w.model.tts_model.infer_codes_one_step(
                torch.tensor([[ctx.frame_idx * 10 + k]]), torch.tensor([[1]]), torch.tensor([[True]]),
                torch.zeros(1, 1, 1, dtype=torch.long), ctx.tts, guidance_enabled=True, ignore_eos_flag_stop=True)
            codes.append(int(c))
        self.get_or_create_state(f.stream_id).text += f" A{codes}"


def make_engine():
    w = RZ.Wrapper()
    w.model.tts_model = SimpleNamespace(infer_codes_one_step=fake_tts_step)
    p = TPipeline(w)
    eng = object.__new__(RZ.ReferenceEngine)
    eng.pipeline, eng.wrapper, eng.tokenizer = p, w, None
    eng.session_tools = RZ.SessionToolBinding(p.tool_registry)
    eng.fc_still_running, eng.fc_records, eng._fc_lock = 0, [], threading.Lock()
    eng._instrument_fc()
    eng.unknown_tool_handler = eng.multi_call_handler = None
    eng.unknown_tool_events, eng.multi_call_events = [], []
    eng._wrap_executor()
    eng.chunk_samples, eng.stream_id, eng._started = RZ.T.CHUNK, 0, False
    eng._pos, eng._last_frame_idx, eng._rec_cursor, eng.llm_cache_info = {"text": 0, "asr": 0, "func": 0}, 0, 0, None
    eng._v2_full_graph = SimpleNamespace(n_slots=8)
    return eng


def solo(sid, n):
    eng = make_engine()
    torch.manual_seed(RZ.T.SESS[sid][2])
    eng.open(RZ.T.SESS[sid][0])
    res = [RZ.obs(eng.step(RZ.T.pcm(sid, k))) for k in range(n)]
    eng.close()
    return res


def test_rendezvous_eartts_stage_batches_after_the_thinker_and_matches_solo(monkeypatch):
    monkeypatch.setattr(EB, "BatchedEarTTS", FakeEBatch)
    n = 4
    sa, sb = solo("A", n), solo("B", n)
    eng = make_engine()
    m = RZ.MultiSessionEngine(eng, cuda_rng=False)
    be = RZ.FakeBackend(2)
    rz = RZ.RendezvousEngine(m, be, batch_eartts=True)
    RZ.open_(rz, "A")
    RZ.open_(rz, "B")
    TTS_LOG.clear()
    order = []
    real_step = be.step
    be.step = lambda frames: (order.append("thinker"), real_step(frames))[1]
    real_run = rz.ebatch.run
    rz.ebatch.run = lambda kws: (order.append("eartts"), real_run(kws))[1]
    pa, pb = [], []
    for k in range(n):
        r = rz.step_jobs({"A": RZ.T.pcm("A", k), "B": RZ.T.pcm("B", k)})
        pa.append(RZ.obs(r["A"]))
        pb.append(RZ.obs(r["B"]))
    assert pa == sa and pb == sb                       # same per-session EarTTS outputs and state
    assert rz.stats["eartts"]["batch_hist"] == {2: 2 * n} and rz.stats["eartts"]["b1"] == 0
    assert order == ["thinker", "thinker", "eartts", "eartts"] * n   # fake chunk: 2 Thinker frames, then 2 EarTTS steps
    # a call from a non-session thread passes through (FC threads / reminders)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", eng.wrapper.model.tts_model.infer_codes_one_step(
        torch.tensor([[3]]), torch.tensor([[1]]), torch.tensor([[True]]), torch.zeros(1, 1, 1, dtype=torch.long),
        FakeKV())))
    t.start()
    t.join(5)
    assert rz.stats["eartts"]["passthrough"] == 1 and box["r"][0].shape == (1, 1, 1)
    rz.close_all()
    assert "infer_codes_one_step" in vars(eng.wrapper.model.tts_model)   # SimpleNamespace attr restored
    assert eng.wrapper.model.tts_model.infer_codes_one_step is fake_tts_step


# ------------------------------------------------------------------------------------------------ --eartts-dtype bf16
def test_bf16_backbone_converts_params_in_place_keeps_buffers_and_slots_follow(tts):
    fsg = FullStepGraph(tts, lmax=128, slots=3)
    bb = tts.tts_model.backbone
    ids = [id(p) for p in bb.parameters()]
    info = fsg.set_backbone_dtype(torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in bb.parameters()) and [id(p) for p in bb.parameters()] == ids
    assert bb.inv_freq.dtype == torch.float64                            # RoPE frequencies keep full precision
    assert tts.tts_model.head.weight.dtype == torch.float64              # heads / embeddings untouched
    assert info["from"] == "torch.float64" and info["to"] == "torch.bfloat16" and "B8_weights_plus_slots_GiB_after" in info
    be = BatchedEarTTS(fsg, graphs=False)
    assert fsg.pool_k.dtype == fsg.pool_mask.dtype == torch.bfloat16 and fsg.slots[0].k.dtype == torch.bfloat16
    be.f.slots[0].busy = True
    with pytest.raises(RuntimeError):
        fsg.set_backbone_dtype(torch.float32)                            # a slot in use: refused
    be.f.slots[0].busy = False


def test_bf16_never_casts_the_callers_cache_in_place_and_stays_finite(tts):
    fsg = FullStepGraph(tts, lmax=128, slots=3)
    fsg.set_backbone_dtype(torch.bfloat16)
    A = Sess(tts, 1, 0)
    shared = RefCache()                                                  # an FP64 shared initial/prompt cache
    shared.key_cache = [torch.randn(2, H_, 6, D_) for _ in range(L_)]
    shared.value_cache = [torch.randn(2, H_, 6, D_) for _ in range(L_)]
    before = [k.clone() for k in shared.key_cache]
    codes, out = tts.infer_codes_one_step(**A.next_in(), past_key_values=shared)
    assert out is not shared and out.key_cache[0].dtype == torch.bfloat16 and out.key_cache[0].shape[2] == 7
    assert all(k.dtype == torch.float64 and torch.equal(k, b) for k, b in zip(shared.key_cache, before))
    assert shared.key_cache[0].shape[2] == 6                             # source untouched (shape and values)
    be = BatchedEarTTS(fsg, graphs=False)
    B, Cs = Sess(tts, 2, 12), Sess(tts, 3, 40)                          # their reference caches: BF16 backbone
    B.fork(fsg, 0)
    Cs.fork(fsg, 1)
    agree = total = 0
    for _ in range(20):
        kb, kc = B.next_in(), Cs.next_in()
        rb, rc = B.step_ref(kb), Cs.step_ref(kc)
        (cb, _), (cc, _) = be.run([dict(kb, past_key_values=B.kv), dict(kc, past_key_values=Cs.kv)])
        for got, ref in ((cb, rb), (cc, rc)):
            assert got.dtype == torch.long and int(got.min()) >= 0 and int(got.max()) < V_
            agree += int((got == ref).sum())
            total += got.numel()
        B.codes, Cs.codes = rb, rc
    assert agree / total > 0.8, agree / total                            # BF16 batched ~ BF16 per-session


# ------------------------------------------------------------------------------------------------ per-session RNG
@pytest.fixture
def rng_on(monkeypatch):
    monkeypatch.setitem(ER.STATE, "enabled", True)
    monkeypatch.setitem(ER.STATE, "orig_gumbel", lambda t, eps=1e-8: -torch.log(-torch.log(torch.rand_like(t) + eps) + eps))
    yield
    ER.CTX.buf, ER.CTX.seed = None, None


def test_counter_noise_is_reproducible_per_row_and_advances_per_step(rng_on):
    x = torch.zeros(3, 1, 8)
    def draw(buf):
        y = x[:buf.shape[0]]
        with ER.rows(buf):
            return ER.randn_like(y), ER.gumbel_like(y)
    b = torch.tensor([[11, 0], [22, 0], [11, 5]])
    n1, g1 = draw(b)
    n2, g2 = draw(b.clone())
    assert torch.equal(n1, n2) and torch.equal(g1, g2)                # same (seed, step) → same noise
    assert not torch.equal(n1[0], n1[1]) and not torch.equal(n1[0], n1[2])   # other seed / other step differ
    alone, _ = draw(torch.tensor([[11, 5]]))
    assert torch.equal(alone[0], n1[2])                                 # a row's noise ignores neighbours/position
    nxt, _ = draw(torch.tensor([[11, 1]]))
    assert not torch.equal(nxt[0], n1[0])                               # step advanced → new samples
    big = torch.zeros(1, 1, 200000)
    with ER.rows(torch.tensor([[7, 3]])):
        z, u = ER.randn_like(big), ER.uniform(big, 99)
    assert abs(float(z.mean())) < 0.02 and abs(float(z.std()) - 1) < 0.02 and 0 < float(u.min()) and float(u.max()) < 1
    seen = set()
    for step in range(50):                                              # no repeats across 50 consecutive steps
        with ER.rows(torch.tensor([[7, step]])):
            seen.add(tuple(ER.uniform(torch.zeros(1, 4), 2).flatten().tolist()))
    assert len(seen) == 50


def _snapshot(fsg, sess):
    return ([t.clone() for t in (fsg.pool_k, fsg.pool_v, fsg.pool_mask)],
            [(s.kv.t, getattr(s.kv, "_v2_rng_step", 0)) for s in sess])


def _restore(fsg, sess, snap):
    for t, c in zip((fsg.pool_k, fsg.pool_v, fsg.pool_mask), snap[0]):
        t.copy_(c)
    for s, (t, st) in zip(sess, snap[1]):
        s.kv.t, s.kv._v2_rng_step = t, st


def _inputs(seed, n):
    g = torch.Generator().manual_seed(seed)
    return [call(g, int(torch.randint(2, NTOK, (1,), generator=g)), int(torch.randint(2, NTOK, (1,), generator=g)),
                 bool(torch.rand(1, generator=g) < 0.6), torch.randint(0, V_, (1, 1, C_), generator=g)) for _ in range(n)]


def _b1_run(fsg, tts, s, ins, monkeypatch):
    monkeypatch.setattr(fsg, "_on_device", lambda x: True)
    monkeypatch.setattr(GUARD, "may_capture", lambda *a: False)        # per-session static path, eager
    codes, hs = [], []
    for k in ins:
        c, _ = fsg.step(**k, past_key_values=s.kv)
        codes.append(c)
        hs.append(tts.last_h[0])
    return codes, hs


def _batch_run(be, tts, order, ins_of):
    codes = {s: [] for s in order}
    hs = {s: [] for s in order}
    for j in range(len(ins_of[order[0]])):
        outs = be.run([dict(ins_of[s][j], past_key_values=s.kv) for s in order])
        for s, (c, kv) in zip(order, outs):
            codes[s].append(c)
            hs[s].append(tts.last_h[kv.idx])
    return codes, hs


def _setup(tts, seeds_slots, mode="fp32"):
    fsg = FullStepGraph(tts, lmax=128, slots=4)
    fsg.set_mode(mode)
    be = BatchedEarTTS(fsg, graphs=False)
    sess = []
    for seed, slot, pre in seeds_slots:
        s = Sess(tts, seed, pre)
        ER.CTX.seed = seed                                              # the session's EarTTS seed (rendezvous open)
        s.fork(fsg, slot)
        s.kv._v2_rng_seed, s.kv._v2_rng_step = seed, 0
        sess.append(s)
    ER.CTX.seed = None
    return fsg, be, sess


def test_same_session_same_seed_same_codes_b1_vs_batched_any_row_any_neighbour(tts, rng_on, monkeypatch):
    tts.noise = True
    fsg, be, (A, B, C) = _setup(tts, [(101, 2, 10), (202, 0, 40), (303, 1, 70)])
    ins = {A: _inputs(1, 12), B: _inputs(2, 12), C: _inputs(3, 12)}
    snap = _snapshot(fsg, (A, B, C))
    ref, _ = _b1_run(fsg, tts, A, ins[A], monkeypatch)                  # A alone, per-session path
    _restore(fsg, (A, B, C), snap)
    with_b, _ = _batch_run(be, tts, [B, A], ins)                        # A batched with B (A not first)
    _restore(fsg, (A, B, C), snap)
    with_c, _ = _batch_run(be, tts, [A, C], ins)                        # neighbour swapped
    for got in (with_b[A], with_c[A]):
        assert all(torch.equal(x, y) for x, y in zip(got, ref))
    assert not all(torch.equal(x, y) for x, y in zip(with_b[B], ref))  # sanity: other sessions differ
    _restore(fsg, (A, B, C), snap)
    A.kv._v2_rng_seed = 999                                             # another seed → other samples
    other, _ = _batch_run(be, tts, [A, C], ins)
    assert not all(torch.equal(x, y) for x, y in zip(other[A], ref))


@pytest.mark.parametrize("bf16", [False, True])
def test_row_isolation_deterministic_sampling_fp32_and_bf16(bf16, monkeypatch):
    """Row/state isolation only (sampling deterministic: argmax): batched hidden states == per-session ones within
    tolerance, whatever the row order and the neighbours."""
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    torch.manual_seed(0)
    tts = Duplex(torch.float32)
    with torch.no_grad():
        fsg, be, (A, B, C) = _setup(tts, [(1, 1, 10), (2, 0, 40), (3, 2, 70)], "bf16" if bf16 else "fp32")
        assert fsg.pool_k.dtype == (torch.bfloat16 if bf16 else torch.float32)
        ins = {A: _inputs(4, 10), B: _inputs(5, 10), C: _inputs(6, 10)}
        snap = _snapshot(fsg, (A, B, C))
        refs = {}
        for s in (A, B, C):
            _restore(fsg, (A, B, C), snap)
            refs[s] = _b1_run(fsg, tts, s, ins[s], monkeypatch)[1]
        tol = 5e-2 if bf16 else 1e-4
        for order in ([A, B, C], [C, A, B], [B, C]):
            _restore(fsg, (A, B, C), snap)
            _, hs = _batch_run(be, tts, order, ins)
            for s in order:
                err = max(float((x - y).abs().max()) for x, y in zip(hs[s], refs[s]))
                assert err < tol, (order.index(s), err)


@pytest.mark.parametrize("mode,kvdt,mdt", [("bf16w", torch.float32, torch.bfloat16),
                                          ("bf16kv", torch.bfloat16, torch.float32)])
def test_dtype_split_modes_run_and_stay_close(mode, kvdt, mdt, monkeypatch):
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    torch.manual_seed(0)
    tts = Duplex(torch.float32)
    with torch.no_grad():
        fsg, be, (A, B) = _setup(tts, [(1, 0, 10), (2, 1, 40)], mode)
        assert fsg.pool_k.dtype == kvdt and fsg.pool_mask.dtype == mdt
        ins = {A: _inputs(4, 8), B: _inputs(5, 8)}
        snap = _snapshot(fsg, (A, B))
        ref = _b1_run(fsg, tts, A, ins[A], monkeypatch)[1]
        _restore(fsg, (A, B), snap)
        _, hs = _batch_run(be, tts, [B, A], ins)
        assert all(torch.isfinite(x).all() for x in hs[A])
        assert max(float((x - y).abs().max()) for x, y in zip(hs[A], ref)) < 5e-2



def test_warmup_and_capture_forwards_draw_the_same_noise(tts, rng_on):
    """A capture runs the step twice in one rows() context (warm-up, then the recorded call): the draw counter must
    restart per forward, else the graph replays other noise than the eager step (found by the ER2 GPU smoke)."""
    tts.noise = True
    fsg, be, (A,) = _setup(tts, [(7, 0, 10)])
    k = _inputs(9, 1)[0]
    A.kv.prepare()
    outs = []
    with ER.rows(torch.tensor([[7, 3]])):
        for _ in range(2):
            fsg.mask_any = bool(k["current_subword_mask"])
            if fsg.mask_any:
                fsg._load_subword(int(k["current_subword_id"]), k["current_subword_id"], k["current_subword_mask"],
                                  True)
            outs.append(fsg.eager_step(*(k[a] for a in EB.ARGS), A.kv, guidance_enabled=True,
                                       generation_config=k["generation_config"], ignore_eos_flag_stop=True)[0])
    fsg.mask_any = None
    assert torch.equal(outs[0], outs[1])
