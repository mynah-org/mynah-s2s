"""Precision formats (simulated), selector grammar and the teacher-forced sensitivity probe on a toy backbone."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mynah_v2.model import precision as P


def test_formats_error_ordering_and_grids():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    err = {f: float((P.SIM_FORMATS[f](w) - w).pow(2).mean()) for f in P.SIM_FORMATS}
    assert err["int8"] < err["fp8"] < err["int4g128"]
    assert err["int4g32"] < err["int4g64"] < err["int4g128"]
    assert err["int4g64c"] <= err["int4g64"] * 1.0001
    assert err["nvfp4"] < err["int4g128"] * 2           # same bit budget class, block 16
    # nvfp4: every value = e2m1 grid point × its block scale
    q = P._nvfp4(w).reshape(64, 16, 16)
    s = q.abs().amax(-1, keepdim=True) / 6
    ratio = (q.abs() / s.clamp_min(1e-30))
    grid = P.E2M1
    assert torch.all((ratio[..., None] - grid).abs().amin(-1) < 1e-3)
    # int4 g128: at most 16 distinct values per group
    q4 = P._int4_group(w, 128).reshape(64, 2, 128)
    assert max(len(torch.unique(g)) for g in q4.reshape(-1, 128)) <= 16


class Mixer(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        if kind == "mamba":
            self.in_proj, self.out_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        elif kind == "mlp":
            self.up_proj, self.down_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        else:
            self.q_proj, self.o_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)


class Layer(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.block_type = kind
        self.mixer = Mixer(d, kind)

    def forward(self, x):
        return x + torch.tanh(sum(m(x) for m in self.mixer.children()))


class ToyStt(nn.Module):
    """forward(input_embeds, cache) -> logits; the cache is a stateful running sum mutated IN PLACE (like Mamba)."""

    def __init__(self, d=32, v=50):
        super().__init__()
        self.llm = nn.Module()
        self.llm.layers = nn.ModuleList([Layer(d, k) for k in ("mamba", "mlp", "mamba", "attention")])
        self.lm_head = nn.Linear(d, v, bias=False)
        self.function_head = nn.Linear(d, v, bias=False)

    def forward(self, input_embeds, cache=None, **kw):
        h = input_embeds
        if cache is not None:
            cache.state = cache.state + h.mean(1, keepdim=True)
            h = h + cache.state
        for layer in self.llm.layers:
            h = layer(h)
        return {"text_logits": self.lm_head(h), "function_logits": self.function_head(h), "cache": cache}


def test_selector_grammar():
    stt = ToyStt()
    lins = P.thinker_linears(stt.llm)
    assert len(P.select(lins, "all", 4)) == 8
    assert P.select(lins, "mamba.in_proj", 4) == {(0, "in_proj"), (2, "in_proj")}
    assert P.select(lins, "all,!mamba", 4) == {(1, "up_proj"), (1, "down_proj"), (3, "q_proj"), (3, "o_proj")}
    assert P.select(lins, "Q0", 4) == {(0, "in_proj"), (0, "out_proj")}
    assert P.select(lins, "L1-2", 4) == {(1, "up_proj"), (1, "down_proj"), (2, "in_proj"), (2, "out_proj")}
    with pytest.raises(ValueError):
        P.select(lins, "bogus", 4)
    assert P.select(lins, "none", 4) == set()
    hl = P.thinker_linears(stt.llm, stt)
    assert P.select(hl, "all", 4) == P.select(lins, "all", 4)          # heads never in "all"
    assert P.select(hl, "heads", 4) == {(-1, "lm_head"), (-1, "function_head")}
    assert P.select(hl, "head.function_head", 4) == {(-1, "function_head")}
    # with the heads in the list (as the probe builds it): quarters / ranges never pick a head and never raise
    # (D2b crashed on 'Q0' because a head linear fell through to the error branch)
    assert P.select(hl, "Q0", 4) == {(0, "in_proj"), (0, "out_proj")}
    assert P.select(hl, "L1-2", 4) == P.select(lins, "L1-2", 4)


def test_d2_preregistered_selectors_resolve():
    """Every selector of the D2 job must resolve on a head-including linear list before a GPU job is queued."""
    stt = ToyStt()
    hl = P.thinker_linears(stt.llm, stt)
    n = 4
    sel = {s: P.select(hl, s, n) for s in ("all,heads", "all", "all,heads", "heads", "Q0", "Q1", "Q2", "Q3",
                                           "all,!attention")}
    quarters = sel["Q0"] | sel["Q1"] | sel["Q2"] | sel["Q3"]
    assert quarters == sel["all"]                                     # the quarters partition the backbone
    assert sel["heads"] == {(-1, "lm_head"), (-1, "function_head")}
    assert sel["all,heads"] == sel["all"] | sel["heads"]
    assert not any(li == -1 for li, _ in sel["all,!attention"])


def test_probe_teacher_forced_golden_unchanged_and_isolated_caches():
    torch.manual_seed(0)
    stt = ToyStt()
    eng = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=stt)))
    xs = [torch.randn(1, 1, 32) for _ in range(6)]
    ref_cache = SimpleNamespace(state=torch.zeros(1, 1, 32))
    ref = [stt(x, cache=ref_cache)["text_logits"] for x in xs]

    probe = P.SensitivityProbe(eng, ["fine=int8@all", "coarse=int4g32@mamba", "none=int8@attention.k_proj",
                                     "ctrl=bf16@all,heads", "heads=int4g32@heads"])
    cache = SimpleNamespace(state=torch.zeros(1, 1, 32))
    outs = [stt(x, cache=cache)["text_logits"] for x in xs]
    probe.remove()
    for a, b in zip(ref, outs):                                      # golden trajectory bit-identical
        assert torch.equal(a, b)
    assert torch.equal(cache.state, ref_cache.state)                 # golden cache advanced exactly once per call
    assert len(probe.records) == 6
    c = {k: [r["c"][k] for r in probe.records] for k in ("fine", "coarse", "none")}
    assert all(r["text_kl"] == 0 and r["text_agree"] for r in c["none"])   # empty selection = golden
    ctrl = [r["c"]["ctrl"] for r in probe.records]                    # exact copy of every linear incl. heads
    assert all(r["text_kl"] < 1e-9 and r["text_agree"] and r["func_agree"] for r in ctrl)
    assert sum(r["c"]["heads"]["text_kl"] for r in probe.records) > 0
    assert sum(r["text_kl"] for r in c["fine"]) < sum(r["text_kl"] for r in c["coarse"])
    # modules restored
    assert all(isinstance(m, nn.Linear) and m.weight.numel() for _, _, _, mixer in P.thinker_linears(stt.llm)
               for m in mixer.children())
    # each candidate kept its own cache, distinct from the golden one
    for cand in probe.cands:
        sc = cand.caches[id(cache)]
        assert sc is not cache


def test_probe_records_margins_and_gold_gap():
    torch.manual_seed(1)
    stt = ToyStt()
    eng = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=stt)))
    probe = P.SensitivityProbe(eng, ["none=int8@attention.k_proj", "coarse=int4g32@all,heads"])
    cache = SimpleNamespace(state=torch.zeros(1, 1, 32))
    for _ in range(5):
        stt(torch.randn(1, 1, 32), cache=cache)
    probe.remove()
    for r in probe.records:
        assert r["g_text_margin"] >= 0 and r["g_text_2nd"] != r["gold_text"]
        n = r["c"]["none"]                     # identical candidate: its gap for the golden token IS the golden margin
        assert n["text_top"] == r["gold_text"] and abs(n["text_gold_gap"] - r["g_text_margin"]) < 1e-4
        c = r["c"]["coarse"]
        assert (c["text_gold_gap"] > 0) == c["text_agree"] or c["text_gold_gap"] == 0


def test_decision_flip_classes():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from thinker_sensitivity import decision_flips, flip_class
    assert flip_class(12, 1) == "turn" and flip_class(2, 12) == "turn"
    assert flip_class(12, 500) == "timing" and flip_class(500, 12) == "timing"
    assert flip_class(500, 501) == "content"
    rec = lambda g, c, m: {"gold_text": g, "g_text_margin": m,
                           "c": {"w8": {"text_agree": g == c, "text_top": c, "text_gold_gap": -0.1, "text_kl": 0.01}}}
    dec = [rec(12, 12, 5.0), rec(12, 1, 0.2), rec(1, 1, 3.0), rec(500, 501, 0.5), rec(2, 2, 0.05)]
    d = decision_flips(dec, "w8")
    assert d["all_text_flips"] == 2 and d["flips_by_class"] == {"turn": 1, "content": 1}
    assert d["flip_list"][0]["step"] == 1 and d["flip_list"][0]["gold_margin"] == 0.2
    assert d["turn_steps"] == 2 and d["turn_gold_margin_min"] == 0.05
