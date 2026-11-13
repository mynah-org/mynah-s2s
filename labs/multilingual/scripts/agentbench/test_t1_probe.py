"""CPU tests for t1_probe.py and observe.py with fake objects (python3 test_t1_probe.py)."""
import copy
import json
import math
import tempfile

import torch

import observe
import t1_probe as P
from t1_probe import BOS, EOS, EOTC, PAD, SOTC

V = 64


class Tok:
    """Toy tokenizer: one id per character (ids 30..), with BOS prepended like some NeMo wrappers."""
    def text_to_ids(self, s):
        return [BOS] + [30 + (ord(c) % 30) for c in s]

    def ids_to_text(self, ids):
        return "".join(chr(i) for i in ids)


class FakeW:
    tokenizer = Tok()


class FakeThinker:
    """Live-path fake: a MUTABLE cache {"pos", "log"} advanced in place by every step (like the hybrid cache). Logits
    are a function of the frame position: text head -> rec[t]; function head -> PAD before s, SOTC at s, then the
    k-th token of `script` at FC frame s+k. `shallow=True` makes clone() return the same object (contamination)."""
    def __init__(self, rec, s, script, perturb=None, shallow=False, nocache_noise=None):
        self.rec, self.s, self.script, self.perturb = rec, s, script, perturb or {}
        self.shallow, self.nocache_noise = shallow, nocache_noise or {}
        self.w = FakeW()

    def encode(self, s):
        ids = self.w.tokenizer.text_to_ids(s)
        return ids[1:] if ids and ids[0] == BOS else ids

    def decode(self, ids):
        return self.w.tokenizer.ids_to_text(ids)

    def _logits(self, t):
        tl = torch.full((V,), -5.0)
        fl = torch.full((V,), -5.0)
        tl[self.perturb.get(("text", t), self.rec.get(t, PAD))] = 5.0
        if t < self.s:
            fl[PAD] = 5.0
        elif t == self.s:
            fl[self.perturb.get(("func", t), SOTC)] = 5.0
        else:
            k = t - self.s - 1
            fl[self.script[k] if k < len(self.script) else PAD] = 5.0
        return tl, fl

    def prefill(self):
        return {"pos": 0, "log": []}

    def step(self, cache, audio_row, agent_prev, func_prev):
        t = cache["pos"]
        cache["pos"] += 1
        cache["log"].append((agent_prev, func_prev))
        tl, fl = self._logits(t)
        return tl, fl, cache

    def clone(self, cache):
        return cache if self.shallow else copy.deepcopy(cache)

    def forward_full(self, audio, agent_prev, func_prev):
        rows = [self._logits(t) for t in range(len(agent_prev))]
        tl, fl = torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows])
        for t, tok in self.nocache_noise.items():
            tl[t] = -5.0
            tl[t, tok] = 5.0
        return tl, fl


def test_helpers():
    assert P.prev_shift([5, 6, 7]) == [PAD, 5, 6]
    assert P.end_frame(10.51) == 132 and P.end_frame(10.56) == 132 and P.end_frame(10.48) == 131
    hist = P.fixed_history({0: 12, 9: EOS, 19: BOS, 94: EOS, 120: 77}, 130)
    assert hist[94] == EOS and hist[120] == PAD and hist[19] == BOS
    p, r = P.renorm({"a": math.log(0.3), "b": math.log(0.1)})
    assert abs(p["a"] - 0.75) < 1e-9 and abs(r - 0.6) < 1e-9
    common, rest = P.name_candidates(lambda s: Tok().text_to_ids(s)[1:], ["ab", "ac"])
    assert rest == {"ab": [30 + ord("b") % 30, 30 + ord('"') % 30, 30 + ord(",") % 30],
                    "ac": [30 + ord("c") % 30, 30 + ord('"') % 30, 30 + ord(",") % 30]}
    try:  # with no closing token, "ab" would be a token prefix of "abc" -> not disjoint
        P.name_candidates(lambda s: [ord(c) for c in s.replace('",', "")], ["ab", "abc"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_trace_parsing():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for e in [{"ev": "session.opened"}, {"ev": "thinker.step", "model_step": 2, "steps": 2, "text_ids": [12, 2]},
                  {"ev": "thinker.step", "model_step": 4, "steps": 2, "text_ids": [1, 12]},
                  {"ev": "thinker.step", "model_step": 4, "steps": 0, "fc_active": True},
                  {"ev": "function.arguments.done", "model_step": 3, "func_ids": [20, 9, 21]},
                  {"ev": "session.opened"}, {"ev": "thinker.step", "model_step": 2, "steps": 2, "text_ids": [9, 9]}]:
            f.write(json.dumps(e) + "\n")
    s = P.trace_session(f.name, 0)
    assert P.frame_tokens(s) == {0: 12, 1: 2, 2: 1, 3: 12}
    assert P.fc_call(s) == (3, [20, 9, 21])
    assert P.frame_tokens(P.trace_session(f.name, 1)) == {0: 9, 1: 9}


def test_fc_inputs_alignment():
    audio = torch.arange(10).float()[:, None].repeat(1, 4)
    sil = torch.full((1, 4), -1.0)
    a, ag, fu = P.fc_inputs(audio, sil, [7] * 10, 5, [40, 41, EOTC])
    assert a.shape[0] == 9 and torch.equal(a[5], audio[5]) and torch.equal(a[6], sil[0])
    assert ag == [PAD] + [7] * 5 + [PAD] * 3             # frame s and FC frames carry PAD as agent token
    assert fu == [PAD] * 6 + [SOTC, 40, 41]               # input at s+1 is SOTC, then the call tokens


def _gate_fixture(perturb=None, **kw):
    s = 20
    rec = {t: PAD for t in range(s + 1)}
    rec.update({3: EOS, 6: BOS, 7: 50, 8: 51, 12: EOS})
    script = [40, 41, 42, EOTC]
    th = FakeThinker(rec, s, script, perturb, **kw)
    return th, rec, (s, [SOTC] + script)


def _gate(**kw):
    old = P.GATE_SOTC
    P.GATE_SOTC = 20
    try:
        th, rec, call = _gate_fixture(**kw)
        return P.gate(th, torch.zeros(40, 4), rec, call, torch.zeros(1, 4))
    finally:
        P.GATE_SOTC = old


def test_gate():
    g = _gate()
    assert g["pass"] and g["clone_isolation_bitwise"] and g["G2"]["mismatches"] == [], g
    g = _gate(perturb={("text", 6): PAD})                     # model misses a BOS -> critical
    assert not g["pass"] and g["G1"]["critical"] == [(6, BOS, PAD)]
    g = _gate(perturb={("text", 7): 99 % V})                  # one ordinary text token differs -> still FAIL
    assert not g["pass"] and not g["G1"]["critical"] and g["G1"]["n_mismatch"] == 1
    g = _gate(perturb={("func", 20): PAD})                    # FC entry missed -> FAIL
    assert not g["pass"] and not g["G1"]["func_sotc_at"]
    g = _gate(shallow=True)                                    # shared cache between branches -> contamination
    assert not g["pass"] and (g["G2"]["mismatches"] or not g["clone_isolation_bitwise"])
    g = _gate(nocache_noise={7: 33})                           # no-cache path differs: reported, not gating
    assert g["pass"] and g["numeric_info_nocache"]["text_argmax_diff_frames_0_s"] == [7]


def test_branch_isolation():
    th = FakeThinker({}, 5, [40, 41, EOTC])
    cache = th.prefill()
    _, cache, snaps = P.run_frames(th, cache, torch.zeros(6, 4), [PAD] * 6, [PAD] * 6, snap={5})
    snap = snaps[5]
    pos0 = snap["pos"]
    l1, a1, _, _ = P.branch_fc(th, snap, torch.zeros(1, 4), [40, 41, EOTC])
    P.branch_fc(th, snap, torch.zeros(1, 4), [9, 9, 9, 9, 9])
    l2, a2, _, _ = P.branch_fc(th, snap, torch.zeros(1, 4), [40, 41, EOTC])
    assert snap["pos"] == pos0 == 6 and a1 == a2 == [40, 41, EOTC] and l1 == l2
    _, _, free, _ = P.branch_fc(th, snap, torch.zeros(1, 4), [40], n_free=5)
    assert free == [41, EOTC]
    # the branch inputs: FC frame 1 gets agent PAD and function SOTC, then the previous forced token
    c = th.clone(snap)
    P.branch_fc(th, c, torch.zeros(1, 4), [40, 41])
    assert c["log"][-1] == (PAD, PAD)                          # the clone passed in is not advanced either


def test_measure_clip():
    tok = Tok()
    common, rest = P.name_candidates(lambda s: tok.text_to_ids(s)[1:])
    gold_seq = common + rest["search_catalog"] + [60, 61, EOTC]
    rec = {t: PAD for t in range(200)}
    th = FakeThinker(rec, 25, gold_seq)
    res = P.measure_clip(th, torch.zeros(200, 4), [PAD] * 400, 22, torch.zeros(1, 4), "search_catalog")
    assert res["sotc_peak_frame"] == 25 and res["sotc_peak_rel"] == 3
    assert res["names"]["top"] == "search_catalog" and res["names"]["gold_rank"] == 1
    assert res["names"]["shared_prefix_identical"]
    assert res["args_greedy_ids"] == [60, 61, EOTC]
    res2 = P.measure_clip(th, torch.zeros(200, 4), [PAD] * 400, 22, torch.zeros(1, 4), None)
    assert "args_greedy_ids" not in res2 and set(res2["names"]["name_logp"]) == set(P.TOOLS)


def test_region_stats():
    x = torch.randn(30, 8)
    r = P.region_stats(x, x.clone(), {"a": (0, 10), "b": (10, 30), "empty": (30, 30)}, x[:1], x[:1])
    assert abs(r["a"]["cos_vs_A"] - 1) < 1e-5 and abs(r["b"]["norm_ratio_vs_A"] - 1) < 1e-5 and "empty" not in r


class _St(dict):
    pass


def test_observe_wrappers():
    class Fusion(torch.nn.Module):
        def forward(self, agent_text_embeds=None, user_audio_embeds=None, user_text_embeds=None, function_embeds=None):
            return user_audio_embeds

    class STT(torch.nn.Module):
        def forward(self, x, cache=None):
            v = torch.zeros(1, x.shape[1], V)
            v[..., SOTC] = 3.0
            return {"text_logits": v, "function_logits": v}

    class W:
        def __init__(self):
            self.model = type("M", (), {})()
            self.model.stt_model = STT()
            self.fusion_module = Fusion()
            self._silence_embedding_cache = torch.ones(1, 1, 4)

        def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr, rnnt_state=None):
            pass

        def _apply_rnnt_turn_taking(self, t, gen_text, is_blank, rnnt_state):
            gen_text[0, t] = BOS                          # a forced BOS

        def model_llm_interface(self, x, cache=None, generated_tokens=None, current_step=None):
            out = self.model.stt_model(x)
            return {"predicted_token": out["text_logits"][:, -1].argmax(-1),
                    "function_predicted_token": out["function_logits"][:, -1].argmax(-1)}

    w = W()
    rec = []
    un = observe.install(w, rec)
    g = torch.full((1, 10), PAD)
    w._apply_rnnt_turn_taking(4, g, torch.tensor([True]), {"blank_count": torch.tensor([7]), "first_turn": torch.tensor([True])})
    w.model_llm_interface(torch.zeros(1, 1, 4), current_step=4)
    w.fusion_module(user_audio_embeds=torch.ones(1, 1, 4))
    un()
    kinds = [r["kind"] for r in rec]
    assert kinds == ["forcing", "logits", "feature"], kinds
    assert rec[0]["pre"] == PAD and rec[0]["post"] == BOS and rec[0]["forced"] and rec[0]["rnnt"]["blank_count"] == 7
    assert rec[1]["t"] == 4 and rec[1]["pred_func"] == SOTC and rec[1]["func"]["SOTC"] > rec[1]["func"]["PAD"]
    assert abs(rec[2]["cos_silence"] - 1) < 1e-6
    assert w.model_llm_interface.__name__ == "model_llm_interface"   # uninstalled


if __name__ == "__main__":
    for f in [test_helpers, test_trace_parsing, test_fc_inputs_alignment, test_gate, test_branch_isolation,
              test_measure_clip,
              test_region_stats, test_observe_wrappers]:
        f()
    print("OK t1 probe tests")
