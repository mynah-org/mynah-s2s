"""CPU tests for the B1 teacher KL: spans, positions, prefix identity with the student, KL math."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_kl as T  # noqa: E402
from session_distill import EOTC, PAD, SOTC, rule_call_text, student_targets  # noqa: E402

TGT = {"name": "find_customer", "arguments": {"full_name": "Kevin Fabbri", "postcode": "83772"}}


def test_char_and_token_spans_with_char_tokens():
    call = rule_call_text(TGT)
    sp = T.char_spans(call, TGT)
    assert call[sp["tool"][0]:sp["tool"][1]] == "find_customer"
    assert call[sp["full_name"][0]:sp["full_name"][1]] == "Kevin Fabbri"
    assert call[sp["postcode"][0]:sp["postcode"][1]] == "83772"
    toks = [ord(c) + 1000 for c in call] + [EOTC]
    decode = lambda ids: "".join(chr(i - 1000) for i in ids if i != EOTC)  # noqa: E731
    ts = T.token_spans(toks, decode, sp)
    assert decode([toks[j] for j in ts["postcode"]]) == "83772"
    assert len(toks) - 1 not in sum(ts.values(), [])


def test_query_is_never_a_span():
    t = {"name": "search_catalog", "arguments": {"query": "blue picture frame"}}
    assert set(T.char_spans(rule_call_text(t), t)) == {"tool"}


def test_student_fc_prefix_equals_teacher_prefix():
    fc = [5001, 5002, 5003, EOTC]
    rec = {"main_text": [PAD] * 30, "main_func": [PAD] * 20 + [SOTC], "fc": fc, "topk": []}
    st = student_targets(rec, 5, 18, 18, 40, False)
    idx = [j for j, k in enumerate(st["kind"]) if k == "fc"]
    assert [st["func_in"][j] for j in idx] == [SOTC] + fc[:-1]          # teacher inputs: SOTC, c0, c1, ...
    assert [st["func_lbl"][j] for j in idx] == fc                       # position j predicts c_j


def test_kl_math_zero_when_equal_and_direction():
    V = 50
    lp_s = torch.log_softmax(torch.randn(3, V), -1)
    ids = lp_s.topk(32, -1).indices
    lp_t = lp_s.gather(1, ids)                                          # teacher == student on its top-32
    tp = torch.softmax(lp_t, -1)
    sl = lp_s.gather(1, ids)
    kl = (tp * (tp.clamp_min(1e-9).log() - sl)).sum(1)
    assert (kl >= -1e-6).all()                                          # renormalised teacher vs full-vocab student >= 0
    sharp = torch.full_like(lp_t, -20.0)
    sharp[:, 0] = 0.0
    tp2 = torch.softmax(sharp, -1)
    kl2 = (tp2 * (tp2.clamp_min(1e-9).log() - sl)).sum(1)
    assert torch.allclose(kl2, -sl[:, 0], atol=1e-4)                    # one-hot teacher -> CE on the teacher's top token
