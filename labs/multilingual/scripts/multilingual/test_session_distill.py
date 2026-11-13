"""CPU tests for the D-V3 pure parts: teacher filter, target alignment, no leakage, masks, param groups."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_distill as S  # noqa: E402
from session_distill import BOS, EOS, EOTC, PAD, SOTC  # noqa: E402

CS, EN_END = 10, 20


def rec(text, func, fc=()):
    return {"main_text": list(text), "main_func": list(func), "fc": list(fc), "topk": [[[PAD], [0.0]]] * len(text)}


def opener():
    return [PAD, BOS, 101, 102, EOS] + [PAD] * 5      # frames 0..9, ends before CS


def test_filter_rejects_barge_in_and_early_call_keeps_tolerance():
    t = opener() + [PAD] * 5 + [BOS, 7, 8] + [PAD] * 20                 # BOS at 15 < EN_END-4 = 16
    assert S.teacher_filter(rec(t, [PAD] * len(t)), CS, EN_END) == (False, "talk_during_caller")
    f = [PAD] * 30
    f[12] = SOTC
    t = opener() + [PAD] * 20
    assert S.teacher_filter(rec(t, f), CS, EN_END) == (False, "call_during_caller")
    t = opener() + [PAD] * 6 + [BOS, 7, EOS] + [PAD] * 20               # BOS at 16 = EN_END-4: boundary slop
    assert S.teacher_filter(rec(t, [PAD] * len(t)), CS, EN_END) == (True, "ok")
    t = opener() + [PAD] * 80
    assert S.teacher_filter(rec(t, [PAD] * len(t)), CS, EN_END) == (False, "no_response")


def test_targets_retime_mask_and_no_leakage():
    t = opener() + [PAD] * 12 + [BOS, 31, 32, EOS] + [PAD] * 10           # reply BOS at 22 = EN_END + 2
    r = rec(t, [PAD] * len(t))
    x_end = 26                                                         # the IT turn is 6 frames longer
    st = S.student_targets(r, CS, EN_END, x_end, 60, is_en=False)
    tl = st["text_lbl"]
    assert tl[:CS] == t[:CS]                                           # opener copied
    assert all(x == PAD for x in tl[CS:x_end - S.TOL])                  # listening during the IT turn
    assert tl[x_end + 2] == BOS and tl[x_end + 3] == 31               # re-timed with the same offset
    n = len(tl)
    for i in range(1, n):                                              # inputs = labels shifted by one (no leakage)
        assert st["agent_in"][i] == (PAD if st["kind"][i] == "fc" else tl[i - 1])
        assert st["func_in"][i] == st["func_lbl"][i - 1]
    assert st["agent_in"][0] == PAD and st["func_in"][0] == PAD
    assert not any(st["kl"])                                           # no KL for non-EN students
    assert all(a == (k in ("lead", "after")) for a, k in zip(st["anchor"], st["kind"]))
    w = dict(zip(tl, st["text_w"]))
    assert w[BOS] == 12.5 and w[EOS] == 7.5 and w[31] == 5.0 and w[PAD] == 1.0
    assert all(st["teacher_frame"][i] is None for i in range(CS, x_end - S.TOL))


def test_targets_first_call_then_stop_and_en_kl():
    t = opener() + [PAD] * 15
    f = [PAD] * len(t)
    f[EN_END + 1] = SOTC
    t = t[:EN_END + 2]
    f = f[:EN_END + 2]
    r = rec(t, f, fc=[501, 502, EOTC])
    st = S.student_targets(r, CS, EN_END, EN_END, 60, is_en=True)
    k = st["kind"]
    i = st["func_lbl"].index(SOTC)
    assert k[i + 1:] == ["fc"] * 3 and st["func_lbl"][i + 1:] == [501, 502, EOTC]
    assert [st["text_w"][j] for j in range(i + 1, len(k))] == [0.0] * 3   # forced PAD text not supervised
    assert st["func_in"][i + 1] == SOTC and st["agent_in"][i + 1] == PAD
    assert st["pos"][-1] == ("s", 2) and len(st["pos"]) == i + 4         # the sample ends after EOTC
    assert all(st["kl"][j] == (k[j] != "fc") for j in range(len(k)))     # EN: KL on every main frame
    assert all(st["teacher_frame"][j] == st["pos"][j][1] for j in range(len(k)) if k[j] != "fc")


def test_kept_keys_excludes_whole_families():
    items = {"a": {"keep": True}, "b": {"keep": False}}
    man = {("a", "en"), ("a", "it"), ("b", "en"), ("b", "it")}
    tr, va = S.kept_keys(items, man, {"a": "train", "b": "train"}, ["en", "it", "fr"])
    assert tr == [("a", "en"), ("a", "it")] and va == []


def test_param_groups_materialised_and_all_get_grads():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from adapters import ResidualAdapter
    ad = ResidualAdapter(d=8, hidden=16)
    torch.nn.init.normal_(ad.down.weight, std=0.1)     # past zero-init so every group has a gradient path
    ln = torch.nn.LayerNorm(8)
    g = S.param_groups(ad, list(ln.parameters()), 1e-4, 0.1, 5e-5)
    assert [x["name"] for x in g] == ["affine", "mlp", "encoder_ln"]
    assert all(isinstance(x["params"], list) and x["params"] for x in g)
    opt = torch.optim.AdamW([{k: v for k, v in x.items() if k != "name"} for x in g])
    assert all(len(pg["params"]) > 0 for pg in opt.param_groups)        # the V2.1 bug: affine group was empty
    y = ad(ln(torch.randn(4, 8))).pow(2).sum()
    y.backward()
    for x in g:
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in x["params"]), x["name"]


def test_opener_end_frame():
    tx = [PAD] * 9 + [BOS, 5, EOS] + [PAD] * 30 + [BOS, 6, 7] + [PAD] * 20
    assert S.opener_end_frame(tx) == 44
    assert S.opener_end_frame(tx[:50]) is None          # still inside the 15-frame quiet window


def test_onset_eos_is_not_barge_in_and_is_kept():
    t = opener() + [PAD] * 2 + [EOS] + [PAD] * 7 + [BOS, 7, EOS] + [PAD] * 20     # EOS at 12 (onset + 2), reply at 20
    r = rec(t, [PAD] * len(t))
    assert S.teacher_filter(r, CS, EN_END) == (True, "ok")
    st = S.student_targets(r, CS, EN_END, 26, 60, is_en=False)
    assert st["text_lbl"][12] == EOS and st["text_w"][12] == 7.5
    assert all(x == PAD for j, x in enumerate(st["text_lbl"][CS:22]) if CS + j != 12)
    t2 = list(t)
    t2[14] = BOS                                                                    # a real onset inside the turn
    assert S.teacher_filter(rec(t2, [PAD] * len(t2)), CS, EN_END) == (False, "talk_during_caller")


def test_constrain_listens_inside_turn_only():
    w = (CS, EN_END - S.TOL)
    assert S.constrain(BOS, PAD, CS + 1, w) == (PAD, PAD, "text")
    assert S.constrain(PAD, SOTC, CS + 2, w) == (PAD, PAD, "sotc")
    assert S.constrain(EOS, PAD, CS + 9, w) == (EOS, PAD, None)            # onset yield marker passes
    assert S.constrain(BOS, PAD, EN_END - S.TOL, w) == (BOS, PAD, None)    # tolerance boundary: free
    assert S.constrain(BOS, PAD, 5, w) == (BOS, PAD, None)                 # opener free


def test_correct_call_grounds_values_and_drops_invented_keys():
    t = '<TOOLCALL>[{"name": "find_customer", "arguments": {"full_name": "Davy DeBois", "postcode": "12345"}}]</TOOLCALL>'
    new, ch = S.correct_call(t, {"name": "Davide Dubois", "postcode": None})
    assert new == '<TOOLCALL>[{"name": "find_customer", "arguments": {"full_name": "Davide Dubois"}}]</TOOLCALL>'
    assert ch == ["fix:full_name", "drop:postcode"]
    t = '<TOOLCALL>[{"name": "search_catalog", "arguments": {"query": "rug"}}]</TOOLCALL>'
    assert S.correct_call(t, {"name": "X Y", "postcode": "1"}) == (t, [])
    assert S.correct_call("<TOOLCALL>[{bad", {})[0] is None


def test_admit_policy():
    teach = opener() + [PAD] * 12 + [BOS, 7, EOS] + [PAD] * 10        # teacher replies at 22, no call
    r = rec(teach, [PAD] * len(teach))
    rule = {"name": "search_catalog", "arguments": {"query": "rug"}}
    st, new = S.admit(r, rule, None, CS, EN_END, 2, [501, EOTC])
    assert st == "teacher_no_call" and new["main_func"][EN_END + 2] == SOTC and new["fc"] == [501, EOTC]
    assert all(x == PAD for x in new["main_text"][CS:EN_END + 3])    # the wrong reply is not supervised
    f = [PAD] * len(teach)
    f[EN_END + 1] = SOTC
    r2 = rec(teach[:EN_END + 2], f[:EN_END + 2], fc=[900, EOTC])
    assert S.admit(r2, rule, "find_customer", CS, EN_END, 2, [501, EOTC])[0] == "teacher_wrong_tool"
    assert S.admit(r2, rule, "search_catalog", CS, EN_END, 2, [501, EOTC])[1]["fc"] == [501, EOTC]
    assert S.admit(r2, None, "search_catalog", CS, EN_END, 2, None)[0] == "quarantine:called_without_rule"
    assert S.admit(r, None, None, CS, EN_END, 2, None)[0] == "agree"
    silent = rec(opener() + [PAD] * 40, [PAD] * 50)
    assert S.admit(silent, None, None, CS, EN_END, 2, None)[0] == "quarantine:no_response"
    assert S.admit(r, rule, None, CS, EN_END, 60, [501, EOTC])[0] == "quarantine:temporal"
    esc = {"name": "escalate_to_human", "arguments": None}
    assert S.admit(r, esc, None, CS, EN_END, 2, None)[0] == "quarantine:no_rule_args"


def test_constrain_rule_none_suppresses_sotc_after_turn():
    w = (CS, EN_END - S.TOL, 10 ** 9)
    assert S.constrain(PAD, SOTC, EN_END + 3, w) == (PAD, PAD, "sotc_after")
    assert S.constrain(BOS, PAD, EN_END + 3, w) == (BOS, PAD, None)          # the reply is free
    assert S.constrain(BOS, SOTC, CS + 1, w) == (PAD, PAD, "text+sotc")


def test_usable_keys_excludes_grammar_defect_and_asr_fail_only_that_clip():
    man = {("a", "en"): {"asr_ok": True}, ("a", "fr"): {"asr_ok": True, "excluded": "grammar_defect"},
           ("a", "it"): {"asr_ok": True}, ("b", "it"): {"asr_ok": False}, ("c", "en"): {}}
    assert S.usable_keys(man) == {("a", "en"), ("a", "it"), ("c", "en")}


def test_regreet_quarantine_only_on_supervised_frames_after_lead():
    dec = {1: "<s>", 2: "</s>", 900: "How can I help you today?", 901: "Sure, which order?"}
    decode = lambda ids: " ".join(dec.get(x, "") for x in ids)
    opener = [S.BOS, 900, S.EOS]
    ok = {"main_text": opener + [S.PAD] * 5 + [S.EOS, S.PAD, S.BOS, 901]}
    bad = {"main_text": opener + [S.PAD] * 5 + [S.EOS, S.PAD, S.BOS, 900, 901]}
    padded = {"main_text": opener + [S.PAD] * 5 + [S.EOS] + [S.PAD] * 4}       # teacher_no_call: reply PADded
    assert not S.regreet(ok, 8, decode)                 # the opener before the lead is never inspected
    assert S.regreet(bad, 8, decode)
    assert not S.regreet(padded, 8, decode)
