"""CPU tests for t2_posttool.py (schedule, free run, outcome, gate) with a positional fake Thinker."""
import torch

import t2_posttool as T2
from t1_probe import BOS, EOTC, PAD, SOTC

V = 200


def onehot(i):
    x = torch.full((V,), -20.0)
    x[i] = 0.0
    return x


def test_schedule_positions_and_inputs():
    hist = [PAD] * 134
    hist[19], hist[20] = BOS, 50
    call = [60 + i for i in range(31)] + [EOTC]
    resp = [100 + i for i in range(63)]
    sc = T2.schedule(hist, 133, call, resp)
    assert len(sc) == 134 + 32 + 64 and sc[-1]["pos"] == 229          # 230 is the first free main frame
    assert sc[20]["agent_in"] == BOS and sc[0]["agent_in"] == PAD       # shifted agent input
    fc1 = [r for r in sc if r["kind"] == "fc1"]
    assert fc1[0]["pos"] == 134 and fc1[0]["func_in"] == SOTC and fc1[1]["func_in"] == call[0]
    assert fc1[-1]["pos"] == 165 and fc1[-1]["expect_func"] == EOTC
    fc2 = [r for r in sc if r["kind"] == "fc2"]
    assert fc2[0]["pos"] == 166 and fc2[0]["func_in"] == EOTC and fc2[0]["forced_func"] == resp[0]
    assert fc2[-1]["pos"] == 229 and fc2[-1]["func_in"] == resp[-1] and fc2[-1]["forced_func"] is None
    assert all(r["agent_in"] == PAD and r["audio"] == ("sil",) for r in fc1 + fc2)


def test_free_run_outcomes():
    def step_reply(cache, a, ap, fp):          # PAD, PAD, BOS, 7, 8 ... never SOTC
        k = cache
        t = PAD if k < 2 else BOS if k == 2 else 7
        return onehot(t), onehot(PAD), k + 1
    rows, _ = T2.free_run(step_reply, 0, [None] * 10, PAD, 22, 10)
    o = T2.outcome(rows)
    assert rows[0]["sotc_margin"] < 0 and rows[0]["lp_func_PAD"] == 0.0 and rows[2]["bos_margin"] > 0
    assert o["first"] == "reply" and o["reply_bos"] == 232 and o["new_call_sotc"] is None and o["reply_ids"][:2] == [BOS, 7]

    def step_call(cache, a, ap, fp):
        return onehot(PAD), onehot(SOTC if cache == 1 else PAD), cache + 1
    rows, _ = T2.free_run(step_call, 0, [None] * 10, PAD, 22, 10)
    o = T2.outcome(rows)
    assert o["first"] == "new_call" and o["new_call_sotc"] == 231 and len(rows) == 2   # stops at the new call

    seen = []

    def step_inputs(cache, a, ap, fp):          # own previous tokens are fed back
        seen.append((ap, fp))
        return onehot(5 + cache), onehot(PAD), cache + 1
    T2.free_run(step_inputs, 0, [None] * 3, PAD, 22, 3)
    assert seen == [(PAD, 22), (5, PAD), (6, PAD)]


def test_gate_check():
    rec = {t: PAD for t in range(270)}
    rec[232], rec[233] = BOS, 1073
    hist_pred = [(PAD, PAD)] * 133 + [(PAD, SOTC)]
    call = [60, 61, EOTC]
    free = [{"pos": p, "text": rec[p] if p >= 232 else PAD, "func": PAD} for p in range(230, 263)]
    g = T2.gate_check(hist_pred, None, call, call, 22, free, rec)
    assert g["pass"], g
    assert not T2.gate_check(hist_pred, None, call, call, 21, free, rec)["pass"]                 # no EOTR at 229
    bad = [dict(r, func=SOTC) if r["pos"] == 231 else r for r in free]
    assert not T2.gate_check(hist_pred, None, call, call, 22, bad, rec)["pass"]                  # new call
    late = [dict(r, text=PAD) if r["pos"] == 232 else r for r in free]
    assert T2.gate_check(hist_pred, None, call, call, 22, late, rec)["P4_text_mismatch"] == [232]


if __name__ == "__main__":
    test_schedule_positions_and_inputs()
    test_free_run_outcomes()
    test_gate_check()
    print("OK t2")
