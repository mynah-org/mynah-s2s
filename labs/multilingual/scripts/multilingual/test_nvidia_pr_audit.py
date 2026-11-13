"""Minimal CPU checks for the NVIDIA PR #15558 / #16597 audit (see .work/multilingual/nvidia-pr-audit.md).

#16597 is a label/weight id mismatch: labels remapped user EOS -> text EOS, the weight still matched user EOS. The
D-V3 trainer builds labels and weights from the SAME constant, with no remapping; these tests pin that."""
import session_distill as S


def _rec():
    return {"main_text": [S.PAD] * 5 + [S.EOS] + [S.PAD] * 6 + [S.BOS, 1000, 1001, S.PAD],
            "main_func": [S.PAD] * 15 + [S.SOTC], "fc": [500, 501, S.EOTC], "topk": []}


def test_eos_label_gets_eos_weight_no_remap():
    st = S.student_targets(_rec(), cs=2, en_end=10, x_end=10, n_audio=40, is_en=True)
    eos = [i for i, t in enumerate(st["text_lbl"]) if t == S.EOS]
    assert eos and all(st["text_w"][i] == S.W_TEXT["eos"] for i in eos)
    assert all(st["text_w"][i] == S.W_TEXT["bos"] for i, t in enumerate(st["text_lbl"]) if t == S.BOS)
    assert set(st["text_lbl"]) <= {S.PAD, S.BOS, S.EOS, 1000, 1001}       # no user-channel ids ('^'/'$') anywhere


def test_onset_eos_kept_at_teacher_offset():
    st = S.student_targets(_rec(), cs=2, en_end=10, x_end=10, n_audio=40, is_en=True)
    assert [p[1] for p, t in zip(st["pos"], st["text_lbl"]) if t == S.EOS] == [5]


def test_function_weights_by_label():
    st = S.student_targets(_rec(), cs=2, en_end=10, x_end=10, n_audio=40, is_en=True)
    for lbl, kd, w in zip(st["func_lbl"], st["kind"], st["func_w"]):
        if kd == "fc":
            assert w == S.W_FUNC["call"]                                    # EOTC included (call class)
        elif lbl == S.SOTC:
            assert w == S.W_FUNC["sotc"]
        else:
            assert lbl == S.PAD and w == S.W_FUNC["pad"]
