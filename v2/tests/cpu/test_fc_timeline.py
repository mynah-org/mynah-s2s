from mynah_v2.model.tokens import EOTC_ID, EOTR_ID, PAD_ID, SOTC_ID
from mynah_v2.runtime.fc_timeline import check_fc_timeline

P = PAD_ID


def _reference_like(call=(101, 102, 103), resp_len=4, free_before=0):
    """text, func, forced for: 2 normal steps, SOTC call EOTC, [free], forced response, EOTR, 2 normal steps."""
    text = [7, 8]
    func = [P, P]
    forced = [False, False]
    for f in (SOTC_ID, *call, EOTC_ID):
        text.append(P); func.append(f); forced.append(False)
    for _ in range(free_before):
        text.append(55); func.append(P); forced.append(False)
    for i in range(resp_len):
        text.append(P); func.append(500 + i); forced.append(True)
    text.append(P); func.append(EOTR_ID); forced.append(False)
    text += [9, 10]; func += [P, P]; forced += [False, False]
    return text, func, forced


def test_reference_sequence_passes():
    text, func, forced = _reference_like()
    rep = check_fc_timeline(text, func, forced_response=forced)
    assert rep.ok, rep.violations
    (s,) = rep.spans
    assert s.call_tokens == [101, 102, 103]
    assert s.eotr_step == s.resp_end + 1


def test_free_text_between_eotc_and_response_is_caught():
    # the vLLM-Omni failure mode (P0 §1a): free frames with agent text after EOTC before the result
    text, func, forced = _reference_like(free_before=3)
    rep = check_fc_timeline(text, func, forced_response=forced)
    assert not rep.ok
    assert any("free step between EOTC" in v for v in rep.violations)
    assert any("agent text" in v for v in rep.violations)


def test_text_must_be_pad_during_call():
    text, func, forced = _reference_like()
    text[3] = 42  # inside call tokens
    rep = check_fc_timeline(text, func, forced_response=forced)
    assert any("agent text 42" in v for v in rep.violations)


def test_missing_eotr_and_unclosed_span():
    text, func, forced = _reference_like()
    i = func.index(EOTR_ID)
    func[i] = P
    rep = check_fc_timeline(text, func, forced_response=forced)
    assert any("no EOTR" in v for v in rep.violations)
    rep2 = check_fc_timeline([P, P, P], [SOTC_ID, 5, 6])
    assert any("not closed" in v for v in rep2.violations)


def test_user_silence_required_on_inserted_frames():
    text, func, forced = _reference_like()
    sil = [f != P or fr for f, fr in zip(func, forced)]
    rep = check_fc_timeline(text, func, user_silence=sil, forced_response=forced)
    assert rep.ok, rep.violations
    sil[3] = False
    rep = check_fc_timeline(text, func, user_silence=sil, forced_response=forced)
    assert any("not silence" in v for v in rep.violations)


def test_stray_eotc_outside_call():
    rep = check_fc_timeline([P, P], [EOTC_ID, P])
    assert any("outside a call" in v for v in rep.violations)
