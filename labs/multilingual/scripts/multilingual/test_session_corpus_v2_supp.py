import re

import lexicon_v2 as L
import session_corpus_v2 as C
import session_corpus_v2_supp as X


def test_supplement_clean_and_separate():
    items = X.build_supp(60, 60, 60, 24, 20261114)
    rep = X.audit(items)
    assert not rep["not_derivable"] and not rep["leaks"] and not rep["oos_with_furniture_word"]
    assert all(i["id"].startswith("s4x-") for i in items)
    assert {i["split"] for i in items if i["slots"].get("phrasing") is not None and i["split"] == "idev"} == {"idev"}
    # held-out phrasings never appear in train/val
    held = {p[0] for p in X.TRANSFER_HELDOUT + X.OOS_HELDOUT}
    assert not any(h in i["text"]["en"] for i in items if i["split"] != "idev" for h in held)
    for i in items:                                   # gender-changing synonyms are agreed with the spoken noun
        for l in ("it", "fr"):
            s = i["slots"]["synonym"].get(l)
            if s and s in X.GENDER_MISMATCH_SYN[l] and L.SYN_GENDER[l][s] == "f":
                assert not re.search(rf"\b(un|il|mon) {re.escape(s)}\b", i["text"][l]), i["text"][l]
    assert X.audit(X.build_supp(60, 60, 60, 24, 20261114))["sha256_texts"] == rep["sha256_texts"]


def test_supplement_does_not_change_main_build():
    a = C.build(30, 20261113, "train")
    X.build_supp(10, 10, 10, 4, 1)
    assert C.build(30, 20261113, "train") == a
