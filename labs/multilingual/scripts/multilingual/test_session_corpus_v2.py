"""CPU tests for the D-V3 corpus v2 generator: agreement, canonical queries, derivability, split disjointness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402
import session_corpus_v2 as C  # noqa: E402

CON = {c[0]: c for c in L.CONCEPTS}
ATT = {a[0]: a for a in L.ATTRS}


def test_agreement_and_articles():
    np_ = C.noun_phrase(CON["armchair"], [ATT["red"]], "it")
    assert np_["ind"] == "una poltrona rossa" and np_["def"] == "la poltrona rossa"
    np_ = C.noun_phrase(CON["mirror"], [], "it")
    assert np_["ind"] == "uno specchio" and np_["def"] == "lo specchio"
    np_ = C.noun_phrase(CON["mirror"], [ATT["small"]], "it")
    assert np_["ind"] == "un piccolo specchio"
    np_ = C.noun_phrase(CON["wall shelf"], [ATT["black"]], "fr")
    assert np_["ind"] == "une étagère murale noire" and np_["def"] == "l'étagère murale noire"
    assert np_["poss"] == "mon étagère murale noire"
    assert C.noun_phrase(CON["armchair"], [], "en")["ind"] == "an armchair"


def test_canonical_query_order_and_no_additions():
    q = C.canonical_query(CON["rug"], [ATT["wooden"], ATT["small"], ATT["grey"]])
    assert q == "small grey wooden rug"
    assert C.canonical_query(CON["sofa bed"], []) == "sofa bed"


def test_postcode_forms():
    assert L.spoken_postcode("75011", "fr", True) == "soixante-quinze zéro onze"
    assert L.spoken_postcode("06113", "it", False) == "zero sei uno uno tre"
    assert L.spoken_postcode("42320", "fr", True) == "quarante-deux trois vingt"
    assert L.spoken_postcode("21991", "it", True) == "ventuno nove novantuno"


def test_build_derivable_and_split_disjoint():
    tr = C.build(300, 1, "train")
    pr = C.build(12, 2, "dev") + C.build(24, 3, "test")
    assert all(not C.derivable(i) for i in tr + pr)
    cs = C.split_concepts()
    assert not any(i["slots"]["concept"] and cs[i["slots"]["concept"]] != "train" for i in tr)
    assert not any(i["slots"]["heldout_combo"] for i in tr)
    trn = {i["slots"]["name"] for i in tr if i["slots"]["name"]}
    prn = {i["slots"]["name"] for i in pr if i["slots"]["probe_type"] == "new_name_digits"}
    assert prn and not (trn & prn)
    assert all(not ({"sofa", "bookcase", "wardrobe", "lamp", "desk", "chair"} & {i["slots"]["concept"]}) for i in tr)
    low = " ".join(i["text"][l].lower() for i in tr for l in C.LANGS)
    assert "mario rossi" not in low and "20135" not in low


def test_dev_test_disjoint_and_complete_queries():
    dv, te = C.build(12, 2, "dev"), C.build(24, 3, "test")
    cd = {i["slots"]["concept"] for i in dv if i["slots"]["probe_type"] == "new_concept"}
    ct = {i["slots"]["concept"] for i in te if i["slots"]["probe_type"] == "new_concept"}
    assert cd and ct and not (cd & ct)
    nd = {i["slots"]["name"] for i in dv if i["slots"]["name"]}
    nt = {i["slots"]["name"] for i in te if i["slots"]["name"]}
    assert not (nd & nt)
    # completeness: a said attribute missing from the query is reported
    it = C.build(1, 5, "train")[0]
    it = dict(it, target={"name": "search_catalog", "arguments": {"query": "rug"}},
              slots=dict(it["slots"], concept="rug", attrs=[]),
              text={"en": "I'm looking for a red rug.", "it": "Cerco un tappeto rosso.", "fr": "Je cherche un tapis rouge."})
    assert any("implicit_or_missing_attr" in b for b in C.derivable(it))


def test_digit_roundtrip_grouped_and_digits():
    import random
    rng = random.Random(0)
    for _ in range(500):
        pc = "".join(rng.choice("0123456789") for _ in range(5))
        for lang in ("en", "it", "fr"):
            for g in (False, True):
                assert L.parse_spoken_digits(L.spoken_postcode(pc, lang, g), lang) == pc


def test_escalate_rule_has_empty_arguments_and_prefix_is_stable():
    a, b = C.build(150, 20261113, "train"), C.build(400, 20261113, "train")
    assert [x["text"] for x in a] == [x["text"] for x in b[:150]]          # pilot families = prefix of the full corpus
    esc = [x for x in b if x["intent"] == "transfer"]
    assert esc and all(x["target"] == {"name": "escalate_to_human", "arguments": {}} for x in esc)


def test_synonym_gender_agreement_it_fr():
    """Card §15.3: articles/adjectives agree with the SPOKEN synonym, also when its gender differs from the base noun."""
    C_ = {c[0]: c for c in L.CONCEPTS}
    A_ = {a[0]: a for a in L.ATTRS}
    # FR banc (m) -> banquette (f)
    np_ = C.noun_phrase(C_["bench"], [A_["blue"]], "fr", "banquette")
    assert np_["ind"] == "une banquette bleue" and np_["gender"] == "f" and np_["poss"] == "ma banquette bleue"
    # FR table de chevet (f) -> chevet (m); FR miroir (m) -> glace (f)
    assert C.noun_phrase(C_["nightstand"], [], "fr", "chevet")["poss"] == "mon chevet"
    assert C.noun_phrase(C_["mirror"], [A_["small"]], "fr", "glace")["ind"] == "une petite glace"
    # IT culla (f) -> lettino (m); IT baule (m) -> cassapanca (f); IT coperta (f) -> plaid (m)
    assert C.noun_phrase(C_["crib"], [], "it", "lettino")["ind"] == "un lettino"
    np_ = C.noun_phrase(C_["chest"], [A_["white"]], "it", "cassapanca")
    assert np_["def"] == "la cassapanca bianca" and np_["poss"] == "la mia cassapanca bianca"
    assert C.noun_phrase(C_["blanket"], [A_["red"]], "it", "plaid")["ind"] == "un plaid rosso"
    # same-gender synonyms and base nouns unchanged
    assert C.noun_phrase(C_["bench"], [A_["blue"]], "it", "panchina")["ind"] == "una panchina blu"
    assert C.noun_phrase(C_["bench"], [A_["blue"]], "fr")["ind"] == "un banc bleu"
    # legacy mode reproduces the frozen (defective) agreement
    assert C.noun_phrase(C_["bench"], [A_["blue"]], "fr", "banquette", legacy_gender=True)["ind"] == "un banquette bleu"


def test_every_synonym_has_a_gender():
    for c in L.CONCEPTS:
        assert all(s in L.SYN_GENDER["it"] for s in c[3]) and all(s in L.SYN_GENDER["fr"] for s in c[5])


def test_fixed_generator_changes_only_synonym_clips():
    a, b = C.build(300, 20261113, "train", True), C.build(300, 20261113, "train", False)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    for x, y in zip(a, b):
        assert x["target"] == y["target"] and x["text"]["en"] == y["text"]["en"]
        for l in ("it", "fr"):
            if x["text"][l] != y["text"][l]:
                assert x["slots"]["synonym"][l] in L.SYN_GENDER[l] or (l == "fr" and " ille " in x["text"][l])


def test_fr_feminine_pronoun_is_elle_and_legacy_reproduces_ille():
    t = C.RET[2][2]
    for np_ in [C.noun_phrase(k, [], "fr", None) for k in L.CONCEPTS]:
        s = C.fill(t, "fr", np_, "X", "1")
        assert " ille " not in s and (" elle " in s) == (np_["gender"] == "f") and (" il " in s) == (np_["gender"] == "m")
        if np_["gender"] == "f":
            assert " ille " in C.fill(t, "fr", np_, "X", "1", legacy=True)
    items = C.build(300, 20261113, "train", False)
    assert not any(" ille " in x["text"]["fr"] for x in items)
