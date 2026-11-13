import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postcode_v2b as P  # noqa: E402
import session_corpus_v2 as C  # noqa: E402
import session_corpus_v2b as B  # noqa: E402
import verify_asr_v2 as V  # noqa: E402

ITEMS = B.build()


def test_postcode_boundaries_roundtrip_and_unambiguous():
    pcs = [f"{i:05d}" for i in range(0, 1000)]                     # leading zeros (00xxx)
    pcs += [f"{d:02d}{r:03d}" for d in (70, 71, 77, 79, 80, 81, 89, 90, 91, 97, 99) for r in range(0, 1000, 7)]
    pcs += [f"{d:02d}{h}{r:02d}" for d in (20, 30, 60) for h in range(10) for r in (0, 1, 10, 18, 70, 80, 99)]
    for lang in P.LANGS:
        for reading in ("digits", "d3s"):
            for pc in pcs:
                assert P.decode(pc, lang, reading) == {pc}, (lang, reading, pc, P.spoken(pc, lang, reading))


def test_v2_grouped_and_h23_are_ambiguous_without_punctuation():
    assert P.decode("00422", "fr", "v2_212") == {"00422", "00802"}        # "quatre vingt-deux"
    assert len(P.decode("02101", "fr", "h23")) == 2                      # "deux cent un"
    assert len(P.decode("80200", "fr", "d3")) == 2                       # "quatre-vingts deux"
    assert P.spoken("80200", "fr", "d3s") == P.spoken("80200", "fr", "digits")


def test_build_mix_ids_names_and_determinism():
    assert Counter(i["intent"] for i in ITEMS) == Counter(B.MIX)
    assert all(re.fullmatch(r"s4b-\d{4}", i["id"]) for i in ITEMS)
    assert not any(w in ("Margaux", "Faure") for i in ITEMS for w in i["slots"]["name"].split())
    assert json.dumps([i["text"] for i in B.build()]) == json.dumps([i["text"] for i in ITEMS])
    assert not any(" ille " in i["text"]["fr"] for i in ITEMS)


def test_audit_clean_against_frozen_v2_plans():
    frozen = []
    for split, n, seed in (("train", 1000, 20261113), ("dev", 12, 20261114), ("test", 24, 20261115)):
        for it in C.build(n, seed, split, True):
            for l in C.LANGS:
                frozen.append({"id": it["id"], "lang": l, "text": it["text"][l], "slots": it["slots"]})
    rep = B.audit(ITEMS, frozen)
    assert all(v == 0 for v in rep.values()), rep


def test_strict_verifier_accepts_the_d3s_reading_when_the_asr_is_verbatim():
    for it in ITEMS:
        for l in C.LANGS:
            row = {"slots": it["slots"], "target": it["target"], "lang": l}
            hyp = re.sub(r"[^\w' -]", " ", it["text"][l].lower())
            for h in (hyp, hyp.replace("-", " ")):
                assert "postcode" not in V.check(row, h, max_k=4), (it["id"], l, h)


def test_frozen_parser_default_unchanged_and_max_k4_reads_four_token_fr_numbers():
    h = "et mon code postal est quatre vingt dix neuf trois neuf deux"
    assert V.L.parse_spoken_digits(h, "fr", True) == "909392"               # frozen v2 behaviour (max_k=3)
    assert V.L.parse_spoken_digits(h, "fr", True, 4) == "99392"


def test_v2b_verification_is_strict_on_numerals_and_wrong_digits():
    assert P.hyp_has_postcode("code postal vingt six un huit merci", "fr", "20618")
    assert not P.hyp_has_postcode("code postal 26 1 8", "fr", "20618")           # the ASR committed to 26
    assert P.hyp_has_postcode("code postal 20 6 1 8", "fr", "20618")
    assert not P.hyp_has_postcode("code postal vingt six un neuf", "fr", "20618")
    assert P.hyp_has_postcode("cap venti sei uno otto", "it", "20618")
    assert P.hyp_has_postcode("postcode twenty six one eight", "en", "20618")


def test_digit_postcodes_cover_every_department():
    import random
    rng = random.Random(0)
    depts = {int(B.make_postcode(rng, False)[:2]) for _ in range(5000)}
    assert depts == set(range(100))
