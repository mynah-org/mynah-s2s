"""CPU tests for the corpus-v2 ASR verification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_asr_v2 as V  # noqa: E402


def row(lang, text, target, **slots):
    base = {"concept": None, "attrs": [], "synonym": {}, "name": None, "postcode": None}
    base.update(slots)
    return {"lang": lang, "text": text, "target": target, "slots": base}


def test_catalog_concept_attrs_accents_and_synonym():
    r = row("fr", "Je cherche une étagère murale noire.", {"name": "search_catalog", "arguments": {"query": "black wall shelf"}},
            concept="wall shelf", attrs=["black"])
    assert V.check(r, "je cherche une etagere murale noire") == []
    assert V.check(r, "je cherche une etagere murale") == ["attr:black"]
    r = row("it", "Cerco un sofà.", {"name": "search_catalog", "arguments": {"query": "sofa"}}, concept="sofa",
            synonym={"it": "sofà"})
    assert V.check(r, "cerco un sofa") == []
    assert V.check(r, "cerco un tavolo") == ["concept"]


def test_name_and_grouped_postcode():
    t = {"name": "find_customer", "arguments": {"full_name": "Camille Ferraro", "postcode": "42320"}}
    r = row("fr", "", t, name="Camille Ferraro", postcode="42320")
    assert V.check(r, "ici camille ferraro code postal quarante deux trois vingt") == []
    assert V.check(r, "ici camille ferraro code postal 42 3 20") == []
    assert V.check(r, "ici camille ferraro code postal quarante deux trois trente") == ["postcode"]
    assert V.check(r, "ici camila ferrero code postal 42320") == []          # fuzzy name (>= 0.8)
    assert V.check(r, "ici paul martin code postal 42320") == ["name"]


def test_classify_causes():
    sl = {"postcode": "42320", "name": "Camille Ferraro"}
    assert V.classify([], "x", 0.0, None, sl, "fr") == "ok"
    assert V.classify(["tts_overlong"], "", None, None, sl, "fr") == "tts_verified"
    assert V.classify(["postcode"], "code postal 4z320", 0.05, {"chars_per_s": 15}, sl, "fr") == "asr_probable"
    assert V.classify(["postcode"], "code postal quarante deux trois", 0.05, {"chars_per_s": 15}, sl, "fr") == "uncertain"
    assert V.classify(["name"], "ici camil ferrara", 0.1, {"chars_per_s": 15}, sl, "fr") == "asr_probable"
    assert V.classify(["concept"], "je cherche", 0.6, {"chars_per_s": 15}, sl, "fr") == "tts_verified"
