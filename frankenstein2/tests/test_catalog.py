"""The talker language folder: languages found by scanning, family preference, base dirs, hot-added folders, and
language.json registering a new language with its phrases."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2 import langs  # noqa: E402
from frankenstein2.tts.catalog import PackageCatalog  # noqa: E402


def pkg(root, code, fam, prec):
    d = root / code / fam / prec
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    return d


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_scan_prefers_24l_then_6l_and_uses_the_base_for_english(tmp_path):
    pkg(tmp_path, "en", "24l", "base")
    pkg(tmp_path, "en", "6l", "base")
    pkg(tmp_path, "it", "24l", "int4")
    pkg(tmp_path, "it", "6l", "int4")
    pkg(tmp_path, "cs", "6l", "int4")
    pkg(tmp_path, "fr", "24l", "int8")                 # not at the wanted precision
    c = PackageCatalog(tmp_path, "int4")
    assert c.languages() == ["cs", "en", "it"]
    assert c.family("it") == "24l" and c.family("cs") == "6l" and c.path("en").name == "base"
    assert c.base_dir("6l") == tmp_path / "en" / "6l" / "base"
    assert PackageCatalog(tmp_path, "int4", families=("6l",)).family("it") == "6l"


def test_a_folder_added_while_running_is_found_by_the_next_request(tmp_path):
    clock = Clock()
    c = PackageCatalog(tmp_path, "int4", rescan_s=5.0, clock=clock)
    assert c.path("pt") is None
    pkg(tmp_path, "pt", "24l", "int4")
    assert c.path("pt") is None                        # within rescan_s: no disk scan per request
    clock.t = 6.0
    assert c.path("pt") is not None and "pt" in c.languages()


def test_language_json_registers_a_new_language_and_overrides_phrases(tmp_path):
    langs.reset()
    try:
        pkg(tmp_path, "sv", "6l", "int4")
        (tmp_path / "sv" / "language.json").write_text(json.dumps(
            {"code": "sv", "name": "Swedish", "asr_locale": "sv-SE",
             "phrases": {"holding": "Ett ögonblick, jag kollar."}}))
        (tmp_path / "it").mkdir()
        (tmp_path / "it" / "language.json").write_text(json.dumps({"code": "it", "phrases": {"reminder": "Quasi."}}))
        PackageCatalog(tmp_path, "int4")
        assert langs.get("sv").name == "Swedish" and langs.get("sv").asr_locale == "sv-SE"
        assert langs.phrase("sv", "holding") == "Ett ögonblick, jag kollar."
        assert langs.phrase("sv", "apology") == langs.phrase("en", "apology")      # missing: English
        assert langs.phrase("it", "reminder") == "Quasi." and langs.phrase("it", "holding") == "Un momento, controllo."
        assert langs.get("it").tested
    finally:
        langs.reset()


def test_every_registered_language_has_its_four_phrases():
    for code in langs.known():
        for kind in langs.PHRASES:
            assert langs.BUILTIN[code].phrases.get(kind), (code, kind)
