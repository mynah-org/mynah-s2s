"""The model setup script's plan (pure: no network, no download)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.scripts.setup_models import POCKET, TESTED, check_langs, plan  # noqa: E402


def test_plan_fetches_only_the_requested_languages():
    steps = {role: (repo, pats) for repo, pats, role in plan(["it", "fr"], "Qwen/Qwen3.5-4B", "alba-mackenna/casual.wav")}
    assert steps["ears"] == ("nineninesix/tyger", None)
    assert steps["talker weights (gated)"][1] == ["languages/italian_24l/model.safetensors",
                                                     "languages/french_24l/model.safetensors"]
    assert steps["talker tokenizers"][1] == ["languages/italian_24l/tokenizer*", "languages/french_24l/tokenizer*"]
    assert steps["reference voice"] == ("kyutai/tts-voices", ["alba-mackenna/casual.wav"])


def test_languages_are_checked_against_the_official_pocket_list():
    assert check_langs(" IT, en ,fr") == ["it", "en", "fr"]
    assert set(TESTED) <= set(POCKET) and len(POCKET) == 7
    with pytest.raises(SystemExit):
        check_langs("it,cs")


def test_engine_language_table_matches_the_setup_script():
    import re
    src = (Path(__file__).resolve().parents[1] / "runtime.py").read_text()
    table = dict(re.findall(r'"(\w\w)": "(\w+_24l|english_2026-09_24l)"', src))
    assert table == POCKET
