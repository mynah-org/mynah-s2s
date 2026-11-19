"""The model setup script's plan (pure: no network, no download)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.scripts.setup_models import POCKET, TESTED, check_langs, llm_repo, pick_families, plan  # noqa: E402


def test_plan_packages_fetches_each_language_in_its_family_and_the_family_bases():
    files = ["it/24l/int4/manifest.json", "it/6l/int4/manifest.json", "cs/6l/int4/manifest.json",
             "en/24l/base/manifest.json", "en/6l/base/manifest.json"]
    fams = pick_families(files, ["it", "cs", "en"])
    assert fams == {"it": "24l", "cs": "6l", "en": "24l"}
    steps = {role: (repo, pats) for repo, pats, role in plan(["it", "cs", "en"], "qwen4b", "alba-mackenna/casual.wav",
                                                              packages_repo="u/pocket-tts-deltas", families=fams)}
    pats = steps["talker language packages"][1]
    assert "it/24l/int4/*" in pats and "cs/6l/int4/*" in pats and "en/24l/base/*" in pats and "en/6l/base/*" in pats
    assert not any(p.startswith("it/6l") for p in pats)
    assert steps["talker Mimi + voice cloning (gated)"][1] == ["languages/english_2026-09_24l/model.safetensors"]
    assert steps["llm"][0] == "Qwen/Qwen3.5-4B"


def test_plan_full_checkpoints_and_llm_choices():
    steps = {role: (repo, pats) for repo, pats, role in plan(["it"], "moe", "v.wav", talker="full")}
    assert steps["llm"][0] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert steps["talker weights (gated)"][1] == ["languages/italian_24l/model.safetensors"]
    assert steps["talker tokenizers"][1] == ["languages/italian_24l/tokenizer*"]
    assert llm_repo("nano") == ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nano")
    assert llm_repo("org/other-model")[1] == "qwen"


def test_languages_are_checked_against_the_official_pocket_list():
    assert check_langs(" IT, en ,fr") == ["it", "en", "fr"]
    assert set(TESTED) <= set(POCKET) and len(POCKET) == 7
    assert check_langs("it,cs,ko") == ["it", "cs", "ko"]               # packages: every registered language
    with pytest.raises(SystemExit):
        check_langs("it,cs", talker="full")                         # full checkpoints: official only
    with pytest.raises(SystemExit):
        check_langs("xx")


def test_engine_language_table_matches_the_setup_script():
    import re
    src = (Path(__file__).resolve().parents[1] / "runtime.py").read_text()
    table = dict(re.findall(r'"(\w\w)": "(\w+_24l|english_2026-09_24l)"', src))
    assert table == POCKET


def test_llm_preset_sets_backend_and_mode_unless_given():
    import argparse
    from frankenstein2.server import build as B

    def resolved(*argv):
        ap = argparse.ArgumentParser()
        B.add_args(ap)
        a = ap.parse_args(list(argv))
        B.resolve_llm(a)
        return a.llm_backend, a.qwen_mode
    assert resolved("--llm-preset", "qwen4b") == ("qwen", "int8")
    assert resolved("--llm-preset", "moe") == ("qwen", "bf16")
    assert resolved("--llm-preset", "nano")[0] == "nano"
    assert resolved("--llm-preset", "qwen4b", "--qwen-mode", "bf16") == ("qwen", "bf16")
