"""The model setup script's plan (pure: no network, no download)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.scripts.setup_models import POCKET, TESTED, check_langs, llm_repo, plan  # noqa: E402


def test_plan_packages_default_fetches_the_base_and_one_package_per_language():
    steps = {role: (repo, pats) for repo, pats, role in plan(["it", "fr"], "qwen4b", "alba-mackenna/casual.wav",
                                                              packages_repo="u/pocket-tts-deltas")}
    assert steps["ears"] == ("nineninesix/tyger", None)
    assert steps["llm"][0] == "Qwen/Qwen3.5-4B"
    assert steps["talker base (gated)"][1] == ["languages/english_2026-09_24l/model.safetensors"]
    assert steps["talker language packages"] == ("u/pocket-tts-deltas", ["w4/italian_24l/*", "w4/french_24l/*"])
    assert steps["reference voice"] == ("kyutai/tts-voices", ["alba-mackenna/casual.wav"])


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
    with pytest.raises(SystemExit):
        check_langs("it,cs")


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
