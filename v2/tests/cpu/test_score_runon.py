"""Run-on / self-talk detection of scripts/score_corpus.py on the C1 shapes (q05 new self-dialogue, q12 golden ramble)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from score_corpus import runon_vs_golden, turn_stats  # noqa: E402

G = "</s><s>Hi! How can I help you today?</s>"


def s(text):
    return {"text": G + text}


def test_q05_self_dialogue_is_new():
    gold = turn_stats(s("<s>The day after Tuesday is Wednesday.</s>"))
    run = turn_stats(s("<s>The day after Tuesday is Wednesday.</s><s>What day of the week is it today?</s>"
                       "<s>Today is Thursday.</s><s>What day comes after Wednesday?</s><s>The day after Wednesday is"))
    assert gold == {"agent_turns": 2, "continuation_turns": 0, "continuation_words": 0, "question_turns": 0}
    assert run["continuation_turns"] == 4 and run["question_turns"] == 2
    r = dict(run)
    assert runon_vs_golden(r, gold) == "new"


def test_q12_golden_ramble_is_pre_existing():
    gold = turn_stats(s("<s>Spiders have eight legs.</s><s>Most spiders have eight eyes, though some species have "
                        "fewer.</s><s>Not all spiders spin silk.</s><s> tarantulas are the largest spiders"))
    run = turn_stats(s("<s>Spiders have eight legs.</s><s>Most spiders have eight eyes.</s><s>They have two main body "
                       "parts.</s><s>They have eight legs, but only six are used for walking.</s><s>They have two"))
    assert runon_vs_golden(dict(run), gold) == "pre_existing"


def test_identical_and_shorter_runs_are_not_flagged():
    t = s("<s>The answer is eighty four.</s>")
    assert runon_vs_golden(turn_stats(t), turn_stats(t)) is None
    gold = turn_stats(s("<s>A.</s><s>B c d.</s>"))
    assert runon_vs_golden(turn_stats(s("<s>A.</s>")), gold) is None
