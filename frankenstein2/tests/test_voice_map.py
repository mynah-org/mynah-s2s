"""Per-language voices: the voice map overrides the server-wide voices for its languages only; the audition ranks
candidates by how well the ASR understood them and never picks the baseline over a native voice."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webui"))

from frankenstein2.lab.voice_audition import picks, rank
from frankenstein2.server import build as B
from frankenstein2.server.engine import Resources, voice_for, voice_names


def _res(vmap=None):
    return Resources(ears=None, make_llm=None, make_talker=None,
                     voices={"default": "/v/default.wav", "caller": "/v/caller.wav"}, voice_map=vmap or {})


def test_voice_for_prefers_the_language_entry_and_falls_back():
    res = _res({"it": {"default": "native:giovanni", "caller": "native:marta"}, "tr": {"caller": "/r/tr_f.wav"}})
    assert voice_for(res, "", "it") == "native:giovanni"           # no voice asked = the language's default
    assert voice_for(res, "caller", "it") == "native:marta"
    assert voice_for(res, "caller", "tr") == "/r/tr_f.wav"
    assert voice_for(res, "", "tr") == ""                          # no default for tr: the talker's own default
    assert voice_for(res, "default", "tr") == "/v/default.wav"
    assert voice_for(res, "caller", "en") == "/v/caller.wav"        # not in the map
    assert voice_for(_res(), "", "it") == ""


def test_voice_names_include_the_map():
    res = _res({"ja": {"default": "/r/ja.wav", "narrator": "/r/n.wav"}})
    assert voice_names(res) == ["caller", "default", "narrator"]


def test_voice_map_file_resolves_relative_paths(tmp_path):
    (tmp_path / "voices").mkdir()
    f = tmp_path / "voices" / "map.json"
    f.write_text(json.dumps({"_sources": "doc only", "ko": {"default": "ko/fleurs_f_1.wav", "caller": "/abs/x.wav"},
                             "it": {"default": "native:giovanni"}}))
    m = B.voice_map(argparse.Namespace(voice_map=str(f)))
    assert set(m) == {"ko", "it"}
    assert m["ko"]["default"] == str((tmp_path / "voices" / "ko" / "fleurs_f_1.wav").resolve())
    assert m["ko"]["caller"] == "/abs/x.wav"
    assert m["it"]["default"] == "native:giovanni"
    assert B.voice_map(argparse.Namespace(voice_map="")) == {}


def test_audition_rank_and_picks():
    rows = [{"voice": "server:caller", "mean": 0.99, "min": 0.98, "s": 9.0, "baseline": True},
            {"voice": "ref:a", "mean": 0.90, "min": 0.80, "s": 9.0, "baseline": False},
            {"voice": "ref:b", "mean": 0.90, "min": 0.85, "s": 9.5, "baseline": False},
            {"voice": "ref:c", "mean": 0.70, "min": 0.60, "s": 8.0, "baseline": False}]
    assert [r["voice"] for r in rank(rows)] == ["server:caller", "ref:b", "ref:a", "ref:c"]
    assert picks(rows) == ["ref:b", "ref:a"]                       # natives first, baseline never picked
    only = [r for r in rows if r["baseline"]]
    assert picks(only) == ["server:caller"]


def test_demo_script_one_sentence_per_line():
    from server import DEMO_SCRIPT
    assert len(DEMO_SCRIPT) == 15
    for lang, lines in DEMO_SCRIPT.items():
        assert len(lines) == 3, lang
        for x in lines:
            inner = x.strip()[:-1]                                  # the line's own end mark aside
            assert not any(c in inner for c in ".?!。？！।"), (lang, x)
