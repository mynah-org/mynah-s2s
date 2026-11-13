"""CLI presets: release is the default, explicit flags win, and a failing optimisation is skipped with a warning."""
import argparse

import pytest

from mynah_v2 import presets
from mynah_v2.model import perf


def _ns(**kw):
    base = {k: None for k in presets.MULTI["release"]}
    base.update(kw)
    return argparse.Namespace(**base)


def test_release_fills_unset_and_keeps_explicit_flags():
    a = _ns(eartts_batch="off")
    got = presets.fill(a, presets.MULTI, "release")
    assert a.graphs == 1 and a.eartts_rng == "session" and a.fc_merge == "on" and a.capture_policy == "precapture"
    assert a.eartts_batch == "off" and got["eartts_batch"] == "off"


def test_reference_preset_and_unknown_preset():
    a = _ns()
    presets.fill(a, presets.MULTI, "reference")
    assert a.graphs == 0 and a.profile == "" and a.eartts_batch == "off"
    with pytest.raises(SystemExit):
        presets.fill(_ns(), presets.MULTI, "fastest")


def test_profile_pool_follows_session_count():
    assert presets.multi_profile_for(4, "optimized-graph-s4") == "optimized-graph-s4"
    assert presets.multi_profile_for(8, "optimized-graph-s4") == "optimized-graph-s8"
    assert presets.multi_profile_for(8, "optimized-graph") == "optimized-graph"


def test_failing_optimisation_is_skipped_not_fatal(monkeypatch, capsys):
    calls = []

    def fake_enable(name, engine):
        calls.append(name)
        if name == "bad":
            raise RuntimeError("unsupported on this GPU")
        return {"ok": name}
    monkeypatch.setattr(perf, "enable", fake_enable)
    out = presets.apply(object(), None, "good,bad,other")
    assert calls == ["good", "bad", "other"]
    assert set(out["active"]) == {"good", "other"} and "bad" in out["skipped"]
    assert "WARNING optimisation 'bad'" in capsys.readouterr().err


def test_release_single_preset_is_enhanced():
    a = argparse.Namespace(mode=None, perf_profile=None, exp=None)
    presets.fill(a, presets.SINGLE, "release")
    assert a.mode == "enhanced" and a.perf_profile == "optimized-graph" and "thinker_graph" in a.exp


def test_strict_mode_fails_on_missing_optimisation(monkeypatch):
    def fake_enable(name, engine):
        if name == "bad":
            raise RuntimeError("unsupported")
        return {}
    monkeypatch.setattr(perf, "enable", fake_enable)
    with pytest.raises(presets.OptimisationUnavailable):
        presets.apply(object(), None, "good,bad", strict=True)
    assert presets.apply(object(), None, "good", strict=True)["skipped"] == {}


def test_release_is_the_fastest_gated_line_and_release_exact_differs_only_in_attention():
    for table in (presets.SINGLE, presets.MULTI):
        rel, exact = table["release"], table["release-exact"]
        assert "eartts_rvq_cache" in rel["exp"] and rel["exp"] == exact["exp"]
        assert (rel["deterministic"], rel["fill_uninit"], rel["sdpa"]) == ("on", "off", "fast")
        assert (exact["deterministic"], exact["fill_uninit"], exact["sdpa"]) == ("on", "off", "default")
        assert {k: v for k, v in rel.items() if k != "sdpa"} == {k: v for k, v in exact.items() if k != "sdpa"}
        assert "eartts_rvq_cache" not in table["reference"]["exp"]


def test_every_preset_exp_is_a_registered_experiment():
    for table in (presets.SINGLE, presets.MULTI):
        for name, p in table.items():
            for e in filter(None, p["exp"].split(",")):
                assert e.split(":")[0] in perf.EXPERIMENTS or e.startswith("wor:"), (name, e)


def test_determinism_flags_default_to_the_preset_and_explicit_flags_win():
    ap = argparse.ArgumentParser()
    presets.add_determinism_args(ap)
    a = ap.parse_args(["--sdpa", "default"])
    for k in ("mode", "perf_profile", "exp"):
        setattr(a, k, None)
    presets.fill(a, presets.SINGLE, "release")
    assert (a.deterministic, a.fill_uninit, a.sdpa) == ("on", "off", "default")
    assert presets.determinism_overrides(a) == {"s2s.deterministic": True}


def test_rnnt_overlap_is_on_in_the_release_presets_and_off_in_reference():
    assert presets.MULTI["release"]["rnnt_overlap"] == presets.MULTI["release-exact"]["rnnt_overlap"] == "on"
    assert presets.MULTI["reference"]["rnnt_overlap"] == "off"
