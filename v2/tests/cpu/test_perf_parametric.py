"""wo:<fmt>:<selector> parametric experiments: parsing, '+' → ',' selector mapping, unknown names still raise."""

import pytest

from mynah_v2.model import perf


def test_parametric_wo_maps_plus_to_comma_and_tags(monkeypatch):
    calls = []
    monkeypatch.setattr(perf, "_wo_swap", lambda e, fmt, sel, tag: calls.append((fmt, sel, tag)) or {"ok": 1})
    monkeypatch.setattr(perf, "_wo_restore", lambda e, tag: calls.append(("restore", tag)))
    assert perf.enable("wo:f8:all+!attention", object()) == {"ok": 1}
    perf.disable("wo:f8:all+!attention", object())
    assert calls[0][:2] == ("f8", "all,!attention")
    assert calls[1] == ("restore", calls[0][2])          # same tag on enable and disable


def test_named_experiments_unchanged_and_unknown_raises():
    assert "thinker_w8" in perf.EXPERIMENTS
    with pytest.raises(KeyError):
        perf.enable("no_such_experiment", object())


def test_resident_mode_does_not_keep_originals_and_refuses_restore(monkeypatch):
    calls = []
    monkeypatch.setattr(perf, "_wo_swap", lambda e, fmt, sel, tag, keep=True: calls.append((fmt, sel, keep)) or {})
    perf.enable("wor:w8:mlp", object())
    assert calls == [("w8", "mlp", False)]
    with pytest.raises(NotImplementedError):
        perf.disable("wor:w8:mlp", object())
