"""CPU tests of the M4 comparison logic (no model): declared differences, tensor differences, C1 row identity."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_parity as M  # noqa: E402


def rec(**kw):
    r = {"modules": [["", "M"]], "tensors": {"param:w": [[2], "torch.bfloat16", "cuda:0", "aa"]}, "cfg": {"s2s": {"x": 1}},
         "fusion": {"type": "AddFusion", "agent_text_weight": 1.0}, "text_pad_id": 12, "silence_sha": "s",
         "prompt_emb": [[1, 3, 4], "bf16", "p"], "patches": {"S2SContextManager.__init__": "nemo.A", "llm.forward": "x.f"},
         "fc_ctx_fix_info": None, "versions": {"torch": "2.10", "engine_path": "/old", "lab_dir": "/o"}}
    for k, v in kw.items():
        r[k] = v
    return r


def test_declared_differences_pass():
    new = rec(fc_ctx_fix_info={"enabled": True}, patches={"S2SContextManager.__init__": "mynah.B", "llm.forward": "x.f"},
              versions={"torch": "2.10", "engine_path": "/new", "lab_dir": "/n"})
    d = M.diff_construction(rec(), new)
    assert d["pass"] and not d["field_diff"] and "fc_ctx_fix_info.enabled" in d["declared_diff"]


def test_undeclared_and_tensor_differences_fail():
    assert not M.diff_construction(rec(), rec(text_pad_id=0))["pass"]
    assert not M.diff_construction(rec(), rec(patches={"S2SContextManager.__init__": "nemo.A", "llm.forward": "y.g"}))["pass"]
    t = rec(tensors={"param:w": [[2], "torch.bfloat16", "cuda:0", "bb"]})
    d = M.diff_construction(rec(), t)
    assert not d["pass"] and d["tensor_diff"] == ["param:w"]


def _write(d, rows, calls):
    d.mkdir(parents=True)
    (d / "construction.json").write_text(json.dumps(rec()))
    (d / "free_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))


def test_c1_compare(tmp_path):
    row = {"arm": "a", "set": "glex_dev", "id": "x", "lang": "it", "tokens": [1, 2], "score": {"exact": False}}
    calls = [{"call": 0, "x": "h", "shape": [1, 1, 4]}]
    _write(tmp_path / "o", [row], calls)
    _write(tmp_path / "n", [row], calls)
    _write(tmp_path / "m", [dict(row, tokens=[1, 3])], calls)
    for new, ok in (("n", 0), ("m", 1)):
        a = type("A", (), {"old": str(tmp_path / "o"), "new": str(tmp_path / new), "mode": "c1",
                           "out": str(tmp_path / f"r_{new}.json")})
        try:
            M.compare(a)
        except SystemExit as e:
            assert e.code == ok


def _write_c2(d, upd_norm=1.0, gnorm=2.0, loss=0.5):
    d.mkdir(parents=True)
    (d / "construction.json").write_text(json.dumps(rec()))
    calls = [{"call": 0, "x": "h", "shape": [1, 2, 4]}, {"ce": 0, "inp": "i", "target": "t", "out": "o", "sum": 1.0},
             {"clip": [[[2], "torch.float32", "v", "g", gnorm]]}, {"update": [["v2", upd_norm, "u"]]}]
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "c2_train").mkdir()
    (d / "c2_train" / "train_log.jsonl").write_text(json.dumps(
        {"step": 1, "batch_ids": [["a|en"]], "loss": loss, "grad": {"g": gnorm}, "comp": {"text:text": loss}}) + "\n")


def test_c2_compare(tmp_path):
    _write_c2(tmp_path / "o")
    cases = {"same": ({}, 0), "update": ({"upd_norm": 1.05}, 1), "grad": ({"gnorm": 2.1}, 1), "loss": ({"loss": 0.51}, 1),
             "within": ({"upd_norm": 1.005, "gnorm": 2.01, "loss": 0.5004}, 0)}
    for name, (kw, code) in cases.items():
        _write_c2(tmp_path / name, **kw)
        a = type("A", (), {"old": str(tmp_path / "o"), "new": str(tmp_path / name), "mode": "c2",
                           "out": str(tmp_path / f"r_{name}.json")})
        try:
            M.compare(a)
            got = 0
        except SystemExit as e:
            got = e.code
        assert got == code, (name, json.load(open(tmp_path / f"r_{name}.json"))["c2"])
