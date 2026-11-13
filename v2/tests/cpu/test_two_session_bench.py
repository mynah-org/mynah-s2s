"""two_session_bench harness on synthetic summaries/traces (no model, no GPU): leak detection, identical solo-vs-pair,
deadline accounting, refuse-overwrite, join/leave roster, I9 memory decomposition, unsupported reference driver."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import two_session_bench as B  # noqa: E402

SPEC = Path(B.DEFAULT_SPEC)
P1 = "p1_same_id_conflict"


def _order_call(oid, rows, cid="c1"):
    res = json.dumps({k: v for k, v in B.order_lookup(rows, oid).items()})
    return {"call_id": cid, "name": "get_order_status", "args": {"order_id": oid}, "status": "completed",
            "result": res}


def _trace(sid, text_steps=(10, 40), fc=(100, 200), asr="what's the status of my order"):
    ev = [{"ev": "session.opened", "t_ns": 0, "session_id": sid},
          {"ev": "asr.generated", "t_ns": 1, "session_id": sid, "text": asr}]
    for i in range(20):
        ev.append({"ev": "thinker.step", "t_ns": 1_000_000 * (i + 1) * 160, "session_id": sid, "latency_ms": 50.0,
                   "fc_active": False, "model_step": 2 * i})
    for s in text_steps:
        ev.append({"ev": "turn.started", "t_ns": s, "session_id": sid, "who": "agent", "model_step": s})
    if fc:
        ev += [{"ev": "function.started", "t_ns": fc[0], "session_id": sid},
               {"ev": "tool.result.consumed", "t_ns": fc[1], "session_id": sid}]
    return ev


def _write(d: Path, summ: dict, trace: list[dict], wav: bytes = b"RIFFfake"):
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summ))
    (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in trace) + "\n")
    (d / "agent.wav").write_bytes(wav)


def _p1_runs(tmp_path, a_text_pair, a_result_rows=None):
    spec = B.load_spec(SPEC, [P1])
    g = spec["groups"][0]
    a, b = g["sessions"]
    a_good = "<s>Hello, how can I help?</s><s>Your order JKL789 has shipped and arrives November 21st.</s>"
    b_good = "<s>Hi there!</s><s>I'm sorry, order JKL789 was cancelled.</s>"
    solo, pair = tmp_path / "solo", tmp_path / "pair"
    _write(solo / "p1A" / "t03", {"text": a_good, "tool_calls": [_order_call("JKL789", a["orders_rows"])],
                                  "consumed": {"c1": 50}, "token_ids": {"text": [1, 5, 2]}}, _trace("t03"))
    _write(solo / "p1B" / "c08", {"text": b_good, "tool_calls": [_order_call("JKL789", b["orders_rows"])],
                                  "consumed": {"c1": 52}, "token_ids": {"text": [1, 6, 2]}}, _trace("c08"))
    _write(pair / P1 / "p1A", {"text": a_text_pair,
                               "tool_calls": [_order_call("JKL789", a_result_rows or a["orders_rows"])],
                               "consumed": {"c1": 50},
                               "token_ids": {"text": [1, 5, 2] if a_text_pair == a_good else [1, 7, 2]}},
           _trace("p1A"))
    _write(pair / P1 / "p1B", {"text": b_good, "tool_calls": [_order_call("JKL789", b["orders_rows"])],
                               "consumed": {"c1": 52}, "token_ids": {"text": [1, 6, 2]}}, _trace("p1B"))
    return spec, g, solo, pair, a_good


DRES = {"driver": "test-batched", "wall_s": 3.2, "per_session_rng": True,
        "steps": [{"tick": i, "ms": 70.0, "sids": ["p1A", "p1B"], "batched": True, "phase": "decode"}
                  for i in range(20)],
        "chunk_log": [], "memory": None}


def test_identical_solo_vs_pair_passes(tmp_path):
    spec, g, solo, pair, _ = _p1_runs(tmp_path, "<s>Hello, how can I help?</s><s>Your order JKL789 has "
                                      "shipped and arrives November 21st.</s>")
    rep = B.compare_all(spec, pair, solo, {P1: DRES}, "auto", 160.0)
    grp = rep["groups"][0]
    assert rep["verdict"] == "PASS", json.dumps(grp)[:2000]
    assert [s["status"] for s in grp["sessions"]] == ["IDENTICAL", "IDENTICAL"]
    assert all(s["I8"] == "PASS" for s in grp["sessions"])
    assert grp["checks"]["real_batched_forward"] is True
    assert grp["perf"]["session_chunks"] == 40


def test_leak_of_other_sessions_tool_result_in_answer_is_flagged(tmp_path):
    spec, g, solo, pair, _ = _p1_runs(tmp_path, "<s>Hello!</s><s>I'm sorry, your order JKL789 was cancelled.</s>")
    rep = B.compare_all(spec, pair, solo, {P1: DRES}, "auto", 160.0)
    a = rep["groups"][0]["sessions"][0]
    assert a["status"] == "LEAK" and a["answer_leak_hit"] and not a["answer_leak_hit_solo"]
    assert rep["verdict"] == "FAIL_LEAK"


def test_leak_of_tool_result_from_other_fixture_is_flagged(tmp_path):
    spec = B.load_spec(SPEC, [P1])
    b_rows = spec["groups"][0]["sessions"][1]["orders_rows"]
    good = "<s>Hello, how can I help?</s><s>Your order JKL789 has shipped and arrives November 21st.</s>"
    spec, g, solo, pair, _ = _p1_runs(tmp_path, good, a_result_rows=b_rows)
    a = B.compare_all(spec, pair, solo, {P1: DRES}, "auto", 160.0)["groups"][0]["sessions"][0]
    assert a["status"] == "LEAK"
    assert a["tool_results"][0]["from_other_fixture"] == ["p1B"]


def test_leak_pattern_also_in_solo_is_inconclusive_not_leak(tmp_path):
    txt = "<s>Hi</s><s>Order JKL789 has shipped, November 21st, nothing was cancelled.</s>"
    spec, g, solo, pair, _ = _p1_runs(tmp_path, txt)
    sd = solo / "p1A" / "t03" / "summary.json"
    s = json.loads(sd.read_text())
    s["text"] = txt
    sd.write_text(json.dumps(s))
    a = B.compare_all(spec, pair, solo, {P1: DRES}, "auto", 160.0)["groups"][0]["sessions"][0]
    assert a["status"] != "LEAK" and any("also in solo" in x for x in a["soft"])


def test_audio_difference_fails_I8_only_when_determinism_expected(tmp_path):
    good = "<s>Hello, how can I help?</s><s>Your order JKL789 has shipped and arrives November 21st.</s>"
    spec, g, solo, pair, _ = _p1_runs(tmp_path, good)
    (pair / P1 / "p1A" / "agent.wav").write_bytes(b"RIFFother")
    a = B.compare_all(spec, pair, solo, {P1: DRES}, "auto", 160.0)["groups"][0]["sessions"][0]
    assert a["I8"] == "FAIL" and a["status"] == "REGRESSION"
    a = B.compare_all(spec, pair, solo, {P1: {**DRES, "per_session_rng": None}}, "auto", 160.0)
    assert a["groups"][0]["sessions"][0]["I8"] == "NOT_GATED"


def test_deadline_miss_accounting():
    log = [{"sid": "A", "chunk_idx": i, "arrival_ns": i * 160_000_000, "done_ns": i * 160_000_000 + ms * 1_000_000}
           for i, ms in enumerate([100, 159, 161, 400])]
    d = B.deadline_misses(log, 160.0)
    assert d["A"] == {"chunks": 4, "misses": 2, "worst_ms": 400.0}
    tr = [{"ev": "thinker.step", "t_ns": i, "latency_ms": x, "fc_active": i == 3}
          for i, x in enumerate([50, 170, 160.0, 300])]
    m = B.step_metrics(tr, 160.0)
    assert m["chunks"] == 4 and m["steps_over_budget"] == 2 and m["model_step_ms_p95"] == 170


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "res"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        B.main(["compare", "--out", str(out), "--run-dir", str(tmp_path), "--solo-dir", str(tmp_path),
                "--groups", P1])
    f = tmp_path / "x.json"
    B.write_once(f, {"a": 1})
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        B.write_once(f, {"a": 2})


def test_roster_join_leave_rejoin_and_deferred_leave():
    spec = B.load_spec(SPEC, ["j1_join_leave_rejoin"])
    g = spec["groups"][0]
    ro = B.Roster([s["sid"] for s in g["sessions"]], g["sequence"])
    busy = {"t": True}
    for t in range(300):
        act = ro.update(t, can_leave=lambda sid, conds: not busy["t"])
        if t == 100:
            busy["t"] = False            # A's leave at tick 95 is deferred until it is idle
        if act:
            ro.record(t, act, rows={sid: i for i, sid in enumerate(act)})
        if t == 150:
            ro.mark_done("j1B")
        if t == 250:
            ro.mark_done("j1A")
            ro.mark_done("j1C")
    got = B.collapse_multi(ro.compositions)
    assert got == [["j1A", "j1B"], ["j1A", "j1B", "j1C"], ["j1B", "j1C"], ["j1A", "j1C"]]
    assert B.is_subsequence(g["expected_compositions"], got)
    assert [e for e in ro.events if "leave" in e][0]["tick"] == 101
    assert not B.is_subsequence([["j1A", "j1C"], ["j1A", "j1B"]], got)


def test_interleave_feeds_active_lanes_and_logs_chunks():
    torch = pytest.importorskip("torch")

    class Eng:
        chunk_samples = 4

        def fc_active(self):
            return False

    class Sess:
        def __init__(self):
            self.engine, self.fed, self._fc_buffer, self.last_fc_chunk = Eng(), [], [], -1
            self._agent_speaking = False

        def feed_chunk(self, c):
            self.fed.append(float(c[0]))

        def flush(self, s):
            pass

    a, b = Sess(), Sess()
    lanes = [B.Lane("A", a, [torch.full((4,), float(i)) for i in range(3)], post_fc_s=0.0),
             B.Lane("B", b, [torch.full((4,), 10.0 + i) for i in range(2)], post_fc_s=0.0)]
    res = B.interleave(lanes, [{"tick": 0, "join": "A"}, {"tick": 1, "join": "B"}], pace="fast")
    assert a.fed == [0.0, 1.0, 2.0] and b.fed == [10.0, 11.0]
    assert [c["sids"] for c in res["compositions"]] == [["A"], ["A", "B"]]
    assert len(res["chunk_log"]) == 5


def test_memory_breakdown_is_decomposed_not_a_single_peak():
    mem = {"n_slots": 2, "weights_bytes": 22 * 2 ** 30, "load_peak_bytes": 30 * 2 ** 30,
           "eartts_kv_bytes_per_slot": 2 ** 28, "backend": {"bytes_per_slot": 2 ** 29, "graph_pool_bytes": 2 ** 30,
                                                           "static_buffers_bytes": 0, "extra": {}},
           "cuda": {"max_allocated": 26 * 2 ** 30, "max_reserved": 28 * 2 ** 30}}
    m = B.memory_breakdown(mem)["components"]
    assert m["thinker_kv_mamba_total"]["gb"] == 1.0 and m["eartts_kv_total"]["gb"] == 0.5
    assert m["allocator_reserve"]["gb"] == 2.0 and m["load_peak"]["gb"] == 30.0
    assert m["unaccounted_in_peak"]["gb"] == 1.5
    assert "weights" in B.memory_breakdown({})["missing"]


def test_reference_interleave_is_unsupported_on_single_stream_engine(monkeypatch):
    fake = types.ModuleType("mynah_v2.model.reference")
    fake.ReferenceEngine = type("ReferenceEngine", (), {})
    monkeypatch.setitem(sys.modules, "mynah_v2.model.reference", fake)
    with pytest.raises(B.PairUnsupported, match="single-stream"):
        B.resolve_driver("reference-interleave")
    with pytest.raises(B.PairUnsupported, match="not available"):
        B.resolve_driver("no_such_pkg.mod:fn")


def test_spec_is_valid_english_with_distinct_audio_and_fixtures():
    spec = B.load_spec(SPEC)
    assert {g["kind"] for g in spec["groups"]} == {"pair", "join_leave"}
    assert any(g.get("require_fc_overlap") for g in spec["groups"])
    assert any(all(not s["tools"] for s in g["sessions"]) for g in spec["groups"])
