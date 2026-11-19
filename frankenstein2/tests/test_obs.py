"""Observability: short lines, the SQLite trace store (events + spans) and its viewer, fed by a real pipeline run."""

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.obs import LineFormatter, LogSink, TraceStore, fanout, show, summarize  # noqa: E402
from frankenstein2.tracing import Tracer  # noqa: E402


def test_summaries_are_short_and_skip_noise():
    long = "parola " * 100
    assert summarize("asr.turn", {"text": long, "reason": "endpoint"}).endswith("(endpoint)")
    assert len(summarize("asr.turn", {"text": long, "reason": "endpoint"})) < 120
    assert summarize("tool.started", {"name": "find_customer", "args": {"full_name": "Mario Rossi", "postcode": "10121"}}) \
        == 'tool: calling find_customer(full_name="Mario Rossi", postcode="10121")'
    line = summarize("tool.completed", {"name": "find_customer", "status": "timed_out", "ms": 10012.5,
                                        "result": json.dumps({"error": "timeout", "message": "x" * 300})})
    assert line.startswith("tool: find_customer -> timed_out in 10012.5 ms") and len(line) < 150
    for ev in ("ears.frame", "ears.hop", "audio.release", "tts.first_audio", "audio.underrun"):
        assert summarize(ev, {}) is None


def test_line_prefix_carries_session_turn_call_and_relative_time():
    f = LineFormatter()
    f("engine.session", {"session_id": "sess-3f2a", "t_ns": 1_000_000_000, "language": "it"})
    f("asr.turn", {"session_id": "sess-3f2a", "t_ns": 3_000_000_000, "turn_id": 2, "text": "ciao", "reason": "endpoint"})
    line = f("tool.started", {"session_id": "sess-3f2a", "t_ns": 4_500_000_000, "call_id": "call1",
                              "name": "search_catalog", "args": {"query": "divano"}})
    assert line == '[s-3f2a turn=2 call=call1 +3.50s] tool: calling search_catalog(query="divano")'


def _events():
    yield "engine.session", 0, {"language": "it"}
    yield "asr.turn", 10, {"turn_id": 1, "text": "dov'è il mio ordine", "reason": "endpoint"}
    yield "llm.request", 10, {"request_id": 1, "turn_id": 1, "n_messages": 1}
    yield "ears.frame", 11, {"p_end": 0.2}
    yield "tool.started", 30, {"call_id": "call1", "name": "find_customer", "args": {"full_name": "Mario Rossi"}}
    yield "tool.completed", 34, {"call_id": "call1", "name": "find_customer", "status": "completed", "ms": 320.0,
                                 "result": '{"found": true}'}
    yield "tts.first_audio", 36, {"sid": "e0t1s0"}
    yield "llm.done", 40, {"request_id": 1, "text": "In consegna domani."}
    yield "ctl.action", 60, {"kind": "CLOSE_TURN", "turn_id": 1, "reason": "played_out"}
    yield "engine.teardown", 70, {"steps": 70}


def test_store_keeps_events_lines_spans_and_full_payloads(tmp_path):
    db = tmp_path / "t.sqlite"
    st = TraceStore(db, flush_s=0.05)
    lines = []
    tr = Tracer(session_id="s1", sink=fanout(st.sink, lambda ev, d: lines.append(ev)))
    for ev, k, f in _events():
        tr.emit(ev, k, **f)
    st.close()
    c = sqlite3.connect(db)
    evs = [r[0] for r in c.execute("SELECT ev FROM events ORDER BY id")]
    assert "ears.frame" not in evs and "tool.completed" in evs            # noise skipped at level normal
    spans = {r[0]: r for r in c.execute("SELECT kind, name, status, call_id, dur_ms FROM spans")}
    assert set(spans) == {"turn", "llm", "tool", "first_audio"}
    assert spans["tool"][1:4] == ("find_customer", "completed", "call1") and spans["tool"][4] >= 0
    data = json.loads(c.execute("SELECT data FROM events WHERE ev='tool.completed'").fetchone()[0])
    assert data["result"] == '{"found": true}'                           # the full payload stays for debugging
    out = []
    show(str(db), "s1", out=out.append)
    assert any("tool: calling find_customer" in x for x in out) and out[0].startswith("[s1")
    out = []
    show(str(db), out=out.append)
    assert out[0].startswith("s1") and "tools=1" in out[0]


def test_text_free_tracer_keeps_text_out_of_every_sink(tmp_path):
    seen = []
    tr = Tracer(session_id="s", store_text=False, sink=lambda ev, d: seen.append(d))
    tr.emit("asr.turn", 1, turn_id=1, text="il mio codice fiscale è ...", reason="endpoint")
    assert "text" not in seen[0]


def test_a_broken_sink_never_breaks_the_session(tmp_path):
    def bad(ev, d):
        raise RuntimeError("boom")
    got = []
    tr = Tracer(session_id="s", sink=fanout(bad, lambda ev, d: got.append(ev)))
    tr.emit("llm.request", 1, request_id=1)
    assert got == ["llm.request"]


def test_log_sink_writes_one_short_line_per_meaningful_event():
    import logging

    class H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, r):
            self.lines.append(r.getMessage())
    lg = logging.getLogger("f2-test-obs")
    lg.setLevel(logging.INFO)
    h = H()
    lg.addHandler(h)
    tr = Tracer(session_id="abc", sink=LogSink(lg).sink)
    for ev, k, f in _events():
        tr.emit(ev, k, **f)
    assert len(h.lines) == 8 and all(len(x) < 160 for x in h.lines)
    assert any("tool: find_customer -> completed in 320.0 ms" in x for x in h.lines)


def test_pipeline_run_lands_in_the_store(tmp_path):
    from frankenstein2.tests.test_pipeline import make, pieces, ev, orders_handler, run
    db = tmp_path / "p.sqlite"
    st = TraceStore(db, flush_s=0.05)
    ears = {"s": {2: {"pieces": pieces(4, "dov'è il mio ordine X Y Z uno due tre"), "events": [ev("ENDPOINT", 11)]}}}
    script = lambda m: [("decided",), ("tool_start", "get_order_status", {"order_id": "X Y Z"}, "call1"),
                        ("tool", "get_order_status", {"order_id": "X Y Z uno due tre"}),
                        ("text", "Il suo ordine è stato spedito.")]
    p, be, llm, talker, tr = make("s", "it", ears, script, tools={"get_order_status": orders_handler({})})
    tr.sink = st.sink
    run(p, 80)
    time.sleep(0.1)
    st.close()
    c = sqlite3.connect(db)
    lines = [r[0] for r in c.execute("SELECT line FROM events WHERE line IS NOT NULL ORDER BY t_ns")]
    assert any('asr: "dov' in x for x in lines) and any("tool: calling get_order_status" in x for x in lines)
    assert any(x.endswith('"Il suo ordine è stato spedito."') for x in lines if "tts:" in x)
