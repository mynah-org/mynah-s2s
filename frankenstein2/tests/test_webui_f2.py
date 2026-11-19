"""The web UI's Frankenstein v2 side (webui/server.py, pure Python): the Traces API over the engine's SQLite store,
the demo-dialogue script, and the page routing (v2 page at /, the lab page at /lab)."""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui"))
sys.path.insert(0, str(ROOT))

import server as W  # noqa: E402
from frankenstein2.obs import TraceStore  # noqa: E402


def make_store(path):
    st = TraceStore(path)
    sid = "f2-abc123"
    t = time.monotonic_ns()
    for i, (ev, d) in enumerate([
        ("engine.session", {"language": "it", "voice": "default"}),
        ("asr.turn", {"turn_id": 1, "text": "avete un divano?", "reason": "endpoint"}),
        ("llm.request", {"request_id": 1, "turn_id": 1, "n_messages": 2}),
        ("tool.started", {"request_id": 1, "name": "search_catalog", "args": {"query": "divano"}, "call_id": "c1"}),
        ("tool.completed", {"request_id": 1, "name": "search_catalog", "call_id": "c1", "status": "completed",
                            "ms": 120.0, "result": "{\"items\": 1}"}),
        ("llm.done", {"request_id": 1, "text": "Sì, il divano Fjordby."}),
        ("engine.teardown", {"steps": 100}),
    ]):
        st.sink(ev, {"ev": ev, "session_id": sid, "t_ns": t + i * 10 ** 8, "model_step": i, **d})
    st.close()
    return sid


def test_trace_sessions_and_one_session(tmp_path):
    db = tmp_path / "f2_traces.sqlite"
    sid = make_store(db)
    rows = W.trace_sessions(str(db))
    assert [r["session_id"] for r in rows] == [sid]
    assert rows[0]["language"] == "it" and rows[0]["tools"] == 1 and rows[0]["errors"] == 0
    one = W.trace_session(str(db), sid)
    evs = [e["ev"] for e in one["events"]]
    assert evs[0] == "engine.session" and "tool.completed" in evs
    assert any(e["line"] and "search_catalog" in e["line"] for e in one["events"])
    assert {s["kind"] for s in one["spans"]} >= {"tool", "llm"}
    assert W.trace_session(str(db), "nope") is None
    assert W.trace_sessions(str(tmp_path / "missing.sqlite")) == []


def test_demo_script_covers_the_served_languages():
    for code in ("it", "en", "fr", "de", "es", "pt", "nl", "ko", "tr", "ja", "cs", "hi", "pl", "ru", "et"):
        lines = W.DEMO_SCRIPT[code]
        assert 2 <= len(lines) <= 4 and all(isinstance(x, str) and x for x in lines)


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def test_routes_v2_page_traces_api_and_lab_page(tmp_path):
    db = tmp_path / "f2_traces.sqlite"
    sid = make_store(db)
    root = ROOT / "webui"
    httpd = W.WebUIHTTPServer(("127.0.0.1", 0), lambda *a, **kw: W.StaticHandler(*a, directory=str(root), **kw),
                              v2_ws=None, profile="demo", audio_root=None)
    httpd.engine, httpd.ws_port, httpd.trace_db = "frankenstein2", 8788, str(db)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        port = httpd.server_address[1]
        st, body = _get(port, "/")
        assert st == 200 and b"Frankenstein v2" in body and b"/f2/f2.js" in body
        st, body = _get(port, "/lab")
        assert st == 200 and b"/static/app.js" in body
        st, body = _get(port, "/api/traces/sessions")
        assert json.loads(body)["sessions"][0]["session_id"] == sid
        st, body = _get(port, f"/api/traces/session/{sid}")
        assert json.loads(body)["session"]["language"] == "it"
        st, body = _get(port, "/api/demo-script?lang=fr")
        d = json.loads(body)
        assert d["lang"] == "fr" and not d["fallback"] and d["lines"][0].startswith("Bonjour")
        st, body = _get(port, "/api/demo-script?lang=xx")
        assert json.loads(body)["fallback"] is True
        cfg = json.loads(_get(port, "/api/config")[1])
        assert cfg["traces"] is True and cfg["engine"] == "frankenstein2"
    finally:
        httpd.shutdown()
