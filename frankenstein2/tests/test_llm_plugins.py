"""LLM plugin pieces: Qwen3.5 call parser + speech splitter + round logic (fake process), code grounding."""

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.llm.qwen import TIMEOUT_PAYLOAD, TOOL_FAILED, QwenLLM, SpeechSplitter, parse_calls  # noqa: E402
from frankenstein2.text.spoken_numbers import ground_codes  # noqa: E402

CALL = ("<tool_call>\n<function=find_customer>\n<parameter=name>\nMario Rossi\n</parameter>\n"
        "<parameter=postcode>\n01201\n</parameter>\n</function>\n</tool_call>")


def test_parse_qwen_xml_call_keeps_leading_zero_codes():
    assert parse_calls(CALL) == [{"name": "find_customer", "arguments": {"name": "Mario Rossi", "postcode": "01201"}}]


def test_splitter_never_speaks_call_markup():
    sp = SpeechSplitter()
    said = []
    for d in ["Un mom", "ento. <tool", "_call>\n<function=x>", "</function>"]:
        s, _ = sp.feed(d)
        said.append(s)
    assert "".join(said) == "Un momento. " and sp.in_call and sp.call.startswith("<tool_call>")


def test_grounding_fixes_the_garbled_postcode_every_model_produced():
    args, fixes = ground_codes({"name": "Mario Rossi", "postcode": "01201"},
                               ["Mi chiamo Mario Rossi e il mio cap è uno zero uno due uno."], "it")
    assert args["postcode"] == "10121" and fixes


class FakeQwenProc:
    def __init__(self, replies):
        self.replies, self.sent, self.box = list(replies), [], []

    def send(self, m):
        self.sent.append(m)
        if m["op"] == "gen":
            text = self.replies.pop(0)
            self.box += [{"ev": "delta", "rid": m["rid"], "text": text[i:i + 7]} for i in range(0, len(text), 7)]
            self.box.append({"ev": "end", "rid": m["rid"]})

    def poll(self, ss):
        out, self.box = self.box, []
        return out


def drain(q, until="done", limit_s=3.0):
    """Poll until an event of kind ``until`` (tools run on a thread pool, so a round may take several polls)."""
    evs, t_end = [], time.monotonic() + limit_s
    while time.monotonic() < t_end:
        evs += q.poll()
        if any(e["ev"] == until for e in evs):
            return evs
        time.sleep(0.002)
    raise AssertionError(f"no {until!r} event: {[e['ev'] for e in evs]}")


def test_qwen_round_trip_runs_tool_and_resumes():
    seen = []
    proc = FakeQwenProc([CALL, "Trovato, signor Rossi."])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (seen.append(a), json.dumps({"ok": 1}))[1]}, "it",
                user_texts=lambda: ["il mio cap è uno zero uno due uno"])
    q.submit(1, [{"role": "user", "content": "Mario Rossi, cap uno zero uno due uno"}])
    evs = drain(q)
    kinds = [e["ev"] for e in evs]
    assert kinds[0] == "decided" and "tool" in kinds and kinds[-1] == "done"
    assert seen == [{"name": "Mario Rossi", "postcode": "10121"}]
    assert "".join(e["text"] for e in evs if e["ev"] == "text") == "Trovato, signor Rossi."
    assert proc.sent[1]["messages"][-1] == {"role": "tool", "content": '{"ok": 1}' + "\n(Answer the caller in Italian.)"}


def test_qwen_history_markup_round_trips_through_the_parser():
    m = QwenLLM.call_markup("find_customer", {"full_name": "Mario Rossi", "postcode": "10121"})
    assert parse_calls(m) == [{"name": "find_customer", "arguments": {"full_name": "Mario Rossi", "postcode": "10121"}}]


FORCED = ("<function=search_catalog>\n<parameter=query>\ndivano tre posti\n</parameter>\n</function>\n</tool_call>")


def test_qwen_forces_the_call_a_turn_only_announced():
    seen = []
    proc = FakeQwenProc(["Cerco il divano a tre posti nel catalogo.", FORCED, "C'è il divano Fjordby."])
    q = QwenLLM(proc, "s", "sys", [], {"search_catalog": lambda a: (seen.append(a), json.dumps({"ok": 1}))[1]}, "it")
    q.submit(1, [{"role": "user", "content": "Avete divani a tre posti?"}])
    evs = drain(q)
    assert proc.sent[1]["assistant_prefix"] == "Cerco il divano a tre posti nel catalogo.\n<tool_call>\n"
    assert seen == [{"query": "divano tre posti"}] and q.forced == 1
    said = "".join(e["text"] for e in evs if e["ev"] == "text")
    assert "<" not in said and said.endswith("C'è il divano Fjordby.")
    hist = proc.sent[2]["messages"]
    assert hist[-2]["role"] == "assistant" and hist[-2]["content"].startswith("Cerco il divano")
    assert parse_calls(hist[-2]["content"]) == [{"name": "search_catalog", "arguments": {"query": "divano tre posti"}}]


def test_qwen_does_not_force_a_plain_answer():
    proc = FakeQwenProc(["Siamo aperti dalle nove alle diciannove."])
    q = QwenLLM(proc, "s", "sys", [], {}, "it")
    q.submit(1, [{"role": "user", "content": "Che orari fate?"}])
    evs = q.poll()
    assert [e["ev"] for e in evs][-1] == "done" and len(proc.sent) == 1 and q.forced == 0


CALL2 = ("<tool_call>\n<function=search_catalog>\n<parameter=query>\ndivano\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_call>\n<function=find_customer>\n<parameter=name>\nMario Rossi\n</parameter>\n</function>\n"
         "</tool_call>")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_qwen_slow_tool_never_blocks_poll_and_reports_the_wait():
    gate, clock = threading.Event(), Clock()
    proc = FakeQwenProc([CALL, "Trovato."])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (gate.wait(5), json.dumps({"ok": 1}))[1]}, "it",
                first_wait_s=1.5, wait_tick_s=1.0, tool_timeout_s=10.0, clock=clock)
    q.submit(1, [{"role": "user", "content": "Mario Rossi"}])
    t0 = time.monotonic()
    evs = q.poll()
    assert time.monotonic() - t0 < 0.5                     # the backend is blocked, poll() is not
    assert [e["ev"] for e in evs] == ["decided", "tool_start"] and len(proc.sent) == 1
    assert evs[1]["name"] == "find_customer" and evs[1]["call_id"] == "call1"
    waits = []
    for t in (1.0, 1.6, 2.0, 2.7, 3.1, 4.0):
        clock.t = t
        waits += [e for e in q.poll() if e["ev"] == "waiting"]
    assert [w["s"] for w in waits] == [1.6, 2.7, 4.0] and waits[0]["names"] == ["find_customer"]
    gate.set()
    evs = drain(q)
    tool = next(e for e in evs if e["ev"] == "tool")
    assert tool["status"] == "completed" and tool["result"] == '{"ok": 1}' and tool["call_id"] == "call1"
    assert proc.sent[1]["messages"][-1] == {"role": "tool", "content": '{"ok": 1}' + "\n(Answer the caller in Italian.)"}


def test_qwen_tool_past_the_deadline_is_stopped_and_the_caller_hears_a_fixed_apology():
    gate, clock = threading.Event(), Clock()
    proc = FakeQwenProc([CALL])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (gate.wait(5), "late")[1]}, "it",
                tool_timeout_s=10.0, first_wait_s=0, clock=clock)
    q.submit(1, [{"role": "user", "content": "Mario Rossi"}])
    q.poll()
    clock.t = 10.5
    evs = drain(q)
    tool = next(e for e in evs if e["ev"] == "tool")
    assert tool["status"] == "timed_out" and tool["result"] == TIMEOUT_PAYLOAD
    said = "".join(e["text"] for e in evs if e["ev"] == "text")
    assert said == TOOL_FAILED["it"] and evs[-1]["info"]["tool_timeout"] and q.timeouts == 1
    assert len(proc.sent) == 1                             # no extra LLM round
    gate.set()
    t_end = time.monotonic() + 2
    while q.late == 0 and time.monotonic() < t_end:
        time.sleep(0.005)
    assert q.late == 1                                     # the backend's answer arrived later: counted, not used


def test_qwen_timeout_reply_model_lets_the_llm_answer_from_the_error():
    gate, clock = threading.Event(), Clock()
    proc = FakeQwenProc([CALL, "Il sistema è lento, riprovi."])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (gate.wait(5), "late")[1]}, "it",
                tool_timeout_s=10.0, first_wait_s=0, timeout_reply="model", clock=clock)
    q.submit(1, [{"role": "user", "content": "Mario Rossi"}])
    q.poll()
    clock.t = 10.5
    evs = drain(q)
    gate.set()
    assert proc.sent[1]["messages"][-1]["content"] == TIMEOUT_PAYLOAD + "\n(Answer the caller in Italian.)"
    assert "".join(e["text"] for e in evs if e["ev"] == "text") == "Il sistema è lento, riprovi."


def test_qwen_ledger_timeout_counts_as_timed_out():
    proc = FakeQwenProc([CALL])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: TIMEOUT_PAYLOAD}, "en")   # bound handler gave up
    q.submit(1, [{"role": "user", "content": "Mario Rossi"}])
    evs = drain(q)
    assert next(e for e in evs if e["ev"] == "tool")["status"] == "timed_out"
    assert "".join(e["text"] for e in evs if e["ev"] == "text") == TOOL_FAILED["en"]


def test_qwen_runs_the_calls_of_a_round_in_parallel_and_answers_in_call_order():
    barrier = threading.Barrier(2, timeout=2)              # each backend waits for the other: only passes if parallel
    def slow(name):
        return lambda a: (barrier.wait(), json.dumps({"tool": name}))[1]
    proc = FakeQwenProc([CALL2, "Ecco."])
    q = QwenLLM(proc, "s", "sys", [], {"search_catalog": slow("search"), "find_customer": slow("customer")}, "it")
    q.submit(1, [{"role": "user", "content": "divano, Mario Rossi"}])
    evs = drain(q)
    assert [e["name"] for e in evs if e["ev"] == "tool"] == ["search_catalog", "find_customer"]
    assert [m["content"] for m in proc.sent[1]["messages"][-2:]] == ['{"tool": "search"}', '{"tool": "customer"}' + "\n(Answer the caller in Italian.)"]   # reminder on the last one


def test_qwen_cancel_while_a_tool_runs_drops_its_result():
    gate = threading.Event()
    proc = FakeQwenProc([CALL, "never"])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (gate.wait(5), "{}")[1]}, "it")
    q.submit(1, [{"role": "user", "content": "Mario Rossi"}])
    q.poll()
    q.cancel(1)
    gate.set()
    time.sleep(0.05)
    assert q.poll() == [] and len(proc.sent) == 1 and q.dropped == 1
    assert not any(m["op"] == "cancel" for m in proc.sent)   # nothing was generating: no cancel to the engine


def test_reply_reminder_names_the_session_language():
    from frankenstein2.llm.qwen import reply_reminder
    assert reply_reminder("ru") == "(Answer the caller in Russian.)"
    assert "Italian" in reply_reminder("it")
