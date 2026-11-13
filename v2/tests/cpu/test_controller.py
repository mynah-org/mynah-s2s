import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_session import CS, FakeEngine  # noqa: E402

from mynah_v2.agent.controller import Controller, FactsMemory  # noqa: E402
from mynah_v2.agent.tools import ToolLedger  # noqa: E402
from mynah_v2.runtime.session import EnhancedOptions, Mode, Session  # noqa: E402
from mynah_v2.tracing.events import Tracer  # noqa: E402


def test_only_completed_non_error_results_become_facts(tmp_path):
    mem = FactsMemory(tmp_path / "f.sqlite")
    led = ToolLedger("s")
    led.on_close.append(mem.add_from_call)
    a = led.open("get_order_status", {"order_id": "XYZ123"}, 1)
    led.complete(a.call_id, json.dumps({"status": "shipped"}))
    b = led.open("get_order_status", {"order_id": "QQQ999"}, 1)
    led.complete(b.call_id, json.dumps({"error": "timeout"}))
    c = led.open("get_order_status", {"order_id": "ABC456"}, 2)
    led.bump_epoch()                       # c becomes stale -> never a fact
    facts = mem.all()
    assert [f.args["order_id"] for f in facts] == ["XYZ123"]
    assert FactsMemory(tmp_path / "f.sqlite").all()[0].result == {"status": "shipped"}  # persisted


def test_latest_result_supersedes_older_for_same_key():
    mem = FactsMemory()
    led = ToolLedger("s")
    led.on_close.append(mem.add_from_call)
    for st in ("processing", "shipped"):
        x = led.open("get_order_status", {"order_id": "XYZ123"}, 1)
        led.complete(x.call_id, json.dumps({"status": st}))
    (f,) = mem.latest_by_key()
    assert f.result == {"status": "shipped"}


def test_summary_and_prompt_fn():
    ctl = Controller("BASE", prompt_fn=lambda extra: f"[{extra}]")
    led = ToolLedger("s")
    led.on_close.append(ctl.on_tool_call_closed)
    x = led.open("get_order_status", {"order_id": "XYZ123"}, 1)
    led.complete(x.call_id, json.dumps({"status": "shipped", "delivery_date": "November 12, 2026"}))
    ctl.on_user_text("where is my order")
    ctl.on_user_text(" XYZ123")
    ctl.on_agent_text("It shipped.")
    s = ctl.summary()
    assert "November 12, 2026" in s and "user: where is my order XYZ123" in s and "agent: It shipped." in s
    p = ctl.rehydration_prompt()
    assert p.startswith("[This conversation is continuing.") and ctl.rehydrations == 1
    assert Controller("BASE").rehydration_prompt() == "BASE"


def test_session_rehydrates_once_threshold_reached_and_reference_ignores_controller():
    eng = FakeEngine(eos_at=None)
    opened = []
    eng.open = lambda p: opened.append(p)
    ctl = Controller("BASE")
    tr = Tracer(None)
    s = Session(eng, tr, Mode.ENHANCED, EnhancedOptions(rehydrate_after_s=0.48), controller=ctl)
    s.open("BASE")
    for _ in range(8):
        s.feed_chunk(torch.zeros(CS))
    names = [e.name for e in tr.events]
    # 0.48 s = 3 chunks per stream; first chunk starts the agent turn (fake BOS), EOS never comes in the fake,
    # so rehydration waits for idle -> none here
    assert names.count("session.rehydrated") == 0
    s._agent_speaking = False
    s.feed_chunk(torch.zeros(CS))
    assert names.count("session.rehydrated") == 0 and [e.name for e in tr.events].count("session.rehydrated") == 1
    assert len(opened) == 2 and s.context_chunks == 1
    ref_eng = FakeEngine(eos_at=None)
    ref_opened = []
    ref_eng.open = lambda p: ref_opened.append(p)
    ref = Session(ref_eng, Tracer(None), Mode.REFERENCE, EnhancedOptions(rehydrate_after_s=0.48), controller=ctl)
    ref.open("BASE")
    ref._agent_speaking = False
    for _ in range(8):
        ref.feed_chunk(torch.zeros(CS))
    assert ref_opened == ["BASE"]                   # REFERENCE: the controller observes, never restarts the model


def test_summary_strips_turn_markers_from_the_text_channel():
    c = Controller("BASE")
    c.on_agent_text("</s><s>Your order has shipped.</s>")
    c.on_user_text("thanks")
    s = c.summary()
    assert "<s>" not in s and "</s>" not in s and "agent: Your order has shipped." in s


def test_history_is_aggregated_per_turn_despite_interleaved_deltas():
    """GPU E4: the refresh summary read 'user: I'd / agent: today? / user: like to' (deltas interleaved)."""
    c = Controller("BASE")
    for role, t in [("agent", "<s>Hi! How can I help"), ("user", "I'd"), ("agent", " you today?</s>"),
                    ("user", " like to know"), ("user", " where my order is"), ("agent", "<s>Could you"),
                    ("user", " the"), ("agent", " give me your order ID?</s>")]:
        (c.on_agent_text if role == "agent" else c.on_user_text)(t)
    lines = c.summary().split("\n")[1:]
    assert lines == ["agent: Hi! How can I help you today?", "user: I'd like to know where my order is",
                     "agent: Could you give me your order ID?", "user: the"]
