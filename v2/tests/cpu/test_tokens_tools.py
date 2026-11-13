import json
import threading

import pytest

from mynah_v2.agent.tools import (CallStatus, InvalidTransition, ToolLedger, ToolRegistry, make_orders_db,
                                  normalize_order_id, orders_tool, ORDERS_SCHEMA)
from mynah_v2.model import tokens as T
from mynah_v2.tracing.events import Tracer


def test_token_constants_and_fusion_weights():
    assert (T.PAD_ID, T.SOTC_ID, T.EOTC_ID, T.EOTR_ID) == (12, 20, 21, 22)
    assert T.FUSION_WEIGHTS == {"user": 1.0, "text": 1.0, "function": 2.0}
    assert T.SAMPLES_PER_FRAME_IN == 1280 and T.SAMPLES_PER_FRAME_OUT == 1764


def test_format_tool_response_matches_reference_offline_format():
    payload = {"order_id": "ABC456", "status": "shipped"}
    s = T.format_tool_response(payload)
    assert s == '<TOOL_RESPONSE>[{"order_id": "ABC456", "status": "shipped"}]</TOOL_RESPONSE>'


@pytest.mark.parametrize("raw,name,args", [
    ('<TOOLCALL>[{"name": "get_order_status", "arguments": {"order_id": "ABC456"}}]</TOOLCALL>',
     "get_order_status", {"order_id": "ABC456"}),
    ('<SPECIAL_20><TOOLCALL>{"name": "get-weather", "parameters": {"city": "Rome"}}</TOOLCALL><SPECIAL_21>',
     "get_weather", {"city": "Rome"}),
    ('[{"name": "f", "arguments": "{\\"x\\": 1}"}]', "f", {"x": 1}),
])
def test_parse_call_text(raw, name, args):
    (c,) = T.parse_call_text(raw)
    assert c.ok and c.name == name and c.arguments == args


def test_parse_call_text_bad_json():
    (c,) = T.parse_call_text("<TOOLCALL>[{name: oops</TOOLCALL>")
    assert not c.ok and c.error.startswith("json")


def test_ledger_lifecycle_and_trace():
    tr = Tracer(None, session_id="s1")
    led = ToolLedger("s1", tr)
    c = led.open("get_order_status", {"order_id": "ABC456"}, turn_id=1, model_step=40)
    assert c.status == CallStatus.PENDING and c.call_id == "s1-c1"
    led.complete(c.call_id, "{}")
    assert c.status == CallStatus.COMPLETED
    with pytest.raises(InvalidTransition):
        led.fail(c.call_id, "late")
    assert [e.name for e in tr.events] == ["tool.pending", "tool.completed"]
    assert tr.events[0].fields["model_step"] == 40


def test_epoch_fence_marks_pending_stale_and_late_result_stale():
    led = ToolLedger("s")
    a = led.open("t", {}, 1)
    led.bump_epoch()
    assert a.status == CallStatus.STALE
    b = led.open("t", {}, 2)
    assert b.epoch == 1
    led.complete(b.call_id, "ok")
    assert b.status == CallStatus.COMPLETED


def test_result_from_older_epoch_is_stale_not_completed():
    led = ToolLedger("s")
    a = led.open("t", {}, 1)
    led.epoch += 1  # fence moved without touching the entry (e.g. concurrent correction)
    led.complete(a.call_id, "ok")
    assert a.status == CallStatus.STALE and a.error == "epoch fence"


def test_registry_bind_wraps_with_ledger_and_reference_signature(tmp_path):
    db = make_orders_db(tmp_path / "o.sqlite")
    reg = ToolRegistry()
    reg.register("get_order_status", orders_tool(db), "Look up an order", ORDERS_SCHEMA, ack_message="Checking.")
    led = ToolLedger("s")
    h = reg.bind(led, turn_id_fn=lambda: 3, step_fn=lambda: 77)["get_order_status"]
    out = json.loads(h({"order_id": "xyz-123"}))
    assert out == {"order_id": "XYZ123", "found": True, "status": "shipped", "delivery_date": "November 12, 2026",
                   "tracking_code": "7319"}
    (c,) = led.calls.values()
    assert c.turn_id == 3 and c.model_step == 77 and c.status == CallStatus.COMPLETED
    assert json.loads(h({"order_id": "nope"}))["found"] is False
    assert reg.schemas[0]["function"]["name"] == "get_order_status"


def test_registry_exception_becomes_error_payload():
    reg = ToolRegistry()

    def boom(_):
        raise ValueError("db down")
    reg.register("x", boom, "d", {"type": "object", "properties": {}})
    led = ToolLedger("s")
    out = reg.bind(led, lambda: 0, lambda: None)["x"]({})
    assert json.loads(out) == {"error": "db down"}
    assert next(iter(led.calls.values())).status == CallStatus.FAILED


def test_ledger_thread_safety():
    led = ToolLedger("s")

    def work():
        for _ in range(200):
            c = led.open("t", {}, 0)
            led.complete(c.call_id, "r")
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(led.calls) == 800 and len({c.call_id for c in led.calls.values()}) == 800


def test_normalize_order_id():
    assert normalize_order_id(" ab c-45 6 ") == "ABC456"


def test_lean_checkpoint_loading_keeps_exactly_the_wrapper_subsets(tmp_path):
    import safetensors.torch as st
    import torch
    from mynah_v2.model.reference import lean_checkpoint_loading
    sd = {"stt_model.llm.w": torch.ones(2), "tts_model.a": torch.zeros(3), "speech_generation.b": torch.ones(1),
          "stt_model.perception.c": torch.ones(4)}
    st.save_file(sd, str(tmp_path / "model.safetensors"))
    other = tmp_path / "x.safetensors"
    st.save_file({"k": torch.ones(1)}, str(other))
    with lean_checkpoint_loading(str(tmp_path)) as lean:
        a = st.load_file(str(tmp_path / "model.safetensors"))
        o = st.load_file(str(other))
        b = st.load_file(str(tmp_path / "model.safetensors"))
        c = st.load_file(str(tmp_path / "model.safetensors"))
    assert set(a) == {"stt_model.llm.w", "stt_model.perception.c"}      # wrapper call 1 keeps non-TTS keys
    assert set(b) == {"tts_model.a"}                                     # wrapper call 2 keeps tts_model.* only
    assert set(c) == set(sd) and set(o) == {"k"}                         # anything else: original behaviour
    assert lean.loaded == [2, 1] and st.load_file is not None
    assert set(st.load_file(str(tmp_path / "model.safetensors"))) == set(sd)  # restored after exit
