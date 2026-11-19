"""Controller v2 driven by synthetic ears / LLM / playout intents."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.control.controller import Controller, Intent, St  # noqa: E402


def kinds(acts):
    return [a.kind for a in acts]


def endpoint(c, k, text="ciao"):
    c.post(Intent("ENDPOINT", "ears", k, payload={"text": text}))
    return c.tick(k)


def test_endpoint_opens_turn_and_requests_reply():
    c = Controller()
    acts = endpoint(c, 5, "dov'è il mio ordine")
    assert kinds(acts) == ["OPEN_TURN", "REQUEST_REPLY"]
    assert acts[1].data["user_text"] == "dov'è il mio ordine" and acts[1].data["request_id"] == 1
    assert c.state is St.THINKING and c.ears_mode == "speaking"


def test_empty_transcript_holds():
    c = Controller()
    assert kinds(endpoint(c, 5, "  ")) == ["HOLD"]
    assert c.state is St.LISTENING


def test_full_turn_closes_only_after_played_out():
    c = Controller()
    endpoint(c, 1)
    c.post(Intent("LLM_TEXT", "llm", 2, payload={"text": "Salve."}, request_id=1))
    assert kinds(c.tick(2)) == ["SPEAK"]
    c.post(Intent("AUDIO_STARTED", "playout", 3, epoch=0))
    c.post(Intent("LLM_DONE", "llm", 3, payload={"text": "Salve."}, request_id=1))
    assert kinds(c.tick(3)) == ["END_TEXT"]
    assert c.state is St.SPEAKING
    c.post(Intent("PLAYED_OUT", "playout", 9, epoch=0))
    assert kinds(c.tick(9)) == ["CLOSE_TURN", "EARS_RESET"]
    assert c.state is St.LISTENING


def test_barge_cancels_flushes_and_resets_ears():
    c = Controller()
    endpoint(c, 1)
    c.post(Intent("LLM_TEXT", "llm", 2, payload={"text": "Il suo ordine "}, request_id=1))
    c.post(Intent("AUDIO_STARTED", "playout", 2, epoch=0))
    c.tick(2)
    c.post(Intent("BARGE", "ears", 6, payload={"onset_s": 0.4, "frame": 5}))
    acts = c.tick(6)
    assert kinds(acts) == ["CANCEL", "EARS_RESET"]
    assert acts[0].data["new_epoch"] == 1 and acts[0].data["heard_possible"] and not acts[0].data["unheard"]
    assert c.epoch == 1 and c.state is St.LISTENING
    # the old request's text and done are stale now
    c.post(Intent("LLM_TEXT", "llm", 7, payload={"text": "è partito."}, request_id=1))
    c.post(Intent("LLM_DONE", "llm", 7, request_id=1))
    assert kinds(c.tick(7)) == ["DROP_STALE", "DROP_STALE"]


def test_barge_before_anything_heard_merges_user_text():
    c = Controller()
    endpoint(c, 1, "Cerco un divano")
    acts = (c.post(Intent("BARGE", "ears", 4, payload={"onset_s": 0.2})), c.tick(4))[1]
    assert acts[0].data["unheard"] and acts[0].data["drop_user_turn"]
    acts = endpoint(c, 9, "a tre posti")
    assert acts[0].data["user_text"] == "Cerco un divano a tre posti" and acts[0].reason == "endpoint_merged"


def test_holding_phrase_once_and_not_after_speech():
    c = Controller()
    endpoint(c, 1)
    c.post(Intent("LLM_DECIDED", "llm", 2, request_id=1))
    assert kinds(c.tick(2)) == ["HOLDING_PHRASE"]
    c.post(Intent("LLM_DECIDED", "llm", 3, request_id=1))
    assert kinds(c.tick(3)) == []
    c2 = Controller()
    endpoint(c2, 1)
    c2.post(Intent("LLM_TEXT", "llm", 2, payload={"text": "Sì"}, request_id=1))
    c2.tick(2)
    c2.post(Intent("LLM_DECIDED", "llm", 3, request_id=1))
    assert "HOLDING_PHRASE" not in kinds(c2.tick(3))


def test_tool_after_cancel_is_recorded_not_dropped():
    c = Controller()
    endpoint(c, 1)
    c.post(Intent("CANCEL", "app", 2))
    c.tick(2)
    c.post(Intent("LLM_TOOL", "llm", 3, payload={"name": "find_customer"}, request_id=1))
    acts = c.tick(3)
    assert kinds(acts) == ["TOOL_RECORD"] and acts[0].data["stale"]


def test_backchannel_only_while_listening():
    c = Controller()
    c.post(Intent("BACKCHANNEL", "ears", 1))
    assert kinds(c.tick(1)) == ["BACKCHANNEL"]
    endpoint(c, 2)
    c.post(Intent("BACKCHANNEL", "ears", 3))
    assert kinds(c.tick(3)) == []


def test_reply_timeout_cancels():
    c = Controller()
    endpoint(c, 1)
    acts = []
    for k in range(2, 160):
        acts += c.tick(k)
    assert "CANCEL" in kinds(acts) and c.state is St.LISTENING


def test_order_of_arrival_does_not_matter():
    base = [Intent("LLM_TEXT", "llm", 2, payload={"text": "a"}, request_id=1),
            Intent("BARGE", "ears", 2, payload={"onset_s": 0.1}),
            Intent("LLM_DONE", "llm", 2, request_id=1),
            Intent("AUDIO_STARTED", "playout", 2, epoch=0)]
    seen = set()
    for perm in itertools.permutations(base):
        c = Controller()
        endpoint(c, 1)
        for it in perm:
            c.post(it)
        seen.add(tuple((a.kind, a.reason) for a in c.tick(2)))
    assert len(seen) == 1
    assert list(seen)[0][0] == ("CANCEL", "barge_in")


def test_tool_wait_reminds_in_silence_at_most_twice_and_keeps_the_reply_alive():
    c = Controller()
    endpoint(c, 1)
    acts = []
    for k in range(2, 260):
        if k >= 20 and k % 12 == 0:                        # a slow tool: "waiting" about every second
            silent = 0 if k < 40 else 30                   # the holding phrase is still playing until step 40
            c.post(Intent("LLM_WAIT", "llm", k, payload={"s": (k - 20) * 0.05, "silent_steps": silent},
                          request_id=1))
        acts += c.tick(k)
    rem = [a for a in acts if a.kind == "HOLDING_PHRASE"]
    assert [a.reason for a in rem] == ["tool_wait", "tool_wait"] and rem[1].data["reminder"] == 2
    assert rem[0].k == 48 and rem[1].k - rem[0].k >= 50        # first in silence, then 4 s apart
    assert "CANCEL" not in kinds(acts)        # without the waits the 12 s reply timeout fires at step 152


def test_tool_wait_after_a_barge_is_stale():
    c = Controller()
    endpoint(c, 1)
    c.post(Intent("CANCEL", "app", 2))
    c.tick(2)
    c.post(Intent("LLM_WAIT", "llm", 3, payload={"s": 4.0}, request_id=1))
    assert "HOLDING_PHRASE" not in kinds(c.tick(3))


def test_tool_wait_reminds_after_a_while_even_if_the_talker_still_trickles():
    c = Controller()
    endpoint(c, 1)
    acts = []
    for k in range(2, 120):
        if k >= 20 and k % 12 == 8:
            c.post(Intent("LLM_WAIT", "llm", k, payload={"s": (k - 20) * 0.08, "silent_steps": 0}, request_id=1))
        acts += c.tick(k)
    rem = [a for a in acts if a.kind == "HOLDING_PHRASE"]
    assert rem and rem[0].data["waited_s"] >= 2.5           # never "silent", reminded at 2.5 s into the call
