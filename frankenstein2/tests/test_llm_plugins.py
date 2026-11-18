"""LLM plugin pieces: Qwen3.5 call parser + speech splitter + round logic (fake process), code grounding."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.llm.qwen import QwenLLM, SpeechSplitter, parse_calls  # noqa: E402
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


def test_qwen_round_trip_runs_tool_and_resumes():
    seen = []
    proc = FakeQwenProc([CALL, "Trovato, signor Rossi."])
    q = QwenLLM(proc, "s", "sys", [], {"find_customer": lambda a: (seen.append(a), json.dumps({"ok": 1}))[1]}, "it",
                user_texts=lambda: ["il mio cap è uno zero uno due uno"])
    q.submit(1, [{"role": "user", "content": "Mario Rossi, cap uno zero uno due uno"}])
    evs = q.poll() + q.poll()
    kinds = [e["ev"] for e in evs]
    assert kinds[0] == "decided" and "tool" in kinds and kinds[-1] == "done"
    assert seen == [{"name": "Mario Rossi", "postcode": "10121"}]
    assert "".join(e["text"] for e in evs if e["ev"] == "text") == "Trovato, signor Rossi."
    assert proc.sent[1]["messages"][-1] == {"role": "tool", "content": '{"ok": 1}'}
