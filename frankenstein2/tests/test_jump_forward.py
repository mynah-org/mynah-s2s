"""Tool-call jump-forward in the in-process LLM: a fake engine "generates" a fixed call one character per token;
the forced markup is fed through extend() and the stream the client sees is the same call. torch (CPU) needed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")

from frankenstein2.runtime import InProcLLM  # noqa: E402

CALL = ("<tool_call>\n<function=find_customer>\n<parameter=full_name>\nMario Rossi\n</parameter>\n"
        "<parameter=postcode>\n10121\n</parameter>\n</function>\n</tool_call>")
V = 256


class Tok:
    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class Eng:
    """Every forward predicts the next character of CALL after what the session has fed so far."""

    def __init__(self):
        self.tok, self.eos_ids, self.sess, self.fed, self.extends, self.steps = Tok(), [0], {}, {}, 0, 0

    def _next(self, s):
        n = len(self.fed[s])
        lg = torch.zeros(V)
        lg[ord(CALL[n]) if n < len(CALL) else 0] = 1.0
        return lg

    def open(self):
        self.fed[0] = []
        return 0

    def chat_ids(self, messages, tools=None):
        return [1]

    def prefill_begin(self, s, ids):
        return 1

    def prefill_step(self, s, chunk):
        return self._next(s)

    def decode_batch(self, sessions, tokens):
        self.steps += 1
        for s, t in zip(sessions, tokens):
            self.fed[s].append(t)
        return torch.stack([self._next(s) for s in sessions])

    def extend(self, s, toks):
        self.extends += 1
        self.fed[s] += toks
        return self._next(s)


class Sched:
    def submit(self, x):
        pass


def run(jf):
    eng = Eng()
    llm = InProcLLM(Sched(), eng)
    llm.jump_forward = jf                     # off by default in the engine; the mechanism stays tested
    llm.control({"op": "gen", "ss": "s", "rid": 1, "messages": [{"role": "user", "content": "x"}]})
    for _ in range(400):
        if not llm._run_one():
            break
    evs = llm.poll("s")
    text = "".join(e["text"] for e in evs if e["ev"] == "delta")
    return text, eng, evs


def test_jump_forward_gives_the_same_call_in_fewer_steps():
    plain, e0, _ = run(False)
    fast, e1, evs = run(True)
    assert plain == CALL and fast == CALL
    assert e1.extends >= 3 and e1.steps < e0.steps - 20          # characters here: markup spans are long
    assert evs[-1]["ev"] == "end"
