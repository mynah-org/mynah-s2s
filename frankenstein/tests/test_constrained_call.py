"""E1l G2: constrained tool-call decoding, model-free (greedy longest-match tokenizer + an LM that 'wants' a text)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.constrained_call import ConstrainedCall  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "find_customer", "parameters": {
        "type": "object", "properties": {"full_name": {"type": "string"}, "postcode": {"type": "string"}},
        "required": ["full_name"]}}},
    {"type": "function", "function": {"name": "search_catalog", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "parameters": {
        "type": "object", "properties": {"reason": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "set_qty", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}, "mode": {"enum": ["add", "set"]}},
        "required": ["n", "mode"]}}},
]

# multi-character pieces, including ones that glue a closing quote to what follows (the hard case)
PIECES = ['"}}]</TOOLCALL>', '"}}', '", "', '",', '": "', '": ', '"', 'find', '_customer', 'search', '_catalog',
          ' Mario', ' Rossi', 'Mario', '101', '21', 'divano', ' a', ' tre', ' posti', 'arguments', 'full_name',
          'postcode', 'query', 'name', '{"', '}}]', '}', ', ', 'set_qty', 'add', 'escalate', '_to_human']


class Vocab:
    def __init__(self):
        self.tok = {}

    def id(self, piece):
        return self.tok.setdefault(piece, len(self.tok) + 10)

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            best = max((p for p in PIECES if text.startswith(p, i)), key=len, default=text[i])
            out.append(self.id(best))
            i += len(best)
        return out

    def decode(self, ids):
        inv = {v: k for k, v in self.tok.items()}
        return "".join(inv[i] for i in ids)


class WantLM:
    """Greedy LM whose preferred continuation is ``target`` (natural tokenization of the remaining text)."""

    def __init__(self, v: Vocab, target: str, fed: str):
        self.v, self.target, self.fed, self.calls = v, target, fed, []

    def _want(self):
        if self.target.startswith(self.fed) and len(self.fed) < len(self.target):
            return self.v.encode(self.target[len(self.fed):])[0]
        return self.v.id("}")

    def pick(self, allowed):
        w = self._want()
        if allowed is None or w in allowed:
            return w
        rest = self.target[len(self.fed):] if self.target.startswith(self.fed) else ""
        ok = [t for t in allowed if rest.startswith(self.v.decode([t]))]   # a real LM also likes shorter pieces
        return max(ok, key=lambda t: len(self.v.decode([t]))) if ok else min(allowed)

    def feed(self, ids):
        self.calls.append(len(ids))
        self.fed += self.v.decode(ids)

    def decode(self, ids):
        return self.v.decode(ids)


def call(target_args, name, tools=TOOLS):
    v = Vocab()
    free = "<TOOLCALL>[" + json.dumps({"name": name, "arguments": target_args}, ensure_ascii=False) + "]</TOOLCALL>"
    lm = WantLM(v, free, "<TOOLCALL>[")
    out, st = ConstrainedCall(tools, v.encode).run(lm)
    return free, out, st, lm


def test_e1k_find_customer_call_reproduced_with_the_optional_key():
    free, out, st, lm = call({"full_name": "Mario Rossi", "postcode": "10121"}, "find_customer")
    assert out == free
    assert lm.fed.startswith(free[: free.index("10121") + 5])          # the cache saw exactly the natural call
    assert st["generated"] < len(Vocab().encode(free))                  # fewer decoded tokens than the free call


def test_e1k_search_catalog_call_reproduced():
    free, out, st, _ = call({"query": "divano a tre posti"}, "search_catalog")
    assert out == free and st["generated"] <= 8


def test_optional_key_skipped_when_the_lm_closes():
    free, out, _, _ = call({"full_name": "Mario Rossi"}, "find_customer")
    assert out == free


def test_no_required_keys_and_name_restricted_to_the_schema():
    v = Vocab()
    lm = WantLM(v, '<TOOLCALL>[{"name": "lookup_weather", "arguments": {}}]', "<TOOLCALL>[")
    out, _ = ConstrainedCall(TOOLS, v.encode).run(lm)
    name = json.loads(out[len("<TOOLCALL>["):-len("]</TOOLCALL>")])["name"]
    assert name in {"find_customer", "search_catalog", "escalate_to_human", "set_qty"}


def test_integer_and_enum_values():
    free, out, _, _ = call({"n": 3, "mode": "set"}, "set_qty")
    assert json.loads(out[len("<TOOLCALL>["):-len("]</TOOLCALL>")]) == {"name": "set_qty",
                                                                        "arguments": {"n": 3, "mode": "set"}}


class LoneQuoteLM(WantLM):
    """E1l failure mode: after an unnatural lone '"' token the model prefers to close the object."""

    def __init__(self, v, target, fed):
        super().__init__(v, target, fed)
        self.last = None

    def feed(self, ids):
        super().feed(ids)
        self.last = ids[-1]

    def pick(self, allowed):
        if self.last == self.v.id('"') and allowed and self.v.id("}") in allowed:
            return self.v.id("}")
        return super().pick(allowed)


def test_optional_key_options_keep_the_natural_tokenization_of_the_closing_quote():
    v = Vocab()
    args = {"full_name": "Mario Rossi", "postcode": "10121"}
    free = "<TOOLCALL>[" + json.dumps({"name": "find_customer", "arguments": args}) + "]</TOOLCALL>"
    out, _ = ConstrainedCall(TOOLS, v.encode).run(LoneQuoteLM(v, free, "<TOOLCALL>["))
    assert out == free
