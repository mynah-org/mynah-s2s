"""Qwen3.5 (2B / 4B, Apache-2.0) as an LLM plugin, for A/B against Nano (the small-LLM tool benchmark found Qwen3.5-4B
on par with Nano 9B on our suites at ~8 GiB bf16, Qwen3.5-2B the light option).

The model needs a newer transformers than the engine's environment, so it runs in its own process
(``qwen_server.py``, JSON lines), like the ears and the talker. This module holds the model-free parts (the
Qwen3.5 XML tool-call parser, the streaming splitter that hides call markup from speech, the round logic) and the
client plugin. Tool calls are executed engine-side through the session's bound handlers (spoken numbers -> digits
and code grounding first), then the results go back for the next round (at most ``max_rounds``).

Qwen3.5 call format: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>\n</tool_call>
STATUS: parser and splitter unit-tested; the process path is NOT VERIFIED on a GPU yet.
"""

from __future__ import annotations

import json
import re

from .base import wrap_tools

TAG = "<tool_call>"
FN = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.S)
PAR = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(raw: str) -> list:
    out = []
    for m in FN.finditer(raw):
        args = {}
        for k, v in PAR.findall(m.group(2)):
            v = v.strip()
            try:
                args[k] = json.loads(v) if re.fullmatch(r"true|false|null|-?\d+(\.\d+)?|[\[{].*", v, re.S) else v
            except ValueError:
                args[k] = v
            if isinstance(args[k], (int, float)) and not isinstance(args[k], bool) and re.fullmatch(r"0\d+", v):
                args[k] = v                       # a leading zero is a code, not a number
        out.append({"name": m.group(1), "arguments": args})
    return out


class SpeechSplitter:
    """Streaming text -> (speakable deltas, call markup). Text after ``<tool_call>`` is never spoken; a partial tag
    at the end is held back until it is decided."""

    def __init__(self):
        self.buf, self.in_call, self.call = "", False, ""

    def feed(self, delta: str) -> tuple:
        if self.in_call:
            self.call += delta
            return "", True
        self.buf += delta
        i = self.buf.find(TAG)
        if i >= 0:
            say, self.call, self.in_call, self.buf = self.buf[:i], self.buf[i:], True, ""
            return say, True
        keep = 0
        for n in range(1, len(TAG)):
            if self.buf.endswith(TAG[:n]):
                keep = n
        say = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep:]
        return say, False

    def flush(self) -> str:
        s, self.buf = ("" if self.in_call else self.buf), ""
        return s


class QwenLLM:  # pragma: no cover - needs the Qwen process
    """Client plugin. ``proc`` speaks: {"op": "gen", "ss", "rid", "messages", "tools"} -> {"ev": "delta", "ss", "rid",
    "text"}* then {"ev": "end", "ss", "rid"}; {"op": "cancel", "ss", "rid"}."""

    def __init__(self, proc, ss: str, system: str, tool_schemas: list, bound: dict, lang: str, max_rounds: int = 3,
                 user_texts=None):
        self.p, self.ss, self.system, self.schemas, self.lang = proc, ss, system, tool_schemas, lang
        self.record: list = []
        self.tools = wrap_tools(bound, lang, self.record, user_texts=user_texts)
        self.max_rounds = max_rounds
        self.jobs: dict = {}
        self.out: list = []
        self.calls: list = []

    def _gen(self, rid, msgs):
        self.p.send({"op": "gen", "ss": self.ss, "rid": rid, "tools": self.schemas,
                     "messages": [{"role": "system", "content": self.system}] + msgs})

    def submit(self, rid, messages):
        self.jobs[rid] = {"msgs": list(messages), "round": 0, "split": SpeechSplitter(), "text": "", "decided": False}
        self._gen(rid, self.jobs[rid]["msgs"])

    def cancel(self, rid):
        if self.jobs.pop(rid, None) is not None:
            self.p.send({"op": "cancel", "ss": self.ss, "rid": rid})

    def poll(self):
        out = []
        for m in self.p.poll(self.ss):
            j = self.jobs.get(m.get("rid"))
            if j is None:
                continue
            rid = m["rid"]
            if m["ev"] == "delta":
                say, in_call = j["split"].feed(m["text"])
                if say:
                    j["text"] += say
                    out.append({"ev": "text", "rid": rid, "text": say})
                if in_call and not j["decided"]:
                    j["decided"] = True
                    out.append({"ev": "decided", "rid": rid})
            elif m["ev"] == "end":
                sp = j["split"]
                calls = parse_calls(sp.call) if sp.in_call else []
                if calls and j["round"] < self.max_rounds:
                    j["msgs"].append({"role": "assistant", "content": sp.call})
                    for c in calls:
                        res = self.tools[c["name"]](c["arguments"]) if c["name"] in self.tools else \
                            json.dumps({"error": f"unknown tool {c['name']}"})
                        rec = self.record[-1] if c["name"] in self.tools else {"args": c["arguments"], "fixes": []}
                        ev = {"ev": "tool", "rid": rid, "name": c["name"], "args": rec["args"],
                              "args_raw": c["arguments"], "fixes": rec.get("fixes"), "result": res,
                              "call_id": f"call{len(self.calls) + 1}"}
                        self.calls.append(ev)
                        out.append(ev)
                        j["msgs"].append({"role": "tool", "content": res})
                    j["round"] += 1
                    j["split"] = SpeechSplitter()
                    self._gen(rid, j["msgs"])
                else:
                    rest = sp.flush()
                    if rest:
                        j["text"] += rest
                        out.append({"ev": "text", "rid": rid, "text": rest})
                    out.append({"ev": "done", "rid": rid, "text": j["text"], "info": {"rounds": j["round"]}})
                    self.jobs.pop(rid, None)
        return out

    def close(self):
        for rid in list(self.jobs):
            self.cancel(rid)
