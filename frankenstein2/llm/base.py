"""LLM plugin interface (text in, text + tool calls out) and a scripted fake for tests.

A plugin instance serves ONE session (its own context, tools, ledger and language); several instances may share one
loaded model. Non-blocking:

  submit(rid, messages)   a reply for the conversation so far (list of chat messages, system prompt excluded)
  cancel(rid)             stop that request as soon as possible (its later events are stale anyway)
  poll() -> [events]      {"ev": "text", "rid", "text"}            reply text delta (spoken)
                          {"ev": "decided", "rid"}                  the tool step chose to call a tool (holding phrase)
                          {"ev": "waiting", "rid", "s", "names"}    a tool is still running after s seconds
                                                                    (the controller may say an on-hold reminder)
                          {"ev": "tool", "rid", "name", "args", "args_raw", "result", "call_id"}   a tool RAN
                          {"ev": "done", "rid", "text", "info"}     the reply is complete
  close()

Tools are executed by the plugin's worker through the session's bound handlers (V2 ToolRegistry / ToolLedger); the
arguments pass through the spoken-numbers -> digits step (text/spoken_numbers.py) in the session language first.
"""

from __future__ import annotations

from typing import Protocol


class LLMPlugin(Protocol):
    def submit(self, rid: int, messages: list) -> None: ...
    def cancel(self, rid: int) -> None: ...
    def poll(self) -> list: ...
    def close(self) -> None: ...


def wrap_tools(bound: dict, lang: str, record: list, user_texts=None) -> dict:
    """Handlers that normalise spoken numbers in the arguments first and ground code-like values on what the caller
    said (``user_texts() -> [str]``); ``record`` gets {name, args_raw, args, fixes}. ``h(args, rec_out=list)`` also
    hands that record to the caller, which is what concurrent calls use (``record[-1]`` may be another call's)."""
    from frankenstein2.text.spoken_numbers import ground_codes, normalize_args

    def mk(name, fn):
        def h(args, rec_out=None):
            norm = normalize_args(args if isinstance(args, dict) else {}, lang)
            fixes = []
            if user_texts is not None:
                norm, fixes = ground_codes(norm, list(user_texts()), lang)
            rec = {"name": name, "args_raw": args, "args": norm, "fixes": fixes}
            record.append(rec)
            if rec_out is not None:
                rec_out.append(rec)
            return fn(norm)
        return h
    return {n: mk(n, f) for n, f in bound.items()}


class ScriptedLLM:
    """Deterministic fake: ``script(messages) -> list of steps``; a step is ("text", str) | ("decided",) |
    ("tool", name, args) | ("wait", n_polls) | ("waiting", seconds, *names) (a slow tool). Steps are released one per poll, so tests control the timing."""

    def __init__(self, script, tools: dict | None = None, lang: str = "en"):
        self.script, self.lang = script, lang
        self.record: list = []
        self.tools = wrap_tools(tools or {}, lang, self.record,
                                user_texts=lambda: [m["content"] for r, ms in self.submitted[-1:] for m in ms
                                                    if m["role"] == "user"])
        self.jobs: list = []
        self.cancelled: set = set()
        self.submitted: list = []
        self.calls: list = []

    def submit(self, rid, messages):
        self.submitted.append((rid, [dict(m) for m in messages]))
        self.jobs.append([rid, list(self.script(messages)), ""])

    def cancel(self, rid):
        self.cancelled.add(rid)

    def poll(self):
        out = []
        if not self.jobs:
            return out
        job = self.jobs[0]
        rid, steps, text = job
        if rid in self.cancelled:
            self.jobs.pop(0)
            return out
        if not steps:
            self.jobs.pop(0)
            return [{"ev": "done", "rid": rid, "text": text, "info": {}}]
        st = steps.pop(0)
        if st[0] == "text":
            job[2] += st[1]
            out.append({"ev": "text", "rid": rid, "text": st[1]})
        elif st[0] == "decided":
            out.append({"ev": "decided", "rid": rid})
        elif st[0] == "tool":
            name, args = st[1], st[2]
            res = self.tools[name](args) if name in self.tools else '{"error": "unknown tool"}'
            rec = self.record[-1] if self.record and name in self.tools else {"args": args}
            self.calls.append({"name": name, "args": rec["args"], "result": res})
            out.append({"ev": "tool", "rid": rid, "name": name, "args": rec["args"], "args_raw": args,
                        "result": res, "call_id": f"c{len(self.calls)}"})
        elif st[0] == "waiting":
            out.append({"ev": "waiting", "rid": rid, "s": st[1], "names": list(st[2:])})
        elif st[0] == "wait" and st[1] > 1:
            steps.insert(0, ("wait", st[1] - 1))
        return out

    def close(self):
        self.jobs = []
