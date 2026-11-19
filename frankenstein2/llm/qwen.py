"""Qwen3.5 (2B / 4B, Apache-2.0) as an LLM plugin, for A/B against Nano (the small-LLM tool benchmark found Qwen3.5-4B
on par with Nano 9B on our suites at ~8 GiB bf16, Qwen3.5-2B the light option).

The model needs a newer transformers than the engine's environment, so it runs in its own process
(``qwen_server.py``, JSON lines), like the ears and the talker. This module holds the model-free parts (the
Qwen3.5 XML tool-call parser, the streaming splitter that hides call markup from speech, the round logic) and the
client plugin. Tool calls are executed engine-side through the session's bound handlers (spoken numbers -> digits
and code grounding first), then the results go back for the next round (at most ``max_rounds``).

Qwen3.5 call format: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>\n</tool_call>
Tool calls run asynchronously on a thread pool (see ``QwenLLM``).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .base import wrap_tools

# Tool backends run here, never on the engine's scheduler thread (which drives the ears, the talker and the LLM of
# every session): a slow backend only delays its own session's reply. Shared by all sessions of the process.
_POOL = None


def tool_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="f2-tool")
    return _POOL


# Reference shapes (VoiceChat engine, mynah_v2.agent.tools): a call still running at the deadline gets this error.
TIMEOUT_PAYLOAD = json.dumps({"error": "timeout", "message": "I'm sorry, that's taking longer than expected."})

# Said to the caller when every call of a round timed out (timeout_reply="canned", the default).
TOOL_FAILED = {
    "it": "Mi dispiace, non sono riuscito a recuperare l'informazione in tempo. Può riprovare tra poco?",
    "fr": "Je suis désolé, je n'ai pas pu récupérer l'information à temps. Pouvez-vous réessayer dans un instant ?",
    "en": "I'm sorry, I couldn't retrieve that information in time. Could you try again in a moment?",
    "de": "Es tut mir leid, ich konnte die Information nicht rechtzeitig abrufen. Versuchen Sie es bitte gleich noch "
          "einmal.",
    "es": "Lo siento, no he podido obtener la información a tiempo. ¿Puede volver a intentarlo en un momento?",
}


def _is_timeout(res) -> bool:
    try:
        return isinstance(res, str) and json.loads(res).get("error") == "timeout"
    except (ValueError, AttributeError):
        return False

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
                # numerals stay strings (postcodes, ids): the backends and the digit grounding work on strings
                args[k] = json.loads(v) if re.fullmatch(r"true|false|null|[\[{].*", v, re.S) else v
            except ValueError:
                args[k] = v
            if isinstance(args[k], (int, float)) and not isinstance(args[k], bool) and re.fullmatch(r"0\d+", v):
                args[k] = v                       # a leading zero is a code, not a number
        out.append({"name": m.group(1), "arguments": args})
    return out


# Tool-call forcing: a turn that ends saying it is searching / checking, without a call, is continued in the same
# assistant turn with "<tool_call>" already written, so the model has to complete the call it announced (the MoE
# narrated "Cerco il divano ..." and stopped in the f2c_025 gate, despite the tool rule in the system prompt).
NARRATES = {
    "it": re.compile(r"\b(cerco|controllo|verifico|sto cercando|sto controllando|vado a (cercare|controllare|vedere)|"
                     r"lasci[ao] che (controlli|verifichi|cerchi)|un (attimo|momento)|attend[ai])\b", re.I),
    "fr": re.compile(r"\b(je (vais )?(vérifie|vérifier|cherche|chercher|regarde|regarder|consulte)|un instant|"
                     r"un moment|je recherche)\b", re.I),
    "en": re.compile(r"\b(let me (check|look|search|find)|one moment|i('ll| will) (check|look|search)|i'm (checking|"
                     r"searching|looking)|checking|searching|looking (it |that )?up)\b", re.I),
}
FORCE_PREFIX = "\n<tool_call>\n"


def narrates_lookup(text: str, lang: str) -> bool:
    return bool(NARRATES.get(lang, NARRATES["en"]).search(text or ""))


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


class QwenLLM:
    """Client plugin. ``proc`` speaks: {"op": "gen", "ss", "rid", "messages", "tools"} -> {"ev": "delta", "ss", "rid",
    "text"}* then {"ev": "end", "ss", "rid"}; {"op": "cancel", "ss", "rid"}.

    Tool calls are asynchronous (as in the VoiceChat engine): the calls of a round start together on the tool pool
    (``tool_start`` event, with the call id) and ``poll()`` never waits for them; nothing sleeps. While they run the
    request is parked: from ``first_wait_s`` a ``waiting`` event goes out every ``wait_tick_s``, and the controller
    decides whether the silence deserves an on-hold reminder (it also keeps the reply timer alive). At
    ``tool_timeout_s`` a call still running gets the reference timeout error (its later result is counted as late,
    never used). If every call of the round timed out, ``timeout_reply="canned"`` (default) ends the reply with
    ``TOOL_FAILED`` in the session language, without another LLM round; ``"model"`` lets the model answer from the
    timeout error. A cancelled request (barge-in) drops its pending results. Tool messages go back in call order."""

    def __init__(self, proc, ss: str, system: str, tool_schemas: list, bound: dict, lang: str, max_rounds: int = 3,
                 user_texts=None, tool_timeout_s: float = 10.0, first_wait_s: float = 1.5, wait_tick_s: float = 1.0,
                 timeout_reply: str = "canned", pool=None, clock=time.monotonic):
        self.p, self.ss, self.system, self.schemas, self.lang = proc, ss, system, tool_schemas, lang
        self.record: list = []
        self.last_msgs: list = []
        self.tools = wrap_tools(bound, lang, self.record, user_texts=user_texts or (
            lambda: [m["content"] for m in self.last_msgs if m.get("role") == "user"]))
        self.max_rounds = max_rounds
        self.force_calls = True
        self.forced = 0
        self.jobs: dict = {}
        self.out: list = []
        self.calls: list = []
        self.tool_timeout_s, self.first_wait_s, self.wait_tick_s = tool_timeout_s, first_wait_s, wait_tick_s
        if timeout_reply not in ("canned", "model"):
            raise ValueError(f"timeout_reply must be canned | model, not {timeout_reply!r}")
        self.timeout_reply = timeout_reply
        self.n_started = 0
        self.timeouts = 0                      # requests closed with the canned "could not retrieve" reply
        self.pool, self.clock = pool, clock
        self.ledger = None                     # the session's ToolLedger when bound (open calls cancelled on close)
        self.late = 0                          # results that arrived after their deadline or their request's cancel
        self.dropped = 0                       # calls whose request was cancelled while they ran

    def _gen(self, rid, msgs, assistant_prefix=None):
        m = {"op": "gen", "ss": self.ss, "rid": rid, "tools": self.schemas,
             "messages": [{"role": "system", "content": self.system}] + msgs}
        if assistant_prefix:
            m["assistant_prefix"] = assistant_prefix
        self.p.send(m)

    @staticmethod
    def call_markup(name: str, args: dict) -> str:
        """A tool call in the history, in Qwen3.5's own format."""
        ps = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n"
                     for k, v in (args or {}).items())
        return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"

    def submit(self, rid, messages):
        self.last_msgs = list(messages)
        self.jobs[rid] = {"msgs": list(messages), "round": 0, "split": SpeechSplitter(), "text": "", "decided": False}
        self._gen(rid, self.jobs[rid]["msgs"])

    def cancel(self, rid):
        j = self.jobs.pop(rid, None)
        if j is None:
            return
        w = j.get("wait")
        if w is not None:                       # parked on tools: nothing is generating, the results are dropped
            for c in w["calls"]:
                if not c["fut"].done():
                    self.dropped += 1
                    c["fut"].add_done_callback(lambda f: self._count_late())
        else:
            self.p.send({"op": "cancel", "ss": self.ss, "rid": rid})

    def _count_late(self):
        self.late += 1

    def _start_tools(self, rid, j, calls, out):
        pool = self.pool or tool_pool()
        w = {"t0": self.clock(), "calls": [], "next_wait": self.first_wait_s}
        for c in calls:
            box: list = []
            x = {"c": c, "rec": box, "call_id": f"call{self.n_started + 1}", "t_done": None}
            self.n_started += 1
            if c["name"] in self.tools:
                fut = pool.submit(self.tools[c["name"]], c["arguments"], box)
            else:
                fut = pool.submit(lambda n=c["name"]: json.dumps({"error": f"unknown tool {n}"}))
            fut.add_done_callback(lambda f, x=x: x.__setitem__("t_done", self.clock()))
            x["fut"] = fut
            w["calls"].append(x)
            out.append({"ev": "tool_start", "rid": rid, "name": c["name"], "args": c["arguments"],
                        "call_id": x["call_id"]})
        j["wait"] = w

    def _check_tools(self, rid, j, out):
        w = j["wait"]
        el = self.clock() - w["t0"]
        if not all(c["fut"].done() for c in w["calls"]) and el < self.tool_timeout_s:
            if self.first_wait_s and el >= w["next_wait"]:          # the controller decides whether to speak
                w["next_wait"] = el + self.wait_tick_s
                out.append({"ev": "waiting", "rid": rid, "s": round(el, 2),
                            "names": [c["c"]["name"] for c in w["calls"] if not c["fut"].done()]})
            return
        del j["wait"]
        statuses = []
        for x in w["calls"]:
            c, fut = x["c"], x["fut"]
            if not fut.done():
                res, status = TIMEOUT_PAYLOAD, "timed_out"
                fut.add_done_callback(lambda f: self._count_late())
            elif fut.exception() is not None:
                res, status = json.dumps({"error": str(fut.exception())}), "failed"
            else:
                res = fut.result()
                status = "timed_out" if _is_timeout(res) else "completed"   # the ledger's own deadline
            statuses.append(status)
            rec = x["rec"][0] if x["rec"] else {"args": c["arguments"], "fixes": []}
            t_end = x["t_done"] if (x["t_done"] is not None and status != "timed_out") else self.clock()
            ev = {"ev": "tool", "rid": rid, "name": c["name"], "args": rec["args"], "args_raw": c["arguments"],
                  "fixes": rec.get("fixes"), "result": res, "status": status,
                  "ms": round((t_end - w["t0"]) * 1000, 1), "call_id": x["call_id"]}
            self.calls.append(ev)
            out.append(ev)
            j["msgs"].append({"role": "tool", "content": res})
        if self.timeout_reply == "canned" and all(st == "timed_out" for st in statuses):
            # nothing came back in time: say so directly, no extra LLM round (it would only add latency)
            msg = TOOL_FAILED.get(self.lang, TOOL_FAILED["en"])
            j["text"] += msg
            self.timeouts += 1
            out.append({"ev": "text", "rid": rid, "text": msg})
            out.append({"ev": "done", "rid": rid, "text": j["text"],
                        "info": {"rounds": j["round"] + 1, "tool_timeout": True}})
            self.jobs.pop(rid, None)
            return
        j["round"] += 1
        j["split"] = SpeechSplitter()
        self._gen(rid, j["msgs"])

    def poll(self):
        out = []
        for rid, j in list(self.jobs.items()):
            if "wait" in j:
                self._check_tools(rid, j, out)
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
                if (not calls and self.force_calls and not j.get("forced") and j["round"] < self.max_rounds
                        and not m.get("cancelled") and narrates_lookup(j["text"] + sp.buf, self.lang)):
                    j["forced"] = True                  # once per request: continue the turn with the call opened
                    self.forced += 1
                    rest = sp.flush()
                    if rest:
                        j["text"] += rest
                        out.append({"ev": "text", "rid": rid, "text": rest})
                    j["said_before_call"] = j["text"]
                    sp2 = SpeechSplitter()
                    sp2.in_call, sp2.call = True, FORCE_PREFIX.lstrip("\n")
                    j["split"] = sp2
                    if not j["decided"]:
                        j["decided"] = True
                        out.append({"ev": "decided", "rid": rid})
                    self._gen(rid, j["msgs"] + [], assistant_prefix=j["text"] + FORCE_PREFIX)
                    continue
                if calls and j["round"] < self.max_rounds:
                    j["msgs"].append({"role": "assistant", "content": (j.pop("said_before_call", "") + "\n" + sp.call)
                                      .lstrip("\n")})
                    self._start_tools(rid, j, calls, out)
                    self._check_tools(rid, j, out)      # fast backends: the round goes on in this same poll
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
        if self.ledger is not None:             # open calls -> cancelled; a later result is recorded as late
            self.ledger.cancel_open("session_closed")
        if getattr(self.p, "releases_slots", False):       # the in-process engine frees the session's slot
            self.p.send({"op": "close", "ss": self.ss})
