"""E1l (PERF G2): constrained decoding of a tool call. The schema forces everything except the values.

The free call step decodes the whole ``<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>`` text token by token
(27-35 tokens in E1j/E1k, mostly names, keys and punctuation). Here:
  * the tool name is chosen by the LM, restricted to the schema's names (a token trie walk: argmax over the allowed
    next ids only);
  * every required key, the punctuation and the quotes are forced (fed, never decoded);
  * after the required keys the LM chooses, again by trie, between closing the object and one of the optional keys
    still unused (schema order);
  * only the values are generated: a string value ends at the first token that contains a quote (the text before the
    quote is kept, the token itself is NOT fed: the forced continuation is re-encoded from the text), a number at the
    first token that is not part of a number, an enum / boolean is a trie walk over its allowed literals.
Forced text is accumulated and fed in one multi-token call only when the next logits are needed.

The LM is any object with ``pick(allowed: set | None) -> int`` (argmax, optionally restricted), ``feed(ids)``
(advance the cache) and ``decode(ids) -> str``; ``encode`` is the tokenizer's (no special tokens). Model-free tests:
tests/test_constrained_call.py.
"""

from __future__ import annotations

import json
import time

NUM_CHARS = set("0123456789.-+eE")


def _props(tool: dict) -> tuple:
    f = tool.get("function", tool)
    p = f.get("parameters") or {}
    props = p.get("properties") or {}
    req = [k for k in (p.get("required") or []) if k in props]
    opt = [k for k in props if k not in req]
    return f["name"], props, req, opt


class ConstrainedCall:
    def __init__(self, tools: list, encode, stop_ids=(), max_value_tokens: int = 24):
        self.tools = {_props(t)[0]: _props(t) for t in tools}
        self.encode = encode
        self.stop_ids = set(stop_ids)
        self.max_value_tokens = max_value_tokens

    # ------------------------------------------------------------------ pieces
    def _trie(self, lm, options: list, st: dict) -> str:
        """Greedy walk restricted to the token sequences of ``options`` (texts). The ids are fed as they are picked;
        once one option is left its remaining ids are returned as pending (fed with the next forced text)."""
        seqs = {o: list(self.encode(o)) for o in options}
        first = seqs[options[0]]
        i = 0
        while all(len(seqs[o]) > i and seqs[o][i] == first[i] for o in options):
            i += 1
        if i:                                      # the shared prefix is forced: one multi-token feed
            lm.feed(first[:i])
            st["forced"] += i
            st["feeds"] += 1
        cand = list(options)
        while True:
            done = [o for o in cand if len(seqs[o]) == i]
            if done:
                return done[0], []
            if len(cand) == 1:
                return cand[0], seqs[cand[0]][i:]
            allowed = {seqs[o][i] for o in cand}
            tid = lm.pick(allowed)
            st["generated"] += 1
            lm.feed([tid])
            st["feeds"] += 1
            cand = [o for o in cand if seqs[o][i] == tid]
            i += 1

    def _flush(self, lm, pending_ids: list, text: str, st: dict) -> None:
        ids = list(pending_ids) + (list(self.encode(text)) if text else [])
        if ids:
            lm.feed(ids)
            st["forced"] += len(ids)
            st["feeds"] += 1

    def _string(self, lm, st: dict) -> tuple:
        """-> (value, text_after_value_already_decided). The token holding the closing quote is not fed."""
        ids: list = []
        for _ in range(self.max_value_tokens):
            tid = lm.pick(None)
            st["generated"] += 1
            if tid in self.stop_ids:
                break
            cur = lm.decode(ids + [tid])
            prev = lm.decode(ids) if ids else ""
            piece = cur[len(prev):] if cur.startswith(prev) else cur
            if '"' in piece or "\n" in piece:
                head = piece.split('"')[0].split("\n")[0]
                return prev + head, head            # head: text of the unfed token before the quote
            ids.append(tid)
            lm.feed([tid])
            st["feeds"] += 1
        return lm.decode(ids) if ids else "", ""

    def _number(self, lm, st: dict) -> tuple:
        ids: list = []
        for _ in range(self.max_value_tokens):
            tid = lm.pick(None)
            st["generated"] += 1
            prev = lm.decode(ids) if ids else ""
            cur = lm.decode(ids + [tid])
            piece = cur[len(prev):]
            if tid in self.stop_ids or not piece.strip() or any(c not in NUM_CHARS for c in piece.strip()):
                return prev.strip(), ""
            ids.append(tid)
            lm.feed([tid])
            st["feeds"] += 1
        return (lm.decode(ids) if ids else "").strip(), ""

    # ------------------------------------------------------------------ the call
    def run(self, lm, prefix_fed: str = "<TOOLCALL>[") -> tuple:
        """The LM state must end right after ``prefix_fed``. -> (call text, stats)."""
        t0 = time.monotonic_ns()
        st = {"forced": 0, "generated": 0, "feeds": 0}
        self._flush(lm, [], '{"name": "', st)
        name, pend = self._trie(lm, [n + '"' for n in self.tools], st)
        name = name[:-1]
        _, props, req, opt = self.tools[name]
        args: dict = {}
        text = ', "arguments": {'
        keys = list(req)
        first = True
        while True:
            if not keys:
                left = [k for k in opt if k not in args]
                if not left:
                    break
                sep = ", " if not first else ""
                # the unfed text (e.g. the closing quote of the last value) is part of every option, so that the
                # options are tokenized as the model would write them (E1l: '"' fed alone, then ', "postcode"'
                # lost against '}': the natural token is '",')
                options = [text + "}"] + [text + f'{sep}"{k}": ' for k in left]
                self._flush(lm, pend, "", st)
                choice, pend = self._trie(lm, options, st)
                choice, text = choice[len(text):], ""
                if choice == "}":
                    break
                key = choice.split('"')[1]
            else:
                key = keys.pop(0)
                text += ("" if first else ", ") + f'"{key}": '
            first = False
            typ = (props.get(key) or {}).get("type", "string")
            enum = (props.get(key) or {}).get("enum")
            if enum or typ == "boolean":
                lits = [text + json.dumps(v, ensure_ascii=False) for v in (enum or [True, False])]
                self._flush(lm, pend, "", st)
                lit, pend = self._trie(lm, lits, st)
                args[key] = json.loads(lit[len(text):])
                text = ""
            elif typ in ("integer", "number"):
                self._flush(lm, pend, text, st)
                pend, text = [], ""
                v, _ = self._number(lm, st)
                try:
                    args[key] = int(v) if typ == "integer" else float(v)
                except ValueError:
                    args[key] = v
            else:
                self._flush(lm, pend, text + '"', st)
                pend = []
                v, head = self._string(lm, st)
                args[key] = v
                text = head + '"'                   # the unfed quote token, re-encoded with what follows
        call = {"name": name, "arguments": args}
        out = "<TOOLCALL>[" + json.dumps(call, ensure_ascii=False) + "]</TOOLCALL>"
        st["ms"] = round((time.monotonic_ns() - t0) / 1e6, 1)
        st["n"] = st["generated"]
        return out, st
