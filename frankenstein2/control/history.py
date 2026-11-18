"""The conversation as the LLM sees it: user turns, tool calls / results, and only what the user HEARD of each
agent turn.

  * a turn opens: the user message is appended (a turn whose request was cancelled before anything was heard is
    removed again, and its text carries into the next turn: the controller merges it);
  * a tool ran: its call and result are appended at once (they stay even if the turn is barged later: the tool did
    run, e.g. a return was registered);
  * a turn ends normally: the assistant message is the reply text (what was spoken = the whole reply);
  * a barge-in: the assistant message is the heard text from the playout (segments played + the heard part of the
    current one), marked with " —" when it was cut; nothing at all if nothing was heard.
Holding phrases and backchannels never enter the history. Model-free; unit-tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

CUT_MARK = " —"


@dataclass
class Conversation:
    messages: list = field(default_factory=list)
    turn_user_idx: dict = field(default_factory=dict)      # turn -> index of its user message
    log: list = field(default_factory=list)

    def open_turn(self, turn: int, user_text: str) -> None:
        self.turn_user_idx[turn] = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})

    def drop_user_turn(self, turn: int) -> None:
        """The turn was cancelled before anything was heard and before any tool ran: its user text carries over."""
        i = self.turn_user_idx.pop(turn, None)
        if i is None or i >= len(self.messages):
            return
        if any(m["role"] == "tool" for m in self.messages[i + 1:]):
            return                                   # a tool ran in that turn: keep the exchange
        del self.messages[i:]
        self.log.append({"drop_user_turn": turn})

    def tool(self, name: str, args: dict, result: str, markup=None) -> None:
        """``markup(name, args)``: the LLM's own call format (default: Nemotron's <TOOLCALL>[...])."""
        if markup is not None:
            content = markup(name, args)
        else:
            content = "<TOOLCALL>[" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "]</TOOLCALL>"
        self.messages.append({"role": "assistant", "content": content})
        self.messages.append({"role": "tool", "content": result})

    def agent(self, text: str, cut: bool = False) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.messages.append({"role": "assistant", "content": text + (CUT_MARK if cut else "")})

    def for_llm(self) -> list:
        """A copy without consecutive same-role user messages (a merged turn already joined the texts)."""
        return [dict(m) for m in self.messages]
