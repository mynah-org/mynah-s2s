"""Read a frozen ``vc_replay_v1`` file (V2 engine trace converted frame by frame) and drive the adapter with it.

Format: line 1 = header {"format": "vc_replay_v1", ...}; then frames {"k", "id", "kind", ...} (the Thinker text
channel, PAD included) and events {"event", "k", ...} (turn.started, function.started, tool.*). Function-channel ids
are not in the replay: ``function.started`` stands in for SOTC and ``tool.result.consumed`` for EOTR.
"""

from __future__ import annotations

import json
from pathlib import Path

from .adapter import TalkerAdapter
from .events import Frame, ToolEvent

EVENT_MAP = {"function.started": "pending", "tool.pending": "pending", "tool.result.injected": "result",
             "tool.result.consumed": "resumed"}


def load(path: Path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    hdr = rows[0]
    if hdr.get("format") != "vc_replay_v1":
        raise ValueError(f"not a vc_replay_v1 file: {hdr.get('format')!r}")
    frames = [r for r in rows[1:] if "kind" in r and "k" in r]
    events = [r for r in rows[1:] if "event" in r]
    return hdr, frames, events


def drive(adapter: TalkerAdapter, frames: list, events: list, speech_frames: frozenset = frozenset()) -> None:
    """Feed frames in order; tool events are applied before the frame with the same k."""
    by_k: dict = {}
    for e in events:
        kind = EVENT_MAP.get(e["event"])
        if kind:
            by_k.setdefault(int(e["k"]), []).append(ToolEvent(int(e["k"]), kind, str(e.get("call_id", ""))))
    for f in frames:
        k = int(f["k"])
        for ev in by_k.pop(k, []):
            adapter.tool(ev)
        adapter.feed(Frame(k, f.get("id"), user_speech=k in speech_frames))
    for k in sorted(by_k):
        for ev in by_k[k]:
            adapter.tool(ev)
