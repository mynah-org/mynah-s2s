"""VoiceChat 11B special tokens and function-call text helpers.

Ids are those of the Nemotron-Nano-9B-v2 tokenizer shipped with checkpoint 443794e (tokenizer.json
``added_tokens``); they are re-checked against the live tokenizer at engine start (``verify_against``).
Semantics follow NeMo Speech 097dfe9e: duplex_stt_model.py (training) and
nemotron_voicechat_inference_wrapper.py:787-801 (inference).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

BOS_ID = 1          # <s>   agent start-of-turn
EOS_ID = 2          # </s>  agent end-of-turn
PAD_ID = 12         # <SPECIAL_12> text/function channel PAD
SOTC_ID = 20        # <SPECIAL_20> start of tool call   (function channel)
EOTC_ID = 21        # <SPECIAL_21> end of tool call     (function channel)
EOTR_ID = 22        # <SPECIAL_22> end of tool response (function channel, predicted with loss)

TOKEN_TEXT = {BOS_ID: "<s>", EOS_ID: "</s>", PAD_ID: "<SPECIAL_12>", SOTC_ID: "<SPECIAL_20>",
              EOTC_ID: "<SPECIAL_21>", EOTR_ID: "<SPECIAL_22>"}

# Fusion weights actually trained (checkpoint config.json 443794e: duplex_{user,text,function}_channel_weight).
FUSION_WEIGHTS = {"user": 1.0, "text": 1.0, "function": 2.0}
# FC loss weights used in training (paper §3.2 == checkpoint config); informational, inference does not use them.
FC_LOSS_WEIGHTS = {"call": 64.0, "sotc": 6.0, "eotc": 6.0, "eotr": 3.0, "pad": 0.3}

FRAME_S = 0.08            # one Thinker step = 80 ms of user audio
INPUT_SR = 16_000
OUTPUT_SR = 22_050
SAMPLES_PER_FRAME_IN = int(FRAME_S * INPUT_SR)     # 1280
SAMPLES_PER_FRAME_OUT = int(FRAME_S * OUTPUT_SR)   # 1764


def verify_against(tokenizer) -> dict[str, int]:
    """Check the constants above against a NeMo/HF tokenizer exposing ``text_to_ids``. Returns mismatches."""
    bad = {}
    for tid, txt in TOKEN_TEXT.items():
        if txt in ("<s>", "</s>"):
            continue
        got = tokenizer.text_to_ids(txt)
        if got != [tid]:
            bad[txt] = got
    return bad


def format_tool_response(payload: Any) -> str:
    """Exact TOOL_RESPONSE text used by the reference (offline_voicechat.py run_fc_offline_inference)."""
    return f"<TOOL_RESPONSE>[{json.dumps(payload)}]</TOOL_RESPONSE>"


@dataclass
class ParsedCall:
    name: str | None
    arguments: dict
    raw: str
    ok: bool
    error: str | None = None


_TC_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def parse_call_text(call_text: str) -> list[ParsedCall]:
    """Parse the function-channel text between SOTC and EOTC (same rules as the reference executor:
    optional <TOOLCALL> wrapper, a JSON list or a single object, ``arguments`` or ``parameters``)."""
    raw = call_text.replace(TOKEN_TEXT[SOTC_ID], "").replace(TOKEN_TEXT[EOTC_ID], "").strip()
    m = _TC_RE.search(raw)
    body = (m.group(1) if m else raw.replace("<TOOLCALL>", "").replace("</TOOLCALL>", "")).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        return [ParsedCall(None, {}, call_text, False, f"json: {e.msg}")]
    items = parsed if isinstance(parsed, list) else [parsed]
    out = []
    for it in items:
        if not isinstance(it, dict):
            out.append(ParsedCall(None, {}, call_text, False, "not an object"))
            continue
        args = it.get("arguments", it.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        name = it.get("name")
        if isinstance(name, str):
            name = name.replace("-", "_")  # reference normalisation (streaming_s2s_pipeline.py _execute_tool_call)
        out.append(ParsedCall(name, args if isinstance(args, dict) else {}, call_text, name is not None))
    return out
