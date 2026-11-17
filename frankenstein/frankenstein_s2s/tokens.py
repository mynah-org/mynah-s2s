"""VoiceChat Thinker special token ids, as the adapter sees them on the text and function channels.

These mirror ``v2/src/mynah_v2/model/tokens.py`` (Nemotron-Nano-9B-v2 tokenizer shipped with the VoiceChat
checkpoint). They are duplicated on purpose so this package stays importable without the V2 engine and without
torch; ``tests/test_tokens.py`` checks the two copies agree when the V2 sources are present.
"""

from __future__ import annotations

BOS_ID = 1    # <s>            agent start of turn (text channel)
EOS_ID = 2    # </s>           agent end of turn   (text channel)
PAD_ID = 12   # <SPECIAL_12>   text / function channel PAD
SOTC_ID = 20  # <SPECIAL_20>   start of tool call  (function channel)
EOTC_ID = 21  # <SPECIAL_21>   end of tool call    (function channel)
EOTR_ID = 22  # <SPECIAL_22>   end of tool response (function channel)

# Ids below this value are the tokenizer's <SPECIAL_n> range: never forwarded as text, never injected as user text.
FIRST_TEXT_ID = 1000

FRAME_S = 0.08  # one Thinker step = 80 ms

SPECIAL_KIND = {BOS_ID: "bos", EOS_ID: "eos", PAD_ID: "pad", SOTC_ID: "sotc", EOTC_ID: "eotc", EOTR_ID: "eotr"}


def kind_of(token_id: int | None) -> str:
    """Classify one text-channel id: bos | eos | pad | sotc | eotc | eotr | special | tok | missing."""
    if token_id is None:
        return "missing"
    if token_id in SPECIAL_KIND:
        return SPECIAL_KIND[token_id]
    if token_id < FIRST_TEXT_ID:
        return "special"
    return "tok"
