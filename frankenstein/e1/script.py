"""E1 session script: one fixed user timeline per language, identical for every arm.

Timeline (seconds of user audio; silence between turns):
  lead 6.0 s  -> the agent opener (VoiceChat opens the call in silence)
  U1  m03_t0  store question ("three-seat sofa")            -> expects search_catalog
  gap 8.0 s
  U2  m01_t0  order status with name + postcode              -> expects find_customer (Mario Rossi, 10121)
  gap 12.0 s
  U3  m05_t0  out of scope (restaurant booking)              -> polite redirect, no tool
  gap 1.6 s   (the agent is expected to be answering U3 here)
  U4  m08_t0  barge-in: the caller talks over the U3 answer  -> the agent must stop; then ask for name + postcode
  gap 8.0 s
The timeline is NOT reactive: every arm receives the same samples at the same frames. Whether U4 really overlaps agent
speech is measured per arm, not assumed.
"""

from __future__ import annotations

FRAME_S = 0.08

TURNS = [
    # (label, agentbench id, gap after (s), purpose)
    ("U1", "m03_t0", 8.0, "store_question"),
    ("U2", "m01_t0", 12.0, "tool_request"),
    ("U3", "m05_t0", 1.6, "out_of_scope"),
    ("U4", "m08_t0", 8.0, "barge_in"),
]
LEAD_S = 6.0
VOICE = {"en": "vivian", "it": "ryan", "fr": "ryan"}
LANGS = ("it", "fr", "en")

GOLD = {
    "U1": {"tool": "search_catalog", "query_any": ["sofa", "couch", "divano", "canap"],
           "must_say_any": [["fjordby"], ["499", "four hundred ninety", "quattrocentonovantanove", "quatre cent"]]},
    "U2": {"tool": "find_customer", "name": "mario rossi", "postcode": "10121",
           "must_say_any": [["2026-11-14", "14 november", "november 14", "14 novembre", "fourteenth", "quattordici",
                             "quatorze", "14"]]},
    "U3": {"tool": None},
    "U4": {"tool": None, "ask_any": ["name", "nome", "nom", "postcode", "post code", "cap", "code postal"]},
}


def timeline(durations: dict) -> list:
    """durations {agentbench id: seconds} -> [{label, id, start_s, end_s, start_k, end_k, purpose}]."""
    t, out = LEAD_S, []
    for label, iid, gap, purpose in TURNS:
        d = durations[iid]
        out.append({"label": label, "id": iid, "start_s": round(t, 3), "end_s": round(t + d, 3),
                    "start_k": int(t / FRAME_S), "end_k": int((t + d) / FRAME_S), "purpose": purpose})
        t += d + gap
    return out


def total_s(durations: dict) -> float:
    return LEAD_S + sum(durations[i] + g for _, i, g, _ in TURNS)
