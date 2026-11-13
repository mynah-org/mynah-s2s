"""Function-calling sub-timeline: reference invariants and a checker for per-step channel tokens.

Training format (NeMo Speech 097dfe9e duplex_stt_model.py:936-1131, 1555-1590; paper §2.2):
  call and response tokens occupy *inserted* frames, one token per frame, in the function channel;
  on those frames the user audio is encoded silence and the agent text channel is PAD;
  order: SOTC, call tokens, EOTC, response tokens (no loss), EOTR (predicted, with loss) right after.
Reference live inference (async two-phase, streaming_s2s_pipeline.py:1540-1900, wrapper:1179-1772):
  the LLM does not step while the tool runs; the response is forced 1 token/step; one free step for EOTR;
  the main timeline resumes after (frame_idx += async_steps).

The checker below takes the three per-step token channels of a run (agent text, function, and a per-step
flag saying whether the user-audio input at that step was silence) and reports violations. It is backend
agnostic: it is used on the NVIDIA reference trace, on our engine, and on synthetic sequences in CPU tests.
Known failure it must catch: free agent text between EOTC and the response (vLLM-Omni backlog, P0 §1a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mynah_v2.model.tokens import EOTC_ID, EOTR_ID, PAD_ID, SOTC_ID


class FCPhase(str, Enum):
    IDLE = "idle"
    CALL = "call"            # SOTC seen, call tokens being generated (fast path)
    WAIT = "wait"            # EOTC seen, tool executing; no timeline positions consumed
    INJECT = "inject"        # response tokens forced one per step
    EOTR = "eotr"            # free step(s) where the model must predict EOTR
    # after EOTR -> IDLE


@dataclass
class FCSpan:
    sotc_step: int
    eotc_step: int | None = None
    resp_start: int | None = None
    resp_end: int | None = None        # last forced response step
    eotr_step: int | None = None
    call_tokens: list[int] = field(default_factory=list)


@dataclass
class TimelineReport:
    spans: list[FCSpan]
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_fc_timeline(text: list[int], func: list[int], user_silence: list[bool] | None = None,
                      forced_response: list[bool] | None = None, eotr_window: int = 1) -> TimelineReport:
    """Validate reference FC invariants over aligned per-step channels.

    text/func: token id per Thinker step (agent text channel, function channel).
    user_silence: per step, True if the user-audio input was the silence embedding (optional).
    forced_response: per step, True if the function token at that step was forced (tool response injection).
    eotr_window: number of free steps allowed after the forced response for EOTR (reference: 1).
    """
    n = len(text)
    if len(func) != n:
        raise ValueError("text and func must be aligned")
    viol: list[str] = []
    spans: list[FCSpan] = []
    cur: FCSpan | None = None
    phase = FCPhase.IDLE
    free_after = 0
    for t in range(n):
        f = func[t]
        forced = bool(forced_response[t]) if forced_response else False
        if phase == FCPhase.IDLE:
            if f == SOTC_ID:
                cur = FCSpan(sotc_step=t)
                spans.append(cur)
                phase = FCPhase.CALL
            elif f in (EOTC_ID, EOTR_ID):
                viol.append(f"step {t}: {'EOTC' if f == EOTC_ID else 'EOTR'} outside a call")
            continue
        # inside a call: agent text must be PAD on every inserted frame
        if text[t] != PAD_ID:
            viol.append(f"step {t}: agent text {text[t]} != PAD during FC phase {phase.value}")
        if user_silence is not None and not user_silence[t]:
            viol.append(f"step {t}: user audio is not silence during FC phase {phase.value}")
        if phase == FCPhase.CALL:
            if f == EOTC_ID:
                cur.eotc_step = t
                phase = FCPhase.WAIT
            elif f in (SOTC_ID, EOTR_ID):
                viol.append(f"step {t}: unexpected {f} inside call tokens")
            else:
                cur.call_tokens.append(f)
        elif phase == FCPhase.WAIT:
            # the step right after EOTC must already carry the forced response (no free frames in between)
            if forced:
                cur.resp_start = cur.resp_end = t
                phase = FCPhase.INJECT
            else:
                viol.append(f"step {t}: free step between EOTC (step {cur.eotc_step}) and the tool response")
        elif phase == FCPhase.INJECT:
            if forced:
                cur.resp_end = t
            else:
                phase = FCPhase.EOTR
                free_after = 0
        if phase == FCPhase.EOTR:
            free_after += 1
            if f == EOTR_ID:
                cur.eotr_step = t
                phase = FCPhase.IDLE
                cur = None
            elif free_after > eotr_window:
                viol.append(f"step {t}: no EOTR within {eotr_window} free step(s) after the response")
                phase = FCPhase.IDLE
                cur = None
    if cur is not None:
        viol.append(f"FC span starting at step {cur.sotc_step} not closed (phase {phase.value})")
    return TimelineReport(spans, viol)
