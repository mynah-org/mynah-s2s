"""Frankenstein S2S: audio-timed VoiceChat duplex loop + text-mode content from a multilingual ASR transcript
(M1-force) -> talker adapter -> Flash talker. Pure-Python control layer only; see ../DESIGN.md."""

from .adapter import AdapterConfig, State, TalkerAdapter
from .driver import FakeTalkerDriver, FlashTalkerDriver, PlaybackGate, TalkerDriver
from .forcing import AgentChannelForcer, ForcedStep, ForceState
from .events import (AppendText, AudioChunk, Cancel, FinalizeSegment, Frame, LogEntry, StartSegment, TalkerDone,
                     ToolEvent)
from .tracing import NullTracer, Tracer, thinker_frame
from .lang import SessionLanguage, language_instruction, session_language, with_language
from .usertext import (InjectionMode, InsertSpan, TranscriptInserter, TurnTaker, UserTextScheduler,
                        frame_inputs)

__all__ = [
    "AdapterConfig", "State", "TalkerAdapter", "FakeTalkerDriver", "FlashTalkerDriver", "PlaybackGate", "TalkerDriver",
    "AppendText", "AudioChunk", "Cancel", "FinalizeSegment", "Frame", "LogEntry", "StartSegment", "TalkerDone",
    "ToolEvent", "SessionLanguage", "language_instruction", "session_language", "with_language", "InjectionMode",
    "TurnTaker", "UserTextScheduler", "frame_inputs", "InsertSpan", "TranscriptInserter", "AgentChannelForcer",
    "ForcedStep", "ForceState", "NullTracer", "Tracer", "thinker_frame",
]
