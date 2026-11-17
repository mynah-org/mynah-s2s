# Control: intents, actions and one controller

The two-LLM engine has several actors that each see part of the conversation:

| actor | what it sees |
|---|---|
| duplex Thinker | the user audio |
| multilingual ASR | the words |
| text LLM (Nano) | the meaning and the tools |
| tool backend | the data |
| talker | what is audible |

None of them may change the turn state directly. They post **intents**. One **controller**
(`frankenstein_s2s/controller.py`) is the single writer of turn state and answers with **actions**.

## 1. Intents (inputs)

| source | intent | meaning | stamp `k` |
|---|---|---|---|
| Thinker | `TAKE_TURN` (payload `bos` or `tool_call_start`) | the duplex model decided to speak or to call a tool (BOS, or function id 20) | the model frame it was predicted at |
| Thinker | `YIELD` | the Thinker's EOS (advisory only: the controller owns EOS) | the model frame |
| ASR | `USER_SPEECH` / `USER_SPEECH_START` / `USER_SPEECH_END` | ASR activity at a frame (barge-in evidence) | the audio frame (emission frame + right context) |
| ASR | `WORDS` | new caller words were heard | the frame the words closed |
| ASR | `EOU` | end of utterance (N blank frames after speech) | the frame it fired |
| Nano | `REPLY_TOKENS(ids)` | streamed reply ids for `request_id` | the frame they reached the controller |
| Nano | `TOOL_CALL(call)` | a tool call for `request_id` | the frame it reached the controller |
| Nano | `REPLY_DONE` | the reply for `request_id` is complete | same |
| tool | `TOOL_RESULT` / `TOOL_TIMEOUT` | the result (or timeout) of `call_id` | same |
| talker | `AUDIO_STARTED` / `AUDIO_ENDED` / `UNDERRUN` | playback facts, tagged with the epoch of the audio | same |
| app | `CANCEL` | the user or the app stops the reply | same |

**Clocks.**
- Every intent is stamped with a **model frame index `k`** (80 ms of user audio). Thinker and ASR intents refer to
  the frame the evidence is about.
- Wall-clock sources (Nano, tools, talker, app) are stamped with the frame at which they were **delivered** to the
  controller, so a decision never uses something before it existed (causal).
- Wall time (`t_ns`) is in every trace line, but never decides anything.

## 2. Actions (outputs)

| action | effect |
|---|---|
| `OPEN_TURN` | new agent turn (`turn` + 1) |
| `REQUEST_REPLY(request_id, user_text)` | ask the text LLM for a reply; `request_id` + 1 on every open and every cancel |
| `SPEAK(ids)` | reply ids to the talker adapter (word commits, burst commit, segments: `adapter.py`) |
| `HOLD(reason)` | a signal was seen but not acted on (back-channel candidate, gate) |
| `DEFER(reason)` | a `TAKE_TURN` kept for later because the ASR still hears speech |
| `CANCEL(flush, new_epoch)` | stop the reply and flush the audio; `epoch` + 1; pending calls → cancelled |
| `CALL_TOOL(call_id)` | dispatch the call (async; the turn stays open) |
| `RESUME_AFTER_TOOL(call_id)` | the result (or timeout) is in; the text LLM continues the same request |
| `CLOSE_TURN(reason)` | controller-owned end of turn (the talker gets `tts_eos`, the Thinker gets EOS) |
| `DROP_STALE(reason)` | an intent belonging to an old epoch, request or call: ignored and traced |

## 3. State machine

```
            TAKE_TURN (gates ok) / EOU fallback
LISTENING ----------------------------------------> PENDING --REPLY_TOKENS--> SPEAKING --REPLY_DONE--> LISTENING
    ^  ^                                               |  ^                       |
    |  |                         TOOL_CALL             |  | RESUME_AFTER_TOOL     | TOOL_CALL
    |  |                                               v  |                       v
    |  +------------------- CANCEL (barge-in / app) - TOOL_WAIT <------------------+
    +--- reply_timeout / CANCEL from PENDING or SPEAKING
```

**Gates for opening a turn** (`_may_open`):
- not within `debounce` frames after a cancel;
- the ASR has been blank for ≥ `min_blank` frames;
- some caller words were heard since the last turn (no empty turns).

When the Thinker says `TAKE_TURN` while the ASR still hears speech, **the sources disagree**. The take-turn is
`DEFER`red and opens as soon as the ASR has been blank for `min_blank` frames, within `take_turn_window` frames.
Otherwise it expires (traced). The ASR `EOU` opens the turn only as a fallback, when the Thinker stays silent.

## 4. Priorities, races, idempotency, staleness

- **Priority within a frame:** barge-in evidence > cancel > tool (call / result / timeout) > speak (tokens / done) >
  yield > take-turn / EOU > the rest.
- **Ordering:** `tick(k)` applies every intent stamped ≤ k after sorting by (priority, source rank, k, uid). The
  arrival order inside a frame therefore never changes the decision. Tests permute same-frame intents and require
  identical actions.
- **Barge-in vs back-channel:** `barge_in_frames` consecutive frames of user speech while a turn is open → `CANCEL`.
  Shorter runs during `SPEAKING` → `HOLD(backchannel_candidate)`, and the reply continues.
- **Epochs:** every cancel bumps `epoch` and `request_id`.
  - `REPLY_TOKENS` / `REPLY_DONE` / `TOOL_CALL` with an old epoch or request → `DROP_STALE`.
  - A `TOOL_RESULT` for a cancelled or unknown call → `DROP_STALE`.
  - Talker audio of an old epoch → `DROP_STALE`. The `PlaybackGate` drops it too.
  - `REPLY_TOKENS` / `REPLY_DONE` of the **current** request during `TOOL_WAIT` are not stale. The worker runs the
    tool itself and may stream the answer before `TOOL_RESULT` lands. They are held (`HOLD reply_during_tool_wait`)
    and released in arrival order on `RESUME_AFTER_TOOL`. A cancel or a new turn discards them. (E1k regression: the
    first reply word was dropped.)
- **Idempotency:** an intent with a `uid` is applied once; a `TOOL_CALL` with a known `call_id` is ignored.
- **Timeouts:**
  - `PENDING` without tokens for `reply_timeout` frames → `CLOSE_TURN(reply_timeout)`;
  - `TOOL_WAIT` without a result for `tool_timeout` frames → `RESUME_AFTER_TOOL(tool_timeout)` (the LLM answers
    with the failure).

## 5. Sequence diagrams (each has a unit test in `tests/test_controller.py`)

**Normal turn** (`test_normal_turn`):
```
ASR     USER_SPEECH x5 (k10-14), WORDS (k14)
Thinker TAKE_TURN bos (k17: ASR blank 3 >= 2)        -> OPEN_TURN, REQUEST_REPLY(req 1)
Nano    REPLY_TOKENS(req 1) (k19)                    -> SPEAK
Nano    REPLY_DONE(req 1) (k24)                      -> CLOSE_TURN
```

**Tool turn** (`test_tool_turn`):
```
Thinker TAKE_TURN tool_call_start (k6)               -> OPEN_TURN, REQUEST_REPLY(req 1)
Nano    TOOL_CALL c1 (k8)                            -> CALL_TOOL c1          [TOOL_WAIT]
tool    TOOL_RESULT c1 (k20)                         -> RESUME_AFTER_TOOL     [PENDING]
Nano    REPLY_TOKENS (k22), REPLY_DONE (k25)         -> SPEAK, CLOSE_TURN
```

**Barge-in during speech** (`test_barge_in_during_speech…`):
```
... SPEAKING (req 1, epoch 0)
ASR     USER_SPEECH k10, k11                         -> HOLD backchannel_candidate x2
ASR     USER_SPEECH k12 (3rd frame)                  -> CANCEL (epoch 1, flush)
Nano    REPLY_TOKENS(req 1, epoch 0) (k13)           -> DROP_STALE old_epoch
```

**Barge-in during tool wait** (`test_barge_in_during_tool_wait…`):
```
... TOOL_WAIT (c1 pending)
ASR     USER_SPEECH x3                               -> CANCEL (c1 -> cancelled)
tool    TOOL_RESULT c1 (late)                        -> DROP_STALE call_cancelled
```

**Back-channel** (`test_backchannel_does_not_cancel`):
```
... SPEAKING
ASR     USER_SPEECH x2 (< barge_in_frames)           -> HOLD x2, no cancel
Nano    REPLY_DONE                                   -> CLOSE_TURN
```

**Long reply / segment restart** (`test_long_reply…`):
```
Nano    REPLY_TOKENS x24                             -> SPEAK x24, state SPEAKING, epoch unchanged
        (segment cuts at sentence ends happen in the talker adapter, not in the controller)
Nano    REPLY_DONE                                   -> CLOSE_TURN
```

**Second turn** (`test_second_turn…`):
```
turn 1 closed; ASR WORDS; Thinker TAKE_TURN          -> OPEN_TURN (turn 2), REQUEST_REPLY(req 2)
```

**Sources disagree** (`test_take_turn_while_asr_hears_speech…`):
```
Thinker TAKE_TURN k5 while ASR speech until k6       -> DEFER asr_speech_recent
k8 (ASR blank 2)                                     -> OPEN_TURN thinker_bos_deferred
```

**Gates, timeouts, races, idempotency** (`test_no_turn_without_new_caller_words…`, `test_reply_and_tool_timeouts`,
`test_race_orderings…`, `test_race_speech_vs_take_turn…`, `test_idempotent_and_traced`).

## 6. Tracing

Every intent is one `ctl.intent` line: kind, source, epoch, request_id, call_id, uid. Every action is one
`ctl.action` line: kind, epoch, turn, reason and its data. Both use the frame index `model_step` and wall `t_ns`, in
the same JSONL as the rest of the engine (`tracing.py`).

## 7. Status

- The controller is pure Python and unit-tested.
- The E1 runner still drives the older per-frame hook (`e1/m1force.py`), whose turn rules match these gates. Moving
  the runner onto this controller is the next integration step.
- NOT VERIFIED on GPU.
