"""Guided opening: replay baseline A's own first agent turn into a session run with another perception front-end.

What it is
  On frames 0..upto, the agent text token that A's Thinker produced is written into `gen_text[t]`. The tokens
  come from A's trace and include A's greeting with its own BOS/EOS.
  - The write happens right BEFORE the wrapper's own `_apply_rnnt_turn_taking(t, ...)`. Its turn logic then sees
    exactly the token A's Thinker emitted and updates first_turn, agent_speaking, the counters and the EOU/fallback
    rules natively, as in A's run.
  - After that call, `gen_text[t]` feeds the TTS on the same frame and the Thinker's agent-text input on the next
    frame. The KV cache therefore holds A's greeting.
  - The function channel is not touched. A's is PAD on these frames; that is checked, not forced.
  - Instance-level hook only: no V2 or NeMo file is edited, and controller policies are unchanged.

Rejected implementation (kept for the record)
  NVIDIA's own redirect queue (`_redirect_tokens_queue`, wrapper:3036-3044) also writes a token per frame. However,
  it `continue`s past the rest of the turn logic.
  - The CPU replay (test_guided_open.py) shows `agent_speaking`, `first_turn`, `nonblank_total` and
    `rolling_density` diverging from A's native run. One reason: A's stray EOS at frame 9 keeps `agent_speaking`
    False through frames 19-49 in the window logic.
  - The divergence persists after the opener (`rolling_density`).
  - The queued EOS also sets `_agent_eos_just_fired`, which A's spontaneous EOS never sets.

What it is not
  - Not a pre-recorded WAV. Audio played to the caller without agent tokens leaves the Thinker's KV cache and the
    controller's `first_turn` exactly as before, so the first-turn fallback would still fire.
  - Not a synthetic controller state such as `first_turn=False` without a greeting. That would be a state no model
    ever produced.

Remaining deviations, measured rather than hidden
  - Before the turn-taking runs, the FC state machine and the ASR-channel forcing see the Thinker's own tokens on
    the injected frames.
    - The function token is the Thinker's; `observe.py` logs it, and the gate requires PAD there.
    - The ASR channel never forces anything under `turn_taking_source: rnnt`.
  - Frames run under an active FC never call the turn-taking, so they cannot be injected. This is impossible in
    the lead-in, but `summarise()` reports any missing frame.
"""
from __future__ import annotations

import json

PAD, BOS, EOS = 12, 1, 2


def opener_from_trace(trace_path, upto: int = 94, session: int = 0) -> list[int]:
    """A's agent text tokens for frames 0..upto from a V2 trace.jsonl.

    `thinker.step` with model_step=N and k text ids covers frames N-k..N-1; steps with no ids are skipped.
    Raises if any frame is missing.
    """
    sessions, cur = [], None
    for line in open(trace_path):
        e = json.loads(line)
        if e["ev"] == "session.opened":
            cur = []
            sessions.append(cur)
        if cur is not None:
            cur.append(e)
    d = {}
    for e in sessions[session]:
        if e["ev"] == "thinker.step" and e.get("text_ids"):
            n = len(e["text_ids"])
            for i, tok in enumerate(e["text_ids"]):
                d[e["model_step"] - n + i] = int(tok)
    missing = [t for t in range(upto + 1) if t not in d]
    if missing:
        raise ValueError(f"opener frames missing in {trace_path}: {missing[:10]}")
    return [d[t] for t in range(upto + 1)]


def install(w, tokens: list[int], rec, check_input: bool = True) -> callable:
    """Install on the wrapper `w`; returns uninstall().

    Install this BEFORE observe.install, so that observe's forcing rows show pre = the Thinker's own token and
    post = the token after injection and native turn-taking.

    check_input (KV-cache propagation)
      After injected frame t, the next single-token call of the Thinker's embedding table, `stt_model.embed_tokens`
      at wrapper:2399 (`gen_text[:, t]` -> the agent-text input of frame t+1), is recorded and compared with the
      post-turn-taking token of frame t. Observation only: the call's output is unchanged.
    """
    orig = w._apply_rnnt_turn_taking
    holder = {}

    def wrapped(t, gen_text, *args, **kw):
        t = int(t)
        if t >= len(tokens):
            return orig(t, gen_text, *args, **kw)
        pre = int(gen_text[0, t])
        pending = bool(getattr(w, "_redirect_tokens_queue", None))   # NVIDIA redirect would overwrite: report it
        gen_text[0, t] = int(tokens[t])
        out = orig(t, gen_text, *args, **kw)
        row = {"kind": "guided", "t": t, "tok": int(tokens[t]), "thinker": pre, "post": int(gen_text[0, t])}
        if pending:
            row["error"] = "native redirect pending"
        rec.append(row)
        holder["expect"] = (t, row["post"])
        return out

    w._apply_rnnt_turn_taking = wrapped
    undo = [lambda: setattr(w, "_apply_rnnt_turn_taking", orig)]
    if check_input:
        emb = w.model.stt_model.embed_tokens
        orig_fwd = emb.forward

        def fwd(ids, *a, **k):
            exp = holder.pop("expect", None)
            if exp is not None:
                try:
                    v = [int(x) for x in ids.reshape(-1).tolist()]
                    rec.append({"kind": "guided_input", "t_in": exp[0] + 1, "ids": v, "expected": exp[1],
                                "ok": v == [exp[1]]})
                except Exception as e:  # noqa: BLE001 - observation must never break the run
                    rec.append({"kind": "guided_input", "t_in": exp[0] + 1, "error": repr(e)})
            return orig_fwd(ids, *a, **k)
        emb.forward = fwd
        undo.append(lambda: emb.__dict__.pop("forward", None))
    return lambda: [u() for u in reversed(undo)]


def summarise(rows: list[dict], tokens: list[int]) -> dict:
    g = [r for r in rows if r.get("kind") == "guided"]
    seen = {r["t"] for r in g if "tok" in r}
    agree = [r for r in g if "tok" in r and r["thinker"] == r["tok"]]
    return {"frames": len(tokens), "injected": len(seen), "missing": [t for t in range(len(tokens)) if t not in seen],
            "post_equals_token": all(r["post"] == r["tok"] for r in g if "tok" in r),
            "overridden_by_turn_logic": [r["t"] for r in g if "tok" in r and r["post"] != r["tok"]],
            "errors": [r for r in g if "error" in r],
            "kv_input_checked": sum(1 for r in rows if r.get("kind") == "guided_input"),
            "kv_input_ok": sum(1 for r in rows if r.get("kind") == "guided_input" and r.get("ok")),
            "kv_input_bad": [r for r in rows if r.get("kind") == "guided_input" and not r.get("ok")][:5],
            "thinker_agrees": len(agree),
            "thinker_disagrees_at": [r["t"] for r in g if "tok" in r and r["thinker"] != r["tok"]][:40]}
