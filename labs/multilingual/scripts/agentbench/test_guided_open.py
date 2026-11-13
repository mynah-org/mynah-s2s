"""CPU tests for guided_open.py.

Part 1 (anywhere): the injector on a fake wrapper.
Part 2 (lab-box CPU, needs the NeMo fork on PYTHONPATH and the live-timing artefacts): closed-loop replay of the REAL
`_apply_rnnt_turn_taking` (NeMo 097dfe9) with a mirror of `_rnnt_step`'s counter update (wrapper:2972-2995).
  2a. Validation: fed the Thinker's own tokens and the RNNT blank flags recorded by observe.py in job 252, the
      replay must reproduce the recorded post-forcing token and turn state on every frame before the call. This
      covers C-ridge (spontaneous opener), D-V2.1 with the fallback on (forced BOS at 98) and D-V2.1 with it off.
  2b. Equivalence:
      - native = A's opener tokens produced by the Thinker on frames 0..94;
      - guided = the D-V2.1 Thinker's tokens with A's opener pre-written by guided_open.install;
      - both are followed by the same continuation and the same blank flags.
      `gen_text`, the full turn state and the EOS flag must match on every frame.
      The rejected redirect-queue variant is replayed too and its divergence is reported, not asserted.
Usage: python3 test_guided_open.py [--live-dir DIR --a-trace TRACE --nemo-yaml YAML]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guided_open as G  # noqa: E402

STATE_KEYS = ("blank_count", "nonblank_consec", "nonblank_total", "speech_confirmed", "agent_speaking", "first_turn")
# turn state read by the logic (2b compares these plus `_turn_text_tokens` on EVERY frame, opener included)
DECISION_KEYS = STATE_KEYS + ("user_first_turn", "rolling_density", "post_eos_fired", "forced_bos")


# ------------------------------------------------------------------------------------------- part 1: fake wrapper
class FakeEmb:
    def forward(self, ids):
        return ids

    def __call__(self, ids):
        return self.forward(ids)


class FakeIds(list):
    def reshape(self, *_):
        return self

    def tolist(self):
        return list(self)


class FakeW:
    """Native-like: the turn logic reads gen_text[t] (recorded in `seen`); NVIDIA's redirect queue overwrites."""

    def __init__(self):
        self._redirect_tokens_queue, self._agent_eos_just_fired = [], False
        self.seen = {}
        self.model = SimpleNamespace(stt_model=SimpleNamespace(embed_tokens=FakeEmb()))

    def _apply_rnnt_turn_taking(self, t, gen_text, is_blank, state):
        self.seen[t] = gen_text[0, t]
        if self._redirect_tokens_queue:
            gen_text[0, t] = self._redirect_tokens_queue.pop(0)


class Row(list):
    def __getitem__(self, k):
        return list.__getitem__(self, k[1]) if isinstance(k, tuple) else list.__getitem__(self, k)

    def __setitem__(self, k, v):
        return list.__setitem__(self, k[1], v) if isinstance(k, tuple) else list.__setitem__(self, k, v)


def test_injector_fake():
    w, rec = FakeW(), []
    toks = [G.PAD, G.EOS, G.PAD, G.BOS, 500, G.EOS]
    un = G.install(w, toks, rec)
    g = Row([G.PAD] * 10)                                      # the Thinker says PAD on every frame
    emb = w.model.stt_model.embed_tokens
    for t in range(10):
        if t > 0:
            emb(FakeIds([g[t - 1]]))                           # next frame's agent-text input = gen_text[t-1]
        w._apply_rnnt_turn_taking(t, g, False, {})
    assert [w.seen[t] for t in range(6)] == toks               # the native turn logic SEES A's tokens
    assert list(g[:6]) == toks and list(g[6:]) == [G.PAD] * 4
    s = G.summarise(rec, toks)
    assert s["injected"] == 6 and s["post_equals_token"] and not s["missing"] and not s["overridden_by_turn_logic"]
    assert s["thinker_agrees"] == 2                             # the Thinker said PAD; tokens 0 and 2 are PAD
    assert s["kv_input_checked"] == 6 and s["kv_input_ok"] == 6   # frames 1..6 read the injected tokens 0..5
    un()
    assert "forward" not in emb.__dict__
    assert w._apply_rnnt_turn_taking.__func__ is FakeW._apply_rnnt_turn_taking


def test_injector_native_pending_and_missing():
    w, rec = FakeW(), []
    toks = [G.PAD, G.BOS, 7, G.EOS]
    G.install(w, toks, rec)
    g = Row([G.PAD] * 6)
    w._redirect_tokens_queue = [99]                            # NVIDIA's own redirect pending -> reported
    w._apply_rnnt_turn_taking(0, g, False, {})
    for t in (1, 3, 4):                                        # frame 2 skipped (e.g. FC in progress)
        w._apply_rnnt_turn_taking(t, g, False, {})
    s = G.summarise(rec, toks)
    assert s["errors"] and s["missing"] == [2] and g[0] == 99 and s["overridden_by_turn_logic"] == [0]


def test_opener_from_trace(tmp=Path("/tmp")):
    p = tmp / "guided_trace_test.jsonl"
    ev = [{"ev": "session.opened"}, {"ev": "thinker.step", "model_step": 2, "text_ids": [12, 1]},
          {"ev": "thinker.step", "model_step": 3, "steps": 0},
          {"ev": "thinker.step", "model_step": 4, "text_ids": [5, 2]}]
    p.write_text("\n".join(json.dumps(e) for e in ev))
    assert G.opener_from_trace(p, upto=3) == [12, 1, 5, 2]
    try:
        G.opener_from_trace(p, upto=4)
        raise AssertionError("expected missing-frame error")
    except ValueError:
        pass


# --------------------------------------------------------------------------------- part 2: real NeMo replay (lab box)
def make_self(cfg: dict):
    st = SimpleNamespace(text_bos_id=G.BOS, text_eos_id=G.EOS, text_pad_id=G.PAD)
    return SimpleNamespace(model_cfg=cfg, model=SimpleNamespace(stt_model=st), _redirect_tokens_queue=[],
                           _agent_eos_just_fired=False, _post_tc_bos_exempt=False, _max_agent_response_frames=0,
                           _max_response_redirect_tokens=[])


def rnnt_counters(torch, s: dict, blank: bool) -> dict:
    """Mirror of `_rnnt_step`'s state update (wrapper:2972-2995) for one frame, given the blank flag."""
    b = torch.tensor([blank])
    n = dict(s)
    n["blank_count"] = torch.where(b, s["blank_count"] + 1, torch.zeros_like(s["blank_count"]))
    n["nonblank_consec"] = torch.where(b, torch.zeros_like(s["nonblank_consec"]), s["nonblank_consec"] + 1)
    n["nonblank_total"] = torch.where(b, s["nonblank_total"], s["nonblank_total"] + 1)
    if not s["agent_speaking"].any():
        n["rolling_density"] = 0.1 * (~b).float() + 0.9 * s["rolling_density"]
    return n


def snap(s: dict) -> dict:
    out = {}
    for k in DECISION_KEYS + ("_turn_text_tokens",):
        v = s.get(k)
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            x = v[0].item()
            out[k] = round(x, 6) if isinstance(x, float) else x
    return out


def install_redirect(w, tokens, rec):
    """REJECTED variant, kept to document why: NVIDIA's redirect queue, one token per frame."""
    orig = w._apply_rnnt_turn_taking

    def wrapped(t, gen_text, *a, **k):
        if t < len(tokens):
            w._redirect_tokens_queue = [int(tokens[t])]
        out = orig(t, gen_text, *a, **k)
        if t < len(tokens):
            rec.append({"kind": "guided", "t": t, "tok": int(tokens[t]), "thinker": -1, "post": int(gen_text[0, t])})
        return out
    w._apply_rnnt_turn_taking = wrapped


def replay(torch, fn, cfg, thinker: list[int], blanks: list[bool], inject: list[int] | None = None,
           how: str = "prewrite"):
    w = make_self(cfg)
    w._apply_rnnt_turn_taking = lambda *a, **k: fn(w, *a, **k)
    rec = []
    if inject is not None:
        if how == "prewrite":
            G.install(w, inject, rec, check_input=False)     # no embedding table in the CPU replay
        else:
            install_redirect(w, inject, rec)
    s = {"blank_count": torch.zeros(1, dtype=torch.long), "nonblank_consec": torch.zeros(1, dtype=torch.long),
         "nonblank_total": torch.zeros(1, dtype=torch.long), "speech_confirmed": torch.zeros(1, dtype=torch.bool),
         "agent_speaking": torch.zeros(1, dtype=torch.bool), "first_turn": torch.ones(1, dtype=torch.bool),
         "user_first_turn": torch.ones(1, dtype=torch.bool), "rolling_density": torch.zeros(1),
         "post_eos_fired": torch.zeros(1, dtype=torch.bool), "forced_bos": torch.zeros(1, dtype=torch.bool)}
    g = torch.full((1, len(thinker)), G.PAD, dtype=torch.long)
    frames = []
    for t, (tok, bl) in enumerate(zip(thinker, blanks)):
        g[0, t] = tok
        s = rnnt_counters(torch, s, bl)
        w._apply_rnnt_turn_taking(t, g, torch.tensor([bl]), s)
        frames.append({"post": int(g[0, t]), "state": snap(s), "flag": bool(w._agent_eos_just_fired)})
    return frames, rec


def observed(run_dir: Path):
    rows = [json.loads(x) for x in open(run_dir / "observe.jsonl")]
    by_t = {}
    for r in rows:
        if r.get("kind") == "forcing" and r.get("fn") == "_apply_rnnt_turn_taking":
            by_t.setdefault(r["t"], r)
    return by_t


def load_fn_cfg(nemo_yaml: Path):
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
        NemotronVoicechatInferenceWrapper as W)
    base = {k: v for k, v in OmegaConf.to_container(OmegaConf.load(nemo_yaml).s2s, resolve=False).items()
            if not isinstance(v, (dict, list))}
    base["rnnt_self_play_suppression"] = False          # wrapper:543
    return W._apply_rnnt_turn_taking, base


def validate_run(session_dir: Path, nemo_yaml: Path, a_trace: Path | None, upto_open: int = 94) -> dict:
    """S0.4: closed-loop replay of a NEW session's own observe rows (Thinker token `pre`, blank flag) through the
    real NeMo turn logic, with the guided pre-write when the session was guided. Frames: the contiguous prefix
    before the first frame without a turn-taking call (FC in progress)."""
    import torch
    fn, cfg = load_fn_cfg(nemo_yaml)
    summ = json.load(open(session_dir / "summary.json"))
    if summ.get("first_turn_fallback") == "off":
        cfg["first_turn_fallback_frames"] = 10 ** 9
    ob = observed(session_dir)
    n = 0
    while n in ob:
        n += 1
    T = range(n)
    inject = G.opener_from_trace(a_trace, upto=upto_open) if summ.get("opening") == "guided" else None
    frames, _ = replay(torch, fn, cfg, [ob[t]["pre"] for t in T], [ob[t]["is_blank"] for t in T], inject=inject)
    bad = [{"t": t, "exp_post": ob[t]["post"], "got_post": frames[t]["post"], "exp": ob[t]["rnnt"],
            "got": {k: frames[t]["state"][k] for k in STATE_KEYS}} for t in T
           if frames[t]["post"] != ob[t]["post"] or any(frames[t]["state"][k] != ob[t]["rnnt"][k] for k in STATE_KEYS)]
    return {"session": session_dir.name, "frames": n, "guided": inject is not None, "mismatches": len(bad),
            "first": bad[:3], "state_95": frames[upto_open + 1]["state"] if n > upto_open + 1 else None}


def part2(live_dir: Path, a_trace: Path, nemo_yaml: Path, upto_call: int = 132, upto_open: int = 94) -> dict:
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
        NemotronVoicechatInferenceWrapper as W)
    fn = W._apply_rnnt_turn_taking
    base = {k: v for k, v in OmegaConf.to_container(OmegaConf.load(nemo_yaml).s2s, resolve=False).items()
            if not isinstance(v, (dict, list))}      # scalar turn-taking keys; nested vLLM blocks unresolved
    base["rnnt_self_play_suppression"] = False          # wrapper:543
    rep = {"cfg_keys": {k: base.get(k) for k in ("force_turn_taking", "force_turn_taking_threshold", "rnnt_eou_frames",
                                                    "rnnt_bou_frames", "asr_eou_first_turn",
                                                    "asr_min_speech_frames_first_turn", "max_agent_response_sec",
                                                    "tts_text_token_ratio_cap")}}
    # 2a validation against the live observe rows (closed loop: the Thinker tokens recorded live are fed back)
    val = {}
    for name, fb in (("cridge_en_m03", None), ("v21_en_m03", None), ("v21_en_m03_fboff", 10 ** 9)):
        cfg = dict(base)
        if fb is not None:
            cfg["first_turn_fallback_frames"] = fb
        ob = observed(live_dir / name)
        T = range(0, upto_call + 1)
        frames, _ = replay(torch, fn, cfg, [ob[t]["pre"] for t in T], [ob[t]["is_blank"] for t in T])
        bad = []
        for t in T:
            exp = ob[t]
            got = frames[t]
            if got["post"] != exp["post"] or any(got["state"][k] != exp["rnnt"][k] for k in STATE_KEYS):
                bad.append({"t": t, "exp_post": exp["post"], "got_post": got["post"],
                            "exp": exp["rnnt"], "got": {k: got["state"][k] for k in STATE_KEYS}})
        val[name] = {"frames": len(T), "mismatches": len(bad), "first": bad[:3],
                     "forced": [t for t in T if ob[t]["pre"] != ob[t]["post"]]}
        assert not bad, (name, bad[:3])
    rep["validation"] = val
    # 2b equivalence: native A opener vs guided injection into D-V2.1's own tokens, same continuation and blanks
    ob = observed(live_dir / "v21_en_m03_fboff")
    T = range(0, upto_call + 1)
    blanks = [ob[t]["is_blank"] for t in T]
    v21 = [ob[t]["pre"] for t in T]
    a_open = G.opener_from_trace(a_trace, upto=upto_open)
    cont = v21[upto_open + 1:]
    eq = {}
    for label, fb in (("fallback_default", None), ("fallback_off", 10 ** 9)):
        cfg = dict(base)
        if fb is not None:
            cfg["first_turn_fallback_frames"] = fb
        nat, _ = replay(torch, fn, cfg, a_open + cont, blanks)
        gui, rec = replay(torch, fn, cfg, v21[:upto_open + 1] + cont, blanks, inject=a_open)
        tok_diff = [t for t in T if nat[t]["post"] != gui[t]["post"]]
        st_diff = [t for t in T if nat[t]["state"] != gui[t]["state"]]   # every key, every frame
        flag_diff = [t for t in T if nat[t]["flag"] != gui[t]["flag"]]
        ttt = [(t, nat[t]["state"]["_turn_text_tokens"], gui[t]["state"]["_turn_text_tokens"]) for t in (upto_open + 1, upto_call)]
        eq[label] = {"gen_text_diff_frames": tok_diff, "decision_state_diff_frames": st_diff,
                     "eos_flag_diff_frames": flag_diff,                      "turn_text_tokens(native,guided)": ttt, "injector": G.summarise(rec, a_open),
                     "forced_after_opener_native": [t for t in T if t > upto_open and nat[t]["post"] != (a_open + cont)[t]],
                     "state_at_95": nat[upto_open + 1]["state"]}
        red, _ = replay(torch, fn, cfg, v21[:upto_open + 1] + cont, blanks, inject=a_open, how="redirect")
        eq[label]["rejected_redirect_variant"] = {
            "gen_text_diff_frames": [t for t in T if nat[t]["post"] != red[t]["post"]],
            "state_diff_frames_during_opener": sum(1 for t in range(upto_open + 1) if any(
                nat[t]["state"][k] != red[t]["state"][k] for k in DECISION_KEYS)),
            "state_diff_frames_after_opener": [t for t in T if t > upto_open and any(
                nat[t]["state"][k] != red[t]["state"][k] for k in DECISION_KEYS)][:5],
            "keys": sorted({k for t in T for k in DECISION_KEYS if nat[t]["state"][k] != red[t]["state"][k]}),
            "eos_flag_diff_frames": [t for t in T if nat[t]["flag"] != red[t]["flag"]][:5]}
        assert not tok_diff and not st_diff and not flag_diff, (label, eq[label])
    rep["equivalence"] = eq
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--a-trace", default=None)
    ap.add_argument("--nemo-yaml", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate-run", action="append", default=[], help="session dir of a new run (S0.4)")
    a = ap.parse_args()
    test_injector_fake()
    test_injector_native_pending_and_missing()
    test_opener_from_trace()
    print("OK part 1 (fake wrapper)")
    if a.validate_run:
        res = [validate_run(Path(d), Path(a.nemo_yaml), Path(a.a_trace) if a.a_trace else None) for d in a.validate_run]
        print(json.dumps(res, indent=1, default=str))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        sys.exit(0 if all(r["mismatches"] == 0 for r in res) else 3)
    if a.live_dir:
        r = part2(Path(a.live_dir), Path(a.a_trace), Path(a.nemo_yaml))
        print(json.dumps(r, indent=1, default=str)[:6000])
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1, default=str))
        print("OK part 2 (real NeMo turn-taking replay)")
