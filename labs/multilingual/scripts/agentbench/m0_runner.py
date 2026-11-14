"""M0-smoke runner: tenant-furniture scenarios through the V2 Session on the NVIDIA VoiceChat engine (GPU).

Same wiring as V2 `scripts/run_stream.py` (V2 snapshot, unmodified): ToolLedger + ToolRegistry.bind + Controller +
Session, realtime pacing (required with tools), ENHANCED by default (bounded tool wait, unknown-tool error), with the
TENANT system prompt and tools rendered by NVIDIA's FC template. Tool handlers are the mock tenant backend
(tenant_backend.py), which enforces identity / confirmation policy and logs every decision.
Multi-turn scenarios: turn WAVs concatenated with a fixed gap (one session, one stream).
Outputs per scenario: <out>/<lang>_<sid>/{trace.jsonl, summary.json, output.wav}; <out>/records.jsonl for
tenant_score.py. Never overwrites an existing --out.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parent.parent / "multilingual"))   # labpaths (MYNAH_HOME / MYNAH_LAB_HOME)
from labpaths import LAB_HOME  # noqa: E402
from tenant_backend import TenantBackend  # noqa: E402

from mynah_v2.agent.controller import Controller  # noqa: E402
from mynah_v2.agent.tools import ToolLedger, ToolRegistry  # noqa: E402
from mynah_v2.model import tokens as T  # noqa: E402
from mynah_v2.model.reference import DEFAULT_NEMO, ReferenceEngine, build_cfg  # noqa: E402
from mynah_v2.runtime.session import EnhancedOptions, Mode, Session, run_paced  # noqa: E402
from mynah_v2.tracing.events import Tracer  # noqa: E402

sys.path.insert(0, str(DEFAULT_NEMO / "examples/speechlm2"))


def load_16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy()).mean(0)
    return AF.resample(wav.unsqueeze(0), sr, 16000)[0] if sr != 16000 else wav


def render_prompt(schemas, system_message):
    import offline_voicechat_fc_infer as fcmod
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import render_fc_system_prompt
    return render_fc_system_prompt(str(fcmod.DEFAULT_TEMPLATE), system_message, schemas)


class JsonlSink(list):
    """A list that also appends each row to a JSONL file as it arrives (flushed), so the evidence survives a job
    timeout. Job 253 lost observe/guided rows that were only written at session end."""

    def __init__(self, path):
        super().__init__()
        self.f = open(path, "w")

    def append(self, row):
        super().append(row)
        try:
            self.f.write(json.dumps(row) + "\n")
            self.f.flush()
        except Exception:  # noqa: BLE001 - observation must never break the run
            pass

    def close(self):
        self.f.close()


FC_FIX_FLAGS = ("S2S_FIX_CONTEXT_PAD", "S2S_FIX_FC_TOKEN_SYNC")


def apply_fc_fix_defaults(environ, disable: bool) -> dict:
    """Post-FC runtime fix flags. Lab default since 2026-11-12 (owner decision after jobs 277-279): **ON**
    (S2S_FIX_CONTEXT_PAD=1, S2S_FIX_FC_TOKEN_SYNC=all), the training-faithful resume contract (PAD, EOTR).

    History:
    - ON after our GPU validation (fc-fix-validation.md);
    - OFF from 14:30, after EOTR re-injection loops (other lab on the official checkpoint; our re-baseline);
    - ON again after the factorial replays (fc-resume-contract.md §11). Those loops are function-head decisions
      after "need info" results, which the official checkpoint shows too. (PAD, EOTR) keeps the legitimate m07
      chain, while (PAD, PAD) loses it on the official checkpoint.

    Rollback: --no-fc-fix forces both to 0 (the upstream NVIDIA contract). An explicit env value always wins. The
    effective flags go into summary.json `runtime_fix` of every session.

    LEGACY on this repository's engine (owner 2026-11-13, D4): the flags act only on the frozen lab NeMo patch. On a
    clean NeMo they do nothing; the engine's own `fc_ctx_fix` (resume (PAD, PAD), default ON for the official
    checkpoint) does the hand-back. Run with --no-fc-fix there. `runtime_fix` records the engine fix and the
    `resume_contract` actually used (live_gates.resume_contract); a run with both fixes active is refused."""
    for k, on in zip(FC_FIX_FLAGS, ("1", "all")):
        if disable:
            environ[k] = "0"
        else:
            environ.setdefault(k, on)
    return {k: environ[k] for k in FC_FIX_FLAGS}


def write_run_config(out, eng, runtime_fix: dict) -> dict:
    """Provenance of a live run (owner 2026-11-13, C3): engine commit + source hash, NeMo commit, checkpoint, preset,
    the engine's FC hand-back fix, the resume contract and the effective config. The legacy lab flags are recorded with
    whether they had any effect, so they can never be mistaken for an active fix."""
    import hashlib
    import os
    import subprocess
    import mynah_v2
    from omegaconf import OmegaConf
    src = Path(mynah_v2.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(src.rglob("*.py")):
        h.update(f.relative_to(src).as_posix().encode() + b"\0" + f.read_bytes())
    git = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = OmegaConf.to_container(eng.cfg, resolve=True)
    rc = {"engine_commit": os.environ.get("MYNAH_ENGINE_COMMIT") or git or None, "engine_src": str(src),
          "engine_src_sha256": h.hexdigest(), "nemo_git": runtime_fix.get("nemo_git"), "nemo_dirty": runtime_fix.get("nemo_dirty"),
          "checkpoint": cfg.get("s2s", {}).get("llm_checkpoint_path") if isinstance(cfg, dict) else None,
          "preset": "none (ReferenceEngine, llm_cache=True, m0_runner defaults)",
          "exp": runtime_fix.get("exp"), "exp_info": runtime_fix.get("exp_info"),
          "engine_fc_ctx_fix": runtime_fix.get("engine_fc_ctx_fix"), "resume_contract": runtime_fix.get("resume_contract"),
          "legacy_lab_flags": {k: runtime_fix.get(k) for k in ("S2S_FIX_CONTEXT_PAD", "S2S_FIX_FC_TOKEN_SYNC")},
          "legacy_lab_flags_effective": runtime_fix.get("legacy_lab_flags_effective"), "effective_cfg": cfg}
    (out / "run_config.json").write_text(json.dumps(rc, indent=1, default=str))
    return rc


def engine_fix_summary(eng) -> dict:
    _fx = getattr(eng, "fc_ctx_fix_info", None) or {}
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in _fx.items() if k != "fix"}
    if isinstance(_fx.get("fix"), dict):
        d.update({k: _fx["fix"].get(k) for k in ("pad_id", "func_pad", "eotr_step")},
                 channels=list(_fx["fix"].get("channels", ())))
    return d


FC_RESUME = {  # the engine's own fc_ctx_fix, parameterised (no engine change); owner 2026-11-13: the 2x2 resume matrix
    "pad_pad": dict(pad_fill=True, channels=("gen_text", "gen_asr_text"), func_pad=True),     # the engine default
    "pad_eotr": dict(pad_fill=True, channels=("gen_text", "gen_asr_text", "gen_function_text"), func_pad=False),
    # EXPERIMENT (owner GO 2026-11-13): the training layout's inserted EOTR slot, then (PAD, PAD)
    "eotr_step": dict(pad_fill=True, channels=("gen_text", "gen_asr_text", "gen_function_text"), func_pad=False,
                      eotr_step=True),
}


# Guard metadata for the tenant tools, kept OUT of the tool schemas: the FC template serialises the schemas into the
# prompt, so an x-source key there would change the model's input. Ids the backend issues must come from a result.
TENANT_GUARD = {"schemas": {"prepare_return": {"properties": {"customer_id": {"x-source": "result"}}},
                            "commit_return": {"properties": {"pending_id": {"x-source": "result"}}}},
                "side_effects": {"commit_return": "pending_id", "escalate_to_human": None}}


def set_fc_resume(eng, contract: str) -> None:
    """Re-arm the engine's FC hand-back fix with the given resume contract before a session ('off' = upstream)."""
    from mynah_v2.model import fc_ctx_fix as FX
    prev = getattr(eng, "fc_ctx_fix_info", None) or {}
    if getattr(eng.wrapper, "_v2_fc_ctx_fix", None) is not None:
        FX.disable(eng)
    info = FX.enable(eng, **FC_RESUME[contract]) if contract != "off" else None
    eng.fc_ctx_fix_info = {"mode": f"runner:{contract}", "official_checkpoint": prev.get("official_checkpoint"),
                           "enabled": contract != "off", **({"fix": info} if info else {})}


FALLBACK_OFF_FRAMES = 10 ** 9   # first_turn_fallback_frames when the first-turn fallback is disabled (config only)


def parse_plan(plan: str, default_lead: float) -> list[list]:
    """'label:langs:scenarios[:lead_s[:fallback[:opening]]]' blocks.

    fallback is in {on, off} (the first-turn RNNT fallback).
    opening is in {native, guided}; guided replays baseline A's opener (guided_open.py, --guided-trace).
    """
    out = []
    for b in plan.split():
        f = b.split(":")
        lead = float(f[3]) if len(f) > 3 and f[3] else default_lead
        fb = f[4] if len(f) > 4 and f[4] else "on"
        op = f[5] if len(f) > 5 and f[5] else "native"
        if fb not in ("on", "off"):
            raise ValueError(f"fallback must be on/off: {b}")
        if op not in ("native", "guided"):
            raise ValueError(f"opening must be native/guided: {b}")
        out.append([f[0], f[1], f[2], lead, fb, op])
    return out


TURN_CFG_KEYS = ("force_turn_taking", "turn_taking_source", "force_turn_taking_threshold", "rnnt_eou_frames",
                 "rnnt_bou_frames", "asr_eou_first_turn", "asr_min_speech_frames_first_turn",
                 "first_turn_fallback_frames", "force_eos_after_frames", "post_eos_fallback_frames",
                 "rnnt_self_play_suppression", "max_agent_response_sec", "tts_text_token_ratio_cap")


def set_model_cfg(cfg, key, value):
    """Set a key on the wrapper's live model config (read per frame by the turn-taking code); returns the old value
    (None if absent). Configuration only, no engine code change."""
    old = cfg.get(key, None)
    try:
        from omegaconf import open_dict
        with open_dict(cfg):
            if value is None:
                cfg.pop(key, None)
            else:
                cfg[key] = value
    except Exception:  # noqa: BLE001 - plain dict
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    return old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True, help="render_qa output dir (manifest.jsonl, ids <sid>_t<k>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="en")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--mode", default="enhanced", choices=["enhanced", "reference"])
    ap.add_argument("--voice", default="vivian")
    ap.add_argument("--voice-nonen", default="ryan", help="TTS voice of the IT/FR user turns (owner: Ryan for IT/FR)")
    ap.add_argument("--gap-s", type=float, default=22.0)
    ap.add_argument("--lead-s", type=float, default=7.0,
                    help="silence before the caller speaks; 2 s (first M0-smoke) made the caller talk over the greeting")
    ap.add_argument("--spec", default=str(HERE / "tenant_furniture.json"))
    ap.add_argument("--force-customer", default=None,
                    help="PAYLOAD-ISOLATION test only: find_customer always returns this verified customer")
    ap.add_argument("--find-formats", default="nested",
                    help="comma list of find_customer payload shapes (tenant_backend.FIND_FORMATS), run in turn")
    ap.add_argument("--tail-s", type=float, default=20.0)
    ap.add_argument("--tool-timeout", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-extra-s", type=float, default=240.0,
                    help="run_paced drain cap after the input (V2 default 90 s cut two post-tool answers on GB10)")
    ap.add_argument("--variant", default="vc", help="label written into the records")
    ap.add_argument("--plan", default="",
                    help="M1: space-separated 'label:langs:scenarios' blocks run in order in ONE process; label 'vc' = "
                         "original front-end, other labels must be defined in --frontend-defs")
    ap.add_argument("--frontend-defs", default="", help="'label=spec' pairs separated by '|' (e2e_qa variant syntax)")
    ap.add_argument("--frontend", default="vc",
                    help="perception front-end, e2e_qa.py variant syntax: 'vc' (original) or e.g. "
                         "'enc=ml;att=56;rnnt=ml;ln=<encoder_ln_best.pt>;adapter=<adapter_best.pt>' (E5 / V2.1 dual "
                         "encoder: RNNT on the frozen ML encoder, LLM path through the trained LN copy + adapter)")
    ap.add_argument("--observe", action="store_true",
                    help="record forcing (pre/post + RNNT counters), head log-probs and features to observe.jsonl")
    ap.add_argument("--guided-trace", default="",
                    help="'scenario=trace.jsonl,...' baseline-A traces whose opener (frames 0..--guided-upto) is "
                         "replayed in blocks with opening=guided")
    ap.add_argument("--guided-upto", type=int, default=94)
    ap.add_argument("--budget-s", type=float, default=0.0,
                    help="do not START a session if elapsed + the longest session so far would exceed this (0 = off)")
    ap.add_argument("--no-fc-fix", action="store_true",
                    help="force the post-FC runtime fix flags to 0 (lab default is already OFF since 2026-11-12 14:30)")
    ap.add_argument("--call-guard", action="store_true",
                    help="(legacy, now the engine's ENHANCED default) call registry + admission control")
    ap.add_argument("--no-call-guard", action="store_true", help="ENHANCED without the engine call guard")
    ap.add_argument("--asr-shadow", action="store_true",
                    help="EXPERIMENT (with --call-guard): compare call slots / intents with the caller ASR, observe only")
    ap.add_argument("--guard-compact", action="store_true",
                    help="EXPERIMENT (with --call-guard): short refusal responses (fewer FC phase-2 steps)")
    ap.add_argument("--perf-profile", default="", help="mynah_v2 perf profile applied with --exp (e.g. optimized-graph)")
    ap.add_argument("--exp", default="",
                    help="comma list of mynah_v2 perf experiments applied once after load (e.g. the release preset's "
                         "list, which includes the fast FC phase 2); empty = none, as before")
    ap.add_argument("--fc-resume-by-label", default="",
                    help="'label=pad_pad|label=pad_eotr|label=eotr_step|label=off': re-arm the engine fc_ctx_fix per plan label; "
                         "'label=guard:by_outcome' (or guard:pad_pad / guard:pad_eotr) sets the guarded session's contract")
    ap.add_argument("--en-dir", default=str(LAB_HOME / "ckpt/asr-en"))
    ap.add_argument("--ml-dir", default=str(LAB_HOME / "ckpt/asr-ml"))
    a = ap.parse_args()
    import os as _os0
    fc_flags = apply_fc_fix_defaults(_os0.environ, a.no_fc_fix)
    out = Path(a.out)
    if (out / "records.jsonl").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    T_START = time.monotonic()
    spec = json.load(open(a.spec))
    scen = [s for s in spec["scenarios"] if not a.scenarios or s["id"] in a.scenarios.split(",")]
    root = Path(a.audio_root)
    man = {(r["id"], r["lang"], r["voice"]): r for r in map(json.loads, open(root / "manifest.jsonl"))}
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")), llm_cache=True)
    print(f"engine loaded in {eng.load_s:.1f}s", flush=True)
    exp_info = None
    if a.exp or a.perf_profile:
        if a.mode != "enhanced":
            raise SystemExit("--exp is ENHANCED only")
        from mynah_v2 import presets as _presets
        exp_info = _presets.apply(eng, a.perf_profile or None, a.exp, strict=False)
        print("EXP", json.dumps(exp_info, default=str)[:2000], flush=True)
    old_fb_default = eng.wrapper.model_cfg.get("first_turn_fallback_frames", None)   # None = code default 50
    import os as _os
    import nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline as _pl
    runtime_fix = {"nemo_pipeline_file": _pl.__file__,                 # which NeMo checkout actually ran
                   "patched": hasattr(_pl, "fc_sync_token_buffers"),
                   "S2S_FIX_CONTEXT_PAD": _os.environ.get("S2S_FIX_CONTEXT_PAD", "0"),
                   "S2S_FIX_FC_TOKEN_SYNC": _os.environ.get("S2S_FIX_FC_TOKEN_SYNC", "0"),
                   "context_text_pad_id": None}
    try:                                                             # provenance: NeMo commit + V2 snapshot actually used
        import subprocess as _sp
        import mynah_v2 as _v2
        _nroot = Path(_pl.__file__).resolve().parents[5]
        runtime_fix["nemo_git"] = _sp.run(["git", "-C", str(_nroot), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
        runtime_fix["nemo_dirty"] = bool(_sp.run(["git", "-C", str(_nroot), "status", "--porcelain"],
                                                 capture_output=True, text=True).stdout.strip())
        runtime_fix["v2_src"] = str(Path(_v2.__file__).resolve().parent)
    except Exception as _e:  # noqa: BLE001 - provenance must never break a run
        runtime_fix["provenance_error"] = repr(_e)
    runtime_fix["engine_fc_ctx_fix"] = engine_fix_summary(eng)     # the engine's own FC hand-back fix (M4: recorded)
    from live_gates import DOUBLE, resume_contract
    runtime_fix["resume_contract"] = resume_contract(runtime_fix)
    runtime_fix["exp"], runtime_fix["exp_info"] = (a.perf_profile, a.exp) if exp_info else None, exp_info
    print("RUNTIME_FIX", runtime_fix, flush=True)
    if runtime_fix["resume_contract"] == DOUBLE:
        raise SystemExit("both FC hand-back fixes are active (lab NeMo patch + engine fc_ctx_fix): refused, results "
                         "would be mislabelled. On this engine pass --no-fc-fix and use a clean NeMo")
    runtime_fix["legacy_lab_flags_effective"] = bool(runtime_fix["patched"])   # S2S_FIX_* act only on the lab patch
    write_run_config(out, eng, runtime_fix)
    if any(v != "0" for v in fc_flags.values()) and not runtime_fix["patched"]:
        raise SystemExit("legacy lab post-FC flags requested but the NeMo on PYTHONPATH is not the lab patch. On this "
                         "repository's engine pass --no-fc-fix: the engine's fc_ctx_fix does the hand-back (PAD, PAD)")
    defs = {"vc": "vc"}
    for kv in filter(None, a.frontend_defs.split("|")):
        k, v = kv.split("=", 1)
        defs[k] = v
    if a.plan:  # 'label:langs:scenarios[:lead_s[:fallback[:opening]]]'
        blocks = parse_plan(a.plan, a.lead_s)
    else:
        defs.setdefault(a.variant, a.frontend)
        blocks = [[a.variant, a.langs, a.scenarios or ",".join(s["id"] for s in scen), a.lead_s, "on", "native"]]
    gtraces = dict(kv.split("=", 1) for kv in filter(None, a.guided_trace.split(",")))
    sys.path.insert(0, str(HERE))
    import guided_open
    openers = {}
    for b in blocks:   # fail fast, before any session: every guided scenario needs an A trace with frames 0..upto
        if b[5] == "guided":
            for sid_ in b[2].split(","):
                if sid_ not in gtraces:
                    raise SystemExit(f"--guided-trace has no trace for {sid_}")
                openers[sid_] = guided_open.opener_from_trace(gtraces[sid_], upto=a.guided_upto)
    fe = ckpt = None
    if any(defs[b[0]] != "vc" for b in blocks):  # Gate A integration: same FrontEnd swap as e2e_qa.py
        sys.path.insert(0, str(HERE.parent / "multilingual"))
        from e2e_qa import apply_variant, parse_variant
        from perception_swap import FrontEnd, load_nemo_dir
        fe = FrontEnd(eng)
        fe.snapshot()
        ckpt = {"en": load_nemo_dir(a.en_dir), "ml": load_nemo_dir(a.ml_dir)}
    by_id = {s["id"]: s for s in spec["scenarios"]}
    import re as _re
    runs = [(lab, lang, by_id[sid], ff, lead, fb, op) for lab, langs, sids, lead, fb, op in blocks
            for lang in langs.split(",") for sid in sids.split(",") for ff in a.find_formats.split(",")]
    current = None
    longest = 0.0
    budget_log = {"skipped": [], "sessions": []}
    for ri, (lab, lang, sc, ff, lead_s, fb, op) in enumerate(runs):
        if a.budget_s and ri > 0 and time.monotonic() - T_START + longest * 1.1 > a.budget_s:
            budget_log["skipped"] = [[r[0], r[1], r[2]["id"], r[5], r[6]] for r in runs[ri:]]
            print("BUDGET: not starting", len(runs) - ri, "sessions", flush=True)
            break
        if lab != current:
            if fe is not None:
                apply_variant(fe, parse_variant(defs[lab]), ckpt)   # restores the original first
            current = lab
            print("FRONTEND", lab, defs[lab], flush=True)
        frontend_info = defs[lab]
        if True:
            sid = f"{lab}_{lang}_{sc['id']}" + ("" if ff == "nested" and "," not in a.find_formats else f"_{ff}") + \
                ("" if lead_s == a.lead_s else f"_lead{lead_s:g}") + ("" if fb == "on" else "_fboff") + \
                ("" if op == "native" else "_guided")
            od = out / sid
            if od.exists():
                raise SystemExit(f"{od} exists (duplicate plan block?)")
            od.mkdir(parents=True)
            parts = [torch.zeros(int(lead_s * 16000))]
            bounds, tcur = [], lead_s
            for k in range(len(sc["turns"])):
                voice = a.voice if lang == "en" else a.voice_nonen
                x = load_16k(root / man[(f"{sc['id']}_t{k}", lang, voice)]["path"])
                bounds.append([round(tcur, 2), round(tcur + x.numel() / 16000, 2)])
                tcur += x.numel() / 16000 + (a.gap_s if k + 1 < len(sc["turns"]) else 0)
                parts.append(x)
                if k + 1 < len(sc["turns"]):
                    parts.append(torch.zeros(int(a.gap_s * 16000)))
            wav = torch.cat(parts)
            act = {"n": 0, "said": [], "fstart": []}  # caller speech events + RNN-T text (backend confirmation)
            backend = TenantBackend(spec, user_activity=lambda: act["n"], user_transcript=lambda: act["said"],
                                    find_format=ff, force_customer=a.force_customer)
            reg = ToolRegistry()
            for t in spec["tools"]:
                reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
            prompt = render_prompt(reg.schemas, spec["system_prompt"])
            tr = Tracer(od / "trace.jsonl", session_id=sid)
            _emit = tr.emit


            def emit(ev, _emit=_emit, **kw):  # instance-level hook, no V2 code change
                if ev == "asr.generated" and _re.search(r"[A-Za-z]", kw.get("text", "")):
                    act["n"] += 1
                    act["said"].append(kw.get("text", ""))
                if ev == "function.started":
                    act["fstart"].append(kw.get("model_step"))
                return _emit(ev, **kw)
            tr.emit = emit
            led = ToolLedger(sid, tr)
            state = {"s": None}
            bound = reg.bind(led, lambda: state["s"].turn_id if state["s"] else 0,
                             lambda: state["s"].model_step if state["s"] else None,
                             timeout_s=a.tool_timeout if a.mode == "enhanced" else None)
            eng.bind_session_tools(dict(bound))
            chunks = []
            enh = EnhancedOptions() if a.mode == "enhanced" else None
            if enh is not None:                # tenant metadata kept OUT of the schemas (prompt unchanged)
                enh.call_guard = not a.no_call_guard
                enh.guard_schemas = TENANT_GUARD["schemas"]
                enh.guard_side_effects = TENANT_GUARD["side_effects"]
                enh.guard_style = "compact" if a.guard_compact else "full"
                _rbl = dict(kv.split("=", 1) for kv in filter(None, a.fc_resume_by_label.split("|")))
                if str(_rbl.get(lab, "")).startswith("guard:"):   # the guarded session's own contract (e.g. by_outcome)
                    enh.resume_with_guard = _rbl[lab].split(":", 1)[1]
                elif a.fc_resume_by_label:     # an explicit lab contract per label wins over the guarded default
                    enh.resume_with_guard = None
            sess = Session(eng, tr, Mode(a.mode), enh, on_audio=lambda x: chunks.append(x.numpy()),
                           controller=Controller(prompt, ledger=led, tools=dict(eng.pipeline.tool_registry)))
            state["s"] = sess
            random.seed(a.seed)
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            old_fb = set_model_cfg(eng.wrapper.model_cfg, "first_turn_fallback_frames",
                                   FALLBACK_OFF_FRAMES if fb == "off" else old_fb_default)
            fb_effective = eng.wrapper.model_cfg.get("first_turn_fallback_frames", "default(50)")
            obs_rows = JsonlSink(od / "observe.jsonl") if a.observe else []
            guided_rows = JsonlSink(od / "guided.jsonl") if op == "guided" else []
            un_obs, un_guided = None, None
            if op == "guided":   # BEFORE observe: observe then logs pre = the Thinker's token, post = after injection
                un_guided = guided_open.install(eng.wrapper, openers[sc["id"]], guided_rows)
            turn_cfg = {k: eng.wrapper.model_cfg.get(k, None) for k in TURN_CFG_KEYS}
            (od / "turn_cfg.json").write_text(json.dumps(turn_cfg, default=str))   # before the session (timeouts)
            if a.observe:
                sys.path.insert(0, str(HERE))
                import observe
                un_obs = observe.install(eng.wrapper, obs_rows)
            res_by_label = dict(kv.split("=", 1) for kv in filter(None, a.fc_resume_by_label.split("|")))
            if lab in res_by_label and not res_by_label[lab].startswith("guard:"):
                set_fc_resume(eng, res_by_label[lab])
            sess_fix = dict(runtime_fix, engine_fc_ctx_fix=engine_fix_summary(eng))
            sess_fix["resume_contract"] = resume_contract(sess_fix)
            print("SESSION_FIX", lab, sess_fix["resume_contract"], sess_fix["engine_fc_ctx_fix"], flush=True)
            if sess_fix["resume_contract"] == DOUBLE:
                raise SystemExit("both FC hand-back fixes active for this session: refused")
            _rs = ((getattr(eng, "fc_ctx_fix_info", None) or {}).get("fix") or {}).get("stats") or {}
            n_resume0 = len(_rs.get("resume_inputs", []))
            n_eotr0 = len(_rs.get("eotr_steps", []))
            t0 = time.monotonic()
            sess.open(prompt)
            run_paced(sess, wav, pace="realtime", tail_silence_s=a.tail_s, max_extra_s=a.max_extra_s, drain=True)
            wall = time.monotonic() - t0
            sess.close("normal")
            cg = sess.guard_summary
            tr.close()
            import live_gates
            fcev = live_gates.fc_events(od / "trace.jsonl")              # FC openings, calls, results, resumes (M4/C3)
            resume_inputs = list(_rs.get("resume_inputs", []))[n_resume0:]
            eotr_steps = list(_rs.get("eotr_steps", []))[n_eotr0:]
            if un_obs is not None:
                un_obs()
                obs_rows.close()
            if un_guided is not None:
                un_guided()
                guided_rows.close()
            set_model_cfg(eng.wrapper.model_cfg, "first_turn_fallback_frames", old_fb)
            longest = max(longest, time.monotonic() - t0)
            agent = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            sf.write(od / "output.wav", agent, T.OUTPUT_SR)
            calls = [{"name": c.name, "args": c.arguments, "status": c.status.value,
                      "result": json.loads(c.result) if c.result and c.result.startswith("{") else c.result,
                      "model_step": c.model_step} for c in led.calls.values()]
            # premature call: the first function start (model step = 80 ms, aligned with the input before any call)
            # versus the end of the caller turn it belongs to
            f0 = act["fstart"][0] * 0.08 if act["fstart"] and act["fstart"][0] is not None else None
            prem = None if f0 is None else any(b0 <= f0 < b1 for b0, b1 in bounds)
            rec = {"scenario": sc["id"], "tier": sc["tier"], "lang": lang, "variant": lab, "mode": a.mode,
                   "turn_bounds_s": bounds, "first_call_s": f0, "premature_call": prem,
                   "user_rnnt": "".join(act["said"]),
                   "frontend": frontend_info, "find_format": ff, "force_customer": a.force_customer, "spec": spec["name"], "lead_s": lead_s,
                   "first_turn_fallback": fb, "first_turn_fallback_frames": fb_effective,
                   "observe_rows": len(obs_rows), "opening": op, "turn_cfg": turn_cfg,
                   "runtime_fix": dict(sess_fix, context_text_pad_id=getattr(
                       getattr(eng.pipeline, "context_manager", None), "text_pad_id", None)),
                   "guided_open": (guided_open.summarise(guided_rows, openers[sc["id"]]) if op == "guided" else None),
                   "calls": calls, "audit": backend.audit, "agent_text": "".join(sess.text),
                   "function_text": "".join(sess.func_text), "wall_s": round(wall, 1),
                   "audio_in_s": round(wav.numel() / 16000, 1), "audio_out_s": round(len(agent) / T.OUTPUT_SR, 1),
                   "stats": {k: v for k, v in vars(sess.stats).items() if k not in ("step_ms", "step_ms_model")},
                   "grounding_violations": sess.controller.grounding_violations,
                   "fc_events": fcev, "fc_resume_inputs": resume_inputs, "fc_eotr_steps": eotr_steps,
                   "call_guard": cg}
            (od / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            with open(out / "records.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps({k: rec[k] for k in ("variant", "scenario", "lang", "first_turn_fallback", "opening", "calls",
                                                  "agent_text", "wall_s")}, ensure_ascii=False)[:900], flush=True)
            budget_log["sessions"].append([lab, lang, sc["id"], fb, op, round(wall, 1)])
    budget_log["elapsed_s"] = round(time.monotonic() - T_START, 1)
    (out / "budget.json").write_text(json.dumps(budget_log, indent=1))


if __name__ == "__main__":
    main()
