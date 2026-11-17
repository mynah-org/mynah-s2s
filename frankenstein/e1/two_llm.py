"""E1d part 2: first end-to-end run of the two-LLM architecture on the Italian track (GPU, same process as the bench).

  timing  the VoiceChat duplex Thinker (NVIDIA native pipeline, ReferenceEngine), free-running on the original audio,
          as a timing shell only: the controller takes the turn on the Thinker's BOS OR its tool-call start
          (function channel id 20, suppressed before the engine's FC state machine sees it), with the multilingual-ASR
          end of utterance as fallback, the ASR speech gate, no-new-words gate and an 8-frame debounce after cancels;
          NVIDIA's RNNT forced turn-taking (3.2 s EOU BOS) is switched off for the session; controller-owned EOS.
  content Nemotron-Nano-9B-v2 base (text mode, native chat template + tools, real tenant handlers through the V2
          ToolRegistry / ToolLedger): the reply and the tool calls. The reply ids (same tokenizer as the Thinker) go
          to the talker adapter as a burst; availability = request frame + measured wall time (epoch-safe: ids of a
          cut turn are dropped).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class NanoBrain:
    """M1Force brain interface (run(messages, decide_tool) -> dict) backed by Nano text mode with native tools."""

    def __init__(self, tok, model, gen, system: str, tools: list, bound: dict, backend=None, parse=None):
        self.tok, self.m, self.gen = tok, model, gen
        self.system, self.tools, self.bound, self.backend, self.parse = system, tools, bound, backend, parse
        self.last_cache = None
        self.calls: list = []

    def run(self, messages: list, decide_tool: bool = True) -> dict:
        t0 = time.monotonic()
        user = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        if self.backend is not None and user:
            self.backend._said.append(user)
        calls, log, n_tok, gen_s = [], [], 0, 0.0
        txt = ""
        for _ in range(3):
            msgs = [{"role": "system", "content": self.system}] + messages
            txt, s, ctx = self.gen(self.tok, self.m, "nano", msgs, self.tools)
            gen_s += s
            n_tok += len(self.tok.encode(txt, add_special_tokens=False))
            tc = self.parse(txt)
            log.append({"gen": txt, "s": s, "ctx": ctx})
            if isinstance(tc, list) and tc:
                messages.append({"role": "assistant", "content": txt})
                for c in tc[:2]:
                    name, args = c.get("name"), c.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:  # noqa: BLE001
                            args = {"_raw": args}
                    res = self.bound[name](args) if name in self.bound else json.dumps({"error": f"unknown {name}"})
                    calls.append({"name": name, "args": args, "result": res, "executed": name in self.bound})
                    messages.append({"role": "tool", "content": res})
                continue
            break
        self.calls += calls
        ids = self.tok.encode(" " + txt if txt else "", add_special_tokens=False)
        wall = (time.monotonic() - t0) * 1e3
        return {"ids": ids, "text": txt, "ctx_len": log[-1]["ctx"] if log else 0, "prefill_ms": 0.0,
                "decode_ms_per_tok": round(gen_s * 1e3 / max(1, n_tok), 1), "tool_margin": None, "func_top1": None,
                "tool": None, "calls": calls, "nano_log": log, "wall_ms": round(wall, 1)}


def run_session(out: Path, eng, tok, model, gen, parse, spec, policy_it: str, asr_dir: Path, cut_s: float,
                lab_agentbench: Path, tool_schemas):  # pragma: no cover - GPU
    import sys
    sys.path.insert(0, str(lab_agentbench))
    from m0_runner import load_16k, set_model_cfg
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.m1force import FrameLog, M1Force
    from e1.runner import adapter_summary, make_adapter, pump, seed_all
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein_s2s.tracing import Tracer as FTracer
    w = eng.wrapper
    L = session_language("it")
    asr = json.load(open(asr_dir / "asr_it.json"))
    wav = load_16k(asr_dir / "track_it.wav")[: int(cut_s * 16000)]
    od = out / "two_llm_it"
    od.mkdir()
    said = []
    backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
    backend._said = said
    reg = ToolRegistry()
    for t in spec["tools"]:
        reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
    tr = V2Tracer(od / "trace.jsonl", session_id="two_llm_it")
    led = ToolLedger("two_llm_it", tr)
    bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
    from m0_runner import render_prompt
    duplex_prompt = render_prompt(reg.schemas, with_language(spec["system_prompt"], L))
    brain = NanoBrain(tok, model, gen, spec["system_prompt"] + "\n" + policy_it + " /no_think", tool_schemas(reg),
                      bound, backend, parse)
    ftr = FTracer(od / "fr_trace.jsonl", session_id="two_llm_it")
    ad, drv, gate = make_adapter(tok, ftr, L.talker_lang)
    frames, blog = FrameLog(od / "frames.jsonl"), FrameLog(od / "brain.jsonl")
    seed_all()
    hook = M1Force(w, brain, tok, asr, ad, ftr, frames, blog, lambda p: ("{}", 0.0), mode="shell", tool_turns=False,
                   bos_gate=True, eos_hold=0, debounce_frames=8, no_new_words_gate=True, fallback=True,
                   sotc_turn=True)
    hook.pump = lambda: pump(ad, drv, gate)
    old = set_model_cfg(w.model_cfg, "force_turn_taking", False)
    uninstall = hook.install()
    eng.bind_session_tools({})
    sess = Session(eng, tr, Mode.REFERENCE, None, on_audio=lambda x: None, controller=None)
    t0 = time.monotonic()
    try:
        sess.open(duplex_prompt)
        run_paced(sess, wav, pace="realtime", tail_silence_s=1.0, max_extra_s=10.0, drain=True)
        pump(ad, drv, gate)
        sess.close("normal")
    finally:
        uninstall()
        tr.close()
        set_model_cfg(w.model_cfg, "force_turn_taking", old)
    frames.close()
    blog.close()
    rec = {"arm": "two_llm", "lang": "it", "cut_s": cut_s, "hook_calls": hook.n_hook, "frames_last": hook.last_t,
           "mismatch": hook.mismatch[:50], "n_mismatch": len(hook.mismatch), "gaps": hook.gaps[:50],
           "n_gaps": len(hook.gaps), "fc_suppressed": hook.fc_suppressed, "cache_checks": hook.cache_checks,
           "bos_suppressed": hook.suppressed, "brain_skipped": hook.skipped, "sotc_turns": hook.sotc_turns,
           "forcer": {"overrides": hook.forcer.overrides, "agreements": hook.forcer.agreements,
                      "log": [list(map(str, x)) for x in hook.forcer.log]},
           "messages": hook.messages, "calls": brain.calls, "backend_audit": backend.audit,
           "ledger": [{"name": c.name, "status": c.status.value} for c in led.calls.values()],
           "adapter": adapter_summary(ad, gate), "wall_s": round(time.monotonic() - t0, 1)}
    ftr.close()
    (od / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    return rec
