"""Load the shared resources of a Frankenstein v2 server (or lab run) from command-line options.

Placement and memory are options, so the same code serves a 24 GB card, a 48 GB card and a unified-memory host:
  --ears-device cuda|cpu        the ears process (Nemotron 0.6B fp32 + GRU): ~2.5 GB on the device
  --nano-precision bf16|int4-decode   the text LLM (bf16 weights ~17 GB; int4-decode adds INT4 copies for decode)
  --talker-device cuda|cpu      Pocket (24-layer: ~1.5 GB per loaded language on the device)
  --pocket-max-langs N          languages kept loaded by the Pocket process (LRU; pocket_swap will plan this)
  --threads N                   CPU threads per process (torch intra-op), for hosts with few vCPUs
Every process reports its own CUDA allocated / reserved / peak and host RSS (``mem`` op), so the per-component
footprint is measured, not assumed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def add_args(ap) -> None:
    ap.add_argument("--lab-scripts", required=True, help="agentbench + multilingual lab scripts (tenant tools)")
    ap.add_argument("--tenant", default="tenant_furniture.json")
    ap.add_argument("--nano-dir", default="")
    ap.add_argument("--llm-backend", default="qwen", choices=["nano", "qwen"],
                    help="nano = Nemotron Nano 9B v2 in this process; qwen = a Qwen3.5 process (own inference engine "
                         "if --qwen-src points at it, else HF generate)")
    ap.add_argument("--qwen-python", default="")
    ap.add_argument("--qwen-src", default="", help="root that contains frankenstein2.llm.qwen_own (own engine)")
    ap.add_argument("--qwen-model", default="")
    ap.add_argument("--qwen-mode", default="int8", help="own engine weights: int8 (default) | bf16 | MoE AWQ 4-bit")
    ap.add_argument("--qwen-slots", type=int, default=8)
    ap.add_argument("--nano-precision", default="int4-decode", choices=["bf16", "int4-decode"])
    ap.add_argument("--nano-graph", action="store_true", help="decode CUDA graph for the spoken reply (v1 G8)")
    ap.add_argument("--ears-python", required=True)
    ap.add_argument("--ears-bundle", required=True)
    ap.add_argument("--ears-device", default="cuda")
    ap.add_argument("--ears-procs", type=int, default=1, help="ears processes; sessions are spread across them")
    ap.add_argument("--talker", default="swap", choices=["pocket", "swap", "flash"],
                    help="pocket = one Pocket model per language; swap = pocket_swap (shared base + language packages)")
    ap.add_argument("--swap-root", default=str(ROOT / "pocket_swap"))
    ap.add_argument("--swap-base", default="", help="pocket_swap base checkpoint (english_2026-09_24l)")
    ap.add_argument("--swap-packages", default="", help="pocket_swap language packages: w4 (default choice), w8, wfp8 or bf16")
    ap.add_argument("--swap-mode", default="ondemand", choices=["ondemand", "materialise"],
                    help="ondemand = mode (b): compressed language terms applied on the fly (default)")
    ap.add_argument("--swap-preset", default="l40s-16g", help="pocket_swap preset: l40s-16g | gb10 | l4-24g")
    ap.add_argument("--pocket-python", default="")
    ap.add_argument("--pocket-hf-home", default="")
    ap.add_argument("--talker-device", default="cuda")
    ap.add_argument("--pocket-max-langs", type=int, default=3)
    ap.add_argument("--flash-python", default="")
    ap.add_argument("--flash-harness", default="")
    ap.add_argument("--flash-root", default="")
    ap.add_argument("--voice", action="append", default=[], help="NAME=reference.wav (first = default)")
    ap.add_argument("--chunking", default="", help="talker chunking override: sentence | whole | progressive")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ep-thr", action="append", default=[], help="LANG=threshold for the endpoint head")
    ap.add_argument("--tool-timeout", type=float, default=8.0,
                    help="seconds a tool may run; then the model gets the reference timeout error (late results are "
                         "recorded in the ledger, never used)")
    ap.add_argument("--tool-reminder-s", type=float, default=2.5,
                    help="an on-hold reminder every N seconds while a tool runs (0 = none); the tool starts ~2 s after "
                         "the holding phrase (call generation), so 4 s left 4-5 s of silence and callers spoke over it")
    ap.add_argument("--tool-delay-s", type=float, default=0.0, help="lab: every tool backend sleeps N seconds first")
    ap.add_argument("--no-tool-rule", dest="tool_rule", action="store_false",
                    help="v2b: without the tool rule in the system prompt and the localized decision question")
    ap.add_argument("--runtime", default="procs", choices=["procs", "single"],
                    help="single = ONE process owns the GPU: our own ears, Qwen and Pocket runtimes under one scheduler "
                         "(frankenstein2/runtime.py); procs = the ears / talker / Qwen processes")
    ap.add_argument("--pocket-weights", default="", help="single: the Pocket cloning snapshot (languages/<lang>/model.safetensors)")
    ap.add_argument("--pocket-tokenizers", default="", help="single: where languages/<lang>/tokenizer.json are")
    ap.add_argument("--pocket-preload", default="it,en,fr", help="single: talker languages loaded at start")
    ap.add_argument("--talker-budget-ms", type=float, default=40.0)
    ap.add_argument("--child-env", action="append", default=[],
                    help="KEY=VALUE for the ears and talker processes (their own environments' caches / headers)")


def child_env(a, extra: dict | None = None) -> dict:
    """The parent's environment minus PYTHONPATH and HF_HUB_CACHE (a child environment has its own cache, given
    as HF_HOME through --child-env / --pocket-hf-home), plus --child-env."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "HF_HUB_CACHE")}
    env.update(dict(x.split("=", 1) for x in getattr(a, "child_env", []) or []))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if a.threads:
        env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    env.update(extra or {})
    return env


def _single(a, log_dir):  # pragma: no cover - GPU
    if getattr(a, "_single", None) is None:
        from frankenstein2 import runtime
        a._single = runtime.start(a, log_dir)
    return a._single


def start_ears(a, log_dir: Path):  # pragma: no cover - process
    """One ears process, or a pool of --ears-procs processes (sessions spread across them)."""
    if a.runtime == "single":
        return _single(a, log_dir)["ears"]
    from frankenstein2.ears.client import EarsPool, EarsProc
    procs = []
    for i in range(max(1, a.ears_procs)):
        p = EarsProc([a.ears_python, str(ROOT / "frankenstein2" / "ears" / "server.py"), "--bundle", a.ears_bundle,
                      "--device", a.ears_device], child_env(a),
                     str(log_dir / ("ears_server.log" if i == 0 else f"ears_server_{i}.log")))
        p.start()
        p.wait_ready(600)                         # one at a time: never two model loads at once
        procs.append(p)
    return procs[0] if len(procs) == 1 else EarsPool(procs)


def start_pocket(a, log_dir: Path):  # pragma: no cover - process
    if a.runtime == "single":
        return _single(a, log_dir)["talker"]
    from frankenstein2.tts.procs import PocketProc
    extra = {"HF_HOME": a.pocket_hf_home} if a.pocket_hf_home else {}
    thr = ["--threads", str(a.threads)] if a.threads else []
    if a.talker == "swap":                       # pocket_swap's own talker process (same JSON-lines protocol)
        argv = [a.pocket_python, "-m", "pocket_swap.talker_server", "--base", a.swap_base, "--packages",
                a.swap_packages, "--mode", a.swap_mode, "--preset", a.swap_preset, "--device", a.talker_device]
        extra = dict(extra, PYTHONPATH=a.swap_root)
    else:
        argv = [a.pocket_python, str(ROOT / "frankenstein2" / "tts" / "pocket_server.py"), "--device",
                a.talker_device, "--max-langs", str(a.pocket_max_langs), *thr]
    p = PocketProc(argv, child_env(a, extra), str(log_dir / f"{a.talker}_server.log"))
    p.start()
    p.wait_ready(600)
    return p


def start_qwen(a, log_dir: Path):  # pragma: no cover - process
    """The Qwen LLM process: our own inference engine (frankenstein2.llm.qwen_own.server under --qwen-src) or the HF
    generate stand-in (llm/qwen_server.py); both speak the same JSON lines."""
    if a.runtime == "single":
        return _single(a, log_dir)["llm"]
    from frankenstein2.tts.procs import PocketProc               # generic JSON-lines process, replies routed by "ss"
    if a.qwen_src:
        argv = [a.qwen_python, "-m", "frankenstein2.llm.qwen_own.server", "--model", a.qwen_model, "--mode",
                a.qwen_mode, "--slots", str(a.qwen_slots)]
        env = child_env(a, {"PYTHONPATH": a.qwen_src})
    else:
        argv = [a.qwen_python, str(ROOT / "frankenstein2" / "llm" / "qwen_server.py"), "--model", a.qwen_model]
        env = child_env(a)
    p = PocketProc(argv, env, str(log_dir / "qwen_server.log"))
    p.start()
    p.wait_ready(900)
    return p


def slowed(handlers: dict, delay_s: float) -> dict:
    """Lab: tool backends that take ``delay_s`` first (a slow CRM / network), to exercise the async tool path."""
    if not delay_s:
        return handlers

    def mk(fn):
        def h(args):
            time.sleep(delay_s)
            return fn(args)
        return h
    return {n: mk(f) for n, f in handlers.items()}


def qwen_factory(a, proc, spec):  # pragma: no cover - process
    sys.path.insert(0, str(ROOT / "frankenstein"))
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from e1.tools_bench import POLICY as P1, tool_schemas
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein2.llm.nano import BRIEF, POLICY, TOOL_RULE
    from frankenstein2.llm.qwen import QwenLLM
    policy = {**P1, **POLICY}

    def make(sid, lang, system_prompt=None):
        L = session_language(lang)
        said: list = []
        backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        backend._said = said
        reg = ToolRegistry()
        hs = slowed(backend.handlers(), getattr(a, "tool_delay_s", 0.0))
        for t in spec["tools"]:
            reg.register(t["name"], hs[t["name"]], t["description"], t["parameters"])
        led = ToolLedger(sid, None)
        tmo = getattr(a, "tool_timeout", 8.0)
        bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=tmo)   # backend in its own thread, ledger timeout
        rule = ("\n" + TOOL_RULE.get(lang, TOOL_RULE["en"])) if a.tool_rule else ""
        system = (with_language(system_prompt or spec["system_prompt"], L) + "\n" + policy.get(lang, policy["en"])
                  + "\n" + BRIEF.get(lang, BRIEF["en"]) + rule)
        llm = QwenLLM(proc, sid, system, tool_schemas(reg), bound, lang, tool_timeout_s=tmo + 1.0,
                      reminder_s=getattr(a, "tool_reminder_s", 2.5))
        llm.backend, llm.ledger = backend, led
        return llm
    return make


def tenant(a):
    lab = Path(a.lab_scripts)
    for d in ("agentbench", "multilingual"):
        if str(lab / d) not in sys.path:
            sys.path.insert(0, str(lab / d))
    return json.load(open(lab / "agentbench" / a.tenant))


def llm_factory(a, shared, spec, on_ledger=None):  # pragma: no cover - GPU
    """-> make_llm(session_id, lang, system_prompt): a NanoLLM with this session's tools, ledger and language."""
    sys.path.insert(0, str(ROOT / "frankenstein"))
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from e1.tools_bench import POLICY as P1, tool_schemas
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein2.llm.nano import BRIEF, POLICY, TOOL_RULE, NanoLLM
    policy = {**P1, **POLICY}

    def make(sid, lang, system_prompt=None):
        L = session_language(lang)
        said: list = []
        backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        backend._said = said
        reg = ToolRegistry()
        hs = slowed(backend.handlers(), getattr(a, "tool_delay_s", 0.0))
        for t in spec["tools"]:
            reg.register(t["name"], hs[t["name"]], t["description"], t["parameters"])
        led = ToolLedger(sid, None)
        bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
        rule = ("\n" + TOOL_RULE.get(lang, TOOL_RULE["en"])) if a.tool_rule else ""
        system = (with_language(system_prompt or spec["system_prompt"], L) + "\n" + policy.get(lang, policy["en"])
                  + "\n" + BRIEF.get(lang, BRIEF["en"]) + rule + " /no_think")
        llm = NanoLLM(shared, system, tool_schemas(reg), bound, lang, backend=backend,
                      localized_step=a.tool_rule)
        llm.backend, llm.ledger = backend, led
        if on_ledger:
            on_ledger(sid, led, backend)
        return llm
    return make


def talker_factory(a, proc):  # pragma: no cover - process
    from frankenstein2.tts.procs import PocketTalker

    def make(sid, lang, voice):
        t = PocketTalker(proc, sid, lang, voice)
        if a.chunking:
            t.chunking = a.chunking
        return t
    return make


def thresholds(a) -> dict:
    out = {}
    for x in a.ep_thr:
        lang, _, v = x.partition("=")
        out[lang] = {"ep_thr": float(v)}
    return out


def voices(a) -> dict:
    return dict(x.split("=", 1) for x in a.voice)
