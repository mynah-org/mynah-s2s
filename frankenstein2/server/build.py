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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def add_args(ap) -> None:
    ap.add_argument("--lab-scripts", required=True, help="agentbench + multilingual lab scripts (tenant tools)")
    ap.add_argument("--tenant", default="tenant_furniture.json")
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--nano-precision", default="int4-decode", choices=["bf16", "int4-decode"])
    ap.add_argument("--nano-graph", action="store_true", help="decode CUDA graph for the spoken reply (v1 G8)")
    ap.add_argument("--ears-python", required=True)
    ap.add_argument("--ears-bundle", required=True)
    ap.add_argument("--ears-device", default="cuda")
    ap.add_argument("--talker", default="pocket", choices=["pocket", "flash"])
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


def child_env(a, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if a.threads:
        env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    env.update(extra or {})
    return env


def start_ears(a, log_dir: Path):  # pragma: no cover - process
    from frankenstein2.ears.client import EarsProc
    p = EarsProc([a.ears_python, str(ROOT / "frankenstein2" / "ears" / "server.py"), "--bundle", a.ears_bundle,
                  "--device", a.ears_device], child_env(a), str(log_dir / "ears_server.log"))
    p.start()
    p.wait_ready(600)
    return p


def start_pocket(a, log_dir: Path):  # pragma: no cover - process
    from frankenstein2.tts.procs import PocketProc
    extra = {"HF_HOME": a.pocket_hf_home} if a.pocket_hf_home else {}
    p = PocketProc([a.pocket_python, str(ROOT / "frankenstein2" / "tts" / "pocket_server.py"), "--device",
                    a.talker_device, "--max-langs", str(a.pocket_max_langs),
                    *(["--threads", str(a.threads)] if a.threads else [])], child_env(a, extra),
                   str(log_dir / "pocket_server.log"))
    p.start()
    p.wait_ready(600)
    return p


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
    from frankenstein2.llm.nano import BRIEF, POLICY, NanoLLM
    policy = {**P1, **POLICY}

    def make(sid, lang, system_prompt=None):
        L = session_language(lang)
        said: list = []
        backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        backend._said = said
        reg = ToolRegistry()
        for t in spec["tools"]:
            reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
        led = ToolLedger(sid, None)
        bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
        system = (with_language(system_prompt or spec["system_prompt"], L) + "\n" + policy.get(lang, policy["en"])
                  + "\n" + BRIEF.get(lang, BRIEF["en"]) + " /no_think")
        llm = NanoLLM(shared, system, tool_schemas(reg), bound, lang, backend=backend)
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
