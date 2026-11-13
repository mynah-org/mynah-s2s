"""Two-session isolation + throughput harness for the V2 true-batching path (runtime/batch_contract.py, I1-I9).

Fixture: eval/two_session_v1.json — groups (pairs + one join/leave/rejoin sequence) of EN corpus_v1/capability_v1 items,
each session with its own rendered WAV, its own orders fixture (conflicting on the ids asked) and its own seed.

Modes (every mode writes a NEW results directory and refuses an existing one):
  solo     runs every session ALONE through scripts/run_stream.py (B=1, one process per session, same settings the pair
           driver gets) -> <out>/<sid>/<item>/{summary.json,trace.jsonl,agent.wav}, <out>/solo_manifest.json.
  pair     runs each group CONCURRENTLY through a pluggable driver, then compares every session with its solo run
           -> <out>/<group>/<sid>/{summary.json,trace.jsonl,agent.wav}, <out>/<group>/driver_result.json,
              <out>/two_session_summary.json.
  compare  CPU only: compares two existing run dirs (pair vs solo, or solo vs solo = determinism noise floor) and
           writes <out>/two_session_summary.json.

Drivers (--driver):
  batched              default: ``mynah_v2.runtime.batch_scheduler:bench_driver`` (scheduler + ThinkerBatchBackend +
                       SessionStateStore; arrives later). Missing -> "UNSUPPORTED", exit 3.
  reference-interleave REFERENCE B=1 streams interleaved chunk by chunk (no batching, a baseline for the scheduler).
                       Runs ONLY if ReferenceEngine exposes ``open_stream(stream_id)``; today it is single-stream, so
                       it exits "UNSUPPORTED" (exit 3) before loading anything. Never faked.
  pkg.module:function  any callable with the driver contract below.

Driver contract: ``driver(group: dict, out_dir: Path, ctx: dict) -> dict``
  group   {"id", "kind", "sessions": [session], "sequence": [...], "expected_compositions": [...]} where session =
          {"sid", "item", "category", "expect", "wav", "tools" ("orders,random" | ""), "orders_rows" (list | None),
           "seed", "tool_delay_s", "leak_regex", "user_leak_regex"}; sequence ticks are 160 ms chunks, a detached
          session consumes no chunks (``Roster`` implements the join/leave/defer/after_done semantics).
  ctx     {"mode", "pace", "tail_silence_s", "chunk_ms", "spec_path", "run_stream": <path>}.
  must write per session <out_dir>/<sid>/summary.json in the run_stream.py schema (``session_summary`` builds it from
  a Session) + trace.jsonl (mynah_v2 Tracer) + agent.wav (22.05 kHz), applying tools/fixture/seed/tool delay exactly
  like run_stream.py; and return
    {"driver": str, "wall_s": float,
     "per_session_rng": bool | None,                     # I8: sampling uses each session's own generator
     "compositions": [{"tick", "sids": [...], "rows": {sid: row} | None}],      # active set per tick (or on change)
     "steps": [{"tick", "ms", "sids": [...], "batched": bool, "phase", "bucket"}],  # one per Thinker forward
     "chunk_log": [{"sid", "chunk_idx", "arrival_ns", "done_ns"}],               # deadline accounting
     "memory": collect_memory(...)}                                              # I9 decomposition inputs
  and raise ``PairUnsupported`` when it cannot run (the harness prints it and exits 3).

Checks per session (pair vs its solo run): tool names + normalized args, args/results looked up in EVERY fixture of the
group (result from another session's fixture = leak), results consumed by this session, leak_regex (other sessions'
fixture facts) in the agent text and user_leak_regex in this session's ASR channel (flagged only if absent in solo),
foreign session ids in the trace, functional score (score_corpus.score_item) pass/regression, greeting step, agent turn
steps, EOS frames, continuation/self-asked turns (run-on vs solo), FC windows (function.started -> tool.result.consumed)
and FC-timeline violations, token/text identity + WER (diagnostic), agent.wav identity (I8: gated only when determinism
is expected). Per group: FC windows really overlapped (p3), composition sequence and row reuse (j1), real B>=2 forwards.
Metrics: aggregate session-chunks/s, per-session step p50/p95, steps over the 160 ms budget, deadline misses
(done - arrival > 160 ms, from chunk_log), forward ms by batch size, memory decomposition (I9) — vs the solo B=1 runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_corpus import decoded, runon_vs_golden, score_item, turn_stats  # noqa: E402
from screen_compare import calls as norm_calls, outcome, trace_info, wer, words  # noqa: E402

DEFAULT_SPEC = HERE.parent / "eval" / "two_session_v1.json"
SUMMARY_NAME = "two_session_summary.json"
BATCHED_DRIVER_ENTRY = "mynah_v2.runtime.batch_scheduler:bench_driver"
EXIT_UNSUPPORTED = 3
EOS_ID = 2               # mynah_v2.model.tokens.EOS_ID (kept literal: no torch import at module level)


class PairUnsupported(RuntimeError):
    """The requested pair driver cannot run on this engine/build (never faked)."""


# ------------------------------------------------------------------------------------------------ spec
def load_spec(path: Path, only: Iterable[str] | None = None) -> dict:
    """Load + validate the fixture; resolve each session against its source item (category, expect, texts)."""
    spec = json.loads(Path(path).read_text())
    if spec.get("lang") != "en":
        raise ValueError("two-session fixture must be English (the official VoiceChat model is EN-only)")
    base = Path(path).parent
    src_items = {}
    for name, rel in spec["sources"].items():
        s = json.loads((base / rel).read_text())
        if s.get("lang") != "en":
            raise ValueError(f"source {name} is not English")
        src_items[name] = {it["id"]: it for it in s["items"]}
    fixtures = spec["fixtures"]
    only = set(only or [])
    groups, seen = [], set()
    for g in spec["groups"]:
        if only and g["id"] not in only:
            continue
        sess = []
        for s in g["sessions"]:
            if s["sid"] in seen:
                raise ValueError(f"duplicate sid {s['sid']}")
            seen.add(s["sid"])
            it = src_items[s["source"]][s["item"]]
            if s.get("fixture") is not None and s["fixture"] not in fixtures:
                raise ValueError(f"{s['sid']}: unknown fixture {s['fixture']}")
            if "orders" in (s.get("tools") or "") and s.get("fixture") is None:
                raise ValueError(f"{s['sid']}: orders tool without a fixture")
            sess.append({**s, "category": it["category"], "expect": s.get("expect", it["expect"]),
                         "texts": [u["text"] for u in it["utterances"]],
                         "orders_rows": fixtures[s["fixture"]] if s.get("fixture") else None})
        items = [(s["source"], s["item"]) for s in sess]
        if len(set(items)) != len(items):
            raise ValueError(f"{g['id']}: sessions must use different audio")
        tooled = [s for s in sess if s.get("tools")]
        fx = [json.dumps(s["orders_rows"], sort_keys=True) for s in tooled if s["orders_rows"] is not None]
        if len(set(fx)) != len(fx):
            raise ValueError(f"{g['id']}: tool sessions must use different fixtures")
        sids = {s["sid"] for s in sess}
        for comp in g.get("expected_compositions", []):
            if not set(comp) <= sids:
                raise ValueError(f"{g['id']}: composition {comp} names unknown sessions")
        for ev in g.get("sequence", []):
            for k in ("join", "leave", "after_done"):
                if k in ev and ev[k] not in sids:
                    raise ValueError(f"{g['id']}: sequence names unknown session {ev[k]}")
        groups.append({**g, "sessions": sess})
    if only and {g["id"] for g in groups} != only:
        raise ValueError(f"unknown group(s): {sorted(only - {g['id'] for g in groups})}")
    return {**spec, "groups": groups, "path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def resolve_wavs(spec: dict, render_dirs: dict[str, str], check: bool = True) -> None:
    for g in spec["groups"]:
        for s in g["sessions"]:
            if s["source"] not in render_dirs:
                raise SystemExit(f"--render-dir {s['source']}=<dir> missing (rendered with scripts/make_corpus.py)")
            p = Path(render_dirs[s["source"]]) / f"{s['item']}.wav"
            if check and not p.exists():
                raise SystemExit(f"missing WAV {p}")
            s["wav"] = str(p)


def new_out_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        raise SystemExit(f"refusing to overwrite existing results dir {p}")
    p.mkdir(parents=True)
    return p


def write_once(p: Path, obj) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ solo
def solo_cmd(s: dict, out_dir: Path, fixture_path: Path | None, a) -> list[str]:
    cmd = [sys.executable, str(HERE / "run_stream.py"), s["wav"], "--out", str(out_dir / s["sid"]),
           "--mode", a.mode, "--pace", a.pace, "--tail-silence", str(a.tail_silence), f"--tools={s['tools']}",
           "--seed", str(s["seed"]), "--tool-delay", str(s.get("tool_delay_s", 0.0))]
    if fixture_path is not None:
        cmd += ["--orders-json", str(fixture_path)]
    for x in a.run_stream_arg:
        cmd += x.split()
    return cmd


def run_solo(spec: dict, a) -> int:
    out = new_out_dir(a.out)
    fx = out / "_fixtures"
    fx.mkdir()
    man = {"mode": "solo", "spec": spec["path"], "spec_sha256": spec["sha256"], "settings": vars(a).copy(),
           "sessions": {}}
    rc_all = 0
    for g in spec["groups"]:
        for s in g["sessions"]:
            fp = None
            if s["orders_rows"] is not None:
                fp = fx / f"{s['sid']}.json"
                fp.write_text(json.dumps(s["orders_rows"]))
            cmd = solo_cmd(s, out, fp, a)
            print("SOLO", s["sid"], " ".join(cmd), flush=True)
            if a.dry_run:
                man["sessions"][s["sid"]] = {"group": g["id"], "cmd": cmd, "rc": None}
                continue
            t0 = time.monotonic()
            rc = subprocess.call(cmd)
            man["sessions"][s["sid"]] = {"group": g["id"], "cmd": cmd, "rc": rc,
                                         "wall_s": round(time.monotonic() - t0, 1)}
            rc_all = rc_all or rc
    write_once(out / "solo_manifest.json", man)
    return rc_all


# ------------------------------------------------------------------------------------------------ roster / interleave
class Roster:
    """Join/leave/rejoin semantics of a group sequence, shared by every driver.

    Events: {"tick": t, "join": sid} | {"tick": t, "leave": sid, "defer_while": [cond...]} | {"after_done": sid,
    "join": sid}. A deferred leave is retried every tick until ``can_leave(sid, conds)``; ``after_done`` fires once the
    named session is done. Sessions without any join event join at tick 0. A done session leaves the active set."""

    def __init__(self, sids: list[str], sequence: list[dict] | None = None):
        seq = list(sequence or [])
        joined = {e["join"] for e in seq if "join" in e}
        seq = [{"tick": 0, "join": s} for s in sids if s not in joined] + seq
        self.pending = seq
        self.active: list[str] = []
        self.done: set[str] = set()
        self.compositions: list[dict] = []
        self.events: list[dict] = []

    def update(self, tick: int, can_leave: Callable[[str, list], bool] = lambda s, c: True) -> list[str]:
        keep = []
        for e in self.pending:
            due = ("tick" in e and e["tick"] <= tick) or ("after_done" in e and e["after_done"] in self.done)
            if not due:
                keep.append(e)
            elif "leave" in e:
                if e["leave"] not in self.active:
                    continue
                if can_leave(e["leave"], e.get("defer_while", [])):
                    self.active.remove(e["leave"])
                    self.events.append({"tick": tick, "leave": e["leave"], "requested_tick": e.get("tick")})
                else:
                    keep.append(e)
            elif "join" in e:
                if e["join"] not in self.active and e["join"] not in self.done:
                    self.active.append(e["join"])
                    self.events.append({"tick": tick, "join": e["join"]})
        self.pending = keep
        return list(self.active)

    def mark_done(self, sid: str) -> None:
        self.done.add(sid)
        if sid in self.active:
            self.active.remove(sid)

    def record(self, tick: int, sids: list[str], rows: dict | None = None) -> None:
        if not self.compositions or self.compositions[-1]["sids"] != sorted(sids):
            self.compositions.append({"tick": tick, "sids": sorted(sids), "rows": rows})

    def finished(self, all_sids: Iterable[str]) -> bool:
        return set(all_sids) <= self.done


def needs_drain(sess, post_fc_s: float) -> bool:
    """Session.drain_fc's continue condition, one chunk at a time (for tick-interleaved drivers)."""
    dur = sess.engine.chunk_samples / 16000
    if sess.engine.fc_active() or sess._fc_buffer:
        return True
    if sess.last_fc_chunk < 0:
        return False
    return (sess.stats.chunks_to_engine - max(sess.last_fc_chunk, sess.last_text_chunk)) * dur < post_fc_s


class Lane:
    def __init__(self, sid: str, sess, chunks: list, post_fc_s: float, max_extra_s: float = 90.0):
        self.sid, self.sess, self.chunks = sid, sess, chunks
        self.k = 0
        self.extra = 0
        self.post_fc_s = post_fc_s
        self.cap = int(max_extra_s / (sess.engine.chunk_samples / 16000))
        self.done = False


def interleave(lanes: list[Lane], sequence: list[dict] | None, pace: str = "realtime", chunk_s: float = 0.16,
               clock=time.monotonic_ns, sleep=time.sleep, max_ticks: int = 100000) -> dict:
    """Tick loop: every tick each ACTIVE lane gets its next chunk (input, then tail silence, then drain silence while
    ``needs_drain``), in lane order; detached lanes consume nothing. Sequential B=1 engine calls — no batching."""
    import torch
    by = {ln.sid: ln for ln in lanes}
    ro = Roster([ln.sid for ln in lanes], sequence)
    chunk_log = []
    t0 = clock()

    def can_leave(sid, conds):
        s = by[sid].sess
        return not (("fc_active" in conds and s.engine.fc_active())
                    or ("agent_speaking" in conds and s._agent_speaking))

    tick = 0
    while not ro.finished(by) and tick < max_ticks:
        active = ro.update(tick, can_leave)
        if pace == "realtime":
            wait = (t0 + int(tick * chunk_s * 1e9) - clock()) / 1e9
            if wait > 0:
                sleep(wait)
        arrival = t0 + int(tick * chunk_s * 1e9)
        stepped = []
        for sid in active:
            ln = by[sid]
            if ln.k < len(ln.chunks):
                c = ln.chunks[ln.k]
            elif ln.extra < ln.cap and needs_drain(ln.sess, ln.post_fc_s):
                c = torch.zeros(ln.sess.engine.chunk_samples)
                ln.extra += 1
            else:
                ln.sess.flush(0.0)
                ln.done = True
                ro.mark_done(sid)
                continue
            ln.sess.feed_chunk(c)
            chunk_log.append({"sid": sid, "chunk_idx": ln.k, "arrival_ns": arrival, "done_ns": clock()})
            ln.k += 1
            stepped.append(sid)
        if stepped:
            ro.record(tick, stepped)
        tick += 1
    return {"ticks": tick, "wall_s": round((clock() - t0) / 1e9, 3), "compositions": ro.compositions,
            "roster_events": ro.events, "chunk_log": chunk_log}


# ------------------------------------------------------------------------------------------------ driver helpers
def collect_memory(backend=None, n_slots: int | None = None, weights_bytes: int | None = None,
                   load_peak_bytes: int | None = None, eartts_kv_bytes_per_slot: int | None = None,
                   perception_bytes_per_slot: int | None = None) -> dict:
    """I9 inputs for the driver result: backend.memory() (BackendMemory) + torch.cuda.memory_stats peaks."""
    m = {"n_slots": n_slots, "weights_bytes": weights_bytes, "load_peak_bytes": load_peak_bytes,
         "eartts_kv_bytes_per_slot": eartts_kv_bytes_per_slot,
         "perception_bytes_per_slot": perception_bytes_per_slot, "backend": None, "cuda": None}
    if backend is not None and hasattr(backend, "memory"):
        import dataclasses
        bm = backend.memory()
        m["backend"] = dataclasses.asdict(bm) if dataclasses.is_dataclass(bm) else dict(bm)
    try:
        import torch
        if torch.cuda.is_available():
            st = torch.cuda.memory_stats()
            m["cuda"] = {"max_allocated": torch.cuda.max_memory_allocated(),
                         "max_reserved": torch.cuda.max_memory_reserved(),
                         "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                         "stats": {k: st.get(k) for k in ("allocated_bytes.all.peak", "reserved_bytes.all.peak",
                                                          "inactive_split_bytes.all.peak", "num_alloc_retries",
                                                          "num_ooms", "active_bytes.all.peak")}}
    except ImportError:
        pass
    return m


def memory_breakdown(mem: dict | None) -> dict:
    """I9: weights / Thinker K/V+Mamba per slot / EarTTS KV per slot / graph pools / allocator reserve / load peak,
    each with its source; never collapsed into one number. Missing components are listed, unaccounted is derived."""
    mem = mem or {}
    be, cu = mem.get("backend") or {}, mem.get("cuda") or {}
    n = mem.get("n_slots")
    GB = 2 ** 30

    def gb(x):
        return None if x is None else round(x / GB, 3)

    def tot(per):
        return None if per is None or n is None else per * n

    rows = {
        "weights": (mem.get("weights_bytes"), "driver"),
        "thinker_kv_mamba_per_slot": (be.get("bytes_per_slot"), "backend.memory().bytes_per_slot"),
        "thinker_kv_mamba_total": (tot(be.get("bytes_per_slot")), "bytes_per_slot x n_slots"),
        "eartts_kv_per_slot": (mem.get("eartts_kv_bytes_per_slot"), "driver"),
        "eartts_kv_total": (tot(mem.get("eartts_kv_bytes_per_slot")), "per slot x n_slots"),
        "perception_per_slot": (mem.get("perception_bytes_per_slot"), "driver"),
        "graph_pools": (be.get("graph_pool_bytes"), "backend.memory().graph_pool_bytes"),
        "static_buffers": (be.get("static_buffers_bytes"), "backend.memory().static_buffers_bytes"),
        "peak_allocated": (cu.get("max_allocated"), "torch.cuda.max_memory_allocated"),
        "peak_reserved": (cu.get("max_reserved"), "torch.cuda.max_memory_reserved"),
        "allocator_reserve": ((cu["max_reserved"] - cu["max_allocated"])
                              if cu.get("max_reserved") is not None and cu.get("max_allocated") is not None else None,
                              "peak reserved - peak allocated"),
        "load_peak": (mem.get("load_peak_bytes"), "driver (peak during model load)"),
    }
    known = ("weights", "thinker_kv_mamba_total", "eartts_kv_total", "graph_pools", "static_buffers")
    out = {k: {"gb": gb(v), "bytes": v, "source": src} for k, (v, src) in rows.items()}
    missing = [k for k, (v, _) in rows.items() if v is None]
    if cu.get("max_allocated") is not None:
        parts = [rows[k][0] for k in known]
        out["unaccounted_in_peak"] = {"gb": gb(cu["max_allocated"] - sum(p or 0 for p in parts)),
                                      "source": "peak_allocated - sum(known components)",
                                      "partial": any(p is None for p in parts)}
    return {"components": out, "missing": missing, "n_slots": n, "backend_extra": be.get("extra")}


def session_summary(sess, ledger, wav_16k, agent, engine, wall_s: float, mode: str, pace: str, wav_path: str,
                    fc_records=None) -> dict:
    """run_stream.py's summary.json schema for a session driven by a pair driver (same keys the comparison reads)."""
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE))
    st = sess.stats
    steps = np.array(st.step_ms_model) if st.step_ms_model else np.zeros(1)
    chunk_ms = 1000 * engine.chunk_samples / 16000
    fc_tl = None
    if fc_records is not None:
        from run_stream import fc_timeline_from_records
        fc_tl = fc_timeline_from_records(fc_records)
    summ = {"wav": wav_path, "mode": mode, "pace": pace, "audio_in_s": round(wav_16k.numel() / 16000, 2),
            "wall_s": round(wall_s, 2), "audio_out_s": round(len(agent) / 22050, 2), "chunk_ms": chunk_ms,
            "step_ms_p50": round(float(np.percentile(steps, 50)), 1),
            "step_ms_p95": round(float(np.percentile(steps, 95)), 1),
            "rtf": round(float(steps.mean()) / chunk_ms, 3),
            "text": "".join(sess.text), "function_text": "".join(sess.func_text), "model_steps": sess.model_step,
            "turns": sess.turn_id, "stats": {k: v for k, v in vars(st).items() if k not in ("step_ms", "step_ms_model")},
            "tool_calls": [{"call_id": c.call_id, "name": c.name, "args": c.arguments, "status": c.status.value,
                            "result": c.result, "error": c.error, "model_step": c.model_step}
                           for c in ledger.calls.values()],
            "fc_timeline": fc_tl,
            "consumed": sess.controller.consumed if sess.controller else {},
            "grounding_violations": sess.controller.grounding_violations if sess.controller else 0,
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
            "token_ids": {"text": list(sess.text_ids), "func": list(sess.func_ids)}}
    return summ


# ------------------------------------------------------------------------------------------------ drivers
def probe_reference_multistream():
    """Return ReferenceEngine if it can host several independent streams, else raise PairUnsupported (cheap: no load)."""
    try:
        from mynah_v2.model.reference import ReferenceEngine
    except Exception as e:  # noqa: BLE001
        raise PairUnsupported(f"reference engine not importable here ({type(e).__name__}: {e})") from e
    if not hasattr(ReferenceEngine, "open_stream"):
        raise PairUnsupported(
            "ReferenceEngine is single-stream: one stream_id, pipeline.open_session() resets the shared pipeline "
            "state, one fc_records list and one tool-registry binding. reference-interleave needs "
            "ReferenceEngine.open_stream(stream_id) -> per-stream Engine handle (chunk_samples/open/step/fc_active/"
            "close/release_fc/bind_session_tools/token_tensors/fc_records). Not faked; use the batched driver.")
    return ReferenceEngine


def reference_interleave_driver(group: dict, out_dir: Path, ctx: dict) -> dict:
    """REFERENCE B=1 streams of ONE loaded model, stepped one after the other inside each 160 ms tick (no batched
    forward). Needs a multi-stream engine; the current ReferenceEngine is single-stream -> PairUnsupported."""
    ReferenceEngine = probe_reference_multistream()
    from mynah_v2.model.reference import build_cfg
    import numpy as np
    import soundfile as sf
    import torch
    from run_stream import load_16k, render_prompt
    from mynah_v2.agent.controller import Controller
    from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolLedger, ToolRegistry, make_orders_db, orders_tool
    from mynah_v2.runtime.session import Mode, Session, iter_chunks
    from mynah_v2.tracing.events import Tracer

    eng = ReferenceEngine(build_cfg(output_dir=str(out_dir / "_nemo")))
    load_peak, weights = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    lanes, aux = [], {}
    tail = ctx["tail_silence_s"]
    for i, s in enumerate(group["sessions"]):
        h = eng.open_stream(i)
        sd = out_dir / s["sid"]
        sd.mkdir(parents=True)
        tr = Tracer(sd / "trace.jsonl", session_id=s["sid"])
        led = ToolLedger(s["sid"], tr)
        reg = ToolRegistry()
        names = [n for n in s["tools"].split(",") if n]
        if "orders" in names:
            db = make_orders_db(sd / "orders.sqlite", rows=[tuple(r) for r in s["orders_rows"]])
            reg.register("get_order_status", orders_tool(db, s.get("tool_delay_s", 0.0)),
                         "Get the status, delivery date and tracking code of a customer order by its order id.",
                         ORDERS_SCHEMA, ack_message="Let me check that order for you.")
        if "random" in names:
            import offline_voicechat_fc_infer as fcmod
            reg.schemas += fcmod.DEFAULT_TOOLS
        holder = {}
        bound = dict(reg.bind(led, lambda h=holder: h["s"].turn_id, lambda h=holder: h["s"].model_step))
        if "random" in names:
            r2 = ToolRegistry()
            r2.register("generate_random_number", eng.pipeline._tool_generate_random_number, "", {})
            bound.update(r2.bind(led, lambda h=holder: h["s"].turn_id, lambda h=holder: h["s"].model_step))
        h.bind_session_tools(bound)
        prompt = render_prompt(reg.schemas)
        agent = []
        sess = Session(h, tr, Mode(ctx["mode"]), on_audio=lambda x, a=agent: a.append(x.numpy()),
                       controller=Controller(prompt, ledger=led, tools=bound))
        holder["s"] = sess
        torch.manual_seed(s["seed"])        # global generator: per-session RNG is NOT isolated here (I8 unknown)
        sess.open(prompt)
        wav = load_16k(s["wav"])
        chunks = list(iter_chunks(wav, h.chunk_samples)) + \
            [torch.zeros(h.chunk_samples)] * int(round(tail / (h.chunk_samples / 16000)))
        lanes.append(Lane(s["sid"], sess, chunks, post_fc_s=tail))
        aux[s["sid"]] = (h, led, tr, wav, agent, sd)
    torch.cuda.reset_peak_memory_stats()
    res = interleave(lanes, group.get("sequence"), pace=ctx["pace"], chunk_s=ctx["chunk_ms"] / 1000)
    for ln in lanes:
        h, led, tr, wav, agent, sd = aux[ln.sid]
        recs = list(h.fc_records)
        ln.sess.close("normal")
        tr.close()
        a = np.concatenate(agent) if agent else np.zeros(0, np.float32)
        sf.write(sd / "agent.wav", a, 22050)
        summ = session_summary(ln.sess, led, wav, a, h, res["wall_s"], ctx["mode"], ctx["pace"],
                               next(s["wav"] for s in group["sessions"] if s["sid"] == ln.sid), recs)
        (sd / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    return {"driver": "reference-interleave", "wall_s": res["wall_s"], "per_session_rng": False,
            "compositions": res["compositions"], "roster_events": res["roster_events"],
            "steps": [], "chunk_log": res["chunk_log"], "memory": collect_memory(n_slots=len(lanes), weights_bytes=weights,
                                                                 load_peak_bytes=load_peak)}


def resolve_driver(name: str) -> Callable:
    if name == "reference-interleave":
        probe_reference_multistream()
        return reference_interleave_driver
    entry = BATCHED_DRIVER_ENTRY if name == "batched" else name
    if ":" not in entry:
        raise PairUnsupported(f"driver must be 'batched', 'reference-interleave' or 'pkg.module:function', got {name}")
    mod, fn = entry.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise PairUnsupported(f"driver {entry} not available ({type(e).__name__}: {e})") from e


# ------------------------------------------------------------------------------------------------ comparison
def find_session_dir(root: Path, sid: str) -> Path | None:
    """<root>/<sid>, <root>/<group>/<sid> or run_stream's <root>/<sid>/<item>/ (dir holding summary.json)."""
    hits = sorted({p.parent for p in Path(root).rglob("summary.json") if sid in p.parent.relative_to(root).parts})
    if len(hits) > 1:
        raise ValueError(f"ambiguous session dirs for {sid}: {hits}")
    return hits[0] if hits else None


def load_trace(d: Path) -> list[dict]:
    p = d / "trace.jsonl"
    return [json.loads(line) for line in p.open()] if p.exists() else []


def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2) if xs else None


def step_metrics(trace: list[dict], chunk_ms: float) -> dict:
    st = [e for e in trace if e["ev"] == "thinker.step"]
    lat = [e.get("latency_ms", 0.0) for e in st]
    model = [e.get("latency_ms", 0.0) for e in st if not e.get("fc_active")]
    ts = [e["t_ns"] for e in st]
    return {"chunks": len(st), "step_ms_p50": pct(lat, 0.5), "step_ms_p95": pct(lat, 0.95),
            "model_step_ms_p50": pct(model, 0.5), "model_step_ms_p95": pct(model, 0.95),
            "steps_over_budget": sum(x > chunk_ms for x in lat),
            "span_s": round((max(ts) - min(ts)) / 1e9, 3) if len(ts) > 1 else None,
            "t_first_ns": min(ts) if ts else None, "t_last_ns": max(ts) if ts else None}


def deadline_misses(chunk_log: list[dict], chunk_ms: float) -> dict:
    out = {}
    for r in chunk_log:
        d = out.setdefault(r["sid"], {"chunks": 0, "misses": 0, "worst_ms": 0.0})
        lat = (r["done_ns"] - r["arrival_ns"]) / 1e6
        d["chunks"] += 1
        d["misses"] += lat > chunk_ms
        d["worst_ms"] = round(max(d["worst_ms"], lat), 1)
    return out


def fc_windows(trace: list[dict]) -> list[tuple[int, int]]:
    fs = [e["t_ns"] for e in trace if e["ev"] == "function.started"]
    fc = [e["t_ns"] for e in trace if e["ev"] == "tool.result.consumed"]
    return list(zip(fs, fc))


def asr_text(trace: list[dict]) -> str:
    return " ".join(e.get("text", "") for e in trace if e["ev"] == "asr.generated").lower()


def order_lookup(rows, oid) -> dict:
    oid = "".join(ch for ch in str(oid).upper() if ch.isalnum())
    for r in rows or []:
        if r[0] == oid:
            return {"order_id": r[0], "found": True, "status": r[1], "delivery_date": r[2], "tracking_code": r[3]}
    return {"order_id": oid, "found": False}


def _args(c) -> dict:
    a = c.get("args") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            a = {}
    return a


def eos_frames(s: dict) -> list[int]:
    ft = s.get("final_tokens") or s.get("token_ids") or {}
    return [i for i, t in enumerate(ft.get("text") or []) if t == EOS_ID]


def wav_compare(a: Path, b: Path) -> dict:
    if not (a.exists() and b.exists()):
        return {"available": False}
    ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
    r = {"available": True, "identical": ha == hb}
    if ha != hb:
        try:
            import numpy as np
            import soundfile as sf
            x, _ = sf.read(a, dtype="float32")
            y, _ = sf.read(b, dtype="float32")
            n = min(len(x), len(y))
            d = np.abs(x[:n] - y[:n])
            nz = np.nonzero(d > 0)[0]
            r.update(len=[len(x), len(y)], max_abs_diff=round(float(d.max()) if n else 0.0, 6),
                     first_diff_s=round(int(nz[0]) / 22050, 3) if len(nz) else None)
        except Exception as e:  # noqa: BLE001 - diagnostics only; byte identity above is the evidence
            r["decode_error"] = f"{type(e).__name__}: {e}"[:200]
    return r


def regex_hit(rx: str | None, text: str) -> bool | None:
    return None if not rx else bool(re.search(rx, text or "", re.I))


def compare_session(s: dict, group: dict, run_dir: Path, base_dir: Path, chunk_ms: float,
                    determinism_expected: bool | None) -> dict:
    sid = s["sid"]
    rd, bd = find_session_dir(run_dir, sid), find_session_dir(base_dir, sid)
    r = {"sid": sid, "item": s["item"], "run_dir": str(rd) if rd else None, "base_dir": str(bd) if bd else None}
    if rd is None or bd is None:
        r.update(status="MISSING", hard=["missing run" if rd is None else "missing solo baseline"], soft=[])
        return r
    run, base = json.loads((rd / "summary.json").read_text()), json.loads((bd / "summary.json").read_text())
    rt, bt = load_trace(rd), load_trace(bd)
    hard, soft, leaks = [], [], []
    others = [o for o in group["sessions"] if o["sid"] != sid]

    # tool calls: names + normalized args
    rc, bc = norm_calls(run), norm_calls(base)
    r["calls"], r["calls_missing"] = rc, [c for c in bc if c not in rc]
    r["calls_invented"] = [c for c in rc if c not in bc]
    if r["calls_missing"] or r["calls_invented"]:
        hard.append("tool calls differ from solo")
    # args / results looked up in EVERY fixture of the group
    own_id = (s.get("expect") or {}).get("order_id")
    other_ids = {(o.get("expect") or {}).get("order_id") for o in others} - {None, own_id}
    r["tool_results"] = []
    for c in run.get("tool_calls", []):
        a = _args(c)
        row = {"name": c.get("name"), "args": a}
        if c.get("name") == "get_order_status" and "order_id" in a:
            own = order_lookup(s["orders_rows"], a["order_id"])
            try:
                got = json.loads(c.get("result") or "null")
            except ValueError:
                got = c.get("result")
            from_other = [o["sid"] for o in others if o.get("orders_rows") is not None
                          and order_lookup(o["orders_rows"], a["order_id"]) == got and got != own]
            row.update(own_fixture_ok=(got == own), from_other_fixture=from_other or None)
            if from_other:
                leaks.append(f"tool result of call {c.get('call_id')} matches fixture of {from_other}")
            elif got != own and c.get("status") == "completed":
                hard.append(f"tool result of call {c.get('call_id')} matches no fixture of this session")
            norm = "".join(ch for ch in str(a["order_id"]).upper() if ch.isalnum())
            if norm in other_ids:
                leaks.append(f"call args order_id {norm} belongs to another session")
        r["tool_results"].append(row)
    done = [c["call_id"] for c in run.get("tool_calls", []) if c.get("status") == "completed"]
    r["results_unconsumed"] = [cid for cid in done if cid not in (run.get("consumed") or {})]
    if r["results_unconsumed"]:
        hard.append("completed tool result(s) not consumed by this session")
    # leaks in the spoken (internal) text and in the user ASR channel, flagged only if absent in solo
    for key, rx, tr_, tb_ in (("answer", s.get("leak_regex"), decoded(run), decoded(base)),
                              ("user_asr", s.get("user_leak_regex"), asr_text(rt), asr_text(bt))):
        hr, hb = regex_hit(rx, tr_), regex_hit(rx, tb_)
        r[f"{key}_leak_hit"], r[f"{key}_leak_hit_solo"] = hr, hb
        if hr and not hb:
            leaks.append(f"{key}: other session's facts /{rx}/ present (absent in solo)")
        elif hr and hb:
            soft.append(f"{key}: leak pattern also in solo (fixture/regex inconclusive)")
    ids = {e.get("session_id") for e in rt if e.get("session_id") is not None}
    r["trace_session_ids"] = sorted(ids)
    if len(ids) > 1:
        leaks.append(f"trace mixes session ids {sorted(ids)}")
    # functional score vs solo
    item = {"id": s["item"], "category": s["category"], "expect": s["expect"]}
    sr, sb = score_item(item, run, None), score_item(item, base, None)
    r["ok"], r["base_ok"] = bool(sr["ok_model"]), bool(sb["ok_model"])
    r["outcome"] = outcome(r["base_ok"], r["ok"])
    r["checks_changed"] = {k: [sb.get(k), sr.get(k)] for k in set(sr) | set(sb)
                           if k not in ("id", "category") and sb.get(k) != sr.get(k)}
    if r["outcome"] == "regression":
        hard.append("functional regression vs solo")
    # turn-taking / timeline
    ti, tb = trace_info(rd / "trace.jsonl"), trace_info(bd / "trace.jsonl")
    r.update(greeting_step=[tb["greeting_step"], ti["greeting_step"]],
             agent_turn_steps_identical=ti["agent_turn_steps"] == tb["agent_turn_steps"],
             n_agent_turns=[len(tb["agent_turn_steps"]), len(ti["agent_turn_steps"])],
             eos_frames_identical=eos_frames(run) == eos_frames(base),
             fc_window_ms=[tb["fc_window_ms"], ti["fc_window_ms"]])
    if ti["greeting_step"] != tb["greeting_step"]:
        soft.append("greeting step shifted")
    if not r["agent_turn_steps_identical"]:
        soft.append("agent turn steps differ")
    gs, rs = turn_stats(base), turn_stats(run)
    r["turns"] = {"solo": gs, "run": rs, "runon_vs_solo": runon_vs_golden(rs, gs)}
    if r["turns"]["runon_vs_solo"] == "new":
        hard.append("new run-on / self-talk vs solo")
    viol = [v for c in ((run.get("fc_timeline") or {}).get("calls") or []) for v in c.get("violations", [])]
    bviol = [v for c in ((base.get("fc_timeline") or {}).get("calls") or []) for v in c.get("violations", [])]
    r["fc_timeline_violations"] = [len(bviol), len(viol)]
    if len(viol) > len(bviol):
        hard.append("new FC-timeline violations")
    # identity diagnostics
    rti, bti = run.get("token_ids") or {}, base.get("token_ids") or {}
    r["tokens_identical"] = bool(rti) and rti == bti
    r["text_identical"] = run.get("text") == base.get("text")
    r["wer_vs_solo"] = wer(words(base.get("text")), words(run.get("text")))
    if not r["text_identical"]:
        soft.append(f"text differs (WER {r['wer_vs_solo']})")
    r["audio"] = wav_compare(rd / "agent.wav", bd / "agent.wav")
    if determinism_expected is None or not r["audio"].get("available"):
        r["I8"] = "NOT_GATED"
    else:
        r["I8"] = "PASS" if r["audio"]["identical"] or not determinism_expected else "FAIL"
        if r["I8"] == "FAIL":
            hard.append("I8: audio differs from solo although determinism is expected")
    if r["audio"].get("available") and not r["audio"]["identical"]:
        soft.append("agent.wav differs from solo")
    r["metrics"] = {"run": step_metrics(rt, chunk_ms), "solo": step_metrics(bt, chunk_ms)}
    r["leaks"], r["hard"], r["soft"] = leaks, hard, soft
    r["status"] = ("LEAK" if leaks else "REGRESSION" if hard else
                   "IDENTICAL" if r["tokens_identical"] and r["text_identical"] and r["calls"] == norm_calls(base)
                   else "EQUIVALENT")
    return r


def collapse_multi(compositions: list[dict]) -> list[list[str]]:
    out = []
    for c in compositions:
        s = sorted(c["sids"])
        if len(s) >= 2 and (not out or out[-1] != s):
            out.append(s)
    return out


def is_subsequence(want: list[list[str]], got: list[list[str]]) -> bool:
    it = iter(got)
    return all(any(sorted(w) == g for g in it) for w in want)


def row_reuse(compositions: list[dict]) -> list[dict]:
    last_owner, out = {}, []
    for c in compositions:
        for sid, row in (c.get("rows") or {}).items():
            prev = last_owner.get(row)
            if prev is not None and prev != sid:
                out.append({"tick": c.get("tick"), "row": row, "sid": sid, "previous_sid": prev})
            last_owner[row] = sid
    return out


def compare_group(g: dict, run_dir: Path, base_dir: Path, chunk_ms: float, driver_res: dict | None,
                  determinism: str) -> dict:
    dres = driver_res or {}
    if determinism == "yes":
        det = True
    elif determinism == "no":
        det = False
    else:
        det = dres.get("per_session_rng")           # auto: what the driver declares (None = unknown, not gated)
    sess = [compare_session(s, g, run_dir, base_dir, chunk_ms, det) for s in g["sessions"]]
    out = {"id": g["id"], "kind": g["kind"], "determinism_expected": det, "sessions": sess, "checks": {}}
    ck = out["checks"]
    traces = {s["sid"]: load_trace(Path(r["run_dir"])) for s, r in zip(g["sessions"], sess) if r.get("run_dir")}
    if g.get("require_fc_overlap") and driver_res is not None:
        w = [fc_windows(traces.get(s["sid"], [])) for s in g["sessions"]]
        ov = any(a0 < b1 and b0 < a1 for a0, a1 in (w[0] if w else []) for b0, b1 in (w[1] if len(w) > 1 else []))
        ck["fc_windows_overlapped"] = ov
    if g.get("expected_compositions") and driver_res is not None:
        got = collapse_multi(dres.get("compositions") or [])
        ck["compositions"] = got
        ck["compositions_ok"] = is_subsequence(g["expected_compositions"], got) if got else None
        ck["row_reuse"] = row_reuse(dres.get("compositions") or [])
    steps = dres.get("steps") or []
    by_n = {}
    for st in steps:
        by_n.setdefault(len(st.get("sids") or []), []).append(st.get("ms", 0.0))
    ck["forwards"] = {str(n): {"n": len(v), "ms_p50": pct(v, 0.5), "ms_p95": pct(v, 0.95)} for n, v in by_n.items()}
    ck["batched_forwards_ge2"] = sum(1 for st in steps if st.get("batched") and len(st.get("sids") or []) >= 2)
    if driver_res is not None and dres.get("driver") != "reference-interleave":
        ck["real_batched_forward"] = ck["batched_forwards_ge2"] > 0    # the gate needs a real B>=2 Thinker forward
    # throughput / deadlines / memory, vs the solo B=1 runs
    m = [r["metrics"] for r in sess if "metrics" in r]
    if m:
        n_run = sum(x["run"]["chunks"] for x in m)
        t0 = [x["run"]["t_first_ns"] for x in m if x["run"]["t_first_ns"]]
        t1 = [x["run"]["t_last_ns"] for x in m if x["run"]["t_last_ns"]]
        wall = dres.get("wall_s") or (round((max(t1) - min(t0)) / 1e9, 3) if t0 and t1 else None)
        n_solo = sum(x["solo"]["chunks"] for x in m)
        wall_solo = sum(x["solo"]["span_s"] or 0 for x in m)
        thr = round(n_run / wall, 2) if wall else None
        thr_solo = round(n_solo / wall_solo, 2) if wall_solo else None
        out["perf"] = {"session_chunks": n_run, "wall_s": wall, "session_chunks_per_s": thr,
                       "solo_b1_sequential_chunks_per_s": thr_solo,
                       "speedup_vs_solo_b1": round(thr / thr_solo, 3) if thr and thr_solo else None,
                       "input_rate_bound_chunks_per_s": round(len(m) * 1000 / chunk_ms, 2),
                       "per_session": {r["sid"]: {k: r["metrics"]["run"][k] for k in
                                                  ("step_ms_p50", "step_ms_p95", "model_step_ms_p50",
                                                   "model_step_ms_p95", "steps_over_budget")}
                                       | {"solo_step_ms_p50": r["metrics"]["solo"]["step_ms_p50"],
                                          "solo_step_ms_p95": r["metrics"]["solo"]["step_ms_p95"]}
                                       for r in sess if "metrics" in r},
                       "deadline_misses": (deadline_misses(dres["chunk_log"], chunk_ms) if dres.get("chunk_log")
                                           else "no chunk_log from driver; see steps_over_budget")}
    out["memory_I9"] = memory_breakdown(dres.get("memory")) if driver_res is not None else None
    if out["memory_I9"] is not None:
        out["memory_I9"]["solo_peak_allocated_gb"] = {r["sid"]: json.loads(
            (Path(r["base_dir"]) / "summary.json").read_text()).get("max_mem_gb") for r in sess if r.get("base_dir")}
    st = [r["status"] for r in sess]
    if "LEAK" in st:
        out["status"] = "FAIL_LEAK"
    elif "REGRESSION" in st or "MISSING" in st:
        out["status"] = "FAIL"
    elif ck.get("fc_windows_overlapped") is False or ck.get("compositions_ok") is False \
            or ck.get("real_batched_forward") is False \
            or ("compositions_ok" in ck and ck["compositions_ok"] is None):
        out["status"] = "INCONCLUSIVE"            # the group did not exercise what it is meant to test
    else:
        out["status"] = "PASS"
    return out


def compare_all(spec: dict, run_dir: Path, base_dir: Path, driver_results: dict, determinism: str,
                chunk_ms: float) -> dict:
    groups = [compare_group(g, run_dir, base_dir, chunk_ms, driver_results.get(g["id"]), determinism)
              for g in spec["groups"]]
    st = [g["status"] for g in groups]
    verdict = ("FAIL_LEAK" if "FAIL_LEAK" in st else "FAIL" if "FAIL" in st
               else "INCONCLUSIVE" if "INCONCLUSIVE" in st else "PASS")
    return {"spec": spec["path"], "spec_sha256": spec["sha256"], "run": str(run_dir), "baseline": str(base_dir),
            "verdict": verdict, "groups": groups,
            "notes": ["evidence = model internal text + tool ledger + traces; no ASR of the spoken audio here "
                      "(run scripts/agent/asr_dirs.py on the agent.wav dirs for the audio evidence)",
                      "realtime pacing caps throughput at input_rate_bound_chunks_per_s"]}


def print_report(rep: dict) -> None:
    for g in rep["groups"]:
        print(f"GROUP {g['id']:28s} {g['status']:13s} checks={json.dumps(g['checks'])[:300]}")
        for r in g["sessions"]:
            print(f"   {r['sid']:5s} {r['status']:10s} leaks={r.get('leaks')} hard={r.get('hard')} "
                  f"soft={r.get('soft')}"[:400])
        if g.get("perf"):
            p = g["perf"]
            print(f"   perf chunks/s={p['session_chunks_per_s']} solo_b1={p['solo_b1_sequential_chunks_per_s']} "
                  f"speedup={p['speedup_vs_solo_b1']}")
    print("VERDICT", rep["verdict"], flush=True)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bench_mode", choices=["solo", "pair", "compare"])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--groups", default="", help="comma list of group ids (default: all)")
    ap.add_argument("--out", required=True, help="NEW results dir (refused if it exists)")
    ap.add_argument("--render-dir", action="append", default=[],
                    help="source=dir of rendered WAVs, e.g. corpus_v1=~/mynah-s2s/data/v2-corpus-v1 (solo/pair)")
    ap.add_argument("--solo-dir", default=None, help="pair/compare: solo results dir (the B=1 baseline)")
    ap.add_argument("--run-dir", default=None, help="compare: run to compare against --solo-dir")
    ap.add_argument("--driver-results", default=None, help="compare: dir holding <group>/driver_result.json")
    ap.add_argument("--driver", default="batched", help="batched | reference-interleave | pkg.module:function")
    ap.add_argument("--engine-mode", dest="mode", default="reference", choices=["reference", "enhanced"],
                    help="Session mode passed to run_stream.py / the driver")
    ap.add_argument("--pace", default="realtime", choices=["realtime", "fast"])
    ap.add_argument("--tail-silence", type=float, default=6.0)
    ap.add_argument("--expect-deterministic", default="auto", choices=["auto", "yes", "no"],
                    help="I8 gate: auto = the driver's per_session_rng declaration (unknown -> not gated)")
    ap.add_argument("--run-stream-arg", action="append", default=[], help="solo: extra run_stream.py args")
    ap.add_argument("--dry-run", action="store_true", help="solo: write the manifest + print commands only")
    a = ap.parse_args(argv)
    spec = load_spec(Path(a.spec), [x for x in a.groups.split(",") if x])
    chunk_ms = float(spec.get("chunk_ms", 160))
    if a.pace == "fast" and any(s.get("tools") for g in spec["groups"] for s in g["sessions"]):
        ap.error("--pace fast is invalid with tool sessions (see run_stream.py); select no-tool groups")
    renders = dict(x.split("=", 1) for x in a.render_dir)

    if a.bench_mode == "solo":
        resolve_wavs(spec, renders, check=not a.dry_run)
        return run_solo(spec, a)

    if a.bench_mode == "compare":
        if not (a.solo_dir and a.run_dir):
            ap.error("compare needs --run-dir and --solo-dir")
        dres = {}
        if a.driver_results:
            for g in spec["groups"]:
                p = Path(a.driver_results) / g["id"] / "driver_result.json"
                if p.exists():
                    dres[g["id"]] = json.loads(p.read_text())
        out = new_out_dir(a.out)
        rep = compare_all(spec, Path(a.run_dir), Path(a.solo_dir), dres, a.expect_deterministic, chunk_ms)
        write_once(out / SUMMARY_NAME, rep)
        print_report(rep)
        return 0 if rep["verdict"] in ("PASS",) else 1

    # pair
    if not a.solo_dir:
        ap.error("pair needs --solo-dir (the solo B=1 baseline of the same groups)")
    try:
        driver = resolve_driver(a.driver)
    except PairUnsupported as e:
        print(f"UNSUPPORTED: {e}", flush=True)
        return EXIT_UNSUPPORTED
    resolve_wavs(spec, renders)
    out = new_out_dir(a.out)
    ctx = {"mode": a.mode, "pace": a.pace, "tail_silence_s": a.tail_silence, "chunk_ms": chunk_ms,
           "spec_path": spec["path"], "run_stream": str(HERE / "run_stream.py")}
    dres = {}
    for g in spec["groups"]:
        gd = out / g["id"]
        gd.mkdir()
        try:
            res = driver(g, gd, ctx)
        except PairUnsupported as e:
            print(f"UNSUPPORTED: {e}", flush=True)
            write_once(gd / "driver_result.json", {"unsupported": str(e)})
            return EXIT_UNSUPPORTED
        write_once(gd / "driver_result.json", res)
        dres[g["id"]] = res
    rep = compare_all(spec, out, Path(a.solo_dir), dres, a.expect_deterministic, chunk_ms)
    rep["driver"] = a.driver
    write_once(out / SUMMARY_NAME, rep)
    print_report(rep)
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
