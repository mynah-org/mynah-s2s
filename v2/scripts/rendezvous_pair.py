"""First integrated GPU gate of the true-batching path: two EN sessions, real batched Thinker forward (B=2), vs solo B=1.

One process, one model load. Sessions, audio, per-session orders fixtures and seeds come from eval/two_session_v1.json
(group id). Both runs use the same exact profile (mamba_host_pos + no_timer_sync), the same realtime pacing (one 160 ms
chunk per tick; FC threads run on wall time) and the raw engine step API (ReferenceEngine.step — the Session/ENHANCED
layer is not involved yet):
  solo : each session alone through the unchanged single-session path (upstream eager Thinker, B=1);
  pair : MultiSessionEngine + BatchedThinker (thinker_batch.py, eager or graphs) + RendezvousEngine: one round per tick,
         every attached decode frame of the two sessions in ONE backend.step.
Per session vs its solo run: text/function token sequences (first divergence), tool calls (name, args, result),
functional score (score_corpus.score_item with the group's expect), leak regex of the other fixture; timing: per-round
wall p50/p95 vs solo per-step, deadline misses (> 160 ms), aggregate throughput; memory: allocated/reserved peaks +
backend.memory(). Writes <out>/pair_gate.json (immutable run dir).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as S
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mynah_v2 import presets
from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolRegistry, make_orders_db, orders_tool
from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k, render_prompt  # noqa: E402
from score_corpus import score_item  # noqa: E402

TICK = 0.16          # pacing tick (s): one 160 ms audio chunk per session per tick; --tick-s overrides
AUDIO_TICK = 0.16    # audio per chunk (workload: tail/drain lengths in chunks never change)


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 1) if xs else None


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _audio_summary(st):
    if not st:
        return None
    n = max(1, st["samples"])
    return {"samples": st["samples"], "nonfinite": st["nonfinite"], "rms": round((st["sumsq"] / n) ** 0.5, 5),
            "max_abs": round(st["max_abs"], 4)}


def _fc_cycles(recs, t0_ns):
    """FC cycles from FCRecords (phase 1 = call tokens, phase 2 = response): durations and how late the answer
    ends vs the realtime arrival of the FC start chunk (start_frame / 2 chunks of TICK, approximate)."""
    out, p1 = [], None
    for r in recs or []:
        if r["phase"] == 1:
            p1 = r
        elif r["phase"] == 2 and p1 is not None:
            arr = None if t0_ns is None else t0_ns + int(p1["start_frame"] / 2 * TICK * 1e9)
            out.append({"start_frame": p1["start_frame"], "p1_steps": p1["steps"],
                        "p1_ms": round((p1["t1_ns"] - p1["t0_ns"]) / 1e6, 1),
                        "gap_ms": round((r["t0_ns"] - p1["t1_ns"]) / 1e6, 1), "p2_steps": r["steps"],
                        "p2_ms": round((r["t1_ns"] - r["t0_ns"]) / 1e6, 1),
                        "total_ms": round((r["t1_ns"] - p1["t0_ns"]) / 1e6, 1),
                        "start_vs_realtime_ms": None if arr is None else round((p1["t0_ns"] - arr) / 1e6, 1),
                        "end_vs_realtime_ms": None if arr is None else round((r["t1_ns"] - arr) / 1e6, 1)})
            p1 = None
    return out


def _rng_counters():
    """Per-session EarTTS noise draws vs draws that fell back to the global generator (no row buffer: e.g. FC-thread
    or first-step DynamicCache paths). None when the per-session RNG is off."""
    from mynah_v2.model import eartts_rng
    st = eartts_rng.STATE
    return {"draws": st["draws"], "fallback_global": st["fallback_global"]} if st["enabled"] else None


def _audio_cmp(a, b):
    """Solo vs pair waveform: identical length? max |diff|, correlation (with --eartts-rng session and identical
    tokens, differences come only from batched numerics)."""
    if not a or not b:
        return None
    x, y = torch.cat(a), torch.cat(b)
    n = min(len(x), len(y))
    x, y = x[:n].double(), y[:n].double()
    d = (x - y).abs()
    c = float(torch.corrcoef(torch.stack([x, y]))[0, 1]) if n > 1 and x.std() > 0 and y.std() > 0 else None
    return {"len_solo": len(torch.cat(a)), "len_pair": len(torch.cat(b)), "max_abs_diff": round(float(d.max()), 5),
            "mean_abs_diff": round(float(d.mean()), 6), "corr": None if c is None else round(c, 5),
            "identical": bool(d.max() == 0)}



def _handoff_summary(h):
    """V2_HANDOFF_TRACE=1: host time per baton hand-off, grouped by (resumed-from phase -> parked-at phase)."""
    if h is None:
        return None
    import statistics
    g = {}
    for sid, prev, nxt, hand_ms, run_ms in h:
        g.setdefault(f"{prev}->{nxt}", []).append((hand_ms, run_ms))
    out = {}
    for k, v in sorted(g.items()):
        run = [x[1] for x in v]
        hand = [x[0] for x in v]
        out[k] = {"n": len(v), "run_ms_p50": round(statistics.median(run), 3), "run_ms_sum": round(sum(run), 1),
                  "handoff_ms_p50": round(statistics.median(hand), 3), "handoff_ms_sum": round(sum(hand), 1)}
    return {"by_transition": out, "n": len(h), "raw": h}

def main():
    global TICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                    help="regression gate (default): fail if a requested optimisation cannot be enabled; --no-strict "
                         "continues without it (WARNING) and records the fallback")
    ap.add_argument("--preset", default="release", choices=list(presets.MULTI),
                    help="release (default) = the configuration of the published benchmarks; reference = no V2 "
                         "execution optimisations. Explicit flags below override the preset")
    ap.add_argument("--spec", default=str(Path(__file__).parents[1] / "eval/two_session_v1.json"))
    ap.add_argument("--audio-root", required=True, help="dir containing corpus_v1/ and capability_v1/ renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc-ctx-fix", default="auto", choices=["auto", "on", "off"],
                    help="FC hand-back fix (selective): auto = ON for the official VoiceChat checkpoint only; off = rollback")
    ap.add_argument("--graphs", type=int, default=None, help="BatchedThinker CUDA graphs (0 = eager)")
    ap.add_argument("--no-call-guard", action="store_true",
                    help="sessions without the per-session call guard (default ON, solo AND pair, as in serving)")
    ap.add_argument("--tail-s", type=float, default=6.0)
    ap.add_argument("--drain-s", type=float, default=40.0,
                    help="after the scripted audio, keep feeding silence while an FC cycle is open (+ tail after it "
                         "closes), up to this many seconds; same rule for solo and pair (FC is wall-clock driven)")
    ap.add_argument("--profile", default=None, help="perf profile for BOTH solo and pair (e.g. optimized-graph-s4); "
                    "default: the exact pair (mamba_host_pos + no_timer_sync)")
    ap.add_argument("--exp", default=None, help="comma list of perf experiments applied after the profile (solo AND pair), "
                    "e.g. fc_ctx_fix")
    ap.add_argument("--breakdown", action=argparse.BooleanOptionalAction, default=None,
                    help="per-component CUDA-event breakdown (perception / Thinker / EarTTS / codec, FC threads tagged) "
                         "for the solo phase and the multi-session phase (runtime/comp_timer.py)")
    ap.add_argument("--max-sessions", type=int, default=8, help="sessions taken from the group (all by default)")
    ap.add_argument("--lmax", type=int, default=4096, help="BatchedThinker static K/V window per row")
    ap.add_argument("--perc-batch", default=None, choices=["off", "on", "on-permel", "graph", "graph-permel"],
                    help="multi-session phase only: batch the cache-aware perception of the sessions ready in a round "
                         "(model/perception_batch.py); on = eager, graph = CUDA graphs per row count pre-captured "
                         "before the sessions open; -permel = mel preprocessor per session")
    ap.add_argument("--eartts-rng", default=None, choices=["global", "session"],
                    help="EarTTS sampling noise: global = upstream (default CUDA generator, shared by every session and "
                         "thread); session = per-session counter-based noise from the session seed and its EarTTS step "
                         "(model/eartts_rng.py), identical whatever the neighbours / row")
    ap.add_argument("--eartts-dtype", default="fp32", choices=["fp32", "bf16", "bf16w", "bf16kv"],
                    help="EarTTS Gemma3 backbone weights (no FP32 copy kept) + static KV slots in this dtype; casts at "
                         "the backbone boundary; MoG/lm_head/embeddings/CFG/subword encoder stay FP32. fp32 = unchanged")
    ap.add_argument("--tick-s", type=float, default=0.16,
                    help="pacing tick (chunk arrival period, s). 0.16 = realtime. A slower GPU can be given the same "
                         "relative slack as a faster one (e.g. GB10 0.48 s ~ Ada 0.16 s); workload unchanged")
    ap.add_argument("--fc-budget", type=int, default=-1,
                    help="multi-session phase: max FC-thread Thinker requests served inside one round (all pumps of "
                         "the round); between rounds only idle time serves them. -1 = unlimited (legacy, default)")
    ap.add_argument("--rnnt-overlap", default=None, choices=["on", "off"],
                    help="run each session's user-side RNN-T decode step on a side stream while the batched Thinker "
                         "runs (bit-identical outputs; preset default: on in release / release-exact)")
    ap.add_argument("--fc-merge", default=None, choices=["on", "off"],
                    help="multi-session phase: queued FC-thread Thinker decode requests of attached sessions join the "
                         "round's batched Thinker step as extra rows (default on; off = legacy B=1 step_one service, "
                         "lower FC cycle latency (~3 s on L40S) but FC bursts in the rounds)")
    ap.add_argument("--fc-force-after-s", type=float, default=1.0,
                    help="FC budget anti-deadlock escape: serve one more FC request when a session has not parked "
                         "for this long while FC requests are queued (counted as forced)")
    presets.add_determinism_args(ap)
    ap.add_argument("--allow-global-rng-batch", action="store_true",
                    help="allow --eartts-batch with --eartts-rng global (only to reproduce VAL1; VAL2 defect)")
    ap.add_argument("--eartts-batch", default=None, choices=["off", "on", "eager"],
                    help="multi-session phase only: ONE EarTTS step over the sessions parked at it (model/eartts_batch.py; "
                         "needs eartts_graph_full; pooled slots). on = graphs per bucket pre-captured before the "
                         "sessions open; eager = no graph. Not exact (batched GEMMs, shared RNG draw)")
    ap.add_argument("--codec-batch", default=None, choices=["off", "on"],
                    help="multi-session phase only: batch the main-path codec decode of the sessions of a round "
                         "(model/codec_batch.py; audio-only, ~1e-6 vs per-session)")
    ap.add_argument("--capture-policy", default=None, choices=["lazy", "barrier", "precapture"],
                    help="CUDA-graph capture policy for the MULTI-session phase (model/capture_guard.py): lazy = legacy; "
                         "barrier = capture only when the capturing thread is the only GPU-issuing one, else eager + "
                         "retry, failed capture recovered; precapture = barrier + every BatchedThinker (n_rows, bucket) "
                         "key and every EarTTS (slot, bucket, mask) key for N sessions captured before the sessions open")
    ap.add_argument("--precapture-min-bucket", type=int, default=0, help="pre-capture only buckets >= this")
    ap.add_argument("--precapture-max-bucket", type=int, default=0, help="pre-capture only buckets <= this (0 = lmax)")
    ap.add_argument("--precapture-min-free-gib", type=float, default=6.0,
                    help="stop pre-capturing (fail soft: remaining keys lazy + barrier-guarded) when the device would "
                         "keep less than this free (sessions still to open + run-time peak)")
    a = ap.parse_args()
    TICK = a.tick_s
    prof_win = None
    if os.environ.get("V2_TORCH_PROFILE"):            # "k0-k1": torch.profiler over pair rounds k0..k1 (diagnostics)
        prof_win = tuple(int(x) for x in os.environ["V2_TORCH_PROFILE"].split("-"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(Path(a.spec).read_text())
    group = next(g for g in spec["groups"] if g["id"] == a.group)
    sess = group["sessions"][: a.max_sessions]       # N-session gate (B = N rows in the batched backend)
    srcs = {k: json.loads((Path(a.spec).parent / v).read_text()) for k, v in spec["sources"].items()}
    preset_info = presets.fill(a, presets.MULTI, a.preset)
    a.profile = presets.multi_profile_for(len(sess), a.profile)
    print("PRESET", a.preset, preset_info, "profile", a.profile, flush=True)

    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides=presets.determinism_overrides(a)),
                          fc_ctx_fix=a.fc_ctx_fix)
    det_info = presets.apply_determinism(a)
    prof, prof_info = None, None
    if prof_win is not None:                             # label before the profile: the fused norms keep the labels
        from mynah_v2.runtime import comp_timer as _ct
        from mynah_v2.runtime import op_profiler
        _ct.PROFILE_LABELS = True
        prof_info = {"window": prof_win, "eartts_labels": op_profiler.label_eartts(eng.wrapper.model.tts_model)}
        print("TORCH_PROFILE " + json.dumps(prof_info), flush=True)
    opt = presets.apply(eng, a.profile or None, a.exp if a.profile else ",".join(["mamba_host_pos", "no_timer_sync", a.exp]),
                        strict=a.strict)
    exp_info = {n: v for n, v in opt["active"].items() if n in a.exp.split(",")}
    eartts_dtype_info = None
    eartts_rng_info = None
    if a.eartts_batch != "off" and a.eartts_rng == "global" and not a.allow_global_rng_batch:
        raise SystemExit("--eartts-batch needs --eartts-rng session (VAL2: global noise depends on the batch "
                         "composition); --allow-global-rng-batch only to reproduce VAL1")
    if a.eartts_batch != "off" and a.eartts_rng == "global":
        # VAL2: with one noise draw for all rows, a session's EarTTS samples depend on who else is in the batch
        # (pair != solo from the first batched frame; one reproducible degraded greeting in M4MIX). Use session.
        print("WARNING --eartts-batch with --eartts-rng global: EarTTS noise depends on the batch composition; "
              "use --eartts-rng session", flush=True)
    if a.eartts_rng == "session":                   # before any EarTTS graph exists (solo phase included)
        from mynah_v2.model import eartts_rng
        eartts_rng_info = eartts_rng.enable(eng)
    if a.eartts_dtype != "fp32":                    # solo AND multi-session phases (FC threads / B=1 path too)
        fsg0 = getattr(eng, "_v2_full_graph", None)
        if fsg0 is None:
            raise SystemExit("--eartts-dtype bf16 needs the eartts_graph_full experiment (profile optimized-graph-*)")
        eartts_dtype_info = fsg0.set_mode(a.eartts_dtype)
        print("EARTTS_DTYPE " + json.dumps(eartts_dtype_info), flush=True)
    gb = lambda x: round(x / 2**30, 2)
    mem_phases = {"after_load": {"alloc_GB": gb(torch.cuda.memory_allocated()), "reserved_GB": gb(torch.cuda.memory_reserved()),
                                 "peak_alloc_GB_load": gb(torch.cuda.max_memory_allocated())}}
    torch.cuda.reset_peak_memory_stats()
    import offline_voicechat_fc_infer as fcmod

    def item_of(s):
        return next(i for i in srcs[s["source"]]["items"] if i["id"] == s["item"])

    def tools_of(s, log):
        reg, tools = ToolRegistry(), {}
        names = [x for x in (s.get("tools") or "").split(",") if x]
        if "orders" in names:
            rows = spec["fixtures"][s["fixture"]] if s.get("fixture") else None
            db = make_orders_db(out / f"orders_{s['sid']}.sqlite", **({"rows": [tuple(r) for r in rows]} if rows else {}))
            fn = orders_tool(db, s.get("tool_delay_s", 0.0))
            reg.register("get_order_status", fn, "Get the status, delivery date and tracking code of a customer order "
                         "by its order id.", ORDERS_SCHEMA, ack_message="Let me check that order for you.")
        if "random" in names:
            reg.schemas += fcmod.DEFAULT_TOOLS
            reg.handlers["generate_random_number"] = eng.pipeline._tool_generate_random_number
        for name, h in reg.handlers.items():
            def wrapped(args, _h=h, _n=name):
                r = _h(args)
                log.append({"name": _n, "args": args, "result": r})
                return r
            tools[name] = wrapped
        return tools, render_prompt(reg.schemas)

    wavs = {s["sid"]: torch.cat([load_16k(Path(a.audio_root) / s["source"] / f"{s['item']}.wav"),
                                 torch.zeros(int(a.tail_s * 16000))]) for s in sess}
    chunks = {sid: list(iter_chunks(w, eng.chunk_samples)) for sid, w in wavs.items()}
    silence = torch.zeros(eng.chunk_samples)
    n_tail, n_drain = int(a.tail_s / AUDIO_TICK), int(a.drain_s / AUDIO_TICK)

    class Feed:
        """Scripted chunks, then silence while the session's FC cycle is open (+ tail once it closes), capped."""
        def __init__(self, sid):
            self.base, self.k, self.fc, self.post, self.extra = chunks[sid], 0, False, None, 0

        def next(self):
            k, self.k = self.k, self.k + 1
            if k < len(self.base):
                return self.base[k]
            if self.extra >= n_drain:
                return None
            if self.fc:
                self.post = None                      # still open: tail restarts after it closes
            elif self.post is None:
                if self.extra == 0:
                    return None                       # no FC open when the script ended: nothing to drain
                self.post = n_tail
            if self.post is not None:
                if self.post <= 0:
                    return None
                self.post -= 1
            self.extra += 1
            return silence

        def report(self):
            return {"drain_chunks": self.extra, "fc_open_at_end": self.fc, "drain_capped": self.extra >= n_drain}

    def collect(acc, o):
        acc["text_ids"] += list(o.new_text_ids)
        acc["func_ids"] += list(o.new_func_ids)
        acc["text"] += o.text_delta
        acc["fc_records"] += len(o.fc_records)
        acc.setdefault("fc_rec", []).extend({"phase": r.phase, "start_frame": r.start_frame, "steps": r.steps,
                                             "t0_ns": r.t_start_ns, "t1_ns": r.t_end_ns} for r in o.fc_records)
        au = getattr(o, "audio", None)                 # audio sanity (finite samples, energy) for precision probes
        if isinstance(au, torch.Tensor) and au.numel():
            a32 = au.detach().float().cpu()
            st = acc.setdefault("audio_stats", {"samples": 0, "nonfinite": 0, "sumsq": 0.0, "max_abs": 0.0})
            fin = torch.isfinite(a32)
            st["samples"] += a32.numel()
            st["nonfinite"] += int((~fin).sum())
            st["sumsq"] += float((a32[fin] ** 2).sum())
            st["max_abs"] = max(st["max_abs"], float(a32[fin].abs().max()) if fin.any() else 0.0)
            acc.setdefault("audio_chunks", []).append(a32)

    # ------------------------------------------------------------------ solo (unchanged single-session path)
    from mynah_v2.runtime.comp_timer import CompTimer
    breakdown = {}
    ct = CompTimer().install(eng) if a.breakdown else None
    diag = None
    if os.environ.get("V2_DIAG", "0") == "1":      # VAL2 diagnostics: per-session codes, syncs, per-call timeline
        from mynah_v2.runtime.diag_recorder import DiagRecorder
        diag = DiagRecorder()
        diag.install_sync()
        diag.install_codes(eng.wrapper.model.tts_model.audio_codec)
    solo = {}
    for s in sess:
        sid, log = s["sid"], []
        tools, prompt = tools_of(s, log)
        acc = {"text_ids": [], "func_ids": [], "text": "", "fc_records": 0, "step_ms": [], "tools": log}
        seed_all(s["seed"])
        if diag is not None:
            diag.solo_sid = sid
        eng.bind_session_tools(tools)
        if tools and not a.no_call_guard:              # same default as the pair (rz.open) and as serving
            from mynah_v2.agent.call_guard import CallGuard
            eng.set_session_guard(CallGuard(session_id=sid))
        eng.open(prompt)
        t0 = time.perf_counter()
        acc["t0_ns"] = time.monotonic_ns()
        acc["e2e"] = []
        feed, k = Feed(sid), -1
        while (c := feed.next()) is not None:
            k += 1
            ts = time.perf_counter()
            o = eng.step(c)
            te = time.perf_counter()
            feed.fc = bool(o.fc_active)
            collect(acc, o)
            acc["step_ms"].append((te - ts) * 1e3)
            acc["e2e"].append({"k": k, "e2e_ms": (te - (t0 + k * TICK)) * 1e3, "fc": bool(o.fc_active)})
            time.sleep(max(0.0, t0 + (k + 1) * TICK - time.perf_counter()))
        acc["feed"] = feed.report()
        eng.close()
        solo[sid] = acc
    if diag is not None:
        diag.uninstall_codes()
        diag.solo_sid = None
    if ct is not None:
        breakdown["solo"] = ct.report(rounds=sum(len(v["step_ms"]) for v in solo.values()))   # per session-chunk
        ct.uninstall()
    torch.cuda.synchronize()
    mem_solo = {"max_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "max_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    torch.cuda.empty_cache()                       # the solo phase's allocator reserve must not count against the pair
    mem_phases["after_solo_empty_cache"] = {"alloc_GB": gb(torch.cuda.memory_allocated()),
                                            "reserved_GB": gb(torch.cuda.memory_reserved())}
    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------ pair (rendezvous + batched Thinker)
    from mynah_v2.model.session_state import MultiSessionEngine
    from mynah_v2.model.thinker_batch import BatchedThinker
    from mynah_v2.runtime.rendezvous import RendezvousEngine
    def snap(tag):                                 # N-sessions memory ladder: same config, same process
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        mem_phases[tag] = {"alloc_GB": gb(torch.cuda.memory_allocated()), "reserved_GB": gb(torch.cuda.memory_reserved()),
                           "device_free_GB": gb(free), "peak_alloc_GB": gb(torch.cuda.max_memory_allocated())}
    snap("N0_before_backend")
    m = MultiSessionEngine(eng)
    from mynah_v2.model.capture_guard import BUDGET, GUARD
    GUARD.configure("lazy" if a.capture_policy == "lazy" else "barrier")   # solo phase above: legacy lazy, unchanged
    BUDGET.min_bucket, BUDGET.max_bucket = a.precapture_min_bucket, (a.precapture_max_bucket or None)
    BUDGET.min_free_gib = a.precapture_min_free_gib
    GUARD.reset_stats()
    bt = BatchedThinker(eng, max_rows=len(sess), lmax=a.lmax, graphs=bool(a.graphs),
                        precapture=a.capture_policy == "precapture" and bool(a.graphs))
    ct = CompTimer().install(eng, backend=bt) if a.breakdown else None   # under the rendezvous interceptor
    rz = RendezvousEngine(m, bt, batch_perception=a.perc_batch != "off",
                          perception_batch_mel=not a.perc_batch.endswith("permel"),
                          perception_graphs=a.perc_batch.startswith("graph"),
                          batch_codec=a.codec_batch == "on",
                          batch_eartts=a.eartts_batch != "off", eartts_graphs=a.eartts_batch == "on",
                          eartts_precapture=a.eartts_batch == "on",
                          fc_budget=None if a.fc_budget < 0 else a.fc_budget, fc_force_after_s=a.fc_force_after_s,
                          fc_merge=a.fc_merge == "on", rnnt_overlap=a.rnnt_overlap == "on")
    if ct is not None and rz.ebatch is not None:
        ct._wrap(rz.ebatch, "run", "eartts_batched")          # n >= 2; lone / first steps stay "eartts"
    rz_eartts_capture = getattr(rz, "eartts_capture", None)
    if diag is not None:                            # outermost: sees each session's own request before parking
        diag.phase = "pair"
        diag.install_codes(eng.wrapper.model.tts_model.audio_codec)
    if rz.ebatch is not None:
        print("EARTTS_BATCH precapture " + json.dumps(rz_eartts_capture, default=str), flush=True)
    if ct is not None and rz.pbatch is not None:
        ct._wrap(rz.pbatch, "run", "perception_batched")       # n >= 2 rounds; lone sessions stay "perception"
    snap("N0_backend_static")
    fsg = getattr(eng, "_v2_full_graph", None)
    eartts_caps0 = dict(fsg.stats) if fsg is not None else {}
    precap = {}
    if a.capture_policy == "precapture" and fsg is not None:
        precap["eartts"] = fsg.precapture(n_slots=len(sess))      # quiet point: no session open yet
        snap("N0_after_eartts_precapture")
        print("PRECAPTURE eartts " + json.dumps(precap["eartts"], default=str), flush=True)
    pair, logs = {}, {}
    for i, s in enumerate(sess):
        logs[s["sid"]] = []
        tools, prompt = tools_of(s, logs[s["sid"]])
        rz.open(s["sid"], prompt, tools=tools, seed=s["seed"], guard=not a.no_call_guard)
        snap(f"N{i + 1}_after_open")               # fixed cost of one more session (prefill done, before any chunk)
        pair[s["sid"]] = {"text_ids": [], "func_ids": [], "text": "", "fc_records": 0, "tools": logs[s["sid"]]}
    rounds, misses, mem_samples = [], 0, []
    e2e = {sid: [] for sid in chunks}
    diag_rounds = []
    if diag is not None and ct is not None:
        ct.start_timeline()
    t0 = time.perf_counter()
    t0_ns = time.monotonic_ns()
    fc_round_trace = []
    # opt-in host profile of the multi-session rounds: V2_CPROFILE=FIRST-LAST (round indices). One cProfile for the
    # process (Python >= 3.12 profiles every thread through sys.monitoring), saved as <out>/cprofile.prof
    cprof_win = tuple(int(x) for x in os.environ["V2_CPROFILE"].split("-")) if os.environ.get("V2_CPROFILE") else None
    cprof = None
    feeds, k = {sid: Feed(sid) for sid in chunks}, -1
    while True:
        k += 1
        jobs = {sid: c for sid, f in feeds.items() if (c := f.next()) is not None}
        if not jobs:
            break
        arrival = t0 + k * TICK
        ts = time.perf_counter()
        nlog = len(rz.job_log)
        if prof_win is not None and k == prof_win[0]:
            torch.cuda.synchronize()
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        if cprof_win is not None and k == cprof_win[0]:
            import cProfile
            cprof = cProfile.Profile()
            cprof.enable()
        res = rz.step_jobs(jobs)
        if cprof is not None and k == cprof_win[1]:
            cprof.disable()
            cprof.dump_stats(str(out / "cprofile.prof"))
            cprof = None
        if prof is not None and k == prof_win[1]:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            prof_info["summary"] = {x: y for x, y in op_profiler.summarize(prof, out, prof_win[1] - prof_win[0] + 1).items()
                                    if x in ("kernel_ms_per_round", "by_label", "by_kind_dtype")}
            print("TORCH_PROFILE_DONE " + json.dumps(prof_info["summary"]["by_label"]), flush=True)
            prof = None
        dt = (time.perf_counter() - ts) * 1e3
        rounds.append(dt)
        fc_round_trace.append(dict(rz.fc_rounds[-1]) if rz.fc_rounds else None)
        if diag is not None:
            diag_rounds.append({"k": k, "ts_ms": round((ts - t0) * 1e3, 3), "dt_ms": round(dt, 3), "sids": sorted(jobs)})
        misses += dt > TICK * 1e3
        for jl in rz.job_log[nlog:]:
            o = res.get(jl["sid"])
            fc = bool(getattr(o, "fc_active", False))
            e2e[jl["sid"]].append({"k": k, "e2e_ms": (jl["t_end"] - arrival) * 1e3,
                                   "start_delay_ms": (jl["t_start"] - arrival) * 1e3,
                                   "b2": jl.get("b2", 0), "b1": jl.get("b1", 0), "fc_frames": jl.get("fc", 0),
                                   "wait_ms": jl["wait_ms"], "thinker_exec_ms": jl["exec_ms"], "fc": fc,
                                   "state": jl.get("state")})
        if k % 10 == 0:
            free, total = torch.cuda.mem_get_info()
            mem_samples.append({"k": k, "alloc_GB": gb(torch.cuda.memory_allocated()),
                                "reserved_GB": gb(torch.cuda.memory_reserved()), "free_GB": gb(free),
                                "peak_alloc_GB": gb(torch.cuda.max_memory_allocated()),
                                "positions": dict(bt.positions()), "evicted": list(bt.evicted)})
        for sid, o in res.items():
            if isinstance(o, BaseException):
                pair[sid].setdefault("errors", []).append(f"chunk {k}: {type(o).__name__}: {str(o)[:300]}")
            else:
                feeds[sid].fc = bool(o.fc_active)
                collect(pair[sid], o)
        if any(isinstance(o, BaseException) for o in res.values()):
            import traceback
            for sid, o in res.items():
                if isinstance(o, BaseException):
                    print(f"SESSION_ERROR {sid} chunk {k}: {type(o).__name__}: {o}", flush=True)
                    traceback.print_exception(type(o), o, o.__traceback__, limit=12)
            print(f"stopped at chunk {k}: a session failed", flush=True)
            break
        rz.pump(until=time.monotonic() + max(0.0, t0 + (k + 1) * TICK - time.perf_counter()))
    torch.cuda.synchronize()
    mem_pair = {"max_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "max_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    try:
        bm = bt.memory()
        mem_pair["backend"] = {k: v for k, v in vars(bm).items()} if bm is not None else None
    except Exception as e:  # noqa: BLE001
        mem_pair["backend"] = f"unavailable: {e}"
    diag_tl = None
    if diag is not None:
        diag.uninstall_codes()                      # before close_all restores the rendezvous codec interceptor
        diag_tl = ct.timeline_records(t_ref=t0) if ct is not None else None
    if ct is not None:
        breakdown["multi"] = ct.report(rounds=len(rounds))                                   # per round (N sessions)
        breakdown["multi_rounds"] = len(rounds)
    stats = dict(rz.stats)
    if rz.pbatch is not None:
        stats["perception_batch"] = dict(rz.pbatch.stats)
    if "codec" in stats:
        stats["codec"] = {**stats["codec"], "batch_hist": {str(k): v for k, v in stats["codec"]["batch_hist"].items()}}
    if "fc_by_sid" in stats:                      # summarize the FC queue/exec lists (p50/p95/sum)
        stats["fc_by_sid"] = {sid: {"n": v["n"], "queue_ms_p50": pct(v["queue_ms"], 0.5), "queue_ms_max": max(v["queue_ms"], default=None),
                                    "queue_ms_p95": pct(v["queue_ms"], 0.95), "queue_ms_sum": round(sum(v["queue_ms"]), 1),
                                    "exec_ms_p50": pct(v["exec_ms"], 0.5), "exec_ms_sum": round(sum(v["exec_ms"]), 1)}
                              for sid, v in stats["fc_by_sid"].items()}
    stats["positions_at_end"] = dict(bt.positions())
    rz_ebatch_stats = rz.ebatch.stats if rz.ebatch is not None else None
    bt_caps = {k: bt.bstats.get(k) for k in ("captures", "lazy_captures", "precaptured", "replays", "eager_steps",
                                             "capture_deferred", "capture_failed_recovered", "capture_ms",
                                             "precapture_sig", "precapture_alloc_delta", "precapture_pool_bytes",
                                             "precapture_keys",
                                             "first_capture_error", "steps_by_bucket", "steps_by_n")}
    try:
        rz.close_all()
        m.close_all()
    except BaseException as e:  # noqa: BLE001 - teardown problems are reported, results still written
        import traceback
        traceback.print_exc()
        stats["teardown_error"] = f"{type(e).__name__}: {e}"
    snap("after_teardown")                          # sessions closed, backend still alive
    try:
        bt.close()
    except Exception as e:  # noqa: BLE001
        stats["backend_close_error"] = f"{type(e).__name__}: {e}"
    if ct is not None:
        ct.uninstall()
    handoff_info = _handoff_summary(rz.handoffs)
    rnnt_overlap_info = dict(rz.rnnt_stats) if rz.rnnt_overlap else None
    del rz, m, bt
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    snap("after_backend_close_empty_cache")         # should return to N0_before_backend (else something stays alive)

    # ------------------------------------------------------------------ compare
    res = {"tick_s": TICK, "group": a.group, "graphs": bool(a.graphs), "profile": a.profile or "exact-pair", "exp": exp_info, "preset": a.preset, "optimisations": opt,
           "fc_ctx_fix_info": {k: v for k, v in eng.fc_ctx_fix_info.items() if k != "fix"}, "breakdown": breakdown, "gpu": torch.cuda.get_device_name(), "sessions": {},
           "rendezvous_stats": {k: (v if not isinstance(v, dict) else {str(x): y for x, y in v.items()})
                                for k, v in stats.items()},
           "memory": {"phases": mem_phases, "solo": mem_solo, "pair": mem_pair, "pair_samples": mem_samples},
           "lmax": a.lmax, "perc_batch": a.perc_batch, "codec_batch": a.codec_batch,
           "eartts_batch": a.eartts_batch, "eartts_dtype": a.eartts_dtype, "eartts_dtype_info": eartts_dtype_info,
           "determinism": det_info,
           "handoffs": handoff_info, "rnnt_overlap": rnnt_overlap_info,
           "eartts_rng": a.eartts_rng, "eartts_rng_info": eartts_rng_info,
           "eartts_rng_counters": _rng_counters(),
           "eartts_rng_composition_dependent": a.eartts_batch != "off" and a.eartts_rng == "global",
           "eartts_batch_stats": (dict(rz_ebatch_stats) if rz_ebatch_stats is not None else None),
           "eartts_batch_capture": rz_eartts_capture}

    if getattr(eng.wrapper, "_v2_fc_profile", None) is not None:
        from mynah_v2.model import fc_profile
        res["fc_profile"] = fc_profile.report(eng)
    fsg = getattr(eng, "_v2_full_graph", None)       # EarTTS full graph: slots really allocated / busy (VRAM per session)
    if fsg is not None:
        res["eartts_slots"] = {"allocated": sum(x is not None for x in fsg.slots), "pool": fsg.n_slots,
                               "busy": sum(1 for x in fsg.slots if x is not None and x.busy),
                               "eager_reasons": dict(fsg.stats.get("eager_reasons", {}))}
    res["capture_policy"] = {
        "policy": a.capture_policy, "guard": GUARD.report(), "thinker_batch": bt_caps, "precapture": precap,
        # EarTTS counters over the multi-session phase only (solo-phase values subtracted)
        "eartts": {k: (fsg.stats.get(k, 0) - eartts_caps0.get(k, 0)) for k in
                   ("captures", "lazy_captures", "replays", "eager", "capture_deferred", "capture_failed_recovered",
                    "precaptured")} | {"first_capture_error": fsg.stats.get("first_capture_error")}
        if fsg is not None else None}
    print("CAPTURE_POLICY " + json.dumps(res["capture_policy"], default=str)[:3000], flush=True)
    fx = getattr(eng.wrapper, "_v2_fc_ctx_fix", None)
    if fx is not None:
        res["fc_ctx_fix_stats"] = dict(fx["stats"])

    def lat(rows, key="e2e_ms"):
        xs = [r[key] for r in rows]
        return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": round(max(xs), 1) if xs else None,
                "misses_gt_160ms": sum(x > TICK * 1e3 for x in xs)}
    res["latency_e2e"] = {}
    for sid in chunks:
        rows = e2e[sid]
        res["latency_e2e"][sid] = {
            "pair_all": lat(rows),
            "pair_b2_chunks": lat([r for r in rows if r["b2"] > 0 and not r["fc"]]),
            "pair_b1_chunks": lat([r for r in rows if r["b2"] == 0 and r["b1"] > 0 and not r["fc"]]),
            "pair_tool_pending_chunks": lat([r for r in rows if r["fc"]]),
            "pair_no_thinker_chunks": lat([r for r in rows if r["b2"] == 0 and r["b1"] == 0 and not r["fc"]]),
            "pair_rendezvous_wait_minus_exec_ms": lat([{"x": r["wait_ms"] - r["thinker_exec_ms"]} for r in rows], "x"),
            "pair_start_delay_ms": lat(rows, "start_delay_ms"),
            "solo_all": lat(solo[sid]["e2e"]),
            "solo_tool_pending_chunks": lat([r for r in solo[sid]["e2e"] if r["fc"]])}
    # interference: each session's chunks while ANOTHER session's FC cycle is open vs not (same round index k)
    fc_k = {sid: {r["k"] for r in rows if r["fc"]} for sid, rows in e2e.items()}
    for sid, rows in e2e.items():
        other = set().union(*[v for o, v in fc_k.items() if o != sid]) if len(fc_k) > 1 else set()
        own = [r for r in rows if not r["fc"]]
        res["latency_e2e"].setdefault(sid, {})
        res["latency_e2e"][sid]["pair_while_other_fc"] = lat([r for r in own if r["k"] in other])
        res["latency_e2e"][sid]["pair_while_other_not_fc"] = lat([r for r in own if r["k"] not in other])
    for s in sess:
        sid, so, pa = s["sid"], solo[s["sid"]], pair[s["sid"]]
        it = dict(item_of(s))
        it["expect"] = s.get("expect", it.get("expect", {}))
        sumz = lambda acc: {"text": acc["text"], "tool_calls": [{"name": t["name"], "args": t["args"]} for t in acc["tools"]]}
        sc_s, sc_p = score_item(it, sumz(so), None), score_item(it, sumz(pa), None)
        div = next((i for i, (x, y) in enumerate(zip(so["text_ids"], pa["text_ids"])) if x != y), None)
        leak = s.get("leak_regex")
        import re
        res["sessions"][sid] = {
            "item": s["item"], "frames": len(so["text_ids"]),
            "text_ids_identical": so["text_ids"] == pa["text_ids"], "first_text_div_frame": div,
            # FC cycles are wall-clock driven (live mode): after the first FC, token identity solo vs pair is not
            # expected; the functional gate there is tool calls + scored answer
            "first_fc_chunk_solo": next((r["k"] for r in so["e2e"] if r["fc"]), None),
            "first_fc_chunk_pair": next((r["k"] for r in e2e[sid] if r["fc"]), None),
            "func_ids_identical": so["func_ids"] == pa["func_ids"],
            "tools_solo": so["tools"], "tools_pair": pa["tools"],
            "tool_calls_identical": [(t["name"], t["args"]) for t in so["tools"]] == [(t["name"], t["args"]) for t in pa["tools"]],
            "ok_solo": bool(sc_s.get("ok_model")), "ok_pair": bool(sc_p.get("ok_model")),
            "leak_in_pair": bool(leak and re.search(leak, pa["text"].lower()) and not re.search(leak, so["text"].lower())),
            "ids_window_at_div": None if div is None else {"solo": so["text_ids"][max(0, div - 6):div + 10],
                                                           "pair": pa["text_ids"][max(0, div - 6):div + 10]},
            "text_solo": so["text"][-400:], "text_pair": pa["text"][-400:], "errors": pa.get("errors"),
            "audio_solo": _audio_summary(so.get("audio_stats")), "audio_pair": _audio_summary(pa.get("audio_stats")),
            "audio_solo_vs_pair": _audio_cmp(so.get("audio_chunks"), pa.get("audio_chunks")),
            "solo_step_ms_p50": pct(so["step_ms"], 0.5), "solo_step_ms_p95": pct(so["step_ms"], 0.95),
            "fc_cycles_solo": _fc_cycles(so.get("fc_rec"), so.get("t0_ns")),
            "fc_cycles_pair": _fc_cycles(pa.get("fc_rec"), t0_ns)}
    res["pair_round_ms_p50"], res["pair_round_ms_p95"] = pct(rounds, 0.5), pct(rounds, 0.95)
    res["pair_round_ms_max"] = round(max(rounds), 1) if rounds else None
    fr = [x for x in fc_round_trace if x]
    hist = {}
    for x in fr:
        hist[x["served"]] = hist.get(x["served"], 0) + 1
    res["fc_budget"] = {"merge": a.fc_merge == "on", "merged": sum(x.get("merged", 0) for x in fr),
                        "fallback_b1": sum(x.get("fallback_b1", 0) for x in fr),
                        "budget": None if a.fc_budget < 0 else a.fc_budget, "force_after_s": a.fc_force_after_s,
                        "served_in_round_hist": {str(k2): v for k2, v in sorted(hist.items())},
                        "served_in_rounds": sum(x["served"] for x in fr),
                        "served_between": sum(x["between_before"] for x in fr),
                        "forced": sum(x["forced"] for x in fr), "q_max": max((x["q_max"] for x in fr), default=0),
                        "q_end_last": fr[-1]["q_end"] if fr else None}
    res["pair_deadline_misses"] = f"{misses}/{len(rounds)}"
    busy_s = sum(rounds) / 1e3
    # throughput over chunks that actually ran the Thinker in the session's main loop (FC/tool-pending and empty
    # rounds excluded: they are cheap and would inflate the number)
    n_pair = sum(1 for rows in e2e.values() for r in rows if not r["fc"])
    busy_pair = sum(r["e2e_ms"] - r["start_delay_ms"] for rows in e2e.values() for r in rows if not r["fc"]) / 1e3
    res["pair_round_count"] = len(rounds)
    res["pair_thinker_chunks"], res["pair_busy_s_thinker_chunks"] = n_pair, round(busy_pair, 2)
    res["pair_throughput_session_chunks_per_busy_s"] = round(n_pair / max(1e-9, sum(rounds) / 1e3), 2)
    solo_rows = [(r, ms) for s in sess for r, ms in zip(solo[s["sid"]]["e2e"], solo[s["sid"]]["step_ms"]) if not r["fc"]]
    res["solo_thinker_chunks"] = len(solo_rows)
    res["solo_throughput_session_chunks_per_busy_s"] = round(len(solo_rows) / max(1e-9, sum(ms for _, ms in solo_rows) / 1e3), 2)
    res["feeds"] = {"solo": {s["sid"]: solo[s["sid"]]["feed"] for s in sess},
                    "pair": {sid: f.report() for sid, f in feeds.items()}}
    # validation artefacts: per-session WAVs (solo + pair, for ASR on saved audio) and the raw per-round / per-chunk
    # trace (backlog = start_delay_ms trend over k; steady state = rounds after the warm-up)
    if os.environ.get("V2_SAVE_ARTEFACTS", "1") == "1":
        import soundfile as sf
        from mynah_v2.model import tokens as T
        for s in sess:
            for tag, acc in (("solo", solo[s["sid"]]), ("pair", pair[s["sid"]])):
                ch = acc.get("audio_chunks")
                if ch:
                    d = out / "wav" / f"{s['sid']}_{s['item']}_{tag}"
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / "output.wav", torch.cat([c.reshape(-1) for c in ch]).numpy(), T.OUTPUT_SR)
        (out / "pair_trace.json").write_text(json.dumps({"rounds_ms": [round(x, 2) for x in rounds], "e2e": e2e,
                                                         "fc_rounds": fc_round_trace},
                                                        default=str))
        res["artefacts"] = {"wav_dir": "wav", "trace": "pair_trace.json"}
    if diag is not None:
        diag.uninstall()
        dd = diag.save(out)
        dd["pair_t0_ms"] = round((t0 - diag.t_ref) * 1e3, 3)
        dd["rounds"], dd["timeline"] = diag_rounds, diag_tl
        (out / "diag.json").write_text(json.dumps(dd, default=str))
        res["artefacts_diag"] = {"codes": "codes.npz", "diag": "diag.json", "codes_info": dd["codes"]}
    (out / "pair_gate.json").write_text(json.dumps(res, indent=1, default=str))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "sessions"} |
                                 {"sessions": {k: {x: v[x] for x in ("text_ids_identical", "first_text_div_frame",
                                                                     "tool_calls_identical", "ok_solo", "ok_pair",
                                                                     "leak_in_pair", "errors")}
                                               for k, v in res["sessions"].items()}}, default=str), flush=True)


if __name__ == "__main__":
    main()
