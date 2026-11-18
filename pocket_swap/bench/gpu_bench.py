"""GPU check of PocketSwapper in both modes (materialise / ondemand): swap timings per tier, footprint, decode-step
latency (p50/p95), time to first audio, RTF, mixed-language concurrent sessions, slot isolation under loads and
evictions, CUDA-graph validity, linear-layer microbench (eager vs CUDA graph), and audio for an ASR content check.
Stages run in priority order and stop when the time budget is spent (what was skipped is recorded).

    python bench/gpu_bench.py --base EN_CLONING.safetensors --base-language english_2026-09_24l \
        --packages packages/ --truth italian_24l=IT.safetensors,... --voice ref.wav --out results/ --budget-s 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_swap.core import SwapCore, TierSpec  # noqa: E402
from pocket_swap.swapper import PocketSwapper  # noqa: E402

IT, FR, EN = "italian_24l", "french_24l", "english_2026-09_24l"
TEXT = {
    IT: ["Il suo ordine numero quattrocentoventitré è stato spedito questa mattina.",
         "La consegna è prevista per giovedì diciassette, tra le nove e le tredici.",
         "Il totale è di centoventotto euro e cinquanta centesimi."],
    FR: ["Votre commande numéro quatre cent vingt-trois a été expédiée ce matin.",
         "La livraison est prévue jeudi dix-sept, entre neuf heures et treize heures.",
         "Le total est de cent vingt-huit euros et cinquante centimes."],
    EN: ["Your order number four hundred twenty-three was shipped this morning.",
         "Delivery is expected on Thursday the seventeenth, between nine and one.",
         "The total is one hundred twenty-eight euros and fifty cents."],
}
SR = 24000


def sync():
    torch.cuda.synchronize()


def mb(x):
    return round(x / 1e6, 1)


def write_wav(path, audio):
    import numpy as np
    import scipy.io.wavfile as wf

    a = audio.float().cpu().numpy().reshape(-1)
    wf.write(path, SR, (np.clip(a, -1, 1) * 32767).astype(np.int16))
    return len(a) / SR


def nbytes(o):
    if torch.is_tensor(o):
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(nbytes(v) for v in o)
    return 0


class StepTimer:
    """Wall time of each autoregressive decode step of a TTSModel instance (prompt passes excluded)."""

    def __init__(self, tts):
        self.steps: list[float] = []
        orig = tts._run_flow_lm_and_increment_step

        def timed(model_state, text_tokens=None, backbone_input_latents=None, audio_conditioning=None):
            t = time.perf_counter()
            out = orig(model_state=model_state, text_tokens=text_tokens,
                       backbone_input_latents=backbone_input_latents, audio_conditioning=audio_conditioning)
            if text_tokens is None and audio_conditioning is None:
                out[1].item()  # the step already syncs on is_eos; make it explicit for timing
                self.steps.append((time.perf_counter() - t) * 1e3)
            return out

        tts._run_flow_lm_and_increment_step = timed


def pctl(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 2) if xs else None


def stream_once(tts, state, text, seed):
    """(audio, ttfa_s, total_s) using the streaming API."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    first = None
    chunks = []
    for c in tts.generate_audio_stream(state, text):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(c)
    total = time.perf_counter() - t0
    return torch.cat(chunks, dim=-1), first, total


def lang_audio(R, out, tag, tts_for, state_for, langs, n_sent=3):
    """Generate n_sent sentences per language; record per-clip TTFA/RTF and decode-step stats per language."""
    res = {}
    for lg in langs:
        tts, st = tts_for(lg), state_for(lg)
        timer = getattr(tts, "_bench_timer", None) or StepTimer(tts)
        tts._bench_timer = timer
        timer.steps.clear()
        clips = []
        for i, txt in enumerate(TEXT[lg][:n_sent]):
            au, ttfa, tot = stream_once(tts, st, txt, 1000 + i)
            sec = write_wav(out / f"{tag}__{lg}__{i}.wav", au)
            clips.append({"i": i, "sec": round(sec, 2), "ttfa_ms": round(ttfa * 1e3, 1), "rtf": round(tot / sec, 3)})
        res[lg] = {"clips": clips, "step_p50_ms": pctl(timer.steps[3:], 50), "step_p95_ms": pctl(timer.steps[3:], 95),
                   "steps": len(timer.steps)}
    R.setdefault("audio", {}).setdefault(tag, {}).update(res)
    print("audio", tag, json.dumps({k: (v["step_p50_ms"], v["step_p95_ms"], [c["rtf"] for c in v["clips"]])
                                    for k, v in res.items()}), flush=True)


def swapper_audio(R, out, tag, sw, voice, langs, n_sent=3):
    for lg in langs:
        sw.open(f"aud-{lg}", lg, voice=voice)
    lang_audio(R, out, tag, lambda lg: sw.model(f"aud-{lg}"), lambda lg: sw.core.binding(f"aud-{lg}").voice_state,
               langs, n_sent)
    for lg in langs:
        sw.close(f"aud-{lg}")


def slot_digest(tts) -> tuple[str, list[int]]:
    h = hashlib.sha256()
    ptrs = []
    items = dict(tts.flow_lm.state_dict(keep_vars=True))
    items.update({"buf." + k: v for k, v in tts.flow_lm.named_buffers()})
    for _, t in sorted(items.items()):
        ptrs.append(t.data_ptr())
        h.update(t.detach().float().sum().cpu().numpy().tobytes())
        h.update(t.detach().float().abs().sum().cpu().numpy().tobytes())
    return h.hexdigest(), ptrs


def linears(tts):
    from pocket_swap.ondemand import DeltaLinear

    return [m for m in tts.flow_lm.transformer.modules() if isinstance(m, (torch.nn.Linear, DeltaLinear))]


def micro(tts, M):
    """Backbone linears only (one decode step's worth), eager vs CUDA graph, batch rows M."""
    mods = linears(tts)
    dt = torch.bfloat16
    xs = {1024: torch.randn(M, 1024, device="cuda", dtype=dt), 4096: torch.randn(M, 4096, device="cuda", dtype=dt)}

    def f():
        with torch.autocast("cuda", dtype=dt):
            for m in mods:
                m(xs[m.in_features])

    def timeit(fn, n=30):
        for _ in range(3):
            fn()
        sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return round((time.perf_counter() - t) / n * 1e3, 3)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    return {"eager_ms": timeit(f), "graph_ms": timeit(g.replay)}


def concurrent(R, out, tag, sw, voice):
    plan = {"A": IT, "B": FR, "C": EN, "D": IT}
    for s, lg in plan.items():
        sw.open(s, lg, voice=voice)
    groups = {str(k): v for k, v in sw.core.group_by_slot(list(plan)).items()}
    res, errs = {}, []

    def worker(s, i):
        try:
            torch.manual_seed(2000 + i)
            t = time.perf_counter()
            a = sw.generate(s, TEXT[plan[s]][i])
            sync()
            res[s] = (a, time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    th = [threading.Thread(target=worker, args=(s, i % 3)) for i, s in enumerate(plan)]
    t0 = time.perf_counter()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0
    audio = 0.0
    for s, (a, _) in res.items():
        audio += write_wav(out / f"{tag}__concurrent__{s}__{plan[s]}.wav", a)
    R.setdefault("concurrent", {})[tag] = {"groups": groups, "wall_s": round(wall, 2), "audio_s": round(audio, 2),
                                          "aggregate_rtf": round(wall / max(audio, 1e-9), 3), "errors": errs}
    print("concurrent", tag, R["concurrent"][tag], flush=True)


def isolation(R, out, tag, sw, voice):
    """Session A generates in its slot while another session loads, switches and evicts: A's slot must not change."""
    errs, res = [], {}
    a = sw.open("A", IT, voice=voice)
    dig = slot_digest(sw.model("A"))

    def gen_a():
        try:
            torch.manual_seed(3000)
            res["A"] = sw.generate("A", TEXT[IT][1])
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    th = threading.Thread(target=gen_a)
    th.start()
    ev = []
    for lg in (FR, EN, FR, EN):
        b = sw.open("X", lg)
        ev.append((lg, b.slot, b.event, b.source, round(b.load_ms, 1)))
    th.join()
    if "A" in res:
        write_wav(out / f"{tag}__isolation__A__{IT}.wav", res["A"])
    R.setdefault("isolation", {})[tag] = {"A_slot": a.slot, "events": ev, "errors": errs,
                                         "A_slot_unchanged": slot_digest(sw.model("A")) == dig}
    print("isolation", tag, R["isolation"][tag], flush=True)
    sw.close("X")
    sw.close("A")


def graph_validity(sw):
    """Capture a graph over the slot's backbone linears, swap the slot's language in place, replay vs eager."""
    sw.core = SwapCore(sw.backend, 1, [TierSpec("pinned", max_items=8)])
    b = sw.open("G", IT)
    tts = sw.backend.slots[b.slot]
    mods = linears(tts)
    xs = {1024: torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16),
          4096: torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)}
    acc = torch.zeros((), device="cuda", dtype=torch.float32)

    def f():
        acc.zero_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for m in mods:
                acc.add_(m(xs[m.in_features]).float().sum())

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    ptr0 = slot_digest(tts)[1]
    g.replay(); sync(); before = acc.item()
    sw.close("G")
    nb = sw.open("G", FR)
    g.replay(); sync(); replay = acc.item()
    f(); sync(); eager = acc.item()
    sw.close("G")
    return {"same_slot": nb.slot == b.slot, "ptrs_unchanged": ptr0 == slot_digest(tts)[1],
            "alloc_gen_unchanged": nb.alloc_gen == b.alloc_gen, "value_changed_by_swap": before != replay,
            "graph_equals_eager_after_swap": abs(replay - eager) <= 1e-3 * max(1.0, abs(eager))}


def tier_timings(sw, reps):
    r = {}
    sw.core = SwapCore(sw.backend, 1, [TierSpec("pinned", max_items=8)])
    m0 = torch.cuda.memory_allocated()
    r["cold_alloc_ms"] = round(sw.open("t", IT).load_ms, 1)
    r["slot_alloc_MB"] = mb(torch.cuda.memory_allocated() - m0)
    r["cold_disk_ms"] = round(sw.open("t", FR).load_ms, 1)

    def loop(expect):
        ms = []
        for _ in range(reps):
            for lg in (IT, FR):
                b = sw.open("t", lg)
                assert b.source == expect, (b.source, expect)
                ms.append(b.load_ms)
        return round(statistics.median(ms), 1)

    r["warm_pinned_ms"] = loop("pinned")
    t = time.perf_counter()
    assert sw.open("t2", FR).event == "hot"
    r["hot_vram_ms"] = round((time.perf_counter() - t) * 1e3, 3)
    r["package_host_MB"] = mb(sw.core.tier("pinned").nbytes / 2)
    sw.close("t"); sw.close("t2")
    sw.core = SwapCore(sw.backend, 1, [TierSpec("ram", max_items=8)])
    sw.open("t", IT); sw.open("t", FR)
    r["warm_ram_ms"] = loop("ram")
    sw.close("t")
    sw.core = SwapCore(sw.backend, 1, [])
    sw.open("t", IT)
    r["disk_pagecache_ms"] = loop("disk")
    sw.close("t")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-language", required=True)
    ap.add_argument("--packages", required=True)
    ap.add_argument("--truth", default="")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stages", default="", help="comma list of stage-name prefixes to run (default: all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()
    R: dict = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "torch": torch.__version__, "skipped": [], "stage_s": {}}
    stopfile = os.environ.get("POCKET_SWAP_BENCH_STOPFILE")
    torch.zeros(1, device="cuda")
    free0, total = torch.cuda.mem_get_info()
    R["device_total_MB"], R["free_after_context_MB"] = mb(total), mb(free0)

    def save():
        (out / "bench.json").write_text(json.dumps(R, indent=1))

    def stage(name, fn, need_s):
        if a.stages and not any(name.startswith(x) for x in a.stages.split(",")):
            return
        el = time.perf_counter() - T0
        if stopfile and Path(stopfile).exists():
            R["skipped"].append(f"{name} (foreign GPU job)")
            return
        if el + need_s > a.budget_s:
            R["skipped"].append(f"{name} (budget: {el:.0f}s used, ~{need_s}s needed)")
            return
        t = time.perf_counter()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            R.setdefault("errors", {})[name] = traceback.format_exc()[-2500:]
            print("STAGE ERROR", name, repr(e), flush=True)
        R["stage_s"][name] = round(time.perf_counter() - t, 1)
        torch.cuda.empty_cache()
        save()

    def mk(prec, mode, n_slots=4, fused=None):
        torch.cuda.empty_cache()
        sync()
        m0 = torch.cuda.memory_allocated()
        t = time.perf_counter()
        sw = PocketSwapper(a.base, a.base_language, str(Path(a.packages) / prec), n_slots=n_slots, device="cuda",
                           dtype=torch.bfloat16, mode=mode, delta_precision=prec if mode == "ondemand" else None,
                           pinned_max_languages=8, fused=fused)
        sync()
        return sw, {"init_s": round(time.perf_counter() - t, 2),
                    "base_plus_mimi_MB": mb(torch.cuda.memory_allocated() - m0),
                    "slot_bytes_MB": mb(sw.backend.slot_bytes)}

    def full_mode(mode, prec, fused=None):
        def run():
            sw, info = mk(prec, mode, fused=fused)
            tag = f"{mode}_{prec}" + ("" if mode == "materialise" else ("_fused" if fused else "_torch"))
            R.setdefault("modes", {})[tag] = info
            info["tiers"] = tier_timings(sw, a.reps)
            print(tag, json.dumps(info), flush=True)
            save()
            sw.core = SwapCore(sw.backend, 4, [TierSpec("pinned", max_items=8)])
            sync(); m0 = torch.cuda.memory_allocated()
            for lg in (IT, FR, EN):
                sw.open(f"f-{lg}", lg, voice=a.voice)
            sync()
            info["three_slots_plus_states_MB"] = mb(torch.cuda.memory_allocated() - m0)
            info["voice_state_MB"] = mb(nbytes(sw.core.binding(f"f-{IT}").voice_state))
            torch.cuda.reset_peak_memory_stats(); m1 = torch.cuda.memory_allocated()
            torch.manual_seed(5); sw.generate(f"f-{IT}", TEXT[IT][2]); sync()
            info["generate_peak_extra_MB"] = mb(torch.cuda.max_memory_allocated() - m1)
            info["allocated_MB"], info["reserved_MB"] = mb(torch.cuda.memory_allocated()), mb(torch.cuda.memory_reserved())
            info["micro"] = {f"M{M}": micro(sw.model(f"f-{IT}"), M) for M in (1, 4)}
            for lg in (IT, FR, EN):
                sw.close(f"f-{lg}")
            save()
            swapper_audio(R, out, tag, sw, a.voice, [IT, FR, EN])
            concurrent(R, out, tag, sw, a.voice)
            for s in list(sw.core.sessions):
                sw.close(s)
            sw.core = SwapCore(sw.backend, 2, [TierSpec("pinned", max_items=8)])
            isolation(R, out, tag, sw, a.voice)
            info["cuda_graph"] = graph_validity(sw)
            print(tag, "graph", info["cuda_graph"], flush=True)
            del sw
        return run

    def light_mode(mode, prec, langs):
        def run():
            sw, info = mk(prec, mode, n_slots=4)
            tag = f"{mode}_{prec}"
            R.setdefault("modes", {})[tag] = info
            sw.open("m", IT, voice=a.voice)
            info["micro"] = {"M1": micro(sw.model("m"), 1)}
            sw.close("m")
            swapper_audio(R, out, tag, sw, a.voice, langs)
            del sw
        return run

    def micro_fp32(m):
        mods = [x for x in m.flow_lm.transformer.modules() if isinstance(x, torch.nn.Linear)]
        xs = {1024: torch.randn(1, 1024, device="cuda"), 4096: torch.randn(1, 4096, device="cuda")}
        f = lambda: [x(xs[x.in_features]) for x in mods]  # noqa: E731
        for _ in range(3):
            f()
        sync(); t = time.perf_counter()
        for _ in range(30):
            f()
        sync()
        return {"eager_ms": round((time.perf_counter() - t) / 30 * 1e3, 3)}

    def truth():
        from pocket_tts import TTSModel
        from pocket_tts.utils.config import CONFIGS_DIR
        import yaml

        for item in filter(None, a.truth.split(",")):
            lg, path = item.split("=", 1)
            cfg = yaml.safe_load((Path(CONFIGS_DIR) / f"{lg}.yaml").read_text())
            cfg["weights_path"] = cfg["weights_path_without_voice_cloning"] = path
            tok = Path(a.packages) / "int8" / lg / "tokenizer.json"  # same file as upstream's (copied at build)
            cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tok)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
            sync(); m0 = torch.cuda.memory_allocated()
            m = TTSModel.load_model(config=f.name).to("cuda")
            sync()
            R.setdefault("truth_model_MB", {})[lg] = mb(torch.cuda.memory_allocated() - m0)
            st = m.get_state_for_audio_prompt(a.voice)
            lang_audio(R, out, "truth_fp32", lambda _: m, lambda _: st, [lg])
            R.setdefault("truth_micro", {})[lg] = micro_fp32(m)
            del m, st
            torch.cuda.empty_cache()

    def fused_check():
        from pocket_swap import fused
        from pocket_swap.ondemand import DeltaLinear
        from pocket_swap.quant import quantize

        res = {"available": fused.available(torch.zeros(1, device="cuda"))}
        for prec in ("bf16", "int8", "int4"):
            for shape in ((4096, 1024), (1024, 4096)):
                base = (torch.randn(*shape, device="cuda") * 0.03).to(torch.bfloat16)
                d = torch.randn(*shape, device="cuda") * 0.03
                ref_l, fu_l = DeltaLinear(base, prec, 128, fused=False), DeltaLinear(base, prec, 128, fused=True)
                for dl in (ref_l, fu_l):
                    dl.load(d.to(torch.bfloat16)) if prec == "bf16" else dl.load(*quantize(d, 8 if prec == "int8" else 4, 128))
                for M in (1, 4):
                    x = torch.randn(M, shape[1], device="cuda", dtype=torch.bfloat16)
                    r, f = ref_l(x).float(), fu_l(x).float()
                    res[f"{prec}_{shape[0]}x{shape[1]}_M{M}_maxrel"] = round(((r - f).abs().max() / r.abs().max()).item(), 5)
        res["ok"] = res["available"] and all(v < 2e-2 for k, v in res.items() if k.endswith("maxrel"))
        R["fused_check"] = res
        print("fused_check", res, flush=True)

    stage("fused_check", fused_check, 20)
    fused_ok = R.get("fused_check", {}).get("ok", False)
    for prec in ("int8", "int4", "bf16"):
        if fused_ok:
            stage(f"ondemand_{prec}_fused", full_mode("ondemand", prec, fused=True), 60)
        else:
            R["skipped"].append(f"ondemand_{prec}_fused (fused kernel unavailable or wrong)")
        stage(f"materialise_{prec}", full_mode("materialise", prec), 50)
        stage(f"ondemand_{prec}_torch", full_mode("ondemand", prec, fused=False), 60)
    stage("truth_fp32", truth, 60)
    stage("materialise_exact(plain bf16)", light_mode("materialise", "exact", [IT, FR, EN]), 40)
    R["max_memory_allocated_MB"] = mb(torch.cuda.max_memory_allocated())
    R["total_s"] = round(time.perf_counter() - T0, 1)
    save()
    print("done", R["total_s"], "skipped", R["skipped"], flush=True)


if __name__ == "__main__":
    main()
