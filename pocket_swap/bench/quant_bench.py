"""GPU check of quantised end-to-end Pocket (weight-only int8 / int4 / fp8, quantised base + quantised delta) against
bf16 and the true checkpoints, plus a capacity probe (N concurrent sessions, per-session state, host RSS during load,
admission control). Audio for an ASR / speaker-similarity / spectral comparison is written per variant.

    python bench/quant_bench.py --base EN_CLONING.safetensors --base-language english_2026-09_24l \
        --packages-root packages/ --truth italian_24l=IT.safetensors,... --voice ref.wav --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_bench import EN, FR, IT, TEXT, lang_audio, mb, micro, nbytes, sync, write_wav  # noqa: E402
from pocket_swap.lru import AdmissionRefused  # noqa: E402
from pocket_swap.swapper import PocketSwapper  # noqa: E402

VARIANTS = [  # tag, package dir, kwargs
    ("bf16_materialise", "exact", dict(mode="materialise")),
    ("od_d8_base_bf16", "int8", dict(mode="ondemand")),
    ("od_w8", "w8", dict(mode="ondemand")),
    ("od_w4", "w4", dict(mode="ondemand")),
    ("od_wfp8", "wfp8", dict(mode="ondemand")),
    ("od_d8_base_int8", "int8", dict(mode="ondemand", base_precision="int8")),
    ("od_d4_base_int4", "int4", dict(mode="ondemand", base_precision="int4")),
]


def host_mem():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            d[k] = round(int(v.split()[0]) / 2 ** 20, 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-language", required=True)
    ap.add_argument("--packages-root", required=True)
    ap.add_argument("--truth", default="")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=600)
    ap.add_argument("--variants", default="")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--only", default="", help="comma list of stage names to run (default: all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()
    R: dict = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "skipped": [], "errors": {}, "host_start": host_mem()}
    stopfile = os.environ.get("POCKET_SWAP_BENCH_STOPFILE")

    def save():
        (out / "bench.json").write_text(json.dumps(R, indent=1, default=str))

    def stage(name, fn, need_s):
        if a.only and name not in a.only.split(","):
            return
        el = time.perf_counter() - T0
        if stopfile and Path(stopfile).exists():
            R["skipped"].append(f"{name} (foreign GPU job)")
            return
        if el + need_s > a.budget_s:
            R["skipped"].append(f"{name} (budget)")
            return
        t = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001
            import traceback

            R["errors"][name] = traceback.format_exc()[-2500:]
            print("STAGE ERROR", name, R["errors"][name][-300:], flush=True)
        R.setdefault("stage_s", {})[name] = round(time.perf_counter() - t, 1)
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        save()

    def fused_check():
        from pocket_swap import fused
        from pocket_swap.ondemand import DeltaLinear
        from pocket_swap.quant import quantize_any

        res = {}
        for bp, lp in (("bf16", "int8"), (None, "int8"), (None, "int4"), (None, "fp8"), ("int8", "int8"),
                       ("int4", "int4")):
            for shape in ((4096, 1024), (1024, 4096)):
                W0 = torch.randn(*shape, device="cuda") * 0.03
                D = torch.randn(*shape, device="cuda") * 0.03
                lay = {}
                for fz in (False, True):
                    if bp is None:
                        dl = DeltaLinear(None, lp, 128, fused=fz, shape=shape, device="cuda")
                        dl.load(*quantize_any(W0 + D, lp, 128))
                    else:
                        bt = (bp,) + quantize_any(W0, bp, 128) if bp != "bf16" else ("bf16", W0.to(torch.bfloat16), None)
                        dl = DeltaLinear(None, lp, 128, fused=fz, base_term=bt, shape=shape, device="cuda")
                        dl.load(*quantize_any(D, lp, 128))
                    lay[fz] = dl
                x = torch.randn(4, shape[1], device="cuda", dtype=torch.bfloat16)
                r, f = lay[False](x).float(), lay[True](x).float()
                res[f"{bp}+{lp}_{shape[0]}x{shape[1]}"] = round(((r - f).abs().max() / r.abs().max()).item(), 5)
        res["fused_available_fp8"] = fused.available(torch.zeros(1, device="cuda"), ("fp8",))
        res["ok"] = all(v < 2e-2 for k, v in res.items() if "x" in k)
        R["fused_check"] = res
        print("fused_check", res, flush=True)

    def variant(tag, pk, kw):
        def run():
            torch.cuda.empty_cache(); sync()
            m0 = torch.cuda.memory_allocated()
            h0 = host_mem()
            t = time.perf_counter()
            sw = PocketSwapper(a.base, a.base_language, str(Path(a.packages_root) / pk), n_slots=3,
                               pinned_max_languages=4, **kw)
            sync()
            info = {"init_s": round(time.perf_counter() - t, 2), "fixed_MB": mb(torch.cuda.memory_allocated() - m0),
                    "base_MB": mb(sw.backend.base_nbytes), "slot_MB": mb(sw.backend.slot_bytes),
                    "precision": sw.backend.delta_precision, "host_after_init": host_mem(), "host_before": h0}
            sync(); m1 = torch.cuda.memory_allocated()
            for lg in (IT, FR, EN):
                sw.open(f"v-{lg}", lg, voice=a.voice)
            sync()
            info["three_langs_MB"] = mb(torch.cuda.memory_allocated() - m1)
            info["micro"] = {f"M{M}": micro(sw.model(f"v-{IT}"), M) for M in (1, 4)}
            R.setdefault("variants", {})[tag] = info
            lang_audio(R, out, tag, lambda lg: sw.model(f"v-{lg}"), lambda lg: sw.core.binding(f"v-{lg}").voice_state,
                       [IT, FR, EN])
            info["peak_MB"] = mb(torch.cuda.max_memory_allocated())
            print(tag, json.dumps({k: v for k, v in info.items() if k != "micro"}), info["micro"], flush=True)
            for lg in (IT, FR, EN):
                sw.close(f"v-{lg}")
            del sw
        return run

    def truth():
        from pocket_tts import TTSModel
        from pocket_tts.utils.config import CONFIGS_DIR
        import yaml

        for item in filter(None, a.truth.split(",")):
            lg, path = item.split("=", 1)
            cfg = yaml.safe_load((Path(CONFIGS_DIR) / f"{lg}.yaml").read_text())
            cfg["weights_path"] = cfg["weights_path_without_voice_cloning"] = path
            cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(Path(a.packages_root) / "int8" / lg / "tokenizer.json")
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
            m = TTSModel.load_model(config=f.name).to("cuda")
            st = m.get_state_for_audio_prompt(a.voice)
            lang_audio(R, out, "truth_fp32", lambda _: m, lambda _: st, [lg])
            del m, st
            torch.cuda.empty_cache()

    def capacity():
        """N sessions over 3 languages generating at once: per-session state, peak memory, aggregate RTF; then
        admission control with a slot budget of 2 slots."""
        from pocket_swap.config import preset

        torch.cuda.empty_cache(); sync()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        h0 = host_mem()
        sw = preset("l40s-16g").build(a.base, a.base_language, str(Path(a.packages_root) / "w8"))
        sync()
        cap = {"preset": "l40s-16g", "packages": "w8", "fixed_MB": mb(torch.cuda.memory_allocated() - m0),
               "host_before": h0, "host_after_init": host_mem()}
        langs = [IT, FR, EN]
        sess = [f"s{i}" for i in range(a.sessions)]
        sync(); m1 = torch.cuda.memory_allocated()
        for i, s in enumerate(sess):
            sw.open(s, langs[i % 3], voice=a.voice)
        sync()
        cap["sessions_bound_MB"] = mb(torch.cuda.memory_allocated() - m1)
        cap["voice_state_MB_each"] = mb(nbytes(sw.core.binding(sess[0]).voice_state))
        res, errs = {}, []

        def worker(s, i):
            try:
                torch.manual_seed(4000 + i)
                res[s] = sw.generate(s, TEXT[langs[i % 3]][i % 3])
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        torch.cuda.reset_peak_memory_stats(); sync(); m2 = torch.cuda.memory_allocated()
        th = [threading.Thread(target=worker, args=(s, i)) for i, s in enumerate(sess)]
        t0 = time.perf_counter()
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.perf_counter() - t0
        audio = sum(write_wav(out / f"capacity__{s}__{langs[i % 3]}__{i % 3}.wav", res[s]) for i, s in enumerate(sess)
                    if s in res)
        cap.update(gen_peak_extra_MB=mb(torch.cuda.max_memory_allocated() - m2), wall_s=round(wall, 2),
                   audio_s=round(audio, 2), aggregate_rtf=round(wall / max(audio, 1e-9), 3), errors=errs,
                   groups={str(k): v for k, v in sw.groups(sess).items()}, host_after_gen=host_mem(),
                   allocated_MB=mb(torch.cuda.memory_allocated()), reserved_MB=mb(torch.cuda.memory_reserved()))
        for s in sess:
            sw.close(s)
        del sw
        torch.cuda.empty_cache()
        # admission: budget for exactly 2 slots, 3 languages requested while the first two are held
        sw = preset("l40s-16g", max_slots=4).build(a.base, a.base_language, str(Path(a.packages_root) / "w8"))
        sw.backend.slot_budget = 2 * sw.backend.slot_bytes
        sw.open("x", IT); sw.open("y", FR)
        try:
            sw.open("z", EN, timeout=0)
            cap["admission"] = "NOT refused"
        except AdmissionRefused as e:
            cap["admission"] = f"refused: {e}"
        sw.close("y")
        b = sw.open("z", EN, timeout=0)
        cap["admission_after_release"] = f"{b.event} slot {b.slot}"
        del sw
        R["capacity"] = cap
        print("capacity", json.dumps(cap, default=str), flush=True)

    stage("fused_check", fused_check, 20)
    want = set(filter(None, a.variants.split(",")))
    for tag, pk, kw in VARIANTS:
        if want and tag not in want:
            continue
        stage(tag, variant(tag, pk, kw), 45)
    stage("truth_fp32", truth, 50)
    stage("capacity", capacity, 60)
    R["total_s"] = round(time.perf_counter() - T0, 1)
    R["host_end"] = host_mem()
    save()
    print("done", R["total_s"], R["skipped"], list(R["errors"]), flush=True)


if __name__ == "__main__":
    main()
