"""G2 + G3 in one GPU job (one engine load, the Thinker graphs captured in G2 are reused by G3).

G2  per-frame load cut, each change checked before its speed counts:
    a) Thinker: the V2 exact profile (mamba_host_pos, no_timer_sync, perception_graph, thinker_graph) on the INT4
       Thinker. Thinker-only sessions on the frozen 52 s Italian track (pace fast, no Nano, no controller): A eager,
       B profile (captures), C profile (replays). Per-frame predicted / emitted text ids must be identical A = B = C.
    b) live ASR: every 4 vs 8 frames, window 12 vs 6 s, offline on the same track: words and characters vs the frozen
       track, ms per call and per frame.
    c) talker: the plain code-predictor loop of G1 (codes identical there) is used when ``--fast-cp``.
G3  the G0 session again with: the profile (only if a) is identical), the chosen ASR stride, the fast predictor,
    the holding phrase not cut, and the audio-aware controller (the turn closes at the end of its audio; speech over
    our audio is a barge-in). Then the content evidence: the ASR transcript of the talker's audio for G3 AND G0.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _norm_words(s: str) -> list:
    import re
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer_cer(ref: str, hyp: str) -> dict:
    rw, hw = _norm_words(ref), _norm_words(hyp)
    rc, hc = "".join(rw), "".join(hw)

    def ed(a, b):
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return {"wer": round(ed(rw, hw) / max(1, len(rw)), 3), "cer": round(ed(rc, hc) / max(1, len(rc)), 3),
            "ref_words": len(rw)}


def thinker_only(eng, wav, prompt, timer, tag):  # pragma: no cover - GPU
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.runner import seed_all
    w = eng.wrapper
    rec = []
    orig = w._apply_rnnt_turn_taking

    def hook(t, gen_text, *a, **kw):
        p = int(gen_text[0, int(t)])
        r = orig(t, gen_text, *a, **kw)
        rec.append((int(t), p, int(gen_text[0, int(t)])))
        return r
    w._apply_rnnt_turn_taking = hook
    seed_all()
    timer.reset()
    tr = V2Tracer(Path(os.environ.get("G23_OUT", "/tmp")) / f"trace_{tag}.jsonl", session_id=tag)
    sess = Session(eng, tr, Mode.REFERENCE, None, on_audio=lambda x: None, controller=None)
    t0 = time.monotonic()
    try:
        sess.open(prompt)
        run_paced(sess, wav, pace="fast", tail_silence_s=1.0, max_extra_s=0.0, drain=True)
        sess.close("normal")
    finally:
        del w._apply_rnnt_turn_taking
    return {"tag": tag, "wall_s": round(time.monotonic() - t0, 1), "frames": len(rec), "ids": rec,
            "comp": timer.report()}


def asr_eval(live, wav, frozen, every, window_s):  # pragma: no cover - GPU
    live.reset(window_s=window_s, every=every)
    words = []
    for k in range(len(wav) // 1280):
        ev = live.step(k, wav)
        words += [{"k": k, "word": x} for x in ev["words"]]
    tl = frozen["timeline"]
    turns = {}
    for i, u in enumerate(tl):
        hi = tl[i + 1]["start_k"] if i + 1 < len(tl) else 10 ** 9
        lw = " ".join(x["word"] for x in words if u["start_k"] <= x["k"] < hi)
        fw = " ".join(x["word"] for x in frozen["words"] if u["start_k"] <= x["k"] < hi)
        lk = [x["k"] for x in words if u["start_k"] <= x["k"] < hi]
        fk = [x["k"] for x in frozen["words"] if u["start_k"] <= x["k"] < hi]
        turns[u["label"]] = {"live": lw, "frozen": fw, **wer_cer(fw, lw),
                             "last_word_delay_frames": (lk[-1] - fk[-1]) if lk and fk else None}
    allf = " ".join(t["frozen"] for t in turns.values())
    alll = " ".join(t["live"] for t in turns.values())
    ms = live.stats["ms"]
    return {"every": live.every, "window_s": window_s, "calls": live.stats["calls"],
            "ms_p50": sorted(ms)[len(ms) // 2] if ms else None, "ms_sum": round(sum(ms), 1),
            "ms_per_frame": round(sum(ms) / max(1, len(wav) // 1280), 2), **wer_cer(allf, alll), "turns": turns}


def main():  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g0-run", required=True, help="the G0 run dir (g0_it/ inside): agent audio + spoken text")
    ap.add_argument("--flash-python", required=True)
    ap.add_argument("--flash-harness", required=True)
    ap.add_argument("--flash-root", required=True)
    ap.add_argument("--flash-voice", required=True)
    ap.add_argument("--fast-cp", action="store_true")
    ap.add_argument("--listen-dir", required=True)
    ap.add_argument("--prebuffer", type=float, default=0.32)
    ap.add_argument("--g3-min-left-s", type=float, default=290.0, help="start G3 only with this much budget left")
    ap.add_argument("--budget-s", type=float, default=585.0)
    a = ap.parse_args()
    T0 = time.monotonic()
    left = lambda: a.budget_s - (time.monotonic() - T0)            # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["G23_OUT"] = str(out)
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "multilingual"))
    sys.path.insert(0, str(lab / "agentbench"))
    import llm_text_bench as B2
    B2.NANO = Path(a.nano_dir)
    import numpy as np
    import soundfile as sf
    import torch
    from m0_runner import load_16k, render_prompt
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from mynah_v2.model.perf import enable
    from mynah_v2.model.quant import quantize_thinker_int4
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.comp_timer import CompTimer
    from e1.fast import safe_no_audio
    from e1.live_asr import LiveASR
    from e1.runner import mem_available_gb
    from e1.tools_bench import POLICY, tool_schemas
    from frankenstein_s2s.lang import session_language, with_language
    res = {"commit": os.environ.get("FRANK_COMMIT"), "mem_before_gb": round(mem_available_gb(), 1)}

    def save():
        (out / "g23.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))

    t0 = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.decode_audio": False}),
                          llm_cache=True)
    safe_no_audio(eng)
    res["int4_thinker"] = quantize_thinker_int4(eng)
    res["engine_load_s"] = round(time.monotonic() - t0, 1)
    res["mem_after_engine_gb"] = round(mem_available_gb(), 1)
    asr_dir = Path(a.asr_dir)
    frozen = json.load(open(asr_dir / "asr_it.json"))
    wav = load_16k(asr_dir / "track_it.wav")[: 52 * 16000]
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    reg = ToolRegistry()
    be = TenantBackend(spec)
    for t in spec["tools"]:
        reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
    prompt = render_prompt(reg.schemas, with_language(spec["system_prompt"], session_language("it")))
    w = eng.wrapper
    from m0_runner import set_model_cfg
    set_model_cfg(w.model_cfg, "force_turn_taking", False)
    print("ENGINE", res["engine_load_s"], "s mem", res["mem_after_engine_gb"], flush=True)

    # ---------------------------------------------------------------- G2 a) Thinker exact profile
    tA = CompTimer()
    pcm = getattr(w, "perception_cache_mgr", None)
    if pcm is not None and hasattr(pcm, "step"):
        tA._wrap(pcm, "step", "perception")
    tA._wrap(w.model.stt_model, "forward", "thinker_b1")
    A = thinker_only(eng, wav, prompt, tA, "A_eager")
    tA.enabled = False
    applied = {}
    for name in ("mamba_host_pos", "no_timer_sync", "perception_graph", "thinker_graph"):
        try:
            applied[name] = {"ok": True, "info": enable(name, eng)}
        except Exception as e:  # noqa: BLE001
            applied[name] = {"ok": False, "error": repr(e)[:300]}
    tB = CompTimer()
    if pcm is not None and hasattr(pcm, "step"):
        tB._wrap(pcm, "step", "perception")
    tB._wrap(w.model.stt_model, "forward", "thinker_b1")
    B = thinker_only(eng, wav, prompt, tB, "B_profile_capture")
    C = thinker_only(eng, wav, prompt, tB, "C_profile_replay")
    tg = getattr(eng, "_v2_thinker_graph", None)
    same = lambda x, y: [r[1:] for r in x["ids"]] == [r[1:] for r in y["ids"]]   # noqa: E731
    first_diff = next((i for i, (p, q) in enumerate(zip(A["ids"], C["ids"])) if p[1:] != q[1:]), None)
    res["thinker"] = {"applied": applied, "graph_stats": dict(tg.stats) if tg is not None else None,
                      "A_eq_B": same(A, B), "A_eq_C": same(A, C), "first_diff_frame_A_C": first_diff,
                      "runs": {r["tag"]: {"wall_s": r["wall_s"], "frames": r["frames"], "comp": r["comp"]}
                               for r in (A, B, C)}}
    profile_ok = res["thinker"]["A_eq_B"] and res["thinker"]["A_eq_C"]
    if not profile_ok:                                   # not identical: back to the eager path for G3
        from mynah_v2.model.perf import disable
        for name in reversed([n for n, v in applied.items() if v["ok"]]):
            try:
                disable(name, eng)
            except Exception as e:  # noqa: BLE001
                applied[name]["disable_error"] = repr(e)[:200]
    res["thinker"]["profile_used_in_g3"] = profile_ok
    save()
    print("G2 THINKER", json.dumps({k: v for k, v in res["thinker"].items() if k != "runs"}, default=str)[:1500],
          json.dumps({k: {c: (v["comp"].get(c) or {}).get("gpu_ms_p50") for c in ("thinker_b1/main", "perception/main")}
                      | {"wall_s": v["wall_s"]} for k, v in res["thinker"]["runs"].items()}), flush=True)

    # ---------------------------------------------------------------- G2 b) live ASR stride / window
    live = LiveASR(str(lab / "multilingual"), lang="it", att_right=3, window_s=12.0)
    res["asr"] = [asr_eval(live, wav, frozen, ev, ws) for ev, ws in ((4, 12.0), (8, 12.0), (8, 6.0))]
    base = res["asr"][0]
    pick = base
    for c in res["asr"][1:]:
        if c["cer"] <= base["cer"] + 0.01 and c["ms_per_frame"] < pick["ms_per_frame"]:
            pick = c
    res["asr_pick"] = {"every": pick["every"], "window_s": pick["window_s"]}
    save()
    print("G2 ASR", json.dumps([{k: v for k, v in c.items() if k != "turns"} for c in res["asr"]]),
          "pick", res["asr_pick"], flush=True)

    # ---------------------------------------------------------------- G3 session
    if left() < a.g3_min_left_s:
        res["g3"] = {"status": "INCOMPLETE", "why": f"budget left {left():.0f} s < {a.g3_min_left_s}"}
        save()
        print("G3 SKIPPED", res["g3"], flush=True)
        return
    t0 = time.monotonic()
    tok, m, _ = B2.load("nano")
    res["nano_load_s"] = round(time.monotonic() - t0, 1)
    res["mem_after_nano_gb"] = round(mem_available_gb(), 1)
    from e1.flash_client import FlashProc
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = FlashProc([a.flash_python, str(Path(__file__).resolve().parent / "flash_server.py"), "--harness",
                      a.flash_harness, "--root", a.flash_root, "--voice", a.flash_voice, "--lang", "it",
                      *(["--fast-cp"] if a.fast_cp else [])], env, str(out / "flash_server.log"))
    proc.start()
    res["flash_ready"] = proc.wait_ready(240)
    res["mem_after_flash_gb"] = round(mem_available_gb(), 1)
    print("G3 LOADS", json.dumps({k: res[k] for k in ("nano_load_s", "mem_after_nano_gb", "flash_ready",
                                                       "mem_after_flash_gb")}), flush=True)
    if res["mem_after_flash_gb"] < 20:
        res["g3"] = {"status": "INCOMPLETE", "why": "memory"}
        proc.close()
        save()
        return
    live.reset(window_s=pick["window_s"], every=pick["every"])
    tB.enabled = False
    timer = CompTimer()
    if pcm is not None and hasattr(pcm, "step"):
        timer._wrap(pcm, "step", "perception")
    timer._wrap(w.model.stt_model, "forward", "thinker_b1")
    timer._wrap(w, "_rnnt_step", "rnnt")
    from e1 import g0_session
    try:
        r = g0_session.run(out, "g3_it", eng, tok, m, spec, POLICY["it"], asr_dir, 52.0, lab / "agentbench",
                           tool_schemas, live, proc, timer=timer, prebuffer_s=a.prebuffer,
                           listen_dir=Path(a.listen_dir), holding_cut="never", audio_aware=True)
    finally:
        proc.close()
    res["g3"] = {"status": "DONE", "g0": r["g0"]}
    save()
    g = r["g0"]
    print("G3", json.dumps({k: g[k] for k in ("frames", "loop_s", "first_audio_after_user_end", "barge_in")},
                           default=str)[:2500], flush=True)
    print("G3 PLAYBACK", json.dumps({k: v for k, v in g["playback"].items() if k != "gap_list"}, default=str)[:1500],
          flush=True)
    print("G3 TALKER", json.dumps(g["talker"], default=str)[:1500], flush=True)

    # ---------------------------------------------------------------- content evidence: ASR of the talker's audio
    from scipy.signal import resample_poly
    content = {}
    for name, run_dir in (("G0", Path(a.g0_run) / "g0_it"), ("G3", out / "g3_it")):
        agent, sr = sf.read(run_dir / "agent_24k.wav", dtype="float32")
        a16 = torch.from_numpy(resample_poly(agent, 2, 3).astype(np.float32))
        hyp = live.transcribe(a16)
        s = json.load(open(run_dir / "summary.json"))
        ref = " ".join(t["text"] for t in s["spoken_turns"])
        content[name] = {"hyp": hyp, "ref": ref, **wer_cer(ref, hyp)}
    res["content"] = content
    save()
    print("CONTENT", json.dumps({k: {x: v[x] for x in ("wer", "cer", "ref_words")} for k, v in content.items()}),
          flush=True)


if __name__ == "__main__":
    main()
