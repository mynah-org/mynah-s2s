"""End-to-end spoken-QA comprehension through the real NVIDIA VoiceChat streaming pipeline, per front-end variant.

One engine (V2 ReferenceEngine: native NVIDIA StreamingS2SPipeline, deterministic, decode_audio=False so EarTTS is
not run — the agent's *internal text* is what is scored here; spoken-audio checks are a separate step), hybrid LLM
cache on (token-identical to the reference path, results/v2/cache-parity). For each variant the speech front end is
swapped in place (perception_swap.FrontEnd), then every clip is streamed in 160 ms chunks followed by trailing
silence. Per clip we keep: agent text, user-ASR (RNNT) text, function-channel text, first agent-token frame.

Score: an item is correct if any answer regex matches the normalised agent text (lower-case, accents stripped).
This measures comprehension of the question, not the language of the reply (recorded separately).

Variant spec (comma-free, ':'-separated fields):
  vc                         original VoiceChat front end
  enc=en|ml[;att=56][;rnnt=vc|ml][;lang=auto|it|...][;ln=orig|<ln.pt>][;map=<file.pt>][;adapter=<file.pt>]
e.g.  "enc=ml;att=56;rnnt=ml"   "enc=ml;att=56;rnnt=ml;map=/path/ridge.pt"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from labpaths import LAB_HOME  # noqa: E402  (MYNAH_HOME / MYNAH_LAB_HOME, no hard-coded home layout)

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perception_swap import FrontEnd, load_nemo_dir  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def score(text: str, answers: list[str]) -> bool:
    t = norm(text)
    return any(re.search(a, t) for a in answers)


def parse_variant(spec: str) -> dict:
    if spec == "vc":
        return {"name": "vc"}
    d = {"name": spec}
    for f in spec.split(";"):
        k, v = f.split("=", 1)
        d[k] = v
    return d


def apply_variant(fe: FrontEnd, v: dict, ckpt: dict, refresh_silence: bool = True):
    """refresh_silence (default since 2026-11-12): recompute the FC-fast-path silence embedding through the active
    front-end (before that date it stayed the VC one after a swap; QA runs had no FC, M1 FC windows did)."""
    _apply_variant(fe, v, ckpt)
    if refresh_silence:
        fe.refresh_silence_embedding()


def _apply_variant(fe: FrontEnd, v: dict, ckpt: dict):
    fe.restore()
    if v["name"] == "vc":
        return
    sd, cfg = ckpt[v["enc"]]
    fe.set_encoder(sd, v["enc"])
    if "att" in v:
        fe.set_att_context([int(v["att"]), 0])
    if v.get("rnnt", "vc") == "ml":
        msd, mcfg = ckpt["ml"]
        fe.set_rnnt_ml(msd, mcfg, v.get("lang", "auto"))
    if "ln" in v:  # E5: dual encoder, LLM path with trained LayerNorms ("orig" = identical copy, parity control)
        ln = None if v["ln"] == "orig" else torch.load(v["ln"], map_location="cpu", weights_only=True)
        fe.set_dual_ln(ln, Path(v["ln"]).name)
    if "map" in v:
        m = torch.load(v["map"], map_location="cpu", weights_only=True)
        fe.set_proj_affine(m["M"], m["m"], f"map:{Path(v['map']).name}")
    if "adapter" in v:
        from adapters import load_adapter
        fe.set_adapter(load_adapter(v["adapter"]), f"adapter:{Path(v['adapter']).name}")


def drive(eng, wav: torch.Tensor, prompt: str, tail_s: float) -> dict:
    from mynah_v2.runtime.session import iter_chunks
    x = torch.cat([wav, torch.zeros(int(tail_s * 16000))])
    eng.open(prompt)
    text, asr, func, ms = "", "", "", []
    first_tok = None
    speech_end_frame = int(np.ceil(len(wav) / 1280))
    chunks = list(iter_chunks(x, eng.chunk_samples))
    for i, c in enumerate(chunks):
        o = eng.step(c, is_last=(i == len(chunks) - 1))
        text += o.text_delta
        asr += o.asr_delta
        func += o.func_delta
        ms.append(o.latency_ms)
        if first_tok is None and o.text_delta.strip():
            first_tok = o.frame_idx
    eng.close()
    return {"agent_text": text.strip(), "user_asr": asr.strip(), "func_text": func.strip(),
            "speech_end_frame": speech_end_frame, "first_text_frame": first_tok,
            "ms_p50": float(np.median(ms)), "ms_total": float(np.sum(ms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-root", required=True, help="dir with manifest.jsonl from render_qa.py")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--langs", default="en,it,fr")
    ap.add_argument("--voices", default="aiden")
    ap.add_argument("--variants", nargs="+", default=["vc"])
    ap.add_argument("--en-dir", default=str(LAB_HOME / "ckpt/asr-en"))
    ap.add_argument("--ml-dir", default=str(LAB_HOME / "ckpt/asr-ml"))
    ap.add_argument("--prompt", default="You are a helpful assistant.")
    ap.add_argument("--tail-s", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mynah_v2.model.llm_cache import enable_hybrid_llm_cache
    from mynah_v2.model.reference import ReferenceEngine, build_cfg

    out = Path(a.out)
    if (out / "rows.jsonl").exists():
        raise SystemExit(f"{out} already holds results; choose a new --out (results are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    answers = {it["id"]: it["answers"] for it in json.load(open(a.corpus))["items"]}
    root = Path(a.qa_root)
    clips = [json.loads(l) for l in open(root / "manifest.jsonl")]
    clips = [c for c in clips if c["split"] == a.split and c["lang"] in a.langs.split(",")
             and c["voice"] in a.voices.split(",")]
    if a.limit:
        clips = clips[: a.limit]
    print(f"{len(clips)} clips, variants {a.variants}", flush=True)

    t0 = time.monotonic()
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"),
                                    overrides={"s2s.deterministic": True, "s2s.decode_audio": False}))
    cache_info = enable_hybrid_llm_cache(eng)
    # decode_audio=False: the pipeline's log_output indexes audio_wave unconditionally (streaming_s2s_pipeline.py:847)
    # and the wrapper returns None -> hand it an empty [B, 0] waveform instead (text/state updates unchanged).
    _orig_log = eng.pipeline.log_output

    def _log_output(frames, audio_wave, *rest, **kw):
        if audio_wave is None:
            audio_wave = torch.zeros(len(frames), 0)
        return _orig_log(frames, audio_wave, *rest, **kw)
    eng.pipeline.log_output = _log_output
    print(f"engine loaded in {time.monotonic() - t0:.0f}s; cache {cache_info}", flush=True)
    fe = FrontEnd(eng)
    fe.snapshot()
    ckpt = {"en": load_nemo_dir(a.en_dir), "ml": load_nemo_dir(a.ml_dir)}

    rows_path = out / "rows.jsonl"
    summary = {}
    for spec in a.variants:
        v = parse_variant(spec)
        apply_variant(fe, v, ckpt)
        print("variant", spec, fe.state, flush=True)
        per = {}
        for c in clips:
            wav, sr = sf.read(root / c["path"], dtype="float32")
            assert sr == 16000
            r = drive(eng, torch.from_numpy(wav), a.prompt, a.tail_s)
            r.update({"variant": spec, "frontend": dict(fe.state), "id": c["id"], "lang": c["lang"],
                      "voice": c["voice"], "cat": c["cat"], "question": c["text"],
                      "correct": score(r["agent_text"], answers[c["id"]])})
            with open(rows_path, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            per.setdefault(c["lang"], []).append(r["correct"])
            print(f"[{spec}] {c['lang']} {c['voice']} {c['id']} {'OK ' if r['correct'] else 'NO '} "
                  f"| asr: {r['user_asr'][:60]!r} | agent: {r['agent_text'][:110]!r}", flush=True)
        summary[spec] = {l: {"acc": round(sum(x) / len(x), 3), "n": len(x)} for l, x in per.items()}
        print("SUMMARY", spec, json.dumps(summary[spec]), flush=True)
    (out / "summary.json").write_text(json.dumps({"split": a.split, "langs": a.langs, "voices": a.voices,
                                                  "prompt": a.prompt, "tail_s": a.tail_s, "cache": cache_info,
                                                  "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
