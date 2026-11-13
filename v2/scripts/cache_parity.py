"""Correctness + speed gate for the V2 incremental Thinker (``mynah_v2.model.llm_cache``).

Same process, same engine, same input, deterministic decoding: run the V2 driver with the reference no-cache path,
then with the hybrid cache, and compare the text/function token ids frame by frame plus the per-step latency.
Saves both agent audios (for OmniASR) and a JSON report per input.
"""

from __future__ import annotations

import argparse
import json
import statistics as S
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mynah_v2.model.llm_cache import disable_hybrid_llm_cache, enable_hybrid_llm_cache
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def drive(eng: ReferenceEngine, wav: torch.Tensor, prompt: str) -> dict:
    eng.open(prompt)
    ids_t, ids_f, ms, audio = [], [], [], []
    chunks = list(iter_chunks(wav, eng.chunk_samples))
    for i, c in enumerate(chunks):
        o = eng.step(c, is_last=(i == len(chunks) - 1))
        ids_t += o.new_text_ids
        ids_f += o.new_func_ids
        ms.append(o.latency_ms)
        audio.append(o.audio.float().cpu().reshape(-1))
    tt = eng.token_tensors()
    if tt is not None:
        gt, _, n, gf = tt
        ids_t = gt[0, :n].tolist()
        ids_f = gf[0, :n].tolist() if gf is not None else ids_f
    eng.close()
    return {"text": ids_t, "func": ids_f, "ms": ms, "audio": torch.cat(audio).numpy() if audio else np.zeros(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="You are a helpful assistant.")
    ap.add_argument("--nondeterministic", action="store_true")
    ap.add_argument("--ssm-dtype", default="float32", choices=["float32", "bfloat16", "model"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = not a.nondeterministic
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": det}),
                          llm_cache=False)
    tok = eng.tokenizer
    report = []
    for w in a.wavs:
        x = load_16k(w)
        disable_hybrid_llm_cache(eng)
        ref = drive(eng, x, a.prompt)
        info = enable_hybrid_llm_cache(eng, ssm_dtype=None if a.ssm_dtype == "model" else getattr(torch, a.ssm_dtype))
        cac = drive(eng, x, a.prompt)
        disable_hybrid_llm_cache(eng)
        stem = Path(w).stem
        for tag, r in (("nocache", ref), ("cache", cac)):
            sf.write(out / f"{stem}_{tag}.wav", r["audio"], 22050)
        n = min(len(ref["text"]), len(cac["text"]))
        div_t = next((i for i in range(n) if ref["text"][i] != cac["text"][i]), None)
        m = min(len(ref["func"]), len(cac["func"]))
        div_f = next((i for i in range(m) if ref["func"][i] != cac["func"][i]), None)
        half = len(ref["ms"]) // 2
        r = {"wav": w, "deterministic": det, "cache": info, "frames": [len(ref["text"]), len(cac["text"])],
             "text_identical": ref["text"] == cac["text"], "func_identical": ref["func"] == cac["func"],
             "first_text_divergence": div_t, "first_func_divergence": div_f,
             "text_agreement": round(sum(ref["text"][i] == cac["text"][i] for i in range(n)) / max(n, 1), 4),
             "nocache_ms": {"p50": S.median(ref["ms"]), "p95": sorted(ref["ms"])[int(.95 * len(ref["ms"]))],
                            "p50_first_half": S.median(ref["ms"][:half]), "p50_second_half": S.median(ref["ms"][half:])},
             "cache_ms": {"p50": S.median(cac["ms"]), "p95": sorted(cac["ms"])[int(.95 * len(cac["ms"]))],
                          "p50_first_half": S.median(cac["ms"][:half]), "p50_second_half": S.median(cac["ms"][half:])},
             "nocache_text": tok.ids_to_text([i for i in ref["text"] if i > 2 and i != 12]) if hasattr(tok, "ids_to_text") else None,
             "cache_text": tok.ids_to_text([i for i in cac["text"] if i > 2 and i != 12]) if hasattr(tok, "ids_to_text") else None}
        report.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        (out / f"{stem}.json").write_text(json.dumps({**r, "nocache_ids": ref["text"], "cache_ids": cac["text"]}))
    (out / "cache_parity.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
