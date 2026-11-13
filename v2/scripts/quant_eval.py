"""EXPERIMENT gate for the int4 Thinker (``mynah_v2.model.quant``): same process, same input, deterministic decoding,
cache ON; first the BF16 baseline, then the backbone linears are quantized in place (irreversible for this process)
and the same inputs run again. Reports token agreement per channel, first divergence, decoded texts, per-chunk
latency and memory. Saves both agent audios for OmniASR. Not a parity claim: quantization changes the model."""

from __future__ import annotations

import argparse
import json
import statistics as S
import sys
from pathlib import Path

import soundfile as sf
import torch

from mynah_v2.model.quant import quantize_thinker_int4, restore_thinker
from mynah_v2.model.reference import ReferenceEngine, build_cfg

sys.path.insert(0, str(Path(__file__).parent))
from cache_parity import drive  # noqa: E402
from run_stream import load_16k, render_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc-prompt", action="store_true")
    ap.add_argument("--variants", default="g128",
                    help="comma list of g{128,64,32}[c][s]: group size, c = clipping search, s = first/last 2 layers BF16")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    tok = eng.tokenizer
    if a.fc_prompt:
        import offline_voicechat_fc_infer as fcmod
        prompt = render_prompt(fcmod.DEFAULT_TOOLS)
    else:
        prompt = "You are a helpful assistant."
    xs = {w: load_16k(w) for w in a.wavs}
    base = {}
    torch.cuda.reset_peak_memory_stats()
    for w, x in xs.items():
        base[w] = drive(eng, x, prompt)
    mem_bf16 = torch.cuda.max_memory_allocated() / 2**30
    nl = len(eng.wrapper.model.stt_model.llm.layers)
    rep = {"peak_alloc_gb": {"bf16": round(mem_bf16, 2)}, "variants": {}}

    def txt(ids):
        return tok.ids_to_text([i for i in ids if i > 2 and i != 12])
    for var in a.variants.split(","):
        g = int(var[1:].rstrip("cs"))
        info, orig = quantize_thinker_int4(eng, max_group=g, clip="c" in var[1:],
                                           skip_layers={0, 1, nl - 2, nl - 1} if "s" in var[1:] else None,
                                           keep_original=True)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for w, x in xs.items():
            q = drive(eng, x, prompt)
            b = base[w]
            n = min(len(b["text"]), len(q["text"]))
            stem = Path(w).stem
            sf.write(out / f"{stem}_bf16.wav", b["audio"], 22050)
            sf.write(out / f"{stem}_int4_{var}.wav", q["audio"], 22050)
            m = min(len(b["func"]), len(q["func"]))
            r = {"variant": var, "wav": w, "frames": n,
                 "text_agreement": round(sum(b["text"][i] == q["text"][i] for i in range(n)) / max(n, 1), 4),
                 "func_agreement": round(sum(b["func"][i] == q["func"][i] for i in range(m)) / max(1, m), 4),
                 "first_text_divergence": next((i for i in range(n) if b["text"][i] != q["text"][i]), None),
                 "bf16_text": txt(b["text"]), "int4_text": txt(q["text"]),
                 "ms_p50": {"bf16": S.median(b["ms"]), "int4": S.median(q["ms"])}}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        rep["variants"][var] = {"quant": info, "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                "inputs": rows}
        print(json.dumps({"variant": var, "quant": info}), flush=True)
        restore_thinker(orig)
    (out / "quant_eval.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
