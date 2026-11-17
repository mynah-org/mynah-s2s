"""E1c step 0 (CPU only): multilingual-ASR transcripts of the agentbench caller clips, per clip.

Same model and decoding path as e1/asr_track.py (lab encoder SpeechEncoder("ml", att=[56, 3]) + the lab greedy RNNT
with the language prompt; predictor reset after a language tag), but one clip at a time with 1 s of lead silence and
2 s of tail, so a transcript never depends on another clip. Output: {"<scenario>_t<k>|<lang>": text}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LANG_IDX = {"en": 0, "it": 15, "fr": 8}


def main():  # pragma: no cover - CPU model run on the GPU host
    import torch
    import soundfile as sf
    import torchaudio.functional as AF
    import yaml
    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--roots", required=True, help="comma list of agentbench roots with manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--att-right", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, a.lab_scripts)
    from encoders import ASR_DIRS, SpeechEncoder
    from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
    torch.set_num_threads(8)
    d = ASR_DIRS["ml"]
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    dec = RNNTDecoder.from_config_dict(OmegaConf.create(cfg["decoder"]))
    jnt = RNNTJoint.from_config_dict(OmegaConf.create(cfg["joint"]))
    dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
    jnt.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("joint.")})
    nd = cfg["model_defaults"]["enc_hidden"]
    pk = torch.nn.Sequential(torch.nn.Linear(nd + 128, 2 * nd), torch.nn.ReLU(), torch.nn.Linear(2 * nd, nd))
    pk.load_state_dict({k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
    dec.eval(), jnt.eval(), pk.eval()
    vocab = list(jnt.vocabulary)
    blank = len(vocab)
    enc = SpeechEncoder("ml", att=[56, a.att_right], device="cpu")

    @torch.no_grad()
    def greedy(e, li):
        oh = torch.zeros(e.shape[0], 128)
        oh[:, li] = 1
        f = pk(torch.cat([e, oh], -1))
        g, st = dec.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)
        out = []
        for t in range(f.shape[0]):
            for _ in range(10):
                k = int(jnt.joint(f[t:t + 1][None], g)[0, 0, 0].argmax())
                if k == blank:
                    break
                p = vocab[k]
                g, st = dec.predict(torch.tensor([[k]]), st, add_sos=False, batch_size=1)
                if p.strip().startswith("<") and p.strip().endswith(">"):
                    g, st = dec.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)
                    break
                out.append(p)
        return "".join(out).replace("▁", " ").strip()

    res, t0 = {}, time.monotonic()
    gen = torch.Generator().manual_seed(0)
    for root in a.roots.split(","):
        for m in map(json.loads, open(Path(root) / "manifest.jsonl")):
            key = f"{m['id']}|{m['lang']}"
            if key in res or m["lang"] not in LANG_IDX:
                continue
            w, sr = sf.read(Path(root) / m["path"], dtype="float32", always_2d=True)
            w = torch.from_numpy(w.T.copy()).mean(0)
            w = AF.resample(w[None], sr, 16000)[0] if sr != 16000 else w
            x = torch.cat([torch.zeros(16000), w, torch.zeros(32000)])
            x = x + torch.randn(x.numel(), generator=gen) * 3e-4
            res[key] = {"asr": greedy(enc(x)["out"], LANG_IDX[m["lang"]]), "ref": m.get("text", "")}
            print(key, "|", res[key]["asr"][:100], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("done", len(res), round(time.monotonic() - t0, 1), "s")


if __name__ == "__main__":
    main()
