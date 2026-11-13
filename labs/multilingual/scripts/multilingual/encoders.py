"""Standalone speech encoders (no LLM): VoiceChat perception (VC), nemotron-speech-streaming-en (EN),
nemotron-3.5-asr-streaming multilingual (ML), built with the NeMo modules of the VoiceChat fork and fed with the
same mel front end. Used for feature extraction (map/adapter fitting, probes); the end-to-end path stays the NVIDIA
pipeline (e2e_qa.py).

Offline full-utterance forward with chunked-limited attention [left, 0] and causal convolutions: by construction of
the cache-aware FastConformer this is the computation the streaming cache reproduces chunk by chunk.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

HOME = Path.home()
VC_CKPT = HOME / "mynah-s2s/hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39"
ASR_DIRS = {"en": HOME / "mynah-s2s/ml/ckpt/asr-en", "ml": HOME / "mynah-s2s/ml/ckpt/asr-ml"}
DEFAULT_ATT = {"vc": [70, 0], "en": [70, 0], "ml": [56, 0]}


def _vc_state():
    enc, pre, proj = {}, {}, {}
    with safe_open(str(VC_CKPT / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("stt_model.perception.encoder."):
                enc[k[len("stt_model.perception.encoder."):]] = f.get_tensor(k)
            elif k.startswith("stt_model.perception.preprocessor."):
                pre[k[len("stt_model.perception.preprocessor."):]] = f.get_tensor(k)
            elif k.startswith("stt_model.perception.proj."):
                proj[k.rsplit(".", 1)[1]] = f.get_tensor(k)
    cfg = json.load(open(VC_CKPT / "config.json"))["model"]["stt"]["model"]["perception"]
    return cfg["encoder"], cfg["preprocessor"], enc, pre, proj


def _asr_state(name):
    d = ASR_DIRS[name]
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}  # noqa: E731
    return cfg["encoder"], cfg["preprocessor"], sub("encoder."), sub("preprocessor."), sub("prompt_kernel.")


class SpeechEncoder(torch.nn.Module):
    def __init__(self, name: str, att=None, device="cuda", dtype=torch.float32):
        super().__init__()
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
        self.name = name
        if name == "vc":
            ecfg, pcfg, esd, psd, extra = _vc_state()
        else:
            ecfg, pcfg, esd, psd, extra = _asr_state(name)
        ecfg = {k: v for k, v in ecfg.items() if k != "_target_"}
        ecfg["att_context_size"] = list(att or DEFAULT_ATT[name])
        pcfg = {k: v for k, v in pcfg.items() if k != "_target_"}
        pcfg["dither"] = 0.0
        self.pre = AudioToMelSpectrogramPreprocessor(**pcfg)
        self.pre.load_state_dict(psd, strict=False)
        self.enc = ConformerEncoder(**ecfg)
        missing, unexpected = self.enc.load_state_dict(esd, strict=True), None
        self.proj = None
        if name == "vc":
            self.proj = torch.nn.Linear(1024, 4480)
            self.proj.load_state_dict(extra)
        self.prompt_kernel = None
        if name == "ml" and extra:
            self.prompt_kernel = torch.nn.Sequential(torch.nn.Linear(1024 + 128, 2048), torch.nn.ReLU(),
                                                     torch.nn.Linear(2048, 1024))
            self.prompt_kernel.load_state_dict(extra)
        self.to(device, dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, layers: list[int] | None = None) -> dict:
        """wav [T] float32 16 kHz -> {'out': [F, 1024], 'L<i>': [F, 1024] for requested layer outputs (0-based)}."""
        x = wav.to(self.device, torch.float32)[None]
        n = torch.tensor([x.shape[1]], device=self.device)
        mel, ml = self.pre(input_signal=x, length=n)
        hooks, store = [], {}
        for i in layers or []:
            hooks.append(self.enc.layers[i].register_forward_hook(
                lambda m, inp, out, i=i: store.__setitem__(f"L{i}", out[0] if isinstance(out, tuple) else out)))
        try:
            e, el = self.enc(audio_signal=mel.to(self.dtype), length=ml)
        finally:
            for h in hooks:
                h.remove()
        F = int(el[0])
        res = {"out": e[0, :, :F].transpose(0, 1).float()}
        for k, v in store.items():
            res[k] = v[0, :F].float()
        return res
