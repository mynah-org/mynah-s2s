"""NVIDIA offline oracle (model-card recipe) without torchaudio I/O.

Same calls as examples/speechlm2/offline_voicechat_{fc_,}infer.py @ 097dfe9e (build_model, encode_system_prompt,
render_fc_system_prompt, run_offline_inference / run_fc_offline_inference, splice_fc_audio_gap); only WAV load/save
use soundfile + torchaudio.functional.resample, because torchaudio 2.10 load/save need torchcodec (no aarch64 wheel
and no FFmpeg on a GB10 machine). Also dumps the per-step token channels for parity checks.
"""

from __future__ import annotations
import os

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

N = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "third_party/agent/nemo-speech"
sys.path.insert(0, str(N / "examples/speechlm2"))


def load_16k(path, device):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy())
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
    return wav, wav.unsqueeze(0).to(device), torch.tensor([wav.shape[0]], device=device)


def main():
    from nemo.collections.speechlm2.inference.utils import offline_voicechat as ov
    import offline_voicechat_fc_infer as fcmod  # NVIDIA defaults (system message + tools)

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc", action="store_true")
    ap.add_argument("--api-response-json")
    ap.add_argument("--tools-json")
    ap.add_argument("--system-message", default=None)
    ap.add_argument("--no-prompt", action="store_true", help="general mode: no system prompt (model card default)")
    ap.add_argument("--stt-bf16", action="store_true",
                    help="cast the STT/Thinker to BF16 (the streaming path's compute dtype; TTS stays as loaded). "
                         "The FP32 model-card run exceeded ~105 GB on GB10 for sample_fc (no-cache recompute).")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    t0 = time.monotonic()
    model = ov.build_model(a.checkpoint, device=dev)
    if a.stt_bf16:
        model.stt_model.to(torch.bfloat16)
    t_load = time.monotonic() - t0
    wav1d, sig, lens = load_16k(a.wav, dev)
    rec = {"wav": a.wav, "load_s": round(t_load, 1), "audio_s": round(wav1d.numel() / 16000, 2),
           "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
           "stt_dtype": str(next(model.stt_model.parameters()).dtype)}
    t1 = time.monotonic()
    if a.fc:
        tools = json.load(open(a.tools_json)) if a.tools_json else fcmod.DEFAULT_TOOLS
        sysmsg = a.system_message or fcmod.DEFAULT_SYSTEM_MESSAGE
        prompt = ov.render_fc_system_prompt(str(fcmod.DEFAULT_TEMPLATE), sysmsg, tools)
        ptoks, plens = ov.encode_system_prompt(model, prompt, device=dev)
        api = json.load(open(a.api_response_json))
        result, fc_out, call_step, resp_step, resp_ids = ov.run_fc_offline_inference(
            model, input_signal=sig, input_signal_lens=lens, prompt_tokens=ptoks, prompt_token_lens=plens,
            api_response=api, device=dev)
        rec.update(fc_output=fc_out, call_step=call_step, response_step=resp_step, resp_token_len=len(resp_ids),
                   prompt_chars=len(prompt))
    else:
        result = ov.run_offline_inference(model, input_signal=sig, input_signal_lens=lens, decode_audio=True)
        call_step = -1
    rec["infer_s"] = round(time.monotonic() - t1, 1)
    rec["text"] = result.get("text", [""])[0]
    for k in ("tokens_text", "tokens_function_pred", "tokens_function", "tokens_asr", "tokens_len"):
        v = result.get(k)
        if isinstance(v, torch.Tensor):
            rec[k] = v[0].tolist() if v.dim() > 1 else v.tolist()
    if result.get("audio") is not None:
        n = int(result["audio_len"][0].item())
        audio = result["audio"][0, :n].float().cpu()
        sf.write(out / "agent_raw.wav", audio.numpy(), 22050)
        if call_step >= 0:
            spliced = ov.splice_fc_audio_gap(audio.unsqueeze(0), call_step, rec["response_step"], rec["resp_token_len"])[0]
            sf.write(out / "agent_spliced.wav", spliced.numpy(), 22050)
        user = AF.resample(wav1d.unsqueeze(0).cpu(), 16000, 22050)[0]
        L = max(user.numel(), audio.numel())
        st = np.stack([np.pad(user.numpy(), (0, L - user.numel())), np.pad(audio.numpy(), (0, L - audio.numel()))], 1)
        sf.write(out / "combined.wav", st, 22050)
        rec["audio_out_s"] = round(n / 22050, 2)
    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "oracle.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("text", "infer_s", "load_s", "max_mem_gb", "dtype")}, ensure_ascii=False))
    if a.fc:
        print("FC:", json.dumps(rec["fc_output"]["tool_calls"], ensure_ascii=False))


if __name__ == "__main__":
    main()
