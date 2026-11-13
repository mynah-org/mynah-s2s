# Notices and credits

The MIT License in `LICENSE` covers **only the code in this repository**. mynah-s2s does not contain or redistribute
model weights, and it does not vendor third-party source code. At runtime it loads components that have their own
licenses; you are responsible for complying with them. This file is informational, not legal advice.

## Model

- **NVIDIA NemotronLabs VoiceChat 11B** — weights published by NVIDIA on Hugging Face
  ([nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)).
  The model card states the **OpenMDW 1.1** license
  ([official text](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)). Read the model card
  for the terms and the intended use. mynah-s2s was developed against revision `443794ea`.
- **NVIDIA Nemotron-Nano-9B-v2** — the VoiceChat checkpoint resolves this model's remote code and tokenizer by name
  ([nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)); see its model
  card for its license (revision `6533e8d`).

## Reference inference code (not vendored)

- **NVIDIA NeMo Speech**, branch `nemotron-labs-voicechat` @ `097dfe9`
  ([NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech)), Apache-2.0. mynah-s2s imports it as a dependency and
  drives its streaming VoiceChat pipeline **without modifying upstream files**: the per-frame step logic and the
  function-call protocol are NVIDIA's reference; mynah-s2s adds the execution engine around it (see
  `docs/architecture.md`). Credit for the model, the protocol and the reference implementation belongs to NVIDIA.

## Portions derived from Apache-2.0 sources

The MIT License does **not** apply to the following portions, which are modified copies of Apache-2.0 code and stay
under the Apache License 2.0 (`LICENSES/Apache-2.0.txt`) with their original copyright notices. Each file carries an
SPDX header saying which part is derived and that it was modified.

| file | portion | source (license) |
|---|---|---|
| `v2/src/mynah_v2/model/perf.py` | causal-mask construction in `mamba_host_pos` | NemotronH remote code `modeling_nemotron_h.py` (Nemotron-Nano-9B-v2 @6533e8d), © HuggingFace Inc. team, © NVIDIA CORPORATION (Apache-2.0) |
| `v2/src/mynah_v2/model/perf.py` | `_make_generate_step` (EarTTS MoG refinement step without host syncs) | NeMo Speech `speechlm2/modules/ear_tts_model.py` @097dfe9, © NVIDIA CORPORATION & AFFILIATES (Apache-2.0) |
| `v2/src/mynah_v2/model/codec_async.py` | `spec_to_wav` ops in `make_cached_spec_to_wav` | NeMo Speech `speechlm2/modules/ear_tts_vae_codec.py` @097dfe9, © NVIDIA CORPORATION & AFFILIATES (Apache-2.0) |
| `v2/tests/cpu/test_codec_async.py` | `_upstream_spec_to_wav` (unmodified reference copy for a bit-identity test) | same as above (Apache-2.0) |

Other files reference upstream line numbers in comments to explain what they interact with; they do not copy it.

## Main libraries

PyTorch (BSD-3-Clause), Hugging Face Transformers (Apache-2.0), mamba-ssm (Apache-2.0), causal-conv1d (BSD-3-Clause),
Triton (MIT), and the NeMo dependency set listed in `v2/pyproject.toml`, each under its own license.

## Evaluation tools (used in our experiments, not required to run the engine)

- Omnilingual ASR, `omniASR_CTC_3B_v2` (Meta, Apache-2.0) for spoken-output checks
  ([facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)).
- Qwen3-TTS 0.6B CustomVoice (Apache-2.0) for synthetic test speech
  ([Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)).
