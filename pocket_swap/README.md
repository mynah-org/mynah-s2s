# pocket_swap

Several Pocket TTS languages on one GPU from **one shared base checkpoint** plus per-language packages. It is a
standalone, opt-in package: nothing else in this repository imports it.

- **Base**: the English 24-layer checkpoint (`english_2026-09_24l`). Its backbone stays resident in bf16 and is
  never written. Mimi (codec, including the voice-cloning encoder) is identical in every language and is loaded once.
- **Language package**: tokenizer + text embedding + non-backbone tensors (verbatim: the vocabularies differ, so
  they cannot be diffed) + the backbone as a delta against the base, stored as `exact`, `bf16`, `int8` or `int4`
  (per-group scales, optional zstd). See `pocket_swap/package.py` for the layout and manifest.
- **Two runtime modes** (per deployment):
  - `ondemand` (mode b, **the default**): a slot holds only its compressed term; every forward rebuilds the weight
    tile inside a fused Triton GEMV (plain-torch fallback on any architecture). Terms:
    - a delta (`bf16`/`int8`/`int4` packages) on the shared base, kept as `base_precision` `bf16`, `int8` or `int4`;
    - or weight-only (`w8`/`w4`/`wfp8` packages): each language's own quantised weights, no base backbone at all.
  - `materialise` (mode a): `W_lang = W_base + dequant(Δ)` is written once into a per-language slot buffer. Per-step
    cost is the same as a plain model; memory is one full bf16 backbone per resident language.
- **Tiers**: VRAM slots (LRU, ref-counted, never evicted while a session holds them; fixed addresses while allocated,
  so CUDA graphs survive language swaps) → pinned host RAM → pageable host RAM → disk. Each tier has byte / count /
  idle-time caps. `pocket_swap/planner.py` sizes them for a given GPU, host and resident stack.
- **Sessions**: each session/call id is bound to one language. A switch, load or eviction never touches a slot that
  another session holds. Voice states are per (language, voice) templates and generation always works on a copy.
  `SwapCore.group_by_slot()` groups a mixed-language batch per slot.

## Engine API

```python
from pocket_swap.config import preset, auto
cfg = preset("l40s-16g")                 # presets: "gb10", "l40s-16g", "l4-24g"; or auto(stack_vram=..., stack_host=...)
sw = cfg.build("en_24l_cloning.safetensors", "english_2026-09_24l", "packages/w8")   # precision read from manifests

sw.open("call-1", "italian_24l", voice="reference.wav")    # bind; raises AdmissionRefused / AllSlotsBusy (timeout=0)
for pcm in sw.stream("call-1", "Buongiorno, come posso aiutarla?", stop=stop_event):
    play(pcm)                                                # float32 mono chunks at sw.sample_rate
sw.switch("call-1", "french_24l")                            # only this session moves; its voice is kept
groups = sw.groups(["call-1", "call-2", "call-3"])           # {slot: [sessions]}: batch per group (one language)
sw.close("call-1"); sw.stats(); sw.mem(); sw.maintenance()   # maintenance(): idle eviction, call from a timer
```

- Each session/call id binds one language and one voice; a switch, load or eviction never touches a slot that
  another session holds (ref-counted). Voice states are per (language, voice) templates; generation works on a copy.
- Admission control: a new slot is allocated only if it fits the slot budget and leaves `vram_reserve` free;
  otherwise an unused slot is reused, or the request waits (`timeout`) / is refused (`AdmissionRefused`).
- Loading streams tensors from safetensors straight to the device (no full checkpoint in host RAM) and builds
  slots on the meta device (no host copy, no fp32 transient).

Build and verify packages (nothing is uploaded):

```
python -m pocket_swap.cli build --base EN.safetensors --base-id REPO@REV:FILE --target IT.safetensors \
    --target-id REPO@REV:FILE --lang italian_24l --tokenizer tokenizer.json --config italian_24l.yaml \
    --precision int8 --out pocket-tts-deltas/
python -m pocket_swap.cli build --base EN.safetensors --base-id REPO@REV:FILE --base-language \
    --lang english_2026-09_24l --tokenizer en_tokenizer.json --config english_2026-09_24l.yaml --out pocket-tts-deltas/
python -m pocket_swap.cli verify --base EN.safetensors --package pocket-tts-deltas/italian_24l --target IT.safetensors
python -m pocket_swap.cli card --root pocket-tts-deltas
python -m pocket_swap.planner --preset l40s-g6e.xlarge --languages 3
```

Tests: `pytest tests/test_core.py tests/test_planner.py` (pure python), `pytest tests/test_package.py` (torch, CPU).
`bench/gpu_bench.py` is the GPU check (timings, footprint, latency, isolation, CUDA graphs, audio for an ASR check).

## Licence of the weights

Pocket TTS weights are published by Kyutai under **CC-BY-4.0**. Packages built by this tool are adaptations of those
weights. They carry the attribution and change notice from `package.LICENCE_NOTICE` in every manifest and in the
generated model card. Use stays subject to Kyutai's terms on the model pages, including the listed prohibited uses
(for example, voice cloning without explicit and lawful consent). This code is MIT like the rest of the repository.
This is not legal advice.
