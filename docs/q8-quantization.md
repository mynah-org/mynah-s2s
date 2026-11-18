# Quantization and fusion notes

This is the technical note for the optional INT4/Q8 paths. The default server
remains BF16 unless a quantization flag, environment variable, or offline pack
is supplied.

## Profiles

`--int4` covers the measured standard Linear paths, dense pointwise codec
projections, selected MoG, and RVQ gather/sum. Temporal codec kernels, the RNNT
decoder, and RVQ argmin/update remain opt-in experiments because their numerical
contracts are more sensitive.

`--int8` covers the same standard scopes with Q8 batched GEMV plus selected MoG,
RVQ gather, codec depthwise and ConvTranspose paths, and the RNNT prediction
LSTM. The Q8 RNNT gate/activation path is fused by default; set
`MYNAH_INT8_RNNT_FUSED=0` to use the fallback path. The Q8 RVQ encoder remains
opt-in with `MYNAH_INT8_RVQ_ENCODE=1` because changing the argmin can change
code selection.

## Offline Q8 bundle

Export once, then load the sibling manifest at startup:

```bash
make w8-pack W8_PACK=/path/to/voicechat-w8.safetensors
make webui-gpu V2_SERVE_ARGS="--int8-pack /path/to/voicechat-w8.safetensors"
```

The bundle serializes the standard Q8 modules and the custom paths that cannot
use an ordinary Linear replacement:

- selected MoG projection and low-rank projection;
- RVQ embedding gather;
- fused RNNT LSTM gates;
- codec depthwise filters;
- grouped/temporal codec ConvTranspose layers.

The remaining unsupported scopes are embeddings and state/cache tensors, which
stay in the original checkpoint. The loader validates the complete manifest
before replacing modules and avoids requantizing the custom paths at startup.

## Validation snapshot

The current custom bundle has the following verified results:

- 629 standard modules + 15 custom modules exported;
- t02 full tool-call gate: `get_order_status(ABC456)`, grounding violations `0`;
- t02 Q8 offline run: RTF `1.643`, p50 step `165.8 ms`;
- GPU specialized tests: `5 passed`, including dense/grouped ConvTranspose,
  RNNT fused B=4, MoG, RVQ, pointwise and depthwise paths.

For quality experiments, the dynamic path also supports a protected W8A16
profile. Run with `--int8 --int8-protected`; it keeps the Thinker text and
function heads in BF16 while retaining the Q8 weight-only paths elsewhere.
This is intentionally not supported by the existing offline bundle yet, and
does not change the default Q8 behavior.

The offline bundle is also wired into the real multi-session runner. Use one
switch; the runner loads one model, opens all sessions, and batches Thinker,
perception, codec, and EarTTS work where possible:

```bash
python v2/scripts/rendezvous_pair.py \
  --group p3_overlapping_tool_calls \
  --audio-root /path/to/agent-audio \
  --out /tmp/q8-pair \
  --int8-pack /path/to/voicechat-w8.safetensors
```

The concurrent gate has passed with two sessions and overlapping tool calls:
both sessions produced the expected tool calls and grounded answers with no
cross-session leak. The path is functionally correct, but the current
multi-session profile is still above a strict 160 ms round budget; the gate
records batch histograms, FC queue time, graph capture cost, memory, and
per-session latency in `pair_gate.json` for further optimization.

## CUDA and Triton compatibility

The Q8 custom kernels use ordinary Triton pointer arithmetic, BF16 inputs, and
FP32 accumulation. They do not contain inline PTX or device-specific matrix
instructions, so the source path is intended for CUDA GPU families supported by
the installed Triton/PyTorch pair, including Ampere-class, Ada/L40S, Hopper,
and newer architectures. The standard Q8 GEMV implementation follows the same
SM80+ portability boundary.

`setup_cuda.sh` detects the active compute capability and builds the
architecture-sensitive `mamba-ssm` and `causal-conv1d` extensions for that
device. The Q8 Triton kernels themselves are JIT-compiled for the active GPU.
Performance is not portable by definition: autotuning and the preferred BF16
or INT4 backend can change between families. Run the GPU specialized tests on a
new family before treating its latency as supported; a successful import alone
is not a performance result.

For comparison with the external GGUF layout, see the
[reference project](https://github.com/mkotyushev/nemotron-voicechat-realtime-gguf#why-gguf).
