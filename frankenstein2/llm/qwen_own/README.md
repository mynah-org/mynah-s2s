# qwen_own — our own inference for Qwen3.5 (dense) and Qwen3.5/3.6 MoE

Runtime dependencies: torch, triton (optional), safetensors, tokenizers and jinja2 (the chat template). config.json,
the tokenizer and the chat template are read by our own code (`config.py`, `tokenizer.py`); HF transformers and vLLM
are used only as test references (the tokenizer/chat template give identical ids to HF's on the three checkpoints).
No serving framework. Triton kernels are
optional (plain-torch fallbacks everywhere); tested on GB10 (sm_121), written for sm_80+ / sm_89 too.

## Models and weight formats

| Model | Formats | Notes |
|---|---|---|
| Qwen3.5-4B (dense) | bf16; weight-only int8 (per channel, Triton GEMV); int4 RTN (not recommended: quality drops) | int8 = bf16 greedy on our checks, faster decode |
| Qwen3.6-35B-A3B (MoE, 256 experts, top-8 + shared) | AWQ int4 (AutoAWQ GEMM layout, the sm_80/89 path); NVFP4 W4A16 (ModelOpt: e2m1 + e4m3/16 + fp32 global, FP8 per-tensor projections; the Blackwell path) | bf16 experts are refused by the loader (compressed only); bf16 MTP experts are quantised to NVFP4 at load |

Architecture covered: gated DeltaNet linear attention (chunked delta rule for prefill, recurrent step for decode,
fp32 state, causal conv window), gated full attention (q/k RMSNorm(1+w), partial RoPE, output gate, GQA via SDPA),
dense MLP or MoE (fp32 softmax router, renormalised top-k, shared expert with a sigmoid gate), MTP head.

## API

### Engine: `QwenOwn`

```python
from frankenstein2.llm.qwen_own import QwenOwn, BatchRunner
eng = QwenOwn(path, device="cuda", mode="bf16"|"int8"|"int4", slots=16, max_len=8192, graphs=True, mtp=False,
              tokenizer=None, stream=None)  # tokenizer: default QwenTokenizer (tokenizers + jinja2, our code);
                                            # stream: torch.cuda.Stream for all of this engine's GPU work
s = eng.open()                       # a session slot: its own KV cache + DeltaNet state (no shared state)
ids = eng.chat_ids(messages, tools)  # chat template, thinking off
logits = eng.prefill(s, ids)         # whole prompt; continues from the session's previous prompt when it extends it
eng.prefill_begin(s, ids); eng.prefill_step(s, budget)   # chunked prefill (returns logits when complete)
logits = eng.decode_batch([s1, s2, ...], [tok1, tok2, ...])   # one token for N sessions, CUDA graph per bucket
for ev in eng.generate(s, ids, max_new=192, stop=("</tool_call>",)): ...      # single-session greedy stream
for ev in eng.generate_mtp(s, ids, ...): ...       # single-session speculative decoding with the MTP head (mtp=True)
eng.close(s); eng.footprint()
```

- `slots` is the maximum number of concurrent sessions; one extra scratch slot pads CUDA-graph batches.
- Graph buckets: batch 1/2/4/8/16 × cache length 512…32768; captured lazily on first use (a short one-time stall).
- Per session memory: DeltaNet state (4B: 49 MiB, MoE: 61 MiB) + KV 32 KiB/token (4B) or 20 KiB/token (MoE).

### Continuous batching: `BatchRunner`

```python
run = BatchRunner(eng, prefill_budget=256, stop=("</tool_call>",))
run.submit(key, ids, max_new=192)    # add a session on the fly (new key = new slot; known key = next turn)
while run.busy():
    for ev in run.step():            # {"key", "text"} deltas ... {"key", "done", "ids", "reason", "first_token_s", ...}
        ...
run.cancel(key)                      # stops that request at the next step
run.close(key)                       # frees the slot
```

Each `step()` gives every session still reading its prompt one prefill chunk (≤ `prefill_budget` tokens), then runs
one batched decode token for all generating sessions. A session stops on end of turn, on its stop strings (the tool
call is complete: run it and submit the next turn), on `max_new` or on cancel. Output per session is identical to
single-session greedy decoding (CPU test `test_batch_runner_matches_single_generation`).

### In-process use

Library only: no process, no global state. Dependencies: torch, safetensors, triton (optional), tokenizers, jinja2. `BatchRunner.step()` does a
bounded amount of work (one prefill chunk per prefilling session + one batched decode token) and returns, so an
external scheduler can interleave it with the ears and the talker; give the engine its own `stream=` to overlap.

### Process: `server.py`

JSON lines, same protocol as `../qwen_server.py` (`gen` / `cancel` / `quit` → `delta` / `end`); one slot per `ss`.

## Validation status (GB10)

- 4B bf16 = HF transformers: top-1 6/6, greedy tokens identical; tool suite = vLLM (25/26 identical calls).
- 4B int8: greedy = bf16 on 6/6 prompts; tool suite 24/26 calls identical to bf16.
- MoE AWQ: tool suite calls identical to vLLM AWQ on 26/26 items.
- MoE AWQ after all kernel changes: tool calls identical to vLLM AWQ on 26/26 suite items.
- MoE NVFP4 + MTP: speculative output identical to plain greedy (6/6); tool calls identical to vLLM NVFP4 on 21/26
  (our FP8 projections run as W8A16, vLLM quantises activations too, so small numeric drift is expected).
- 4B: re-checked equal to HF transformers greedy (6/6) after the fused DeltaNet kernels.
- CPU tests: `frankenstein2/tests/test_qwen_own.py` (tiny random models).
