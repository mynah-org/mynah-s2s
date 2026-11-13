"""CUDA graph for the full-buffer (no perception cache) perception call.

The exact path re-encodes the whole 71-frame (90 880 samples) sliding window every chunk: preprocessor (mel, dither only
in training) → FastConformer → modality adapter → proj, FP32. NVIDIA graphs perception only in its cache mode
(``use_perception_cudagraph`` requires ``use_perception_cache``), which changes inputs and is NOT exact. Here the same
module runs on the same input with the same kernels; only the launches are replayed.

Scope, by construction:
* graphed only when the call is exactly ``forward(input_signal=[1, full_len] CUDA, input_signal_length=[1] CUDA,
  return_encoder_emb=bool)``; every other call (buffer still filling at session start, other shapes, CPU, grad
  enabled, extra arguments) runs eagerly through the original forward;
* stateless: the module keeps no per-session state (no cache mode), so sessions need no reset;
* the length is COPIED into the graph input (no host read); NeMo's own host check in ``normalize_batch`` is skipped
  while capturing (``is_current_stream_capturing``), it runs in the eager warm-up;
* one capture per (shape, dtype, return_encoder_emb); the warm-up eager call is the result of the capturing call;
* outputs are cloned before returning (the next replay overwrites the static outputs); a lock + an event serialize
  copy-in → replay → clone across threads and streams.
"""

from __future__ import annotations

import threading

import torch

FRAME_SAMPLES = 1280            # 80 ms at 16 kHz (wrapper FRAME_SIZE_SAMPLES)
DEFAULT_BUFFER_FRAMES = 71      # wrapper DEFAULT_BUFFER_SIZE_FRAMES


class PerceptionGraph:
    def __init__(self, perception: torch.nn.Module, full_len: int):
        if "forward" in perception.__dict__:
            raise RuntimeError("perception_graph: forward already patched")
        self.mod = perception
        self.full_len = int(full_len)
        self.eager = perception.forward
        self.graphs: dict = {}
        self.stats = {"captures": 0, "replays": 0, "eager": 0}
        self._lock = threading.Lock()
        self._last: torch.cuda.Event | None = None
        perception.forward = self.forward

    def remove(self) -> None:
        if self.mod.__dict__.get("forward") == self.forward:
            del self.mod.forward
        self.graphs.clear()

    def _graphable(self, args, kw) -> bool:
        if args or set(kw) - {"input_signal", "input_signal_length", "return_encoder_emb"}:
            return False
        sig, ln = kw.get("input_signal"), kw.get("input_signal_length")
        return (isinstance(sig, torch.Tensor) and isinstance(ln, torch.Tensor) and sig.is_cuda and ln.is_cuda
                and sig.dim() == 2 and sig.shape[0] == 1 and sig.shape[1] == self.full_len and tuple(ln.shape) == (1,)
                and not torch.is_grad_enabled() and not torch.cuda.is_current_stream_capturing())

    def forward(self, *args, **kw):
        if not self._graphable(args, kw):
            self.stats["eager"] += 1
            return self.eager(*args, **kw)
        sig, ln = kw["input_signal"], kw["input_signal_length"]
        rte = bool(kw.get("return_encoder_emb", False))
        key = (tuple(sig.shape), sig.dtype, ln.dtype, rte)
        with self._lock:
            cur = torch.cuda.current_stream()
            if self._last is not None:
                cur.wait_event(self._last)               # previous user's clone done before we overwrite the inputs
            g = self.graphs.get(key)
            if g is None:
                static_sig, static_ln = sig.clone(), ln.clone()
                s = torch.cuda.Stream()
                s.wait_stream(cur)
                with torch.cuda.stream(s):               # warm-up (allocations, cudnn autotune, pos-enc resize)
                    out = self.eager(input_signal=static_sig, input_signal_length=static_ln, return_encoder_emb=rte)
                    out = tuple(t.clone() for t in out)
                cur.wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    static_out = self.eager(input_signal=static_sig, input_signal_length=static_ln,
                                            return_encoder_emb=rte)
                self.graphs[key] = (graph, static_sig, static_ln, static_out)
                self.stats["captures"] += 1
            else:
                graph, static_sig, static_ln, static_out = g
                static_sig.copy_(sig)
                static_ln.copy_(ln)
                graph.replay()
                out = tuple(t.clone() for t in static_out)
                self.stats["replays"] += 1
            ev = torch.cuda.Event()
            ev.record(cur)
            self._last = ev
            return out
