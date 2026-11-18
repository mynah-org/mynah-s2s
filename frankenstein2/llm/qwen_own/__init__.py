"""Own torch inference for Qwen3.5 (dense, hybrid gated DeltaNet + gated attention) and Qwen3.5/3.6 MoE (AWQ 4-bit
experts): `QwenOwn` (engine.py). No serving framework; torch + the checkpoint's HF tokenizer. See engine.py."""

from .batch import BatchRunner
from .engine import QwenOwn

__all__ = ["QwenOwn", "BatchRunner"]
