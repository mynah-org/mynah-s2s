"""pocket_swap: several Pocket TTS languages on one GPU from one base checkpoint plus per-language packages.

Pure-python bookkeeping (lru, core) imports without torch; package/swapper need torch (and pocket_tts for the
swapper). Opt-in: nothing else in the repository imports this package.
"""

from .core import Binding, SwapCore  # noqa: F401
from .lru import AllSlotsBusy, CacheFull, RefLRU, SlotPool  # noqa: F401

__all__ = ["Binding", "SwapCore", "AllSlotsBusy", "CacheFull", "RefLRU", "SlotPool"]
