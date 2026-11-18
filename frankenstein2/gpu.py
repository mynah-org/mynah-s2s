"""GPU -> host waits that do not hold the GIL.

The one-process engine runs the scheduler (ears + talker) and the LLM on two Python threads. A blocking read of a
CUDA result (``.tolist()``, ``.item()``, ``bool(t)``) waits for the GPU inside the C++ call with the GIL held, so the
other thread cannot run a single Python line until that wait ends: with the LLM decoding continuously next to the
ears, the ears' hop slowed down by orders of magnitude (lab/ears_contention.py). ``host()`` first waits on a CUDA
event with short sleeps (the GIL is released while sleeping), then reads the already-finished result.
"""

from __future__ import annotations

import os
import time

POLITE = os.environ.get("F2_POLITE_SYNC", "1") != "0"


def ready(stream=None) -> None:
    """Return once the work queued so far on ``stream`` (default: the current stream) has finished."""
    if not POLITE:
        return
    import torch
    if not torch.cuda.is_available():
        return
    ev = torch.cuda.Event()
    ev.record(stream if stream is not None else torch.cuda.current_stream())
    while not ev.query():
        time.sleep(0.0002)


def host(t):
    """``t.tolist()`` without holding the GIL while the GPU computes it."""
    if getattr(t, "is_cuda", False):
        ready()
    return t.tolist()
