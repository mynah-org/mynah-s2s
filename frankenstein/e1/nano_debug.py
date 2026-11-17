"""E1h step 1: is the Nano worker's incremental decode exact? One fixed prompt, first N greedy ids:

  ref_generate   HF generate (greedy), BEFORE the worker patches the backbone
  ref_nocache    full recompute each step (no cache at all): the ground truth of the arithmetic path we use
  inc_default    the worker's cached decode, main thread, default stream
  inc_stream     the same in a thread on the worker's own CUDA stream
  inc_concurrent inc_stream while the main thread steps the duplex Thinker (its own cache), as in the session
Each variant logs its ids, the first divergence from ref_nocache, and the cache's position / devices / dtypes.
"""

from __future__ import annotations

import threading
import time


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_debug(tok, m, eng, system, tools, n: int = 20) -> dict:  # pragma: no cover - GPU
    import torch
    from e1.nano_worker import NanoWorker
    msgs = [{"role": "user", "content": "Buongiorno, cerco un divano a tre posti. Cosa avete?"}]
    ids = tok.apply_chat_template([{"role": "system", "content": system}] + msgs, tools=tools,
                                  add_generation_prompt=True)
    dev = next(m.parameters()).device
    res = {"ctx": len(ids)}
    with torch.inference_mode():
        g = m.generate(torch.tensor([ids], device=dev), max_new_tokens=n, min_new_tokens=n, do_sample=False,
                       pad_token_id=0)
        res["ref_generate"] = g[0, len(ids):].tolist()
        cur = list(ids)
        out = []
        for _ in range(n):
            h = m.backbone(input_ids=torch.tensor([cur], device=dev)).last_hidden_state
            nxt = int(m.lm_head(h[:, -1]).float().argmax(-1))
            out.append(nxt)
            cur.append(nxt)
        res["ref_nocache"] = out
    wk = NanoWorker(tok, m, system, tools, bound={}, backend=None, parse=None, max_new=n, tool_step=False)

    def inc():
        ids_out, st = wk._greedy(-1, ids, n, False, stop_text=None, ignore_stops=True, cache_info=True)
        return ids_out, st

    with torch.inference_mode():
        r, st = inc()
    res["inc_default"], res["inc_default_info"] = r, st
    box = {}

    def th():
        box["r"] = _in_stream(wk, inc)
    t = threading.Thread(target=th)
    t.start()
    t.join()
    res["inc_stream"], res["inc_stream_info"] = box["r"]
    # concurrent: the duplex Thinker steps in the main thread meanwhile (own V2 cache, random inputs)
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    stt = eng.wrapper.model.stt_model
    p = next(stt.llm.parameters())
    tcache = _hybrid_cache_cls(stt.llm)(stt.llm.config, 1, dtype=p.dtype, device=p.device, ssm_dtype=torch.float32)
    box2 = {}
    t = threading.Thread(target=lambda: box2.__setitem__("r", _in_stream(wk, inc)))
    steps = 0
    with torch.inference_mode():
        hdim = stt.embed_tokens.weight.shape[1]
        stt(torch.randn(1, 8, hdim, device=p.device, dtype=p.dtype) * 0.01, cache=tcache)
        t.start()
        t0 = time.monotonic()
        while t.is_alive() and time.monotonic() - t0 < 60:
            stt(torch.randn(1, 1, stt.embed_tokens.weight.shape[1], device=p.device, dtype=p.dtype) * 0.01,
                cache=tcache)
            steps += 1
        t.join()
    res["inc_concurrent"], res["inc_concurrent_info"] = box2["r"]
    res["thinker_steps_during"] = steps
    ref = res["ref_nocache"]
    for k in ("ref_generate", "inc_default", "inc_stream", "inc_concurrent"):
        res[f"div_{k}"] = first_div(res[k], ref)
    res["text"] = {k: tok.decode(res[k]) for k in ("ref_generate", "ref_nocache", "inc_default", "inc_stream",
                                                   "inc_concurrent")}
    res["exact"] = all(res[f"div_{k}"] is None for k in ("inc_default", "inc_stream", "inc_concurrent"))
    return res, wk


def _in_stream(wk, fn):
    import contextlib
    import torch
    ctx = torch.cuda.stream(wk.stream) if wk.stream is not None else contextlib.nullcontext()
    with ctx, torch.inference_mode():
        return fn()
