"""E1g: Nano off the frame loop. A worker thread with its own CUDA stream runs the text LLM with a hybrid KV/SSM cache
and STREAMING greedy decode (one forward per token, no HF generate), pushing every reply token with its wall time.

Interface used by the per-frame hook (non-blocking):
  submit(rid, messages)   queue a request (a copy of the history is taken)
  cancel(rid)             stop decoding that request as soon as possible (epoch: its tokens become stale)
  poll() -> [events]      ("tok", rid, id, t_ns) | ("tool", rid, call dict) | ("done", rid, text, new_messages, info)
The worker runs on the DEFAULT CUDA stream (see the note in __init__): kernels of the Thinker and of Nano
interleave on one stream, each thread's own ops stay in order.
Per request: an explicit tool step (call-or-reply question on a copy, 3 tokens), on "call" the call text with the
<TOOLCALL>[ prefix forced, the call executed through the real handlers (V2 ToolRegistry / ToolLedger), then the reply.
Cache: the V2 hybrid-cache fix (mynah_v2.model.llm_cache) applied to the Nano backbone instance; each generation
prefills its full context once (no prefix cache across requests yet: a clone of the hybrid cache plus a multi-token
continuation is not supported by that fix outside its experimental path).
"""

from __future__ import annotations

import json
import queue
import threading
import time

from e1.tool_step_probe import STEP_Q

STOPS = (12, 11, 2)


class QueueWorker(threading.Thread):
    """Thread + request queue + cancel / close, model-free (unit-tested). Subclasses implement _serve()."""

    def __init__(self, name: str = "worker"):
        super().__init__(daemon=True, name=name)
        self.q_in: queue.Queue = queue.Queue()
        self.q_out: queue.Queue = queue.Queue()
        self.cancelled: set = set()
        self.calls: list = []
        self.stop_flag = False
        self.served: list = []

    def submit(self, rid: int, messages: list) -> None:
        self.q_in.put((rid, [dict(m) for m in messages], time.monotonic_ns()))

    def cancel(self, rid) -> None:
        if rid is not None:
            self.cancelled.add(rid)

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.q_out.get_nowait())
            except queue.Empty:
                return out

    def close(self, timeout: float = 30.0) -> None:
        """Stop after the current request and wait for the thread (E1i aborted at interpreter exit with the worker
        still decoding: 'terminate called without an active exception')."""
        self.stop_flag = True
        self.cancelled.update(range(0, 10 ** 6))
        self.q_in.put(None)
        if self.is_alive():
            self.join(timeout)

    def run(self):
        while not self.stop_flag:
            job = self.q_in.get()
            if job is None:
                break
            rid, msgs, t_req = job
            if rid in self.cancelled:                     # superseded before it started: never served
                self.q_out.put(("skipped", rid))
                continue
            self.served.append(rid)
            try:
                self._serve_job(rid, msgs, t_req)
            except Exception as e:  # noqa: BLE001 - a failed request must not kill the worker
                self.q_out.put(("done", rid, "", [], {"error": repr(e)}))

    def _serve_job(self, rid, msgs, t_req):
        self._serve(rid, msgs, t_req)

    # ------------------------------------------------------------------ E1k: KV / SSM reuse across the steps of a turn
    def _prefill(self, ids):
        torch = self.torch
        cache = self.cache_cls(self.llm.config, 1, dtype=self.dtype, device=self.dev, ssm_dtype=torch.float32)
        h = self.llm(inputs_embeds=self.llm.embeddings(torch.tensor([ids], device=self.dev)), cache_params=cache,
                     use_cache=True).last_hidden_state
        return cache, h

    def _continue(self, cache, ids):
        """T known positions on an existing cache in one call (V2's continuation_forward: GEMMs on T rows, the Mamba
        recurrence advanced token by token with the decode kernels, explicit causal mask for attention)."""
        from mynah_v2.model.fc_phase2_prefill import continuation_forward
        return continuation_forward(self.llm, self.llm.embeddings(self.torch.tensor([ids], device=self.dev)), cache,
                                    chunk=16)

    def _decode_from(self, rid, cache, h, max_new, emit, stop_text=None):
        torch = self.torch
        t0 = time.monotonic_ns()
        logits = self.m.lm_head(h[:, -1]).float()
        out, t_first = [], None
        for _ in range(max_new):
            if rid in self.cancelled:
                break
            nxt = int(logits.argmax(-1))
            if nxt in STOPS:
                break
            out.append(nxt)
            if t_first is None:
                t_first = time.monotonic_ns()
            if emit:
                self.q_out.put(("tok", rid, nxt, time.monotonic_ns()))
            if stop_text and stop_text in self.tok.decode(out[-8:]):
                break
            hh = self.llm(inputs_embeds=self.llm.embeddings(torch.tensor([[nxt]], device=self.dev)), cache_params=cache,
                          use_cache=True).last_hidden_state
            logits = self.m.lm_head(hh[:, -1]).float()
        torch.cuda.current_stream().synchronize()
        return out, {"n": len(out), "first_ms": round(((t_first or time.monotonic_ns()) - t0) / 1e6, 1),
                     "ms": round((time.monotonic_ns() - t0) / 1e6, 1)}

    def _branch(self, base, cp, ids):
        """A copy of the shared-prefix cache continued with ids[cp:] -> (cache, hidden). ids[:cp] must equal the
        prefix that built ``base``."""
        import copy
        c = copy.deepcopy(base)
        return c, self._continue(c, ids[cp:])

    def _serve_reuse(self, rid, msgs, t_req):
        info = {"t_req_ns": t_req, "t_start_ns": time.monotonic_ns(), "steps": [], "reuse": True}
        new_msgs = []
        user = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "user" else ""
        if self.backend is not None and user:
            self.backend._said.append(user)
        ctx = msgs if msgs else [{"role": "user", "content": "[call connected]"}]
        reply_ids = self._ids(ctx)
        pre = self.tok.encode("<TOOLCALL>[", add_special_tokens=False)
        dec_ids = self._ids(msgs + [{"role": "user", "content": STEP_Q}]) if (self.tool_step and user) else reply_ids
        cp = 0
        for a, b in zip(dec_ids, reply_ids):
            if a != b:
                break
            cp += 1
        cp = min(cp, len(reply_ids) - 1, len(dec_ids) - 1)       # every branch continues with >= 1 token
        t0 = time.monotonic_ns()
        base, _ = self._prefill(reply_ids[:cp])
        info["shared_prefix"] = cp
        info["prefill_ms"] = round((time.monotonic_ns() - t0) / 1e6, 1)
        if self.tool_step and user:
            c, h = self._branch(base, cp, dec_ids)
            d, st = self._decode_from(rid, c, h, 3, False)
            dtxt = self.tok.decode(d, skip_special_tokens=True)
            info["steps"].append(dict(st, step="decide", out=dtxt, suffix=len(dec_ids) - cp))
            if "TOOL" in dtxt.upper() and rid not in self.cancelled:
                c, h = self._branch(base, cp, reply_ids + pre)
                cl, st = self._decode_from(rid, c, h, 60, False, stop_text="</TOOLCALL>")
                ctext = "<TOOLCALL>[" + self.tok.decode(cl, skip_special_tokens=False)
                if "</TOOLCALL>" in ctext:
                    ctext = ctext[: ctext.index("</TOOLCALL>") + len("</TOOLCALL>")]
                info["steps"].append(dict(st, step="call", out=ctext))
                tc = self.parse(ctext)
                if isinstance(tc, list) and tc:
                    new_msgs.append({"role": "assistant", "content": ctext})
                    for call in tc[:2]:
                        name, args = call.get("name"), call.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:  # noqa: BLE001
                                args = {"_raw": args}
                        t1 = time.monotonic_ns()
                        res = self.bound[name](args) if name in self.bound else json.dumps({"error": f"unknown {name}"})
                        rec = {"name": name, "args": args, "result": res, "executed": name in self.bound,
                               "exec_ms": round((time.monotonic_ns() - t1) / 1e6, 1)}
                        self.calls.append(rec)
                        new_msgs.append({"role": "tool", "content": res})
                        self.q_out.put(("tool", rid, rec))
                    reply_ids = self._ids(msgs + new_msgs)
        c, h = self._branch(base, cp, reply_ids)               # the reply folds the tool result in on the same prefix
        r, st = self._decode_from(rid, c, h, self.max_new, True)
        info["steps"].append(dict(st, step="reply", suffix=len(reply_ids) - cp))
        info["t_done_ns"] = time.monotonic_ns()
        self.q_out.put(("done", rid, self.tok.decode(r, skip_special_tokens=True), new_msgs, info))

    def _serve(self, rid, msgs, t_req):
        if self.reuse:
            return self._serve_reuse(rid, msgs, t_req)  # pragma: no cover - abstract
        raise NotImplementedError


class NanoWorker(QueueWorker):
    def __init__(self, tok, model, system: str, tools: list, bound: dict, backend=None, parse=None,
                 max_new: int = 80, tool_step: bool = True, reuse: bool = False):
        super().__init__(name="nano-worker")
        self.reuse = reuse
        import torch
        from mynah_v2.model.llm_cache import _hybrid_cache_cls, cache_passing_attention, position_injecting_forward
        self.torch = torch
        self.tok, self.m = tok, model
        self.system, self.tools, self.bound, self.backend, self.parse = system, tools, bound, backend, parse
        self.max_new, self.tool_step = max_new, tool_step
        llm = model.backbone
        if not hasattr(llm.forward, "__wrapped_v2__"):
            llm.forward = position_injecting_forward(llm.forward, holder=llm)
        for layer in llm.layers:
            if getattr(layer, "block_type", None) == "attention" and not hasattr(layer.mixer.forward, "__wrapped_v2__"):
                layer.mixer.forward = cache_passing_attention(layer.mixer.forward, llm)
        self.llm = llm
        self.cache_cls = _hybrid_cache_cls(llm)
        p = next(model.parameters())
        self.dtype, self.dev = p.dtype, p.device
        # NO private CUDA stream (E1h): the NemotronH remote code runs every block inside
        # torch.cuda.stream(torch.cuda.default_stream(...)) (modeling_nemotron_h.py:767), so on a side stream the
        # block kernels and our embedding / attention-cache / lm_head ops ran on two streams without ordering; the
        # decode diverged from token 1. On the default stream the cached decode equals the no-cache greedy ids.
        self.stream = None

    def _serve_job(self, rid, msgs, t_req):
        with self.torch.inference_mode():
            self._serve(rid, msgs, t_req)

    def _ids(self, msgs, extra=None):
        ids = self.tok.apply_chat_template([{"role": "system", "content": self.system}] + msgs, tools=self.tools,
                                           add_generation_prompt=True)
        return list(ids) + list(extra or [])

    def _greedy(self, rid, ids, max_new, emit, stop_text=None, ignore_stops=False, cache_info=False):
        torch = self.torch
        cache = self.cache_cls(self.llm.config, 1, dtype=self.dtype, device=self.dev, ssm_dtype=torch.float32)
        emb = self.llm.embeddings(torch.tensor([ids], device=self.dev))
        t0 = time.monotonic_ns()
        h = self.llm(inputs_embeds=emb, cache_params=cache, use_cache=True).last_hidden_state
        logits = self.m.lm_head(h[:, -1]).float()
        out, t_first = [], None
        for _ in range(max_new):
            if rid in self.cancelled:
                break
            nxt = int(logits.argmax(-1))
            if nxt in STOPS and not ignore_stops:
                break
            out.append(nxt)
            if t_first is None:
                torch.cuda.current_stream().synchronize()
                t_first = time.monotonic_ns()
            if emit:
                self.q_out.put(("tok", rid, nxt, time.monotonic_ns()))
            if stop_text and stop_text in self.tok.decode(out[-8:]):
                break
            e = self.llm.embeddings(torch.tensor([[nxt]], device=self.dev))
            h = self.llm(inputs_embeds=e, cache_params=cache, use_cache=True).last_hidden_state
            logits = self.m.lm_head(h[:, -1]).float()
        torch.cuda.current_stream().synchronize()
        t_end = time.monotonic_ns()
        info = {"ctx": len(ids), "n": len(out), "first_ms": round(((t_first or t_end) - t0) / 1e6, 1),
                "ms": round((t_end - t0) / 1e6, 1)}
        if cache_info:
            keys = [k for k in getattr(cache, "key_cache", []) if isinstance(k, torch.Tensor) and k.numel()]
            convs = [c for c in getattr(cache, "conv_states", []) if isinstance(c, torch.Tensor) and c.numel()]
            info.update(cache_pos=int(getattr(cache, "_v2_pos", -1)), n_key=len(keys), n_conv=len(convs),
                        key_shape=list(keys[0].shape) if keys else None,
                        key_dev=str(keys[0].device) if keys else None, key_dtype=str(keys[0].dtype) if keys else None,
                        conv_dtype=str(convs[0].dtype) if convs else None, cache_cls=type(cache).__name__,
                        stream=str(torch.cuda.current_stream()))
        return out, info

    # ------------------------------------------------------------------ E1k: KV / SSM reuse across the steps of a turn
    def _prefill(self, ids):
        torch = self.torch
        cache = self.cache_cls(self.llm.config, 1, dtype=self.dtype, device=self.dev, ssm_dtype=torch.float32)
        h = self.llm(inputs_embeds=self.llm.embeddings(torch.tensor([ids], device=self.dev)), cache_params=cache,
                     use_cache=True).last_hidden_state
        return cache, h

    def _continue(self, cache, ids):
        """T known positions on an existing cache in one call (V2's continuation_forward: GEMMs on T rows, the Mamba
        recurrence advanced token by token with the decode kernels, explicit causal mask for attention)."""
        from mynah_v2.model.fc_phase2_prefill import continuation_forward
        return continuation_forward(self.llm, self.llm.embeddings(self.torch.tensor([ids], device=self.dev)), cache,
                                    chunk=16)

    def _decode_from(self, rid, cache, h, max_new, emit, stop_text=None):
        torch = self.torch
        t0 = time.monotonic_ns()
        logits = self.m.lm_head(h[:, -1]).float()
        out, t_first = [], None
        for _ in range(max_new):
            if rid in self.cancelled:
                break
            nxt = int(logits.argmax(-1))
            if nxt in STOPS:
                break
            out.append(nxt)
            if t_first is None:
                t_first = time.monotonic_ns()
            if emit:
                self.q_out.put(("tok", rid, nxt, time.monotonic_ns()))
            if stop_text and stop_text in self.tok.decode(out[-8:]):
                break
            hh = self.llm(inputs_embeds=self.llm.embeddings(torch.tensor([[nxt]], device=self.dev)), cache_params=cache,
                          use_cache=True).last_hidden_state
            logits = self.m.lm_head(hh[:, -1]).float()
        torch.cuda.current_stream().synchronize()
        return out, {"n": len(out), "first_ms": round(((t_first or time.monotonic_ns()) - t0) / 1e6, 1),
                     "ms": round((time.monotonic_ns() - t0) / 1e6, 1)}

    def _branch(self, base, cp, ids):
        """A copy of the shared-prefix cache continued with ids[cp:] -> (cache, hidden). ids[:cp] must equal the
        prefix that built ``base``."""
        import copy
        c = copy.deepcopy(base)
        return c, self._continue(c, ids[cp:])

    def _serve_reuse(self, rid, msgs, t_req):
        info = {"t_req_ns": t_req, "t_start_ns": time.monotonic_ns(), "steps": [], "reuse": True}
        new_msgs = []
        user = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "user" else ""
        if self.backend is not None and user:
            self.backend._said.append(user)
        ctx = msgs if msgs else [{"role": "user", "content": "[call connected]"}]
        reply_ids = self._ids(ctx)
        pre = self.tok.encode("<TOOLCALL>[", add_special_tokens=False)
        dec_ids = self._ids(msgs + [{"role": "user", "content": STEP_Q}]) if (self.tool_step and user) else reply_ids
        cp = 0
        for a, b in zip(dec_ids, reply_ids):
            if a != b:
                break
            cp += 1
        cp = min(cp, len(reply_ids) - 1, len(dec_ids) - 1)       # every branch continues with >= 1 token
        t0 = time.monotonic_ns()
        base, _ = self._prefill(reply_ids[:cp])
        info["shared_prefix"] = cp
        info["prefill_ms"] = round((time.monotonic_ns() - t0) / 1e6, 1)
        if self.tool_step and user:
            c, h = self._branch(base, cp, dec_ids)
            d, st = self._decode_from(rid, c, h, 3, False)
            dtxt = self.tok.decode(d, skip_special_tokens=True)
            info["steps"].append(dict(st, step="decide", out=dtxt, suffix=len(dec_ids) - cp))
            if "TOOL" in dtxt.upper() and rid not in self.cancelled:
                c, h = self._branch(base, cp, reply_ids + pre)
                cl, st = self._decode_from(rid, c, h, 60, False, stop_text="</TOOLCALL>")
                ctext = "<TOOLCALL>[" + self.tok.decode(cl, skip_special_tokens=False)
                if "</TOOLCALL>" in ctext:
                    ctext = ctext[: ctext.index("</TOOLCALL>") + len("</TOOLCALL>")]
                info["steps"].append(dict(st, step="call", out=ctext))
                tc = self.parse(ctext)
                if isinstance(tc, list) and tc:
                    new_msgs.append({"role": "assistant", "content": ctext})
                    for call in tc[:2]:
                        name, args = call.get("name"), call.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:  # noqa: BLE001
                                args = {"_raw": args}
                        t1 = time.monotonic_ns()
                        res = self.bound[name](args) if name in self.bound else json.dumps({"error": f"unknown {name}"})
                        rec = {"name": name, "args": args, "result": res, "executed": name in self.bound,
                               "exec_ms": round((time.monotonic_ns() - t1) / 1e6, 1)}
                        self.calls.append(rec)
                        new_msgs.append({"role": "tool", "content": res})
                        self.q_out.put(("tool", rid, rec))
                    reply_ids = self._ids(msgs + new_msgs)
        c, h = self._branch(base, cp, reply_ids)               # the reply folds the tool result in on the same prefix
        r, st = self._decode_from(rid, c, h, self.max_new, True)
        info["steps"].append(dict(st, step="reply", suffix=len(reply_ids) - cp))
        info["t_done_ns"] = time.monotonic_ns()
        self.q_out.put(("done", rid, self.tok.decode(r, skip_special_tokens=True), new_msgs, info))

    def _serve(self, rid, msgs, t_req):
        if self.reuse:
            return self._serve_reuse(rid, msgs, t_req)
        info = {"t_req_ns": t_req, "t_start_ns": time.monotonic_ns(), "steps": []}
        new_msgs = []
        user = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "user" else ""
        if self.backend is not None and user:
            self.backend._said.append(user)
        ctx = msgs if msgs else [{"role": "user", "content": "[call connected]"}]
        if self.tool_step and user:
            d, st = self._greedy(rid, self._ids(msgs + [{"role": "user", "content": STEP_Q}]), 3, False)
            dtxt = self.tok.decode(d, skip_special_tokens=True)
            info["steps"].append(dict(st, step="decide", out=dtxt))
            if "TOOL" in dtxt.upper() and rid not in self.cancelled:
                pre = self.tok.encode("<TOOLCALL>[", add_special_tokens=False)
                c, st = self._greedy(rid, self._ids(msgs, pre), 60, False, stop_text="</TOOLCALL>")
                ctext = "<TOOLCALL>[" + self.tok.decode(c, skip_special_tokens=False)
                if "</TOOLCALL>" in ctext:
                    ctext = ctext[: ctext.index("</TOOLCALL>") + len("</TOOLCALL>")]
                info["steps"].append(dict(st, step="call", out=ctext))
                tc = self.parse(ctext)
                if isinstance(tc, list) and tc:
                    new_msgs.append({"role": "assistant", "content": ctext})
                    for call in tc[:2]:
                        name, args = call.get("name"), call.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:  # noqa: BLE001
                                args = {"_raw": args}
                        t1 = time.monotonic_ns()
                        res = self.bound[name](args) if name in self.bound else json.dumps({"error": f"unknown {name}"})
                        rec = {"name": name, "args": args, "result": res, "executed": name in self.bound,
                               "exec_ms": round((time.monotonic_ns() - t1) / 1e6, 1)}
                        self.calls.append(rec)
                        new_msgs.append({"role": "tool", "content": res})
                        self.q_out.put(("tool", rid, rec))
                    ctx = msgs + new_msgs
        r, st = self._greedy(rid, self._ids(ctx), self.max_new, True)
        info["steps"].append(dict(st, step="reply"))
        info["t_done_ns"] = time.monotonic_ns()
        self.q_out.put(("done", rid, self.tok.decode(r, skip_special_tokens=True), new_msgs, info))
