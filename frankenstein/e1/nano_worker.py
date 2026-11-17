"""E1g: Nano off the frame loop. A worker thread with its own CUDA stream runs the text LLM with a hybrid KV/SSM cache
and STREAMING greedy decode (one forward per token, no HF generate), pushing every reply token with its wall time.

Interface used by the per-frame hook (non-blocking):
  submit(rid, messages)   queue a request (a copy of the history is taken)
  cancel(rid)             stop decoding that request as soon as possible (epoch: its tokens become stale)
  poll() -> [events]      ("tok", rid, id, t_ns) | ("tool", rid, call dict) | ("done", rid, text, new_messages, info)
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


class NanoWorker(threading.Thread):
    def __init__(self, tok, model, system: str, tools: list, bound: dict, backend=None, parse=None,
                 max_new: int = 80, tool_step: bool = True):
        super().__init__(daemon=True, name="nano-worker")
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
        self.q_in: queue.Queue = queue.Queue()
        self.q_out: queue.Queue = queue.Queue()
        self.cancelled: set = set()
        self.calls: list = []
        self.stream = torch.cuda.Stream()
        self.stop_flag = False

    # ------------------------------------------------------------------ API (frame-loop thread)
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

    def close(self) -> None:
        self.stop_flag = True
        self.q_in.put(None)

    # ------------------------------------------------------------------ worker
    def run(self):
        torch = self.torch
        while not self.stop_flag:
            job = self.q_in.get()
            if job is None:
                break
            rid, msgs, t_req = job
            try:
                with torch.cuda.stream(self.stream), torch.inference_mode():
                    self._serve(rid, msgs, t_req)
            except Exception as e:  # noqa: BLE001 - a failed request must not kill the worker
                self.q_out.put(("done", rid, "", [], {"error": repr(e)}))

    def _ids(self, msgs, extra=None):
        ids = self.tok.apply_chat_template([{"role": "system", "content": self.system}] + msgs, tools=self.tools,
                                           add_generation_prompt=True)
        return list(ids) + list(extra or [])

    def _greedy(self, rid, ids, max_new, emit, stop_text=None):
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
            if nxt in STOPS:
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
        return out, {"ctx": len(ids), "n": len(out), "first_ms": round(((t_first or t_end) - t0) / 1e6, 1),
                     "ms": round((t_end - t0) / 1e6, 1)}

    def _serve(self, rid, msgs, t_req):
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
