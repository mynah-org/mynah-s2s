"""QwenOwn: our own inference engine for Qwen3.5-4B (bf16 / int8 / int4 weight-only) and Qwen3.6-35B-A3B (AWQ 4-bit
experts). torch + the checkpoint's HF tokenizer and chat template; no serving framework.

  eng = QwenOwn(path, mode="bf16", slots=4, max_len=8192)
  sid = eng.open()                                    # a session slot (KV cache + DeltaNet state)
  for ev in eng.generate(sid, eng.chat_ids(messages, tools), max_new=192, stop=("</tool_call>",)):
      ...                                             # {"text": delta} ... then {"done": True, "ids": [...], ...}
  eng.close(sid)

Sessions keep their state between requests: a new prompt that extends the previous prompt of the same session is
prefilled from the end of that prompt (DeltaNet states are restored from a snapshot taken there; the KV cache past it is
overwritten), so each turn only pays for its new tokens. `decode_batch` steps several sessions together (one token
each). Greedy decoding.
"""

from __future__ import annotations

import time

import torch

from .config import load_cfg
from .model import Model
from .weights import load_weights


class QwenOwn:
    GRAPH_B = (1, 2, 4, 8, 16)
    GRAPH_L = (512, 1024, 2048, 4096, 8192, 16384, 32768)

    def __init__(self, path, device="cuda", mode="bf16", slots=4, max_len=8192, group=128, graphs=True):
        from transformers import AutoTokenizer
        t0 = time.monotonic()
        self.path, self.device = str(path), torch.device(device)
        self.cfg = load_cfg(path)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        W, info = load_weights(path, self.device)
        self.quant = info
        awq_group = info.get("group_size") if info.get("quant_method") == "awq" else None
        self.model = Model(self.cfg, W, self.device, mode=mode, group=group, awq_group=awq_group)
        if W:
            raise ValueError(f"unused tensors: {sorted(W)[:8]}")
        self.state = self.model.new_state(slots + 1, max_len)              # + one scratch slot (graph padding)
        self.scratch = slots
        self.max_len = max_len
        self.free = list(range(slots))
        self.graphs: dict = {}
        self.use_graphs = graphs and self.device.type == "cuda"
        self.sess: dict = {}
        self.mode = mode
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = sorted({*self.cfg.eos_ids, *([im_end] if isinstance(im_end, int) else [])})
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        self.load_s = round(time.monotonic() - t0, 1)

    # ---- sessions ----
    def open(self) -> int:
        s = self.free.pop(0)
        self.sess[s] = {"ids": [], "snap": None}
        self._zero(s)
        return s

    def close(self, s):
        self.sess.pop(s, None)
        self.free.append(s)

    def _zero(self, s):
        for t in (*self.state["conv"].values(), *self.state["rec"].values()):
            t[s].zero_()

    def _snapshot(self, s, n_ids):
        st = self.state
        self.sess[s]["snap"] = {"n": n_ids, "conv": {i: t[s].clone() for i, t in st["conv"].items()},
                                "rec": {i: t[s].clone() for i, t in st["rec"].items()}}

    def _restore(self, s):
        sn = self.sess[s]["snap"]
        for i, t in self.state["conv"].items():
            t[s].copy_(sn["conv"][i])
        for i, t in self.state["rec"].items():
            t[s].copy_(sn["rec"][i])
        return sn["n"]

    # ---- prompts ----
    def chat_ids(self, messages, tools=None, thinking=False) -> list:
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=True,
                                            enable_thinking=thinking, return_dict=False)

    # ---- forward ----
    def prefill(self, s, ids: list) -> torch.Tensor:
        """Feed `ids` (the whole prompt) to session s, reusing the snapshot when the prompt extends the last one."""
        se = self.sess[s]
        start = 0
        if se["snap"] is not None and ids[:se["snap"]["n"]] == se["ids"][:se["snap"]["n"]] and len(ids) > se["snap"]["n"]:
            start = self._restore(s)
        else:
            self._zero(s)
        if len(ids) > self.max_len:
            raise ValueError("prompt longer than max_len")
        new = torch.tensor([ids[start:]], device=self.device)
        pos = torch.arange(start, len(ids), device=self.device)[None]
        slots = torch.tensor([s], device=self.device)
        lens = torch.tensor([len(ids)], device=self.device)
        logits = self.model.forward(new, self.state, slots, pos, lens, prefill=True)
        se["ids"] = list(ids)
        self._snapshot(s, len(ids))
        return logits[0]

    def decode_batch(self, sessions: list, tokens: list) -> torch.Tensor:
        """One token for each session (in order). Returns logits [B, V]. With CUDA graphs: the batch is padded to a
        captured size with the scratch slot, the attention reads a static cache-length bucket."""
        dev = self.device
        B = len(sessions)
        cur = [len(self.sess[s]["ids"]) for s in sessions]
        L = next((x for x in self.GRAPH_L if x >= max(cur) + 1), self.max_len)
        L = min(L, self.max_len)
        if self.use_graphs and B <= self.GRAPH_B[-1]:
            Bg = next(x for x in self.GRAPH_B if x >= B)
            g = self.graphs.get((Bg, L)) or self._capture(Bg, L)
            g["ids"][:B, 0] = torch.tensor(tokens, device=dev)
            g["pos"][:B, 0] = torch.tensor(cur, device=dev)
            g["lens"][:B] = g["pos"][:B, 0] + 1
            g["slots"][:B] = torch.tensor(sessions, device=dev)
            g["ids"][B:], g["pos"][B:], g["lens"][B:], g["slots"][B:] = 0, 0, 1, self.scratch
            g["graph"].replay()
            logits = g["out"][:B]
        else:
            pos = torch.tensor([[c] for c in cur], device=dev)
            logits = self.model.forward(torch.tensor([[t] for t in tokens], device=dev), self.state,
                                        torch.tensor(sessions, device=dev), pos, pos[:, 0] + 1, prefill=False, kv_len=L)
        for s, t in zip(sessions, tokens):
            self.sess[s]["ids"].append(t)
        return logits

    def _capture(self, B, L):
        dev = self.device
        g = {"ids": torch.zeros(B, 1, dtype=torch.long, device=dev), "pos": torch.zeros(B, 1, dtype=torch.long, device=dev),
             "lens": torch.ones(B, dtype=torch.long, device=dev),
             "slots": torch.full((B,), self.scratch, dtype=torch.long, device=dev)}
        run = lambda: self.model.forward(g["ids"], self.state, g["slots"], g["pos"], g["lens"], prefill=False,  # noqa: E731
                                         kv_len=L)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graphs.get("pool")):
            g["out"] = run()
        self.graphs.setdefault("pool", graph.pool())
        g["graph"] = graph
        self.graphs[(B, L)] = g
        return g

    # ---- generation ----
    @torch.no_grad()
    def generate(self, s, ids, max_new=192, stop=("</tool_call>",), stop_flag=None):
        """Greedy stream for one session: yields {"text": delta} and finally {"done": True, ...timings}."""
        t0 = time.perf_counter()
        logits = self.prefill(s, ids)
        eos = set(self.eos_ids)
        out, text, sent = [], "", 0
        t_first = None
        reason = "max"
        for _ in range(max_new):
            tok = int(torch.argmax(logits))
            if tok in eos:
                reason = "eos"; break
            out.append(tok)
            text = self.tok.decode(out, skip_special_tokens=False)
            if t_first is None:
                t_first = time.perf_counter() - t0
            if not text.endswith("�") and len(text) > sent:
                yield {"text": text[sent:]}
                sent = len(text)
            if any(x in text for x in stop):
                reason = "stop"; break
            if stop_flag is not None and stop_flag():
                reason = "cancel"; break
            logits = self.decode_batch([s], [tok])[0]
        dt = time.perf_counter() - t0
        yield {"done": True, "ids": out, "text": text, "reason": reason, "first_token_s": t_first, "total_s": dt,
               "n": len(out), "prompt_n": len(ids)}

    # ---- facts ----
    def footprint(self) -> dict:
        from .quant import QLinear
        wb = 0
        for d in self.model.layers:
            for v in d.values():
                if isinstance(v, QLinear):
                    wb += v.nbytes()
                elif isinstance(v, torch.Tensor):
                    wb += v.numel() * v.element_size()
                elif isinstance(v, dict):
                    wb += sum(t.numel() * t.element_size() for t in v.values())
        wb += self.model.embed.numel() * 2 + (0 if self.cfg.tie_embeddings else self.model.lm_head.nbytes())
        st = self.state
        per_slot_state = sum(t[0].numel() * t.element_size() for t in (*st["conv"].values(), *st["rec"].values()))
        per_token_kv = sum(t[0, :, 0].numel() * t.element_size() for t in (*st["k"].values(), *st["v"].values()))
        return {"weights_gib": round(wb / 2 ** 30, 3), "deltanet_state_mib_per_session": round(per_slot_state / 2 ** 20, 2),
                "kv_kib_per_token": round(per_token_kv / 1024, 2),
                "kv_mib_per_session_at_max_len": round(per_token_kv * self.max_len / 2 ** 20, 1)}
