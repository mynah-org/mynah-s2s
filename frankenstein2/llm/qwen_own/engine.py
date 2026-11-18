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

import contextlib
import functools
import time

import torch


def _on_stream(fn):
    """Run an engine method on the engine's CUDA stream (if one was given)."""
    @functools.wraps(fn)
    def w(self, *a, **k):
        ctx = torch.cuda.stream(self.stream) if getattr(self, "stream", None) is not None else contextlib.nullcontext()
        with ctx:
            return fn(self, *a, **k)
    return w

from .config import load_cfg
from .model import Model
from .weights import load_weights


class QwenOwn:
    GRAPH_B = (1, 2, 4, 8, 12, 16)
    GRAPH_L = (512, 1024, 2048, 4096, 8192, 16384, 32768)

    def __init__(self, path, device="cuda", mode="bf16", slots=4, max_len=8192, group=128, graphs=True, mtp=False,
                 tokenizer=None, stream=None):
        """tokenizer: any object with apply_chat_template / decode / convert_tokens_to_ids (default: QwenTokenizer,
        tokenizers + jinja2 on the checkpoint's own files). stream: a torch.cuda.Stream all of this engine's GPU work is issued on (an
        external scheduler can overlap it with other models); None = the current stream."""
        t0 = time.monotonic()
        self.path, self.device = str(path), torch.device(device)
        self.stream = stream
        self.cfg = load_cfg(path)
        if tokenizer is None:                                          # tokenizers + jinja2: no transformers
            from .tokenizer import QwenTokenizer
            tokenizer = QwenTokenizer(self.path)
        self.tok = tokenizer
        W, info = load_weights(path, self.device, mtp=mtp)
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
        self.graphs_frozen = False
        self.prefill_chunk = 1024
        self.use_mtp = mtp
        self.mtp_stats = {"steps": 0, "accepted": 0}
        self.sess: dict = {}
        self.prefixes: list = []
        self.prefix_hits = 0
        self.live_hits = 0
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

    # ---- shared prompt prefix (system prompt + tools), computed once and copied into each new session ----
    @_on_stream
    def add_prefix(self, ids: list) -> None:
        """Prefill `ids` once (in the scratch slot) and keep the resulting state: the linear-attention states and the
        attention K/V of its positions. A later prompt that starts with `ids` copies this state instead of prefilling
        those tokens again (sessions of one tenant share their system prompt and tool schemas)."""
        ids = list(ids)
        if not ids or any(p["ids"] == ids for p in self.prefixes):
            return
        s, n = self.scratch, len(ids)
        self._zero(s)
        for a in range(0, n, self.prefill_chunk):
            b = min(n, a + self.prefill_chunk)
            self.model.forward(torch.tensor([ids[a:b]], device=self.device), self.state,
                               torch.tensor([s], device=self.device), torch.arange(a, b, device=self.device)[None],
                               torch.tensor([b], device=self.device), prefill=True)
        st = self.state
        self.prefixes.append({"ids": ids, "n": n,
                              "conv": {i: t[s].clone() for i, t in st["conv"].items()},
                              "rec": {i: t[s].clone() for i, t in st["rec"].items()},
                              "k": {i: t[s, :, :n].clone() for i, t in st["k"].items()},
                              "v": {i: t[s, :, :n].clone() for i, t in st["v"].items()}})

    def shared_prefix(self, system: dict, tools=None) -> list:
        """Token ids that every prompt with this system message and tools starts with: the common part of two renders
        that differ only in the user message (the template needs a user turn, so the system block alone can't be
        rendered)."""
        a = self.chat_ids([system, {"role": "user", "content": "a"}], tools)
        b = self.chat_ids([system, {"role": "user", "content": "zz"}], tools)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return list(a[:n])

    def _from_prefix(self, s, ids) -> int:
        best = None
        for p in self.prefixes:
            if p["n"] < len(ids) and ids[:p["n"]] == p["ids"] and (best is None or p["n"] > best["n"]):
                best = p
        if best is None:
            return 0
        st, n = self.state, best["n"]
        for i, t in st["conv"].items():
            t[s].copy_(best["conv"][i])
        for i, t in st["rec"].items():
            t[s].copy_(best["rec"][i])
        for i, t in st["k"].items():
            t[s, :, :n].copy_(best["k"][i])
        for i, t in st["v"].items():
            t[s, :, :n].copy_(best["v"][i])
        self.prefix_hits += 1
        return n

    # ---- prompts ----
    def chat_ids(self, messages, tools=None, thinking=False) -> list:
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=True,
                                            enable_thinking=thinking, return_dict=False)

    # ---- forward ----
    def prefill(self, s, ids: list) -> torch.Tensor:
        """Feed `ids` (the whole prompt) to session s, reusing the snapshot when the prompt extends the last one.
        Returns the next-token logits [V]."""
        self.prefill_begin(s, ids)
        while True:
            out = self.prefill_step(s, self.prefill_chunk)
            if out is not None:
                return out

    def prefill_begin(self, s, ids: list) -> int:
        """Start (or restart) a prompt for session s; returns how many tokens are left to prefill (the part after the
        session's snapshot when the prompt extends its previous prompt). Pair with prefill_step()."""
        se = self.sess[s]
        if len(ids) > self.max_len:
            raise ValueError("prompt longer than max_len")
        start = 0
        live = len(se["ids"])
        mtp_on = self.model.mtp is not None and self.use_mtp
        if live and not mtp_on and "pending" not in se and len(ids) > live and ids[:live] == se["ids"]:
            # continue from the live state: the new prompt extends everything this session has already fed (its last
            # prompt + the tokens it generated, e.g. a tool call); only the new part (the tool result) is prefilled,
            # as one continuation (VoiceChat V2's "fast phase 2" idea, here without re-feeding the call tokens)
            start = live
            self.live_hits += 1
        elif se["snap"] is not None and ids[:se["snap"]["n"]] == se["ids"][:se["snap"]["n"]] and len(ids) > se["snap"]["n"]:
            start = self._restore(s)
        else:
            self._zero(s)
            if not (self.model.mtp is not None and self.use_mtp):         # MTP K/V of the prefix is not kept
                start = self._from_prefix(s, ids)
        se["pending"], se["cursor"], se["hid_prev"] = list(ids), start, None
        return len(ids) - start

    @_on_stream
    def prefill_step(self, s, budget: int):
        """Prefill up to `budget` more prompt tokens of session s (so prefill can be interleaved with other sessions'
        decode steps). Returns None while tokens remain, then the next-token logits [V]."""
        se = self.sess[s]
        ids, a = se["pending"], se["cursor"]
        b = min(len(ids), a + max(1, budget))
        mtp = self.model.mtp is not None and self.use_mtp
        out = self.model.forward(torch.tensor([ids[a:b]], device=self.device), self.state,
                                 torch.tensor([s], device=self.device), torch.arange(a, b, device=self.device)[None],
                                 torch.tensor([b], device=self.device), prefill=True, hidden=mtp)
        logits, hid = out if mtp else (out, None)
        if mtp and b < len(ids):                                        # MTP over this chunk: (token i+1, hidden i)
            self._mtp_fill(s, ids[a + 1:b + 1], hid, a + 1)
        se["cursor"], se["hid_prev"] = b, (a, hid)
        if b < len(ids):
            return None
        se["ids"] = list(ids)
        self._snapshot(s, len(ids))
        if mtp:
            se["mtp_tail"] = se["hid_prev"]                             # last chunk: filled once the next token is known
        se.pop("pending", None)
        return logits[0]

    @_on_stream
    def _mtp_fill(self, s, toks, hid, p0):
        """Run the MTP head over tokens `toks` paired with main hidden states `hid` [1, T, H], positions p0.. ."""
        T = len(toks)
        ids = torch.tensor([toks], device=self.device)
        pos = torch.arange(p0, p0 + T, device=self.device)[None]
        return self.model.mtp_forward(ids, hid[:, :T], self.state, torch.tensor([s], device=self.device), pos,
                                      torch.tensor([p0 + T], device=self.device), prefill=True)[0]

    @_on_stream
    def decode_batch(self, sessions: list, tokens: list) -> torch.Tensor:
        """One token for each session (in order). Returns logits [B, V]. With CUDA graphs: the batch is padded to a
        captured size with the scratch slot, the attention reads a static cache-length bucket."""
        dev = self.device
        B = len(sessions)
        cur = [len(self.sess[s]["ids"]) for s in sessions]
        L = next((x for x in self.GRAPH_L if x >= max(cur) + 1), self.max_len)
        L = min(L, self.max_len)
        g = None
        if self.use_graphs and B <= self.GRAPH_B[-1]:
            Bg = next(x for x in self.GRAPH_B if x >= B)
            g = self.graphs.get((Bg, L))
            if g is None and not self.graphs_frozen:
                g = self._capture(Bg, L)
        if g is not None:
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

    def warm_graphs(self, max_len: int | None = None) -> int:
        """Capture every decode graph (batch bucket x cache-length bucket up to max_len) now, then freeze: a missing
        bucket later runs eager instead of being captured. In the one-process engine other threads use the GPU while
        the LLM decodes, and a capture during their work fails (a device sync is not allowed while a stream
        captures), so all captures happen before the sessions start."""
        top = min(max_len or self.max_len, self.max_len)
        n = 0
        for L in [x for x in self.GRAPH_L if x <= top] + ([top] if top not in self.GRAPH_L else []):
            for B in self.GRAPH_B:
                if (B, L) not in self.graphs:
                    self._capture(B, L)
                    n += 1
        self.graphs_frozen = True
        return n

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
        with torch.cuda.graph(graph, pool=self.graphs.get("pool"), capture_error_mode="thread_local"):
            g["out"] = run()
        self.graphs.setdefault("pool", graph.pool())
        g["graph"] = graph
        self.graphs[(B, L)] = g
        return g

    # ---- speculative decoding with the MTP head (greedy, lossless) ----
    @_on_stream
    def _verify(self, s, t, d, p, L):
        """Main model on [t, d] at positions p, p+1 -> (logits [2, V], hidden [1, 2, H], rollback states)."""
        dev = self.device
        key = ("verify", L)
        if self.use_graphs:
            g = self.graphs.get(key) or self._capture_spec(key, L)
            g["ids"][0, 0], g["ids"][0, 1] = t, d
            g["pos"][0, 0], g["pos"][0, 1] = p, p + 1
            g["lens"][0] = p + 2
            g["slots"][0] = s
            g["graph"].replay()
            return g["out"][0][0], g["out"][1], g["rb"]
        out = self.model.forward(torch.tensor([[t, d]], device=dev), self.state, torch.tensor([s], device=dev),
                                 torch.tensor([[p, p + 1]], device=dev), torch.tensor([p + 2], device=dev),
                                 prefill=False, kv_len=L, all_logits=True, hidden=True)
        return out[0][0], out[1], self.state["rollback"]

    @_on_stream
    def _draft(self, s, toks, hid, p, L):
        """MTP head on 1 or 2 tokens at positions p.. -> draft logits [V]."""
        dev = self.device
        T = len(toks)
        key = ("mtp", T, L)
        if self.use_graphs:
            g = self.graphs.get(key) or self._capture_spec(key, L)
            g["ids"][0, :T] = torch.tensor(toks, device=dev)
            g["pos"][0, :T] = torch.arange(p, p + T, device=dev)
            g["lens"][0] = p + T
            g["slots"][0] = s
            g["hid"].copy_(hid[:, :T])
            g["graph"].replay()
            return g["out"][0]
        return self.model.mtp_forward(torch.tensor([toks], device=dev), hid[:, :T], self.state,
                                      torch.tensor([s], device=dev), torch.arange(p, p + T, device=dev)[None],
                                      torch.tensor([p + T], device=dev), prefill=False, kv_len=L)[0]

    def _capture_spec(self, key, L):
        dev, H = self.device, self.cfg.hidden
        T = 2 if key[0] == "verify" else key[1]
        g = {"ids": torch.zeros(1, T, dtype=torch.long, device=dev),
             "pos": torch.arange(T, device=dev)[None].clone(), "lens": torch.full((1,), T, dtype=torch.long, device=dev),
             "slots": torch.full((1,), self.scratch, dtype=torch.long, device=dev),
             "hid": torch.zeros(1, T, H, dtype=torch.bfloat16, device=dev)}
        if key[0] == "verify":
            run = lambda: self.model.forward(g["ids"], self.state, g["slots"], g["pos"], g["lens"], prefill=False,  # noqa
                                             kv_len=L, all_logits=True, hidden=True)
        else:
            run = lambda: self.model.mtp_forward(g["ids"], g["hid"], self.state, g["slots"], g["pos"], g["lens"],  # noqa
                                                 prefill=False, kv_len=L)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graphs.get("pool"), capture_error_mode="thread_local"):
            g["out"] = run()
        self.graphs.setdefault("pool", graph.pool())
        g["graph"] = graph
        if key[0] == "verify":
            g["rb"] = dict(self.state["rollback"])                   # static tensors written by every replay
        self.graphs[key] = g
        return g

    def _rollback(self, s, rb):
        """Keep the main model's DeltaNet state after the first verified token only (draft rejected)."""
        for i, keep in rb.items():
            rec, conv = keep[0]
            self.state["rec"][i][s].copy_(rec[0])
            self.state["conv"][i][s].copy_(conv[0])

    @torch.no_grad()
    def generate_mtp(self, s, ids, max_new=192, stop=("</tool_call>",), stop_flag=None):
        """Greedy speculative decoding: the MTP head drafts one token, the main model verifies [token, draft] in one
        forward; the draft is kept only if it equals the main model's own greedy token, so the output is the plain
        greedy output (up to bf16 numerics of a 2-token vs 1-token forward)."""
        t0 = time.perf_counter()
        logits = self.prefill(s, ids)
        se = self.sess[s]
        eos = set(self.eos_ids)
        out, text, sent, reason, t_first = [], "", 0, "max", None
        n = len(ids)
        t = int(torch.argmax(logits))
        a, hid = se["mtp_tail"]
        L = lambda p: min(next((x for x in self.GRAPH_L if x >= p + 3), self.max_len), self.max_len)  # noqa: E731
        dl = self._mtp_fill(s, ids[a + 1:] + [t], hid, a + 1)            # last prompt chunk + the first token
        steps = acc = 0

        def emit(tok):
            nonlocal text, sent, t_first
            out.append(tok)
            text = self.tok.decode(out, skip_special_tokens=False)
            t_first = t_first if t_first is not None else time.perf_counter() - t0
        p = n
        while len(out) < max_new:
            if t in eos:
                reason = "eos"; break
            emit(t)
            if not text.endswith("\ufffd") and len(text) > sent:
                yield {"text": text[sent:]}; sent = len(text)
            if any(x in text for x in stop):
                reason = "stop"; break
            if stop_flag is not None and stop_flag():
                reason = "cancel"; break
            d = int(torch.argmax(dl))
            lg, hid2, rb = self._verify(s, t, d, p, L(p))
            steps += 1
            y0 = int(torch.argmax(lg[0]))
            if y0 == d and d not in eos:
                acc += 1
                emit(d)
                if not text.endswith("\ufffd") and len(text) > sent:
                    yield {"text": text[sent:]}; sent = len(text)
                if any(x in text for x in stop) or len(out) >= max_new:
                    se["ids"] += [t, d]
                    reason = "stop" if any(x in text for x in stop) else "max"; break
                y1 = int(torch.argmax(lg[1]))
                dl = self._draft(s, [d, y1], hid2, p + 1, L(p))
                se["ids"] += [t, d]
                t, p = y1, p + 2
            else:
                self._rollback(s, rb)
                dl = self._draft(s, [y0], hid2, p + 1, L(p))
                se["ids"] += [t]
                t, p = y0, p + 1
        self.mtp_stats["steps"] += steps
        self.mtp_stats["accepted"] += acc
        yield {"done": True, "ids": out, "text": text, "reason": reason, "first_token_s": t_first,
               "total_s": time.perf_counter() - t0, "n": len(out), "prompt_n": len(ids), "mtp_steps": steps,
               "mtp_accepted": acc}

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
