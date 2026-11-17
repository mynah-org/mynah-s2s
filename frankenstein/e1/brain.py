"""Text-mode brain: the SAME loaded VoiceChat Thinker modules, driven on the plain text path with its OWN cache.

  context  Nano-v2 chat template; system = the session prompt + " /no_think"; history of user turns (multilingual ASR
           text) and the agent replies actually spoken; generation prompt.
  reply    llm(embed_tokens(ids)) -> lm_head (the training text_to_text path), greedy, id 12 (= Nano chat EOS = duplex
           PAD) banned, stop on </s> (2) or <SPECIAL_11> (11), max ``max_new`` tokens.
  tools    T1 (DESIGN.md 4.3): decision = function head at the last context position, SOTC (20) vs PAD (12);
           call = the duplex FC fast-path input layout (fusion(agent PAD, audio = the wrapper's silence embedding,
           function = previous function token)) read by function_head, SOTC + the format prefix forced, greedy to
           EOTC (21) -- the forced-call harness of the multilingual lab (a1b_forced_call.py run_fh), batch 1.
Cache isolation: every call builds a NEW hybrid cache object; ``fingerprint`` hashes a cache's tensors so the runner
can prove the duplex cache is untouched by a brain call.
"""

from __future__ import annotations

import json
import re
import time

import torch

PAD, BOS, EOS, SOTC, EOTC, SPECIAL_11 = 12, 1, 2, 20, 21, 11
CALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def fingerprint(cache) -> str:
    """Cheap content hash of a hybrid cache (Mamba conv/ssm states + attention K/V): sum and abs-sum per tensor."""
    if cache is None:
        return "none"
    parts = []
    for name in ("conv_states", "ssm_states", "key_cache", "value_cache"):
        for t in getattr(cache, name, []) or []:
            if isinstance(t, torch.Tensor) and t.numel():
                tf = t.float()
                parts.append(f"{tf.sum().item():.6e}/{tf.abs().sum().item():.6e}/{tuple(t.shape)}")
    parts.append(str(getattr(cache, "_v2_pos", None)))
    import hashlib
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class TextBrain:
    def __init__(self, eng, tok, system_prompt: str, max_new: int = 90, max_call: int = 56):
        from mynah_v2.model.llm_cache import _hybrid_cache_cls
        w = eng.wrapper
        self.w, self.stt, self.fusion, self.tok = w, w.model.stt_model, w.fusion_module, tok
        self.e = self.stt.embed_tokens
        p = next(self.stt.llm.parameters())
        self.dtype, self.dev = p.dtype, p.device
        self.cache_cls = _hybrid_cache_cls(self.stt.llm)
        self.system = system_prompt + " /no_think"
        self.max_new, self.max_call = max_new, max_call
        if getattr(w, "_silence_embedding_cache", None) is None:
            w._create_silence_embedding_cache()
        self.sil = w._silence_embedding_cache.reshape(-1).to(self.dev, self.dtype)
        self.pad_e = self.e(torch.tensor([PAD], device=self.dev))[0].to(self.dtype)
        a = tok.encode('<TOOLCALL>' + json.dumps([{"name": "find_customer", "arguments": {}}]), add_special_tokens=False)
        b = tok.encode('<TOOLCALL>' + json.dumps([{"name": "search_catalog", "arguments": {}}]), add_special_tokens=False)
        k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
        self.call_prefix = a[:k]                      # '<TOOLCALL>[{"name": "' as the tokenizer splits it
        self.last_cache = None

    def context(self, messages: list) -> list:
        return self.tok.apply_chat_template([{"role": "system", "content": self.system}] + messages,
                                            add_generation_prompt=True, tokenize=True)

    def _new_cache(self):
        return self.cache_cls(self.stt.llm.config, 1, dtype=self.dtype, device=self.dev, ssm_dtype=torch.float32)

    def _fwd(self, x, cache):
        ans = self.stt(x, cache=cache)
        return ans, ans.get("cache", cache)

    @torch.no_grad()
    def run(self, messages: list, decide_tool: bool = True) -> dict:
        """One brain turn. Returns {'ctx_len', 'prefill_ms', 'tool': {...} | None, 'ids': [...] (reply, only if no
        call), 'text', 'decode_ms_per_tok', 'cache_fp'}. A call is returned WITHOUT a reply: the runner executes the
        tool, appends the result and calls run() again (resume)."""
        ids = self.context(messages)
        cache = self._new_cache()
        self.last_cache = cache
        t0 = time.monotonic()
        x = self.e(torch.tensor([ids], device=self.dev)).to(self.dtype)
        ans, cache = self._fwd(x, cache)
        torch.cuda.synchronize()
        out = {"ctx_len": len(ids), "prefill_ms": round((time.monotonic() - t0) * 1e3, 1), "tool": None}
        fl = ans["function_logits"][0, -1].float()
        margin = float(fl[SOTC] - fl[PAD])
        out["tool_margin"] = round(margin, 3)
        out["func_top1"] = int(fl.argmax())
        if decide_tool and margin > 0:
            out["tool"] = self._call(cache)
            return out
        lg = ans["text_logits"][0, -1].float()
        gen, t1 = [], time.monotonic()
        for _ in range(self.max_new):
            lg[PAD] = -float("inf")
            nxt = int(lg.argmax())
            if nxt in (EOS, SPECIAL_11):
                break
            gen.append(nxt)
            ans, cache = self._fwd(self.e(torch.tensor([[nxt]], device=self.dev)).to(self.dtype), cache)
            lg = ans["text_logits"][0, -1].float()
        torch.cuda.synchronize()
        out["ids"] = gen
        out["text"] = self.tok.decode(gen)
        out["decode_ms_per_tok"] = round((time.monotonic() - t1) * 1e3 / max(1, len(gen)), 1)
        out["cache_fp"] = fingerprint(cache)
        return out

    def _call(self, cache) -> dict:
        t0 = time.monotonic()
        seq = [SOTC] + list(self.call_prefix)
        gen = []
        i = 0
        while True:
            v = seq[i] if i < len(seq) else gen[-1]
            x = self.fusion(agent_text_embeds=self.pad_e[None, None], user_audio_embeds=self.sil[None, None],
                            user_text_embeds=None,
                            function_embeds=self.e(torch.tensor([[v]], device=self.dev)).to(self.dtype))
            ans, cache = self._fwd(x, cache)
            i += 1
            if i >= len(seq):
                nxt = int(ans["function_logits"][0, -1].float().argmax())
                gen.append(nxt)
                if nxt == EOTC or len(gen) >= self.max_call:
                    break
        torch.cuda.synchronize()
        text = self.tok.decode(list(self.call_prefix) + [g for g in gen if g != EOTC])
        if not text.rstrip().endswith("</TOOLCALL>"):
            text = text + "</TOOLCALL>" if "]" in text else text
        parsed = None
        m = CALL_RE.search(text)
        try:
            parsed = json.loads(m.group(1))[0] if m else None
        except Exception:  # noqa: BLE001 - the call text is model output
            parsed = None
        return {"text": text, "ids": gen, "eotc": bool(gen) and gen[-1] == EOTC, "parsed": parsed,
                "ms": round((time.monotonic() - t0) * 1e3, 1)}
