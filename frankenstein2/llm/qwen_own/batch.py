"""BatchRunner: continuous batching of many conversations on one QwenOwn engine (one process, one GPU).

  run = BatchRunner(eng, prefill_budget=256)
  run.submit("sess-A", prompt_ids, max_new=192)      # opens a slot for a new key; a known key keeps its slot and
  run.submit("sess-B", prompt_ids)                    #   continues from its previous prompt's snapshot
  while run.busy():
      for ev in run.step():                           # {"key", "text"} deltas, then {"key", "done", "ids", "reason", ...}
          ...
  run.cancel("sess-A"); run.close("sess-B")           # cancel stops the request; close frees the slot

One step = (1) one prefill chunk of at most `prefill_budget` tokens for each session still reading its prompt (so a long
prompt never blocks the others' decoding for more than one chunk), then (2) one batched decode token for every
session that is generating (CUDA graph per batch-size bucket). Greedy; a session stops on end of turn, on a stop
string (default "</tool_call>": the call is complete and can be run), on max_new or on cancel. Speculative MTP
decoding is single-session (QwenOwn.generate_mtp) and not used here.
"""

from __future__ import annotations

import time

import torch


class BatchRunner:
    def __init__(self, eng, prefill_budget: int = 256, stop=("</tool_call>",)):
        self.eng, self.budget, self.stop = eng, prefill_budget, tuple(stop)
        self.slot: dict = {}                    # key -> engine slot
        self.job: dict = {}                     # key -> request state

    # ---- control ----
    def submit(self, key, ids: list, max_new: int = 192, stop=None):
        if key not in self.slot:
            if not self.eng.free:
                raise RuntimeError("no free session slot")
            self.slot[key] = self.eng.open()
        self.eng.prefill_begin(self.slot[key], ids)
        self.job[key] = {"phase": "prefill", "out": [], "text": "", "sent": 0, "max_new": max_new, "t0": time.perf_counter(),
                         "t_first": None, "stop": tuple(stop) if stop is not None else self.stop, "prompt_n": len(ids)}

    def cancel(self, key):
        j = self.job.get(key)
        if j is not None:
            j["cancel"] = True

    def close(self, key):
        self.job.pop(key, None)
        s = self.slot.pop(key, None)
        if s is not None:
            self.eng.close(s)

    def busy(self) -> bool:
        return bool(self.job)

    # ---- one scheduling step ----
    @torch.no_grad()
    def step(self) -> list:
        ev = []
        for key, j in list(self.job.items()):
            if j.get("cancel"):
                ev.append(self._done(key, "cancel"))
                continue
            if j["phase"] == "prefill":
                lg = self.eng.prefill_step(self.slot[key], self.budget)
                if lg is not None:
                    j["phase"], j["next"] = "decode", int(torch.argmax(lg))
                    ev += self._emit(key, j)
        dec = [k for k, j in self.job.items() if j["phase"] == "decode" and "next" in j]
        if dec:
            logits = self.eng.decode_batch([self.slot[k] for k in dec], [self.job[k].pop("next") for k in dec])
            toks = logits.argmax(-1).tolist()
            for k, t in zip(dec, toks):
                self.job[k]["next"] = t
                ev += self._emit(k, self.job[k])
        return ev

    def _emit(self, key, j) -> list:
        t = j["next"]
        if t in self.eng.eos_ids:
            j.pop("next", None)
            return [self._done(key, "eos")]
        j["out"].append(t)
        j["t_first"] = j["t_first"] or time.perf_counter() - j["t0"]
        j["text"] = self.eng.tok.decode(j["out"], skip_special_tokens=False)
        ev = []
        if not j["text"].endswith("�") and len(j["text"]) > j["sent"]:
            ev.append({"key": key, "text": j["text"][j["sent"]:]})
            j["sent"] = len(j["text"])
        if any(x in j["text"] for x in j["stop"]):
            j.pop("next", None)
            ev.append(self._done(key, "stop"))
        elif len(j["out"]) >= j["max_new"]:
            j.pop("next", None)
            ev.append(self._done(key, "max"))
        return ev

    def _done(self, key, reason) -> dict:
        j = self.job.pop(key)
        if j.get("next") is not None and reason == "cancel":
            pass
        return {"key": key, "done": True, "ids": j["out"], "text": j["text"], "reason": reason,
                "first_token_s": j["t_first"], "total_s": time.perf_counter() - j["t0"], "n": len(j["out"]),
                "prompt_n": j["prompt_n"]}
