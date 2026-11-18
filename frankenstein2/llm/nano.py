"""Nemotron Nano 9B v2 (text mode) as an LLM plugin: Frankenstein v1's worker (frankenstein/e1/nano_worker.py,
imported read-only) with its tool step, schema-constrained calls and brief replies, behind the plugin interface.

``NanoShared`` = the loaded model (one per process) + precision profile:
    "bf16"          bf16 weights only
    "int4-decode"   bf16 weights for prefill + INT4 copies for the one-row decode steps (v1 G6/G8: the tool suite
                    identical to bf16); memory = bf16 + INT4
  and a lock that serialises requests across sessions (the INT4 prefill path shares a dequant scratch buffer per
  weight shape; no cross-session batching yet).
``NanoLLM`` = one session: its system prompt (tenant prompt + language policy + brief-reply rule), its tools (V2
ToolRegistry/Ledger bound handlers, wrapped by the spoken-numbers -> digits step in the session language), its own
worker thread and request queue. Text is streamed as deltas of the detokenised reply.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .base import wrap_tools

V1 = Path(__file__).resolve().parents[2] / "frankenstein"

BRIEF = {
    "it": ("Rispondi come in una telefonata: al massimo due frasi brevi, senza elenchi puntati, senza markdown e "
           "senza simboli; dì i prezzi a parole."),
    "fr": ("Réponds comme au téléphone : deux phrases courtes au maximum, sans listes, sans markdown ni symboles ; "
           "dis les prix en toutes lettres."),
    "en": "Answer as on a phone call: at most two short sentences, no lists, no markdown, no symbols.",
}
POLICY = {
    "fr": ("Réponds toujours en français, avec des phrases courtes. N'invente jamais de noms, codes postaux, numéros "
           "de commande, dates ou statuts : utilise seulement ce que dit le client et les résultats des outils. S'il "
           "manque une information, demande-la. Avant une action qui change quelque chose, demande confirmation."),
}


def _v1_path():
    if str(V1) not in sys.path:
        sys.path.insert(0, str(V1))


class NanoShared:  # pragma: no cover - GPU
    def __init__(self, nano_dir: str, lab_multilingual: str, precision: str = "int4-decode", graph: bool = False):
        _v1_path()
        sys.path.insert(0, str(lab_multilingual))
        import torch
        import llm_text_bench as B2
        B2.NANO = Path(nano_dir)
        self.B2, self.torch = B2, torch
        self.tok, self.model, _ = B2.load("nano")
        self.mem = {"bf16_allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3)}
        self.precision = precision
        if precision == "int4-decode":
            from e1.int4_dispatch import quantize_int4_decode_only
            self.int4 = quantize_int4_decode_only(self.model.backbone)
        elif precision != "bf16":
            raise ValueError(f"unknown Nano precision {precision!r} (bf16 | int4-decode)")
        self.mem["allocated_gb"] = round(torch.cuda.memory_allocated() / 2 ** 30, 3)
        self.graph = None
        if graph:
            from e1.nano_graph import NanoGraph
            self.graph = NanoGraph(self.model.backbone, lmax=4096)
        self.lock = threading.Lock()


class NanoLLM:  # pragma: no cover - GPU (the event conversion is model-free and tested through ScriptedLLM)
    def __init__(self, shared: NanoShared, system: str, tool_schemas: list, bound: dict, lang: str,
                 backend=None, max_new: int = 60, soft_cap: int = 24):
        _v1_path()
        from e1.nano_worker import NanoWorker
        self.sh, self.lang = shared, lang
        self.record: list = []
        tools = wrap_tools(bound, lang, self.record,
                           user_texts=lambda: list(getattr(backend, "_said", []) or []))
        w = NanoWorker(shared.tok, shared.model, system, tool_schemas, tools, backend, shared.B2.parse_toolcall,
                       max_new=max_new, tool_step=True, constrained=True)
        w.reply_soft_cap = soft_cap
        w.graph, w.graph_reply_only = shared.graph, shared.graph is not None
        orig = w._serve_job

        def serialized(rid, msgs, t_req):
            with shared.lock:
                return orig(rid, msgs, t_req)
        w._serve_job = serialized
        self.w = w
        self.ids: dict = {}
        self.text: dict = {}
        self.n_calls = 0
        self.calls: list = []
        w.start()

    def submit(self, rid, messages):
        self.ids[rid], self.text[rid] = [], ""
        self.w.submit(rid, messages)

    def cancel(self, rid):
        self.w.cancel(rid)

    def poll(self):
        out = []
        for ev in self.w.poll():
            kind, rid = ev[0], ev[1]
            if kind == "tok":
                ids = self.ids.setdefault(rid, [])
                ids.append(ev[2])
                txt = self.sh.tok.decode(ids, skip_special_tokens=True)
                prev = self.text.get(rid, "")
                if txt.startswith(prev) and len(txt) > len(prev) and not txt.endswith("�"):
                    out.append({"ev": "text", "rid": rid, "text": txt[len(prev):], "t_ns": ev[3]})
                    self.text[rid] = txt
            elif kind == "decided":
                out.append({"ev": "decided", "rid": rid})
            elif kind == "tool":
                rec = ev[2]
                self.n_calls += 1
                norm = next((r["args"] for r in reversed(self.record) if r["name"] == rec["name"]), rec["args"])
                fixes = next((r.get("fixes") for r in reversed(self.record) if r["name"] == rec["name"]), [])
                c = {"ev": "tool", "rid": rid, "name": rec["name"], "args": norm, "args_raw": rec["args"], "fixes": fixes,
                     "result": rec["result"], "executed": rec.get("executed"), "call_id": f"call{self.n_calls}"}
                self.calls.append(c)
                out.append(c)
            elif kind == "done":
                full = ev[2]
                prev = self.text.get(rid, "")
                if full.startswith(prev) and len(full) > len(prev):
                    out.append({"ev": "text", "rid": rid, "text": full[len(prev):]})
                out.append({"ev": "done", "rid": rid, "text": full, "info": ev[4]})
            elif kind == "skipped":
                out.append({"ev": "done", "rid": rid, "text": "", "info": {"skipped": True}})
        return out

    def close(self):
        self.w.close(timeout=30)
