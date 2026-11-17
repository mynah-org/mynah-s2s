"""E1c part 1 (GPU job): native tool calling in TEXT mode, extending the multilingual lab's B2 harness.

Reuses, unchanged (imported, not copied): labs/multilingual/scripts/multilingual/llm_text_bench.py
  load(which)        Nano-9B-v2 (official) or the same Nano with every backbone / embedding / LM-head tensor replaced
                     by the VoiceChat Thinker's ("vc-text"); its docstring caveat applies: a vc-text drop mixes
                     language/knowledge loss with a template mismatch (the duplex Thinker never saw the chat template)
  parse_toolcall     the <TOOLCALL>[...]</TOOLCALL> parser
  the generation recipe (apply_chat_template(..., tools=...), greedy, HF generate)
and the agentbench suite: tenant_furniture.json (system prompt, tools, EN/IT/FR turns, gold), tenant_backend.py
(deterministic handlers with identity / confirmation policy) scored later by tenant_score.py; plus the V2 SQLite order
tool (mynah_v2.agent.tools make_orders_db / orders_tool) with NVIDIA's generate_random_number, EN + Italian turns.
Calls run through the V2 ToolRegistry / ToolLedger (the engine's tool-loop objects) bound to those handlers; results go
back with the chat template's tool role; up to 3 generations per user turn; the final text is the spoken reply.
Arms: nano (text), nano_asr (Nano on the multilingual-ASR transcript of the same turns, where audio exists), vc (vc-text,
id 12 = duplex PAD allowed as end only after a few tokens).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

POLICY = {
    "it": ("Rispondi sempre in italiano, con frasi brevi. Non inventare mai nomi, CAP, numeri d'ordine, date o stati: "
           "usa solo quello che dice il cliente e i risultati degli strumenti. Se manca un dato, chiedilo. Prima di "
           "un'azione che cambia qualcosa, chiedi conferma."),
    "en": ("Always answer in English, with short sentences. Never invent names, postcodes, order numbers, dates or "
           "statuses: use only what the caller says and the tool results. If a detail is missing, ask for it. Before "
           "an action that changes something, ask for confirmation."),
}

TENANT_IDS = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m11"]
ORDERS = [  # V2 FC suite turns (capability_v1 c07 / corpus_v1 r01 / c09 / q09) + XYZ123 lookup; Italian versions added
    {"id": "o1", "en": "Where is my order Q Q Q nine nine nine?", "it": "Dov'è il mio ordine Q Q Q nove nove nove?",
     "gold": {"tool": "get_order_status", "args": {"order_id": "QQQ999"}, "found": False}},
    {"id": "o2", "en": "Where is my order X Y Z one two three?", "it": "Dov'è il mio ordine X Y Z uno due tre?",
     "gold": {"tool": "get_order_status", "args": {"order_id": "XYZ123"}, "found": True,
              "must_say_any": ["12", "twelve", "dodici", "november", "novembre", "shipped", "spedito"]}},
    {"id": "o3", "en": "Give me a random number between one and ten.", "it": "Dammi un numero casuale tra uno e dieci.",
     "gold": {"tool": "generate_random_number", "args": {"min": 1, "max": 10}}},
    {"id": "o4", "en": "I was going to ask about an order, but never mind. Give me a random number between one and five "
                       "instead.",
     "it": "Volevo chiederti di un ordine, ma lascia stare. Dammi invece un numero casuale tra uno e cinque.",
     "gold": {"tool": "generate_random_number", "args": {"min": 1, "max": 5}, "forbid": "get_order_status"}},
]
ASR_KEYS = {"it": {"m01": ["m01_t0"], "m02": ["m02_t0"], "m03": ["m03_t0"], "m05": ["m05_t0"], "m06": ["m06_t0"],
                   "m07": ["m07_t0"], "m08": ["m08_t0"], "m11": ["m11_t0"]}}


def tool_schemas(reg) -> list:
    """The registry's schemas in the chat template's tools format."""
    out = []
    for s in reg.schemas:
        f = s.get("function", s)
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                     "parameters": f.get("parameters", {})}})
    return out


def main():  # pragma: no cover - GPU
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True, help="labs/multilingual/scripts (multilingual/ and agentbench/)")
    ap.add_argument("--asr", required=True, help="e1/asr_clips.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="nano,nano_asr,vc")
    ap.add_argument("--langs", default="it,en")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--budget-s", type=float, default=480)
    ap.add_argument("--nano-dir", default=None, help="Nano-9B-v2 snapshot with weights (overrides the lab default)")
    ap.add_argument("--two-llm-asr-dir", default=None, help="E1d part 2: e1_asr track dir; runs the two-LLM session")
    ap.add_argument("--two-llm-cut-s", type=float, default=52.0)
    ap.add_argument("--two-llm-fixes", action="store_true",
                    help="E1f: blank-frames turn opening, explicit tool step, no synthetic opener turn")
    ap.add_argument("--two-llm-need-s", type=float, default=170.0, help="do not start part 2 with less than this left")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "multilingual"))
    sys.path.insert(0, str(lab / "agentbench"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # package root: e1.*, frankenstein_s2s.*
    import llm_text_bench as B2
    if a.nano_dir:
        B2.NANO = Path(a.nano_dir)          # same revision, weights downloaded to this host's own cache
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolLedger, ToolRegistry, make_orders_db, orders_tool
    from mynah_v2.tracing.events import Tracer
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    scen = {s["id"]: s for s in spec["scenarios"]}
    asr = json.load(open(a.asr))
    nvidia_random = None
    try:
        sys.path.insert(0, str(Path.home() / "mynah-s2s/third_party/agent/nemo-speech/examples/speechlm2"))
        import offline_voicechat_fc_infer as fcmod
        from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline
        nvidia_random = StreamingS2SPipeline._tool_generate_random_number
        random_schema = fcmod.DEFAULT_TOOLS[0]["function"]
        orders_system = fcmod.DEFAULT_SYSTEM_MESSAGE
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"NeMo FC defaults unavailable: {e}")
    db = make_orders_db(out / "orders.sqlite")
    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()          # noqa: E731
    pkg = Path(__file__).resolve().parents[1]
    freeze = {"commit": __import__("os").environ.get("FRANK_COMMIT"), "argv": sys.argv, "seed": 1234,
              "harness_sha256": {str(p.relative_to(pkg)): sha(p) for p in sorted(pkg.rglob("*.py"))
                                 if "__pycache__" not in str(p)},
              "lab_sha256": {n: sha(lab / n) for n in ("multilingual/llm_text_bench.py", "agentbench/tenant_backend.py",
                                                      "agentbench/tenant_furniture.json")},
              "asr_sha256": sha(a.asr), "nano": str(B2.NANO), "vc": str(B2.VC),
              "nano_weights_note": "Nano-9B-v2 revision 6533e8de downloaded from HF to this host's own cache "
                                   "(owner-accepted); the shared cache holds only its config/tokenizer/code",
              "nano_index_sha256": sha(B2.NANO / "model.safetensors.index.json")}
    (out / "freeze.json").write_text(json.dumps(freeze, indent=1))

    def build(domain, lang, sid):
        """-> (system text, tools for the template, bound handlers, backend, ledger)."""
        tr = Tracer(None, session_id=sid)
        led = ToolLedger(sid, tr)
        reg = ToolRegistry()
        backend = None
        if domain == "tenant":
            said = []
            backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
            backend._said = said
            for t in spec["tools"]:
                reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
            system = spec["system_prompt"]
        else:
            reg.register("get_order_status", orders_tool(db),
                         "Get the status, delivery date and tracking code of a customer order by its order id.",
                         ORDERS_SCHEMA)
            reg.register("generate_random_number", nvidia_random, random_schema["description"],
                         random_schema["parameters"])
            system = orders_system
        bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
        return system + "\n" + POLICY[lang] + " /no_think", tool_schemas(reg), bound, backend, led

    items = []
    for lang in a.langs.split(","):
        for sid in TENANT_IDS:
            items.append({"domain": "tenant", "id": sid, "lang": lang,
                          "turns": [t[lang] for t in scen[sid]["turns"]]})
        for o in ORDERS:
            items.append({"domain": "orders", "id": o["id"], "lang": lang, "turns": [o[lang]], "gold": o["gold"]})

    def asr_turns(it):
        keys = ASR_KEYS.get(it["lang"], {}).get(it["id"])
        if it["domain"] != "tenant" or not keys:
            return None
        txt = [asr.get(f"{k}|{it['lang']}", {}).get("asr") for k in keys]
        return txt if all(txt) else None

    @torch.no_grad()
    def gen(tok, m, which, msgs, tools):
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt").to("cuda")
        kw = {"eos_token_id": [12, 11, 2]}
        if which == "vc":
            kw["min_new_tokens"] = 4          # id 12 = VoiceChat frame PAD: not an end before a few tokens
        t0 = time.monotonic()
        g = m.generate(ids, max_new_tokens=a.max_new, do_sample=False, pad_token_id=0, **kw)
        txt = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
        import re
        txt = re.split(r"<SPECIAL_1[12]>|</s>|<extra_id_1>", txt)[0]
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        if "</TOOLCALL>" in txt:
            txt = txt[: txt.index("</TOOLCALL>") + len("</TOOLCALL>")]
        return txt, round(time.monotonic() - t0, 2), int(ids.shape[1])

    status = {"done": [], "skipped": []}
    loaded = None
    durations = []
    for arm in [x for x in a.arms.split(",") if x]:
        which = "vc" if arm == "vc" else "nano"
        est = (max(durations) if durations else 200) + (90 if which != loaded else 0)
        if time.monotonic() - T0 + est > a.budget_s:
            status["skipped"].append(arm)
            print("BUDGET: skip", arm, flush=True)
            continue
        t_arm = time.monotonic()
        if which != loaded:
            if loaded is not None:
                del m
                torch.cuda.empty_cache()
            t0 = time.monotonic()
            tok, m, info = B2.load("vcsup" if which == "vc" else "nano")
            info["load_s"] = round(time.monotonic() - t0, 1)
            (out / f"info_{which}.json").write_text(json.dumps(info))
            loaded = which
            print("LOADED", which, info, flush=True)
        rows = out / f"rows_{arm}.jsonl"
        for it in items:
            turns = it["turns"]
            if arm == "nano_asr":
                turns = asr_turns(it)
                if turns is None:
                    continue
            random.seed(1234)
            system, tools, bound, backend, led = build(it["domain"], it["lang"], f"{arm}_{it['domain']}_{it['id']}_{it['lang']}")
            msgs = [{"role": "system", "content": system}]
            log, replies, calls = [], [], []
            for u in turns:
                if backend is not None:
                    backend._said.append(u)
                msgs.append({"role": "user", "content": u})
                for step in range(3):
                    txt, s, ctx = gen(tok, m, which, msgs, tools)
                    tc = B2.parse_toolcall(txt)
                    log.append({"gen": txt, "s": s, "ctx": ctx, "toolcall": tc if tc != "unparsable" else "unparsable"})
                    if isinstance(tc, list) and tc:
                        msgs.append({"role": "assistant", "content": txt})
                        for c in tc[:2]:
                            name, args = c.get("name"), c.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:  # noqa: BLE001
                                    args = {"_raw": args}
                            res = bound[name](args) if name in bound else json.dumps({"error": f"unknown tool {name}"})
                            calls.append({"name": name, "args": args, "result": res, "executed": name in bound})
                            msgs.append({"role": "tool", "content": res})
                        continue
                    msgs.append({"role": "assistant", "content": txt})
                    replies.append(txt)
                    break
            rec = {"arm": arm, "domain": it["domain"], "scenario": it["id"], "lang": it["lang"], "turns": turns,
                   "calls": calls, "agent_text": " ".join(replies), "replies": replies, "log": log,
                   "audit": backend.audit if backend is not None else [],
                   "ledger": [{"name": c.name, "status": c.status.value} for c in led.calls.values()],
                   "gold": it.get("gold")}
            with open(rows, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(arm, it["lang"], it["domain"], it["id"], "| calls", [(c["name"], c["args"]) for c in calls],
                  "|", rec["agent_text"][:140], flush=True)
        durations.append(time.monotonic() - t_arm)
        status["done"].append([arm, round(durations[-1], 1)])
    if a.two_llm_asr_dir:
        from e1.runner import mem_available_gb
        from e1 import two_llm
        left = a.budget_s - (time.monotonic() - T0)
        if left < a.two_llm_need_s:
            status["skipped"].append("two_llm")
            print("BUDGET: skip two_llm", round(left), flush=True)
        else:
            # engine FIRST (it loaded in ~55 s alone; with Nano resident it ran past 120 s in e1d-002), then Nano
            loads = {"mem_before_engine_gb": round(mem_available_gb(), 1)}
            print("MEM before engine", loads["mem_before_engine_gb"], "GB", flush=True)
            from mynah_v2.model.reference import ReferenceEngine, build_cfg
            t0 = time.monotonic()
            eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")), llm_cache=True)
            loads["engine_load_s"] = round(time.monotonic() - t0, 1)
            loads["mem_after_engine_gb"] = round(mem_available_gb(), 1)
            print("LOADED engine", loads, flush=True)
            if loaded is None:
                if loads["mem_after_engine_gb"] < 30:
                    raise SystemExit(f"not enough memory for Nano: {loads['mem_after_engine_gb']} GB")
                t0 = time.monotonic()
                tok, m, info = B2.load("nano")
                loads["nano_load_s"] = round(time.monotonic() - t0, 1)
                loaded = "nano"
            loads["mem_after_nano_gb"] = round(mem_available_gb(), 1)
            print("LOADED nano", loads, flush=True)
            (out / "loads.json").write_text(json.dumps(loads, indent=1))
            rec = two_llm.run_session(out, eng, tok, m, gen, B2.parse_toolcall, spec, POLICY["it"],
                                      Path(a.two_llm_asr_dir), a.two_llm_cut_s, lab / "agentbench", tool_schemas,
                                      fixes=a.two_llm_fixes)
            status["done"].append(["two_llm", rec["wall_s"]])
            print("DONE two_llm", rec["wall_s"], "s |", json.dumps(rec["messages"], ensure_ascii=False)[:1500],
                  flush=True)
    status["elapsed_s"] = round(time.monotonic() - T0, 1)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    print("STATUS", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
