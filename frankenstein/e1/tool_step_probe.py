"""E1f CPU probe: why did Nano stop calling tools inside the two-LLM session, and which fix helps?

Replays the EXACT message lists of the e1e session (runs/e1e_001/two_llm_it/summary.json) at the two decision points
where a tool was needed (U1 catalog request, U2 order status) with the same system prompt, tools block and chat
template as the session, on CPU (Nano-9B-v2, one forward per variant, no generation loop), and reads the first
reply token: '<' (the first token of <TOOLCALL>) means "call", anything else means "reply". Variants:
  exact        the session history as it was
  no_opener    without the "[call connected]" turn and its greeting
  no_promise   earlier assistant turns that promised to look something up without a call are left out
  tool_step    an explicit decision step: the same context plus an internal question "tool or reply?", read as
               the logit of TOOL vs REPLY (the call itself would then be generated with the tools block)
Output: one JSON per (turn, variant) with p('<'), the top-5 first tokens, and for tool_step the TOOL/REPLY logits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STEP_Q = ("[Controllo interno, non dal cliente] Per rispondere all'ultima richiesta del cliente devi prima chiamare "
          "uno strumento? Rispondi con una sola parola: TOOL oppure REPLY.")


def variants(msgs_before: list, user: str) -> dict:
    exact = msgs_before + [{"role": "user", "content": user}]
    no_open = [m for m in msgs_before if m["content"] != "[call connected]"]
    if no_open and no_open[0]["role"] == "assistant":
        no_open = no_open[1:]
    promise = ("un momento", "cerco", "cercare", "verifico", "controllo")
    no_prom = [m for i, m in enumerate(msgs_before)
               if not (m["role"] == "assistant" and any(p in m["content"].lower() for p in promise))]
    # keep user/assistant alternation: drop a user turn left without its answer only at the very end
    return {"exact": exact, "no_opener": no_open + [{"role": "user", "content": user}],
            "no_promise": no_prom + [{"role": "user", "content": user}],
            "tool_step": exact + [{"role": "assistant", "content": "…"}, {"role": "user", "content": STEP_Q}]}


def main():  # pragma: no cover - CPU model run on the GPU host
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--session", required=True, help="e1e two_llm_it/summary.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "agentbench"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from e1.tools_bench import POLICY, tool_schemas
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(16)
    if not torch.cuda.is_available():
        # the NemotronH remote code wraps its Mamba mixer in torch.cuda.stream(...) unconditionally: make that a
        # no-op on a CPU-only process (no numerical effect)
        import contextlib
        torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
        torch.cuda.default_stream = lambda *a, **k: None
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    reg = ToolRegistry()
    be = TenantBackend(spec)
    for t in spec["tools"]:
        reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
    tools = tool_schemas(reg)
    system = spec["system_prompt"] + "\n" + POLICY["it"] + " /no_think"
    msgs = json.load(open(a.session))["messages"]
    # decision points: the user turns U1 (index 2) and U2 (index 4) with everything before them
    points = {"U1_catalog": (msgs[:2], msgs[2]["content"]), "U2_order": (msgs[:4], msgs[4]["content"])}
    tok = AutoTokenizer.from_pretrained(a.nano_dir)
    t0 = time.monotonic()
    m = AutoModelForCausalLM.from_pretrained(a.nano_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                             device_map="cpu").eval()
    print("loaded", round(time.monotonic() - t0, 1), "s", flush=True)
    lt = tok.encode("<", add_special_tokens=False)[0]
    tool_ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("TOOL", " TOOL")}
    reply_ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("REPLY", " REPLY")}
    rows = []
    for point, (before, user) in points.items():
        for name, ms in variants(before, user).items():
            ids = tok.apply_chat_template([{"role": "system", "content": system}] + ms, tools=tools,
                                          add_generation_prompt=True, return_tensors="pt")
            t1 = time.monotonic()
            with torch.no_grad():
                lg = m(ids).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            top = torch.topk(p, 5)
            row = {"point": point, "variant": name, "ctx": int(ids.shape[1]), "p_toolcall_start": round(float(p[lt]), 4),
                   "top5": [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
                   "s": round(time.monotonic() - t1, 1)}
            if name == "tool_step":
                row["tool_logit"] = round(max(float(lg[i]) for i in tool_ids.values()), 3)
                row["reply_logit"] = round(max(float(lg[i]) for i in reply_ids.values()), 3)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
