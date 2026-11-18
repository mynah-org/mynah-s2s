"""CPU probe of the text LLM's tool decision (one forward per variant, no GPU): why the catalogue question gets a
spoken "I'm searching" instead of a search_catalog call, and which prompt fix makes the decision step say TOOL.

For each (question, variant) it builds the worker's exact decision context (system prompt as the server builds it
+ tools block + conversation + the internal "TOOL or REPLY?" question) and reads the TOOL vs REPLY logits of the
first answer token. Variants: the system prompt as in the runs, + the per-language tool rule (llm/nano.py
TOOL_RULE), the decision question with the tool list, the user text cleaned (capital, punctuation).

usage: python tool_step_cpu.py --lab-scripts DIR --nano-dir DIR --out out.json [--threads 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frankenstein"))

QUESTIONS = {
    "it_catalog_asr": ("it", "E cerco un divano a tre posti , cosa avete"),
    "it_catalog_clean": ("it", "Cerco un divano a tre posti, cosa avete?"),
    "it_order": ("it", "buongiorno, vorrei sapere quando arriva il mio ordine. Mi chiamo Mario Rossi e il mio cap è 10121"),
    "it_oos": ("it", "può prenotare un tavolo al ristorante per stasera"),
    "en_catalog": ("en", "I'm looking for a three-seat sofa, what do you have?"),
    "fr_catalog": ("fr", "Je cherche un canapé trois places, qu'est-ce que vous avez ?"),
}


def main():  # pragma: no cover - CPU model run on the GPU host
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-scripts", required=True)
    ap.add_argument("--nano-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    import contextlib
    torch.cuda.stream = lambda *x, **k: contextlib.nullcontext()
    torch.cuda.default_stream = lambda *x, **k: None
    torch.set_num_threads(a.threads)
    lab = Path(a.lab_scripts)
    sys.path.insert(0, str(lab / "agentbench"))
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    from e1.tools_bench import POLICY as P1, tool_schemas
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein2.llm.nano import BRIEF, POLICY, STEP_Q, TOOL_RULE, step_question
    from transformers import AutoModelForCausalLM, AutoTokenizer
    policy = {**P1, **POLICY}
    spec = json.load(open(lab / "agentbench" / "tenant_furniture.json"))
    reg = ToolRegistry()
    be = TenantBackend(spec)
    for t in spec["tools"]:
        reg.register(t["name"], be.handlers()[t["name"]], t["description"], t["parameters"])
    tools = tool_schemas(reg)
    tok = AutoTokenizer.from_pretrained(a.nano_dir)
    t0 = time.monotonic()
    m = AutoModelForCausalLM.from_pretrained(a.nano_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                             device_map="cpu").eval()
    print("loaded", round(time.monotonic() - t0, 1), flush=True)

    def gated_rmsnorm(self, hidden_states, gate=None):      # the Triton rmsnorm_fn needs a GPU: same math in torch
        x = hidden_states.float()
        if gate is not None:
            x = x * torch.nn.functional.silu(gate.float())
        g = self.group_size
        xs = x.view(*x.shape[:-1], x.shape[-1] // g, g)
        xs = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (xs.view_as(x) * self.weight.float()).to(hidden_states.dtype)
    for mod in m.modules():
        if type(mod).__name__ == "MambaRMSNormGated":
            mod.forward = gated_rmsnorm.__get__(mod)
    tid = [tok.encode(w, add_special_tokens=False)[0] for w in ("TOOL", " TOOL")]
    rid = [tok.encode(w, add_special_tokens=False)[0] for w in ("REPLY", " REPLY")]
    rows = []
    for qname, (lang, user) in QUESTIONS.items():
        L = session_language(lang)
        base = with_language(spec["system_prompt"], L) + "\n" + policy.get(lang, policy["en"]) + "\n" + \
            BRIEF.get(lang, BRIEF["en"])
        variants = {
            "as_run": (base + " /no_think", STEP_Q),
            "tool_rule": (base + "\n" + TOOL_RULE.get(lang, TOOL_RULE["en"]) + " /no_think", STEP_Q),
            "tool_rule_step_lang": (base + "\n" + TOOL_RULE.get(lang, TOOL_RULE["en"]) + " /no_think",
                                    step_question(lang)),
        }
        for vname, (system, q) in variants.items():
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user},
                    {"role": "user", "content": q}]
            ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt")
            t1 = time.monotonic()
            with torch.no_grad():
                lg = m(ids).logits[0, -1].float()
            row = {"q": qname, "variant": vname, "ctx": int(ids.shape[1]),
                   "tool": round(max(float(lg[i]) for i in tid), 3), "reply": round(max(float(lg[i]) for i in rid), 3),
                   "s": round(time.monotonic() - t1, 1)}
            row["decision"] = "TOOL" if row["tool"] > row["reply"] else "REPLY"
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
