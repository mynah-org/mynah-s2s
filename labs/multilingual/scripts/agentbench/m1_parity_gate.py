"""M1 parity gate (CPU): exit 0 only if
  (1) the front-end swap is clean: the original VC on m03 gives identical agent text, function text and tool calls
      before any swap and after restoring from the V2.1 front-end (same process, same seed);
  (2) the RNN-T branch is isolated: V2.1 (trained LN copy + adapter on the LLM path) and the untrained dual front-end
      (ln=orig, no adapter) produce the identical user RNN-T transcript (both read the frozen ML encoder).
Usage: m1_parity_gate.py <records.jsonl> <out.json>"""
import json
import sys
from pathlib import Path

recs = [json.loads(l) for l in open(sys.argv[1])]
m03 = [r for r in recs if r["scenario"] == "m03" and r["lang"] == "en"]
vc = [r for r in m03 if r["variant"] == "vc"]
by = {r["variant"]: r for r in m03 if r["variant"] != "vc"}
key = lambda r: (r["agent_text"], r["function_text"], json.dumps([(c["name"], c["args"]) for c in r["calls"]]))  # noqa: E731
res = {"vc_runs": len(vc),
       "swap_clean": len(vc) == 2 and key(vc[0]) == key(vc[-1]),
       "rnnt_identical_v21_vs_mlorig": "v21" in by and "mlorig" in by and by["v21"]["user_rnnt"] == by["mlorig"]["user_rnnt"],
       "rnnt": {k: r["user_rnnt"][:160] for k, r in by.items()}}
res["go"] = res["swap_clean"] and res["rnnt_identical_v21_vs_mlorig"]
out = Path(sys.argv[2])
if out.exists():
    raise SystemExit(f"{out} exists")
out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, ensure_ascii=False))
sys.exit(0 if res["go"] else 1)
