"""E5 GO gate (CPU only). Passes (exit 0) only if:
  1. the dual-encoder parity run (ln=orig + a1) reproduces the single-encoder a1 rows exactly on the same clips
     (agent text, RNNT transcript, function text, first text frame);
  2. the adapter init hash is identical across two independent processes (=> equal to a1's init, see experiments.md);
  3. the smoke GO checks all passed.
Writes <out> (json verdict); never overwrites."""
import json
import sys
from pathlib import Path

R = Path.home() / "mynah-s2s/ml/results"
out = Path(sys.argv[1])
if out.exists():
    raise SystemExit(f"{out} exists")
ref = {(r["lang"], r["id"]): r for r in map(json.loads, open(R / "e2e-rv-dev/rows.jsonl"))}
par = [json.loads(l) for l in open(R / "e2e-rv-dual-parity/rows.jsonl")]
keys = ("agent_text", "user_asr", "func_text", "first_text_frame")
diff = [{"lang": p["lang"], "id": p["id"], "field": k, "ref": ref[(p["lang"], p["id"])][k], "dual": p[k]}
        for p in par for k in keys if p[k] != ref[(p["lang"], p["id"])][k]]
h = [json.load(open(R / f"distill/inithash-{k}/init_hash.json"))["init_hash"] for k in (1, 2)]
smoke = json.load(open(R / "distill/e5-smoke/smoke.json"))["go_checks"]
v = {"parity_clips": len(par), "parity_diffs": diff, "init_hashes": h, "smoke": smoke,
     "go": not diff and len(par) == 32 and h[0] == h[1] and all(smoke.values())}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(v, indent=1, ensure_ascii=False))
print(json.dumps({k: v[k] for k in ("parity_clips", "init_hashes", "smoke", "go")}), len(diff), "diffs")
sys.exit(0 if v["go"] else 1)
