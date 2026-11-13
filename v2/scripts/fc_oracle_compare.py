"""#9 oracle for the tokens INSIDE a tool call: NVIDIA's offline/training-format FC inference vs our live path.

The oracle is produced by ``scripts/oracle_offline.py --fc`` (NVIDIA ``run_fc_offline_inference``: pass 1 detects the
call on the full input, pass 2 re-runs with the call and the response inserted on the timeline, gap 0 — the format the
model was trained on). It shares no code with V2 except the checkpoint, so it is an independent oracle — but not a
token-identity oracle for the live path: the live reference pipeline conditions the call on silence embeddings,
injects the response at LLM speed and rewrites numbers to words (``fc_convert_num_to_text``).

This script runs the live path (REFERENCE session, deterministic, realtime pace, same prompt = NVIDIA default
system message + tools, the tool answering the SAME fixed payload as the oracle) and compares:
  call name + arguments, SOTC frame, response accepted (EOTR right after the forced response), and whether the agent
  text after the call voices the result value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from mynah_v2.model import tokens as T
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import Mode, Session, run_paced
from mynah_v2.tracing.events import Tracer

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import fc_timeline_from_records, load_16k, render_prompt  # noqa: E402

NUM_WORDS = {"42": "forty-two"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--oracle", required=True, help="oracle.json from oracle_offline.py --fc")
    ap.add_argument("--api-response", required=True, help='{"tool_name": ..., "response": {...}} (same as the oracle)')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    api = json.load(open(a.api_response))
    orc = json.load(open(a.oracle))
    import offline_voicechat_fc_infer as fcmod
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    fixed = json.dumps(api["response"])
    eng.pipeline.tool_registry[api["tool_name"]] = lambda args: fixed
    tr = Tracer(out / "trace.jsonl", session_id="fc_oracle")
    s = Session(eng, tr, Mode.REFERENCE)
    s.open(render_prompt(fcmod.DEFAULT_TOOLS))
    run_paced(s, load_16k(a.wav), pace="realtime", tail_silence_s=8)
    recs = list(eng.fc_records)
    tt = eng.token_tensors()
    s.close()
    tr.close()
    tl = fc_timeline_from_records(recs)
    live_calls = [T.parse_call_text("".join(s.func_text))] if s.func_text else []
    p1 = next((r for r in recs if r.phase == 1), None)
    p2 = next((r for r in recs if r.phase == 2), None)
    live_call = None
    ftxt = "".join(s.func_text)
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", ftxt, re.S)
    if m:
        parsed = T.parse_call_text(m.group(0))
        if parsed and parsed[0].ok:
            live_call = {"name": parsed[0].name, "arguments": parsed[0].arguments}
    oc = orc.get("fc_output", {}).get("tool_calls", [])
    oracle_call = {"name": oc[0]["name"], "arguments": oc[0]["arguments"]} if oc else None
    live_text = "".join(s.text)
    after_live = live_text.split("</s>")[-1] if live_text else ""
    val = str(next(iter(api["response"].values())))
    voiced_live = val in live_text or NUM_WORDS.get(val, "@@") in live_text.lower()
    voiced_oracle = val in orc.get("text", "") or NUM_WORDS.get(val, "@@") in orc.get("text", "").lower()
    res = {
        "oracle_call": oracle_call, "live_call": live_call,
        "call_identical": oracle_call == live_call and oracle_call is not None,
        "oracle_sotc_frame": orc.get("call_step"), "live_sotc_frame": (tl["calls"][0]["start_frame"] if tl["calls"] else None),
        "live_timeline_violations": [c["violations"] for c in tl["calls"]],
        "live_response_steps": p2.steps if p2 else None, "live_response_text": (p2.tool_response_text or "")[:200] if p2 else None,
        "oracle_response_tokens": orc.get("resp_token_len"),
        "result_value": val, "voiced_live": voiced_live, "voiced_oracle": voiced_oracle,
        "oracle_text": orc.get("text"), "live_text": live_text, "live_after_call": after_live,
        "note": "oracle = NVIDIA offline training-format FC inference (independent code); live = NVIDIA streaming "
                "pipeline driven by V2. Not expected token-identical (silence conditioning, num-to-words, timing).",
    }
    (out / "fc_oracle.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("oracle_call", "live_call", "call_identical", "oracle_sotc_frame",
                                          "live_sotc_frame", "live_timeline_violations", "voiced_live",
                                          "voiced_oracle")}, ensure_ascii=False), flush=True)
    print("ORACLE_TEXT", res["oracle_text"], flush=True)
    print("LIVE_TEXT", live_text, flush=True)


if __name__ == "__main__":
    main()
