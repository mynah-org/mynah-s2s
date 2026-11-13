"""Per-session timeline from a V2 trace: agent text segments, ASR, turn and function events by model step (80 ms)."""
import json
import sys

for path in sys.argv[1:]:
    ev = [json.loads(l) for l in open(path)]
    sess, cur = [], None
    for e in ev:
        if e["ev"] == "session.opened":
            cur = []
            sess.append(cur)
        cur.append(e)
    s = sess[0]
    print("=====", path, "sessions:", len(sess))
    rms_on = [e["audio_frame"] for e in s if e["ev"] == "audio.received" and e.get("rms", 0) > 0.003]
    if rms_on:
        print(f"  user speech frames (rms>0.003): audio_frame {rms_on[0]}..{rms_on[-1]} "
              f"= model_step {2 * rms_on[0]}..{2 * rms_on[-1] + 1} = {0.16 * rms_on[0]:.2f}..{0.16 * (rms_on[-1] + 1):.2f}s")
    text, t0 = [], None
    for e in s:
        st = e.get("model_step")
        if e["ev"] == "text.generated":
            text.append((st, e["text"]))
        elif e["ev"] in ("asr.generated", "turn.started", "function.started", "tool.pending", "tool.result.injected",
                         "tool.result.consumed"):
            extra = {k: e[k] for k in ("text", "who", "tool", "args", "turn_id") if k in e}
            print(f"  step {st:4} ({st * 0.08:6.2f}s) {e['ev']:22} {extra}")
    # agent text grouped by contiguous steps
    print("  agent text by step:")
    line, last = "", None
    for st, t in text:
        if last is not None and st - last > 6:
            print(f"    [{start:4}-{last:4} | {start * .08:5.2f}-{last * .08:5.2f}s] {line}")
            line = ""
        if not line:
            start = st
        line += t
        last = st
    if line:
        print(f"    [{start:4}-{last:4} | {start * .08:5.2f}-{last * .08:5.2f}s] {line}")
    # non-pad ids per thinker step (fc_active)
    fc = [e["model_step"] for e in s if e["ev"] == "thinker.step" and e.get("fc_active")]
    if fc:
        print(f"  fc_active steps {fc[0]}..{fc[-1]} ({len(fc)} step events)")
