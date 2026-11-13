"""For every session trace: first agent text step vs first user speech frame; agent text overlapping user speech."""
import glob
import json
import sys

rows = []
for root in sys.argv[1:]:
    for path in sorted(glob.glob(f"{root}/*/trace.jsonl")):
        ev = [json.loads(l) for l in open(path)]
        sess, cur = [], None
        for e in ev:
            if e["ev"] == "session.opened":
                cur = []
                sess.append(cur)
            cur.append(e)
        s = sess[0]
        on = [e["audio_frame"] for e in s if e["ev"] == "audio.received" and e.get("rms", 0) > 0.003]
        u0 = 2 * on[0] if on else None
        u1 = 2 * on[-1] + 1 if on else None
        txt = [(e["model_step"], e["text"]) for e in s if e["ev"] == "text.generated" and e["text"].strip("</s> ")]
        first = txt[0][0] if txt else None
        before = "".join(t for st, t in txt if u0 is not None and st < u0)
        during = "".join(t for st, t in txt if u0 is not None and u0 <= st <= u1)
        fc = [e["model_step"] for e in s if e["ev"] == "function.started"]
        rows.append((path.split("/")[-3] + "/" + path.split("/")[-2], u0, u1, first, len(before), len(during), fc[:1]))
print(f"{'session':40} {'user_on':>7} {'user_off':>8} {'1st_txt':>7} {'chars<on':>8} {'chars_overlap':>13} first_fc")
for r in rows:
    print(f"{r[0]:40} {r[1]!s:>7} {r[2]!s:>8} {r[3]!s:>7} {r[4]:8} {r[5]:13} {r[6]}")
