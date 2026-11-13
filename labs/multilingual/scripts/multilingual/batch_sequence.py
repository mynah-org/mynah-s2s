"""Reconstruct the deterministic training batch sequence of session_distill train (CPU): the kept train keys
(kept_keys over the teacher items, usable manifest keys and splits) and random.Random(1).sample(tr, batch) per
micro-batch, as the training loop draws them (no other consumer of that RNG before the loop when --smoke-first is
not used). Usage: batch_sequence.py --prep DIR --data DIR --steps 120 --accum 2 --batch 4 --out seq.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_distill import kept_keys, usable_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--langs", default="en,it,fr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.data) / "manifest.jsonl"))}
    split = {r["id"]: r["split"] for r in man.values()}
    tr, va = kept_keys(prep["items"], usable_keys(man), split, a.langs.split(","))
    rng = random.Random(1)
    seq = [[f"{i}|{l}" for i, l in rng.sample(tr, a.batch)] for _ in range(a.steps * a.accum)]
    h = hashlib.sha256(json.dumps(seq).encode()).hexdigest()
    Path(a.out).write_text(json.dumps({"n_train": len(tr), "sha256": h, "seq": seq}))
    print("train keys", len(tr), "micro-batches", len(seq), "sha256", h[:16], "first", seq[0])


if __name__ == "__main__":
    main()
