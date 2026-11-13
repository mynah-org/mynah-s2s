"""Bit-identity check between two multi-session gate runs (rendezvous_pair.py with V2_SAVE_ARTEFACTS=1).

Compares the EarTTS audio codes saved in each run's codes.npz, per stream (solo__<sid>, pair__<sid>, FC streams), and
prints `=` when identical or the first differing code row and both lengths. This is the oracle used to promote a
change as bit-exact: run the same group (e.g. m4_notools) before and after the change with the same preset.
Sessions with tool calls are not a valid bit oracle (they already differ run to run in the baseline).

usage: python v2/scripts/tools/compare_codes.py RUN_A:RUN_B [RUN_C:RUN_D ...]
"""
import sys

import numpy as np


def load(run: str) -> dict:
    z = np.load(run.rstrip("/") + "/codes.npz")
    return {k: z[k] for k in z.files if not k.endswith("t_ms") and not k.endswith("__call")}


def compare(a: str, b: str) -> tuple[bool, str]:
    A, B = load(a), load(b)
    out, same = [], True
    for k in sorted(set(A) | set(B)):
        x, y = A.get(k), B.get(k)
        if x is None or y is None:
            out.append(f"{k}:missing-in-{'A' if x is None else 'B'}")
            same = False
            continue
        n = min(len(x), len(y))
        first = next((i for i in range(n) if not np.array_equal(x[i], y[i])), None)
        if x.shape == y.shape and first is None:
            out.append(f"{k}:=")
        else:
            same = False
            out.append(f"{k}:diff@{first} len {len(x)}/{len(y)}")
    return same, " ".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for pair in sys.argv[1:]:
        a, b = pair.split(":", 1)
        same, line = compare(a, b)
        print(("IDENTICAL " if same else "DIFFERENT ") + f"{a} vs {b} | {line}")
        rc |= 0 if same else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
