"""Lab paths (owner 2026-11-13, after M4): no hard-coded home layout.

MYNAH_HOME      the engine's root (hf-cache, third_party, data), same variable and default as the engine.
MYNAH_LAB_HOME  the lab's root (data, ckpt, results): default $MYNAH_HOME/ml, the historical layout.
With neither variable set, every path is exactly what it was before (~/mynah-s2s and ~/mynah-s2s/ml).
"""
from __future__ import annotations

import os
from pathlib import Path

MYNAH_HOME = Path(os.environ.get("MYNAH_HOME") or Path.home() / "mynah-s2s")
LAB_HOME = Path(os.environ.get("MYNAH_LAB_HOME") or MYNAH_HOME / "ml")
HF_CACHE = MYNAH_HOME / "hf-cache"

_HISTORICAL = "/mynah-s2s/ml/"


def resolve(p) -> Path:
    """A data path as stored in the lab's manifests / symlinks. Archived corpora reuse clips through ABSOLUTE symlinks
    into the historical layout (``<home>/mynah-s2s/ml/...``); when that target does not exist (another machine, an
    archive copy) the same file is looked up under ``LAB_HOME``. Existing paths are returned unchanged."""
    p = Path(p)
    if p.exists():
        return p
    target = os.readlink(p) if p.is_symlink() else str(p)
    i = target.find(_HISTORICAL)
    if i >= 0:
        alt = LAB_HOME / target[i + len(_HISTORICAL):]
        if alt.exists():
            return alt
    return p
