"""The talker's language folder: which languages exist and where their packages are, found by scanning (no config).

Layout (docs/frankenstein2-languages.md):

    <root>/
      en/
        language.json
        24l/base/                the English 24-layer model verbatim (exact): the shared base of the 24l deltas
        6l/base/                 the English 6-layer model: the base of the 6l deltas
      it/
        language.json            name, ASR locale, fixed phrases (registered in frankenstein2.langs)
        24l/int4/ 24l/int8/ 24l/bf16/     deltas against en/24l/base (pocket_swap format)
        6l/int4/ ...                      deltas against en/6l/base
      cs/6l/int4/ ...            a language that only exists as a 6-layer model

A language is served by the first family in ``families`` (default 24l, then 6l) that has the wanted precision, or
``base`` (the base language). A code the catalog does not know triggers a re-scan (at most every ``rescan_s``), so a
folder copied in while the server runs is used by the next session that asks for it. ``<code>@<n>`` folders are
copies of a language (lab).
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import langs

BASE_CODE = "en"


class PackageCatalog:
    def __init__(self, root, precision: str = "int4", families=("24l", "6l"), rescan_s: float = 5.0,
                 clock=time.monotonic):
        self.root, self.precision, self.families = Path(root), precision, tuple(families)
        self.rescan_s, self.clock = rescan_s, clock
        self.entries: dict = {}            # name ("it", "it@03") -> (package dir, family)
        self.last_scan = None
        self.scans = 0
        self.scan()

    def _pick(self, d: Path):
        for fam in self.families:
            for prec in (self.precision, "base"):
                p = d / fam / prec
                if (p / "manifest.json").exists():
                    return p, fam
        return None

    def scan(self) -> list:
        """Re-read the folder; -> the names found. Registers each language.json (codes without '@')."""
        found = {}
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                if not d.is_dir():
                    continue
                hit = self._pick(d)
                if hit is not None:
                    found[d.name] = hit
                lj = d / "language.json"
                if "@" not in d.name and lj.exists():
                    try:
                        langs.register(langs.from_json(lj))
                    except (ValueError, KeyError):
                        pass                    # a broken language.json: the built-in values stay
        self.entries = found
        self.last_scan = self.clock()
        self.scans += 1
        return list(found)

    def _get(self, name):
        e = self.entries.get(name)
        if e is None and (self.last_scan is None or self.clock() - self.last_scan >= self.rescan_s):
            self.scan()
            e = self.entries.get(name)
        return e

    def path(self, name: str):
        """The package dir serving ``name``, re-scanning once if it is unknown; None if there is none."""
        e = self._get(name)
        return None if e is None else e[0]

    def family(self, name: str):
        e = self._get(name)
        return None if e is None else e[1]

    def base_dir(self, family: str):
        """The shared base backbone package of a family (``en/<family>/base``)."""
        p = self.root / BASE_CODE / family / "base"
        return p if (p / "manifest.json").exists() else None

    def languages(self) -> list:
        """Codes served (copies excluded); re-scans if the last scan is older than ``rescan_s``."""
        if self.last_scan is None or self.clock() - self.last_scan >= self.rescan_s:
            self.scan()
        return sorted(n for n in self.entries if "@" not in n)
