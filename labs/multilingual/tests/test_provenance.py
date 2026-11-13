"""PROVENANCE.tsv is the migration record: every lab file with its source blob, sha256 and status. This check fails if a
file marked identical differs from its source, if an edited file has no reason, or if any file under scripts/ is
missing from the record or has changed since the migration without its row being updated."""
import csv
import hashlib
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def rows():
    with open(LAB / "PROVENANCE.tsv", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_every_file_recorded_and_unchanged():
    rec = {r["dest"]: r for r in rows() if r["status"] != "not-migrated"}
    files = sorted(p.relative_to(LAB).as_posix() for p in (LAB / "scripts").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
    assert files == sorted(rec), set(files) ^ set(rec)
    for f in files:
        assert hashlib.sha256((LAB / f).read_bytes()).hexdigest() == rec[f]["dest_sha256"], f


def test_status_is_consistent():
    for r in rows():
        if r["status"] == "identical":
            assert r["source_sha256"] == r["dest_sha256"], r["dest"]
        elif r["status"] == "edited":
            assert r["source_sha256"] != r["dest_sha256"] and r["note"], r["dest"]
        elif r["status"] == "new":
            assert r["source_git_blob"] == "-" and r["note"], r["dest"]
        else:
            assert r["status"] == "not-migrated" and r["note"], r
