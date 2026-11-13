"""labpaths: defaults equal the historical layout; MYNAH_HOME / MYNAH_LAB_HOME override it (no symlink needed)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = "import labpaths as L; print(L.MYNAH_HOME, L.LAB_HOME, L.HF_CACHE, sep='|')"


def run(env):
    e = {k: v for k, v in os.environ.items() if k not in ("MYNAH_HOME", "MYNAH_LAB_HOME")}
    e.update(env)
    return subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=e, capture_output=True, text=True, check=True).stdout.strip().split("|")


def test_defaults_are_the_historical_layout():
    h = Path.home() / "mynah-s2s"
    assert run({}) == [str(h), str(h / "ml"), str(h / "hf-cache")]


def test_env_overrides():
    assert run({"MYNAH_HOME": "/x/root"}) == ["/x/root", "/x/root/ml", "/x/root/hf-cache"]
    assert run({"MYNAH_HOME": "/x/root", "MYNAH_LAB_HOME": "/archive/ml"}) == ["/x/root", "/archive/ml", "/x/root/hf-cache"]


def test_no_hard_coded_home_layout_left():
    hits = []
    for d in (HERE, HERE.parent / "agentbench"):
        for f in d.glob("*.py"):
            if f.name in ("labpaths.py", "test_labpaths.py"):
                continue
            for i, ln in enumerate(f.read_text().splitlines(), 1):
                if "Path.home()" in ln and not ln.startswith("HOME = Path.home()") or '"mynah-s2s/' in ln:
                    hits.append(f"{f.name}:{i}")
    assert hits == [], hits
