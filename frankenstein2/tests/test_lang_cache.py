"""Talker language cache: concurrent loads (the caller never waits), LRU tiers, never evicting a language in use."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.tts.lang_cache import TalkerLangs  # noqa: E402


class Obj:
    def __init__(self, name):
        self.name, self.closed = name, False

    def close(self):
        self.closed = True


def make(gates=None, vram=2, host=3, fail=()):
    fetched, built = [], []

    def fetch(name):
        if gates and name in gates:
            gates[name].wait(5)
        if name in fail:
            raise OSError("no such package")
        fetched.append(name)
        return f"host:{name}"

    def build(name, h):
        built.append((name, h))
        return Obj(name)
    return TalkerLangs(fetch, build, vram_langs=vram, host_langs=host), fetched, built


def settle(c, names, in_use=lambda n: False, limit=3.0):
    t_end = time.monotonic() + limit
    while time.monotonic() < t_end:
        c.poll(in_use)
        if all(n in c.resident for n in names):
            return
        time.sleep(0.002)
    raise AssertionError(f"not resident: {names}, {c.stats()}")


def test_a_slow_load_never_blocks_the_caller_and_other_languages_stay_usable():
    gate = threading.Event()
    c, _, _ = make({"fr": gate})
    c.ensure("it")
    t0 = time.monotonic()
    assert c.want("fr") == "loading"                 # fr is stuck on the disk ...
    for _ in range(50):
        c.poll()
        assert c.get("it") is not None               # ... and it keeps serving
    assert time.monotonic() - t0 < 0.5
    gate.set()
    settle(c, ["fr"])
    assert c.stats()["loads"] == 2 and c.loads[-1]["wait_ms"] is not None


def test_lru_eviction_skips_languages_in_use_and_soft_caps():
    c, _, _ = make(vram=2)
    for n in ("it", "fr"):
        c.want(n)
    settle(c, ["it", "fr"])
    c.get("it")                                      # it is now most recent
    c.want("en")
    settle(c, ["en"], in_use=lambda n: n == "it")
    assert list(c.resident) == ["it", "en"] and c.evictions == 1     # fr: least recent and unused
    busy = {"it", "en"}
    c.want("de")
    settle(c, ["de"], in_use=lambda n: n in busy)
    assert len(c.resident) == 3                      # everything in use: over the cap rather than evict a live one
    busy.clear()
    c.poll(lambda n: False)
    assert len(c.resident) == 2


def test_reload_after_eviction_comes_from_the_host_tier():
    c, fetched, built = make(vram=1, host=2)
    for n in ("it", "fr", "it"):
        c.want(n)
        settle(c, [n])
    assert fetched == ["it", "fr"]                   # the second "it" did not touch the disk
    tiers = [x["tier"] for x in c.loads]
    assert tiers == ["disk", "disk", "host"] and c.stats()["by_tier"]["host"]["n"] == 1


def test_host_tier_is_bounded():
    c, fetched, _ = make(vram=1, host=1)
    for n in ("it", "fr", "it"):
        c.want(n)
        settle(c, [n])
    assert fetched == ["it", "fr", "it"] and c.host_evictions == 2


def test_a_broken_language_fails_alone():
    c, _, _ = make(fail={"xx"})
    c.want("xx")
    c.want("it")
    settle(c, ["it"])
    for _ in range(100):
        c.poll()
        if "xx" in c.failed:
            break
        time.sleep(0.002)
    assert c.want("xx").startswith("failed") and c.get("it") is not None
