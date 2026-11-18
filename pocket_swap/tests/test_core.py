"""Pure-python tests (no torch): RefLRU, SlotPool, SwapCore session binding with a fake backend."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_swap.core import SwapCore, TierSpec  # noqa: E402
from pocket_swap.lru import AllSlotsBusy, CacheFull, RefLRU, SlotPool  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeBackend:
    def __init__(self):
        self.reads, self.allocs, self.frees, self.loads, self.voices = [], [], [], [], []
        self.slot_lang: dict[int, str] = {}
        self.fail_on: str | None = None
        self.conversions: list = []

    def read_package(self, lang):
        self.reads.append(lang)
        return {"lang": lang}, 100

    def to_tier(self, pkg, tier):
        self.conversions.append((pkg["lang"], tier))
        return dict(pkg, tier=tier), 100

    def allocate_slot(self, i):
        self.allocs.append(i)

    def free_slot(self, i):
        self.frees.append(i)
        self.slot_lang.pop(i, None)

    def materialise(self, i, lang, pkg):
        assert pkg["lang"] == lang
        if lang == self.fail_on:
            raise RuntimeError("boom")
        self.loads.append((i, lang))
        self.slot_lang[i] = lang

    def voice_state(self, i, lang, voice):
        assert self.slot_lang[i] == lang
        self.voices.append((lang, voice))
        return f"state:{lang}:{voice}"


# -- RefLRU -------------------------------------------------------------------------------------------------
def test_lru_order_and_byte_cap():
    ev = []
    c = RefLRU(max_bytes=300, on_evict=lambda k, v: ev.append(k))
    for k in "abc":
        c.put(k, k.upper(), 100)
    assert c.get("a") == "A"  # a becomes most recent
    c.put("d", "D", 100)
    assert ev == ["b"] and c.keys() == ["c", "a", "d"]


def test_lru_pinned_entries_never_evicted():
    c = RefLRU(max_items=2)
    c.put("a", 1, 1)
    c.put("b", 2, 1)
    c.acquire("a")
    c.acquire("b")
    with pytest.raises(CacheFull):
        c.put("c", 3, 1)
    assert c.keys() == ["a", "b"]
    c.release("b")
    c.put("c", 3, 1)
    assert "b" not in c and "a" in c
    with pytest.raises(RuntimeError):
        c.pop("a")


def test_lru_idle_eviction():
    clk = Clock()
    c = RefLRU(idle_s=10, clock=clk)
    c.put("a", 1, 1)
    c.put("b", 2, 1)
    c.acquire("b")
    clk.t = 11
    assert c.evict_idle() == ["a"]
    c.release("b")
    clk.t = 15
    assert c.evict_idle() == []
    clk.t = 21.5
    assert c.evict_idle() == ["b"]


def test_lru_release_without_acquire():
    c = RefLRU()
    c.put("a", 1, 1)
    with pytest.raises(RuntimeError):
        c.release("a")


# -- SlotPool -----------------------------------------------------------------------------------------------
def test_slotpool_events():
    clk = Clock()
    p = SlotPool(2, clock=clk)
    s, e = p.choose("it")
    assert e == "alloc"
    p.commit(s, "it", e)
    assert p.choose("it") == (s, "hot")
    s2, e2 = p.choose("fr")
    assert e2 == "alloc" and s2 is not s
    p.commit(s2, "fr", e2)
    clk.t = 5
    p.ref(s2, "x")
    s3, e3 = p.choose("en")
    assert (s3, e3) == (s, "evict")  # the unreferenced slot, even though it is not the oldest overall
    p.ref(s, "y")
    with pytest.raises(AllSlotsBusy):
        p.choose("en")


def test_slotpool_alloc_gen_and_free():
    clk = Clock()
    p = SlotPool(1, idle_s=3, clock=clk)
    s, e = p.choose("it")
    p.commit(s, "it", e)
    assert s.alloc_gen == 1 and s.load_gen == 1
    clk.t = 4
    assert p.idle_slots() == [s]
    p.free(s)
    s, e = p.choose("fr")
    assert e == "alloc"
    p.commit(s, "fr", e)
    assert s.alloc_gen == 2 and s.load_gen == 2


# -- SwapCore -----------------------------------------------------------------------------------------------
def make(n=4, **kw):
    be = FakeBackend()
    return be, SwapCore(be, n_slots=n, clock=kw.pop("clock", Clock()), **kw)


def test_same_language_shares_slot_and_loads_once():
    be, c = make()
    b1 = c.bind("s1", "it", voice="v1")
    b2 = c.bind("s2", "it", voice="v1")
    assert b1.slot == b2.slot and b1.event == "alloc" and b2.event == "hot" and b2.source == "vram"
    assert be.reads == ["it"] and be.loads == [(b1.slot, "it")]
    assert be.voices == [("it", "v1")]  # voice state cached per (language, voice)
    assert c.group_by_slot(["s1", "s2"]) == {b1.slot: ["s1", "s2"]}


def test_four_sessions_three_languages_grouping():
    be, c = make()
    c.bind("a", "it")
    c.bind("b", "fr")
    c.bind("c", "en")
    c.bind("d", "it")
    g = c.group_by_slot(["a", "b", "c", "d"])
    assert sorted(len(v) for v in g.values()) == [1, 1, 2]
    assert len(set(c.resident().values()) - {None}) == 3


def test_switch_one_session_does_not_disturb_others():
    be, c = make(n=2)
    a = c.bind("a", "it", voice="va")
    c.bind("b", "it", voice="vb")
    loads_before = list(be.loads)
    sw = c.bind("a", "fr")  # keep voice? core needs voice passed; the swapper layer re-passes it
    assert sw.slot != a.slot and c.binding("b").slot == a.slot
    assert be.loads == loads_before + [(sw.slot, "fr")]
    assert c.pool.slots[a.slot].refs == 1 and c.pool.slots[sw.slot].refs == 1


def test_switch_rebuilds_voice_state_for_new_language():
    be, c = make()
    c.bind("a", "it", voice="v")
    b = c.bind("a", "fr", voice="v")
    assert b.voice_state == "state:fr:v" and ("fr", "v") in be.voices


def test_never_evicts_slot_in_use_and_waits_or_raises():
    be, c = make(n=2)
    c.bind("a", "it")
    c.bind("b", "fr")
    with pytest.raises(AllSlotsBusy):
        c.bind("c", "en", timeout=0)
    assert c.resident() == {0: "it", 1: "fr"}
    done = {}

    def waiter():
        done["b"] = c.bind("c", "en", timeout=5)

    t = threading.Thread(target=waiter)
    t.start()
    c.unbind("b")
    t.join(5)
    assert done["b"].event == "evict" and done["b"].slot == 1 and c.resident()[0] == "it"


def test_sole_user_can_switch_when_all_slots_busy():
    be, c = make(n=1)
    c.bind("a", "it")
    b = c.bind("a", "fr", timeout=0)
    assert b.event == "evict" and c.resident() == {0: "fr"}


def test_ram_tier_hits_after_vram_eviction():
    be, c = make(n=1, tiers=[TierSpec("pinned", max_items=4)])
    c.bind("a", "it")
    c.unbind("a")
    c.bind("a", "fr")
    c.unbind("a")
    b = c.bind("a", "it")
    assert b.source == "pinned" and be.reads == ["it", "fr"]


def test_ram_cap_zero_serves_from_disk_every_time():
    be, c = make(n=1, tiers=[TierSpec("pinned", max_bytes=50)])  # packages are 100 B: never kept
    c.bind("a", "it")
    c.unbind("a")
    c.bind("a", "fr")
    c.unbind("a")
    b = c.bind("a", "it")
    assert b.source == "disk" and be.reads == ["it", "fr", "it"]


def test_idle_maintenance_frees_only_unused_slots():
    clk = Clock()
    be, c = make(n=2, clock=clk, slot_idle_s=10, tiers=[TierSpec("pinned", max_items=4, idle_s=20)])
    c.bind("a", "it")
    c.bind("b", "fr")
    c.unbind("b")
    clk.t = 11
    r = c.maintenance()
    assert r["slots_freed"] == [1] and be.frees == [1] and c.resident() == {0: "it", 1: None}
    clk.t = 25
    assert c.maintenance()["host_evicted"]["pinned"] == ["it", "fr"]  # RAM copies idle even if "it" is in VRAM
    b = c.bind("b", "fr")
    assert b.event == "alloc" and b.source == "disk" and b.alloc_gen == 2


def test_failed_materialise_leaves_no_language_in_slot():
    be, c = make(n=1)
    c.bind("a", "it")
    c.unbind("a")
    be.fail_on = "fr"
    with pytest.raises(RuntimeError):
        c.bind("a", "fr")
    assert c.resident() == {0: None} and "a" not in c.sessions
    be.fail_on = None
    assert c.bind("a", "it").event == "reuse"


def test_host_tiers_demote_and_promote():
    be, c = make(n=1, tiers=[TierSpec("pinned", max_items=1), TierSpec("ram", max_items=1)])
    for lang in ("it", "fr", "en"):
        c.bind("a", lang)
    # it: pinned -> ram (when fr came) -> dropped (when en came); fr: pinned -> ram; en: pinned
    assert c.host_resident() == {"pinned": ["en"], "ram": ["fr"]}
    b = c.bind("a", "fr")
    assert b.source == "ram" and c.host_resident() == {"pinned": ["fr"], "ram": ["en"]}
    b = c.bind("a", "it")
    assert b.source == "disk" and be.reads == ["it", "fr", "en", "it"]
    assert ("fr", "pinned") in be.conversions and ("en", "ram") in be.conversions


def test_no_host_tiers_means_disk_every_time():
    be, c = make(n=1, tiers=[])
    c.bind("a", "it")
    c.bind("a", "fr")
    assert c.bind("a", "it").source == "disk" and be.reads == ["it", "fr", "it"]


def test_idle_pinned_entry_is_demoted_to_ram():
    clk = Clock()
    be, c = make(n=1, clock=clk, tiers=[TierSpec("pinned", idle_s=5), TierSpec("ram")])
    c.bind("a", "it")
    clk.t = 6
    c.maintenance()
    assert c.host_resident() == {"pinned": [], "ram": ["it"]}


class SlowBackend(FakeBackend):
    """Materialise takes time and writes the slot word by word, so a concurrent reader would see a torn slot."""

    def __init__(self, width=50):
        super().__init__()
        self.width = width
        self.weights: dict[int, list] = {}

    def allocate_slot(self, i):
        super().allocate_slot(i)
        self.weights[i] = [None] * self.width

    def materialise(self, i, lang, pkg):
        import time as _t
        for j in range(self.width):
            self.weights[i][j] = lang
            if j % 10 == 0:
                _t.sleep(0.0005)
        super().materialise(i, lang, pkg)


def test_concurrent_sessions_it_en_fr_isolated_under_loads_and_evictions():
    """Generators read their slot continuously while other sessions bind, switch and get evicted."""
    import random
    import time as _t

    be = SlowBackend()
    c = SwapCore(be, n_slots=3, tiers=[TierSpec("pinned", max_items=2)])
    langs = ["it", "en", "fr", "de", "es"]
    errors, stop = [], threading.Event()

    def generator(session, lang):
        try:
            b = c.bind(session, lang, voice=f"v-{session}", timeout=10)
            state = b.voice_state
            for _ in range(300):  # a "generation": every step reads the whole slot
                assert c.binding(session).slot == b.slot
                w = be.weights[b.slot]
                assert all(x == lang for x in w), (session, lang, set(w))
                assert c.binding(session).voice_state is state
            c.unbind(session)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    def churner(k):
        rnd = random.Random(k)
        while not stop.is_set():
            lg = rnd.choice(langs)
            try:
                c.bind(f"churn{k}", lg, timeout=0)
                _t.sleep(0.0002)
                c.unbind(f"churn{k}")
            except Exception:  # AllSlotsBusy is expected while generators hold slots
                pass

    ch = [threading.Thread(target=churner, args=(k,)) for k in range(2)]
    [t.start() for t in ch]
    for round_ in range(5):
        gens = [threading.Thread(target=generator, args=(f"s{round_}{i}", lg))
                for i, lg in enumerate(["it", "en", "fr", "it"])]
        [t.start() for t in gens]
        [t.join(30) for t in gens]
    stop.set()
    [t.join(5) for t in ch]
    assert not errors, errors[:3]
    assert all(s.refs >= 0 for s in c.pool.slots)
    assert c.stats.slot_evictions > 0  # churners really forced reloads while generators ran


def test_mixed_language_batch_grouped_per_slot():
    be, c = make(n=4)
    plan = {"c1": "it", "c2": "en", "c3": "fr", "c4": "it", "c5": "en"}
    for s, lg in plan.items():
        c.bind(s, lg)
    g = c.group_by_slot(list(plan))
    for slot, members in g.items():
        assert len({plan[m] for m in members}) == 1  # one language per group
    assert sorted(sorted(v) for v in g.values()) == [["c1", "c4"], ["c2", "c5"], ["c3"]]
