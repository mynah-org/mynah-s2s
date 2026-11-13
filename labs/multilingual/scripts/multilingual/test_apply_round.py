import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_round as A  # noqa: E402


def _row(i, lang, ok, excluded=None):
    r = {"id": i, "lang": lang, "text": f"t{i}{lang}", "path": f"{lang}/v/{i}.wav", "dur": 2.0, "asr_ok": ok,
         "cause_history": ["uncertain"] if not ok else ["ok"], "target": {"name": "x"}}
    if excluded:
        r["excluded"] = excluded
    return r


def test_retry_rows_only_failed_clips_of_the_langs_including_excluded():
    base = [_row("a", "fr", False), _row("a", "en", False), _row("b", "it", True), _row("c", "fr", False, "gd")]
    plan = {(r["id"], r["lang"]): {"id": r["id"], "lang": r["lang"], "text": r["text"]} for r in base}
    got = {(r["id"], r["lang"]) for r in A.retry_rows(base, plan, {"it", "fr"})}
    assert got == {("a", "fr"), ("c", "fr")}


def test_apply_replaces_only_verified_ok_and_keeps_exclusion_and_target():
    base = [_row("a", "fr", False), _row("c", "fr", False, "gd"), _row("d", "it", False), _row("b", "it", True)]
    new = {(i, l): {"text": f"t{i}{l}", "path": f"{l}/v/{i}.wav", "dur": 3.0} for i, l in [("a", "fr"), ("c", "fr"),
                                                                                         ("d", "it")]}
    ver = {"a|fr": {"ok": True}, "c|fr": {"ok": True}, "d|it": {"ok": False, "cause": "uncertain"}}
    rows, stat = A.apply(base, new, ver, 3)
    by = {(r["id"], r["lang"]): r for r in rows}
    assert by[("a", "fr")]["asr_ok"] and by[("a", "fr")]["render_round"] == 3 and by[("a", "fr")]["dur"] == 3.0
    assert by[("c", "fr")]["excluded"] == "gd" and by[("c", "fr")]["asr_ok"]
    assert not by[("d", "it")]["asr_ok"] and by[("d", "it")]["cause_history"][-1] == "uncertain"
    assert by[("b", "it")] == base[3] and by[("a", "fr")]["target"] == {"name": "x"}
    assert stat["recovered:fr"] == 2 and "recovered:it" not in stat
