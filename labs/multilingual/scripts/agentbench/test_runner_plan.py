"""CPU tests for m0_runner plan parsing and the live-config override (python3 test_runner_plan.py; needs omegaconf)."""
from pathlib import Path

src = (Path(__file__).resolve().parent / "m0_runner.py").read_text()
ns: dict = {}
ns["json"] = __import__("json")
exec(src[src.index("class JsonlSink"):src.index("def main():")], ns)   # helpers only, no engine imports


def test_parse_plan():
    b = ns["parse_plan"]("cridge:en:m03::off v21:en:m03,m08 x:en:m03:2:on v21:en:m03:::guided", 7.0)
    assert b == [["cridge", "en", "m03", 7.0, "off", "native"], ["v21", "en", "m03,m08", 7.0, "on", "native"],
                 ["x", "en", "m03", 2.0, "on", "native"], ["v21", "en", "m03", 7.0, "on", "guided"]], b
    for bad in ("a:en:m03:7:maybe", "a:en:m03:7:on:scripted"):
        try:
            ns["parse_plan"](bad, 7.0)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_set_model_cfg():
    from omegaconf import OmegaConf
    c = OmegaConf.create({"a": 1})
    OmegaConf.set_struct(c, True)                       # struct config, as the wrapper's
    assert ns["set_model_cfg"](c, "first_turn_fallback_frames", 10 ** 9) is None
    assert c.first_turn_fallback_frames == 10 ** 9 and c.get("first_turn_fallback_frames", 50) == 10 ** 9
    assert ns["set_model_cfg"](c, "first_turn_fallback_frames", None) == 10 ** 9
    assert c.get("first_turn_fallback_frames", 50) == 50   # restored to the code default
    d = {}
    ns["set_model_cfg"](d, "k", 5)
    assert d == {"k": 5}


def test_fc_fix_defaults():
    e = {}
    assert ns["apply_fc_fix_defaults"](e, False) == {"S2S_FIX_CONTEXT_PAD": "1", "S2S_FIX_FC_TOKEN_SYNC": "all"}
    e = {"S2S_FIX_FC_TOKEN_SYNC": "text", "S2S_FIX_CONTEXT_PAD": "1"}        # an explicit env value wins
    assert ns["apply_fc_fix_defaults"](e, False) == {"S2S_FIX_CONTEXT_PAD": "1", "S2S_FIX_FC_TOKEN_SYNC": "text"}
    e = {"S2S_FIX_CONTEXT_PAD": "1"}
    assert ns["apply_fc_fix_defaults"](e, True) == {"S2S_FIX_CONTEXT_PAD": "0", "S2S_FIX_FC_TOKEN_SYNC": "0"}


def test_jsonl_sink(tmp="/tmp/jsonl_sink_test.jsonl"):
    s = ns["JsonlSink"](tmp)
    s.append({"a": 1})
    assert open(tmp).read() == '{"a": 1}\n' and s == [{"a": 1}]   # on disk before close (timeout-safe)
    s.close()


if __name__ == "__main__":
    test_fc_fix_defaults()
    test_jsonl_sink()
    test_parse_plan()
    test_set_model_cfg()
    print("OK runner plan/cfg")
