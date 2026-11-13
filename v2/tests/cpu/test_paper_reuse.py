"""Dedicated checks for paper mechanisms that V2 reuses from NVIDIA unchanged (matrix #15, #25). Need the NVIDIA
source checkout / checkpoint config on a GB10 machine (skipped elsewhere)."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

NEMO = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "third_party/agent/nemo-speech"
CKPT = Path(os.environ.get("V2_CKPT", Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39"))
SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture(scope="module")
def render():
    if not (NEMO / "examples/speechlm2/function_calling/template.jinja").exists():
        pytest.skip("NVIDIA source checkout not present")
    sys.path.insert(0, str(NEMO / "examples/speechlm2"))
    sys.path.insert(0, str(SCRIPTS))
    pytest.importorskip("nemo")
    from run_stream import render_prompt
    return render_prompt


def test_15_prompt_follows_paper_app_d2_template(render):
    """App. D.2: system message, then <AVAILABLE_TOOLS>[...] with the OpenAI tool definitions (no 'type' key),
    the <TOOLCALL> call format and the <TOOL_RESPONSE> response format."""
    from mynah_v2.agent.tools import ORDERS_SCHEMA, ToolRegistry
    reg = ToolRegistry()
    reg.register("get_order_status", lambda a: "", "Get the status of an order.", ORDERS_SCHEMA)
    p = render(reg.schemas, system_message="SYS")
    assert p.startswith("SYS\n\nYou can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[")
    tools = json.loads(re.search(r"<AVAILABLE_TOOLS>(\[.*\])</AVAILABLE_TOOLS>", p, re.S).group(1))
    assert tools == [{"name": "get_order_status", "description": "Get the status of an order.",
                      "parameters": ORDERS_SCHEMA}]
    assert '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}' in p
    assert "<TOOL_RESPONSE>[" in p


def test_15_default_prompt_is_nvidias_own(render):
    import offline_voicechat_fc_infer as fcmod
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import render_fc_system_prompt
    assert render(fcmod.DEFAULT_TOOLS) == render_fc_system_prompt(str(fcmod.DEFAULT_TEMPLATE),
                                                                  fcmod.DEFAULT_SYSTEM_MESSAGE, fcmod.DEFAULT_TOOLS)


def test_25_speaker_is_the_checkpoints_baked_voice():
    """§2.3 speaker prompt: the released checkpoint ships a pre-baked latent (inference_speaker_name) instead of a
    3 s reference clip; V2 must select exactly that one and pass no external reference audio."""
    cfgp = CKPT / "config.json"
    if not cfgp.exists():
        pytest.skip("checkpoint not present")
    pytest.importorskip("omegaconf")
    m = json.loads(cfgp.read_text())["model"]
    from mynah_v2.model.reference import build_cfg
    c = build_cfg(ckpt=CKPT)
    assert c.s2s.speaker_name == m["inference_speaker_name"] == "Aria"
    assert c.s2s.speaker_reference == "" and m.get("inference_speaker_reference", "") == ""
