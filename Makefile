# mynah-s2s V2 — common tasks. `make` or `make help` lists them.
# Everything lives under MYNAH_HOME (default ~/mynah-s2s; keep this repository clone elsewhere).
# Extra flags for the underlying scripts: make demo ARGS="--preset reference", make multi ARGS="--fc-merge off".
SHELL := /bin/bash
M := bash v2/scripts/mynah.sh
ITEM ?= t02
WS_ITEM ?= q03
GROUP ?=

.DEFAULT_GOAL := help
.PHONY: help setup setup-all setup-tts setup-asr doctor ws-check corpus test-cpu test-gpu demo e2e-fc e2e-corpus multi multi-notools \
        multi-mixed custom serve int4-pack w8-pack client webui webui-gpu asr score reproduce \
        f2-setup f2-plan f2-serve f2-webui f2-traces f2-lang-suite test-f2 test-f2-torch test-f2-gpu

help: ## show this list
	@echo "mynah-s2s (work in progress)"
	@echo "  Frankenstein v2 (recommended): make f2-setup -> source the env file -> make f2-serve / make f2-webui"
	@echo "  VoiceChat engine core runtime: make setup -> make doctor -> make serve (your own audio / client)"
	@echo "  test suites:       make setup-all -> make doctor -> make corpus -> make demo -> e2e-fc / multi / custom"
	@grep -E '^[a-z][a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup: ## core runtime: engine venv, NeMo Speech, CUDA kernels for your GPU, VoiceChat weights (~42 GB)
	bash v2/scripts/setup_cuda.sh
setup-all: ## core runtime + synthetic test-audio TTS + evaluation ASR (everything the test suites use)
	bash v2/scripts/setup_cuda.sh --all
setup-tts: ## add / repair only the test-audio TTS (Qwen3-TTS 0.6B)
	bash v2/scripts/setup_cuda.sh --only tts
setup-asr: ## add / repair only the evaluation ASR (Omnilingual ASR CTC 3B v2, own venv)
	bash v2/scripts/setup_cuda.sh --only asr
doctor: ## check the core runtime (required) and the test TTS / ASR / audio (optional); prints the fix
	@$(M) doctor

corpus: ## render the synthetic test audio (40 corpus items, 9 capability items, 7 multi-call conversations)
	$(M) corpus
test-cpu: ## unit tests, no GPU or weights needed
	$(M) test-cpu
test-gpu: ## GPU tests (weights needed)
	$(M) test-gpu

# ---- Frankenstein v2: the recommended engine (multilingual, one GPU process) -----------------------------------
# F2_PYTHON: a Python with frankenstein2/requirements.txt installed (torch with CUDA for serving; only pytest +
# numpy for test-f2). F2_ENV: the paths written by f2-setup (it records F2_LLM, read by the server).
# F2_LLM: qwen4b (recommended: Qwen3.5-4B int8) | moe (Qwen3.6-35B-A3B NVFP4, Blackwell) | moe-awq (same MoE in
# AWQ int4, any recent GPU) | nano (Nemotron Nano 9B v2, legacy). Sizes and memory: docs/frankenstein2.md.
F2_PYTHON ?= python3
F2_LLM ?= qwen4b
F2_LANGS ?= it,en,fr
F2_HF_HOME ?= $(HOME)/mynah-s2s/hf
F2_ENV ?= $(F2_HF_HOME)/f2.env
F2_TRACE_DIR ?= $(CURDIR)/.f2-traces
F2_ARGS ?=
f2-setup: ## Frankenstein v2: download ears + the F2_LLM brain + Pocket TTS packages for F2_LANGS (it,en,fr)
	$(F2_PYTHON) frankenstein2/scripts/setup_models.py --llm $(F2_LLM) --langs $(F2_LANGS) --hf-home $(F2_HF_HOME)
f2-plan: ## Frankenstein v2: what f2-setup would download for F2_LLM / F2_LANGS, with sizes (nothing downloaded)
	$(F2_PYTHON) frankenstein2/scripts/setup_models.py --llm $(F2_LLM) --langs $(F2_LANGS) --hf-home $(F2_HF_HOME) --dry-run
f2-serve: ## Frankenstein v2 WebSocket server on :8772 (paths from F2_ENV, extra flags in F2_ARGS)
	set -a; . "$(F2_ENV)"; set +a; PYTHONPATH="$(CURDIR)/v2/src" HF_HUB_OFFLINE=1 $(F2_PYTHON) \
		frankenstein2/server/serve.py --port 8772 --trace-dir "$(F2_TRACE_DIR)" --pocket-preload "$$F2_LANGS" $(F2_ARGS)
f2-webui: ## Frankenstein v2 server + web UI (http://127.0.0.1:8787); the Debug log switch shows engine lines
	set -a; . "$(F2_ENV)"; set +a; $(MAKE) webui-gpu ENGINE=frankenstein2 WEBUI_PYTHON="$(F2_PYTHON)" \
		FRANK2_SERVE_ARGS="--pocket-preload $$F2_LANGS $(F2_ARGS)"
f2-traces: ## Frankenstein v2 trace store: sessions; SESSION=<id suffix> for a timeline, SPANS=1 for durations
	$(F2_PYTHON) -m frankenstein2.obs "$(F2_TRACE_DIR)/f2_traces.sqlite" $(if $(SESSION),--session $(SESSION)) \
		$(if $(SPANS),--spans)
test-f2: ## Frankenstein v2 unit tests: pure Python + numpy, no torch / GPU / weights (what CI runs)
	$(F2_PYTHON) -m pytest -q frankenstein2/tests
test-f2-torch: ## Frankenstein v2 unit tests + the torch CPU tests on tiny random weights
	$(F2_PYTHON) -m pytest -q frankenstein2/tests --torch
f2-lang-suite: ## Frankenstein v2 language suite (opt-in, GPU): talker, LLM and e2e per language in F2_LANGS
	set -a; . "$(F2_ENV)"; set +a; PYTHONPATH="$(CURDIR)/v2/src" HF_HUB_OFFLINE=1 $(F2_PYTHON) frankenstein2/lab/lang_suite.py \
		--langs "$(F2_LANGS)" --out "$(F2_TRACE_DIR)/lang-suite-$$(date +%Y%m%d-%H%M%S)" $(F2_ARGS)
test-f2-gpu: ## Frankenstein v2 unit tests + GPU smoke tests on the real models (after f2-setup)
	set -a; . "$(F2_ENV)"; set +a; $(F2_PYTHON) -m pytest -q frankenstein2/tests --gpu

demo: ## one session, one item (ITEM=t02 order lookup; try q03, t13, s05): agent text, tool calls, ASR, combined.wav
	$(M) demo $(ITEM)
e2e-fc: ## single-session tool-call suite: lookups, not-found, correction mid-call, follow-up, consecutive calls
	$(M) e2e-fc
e2e-corpus: ## single-session full corpus (40 items: reasoning, small talk, tools) + ASR + score (~30 min)
	$(M) e2e-corpus
multi: ## 4 concurrent sessions: without tools and mixed (2 with tool calls); round latency, isolation, calls
	$(M) multi $(GROUP)
multi-notools: ## 4 concurrent sessions, no tool calls
	$(M) multi m4_notools
multi-mixed: ## 4 concurrent sessions, two of them make a tool call
	$(M) multi m4_mixed
custom: ## custom scenario + persona prompt (default: v2/eval/examples/furniture_store*); SPEC=... MSG=... to change
	$(M) custom $(SPEC) $(MSG)

serve: ## realtime WebSocket server on :8765 (one session at a time); stop with Ctrl-C
	ARGS="$(V2_SERVE_ARGS)" $(M) serve

INT4_PACK ?= $(HOME)/mynah-s2s/artifacts/voicechat-int4.safetensors
INT4_ARGS ?=
int4-pack: ## export offline native INT4 bundle + JSON manifest (INT4_ARGS=...)
	ARGS="$(INT4_ARGS)" $(M) int4-pack "$(INT4_PACK)"

W8_PACK ?= $(HOME)/mynah-s2s/artifacts/voicechat-w8.safetensors
W8_ARGS ?=
w8-pack: ## export offline per-channel Q8 bundle + JSON manifest (W8_ARGS=...)
	ARGS="$(W8_ARGS)" $(M) w8-pack "$(W8_PACK)"

WEBUI_ARGS ?=
WEBUI_GPU_LOG ?= /tmp/mynah-v2-webui-gpu.log
WEBUI_PYTHON ?= $(if $(wildcard $(HOME)/mynah-s2s/.venv-v2/bin/python),$(HOME)/mynah-s2s/.venv-v2/bin/python,python3)
WEBUI_PROFILE ?= demo
WEBUI_GPU_PROFILE ?= int4
WEBUI_AUDIO_ROOT ?= $(HOME)/mynah-s2s/data/agent
# The WebUI path targets the measured lower-latency INT4 configuration by default.
# Direct `make serve` remains BF16 unless the caller explicitly passes --int4.
V2_SERVE_ARGS ?= --int4 --cudnn-benchmark
# ENGINE=frankenstein: the Frankenstein engine (streaming ASR + Thinker as timing shell + text LLM + talker) behind
# the same WebSocket protocol, from frankenstein/ in this tree (FRANK_DIR); FRANK_SERVE_ARGS = its model paths
# (see frankenstein/server/serve.py: --lab-scripts, --nano-dir, --flash-python, --flash-harness, --flash-root, ...). The UI's LANGUAGE / VOICE selectors are honoured by this engine only.
ENGINE ?= voicechat
FRANK_DIR ?= $(CURDIR)
FRANK_PORT ?= 8771
FRANK_SERVE_ARGS ?=
# ENGINE=frankenstein2: Frankenstein v2 (Nemotron streaming ASR + Tyger turn-taking GRU, text LLM with tools, Pocket
# talker), from frankenstein2/ (FRANK2_SERVE_ARGS = its model paths, see frankenstein2/server/build.py).
FRANK2_PORT ?= 8772
FRANK2_SERVE_ARGS ?=
ifeq ($(ENGINE),frankenstein)
WEBUI_SERVE_CMD = cd $(FRANK_DIR) && PYTHONPATH=$(FRANK_DIR)/v2/src $(WEBUI_PYTHON) frankenstein/server/serve.py --port $(FRANK_PORT) $(FRANK_SERVE_ARGS)
WEBUI_V2_WS = ws://127.0.0.1:$(FRANK_PORT)/v1/session
else ifeq ($(ENGINE),frankenstein2)
WEBUI_SERVE_CMD = cd $(FRANK_DIR) && PYTHONPATH=$(FRANK_DIR)/v2/src $(WEBUI_PYTHON) frankenstein2/server/serve.py --port $(FRANK2_PORT) --trace-dir $(FRANK_DIR)/.f2-traces $(FRANK2_SERVE_ARGS)
WEBUI_V2_WS = ws://127.0.0.1:$(FRANK2_PORT)/v1/session
else
WEBUI_SERVE_CMD = ARGS="$(V2_SERVE_ARGS)" $(M) serve
WEBUI_V2_WS = ws://127.0.0.1:8765/v1/session
endif
webui: ## local HTML5 UI + Python bridge on http://127.0.0.1:8787 (no Node.js)
	@if command -v uv >/dev/null 2>&1; then uv run --project webui --no-sync python webui/server.py --profile $(WEBUI_PROFILE) --audio-root "$(WEBUI_AUDIO_ROOT)" $(WEBUI_ARGS); else $(WEBUI_PYTHON) webui/server.py --profile $(WEBUI_PROFILE) --audio-root "$(WEBUI_AUDIO_ROOT)" $(WEBUI_ARGS); fi
webui-gpu: ## start the V2 GPU server and the web UI together; stop both with Ctrl-C
	@set -e; \
	( $(WEBUI_SERVE_CMD) ) >$(WEBUI_GPU_LOG) 2>&1 & gpu_pid=$$!; \
	trap 'kill $$gpu_pid 2>/dev/null || true' EXIT INT TERM; \
	until grep -q '^READY' $(WEBUI_GPU_LOG); do \
		kill -0 $$gpu_pid 2>/dev/null || { tail -40 $(WEBUI_GPU_LOG); exit 1; }; \
		sleep 2; \
	done; \
	if command -v uv >/dev/null 2>&1; then uv run --project webui --extra gpu python webui/server.py --v2-ws $(WEBUI_V2_WS) --engine $(ENGINE) --profile $(WEBUI_GPU_PROFILE) --audio-root "$(WEBUI_AUDIO_ROOT)" $(WEBUI_ARGS); else $(WEBUI_PYTHON) webui/server.py --v2-ws $(WEBUI_V2_WS) --engine $(ENGINE) --profile $(WEBUI_GPU_PROFILE) --audio-root "$(WEBUI_AUDIO_ROOT)" $(WEBUI_ARGS); fi
client: ## stream a WAV to the server at real-time pace: make client WAV=path/to.wav
	$(M) client $(WAV)
ws-check: ## non-interactive server + client round trip on one corpus item (WS_ITEM=q03 by default); fails on dropped audio / no answer
	$(M) ws-check $(WS_ITEM)
asr: ## evaluation ASR of a run directory: make asr DIR=...
	$(M) asr $(DIR)
score: ## score a corpus run directory: make score DIR=...
	$(M) score $(DIR)

reproduce: ## installs missing TTS/ASR, then CPU+GPU tests, corpus, demo, tool calls, multi-session, custom, server (~1.5 h)
	@$(M) doctor
	$(M) extras
	$(M) test-cpu
	$(M) test-gpu
	$(M) corpus
	$(M) demo
	$(M) e2e-fc
	$(M) multi
	$(M) custom
	$(M) ws-check
