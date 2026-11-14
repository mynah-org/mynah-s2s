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
        multi-mixed custom serve int4-pack client webui webui-gpu asr score reproduce

help: ## show this list
	@echo "mynah-s2s V2 (work in progress)"
	@echo "  core runtime only: make setup -> make doctor -> make serve (your own audio / client)"
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

WEBUI_ARGS ?=
WEBUI_GPU_LOG ?= /tmp/mynah-v2-webui-gpu.log
WEBUI_PYTHON ?= $(if $(wildcard $(HOME)/mynah-s2s/.venv-v2/bin/python),$(HOME)/mynah-s2s/.venv-v2/bin/python,python3)
# The WebUI path targets the measured GB10 configuration by default. Direct
# `make serve` remains BF16 unless the caller explicitly passes --int4.
V2_SERVE_ARGS ?= --int4 --cudnn-benchmark
webui: ## local HTML5 UI + Python bridge on http://127.0.0.1:8787 (no Node.js)
	@if command -v uv >/dev/null 2>&1; then uv run --project webui --no-sync python webui/server.py $(WEBUI_ARGS); else $(WEBUI_PYTHON) webui/server.py $(WEBUI_ARGS); fi
webui-gpu: ## start the V2 GPU server and the web UI together; stop both with Ctrl-C
	@set -e; \
	ARGS="$(V2_SERVE_ARGS)" $(M) serve >$(WEBUI_GPU_LOG) 2>&1 & gpu_pid=$$!; \
	trap 'kill $$gpu_pid 2>/dev/null || true' EXIT INT TERM; \
	until grep -q '^READY' $(WEBUI_GPU_LOG); do \
		kill -0 $$gpu_pid 2>/dev/null || { tail -40 $(WEBUI_GPU_LOG); exit 1; }; \
		sleep 2; \
	done; \
	if command -v uv >/dev/null 2>&1; then uv run --project webui --extra gpu python webui/server.py --v2-ws ws://127.0.0.1:8765/v1/session $(WEBUI_ARGS); else $(WEBUI_PYTHON) webui/server.py --v2-ws ws://127.0.0.1:8765/v1/session $(WEBUI_ARGS); fi
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
