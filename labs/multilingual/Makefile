# labs/multilingual: CPU tests (engine venv with NeMo Speech; PYTHONPATH = the engine in this repository).
# Each directory runs on its own, from inside it: the lab tests import their sibling modules by name (`agentbench`,
# `session_distill`, ...), as they always did in the lab.
ROOT := $(abspath ../..)
PY ?= python3
RUN = PYTHONPATH=$(ROOT)/v2/src:$$PYTHONPATH CUDA_VISIBLE_DEVICES= $(PY) -m pytest -q -p no:cacheprovider
.PHONY: test
test:
	cd $(CURDIR)/tests && $(RUN) .
	cd $(CURDIR)/scripts/multilingual && $(RUN) .
	cd $(CURDIR)/scripts/agentbench && $(RUN) .
