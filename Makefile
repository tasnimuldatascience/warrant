# Everything a reviewer needs, in the order they need it.
# Targets appear here only when the code behind them exists.
PY ?= python
CFG ?= configs/default.yaml
SPLIT ?= test

.PHONY: help install survey fetch build index diff eval generation latency autopsy \
        serve test invariants lint fmt clean all

help:
	@echo "install     editable install with dev extras"
	@echo "survey      how much amendment history each eCFR part actually has"
	@echo "fetch       download eCFR point-in-time snapshots (cached, ~10 min first run)"
	@echo "build       parse snapshots into the bitemporal store"
	@echo "index       embed the store for dense retrieval"
	@echo "diff        classify what changed, and report the discard rate"
	@echo "eval        score every bucket on the held-out split, with paired ablations"
	@echo "generation  hallucination, citation precision, abstention"
	@echo "latency     latency vs quality per configuration"
	@echo "autopsy     localize every failure to a stage; print the failure budget"
	@echo "serve       run the HTTP API on :8000"
	@echo "gate        fail if quality regressed below the recorded floor"
	@echo "abstention  risk-coverage, calibration, ECE"
	@echo "test        run the unit suite (offline)"
	@echo "invariants  the deterministic correctness gates"
	@echo "all         fetch -> build -> index -> eval -> autopsy"

install:
	$(PY) -m pip install -e ".[dev,serve]"

survey:
	$(PY) -m warrant.cli corpus survey -c $(CFG)

fetch:
	$(PY) -m warrant.cli corpus fetch -c $(CFG)

build:
	$(PY) -m warrant.cli corpus build -c $(CFG) --rebuild

index:
	$(PY) -m warrant.cli index build -c $(CFG)

diff:
	$(PY) -m warrant.cli corpus diff -c $(CFG)

eval:
	$(PY) -m warrant.cli eval run -c $(CFG) --split $(SPLIT)

generation:
	$(PY) -m warrant.cli eval generation -c $(CFG) --split $(SPLIT)

latency:
	$(PY) -m warrant.cli eval latency -c $(CFG) --split $(SPLIT)

autopsy:
	$(PY) -m warrant.cli autopsy run -c $(CFG)

serve:
	$(PY) -m warrant.cli serve -c $(CFG)

test:
	$(PY) -m pytest -q

invariants:
	$(PY) -m pytest -q tests/invariants -m ""

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

clean:
	rm -rf runs/* .pytest_cache .ruff_cache

all: fetch build index eval autopsy

gate:
	$(PY) -m warrant.cli eval gate -c $(CFG) --split $(SPLIT)

abstention:
	$(PY) -m warrant.cli eval abstention -c $(CFG)
