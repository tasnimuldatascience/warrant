# Everything a reviewer needs, in the order they need it.
# Targets appear here only when the code behind them exists.
PY ?= python
CFG ?= configs/default.yaml

.PHONY: help install survey fetch build index eval autopsy test invariants lint fmt clean all

help:
	@echo "install     editable install with dev extras"
	@echo "survey      how much amendment history each eCFR part actually has"
	@echo "fetch       download eCFR point-in-time snapshots (cached, ~10 min first run)"
	@echo "build       parse snapshots into the bitemporal store"
	@echo "index       embed the store for dense retrieval"
	@echo "diff        classify snapshot-to-snapshot change; report the discard rate"
	@echo "eval        score every benchmark bucket, with ablations"
	@echo "autopsy     localize every failure to a stage; print the failure budget"
	@echo "test        run the unit suite (offline)"
	@echo "invariants  run the deterministic correctness gates against a built corpus"
	@echo "all         fetch -> build -> index -> eval -> autopsy"

install:
	$(PY) -m pip install -e ".[dev]"

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
	$(PY) -m warrant.cli eval run -c $(CFG)

autopsy:
	$(PY) -m warrant.cli autopsy run -c $(CFG)

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
