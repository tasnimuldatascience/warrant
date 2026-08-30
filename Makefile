# Everything a reviewer needs, in the order they need it.
# Targets appear here only when the code behind them exists.
PY ?= python
CFG ?= configs/default.yaml

.PHONY: help install survey fetch diff test lint fmt clean

help:
	@echo "install   editable install with dev extras"
	@echo "survey    how much amendment history each eCFR part actually has"
	@echo "fetch     download eCFR point-in-time snapshots (cached, ~10 min first run)"
	@echo "diff      classify snapshot-to-snapshot change; report the discard rate"
	@echo "test      run the test suite (offline)"
	@echo "lint      ruff"

install:
	$(PY) -m pip install -e ".[dev]"

survey:
	$(PY) -m warrant.cli corpus survey -c $(CFG)

fetch:
	$(PY) -m warrant.cli corpus fetch -c $(CFG)

diff:
	$(PY) -m warrant.cli corpus diff -c $(CFG)

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

clean:
	rm -rf runs/* .pytest_cache .ruff_cache
