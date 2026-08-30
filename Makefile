# Everything a reviewer needs, in the order they need it.
PY ?= python

.PHONY: help install fetch build index bench eval autopsy serve test lint fmt clean all

help:
	@echo "install   editable install with dev extras"
	@echo "fetch     download eCFR point-in-time snapshots (cached, ~10 min first run)"
	@echo "build     parse snapshots into the bitemporal chunk store"
	@echo "index     build lexical + dense indexes over the store"
	@echo "bench     mine amendment diffs into the temporal benchmark"
	@echo "eval      score the default system on all benchmark buckets"
	@echo "autopsy   localize every failure to a pipeline stage; print the failure budget"
	@echo "serve     API + dashboard on :8000"
	@echo "test      run the test suite"
	@echo "all       fetch -> build -> index -> bench -> eval -> autopsy"

install:
	$(PY) -m pip install -e ".[dev]"

fetch:
	$(PY) -m warrant.cli corpus fetch -c configs/default.yaml

build:
	$(PY) -m warrant.cli corpus build -c configs/default.yaml

index:
	$(PY) -m warrant.cli index build -c configs/default.yaml

bench:
	$(PY) -m warrant.cli bench mine -c configs/default.yaml

eval:
	$(PY) -m warrant.cli eval run -c configs/default.yaml

autopsy:
	$(PY) -m warrant.cli autopsy run -c configs/default.yaml

serve:
	$(PY) -m warrant.cli serve -c configs/default.yaml

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

clean:
	rm -rf runs/* .pytest_cache .ruff_cache

all: fetch build index bench eval autopsy
