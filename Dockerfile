# syntax=docker/dockerfile:1
#
# Two images from one file, because the two things this project does have nothing to do
# with each other's dependency weight. Ingest, apparatus stripping, the bitemporal SQLite
# store, BM25/FTS5 retrieval and both predicates (as-of, applicability) run on the standard
# library plus a handful of pure-Python/wheel packages -- no torch, no GPU. Dense retrieval,
# the cross-encoder reranker and generation need torch, which pyproject.toml's own comment
# prices at ~2 GB. Shipping one image would make every reviewer who just wants to see the
# predicates work pay a 2 GB download they didn't ask for.
#
#   docker build --target lexical -t warrant:lexical .   # default target, no torch
#   docker build --target neural  -t warrant:neural  .   # + dense, rerank, generation
#
# `docker build .` with no --target builds `lexical`, because it is the last stage below --
# on purpose, so the smaller and safer image is what an unqualified build produces.
#
# Base pinned by digest, not just a minor version: a floating "3.12-slim-bookworm" tag is
# reproducible today and a different image next month, which is exactly what a deployment
# artifact must not be. Digest resolved from Docker Hub's manifest-list for
# python:3.12-slim-bookworm on 2026-08-30; re-resolve it deliberately, don't let it drift.
ARG PY_BASE=python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

# -- base: the repo, and nothing installed yet --------------------------------------------
#
# One COPY of the source for both variants, so a change to src/ busts one layer instead of
# two independent ones. Everything actually excluded from the build context (data/, git,
# node_modules, caches) is listed in .dockerignore, not here.
FROM ${PY_BASE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Fixed uid/gid rather than `useradd -m`'s auto-assigned one: a volume built by uid 999 on
# one host and mounted by uid 1000 on another is a permission error nobody can reproduce.
RUN groupadd --gid 1000 warrant && \
    useradd --uid 1000 --gid warrant --no-create-home --shell /usr/sbin/nologin warrant
COPY . .

# One entrypoint, shared by both final stages, parameterised by the WARRANT_MODE each stage
# bakes in. It exists because the lexical image has no sentence-transformers installed, and
# configs/default.yaml -- which this Dockerfile does not own and will not edit -- ships with
# index.dense.enabled and index.rerank.enabled both true. Constructing the retriever against
# that config in the lexical image would ImportError on the first request that reached
# retrieval, not at startup, which is the worst place to discover a packaging mismatch. So
# the lexical entrypoint reads the real config at container start, flips just those two
# booleans in memory, and writes the result to /tmp -- the tracked YAML on disk is never
# touched, and every other setting (corpus.parts, retrieve.final_k, the fusion weights) comes
# from the one file the rest of the team owns.
RUN cat <<'EOF' > /usr/local/bin/warrant-entrypoint.sh && chmod +x /usr/local/bin/warrant-entrypoint.sh
#!/bin/sh
set -eu
MODE="${WARRANT_MODE:-neural}"
CONFIG="configs/default.yaml"

if [ "$MODE" = "lexical" ]; then
    CONFIG="$(mktemp /tmp/warrant-lexical.XXXXXX.yaml)"
    python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

cfg = yaml.safe_load(pathlib.Path("configs/default.yaml").read_text(encoding="utf-8")) or {}
cfg.setdefault("index", {}).setdefault("dense", {})["enabled"] = False
cfg.setdefault("index", {}).setdefault("rerank", {})["enabled"] = False
pathlib.Path(sys.argv[1]).write_text(yaml.dump(cfg), encoding="utf-8")
PY
    exec warrant serve -c "$CONFIG" --host 0.0.0.0 --port 8000 --no-generate "$@"
fi

exec warrant serve -c "$CONFIG" --host 0.0.0.0 --port 8000 "$@"
EOF

# -- builder-lexical: core deps only, no torch ---------------------------------------------
#
# build-essential lives only in this stage. PyStemmer is a C extension and most of the rest
# ship manylinux wheels, but a builder that can compile if a wheel is missing for the
# platform is one dependency bump away from being needed, and it costs nothing in the final
# image because only /opt/venv is copied out.
FROM base AS builder-lexical
RUN apt-get update && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# `serve` is not optional despite being an extra: fastapi and uvicorn live in it, and without
# it there is no HTTP API to containerise. No `dev` (pytest/ruff -- CI's job, not the image's)
# and no `sources` (PyMuPDF/RapidOCR, for PDF/OCR ingestion of tiers the base P0 corpus
# doesn't need): `warrant corpus build` gives the full eCFR P0 corpus with neither.
RUN pip install --no-cache-dir -e ".[serve]"

# -- builder-neural: + dense retrieval, reranking, generation -------------------------------
FROM base AS builder-neural
RUN apt-get update && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# neural = dense encoder + cross-encoder reranker. generate = the answer model. Both pull
# torch; installed together so this is the one stage that pays the ~2 GB, once.
RUN pip install --no-cache-dir -e ".[serve,neural,generate]"

# -- lexical: the runtime image, no torch, no GPU --------------------------------------------
#
# Fresh FROM the pinned base, not FROM builder-lexical: build-essential and pip's own caches
# must not ride along into the image a reviewer actually pulls.
FROM ${PY_BASE} AS lexical
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    WARRANT_MODE=lexical
WORKDIR /app
# `make` so `make fetch && make build` inside the container is the literal command the
# README already documents, not a Docker-specific paraphrase of it.
RUN apt-get update && apt-get install --no-install-recommends -y make \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1000 warrant && \
    useradd --uid 1000 --gid warrant --no-create-home --shell /usr/sbin/nologin warrant
COPY --from=builder-lexical /opt/venv /opt/venv
COPY --from=base /usr/local/bin/warrant-entrypoint.sh /usr/local/bin/warrant-entrypoint.sh
COPY --chown=warrant:warrant --from=base /app /app
# data/ is where a corpus (built on the host, or inside this container -- see docs/DEPLOY.md)
# lives. Owned by the runtime user before USER switches, so a fresh named volume mounted here
# inherits writable ownership instead of root's.
RUN mkdir -p /app/data && chown warrant:warrant /app/data
VOLUME ["/app/data"]
USER warrant
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/warrant-entrypoint.sh"]

# -- neural: + dense retrieval, reranking, generation (~2 GB heavier) -----------------------
FROM ${PY_BASE} AS neural
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    WARRANT_MODE=neural
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y make \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1000 warrant && \
    useradd --uid 1000 --gid warrant --no-create-home --shell /usr/sbin/nologin warrant
COPY --from=builder-neural /opt/venv /opt/venv
COPY --from=base /usr/local/bin/warrant-entrypoint.sh /usr/local/bin/warrant-entrypoint.sh
COPY --chown=warrant:warrant --from=base /app /app
RUN mkdir -p /app/data && chown warrant:warrant /app/data
VOLUME ["/app/data"]
USER warrant
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/warrant-entrypoint.sh"]
