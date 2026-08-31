# Deploying warrant

This has **never served real traffic**. Every capacity number below came from a synthetic
load generator (`warrant.bench.load`) run against `localhost` on one laptop with one RTX 5070
Laptop GPU. Treat it as the shape of the ceiling, not a number a production SLO should be set
against without re-measuring on the hardware you actually run.

## Two images

`Dockerfile` is multi-stage and builds two images from one file — see its header comment for
the full reasoning. `docker build .` with no `--target` builds `lexical`.

| target | what it runs | needs | size |
|---|---|---|---|
| `lexical` (default) | ingest, apparatus stripping, the bitemporal store, BM25/FTS5 retrieval, both predicates (as-of, applicability), the failure budget | CPU only | **~350–450 MB, estimated** |
| `neural` | + dense retrieval, cross-encoder reranking, generation | CPU works; GPU is what the numbers below were measured on | **~2.3–2.6 GB, estimated** |

**These sizes are estimates, not measurements.** Docker was not available in the environment
that wrote this file — `docker build` was never run, so neither image has actually been built
or booted. Before trusting this document, run:

```bash
docker build --target lexical -t warrant:lexical .
docker build --target neural  -t warrant:neural  .
docker images warrant
```

and update the table above with what those report. The estimate is `torch>=2.4`'s own
~2 GB (pyproject.toml's comment on the `neural` extra) plus the base image and the rest of
the dependency tree; nothing here has confirmed it.

Build once you actually need the neural target — the lexical image is not a crippled demo.
On the held-out split, lexical-only matches the full pipeline's sufficiency (96.7% either
way) at a fraction of the latency (14.1 ms p50 vs. 87.4 ms with reranking — see the README's
latency table). Reach for `neural` when you specifically want to see dense retrieval, the
reranker, or generated answers with citations.

## Resources

| | lexical | neural (CPU) | neural (GPU) |
|---|---|---|---|
| RAM | ~512 MB working set is generous for a ~13k-chunk-version corpus | budget 6–8 GB — Qwen2.5-1.5B, the encoder and the reranker all land in system RAM with no GPU to hold them | ~5 GB **GPU** memory resident (bge-small 127 MB + MiniLM cross-encoder 87 MB + Qwen2.5-1.5B fp16 ~3.1 GB + CUDA context), host RAM stays modest |
| disk, corpus | ~100 MB for this repo's configured scope (Title 5, 26 parts, 2017– floor): ~56 MB raw eCFR XML cache + ~16 MB SQLite store. Reserve **≥1 GB** — a wider `corpus.parts` or a longer history window grows the XML cache | same lexical footprint | + ~20–25 MB of dense vectors (`data/dense`) |
| generation throughput | not applicable — the lexical image has no generator | **unmeasured**. Nothing in `results/eval-010-capacity.md` was run on CPU; expect it to be much slower than the GPU numbers below, possibly by an order of magnitude | **the only configuration actually measured** — see below |

The generation numbers this repo publishes are GPU numbers. Running `neural` without a GPU
gets you dense retrieval and reranking at a reasonable cost (those stages are small encoders)
and a generator that will answer, just not at any pace this document can promise.

## Getting a corpus in

A fresh container — either image — has no corpus. `data/` is gitignored (it's reproducible,
and it's hundreds of megabytes of XML that doesn't belong in git or in an image layer), so
`/ready` correctly returns 503 on first boot: *"corpus is empty; run `make build`."* A
container that silently starts and answers nothing is the failure this section exists to
prevent — don't skip it.

Two ways to populate the `warrant-data` volume `compose.yaml` declares:

**Recommended — build inside the container.** This is the only path that needs nothing but
Docker: no host Python, no host network policy to reconcile with the container's. One-time,
after `docker compose up -d`:

```bash
docker compose exec api make fetch    # eCFR point-in-time snapshots, cached, ~10 min once
docker compose exec api make build    # parse into the bitemporal store
docker compose exec api make index    # neural image only — embeds for dense retrieval, needs torch
```

(`make` is installed in both runtime images specifically so this is the literal command the
README already documents, not a Docker-specific paraphrase.) The volume persists across
`docker compose down` / `up`, so this runs once per volume, not once per container start.

**Alternative — build on the host, bind-mount the result.** Faster if you already have a
Python dev environment set up and are iterating on the pipeline itself: run `make fetch &&
make build && make index` on the host as the README describes, then point the volume at it
instead of the named `warrant-data` volume:

```yaml
volumes:
  - ./data:/app/data   # instead of warrant-data:/app/data
```

This avoids re-fetching from eCFR inside the container and lets you rebuild images without
losing a corpus you already trust. Prefer it once you're doing this more than once; prefer
the in-container path for the first run, because it's the one that asks a reviewer to trust
nothing but Docker.

Either way, confirm the corpus landed before pointing traffic at the service:

```bash
curl -s http://localhost:8000/ready | python -m json.tool
```

`uncovered_chunks: null` means a lexical-only deployment (no dense index to be missing from)
— not zero. On `neural`, a non-null nonzero value means the dense index is one `index build`
behind the store; retrieval still works, just half-fused for those chunks.

## What the measured ceiling actually is

From `results/eval-010-capacity.md`, GPU-measured, unbatched, one generation at a time
(the server enforces this with a semaphore — it's not a suggestion):

- **29.2–29.9 tokens/s**, isolated, over an average ~205 output tokens per answer
- **6.6 s per answer** (mean 7.06 s under load, 6.6 s median in isolation)
- **7.7 requests/minute** measured ceiling (6.7–9.4 across runs), **stable band ≤ 6/min** —
  above 6/min the queue starts growing and doesn't drain within the run
- Admission control refuses with `503` and `Retry-After`, but **does not refuse promptly**:
  its measured floor is 20.1 s and at 3× load the median refused caller waits 65 s for a
  "come back later." If you're fronting this with a real load balancer, do not assume a
  fast-fail; budget for the queue described in eval-010 §2.
- Retrieval-only (`/api/ask?generate=false`) is a different class entirely: ~82 ms per
  request, no GPU, knee at ~18 req/s on a quiet machine. It is *not* protected from the
  generation path today — a generation overload starves it too, because both share the same
  four-token thread pool (`THREAD_LIMIT` in `serve/api.py`). See eval-010 §3 before assuming
  the read endpoints degrade independently of `/api/ask`.

If you need a number to put in a capacity plan: **this is a seven-requests-a-minute service**
on the one GPU it has been measured on, not a service you point a crawler or a demo audience
at without a queue in front of it.

## What to watch

- **`/health`** — liveness only. `async def`, touches nothing. A missing corpus or a
  generation overload cannot fail it; `compose.yaml`'s healthcheck uses this and only this.
- **`/ready`** — readiness. `503` until the corpus exists and models are built; reports
  `corpus`, `models`, `chunks`, and `uncovered_chunks` (see above). Not wired as a Docker
  `HEALTHCHECK` here on purpose — Compose has one probe slot and conflating liveness with
  readiness restarts a container that's merely waiting on a corpus, forever. Check it by hand
  or wire it into whatever orchestrates this beyond Compose.
- **`/metrics`** — Prometheus text, hand-emitted, labels bounded by construction (route
  templates, never raw paths — a 404 sweep can't mint a new series per request). Includes
  `warrant_requests_total`, `warrant_request_duration_ms`, `warrant_stage_duration_ms`,
  `warrant_generate_duration_s`, `warrant_admission_rejected_total{reason=...}`,
  `warrant_corpus_chunks`, `warrant_uncovered_chunks`, `warrant_ready`.
- **Logs** — JSON, one line per event, carrying `request_id` (every request, including ones
  rejected before they reach a handler) and `trace_id` (only requests that got far enough to
  be recorded — the asymmetry is deliberate: it's what separates "answered wrongly" from
  "never got to answer"). `docker compose logs -f api` gets you these as-is; pipe to whatever
  log shipper you use, there's no special container-side handling to configure.

## Commands, end to end

```bash
git clone https://github.com/tasnimuldatascience/warrant
cd warrant

docker compose up -d --build            # lexical target by default
docker compose exec api make fetch      # ~10 min, once, cached
docker compose exec api make build

curl -s http://localhost:8000/health
curl -s http://localhost:8000/api/meta | python -m json.tool
curl -s http://localhost:8000/ready | python -m json.tool

open http://localhost:8000/             # the UI, served from the same process
```

For the full pipeline: `WARRANT_TARGET=neural docker compose up -d --build`, then also run
`docker compose exec api make index` before expecting dense retrieval or generation to work
(`generate=true` is the API default once a corpus and models exist).

## Not covered here

- Horizontal scaling, a reverse proxy, TLS termination, auth — this API is unauthenticated
  and read-only by design (see the README's "What this is not"), and none of that has been
  built or measured.
- Ingesting the non-eCFR tiers (statute, Federal Register notices, OPM guidance, govinfo
  scans) needs `pip install -e ".[sources]"` (PyMuPDF, RapidOCR) beyond what either image
  installs — out of scope for this deployment path; see `warrant corpus ingest --help`.
- A CPU-only torch build would shrink the `neural` image considerably below the ~2 GB
  estimate above. Not attempted here — it's a real optimization, just a separate decision
  from "does a deployment artifact exist at all."
