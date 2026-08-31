# Eval 023 — a fifth screen for the 281 series nothing rendered

**Date:** 2026-08-31
**Scope:** `ui/src/metrics.ts` (new), `ui/src/metrics.test.ts` (new, 16 tests),
`ui/src/screens/Observability.tsx` (new), `ui/src/api.ts`, `ui/src/app.tsx`, `ui/src/lib.ts`,
`ui/src/styles.css`, `ui/tsconfig.app.json`, `ui/package.json`, rebuilt `ui/dist`. No file
under `src/warrant/`, `tests/`, `configs/`, `benchmarks/`, `media/`, `docs/`, `README.md`,
`ARCHITECTURE.md` or `ui/src/screens/Ask.tsx` was touched, and no server change was needed —
see "What the server did not need to change" below.
**Reproduce:** `npm run typecheck` and `npm run build` in `ui/`; `npm run test` (or
`node --test src/metrics.test.ts`) for the parser suite; `ruff check src tests` and
`python -m pytest` for the untouched Python side; live verification below was run against
`python -m warrant.cli serve --port 8016 --no-warm` and the real corpus
(`data/warrant.sqlite3`, 13,212 chunk versions), not synthetic fixtures.

## The problem

`serve/metrics.py` hand-emits a Prometheus scrape at `/metrics` — nine declared series,
expanding to well over 200 label combinations on real traffic — and nothing before this
screen rendered a single one of them. Trace already makes the project's central claim
concrete for *one* request: which stage is answerable for the answer. Observability makes
the same claim at the aggregate — which stage is slow, how often admission control has had
to refuse, whether the cache is earning its keep — from counters the server was already
paying to maintain.

## The design

**A hand-rolled parser, because the input is hand-rolled too.** `metrics.py`'s own docstring
argues against a client library for the *server* side of this format; the same argument holds
for the browser. `parseMetrics` (`ui/src/metrics.ts`) reads `# HELP`/`# TYPE` comments and
sample lines directly, with a real label-block scanner rather than a `split(",")` — label
values are backslash-escaped for `\`, `"` and `\n` per `_escape()` in `metrics.py`, so a value
containing a comma or an embedded quote is legal on the wire and a regex-per-line parser
would silently mis-tokenize it. Histogram assembly reconstructs `HistogramSeries` (buckets,
sum, count) by label set from the four line-shapes `Registry.render()` emits, and every
lookup — `findSample`, `findHistogram`, `filterSamples` — takes the parsed structure as a
plain argument. Nothing in `metrics.ts` touches React, `fetch`, or the DOM; `useMetricsPoll`
in `Observability.tsx` is the only place any of that happens.

**Quantiles are computed, not requested.** This server does not run a Prometheus quantile
function — the browser has raw cumulative bucket counts and nothing else. `bucketQuantile`
implements the same linear-interpolation-within-a-bucket assumption `histogram_quantile`
makes, and reports it: every `Quantile` carries a `clipped` flag, true exactly when the
target rank falls in the `+Inf` overflow bucket, where there is no upper edge to interpolate
against. The UI renders a clipped value with a leading "≥" rather than a bare number, and the
stage-latency panel states in prose that these are bucket-interpolated estimates, dense where
`LATENCY_BUCKETS_MS` is dense (1–40ms) and coarse past 250ms — the same honesty the README
already practices by publishing p-values next to features it disabled.

**A malformed or partial scrape degrades, it does not crash or lie.** Every line that fails to
parse as a HELP/TYPE comment or a sample is skipped and recorded verbatim (capped at 40) in
`ParsedMetrics.errors`; the screen surfaces a count and the first few offending lines rather
than silently dropping them or throwing past the rest of a scrape that parsed fine. A gauge
that was never `set()` this run (`warrant_uncovered_chunks` when no dense index is loaded)
does not appear on the wire at all — `findSample` returns `null` for it, distinct from a
sample that was explicitly written as `0`, and the corpus section renders that distinction in
prose rather than collapsing both to "0".

**Polling is honest about staleness.** `useMetricsPoll` fetches on a 5s interval, skips a tick
entirely when `document.hidden`, and catches up immediately on `visibilitychange` rather than
waiting out the rest of a stale interval. A failed scrape sets `error` without touching the
last successful `text`, and the screen's top-level render checks `error` *before* checking
whether parsed data exists — so a failure shows as `<Failure>`, never as the previous scrape's
numbers presented as current. The toolbar's "as of" timestamp and pause control are rendered
unconditionally, above that branch, so a reader can retry or pause during an outage.

## What renders, in the order the README's own argument runs

1. **Stage latency** (`warrant_stage_duration_ms`) — a ledger, not a chart: stage,
   observation count, p50, p95, and a single relative-magnitude bar per row (plain CSS, no
   library). `predicates`, `lexical`, `dense`, `fusion`, `rerank`, matching the stage names
   `serve/api.py` actually observes (`if stage != "total"`).
2. **Requests, by endpoint and status** (`warrant_requests_total{endpoint,status}`) — one row
   per route template (never a raw path, matching `_route`'s own cardinality guard), columns
   for whichever status classes actually appear, a 5xx cell rendered as a stamp tag.
3. **Generation** (`warrant_generate_duration_s`) — p50/p95, mean, and this process's own
   implied throughput (`60 / mean`) set beside the README's measured ceiling of 7.7/min
   (6.7–9.4 CI, eval-010) — the ceiling is cited as a constant, not re-derived from a handful
   of live observations that could not carry the same weight.
4. **Admission control** (`warrant_admission_rejected_total{reason}`) — `queue_full` and
   `deadline` against the total generation attempts implied by the generate histogram's own
   count. When `deadline` is zero, the panel says so in prose as a claim worth showing, not a
   default worth hiding — matching how the README already treats this exact counter.
5. **Answer cache** (`warrant_cache_total{outcome}`) — hit rate, hit vs. miss broken out with
   the existing `Meter` component (reused verbatim from Ask/Trace, no new bar primitive
   needed for this one).
6. **Corpus** (`warrant_corpus_chunks`, `warrant_uncovered_chunks`, `warrant_ready`) — sampled
   fresh at scrape time per the server's own docstring, with `uncovered_chunks === null`
   rendered as "no dense index," never as "0 missing."

A top-level note appears when total request volume is zero: "this process has served no
requests since it started," so a fresh process reads as *nothing has happened*, not as
*everything failed*. Every section's own zero-cells use the same dim-gray convention Trace
already established for `stages.filter(s => s.out === 0)`.

## Verified against a running server, real traffic

`python -m warrant.cli serve --port 8016 --no-warm`, against `data/warrant.sqlite3`. Driven
traffic: three retrieval-only `/api/ask` calls (200/200/200), one 404 against
`/api/section/nope.999`, two full `/api/ask/stream` generations, and two `/api/ask` calls for
the same question (a cache miss then a cache hit — 5.41s then 5.7ms). The resulting `/metrics`
scrape (241 lines, 223 samples) was fed straight through `parseMetrics` from `ui/src`:

- **Zero parse errors** on the real scrape.
- All five stages present with sane magnitudes — `predicates` p50 0.75ms / p95 9.25ms,
  `lexical` p50 28.75ms / p95 38.9ms, `rerank` p50 60ms / p95 145ms, `dense` p95 ≈2.05s
  (the cold-start `SentenceTransformer` load under `--no-warm`, exactly the seconds-not-
  milliseconds README warns `Ready.uncovered_chunks` callers about) — none clipped.
- `warrant_requests_total` reconstructed the exact per-endpoint counts the driven traffic
  produced, including the self-referential `/metrics` 2xx from the scrape's own prior fetch
  and the one 4xx on `/api/section/{section_id}`.
- `warrant_generate_duration_s`: count 3, sum 18.0271s — matches the two streamed generations
  plus the one non-streamed cache-miss call.
- `warrant_cache_total`: `hit=1, miss=1`, exactly the driven pair.
- `warrant_admission_rejected_total`: **no samples at all** on this run — correctly rendered
  as "declared, never observed," not coerced to a false zero.
- `warrant_corpus_chunks=13212`, `warrant_uncovered_chunks=0` (a dense index was loaded),
  `warrant_ready=1` — all three matched a direct `curl /metrics` read byte-for-byte.

The Chrome extension needed for an in-browser screenshot was not connected in this
environment, so the render itself was not visually screenshotted; `npm run typecheck` and
`npm run build` both pass clean against the finished component tree, and the section
components were exercised against this exact real, non-trivial parsed payload (all six
branches — including the "no admission rejections yet," "uncovered is 0 not null," and
"cache has data" paths) with correct output values, which is the strongest verification
available without that tool. `ruff check src tests` and `python -m pytest` (full suite) both
stayed clean, as neither was touched.

## The parser's own tests

`ui/src/metrics.test.ts`, 16 cases, run with `node --test` (Node 23.6+'s native TypeScript
stripping — no `ts-node`, no `vitest`, no dependency this project does not already have).
Covers: multi-label counters, an unlabelled gauge that is absent vs. present-as-zero,
histogram assembly from all four line shapes, quantile interpolation at a known rank,
clipped/+Inf-bucket quantiles, an all-zero histogram (declared, scraped, never observed) with
no quantile rather than `NaN`, a malformed line skipped and recorded rather than thrown,
escaped-quote/backslash/comma round-tripping through a label value, `+Inf`/`-Inf`/`NaN` as
legal values against genuine garbage, duplicate sample lines both surviving (a torn scrape
concatenating two bodies must not silently pick one), HELP/TYPE with no following samples, an
unrecognised `TYPE` falling back to `untyped` instead of being rejected, a histogram missing
`_sum` reporting `sum: null` rather than `0`, a stray OpenMetrics `# EOF` not counted as an
error, and empty input. `tsc -b` excludes `*.test.ts` (`tsconfig.app.json`) since typing
`node:test`/`node:assert` needs `@types/node`, which nothing else in this project depends on
and this task should not add — the test file runs correctly without it because Node's
type-stripping mode never type-checks, it only strips annotations.

## What the server did not need to change

Nothing. Every series this screen renders — `warrant_stage_duration_ms{stage}`,
`warrant_requests_total{endpoint,status}`, `warrant_generate_duration_s`,
`warrant_admission_rejected_total{reason}`, `warrant_cache_total{outcome}`,
`warrant_corpus_chunks`, `warrant_uncovered_chunks`, `warrant_ready` — was already declared in
`serve/metrics.py` and already populated by real call sites in `serve/api.py`. The one
histogram deliberately left unrendered, `warrant_request_duration_ms` (whole-request wall
time, as opposed to per-stage), is outside this task's stated priority list; it needs no
server change either, only a seventh section, whenever this screen grows one.

## What this deliberately does not build

**No chart library, no new dependency.** The one visual encoding beyond text — the relative
p95 bar in the stage-latency table — is two `div`s and the same box-model approach `Trace.tsx`
already uses for its funnel's time column, generalized into `.bar`/`.bar__fill` in
`styles.css` rather than reused under the funnel-specific class names.

**No push, no WebSocket.** `/metrics` is a scrape target by design (`serve/metrics.py`'s own
docstring: "scrape cost is paid by the scraper, on its own interval"); polling on a fixed
interval that stops for a hidden tab respects that design rather than asking the server to
carry a live-update channel it was never built to serve.

**No recomputation of README numbers.** The generation throughput panel shows this process's
own observed mean beside the README's published ceiling, explicitly labelled as two different
things — the same discipline `Trace.tsx`'s budget panel already applies ("read from a recorded
run... never recomputed per request").
