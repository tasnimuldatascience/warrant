/**
 * Tests for the Prometheus text parser -- the part most likely to be quietly wrong, per the
 * brief that asked for it. Run with `node --test src/metrics.test.ts` (Node 23.6+ strips TS
 * types natively; no ts-node, no vitest, no new dependency). Not wired into `npm run build`
 * or `npm run typecheck` -- excluded in tsconfig.app.json because `node:test`/`node:assert`
 * need `@types/node`, which this project does not depend on and this task may not add.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseMetrics,
  findSample,
  filterSamples,
  findHistogram,
  bucketQuantile,
} from "./metrics.ts";

// A slice of real output shaped like `Registry.render()`: one counter with two label sets,
// one gauge, and one histogram with three buckets plus +Inf, sum and count.
const SAMPLE = `# HELP warrant_requests_total HTTP requests served.
# TYPE warrant_requests_total counter
warrant_requests_total{endpoint="/api/ask",status="2xx"} 12
warrant_requests_total{endpoint="/api/ask",status="5xx"} 1
# HELP warrant_ready 1 when this process can answer, 0 otherwise.
# TYPE warrant_ready gauge
warrant_ready 1
# HELP warrant_stage_duration_ms Wall time per retrieval stage, as recorded on the trace.
# TYPE warrant_stage_duration_ms histogram
warrant_stage_duration_ms_bucket{stage="lexical",le="10"} 3
warrant_stage_duration_ms_bucket{stage="lexical",le="20"} 8
warrant_stage_duration_ms_bucket{stage="lexical",le="+Inf"} 10
warrant_stage_duration_ms_sum{stage="lexical"} 145.5
warrant_stage_duration_ms_count{stage="lexical"} 10
`;

test("parses counters with distinct label sets as separate samples", () => {
  const m = parseMetrics(SAMPLE);
  assert.equal(m.type["warrant_requests_total"], "counter");
  const ask = filterSamples(m, "warrant_requests_total", { endpoint: "/api/ask" });
  assert.equal(ask.length, 2);
  const twoXX = filterSamples(m, "warrant_requests_total", { endpoint: "/api/ask", status: "2xx" });
  assert.deepEqual(twoXX.map((s) => s.value), [12]);
});

test("finds an unlabelled gauge by exact (empty) label match", () => {
  const m = parseMetrics(SAMPLE);
  assert.equal(findSample(m, "warrant_ready"), 1);
  // A gauge that was never `set()` this run doesn't appear on the wire at all -- that must
  // read as "no value", not as zero.
  assert.equal(findSample(m, "warrant_uncovered_chunks"), null);
});

test("assembles a histogram's buckets, sum and count from four line-shapes", () => {
  const m = parseMetrics(SAMPLE);
  const h = findHistogram(m, "warrant_stage_duration_ms", { stage: "lexical" });
  assert.ok(h);
  assert.deepEqual(
    h!.buckets.map((b) => [b.le, b.cumulative]),
    [
      [10, 3],
      [20, 8],
      [Infinity, 10],
    ],
  );
  assert.equal(h!.sum, 145.5);
  assert.equal(h!.count, 10);
  assert.equal(findHistogram(m, "warrant_stage_duration_ms", { stage: "dense" }), null);
});

test("bucket quantile interpolates linearly within the bucket the rank falls in", () => {
  const m = parseMetrics(SAMPLE);
  const h = findHistogram(m, "warrant_stage_duration_ms", { stage: "lexical" })!;
  // rank(0.5) = 5, which falls between le=10 (cum 3) and le=20 (cum 8): 2 of that bucket's 5
  // observations in, i.e. 40% of the way from 10 to 20.
  const p50 = bucketQuantile(h, 0.5)!;
  assert.equal(p50.clipped, false);
  assert.ok(Math.abs(p50.value - 14) < 1e-9);
  // rank(0.1) = 1, inside the first bucket [0,10] holding 3: 1/3 of the way in.
  const p10 = bucketQuantile(h, 0.1)!;
  assert.ok(Math.abs(p10.value - 10 / 3) < 1e-9);
});

test("a quantile whose rank lands in the +Inf bucket is reported as a clipped lower bound", () => {
  const text = `# TYPE x histogram
x_bucket{le="10"} 1
x_bucket{le="+Inf"} 100
x_sum 500
x_count 100
`;
  const m = parseMetrics(text);
  const h = findHistogram(m, "x")!;
  const p95 = bucketQuantile(h, 0.95)!;
  assert.equal(p95.clipped, true);
  assert.equal(p95.value, 10); // "at least 10", not a number that looks precise
});

test("an empty histogram (declared, never observed) yields no quantile rather than NaN", () => {
  const text = `# TYPE x histogram
`;
  const m = parseMetrics(text);
  assert.equal(m.histograms["x"], undefined);
});

test("a histogram with all-zero buckets (declared and scraped, never observed) yields no quantile", () => {
  const text = `# TYPE x histogram
x_bucket{le="10"} 0
x_bucket{le="+Inf"} 0
x_sum 0
x_count 0
`;
  const m = parseMetrics(text);
  const h = findHistogram(m, "x")!;
  assert.equal(bucketQuantile(h, 0.5), null);
});

test("a malformed line is skipped and recorded, not thrown", () => {
  const text = `warrant_ready 1
this is not a metric line at all {{{
warrant_corpus_chunks 13212
`;
  const m = parseMetrics(text);
  assert.equal(findSample(m, "warrant_ready"), 1);
  assert.equal(findSample(m, "warrant_corpus_chunks"), 13212);
  assert.equal(m.errors.length, 1);
  assert.match(m.errors[0], /not a metric line/);
});

test("label values carrying an escaped quote, backslash and comma round-trip", () => {
  const text = `x{k="a\\"b,c\\\\d"} 1
`;
  const m = parseMetrics(text);
  assert.equal(m.samples["x"][0].labels.k, 'a"b,c\\d');
});

test("+Inf, -Inf and NaN are valid sample values, and garbage is not", () => {
  const text = `a 1
b +Inf
c -Inf
d NaN
e notanumber
`;
  const m = parseMetrics(text);
  assert.equal(findSample(m, "b"), Infinity);
  assert.equal(findSample(m, "c"), -Infinity);
  assert.ok(Number.isNaN(findSample(m, "d")));
  assert.equal(m.samples["e"], undefined);
  assert.equal(m.errors.length, 1);
});

test("duplicate metric lines for the same label set both survive as separate samples", () => {
  // Registry.render() never emits a duplicate, but a scrape torn across two writes could
  // concatenate two full bodies -- this must not silently pick one and hide the other.
  const text = `x 1
x 2
`;
  const m = parseMetrics(text);
  assert.equal(m.samples["x"].length, 2);
});

test("a HELP or TYPE line with no following samples parses without error", () => {
  const text = `# HELP warrant_uncovered_chunks Believed chunks the dense index has no vector for.
# TYPE warrant_uncovered_chunks gauge
`;
  const m = parseMetrics(text);
  assert.equal(m.type["warrant_uncovered_chunks"], "gauge");
  assert.equal(m.errors.length, 0);
  assert.equal(findSample(m, "warrant_uncovered_chunks"), null);
});

test("an unrecognised TYPE value is recorded as untyped rather than rejected", () => {
  const text = `# TYPE x summary
x{quantile="0.5"} 3
`;
  const m = parseMetrics(text);
  assert.equal(m.type["x"], "summary");
});

test("a histogram missing its _sum still reports buckets and count, with sum null", () => {
  const text = `# TYPE x histogram
x_bucket{le="+Inf"} 4
x_count 4
`;
  const m = parseMetrics(text);
  const h = findHistogram(m, "x")!;
  assert.equal(h.sum, null);
  assert.equal(h.count, 4);
  assert.equal(bucketQuantile(h, 0.5)?.value, 0); // rank(0.5)=2 <= cum 4 on the only (+Inf) bucket
});

test("an OpenMetrics EOF marker and other stray comments do not become errors", () => {
  const text = `warrant_ready 1
# EOF
`;
  const m = parseMetrics(text);
  assert.equal(m.errors.length, 0);
});

test("empty input parses to an empty, valid result", () => {
  const m = parseMetrics("");
  assert.deepEqual(m.samples, {});
  assert.deepEqual(m.histograms, {});
  assert.equal(m.errors.length, 0);
});
