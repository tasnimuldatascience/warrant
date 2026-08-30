# Eval 003 — corrected statistics, and a claim withdrawn

**Date:** 2026-08-30
**Supersedes:** the confidence intervals and the reranker conclusion in
[eval-002](eval-002-failure-budget.md). The failure-budget mechanism and the as-of ablation
in that document stand and are strengthened here.
**Reproduce:** `make build && make index && make eval`

Six independent audits went at this repository. The measurement layer reproduced to the
digit; three methodological defects and one corpus defect did not survive.

## What changed in the corpus

The chunker read only `<P>` elements. eCFR also carries body text in flush paragraphs
(`FP`, `FP-1`, `FP-2`, `FP-DASH`) and in `TABLE`, and **18,705 words — 4.5% of the corpus —
were never ingested**: 88% of §532.313, 46% of §531.214, concentrated in the Federal Wage
System parts the applicability story is built on.

That loss was invisible to every instrument here, and necessarily so. The failure budget's
`ingestion` row asks whether a gold chunk is in the store, and gold chunks are minted by the
same parser — **text the parser never emitted could never be missed, so the row could only
ever read zero.** A row that cannot be non-zero is not measuring anything.

Corpus is now 13,145 chunk versions (was 12,858), coverage 96.8% of section body text, with
a CI invariant asserting a 95% floor so it cannot silently regress.

## What changed in the statistics

### Intervals were 3–4× too narrow

The temporal bucket's items are not independent trials. 737 items come from **95 sections**,
and one section — §531.603, on locality pay areas — supplies over a third of them. An
item-level bootstrap treats them as independent and reports an interval far tighter than the
data supports.

| | published in eval-002 | corrected (section-clustered) |
|---|:---:|:---:|
| temporal sufficiency | 76.4% (73.2–79.3) | **76.9% (68.3–93.9)** |

The point estimate barely moved. The interval is about 3.5× wider, and that is the honest
resolution of this benchmark: differences under roughly nine points cannot be called.
eval-002's "differences under about three points are inside the noise" understated it by a
factor of three.

### Every 0.0% distractor rate was a tautology

Zero of the temporal items have their distractor admitted by the as-of predicate, and zero of
the scope-exclusion items have theirs admitted by the applicability predicate. The two
versions' validity intervals are disjoint by construction and the excluded part is removed by
a `NOT IN` clause. **The distractor could not be retrieved because it was never a
candidate** — `0.0%` restated the WHERE clause.

Those cells are now marked `*` and read *by construction*. The ablation rows, where the
predicate is switched off and the distractor genuinely becomes reachable, are the real
measurement and they are untouched: **62.8% wrong-version citations without the as-of
predicate, 100.0% without applicability.**

Boundary intervals were also wrong. A percentile bootstrap over an all-true vector returns
zero width, so `100.0–100.0` was published for 130/130. Wilson gives 97.1–100.0.

### Comparisons were unpaired, and one does not survive pairing

Every configuration is scored on identical items, so the comparisons are paired and reading
two marginal intervals for overlap discards most of the resolution. Paired, section-clustered:

| removing | delta | 95% CI | won | lost | p | verdict |
|---|---:|:---:|---:|---:|---:|---|
| the as-of predicate | **+13.8** | 3.2 – 21.7 | 105 | 3 | 1.3e-27 | **carries its weight** |
| the cross-encoder | +0.5 | −2.4 – 2.2 | 68 | 64 | 0.79 | **not measurable** |

The as-of predicate survives properly-clustered paired inference comfortably — the pairing
makes that claim *stronger*, not weaker.

**The cross-encoder does not.** This overturns a published reading. eval-002's budget charged
`rerank` with 64 of 170 failures — the plurality, 37.6% — and named it the next intervention
target. That accounting is one-sided: it counts the 64 items the reranker demoted out of the
final k and ignores the 68 it promoted in. Net, the reranker moves this bucket by half a
point with p = 0.79, while costing roughly 80% of retrieval latency and ~2 GB of a co-resident
8 GB budget.

The failure budget diagnosed exactly this class of error one revision earlier — the
reranker-versus-truncation reattribution — and stated the principle: *a budget row and a
direct ablation disagreeing is the argument for running both*. It was not applied to the row
that survived. Running both is now part of `make eval`.

## What this does not change

- The failure budget mechanism, and the before/after shift it drove (236 → 170 failures).
  Widening the fused head and final cut is confirmed at **+9.2 points** paired.
- Every number in [spike-001](spike-001-amendment-viability.md).
- The applicability ablation: 100.0% wrong-part citations without the predicate.
- The corpus, ingest, and bitemporal invariants.

## Open, and stated rather than hidden

- **The headline is tuned on the evaluation set.** `rerank_top_k` and `final_k` were chosen
  by reading the failure budget over the same items the result is reported on. There is no
  dev/test split. Until there is, 76.9% is a development number.
- **`shared_query` is not version-neutral**, contrary to its own docstring. Measured over 349
  pairs with the predicate off, the before-side outranks the after-side 222 to 127
  (p < 1e-6): the term selection walks the before text in document order, and BM25 length
  normalisation favours the shorter pre-amendment paragraph. Matched-pair, the effect on the
  headline is not significant (p = 0.44), but the ablation digits inherit it.
- **36% of temporal items share a query with an item that has different gold**, because for
  short amendments the shared vocabulary degenerates — 21 pairs reduce to the section heading
  alone. Those items cannot in principle be answered correctly.
- **`generated` at 100% is saturated at k=1** and cannot discriminate. It is a reachability
  assertion, not a benchmark row.
- **`acceptable_evidence` is a singleton everywhere**, so "sufficiency" is exactly recall@k.
  Pooling is designed and not built.
- The `service` scope facet is never exercised: no part offers a contrasting value, so those
  items are silently skipped and the bucket covers 5 of 8 restricted parts.
