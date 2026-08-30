# Eval 004 — held-out results, and the generator measured

> **Superseded serving figures.** This report quotes 21.3 tok/s and a 3-per-minute
> ceiling. Both are wrong: an isolated re-derivation measured 29.2–29.9 tok/s over
> ~205 output tokens, so an answer is 6.6 s and the ceiling is 7.7 req/min. The text
> below is left as it was written, because a results doc is a record of what was
> measured on a day, and editing it to agree with a later number falsifies that
> record. See [eval-010](eval-010-capacity.md).

**Date:** 2026-08-30
**Supersedes:** the bucket numbers in [eval-002](eval-002-failure-budget.md) and
[eval-003](eval-003-corrected-statistics.md). The mechanisms in both stand; the numbers moved
because the benchmark got stricter and a split now exists.
**Reproduce:** `make build && make index && make eval && make autopsy && make generation`

Everything below is on the **held-out test split**. Retrieval parameters were chosen by
reading the failure budget, so the budget runs on `dev` and results are reported from `test`
— `make autopsy` and `make eval` default to opposite sides for that reason. The split is by
**section**: two sides of one amendment share a query, so an item-level split would put one
in each half and leak the answer across the boundary.

## Corpus

13,145 chunk versions · 26 parts of 5 CFR chapter I · 96.8% of section body text captured,
asserted by a CI invariant at a 95% floor.

## Buckets

| bucket | n | sections | sufficiency | 95% CI | wrong version |
|---|---:|---:|---:|:---:|---:|
| temporal | 229 | 47 | **97.8%** | 94.9–100.0 | 0.0%* |
| human | 29 | 22 | 79.3% | 61.8–96.2 | 0.0%* |
| scope | 42 | 42 | 100.0% | 97.0–100.0 | 0.0%* |
| scope-exclusion | 42 | 42 | n/a | | 0.0%* |

\* enforced by construction — the distractor was never admitted by the predicates, so the
rate restates the query. The ablations below are where those predicates are actually
measured. Intervals are a section-clustered bootstrap.

The temporal bucket rose from 76.9% to 97.8% not because retrieval improved but because the
benchmark stopped containing unanswerable items. Mining now discards, and reports, what it
drops:

| discarded | count | why |
|---|---:|---|
| no counterpart | 647 | a pure addition or deletion cannot test temporal discrimination |
| short query | 192 | fewer than 5 shared content terms — the query cannot identify its own gold |
| tiny change | 147 | below the substantive threshold |
| ambiguous | 8 | identical `(query, as_of)` with a different gold; at least one must fail |
| duplicate | 3 | same query, same date, same gold — one trial counted twice |

The `generated` bucket is gone. Its queries were built from the paragraph they retrieve, it
scored 100.0% and 97.7% at k=1, and no configuration could lose a point. It survives as a
corpus reachability invariant, which is what it always was.

## The predicates, paired and clustered

Both measures, because a predicate can be decisive on one and invisible on the other:

| removing | measure | delta | 95% CI | won / lost | p | verdict |
|---|---|---:|:---:|---:|---:|---|
| as-of predicate | sufficiency | +2.2 | 0.0–5.5 | 5 / 0 | 0.063 | not measurable |
| as-of predicate | **no wrong version** | **+96.1** | 92.8–99.5 | **220 / 0** | 1.2e-66 | **carries its weight** |
| cross-encoder | sufficiency | +1.3 | 0.0–4.0 | 4 / 1 | 0.375 | not measurable |
| cross-encoder | no wrong version | +0.0 | 0.0–0.0 | 0 / 0 | 1.0 | not measurable |

**Sufficiency alone would have called the as-of predicate useless.** With `final_k: 16`
several versions of a section fit in the result list at once, so removing the predicate barely
moves whether the right paragraph is present — it changes whether the wrong one is present
beside it. Reporting one measure would have dismissed the predicate on the very bucket built
to prove it works.

The cross-encoder remains unmeasurable on both, confirming eval-003.

## The generator

Nothing measured this before. Held-out human items, 29 questions:

| measure | value | 95% CI |
|---|---:|:---:|
| claims emitted | 67 | |
| hallucination rate | **1.5%** | 0.3–8.0 |
| citation precision | **98.5%** | 92.1–99.7 |
| unparseable responses | 0 of 29 | |

| abstention | count | |
|---|---:|---|
| answered, evidence present | 23 | correct |
| abstained, evidence absent | 0 | correct |
| abstained, evidence present | 0 | wrong — it had the answer |
| **answered, evidence absent** | **6** | **wrong — answered anyway** |

**The model never abstains.** A 1.5% hallucination rate only means something if the system
also declines when it should, and on 6 of 29 questions it wrote an answer from context that
did not contain sufficient evidence. That is the dangerous quadrant in a regulatory system,
and it is now a measured number rather than an assumption. Abstention is the next thing to
fix, and the calibrated verifier in ARCHITECTURE section 5 is what would fix it.

## Latency against quality

Per-stage wall clock from the trace, same 60 queries, same items:

| configuration | p50 | p95 | sufficiency | wrong version |
|---|---:|---:|---:|---:|
| lexical only | **23.8 ms** | 31.5 ms | 96.7% | 0.0% |
| + dense | 41.0 ms | 51.2 ms | 96.7% | 0.0% |
| + cross-encoder | 71.0 ms | 131.1 ms | 98.3% | 0.0% |

On this bucket **lexical-only is three times faster at the same quality**, and the
cross-encoder buys 1.6 points for 30 ms and a 2 GB co-resident model. This is the frontier
ARCHITECTURE section 10 says a shedding policy must be decided from; before this run there
was no measurement to decide from, and the policy was a paragraph.

Generation is a different order of magnitude: 21.3 tokens/s unbatched, so a full answer is
~20 s and the serving ceiling is **three requests per minute**. The API admits under a
semaphore and refuses with 503 + `Retry-After` rather than queueing silently.

## Failure budget

Run on **dev** — 119 items, 3 failures, 97.5% satisfied: `rerank` 2, `truncation` 1.
`ingestion`, `applicability` and `temporal` are all zero, and the ladder now runs through
`generation` and `grounding` so a model failure is attributable rather than invisible.

The budget is thin now, which is the point: it was the instrument that found the items it is
no longer catching.

## Still open

- **Pooling is not implemented.** Every evidence set is a singleton, so "sufficiency" is
  exactly recall@k. Named honestly rather than renamed.
- **`shared_query` bias is mitigated, not eliminated.** Terms are now ordered by corpus
  document frequency, which is symmetric; BM25 length normalisation still favours the shorter
  pre-amendment paragraph and that is a property of the scorer, not the query.
- **The corpus is a live target.** Numbers are against eCFR as of issue date 2026-08-25.
  A run next month ingests amendments published since and the bucket sizes will differ.
  Model revisions are pinnable in config and are unpinned by default.
