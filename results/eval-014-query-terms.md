# eval-014 — Dropping high-frequency query terms

**Question.** The scale study ([eval-011](eval-011-scale.md)) found the lexical stage is the
first thing to break as the corpus grows, because `fts_query` ORs every query term together
and one common word admits every document containing it. Does dropping terms above a
document-frequency ceiling recover the latency, and what does it cost in retrieval quality?

**Answer.** It recovers 43% of lexical latency and costs a difference in sufficiency that is
not measurable. It ships **off**, and the last section says why that is the honest call rather
than a timid one.

---

## The mechanism

The FTS index is `tokenize='porter unicode61'`, and its vocabulary view reports how many
indexed documents contain each stem. On the live corpus:

| stem | documents | share of 13,212 indexed |
|---|---:|---:|
| `the` | 12,037 | **91.1%** |
| `employe` | 6,691 | **50.6%** |
| `schedul` | 1,730 | 13.1% |
| `annual` | 1,147 | 8.7% |
| `restor` | 243 | 1.8% |
| `notwithstand` | 106 | 0.8% |

Terms are ORed, so a query containing "the" MATCHes 91% of the corpus before any other term
is considered. `ORDER BY bm25(...) LIMIT :k` has no top-k pruning, so FTS5 scores all of
them. eval-011 measured the cost as exactly linear at 2.0 µs/row — 24.6 ms at 13k chunks,
1,008 ms at 500k, crossing the published full-pipeline p50 at roughly 35k rows.

## Latency

Lexical stage only, 120 held-out test items × 3 repetitions, dense and reranking off so the
measurement is of the stage that changed. The machine was running other work, so treat the
absolute numbers as an upper bound and the *ratios* as the result.

| `max_document_frequency` | p50 | p95 | vs. none |
|---|---:|---:|---:|
| none (shipped) | 25.18 ms | 40.73 ms | — |
| 0.50 | 20.44 ms | 30.79 ms | **+18.8%** |
| 0.30 | 19.47 ms | 29.43 ms | **+22.7%** |
| 0.20 | 18.42 ms | 27.96 ms | **+26.8%** |
| 0.10 | 14.29 ms | 24.03 ms | **+43.2%** |

### The first version of this table showed the filter making things *slower*

Worth recording, because it is the more instructive result. The initial implementation
called `indexed_documents()` per query to get the denominator, and that is
`SELECT COUNT(*) FROM chunk_fts` — a full scan of the FTS index, computed fresh every time
to obtain a **constant**. Measured, the filter came out 5–10% slower than doing nothing at
every threshold:

```
      none       25.95ms
      0.50       28.53ms    -9.9% vs none
      0.30       27.22ms    -4.9% vs none
      0.20       27.17ms    -4.7% vs none
      0.10       25.21ms    +2.9% vs none
```

An optimisation that is slower than the thing it replaces is easy to ship if the only number
checked is the one it was supposed to improve. Caching the count against the store's write
counter — the same invalidation the admitted set uses — is the whole difference between the
two tables.

## Quality

262 held-out test items (233 temporal, 29 human), scored with the same retriever
configuration, paired and section-clustered. `delta` is baseline minus variant, so a positive
delta means the filter lost ground.

| `max_document_frequency` | sufficiency | 95% CI | delta | 95% CI | won / lost | p |
|---|---:|:---:|---:|:---:|---:|---:|
| none | 93.5% | 89.6–96.7 | — | | | |
| 0.50 | 93.9% | 90.0–96.9 | −0.4 | −1.0–0.0 | 0 / 1 | 1.000 |
| 0.30 | 94.3% | 90.4–97.3 | −0.8 | −1.9–0.0 | 0 / 2 | 0.500 |
| 0.20 | 93.9% | 89.9–96.8 | −0.4 | −1.6–0.9 | 1 / 2 | 1.000 |
| 0.10 | 92.7% | 88.7–96.2 | +0.8 | 0.0–2.2 | 3 / 1 | 0.625 |

Nothing here is measurable. Across four thresholds, at most **four of 262 items** move in
either direction.

**This is an underpowered test, and saying "no measurable difference" is not the same as
saying "no difference".** With 262 items from 92 sections, a real 1-point harm would fail to
reach significance most of the time. What the table licenses is a bound — the filter is not
costing five points — not a claim that it costs nothing.

## Why it ships off

Every ingredient for shipping it on is present: a real latency win, no measurable quality
cost, and an implementation whose own overhead has been measured and removed. It is off
anyway, for one reason.

**At this corpus size, lexical latency is not the constraint.** Retrieval is 18.4 ms p50 and
generation is about nineteen seconds. Halving a stage that is a thousandth of a request buys
a user nothing, and it is a *ranking* change — the query genuinely differs — bought with a
quality test too small to detect a small harm. Trading an unfalsifiable quality risk for an
invisible latency gain is a bad trade, however good the percentage looks.

That changes with the corpus. eval-011 puts the crossover at roughly 35k rows, where the
lexical stage passes the whole current pipeline's p50. At 150k it is the dominant cost of
every evaluation pass, and a benchmark run that goes from 24 seconds to 14 minutes changes
how often anyone measures anything. The setting exists, is hashed into the config, and is
measured, so the decision at that point is a config change with a number already attached,
rather than a fresh investigation under time pressure.

`retrieve.max_document_frequency: 0.20` is the recommended value when that day comes: it
keeps `schedul` at 13.1% and drops `employe` at 50.6%, takes 27% off the stage, and is the
threshold where the paired comparison is flattest.

## Reproducing

```bash
python -m warrant.cli eval gate --record        # floor for the shipped configuration
# then set retrieve.max_document_frequency in configs/default.yaml and
python -m warrant.cli eval gate                 # reports incomparable; the hash moved
```

The gate refusing to compare across this setting is correct and is the point: it is a
ranking change, and a floor recorded without it does not describe a run with it.
