# Eval 011 — does any of this hold at 10x?

**Date:** 2026-08-30
**Module:** `src/warrant/bench/scale.py`
**Reproduce:**

```bash
# the sweep, and the diagnostics that name the causes
python -m warrant.bench.scale --out runs/scale --sizes 13145,50000,150000,500000 \
    --queries 60 --seed 1 --diagnose \
    --real runs/scale/copy-of-warrant.sqlite3 --real-index data/dense

# the anchor alone, at the query count the calibration table was taken with
python -m warrant.bench.scale --out runs/anchor --sizes 13145 --queries 150 --seed 5 \
    --real runs/scale/copy-of-warrant.sqlite3 --real-index data/dense
```

No network, no paid service, no new dependency. `--real` takes a **copy** of the store rather
than the store: the sweep only reads, but a benchmark that can only be trusted because of what
its code does is a benchmark one bug away from destroying the corpus it is measuring. Peak
resident memory at 500k is about 3 GB and the scratch store is 424 MB.

---

## The question

Every number in `results/` was taken against 13,145 chunk versions across 26 parts of 5 CFR.
Four design decisions are size-dependent and none of them had been measured:

| | |
|---|---|
| `Store.candidate_ids` | materialises the admitted set as a Python `set[int]`, and caches up to 64 of them keyed on (as-of, sources, authority) |
| `DenseIndex.search` | exact: a full `(n, 384)` matmul per query, predicate applied by `np.isin` |
| `Store.search` | `ORDER BY bm25(chunk_fts) LIMIT :k` over FTS5 |
| the benchmark and the bootstrap | scale with the corpus |

Fetching more of the CFR is a ten-minute network job per part and would not answer the
question sooner. So `bench/scale.py` generates a corpus with the **statistical shape** of the
real one at an arbitrary size, and measures the mechanism against it.

### What a synthetic corpus licenses, and what it does not

It bounds **cost**: index size, build time, per-stage latency, resident memory. Those depend
on how many rows there are, how long they are, how many vocabulary types there are, and how
postings are spread across them — all of which are reproduced here, and none of which care
whether the sentences mean anything.

It says **nothing about retrieval quality at scale.** Recall against synthetic questions over
synthetic prose measures the generator, not the system. No sufficiency, no wrong-version rate
and no failure budget appears anywhere in this document, and none should be inferred from it.
Whether the as-of predicate still moves the wrong-version rate by 96 points at 150k needs real
text and real questions; that is a different job and this is not a down payment on it.

One stage is worse than merely unmeasured for quality: the **cross-encoder's cost depends on
the text**, not on the corpus size, and synthetic pseudo-words tokenise into more subword
tokens than English does. The reranker figure below is inflated on synthetic input and is
reported for the real store only.

---

## Calibration: the generator against the real store

The shape is measured from `data/warrant.sqlite3` and baked into `CorpusShape.REAL_5CFR`
(`CorpusShape.measure` recomputes it from any store). The generator reproduces the token
length distribution, the section/paragraph nesting, the number of valid-time versions per
paragraph, the in-force fraction, the snapshot-date growth, and the vocabulary's Zipfian
character with Heaps-law growth. The anchor point of every sweep is the **real** store, run
through the identical harness, so the two can be compared instead of assumed equivalent.

| | real 5 CFR | synthetic, same size |
|---|---:|---:|
| chunk versions | 13,145 | 13,145 |
| paragraphs | 10,185 | 10,152 |
| sections | 1,320 | 1,286 |
| parts | 26 | 19 |
| distinct snapshot dates | 66 | 67 |
| in force | 9,961 | 9,923 |
| vocabulary types | 17,173 | 17,205 |
| tokens | 514,629 | 513,908 |
| tokens/chunk mean | 39.15 | 39.10 |
| median / p10 / p90 / max | 29 / 9 / 79 / 1013 | 29 / 9 / 78 / 1001 |
| under 30 tokens | 50.3% | 50.9% |
| "the", share of tokens | 6.92% | 6.92% |
| db on disk | 10.70 MB | 11.2 MB |
| FTS5 index | 1.66 MB | 2.1 MB |
| **rows an `fts_query` matches** | **88.4%** | **90.4%** |
| admitted set | 0.80 MB | 0.80 MB |
| full 64-entry cache | 50.7 MB | 51.8 MB |

Latency at the anchor, re-run on a quiet machine at 150 queries a side, with 95% bootstrap
intervals (this is the one place a tight number was worth waiting for):

| stage, p50 ms | real 5 CFR | synthetic |
|---|---:|---:|
| predicates, cold | 6.55 [6.49, 6.65] | 7.05 [6.89, 7.36] |
| lexical | 24.55 [23.73, 25.12] | 25.35 [24.99, 26.16] |
| lexical, content-only | 12.00 [11.13, 12.90] | 8.53 [8.22, 8.74] |
| dense | 2.23 [2.21, 2.27] | 2.32 [2.27, 2.38] |
| `np.isin` predicate mask | 0.22 | 0.24 |
| fusion | 0.18 | 0.18 |

The real store's 24.55 ms lexical p50 lands on the 23.8 ms `README.md` publishes for the
lexical-only configuration, which is the check that the harness is timing the same thing the
rest of the project timed.

Three rows disagree by more than noise, and each disagrees in a stated direction:

- **FTS5 index, +27%.** The porter tokeniser collapses real English morphology —
  *employee/employees* stem together — and independently generated pseudo-words do not. The
  synthetic index is conservative, which is the right direction for a bound.
- **Content-only lexical, −29%.** Beyond the empirically-fixed top 100, the synthetic
  vocabulary is pure Zipf, and real English has a fatter mid-frequency band than Zipf. So the
  content-only *control* understates real cost, which makes the stopword remedy below look
  slightly better than it is. The natural-query column, which is the one the findings rest
  on, agrees to 3.3%.
- **Parts, 19 against 26.** Sections-per-part is drawn from a 26-value table, and 19 draws
  from a distribution running 7 to 190 are noisy. It converges: at 500k the generator
  produces 49.76 sections per part against a measured 50.77, and 7.81 paragraphs per section
  against 7.72.

The contended sweep below reported dense at 3.81 ms synthetic against 2.47 ms real, on two
matrices of identical shape running identical code. The quiet re-run puts them at 2.32 and
2.23. That gap was the machine, and it is the reason the anchor was re-measured rather than
explained.

Seed sensitivity, three seeds at 50,000 rows: lexical p50 90.96 / 92.69 / 87.38 ms,
predicates cold 24.68 / 24.60 / 22.51 ms, FTS index 7.65 / 7.83 / 7.89 MB. Seed variation is
smaller than the run-to-run variation from machine contention.

**Contention warning.** The sweep below was taken with six other agent processes on the
machine. Two full sweeps at identical settings differed by 5–10% on every latency, in the same
direction. The p95 columns are contended and should be read as upper bounds, not as this
system's p95. Only the anchor table above was re-taken on a quiet machine; re-running the
whole sweep quietly would move every row down by a few per cent and change no conclusion,
because the findings are ratios and slopes rather than absolute milliseconds. The published
numbers this document compares against (`README.md`: 23.8 / 41.0 / 71.0 ms p50, 131.1 ms p95)
were themselves taken on a contended machine — `docs/RESUME.md` says so.

---

## The curves

### Build and disk

| corpus | rows | parts | in force | build s | rows/s | db MB | FTS MB | text MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real 5 CFR | 13,145 | 26 | 9,961 | — | — | 10.7 | 1.7 | 3.7 |
| synthetic | 13,145 | 19 | 9,923 | 1.9 | 6,890 | 11.2 | 2.1 | 3.9 |
| synthetic | 50,000 | 105 | 37,892 | 7.3 | 6,839 | 42.6 | 7.6 | 14.9 |
| synthetic | 150,000 | 292 | 113,888 | 23.9 | 6,278 | 127.7 | 23.0 | 44.8 |
| synthetic | 500,002 | 996 | 378,749 | 80.1 | 6,246 | 424.0 | 74.1 | 149.7 |

Insert throughput falls 9% across a 38x range — the FTS5 `AFTER INSERT` trigger and the six
b-tree indexes cost a slowly growing amount per row, and nothing here is superlinear. 848
bytes of database per chunk version, of which 148 are the FTS5 index. **Ingest does not
break.** A 5-million-row store is a quarter-hour build and 4.2 GB on disk.

Embedding is not measured: the vectors here are random, so no encoder runs. `README.md`
reports ~10 s for 13,145 chunks on a laptop GPU; that is linear in rows and extrapolates to
about six minutes at 500k. It is an extrapolation, not a measurement.

### Stage latency (ms, p50 / p95, 60 queries per point)

Natural queries — drawn from the corpus's own unigram distribution, so they carry function
words, which is what a real question does:

| corpus | predicates cold | predicates warm | lexical | dense | `isin` alone | fusion |
|---|---:|---:|---:|---:|---:|---:|
| real 5 CFR | 8.08 / 10.05 | 0.02 / 0.03 | 28.34 / 35.27 | 2.47 / 6.35 | 0.26 / 0.41 | 0.20 / 0.28 |
| synthetic 13,145 | 7.94 / 10.73 | 0.02 / 0.04 | 28.73 / 32.75 | 3.81 / 6.73 | 0.27 / 0.46 | 0.19 / 0.30 |
| synthetic 50,000 | 29.83 / 41.19 | 0.03 / 0.04 | 97.08 / 117.44 | 9.89 / 20.18 | 1.24 / 1.89 | 0.18 / 0.30 |
| synthetic 150,000 | 97.37 / 127.59 | 0.03 / 0.05 | 307.85 / 388.14 | 24.98 / 44.01 | 4.07 / 8.14 | 0.20 / 0.34 |
| synthetic 500,002 | 335.92 / 358.31 | 0.03 / 0.07 | **1007.81 / 1181.21** | 82.31 / 101.32 | 14.02 / 16.64 | 0.21 / 0.29 |

95% bootstrap intervals on the two that matter, at 500k: lexical p50 973.7–1033.2, p95
1107.0–1244.8; dense p50 80.5–85.8, p95 94.0–108.8.

Content-only queries — the same draw with the top 100 ranks removed. Not a realistic
workload; it is the control that separates *the corpus got bigger* from *`fts_query` ORed
"the" into the query*:

| corpus | lexical p50 / p95 | vs natural |
|---|---:|---:|
| real 5 CFR | 15.32 / 23.06 | 0.54x |
| synthetic 13,145 | 10.64 / 13.38 | 0.37x |
| synthetic 50,000 | 28.06 / 40.03 | 0.29x |
| synthetic 150,000 | 82.82 / 119.21 | 0.27x |
| synthetic 500,002 | 265.09 / 363.28 | 0.26x |

Cross-encoder, real store, 30 queries, 50 pairs per query: **49.1 ms p50, 181.3 ms p95.** It
scores exactly `rerank_top_k` pairs whatever the corpus size, so it is O(1) in rows by
construction; the synthetic measurements (78.5 ms at 13k, 90.1 ms at 150k) are inflated by
subword tokenisation of pseudo-words and are not evidence either way.

Every stage is linear in rows. Log-log slopes over the full 13k–500k range: lexical 0.98,
cold predicates 1.03, dense 0.84 — and 0.92 for dense over 50k–500k, the difference being a
fixed ~1.7 ms floor that dominates at the small end. Marginal cost per chunk version, from
the 150k–500k segment:

| stage | µs per row |
|---|---:|
| lexical, natural query | 2.00 |
| cold predicate scan | 0.68 |
| lexical, content-only | 0.52 |
| dense (matmul + mask) | 0.164 |
| `np.isin` mask alone | 0.028 |

### Memory

| corpus | admitted | admitted set | full 64-entry cache | RSS delta, same | dense matrix | process RSS |
|---|---:|---:|---:|---:|---:|---:|
| real 5 CFR | 9,961 | 0.80 MB | 50.7 MB | 49.8 MB | 20.2 MB | 80.9 MB |
| synthetic 13,145 | 9,923 | 0.80 MB | 51.8 MB | 42.4 MB | 20.2 MB | 85.2 MB |
| synthetic 50,000 | 37,892 | 3.16 MB | 203.6 MB | 199.5 MB | 76.8 MB | 153.6 MB |
| synthetic 150,000 | 113,888 | 7.38 MB | 477.1 MB | 500.8 MB | 230.4 MB | 316.2 MB |
| synthetic 500,002 | 378,749 | 27.38 MB | **1,767.8 MB** | 1,870.8 MB | 768.0 MB | 936.8 MB |

The cache is weighed analytically — `sys.getsizeof` over every set and every element — because
`tracemalloc` records a traceback per allocated block and a full cache at 500k is 24 million
live `int` objects. The RSS delta column is the independent check and agrees within 6%
everywhere; at the 13k anchor `tracemalloc` was affordable and agreed too.

**72.3 bytes per admitted row**, of which 28 are the `int` object and the rest is the set's
open-addressed table. 1,536 bytes per row for the dense matrix.

---

## The first thing that breaks: lexical retrieval, between 2.7x and 5x

Not "it gets slower". The component is `Store.search`, and the reason is one clause:

```sql
SELECT c.*, bm25(chunk_fts) AS score FROM chunk_fts JOIN chunk c ON …
WHERE chunk_fts MATCH :q … ORDER BY score LIMIT :k
```

`LIMIT :k` does not bound the work. FTS5 has no top-k pruning on this plan: to honour
`ORDER BY bm25(...)` it must score **every** row the MATCH selects. And `fts_query` ORs every
query token together, function words included, so a ten-token question selects nearly the
whole corpus:

| corpus | rows an `fts_query` matches | as a fraction |
|---|---:|---:|
| real 5 CFR | 11,620 | 88.4% |
| synthetic 13,145 | 11,884 | 90.4% |
| synthetic 50,000 | 45,300 | 90.6% |
| synthetic 150,000 | 135,700 | 90.5% |
| synthetic 500,002 | 452,300 | 90.5% |

The fraction is flat, so the work is exactly linear in corpus size, and `candidates_lexical:
100` buys nothing back. At 500k a query that returns 100 rows has scored 452,300.

Against the latency frontier this project already published — lexical-only 23.8 ms p50, full
pipeline 71.0 ms p50 and 131.1 ms p95 — the lexical stage alone, at 2.00 µs per row:

| crosses | at | multiple of today |
|---|---:|---:|
| the current full-pipeline p50 (71.0 ms) | ~35,000 rows | 2.7x |
| the current full-pipeline p95 (131.1 ms) | ~65,000 rows | 4.9x |
| 1 second p50 | ~500,000 rows | 38x |

**So the answer to "does it hold at 10x" is: this stage does not, and it stops holding at
about 3x.** Nothing else measured here breaks before 10x.

### What it breaks, precisely

Not the serving path. Generation runs at 21.3 tokens/s unbatched — roughly 20 seconds an
answer, three requests a minute — so even a full second of retrieval at 500k is 5% of what a
user waits for, and the API's admission-control semaphore is sized by generation, not by this.

What it breaks is the **evaluation apparatus**, which runs retrieval thousands of times with
no generator attached and is this project's actual product. One pass over the 851 mined
benchmark items:

| corpus | one pass |
|---|---:|
| 13,145 | 24 s |
| 50,000 | 83 s |
| 150,000 | 4.4 min |
| 500,002 | 14.3 min |

`make eval` runs several passes for the paired ablations, and the interventional autopsy runs
a depth sweep on top of that. A failure budget that takes an afternoon is a failure budget
nobody reruns after a commit, and ARCHITECTURE.md section 7 is explicit that a budget which
does not redirect the next commit is decoration.

---

## Second: the admitted-set cache, which breaks on memory rather than on time

The cache works. Warm predicate latency is **0.02–0.03 ms at every size measured**, flat
across a 38x range, against a cold scan that reaches 335.92 ms. The 9.74 ms → 0.02 ms
improvement recorded in `docs/RESUME.md` holds at 500k as 335.92 ms → 0.03 ms.

It costs two things.

**Resident memory: 50.7 MB today, 1.77 GB at 500k.** That is 3.54 KB of cache per chunk
version in the store. The hardware envelope in ARCHITECTURE.md section 13 budgets ~6.5 GB of
co-resident model weights on a 31 GB machine; the cache alone reaches that at about 1.8M rows,
and at 500k it is already 2.3x the dense matrix it exists to filter.

The representation is the whole cost. The same predicate over the same 378,749 admitted rows,
three ways — measured at every size in the sweep, 500k shown:

| | bytes per entry | a full 64-entry cache | time to apply, p50 |
|---|---:|---:|---:|
| `set[int]` + `np.isin` — what the code does | 27.38 MB | 1,767.8 MB | 15.51 ms |
| sorted `int64` array + `searchsorted` | 3.03 MB | 194 MB | 25.40 ms |
| **`bool` mask indexed by row id** | **0.50 MB** | **32 MB** | **0.29 ms** |

55x less memory and 53x less time than what is there now. The obvious-looking middle option is
*worse* on time than the thing it would replace, which is why it is in the table: `np.isin`
already sorts internally, and re-implementing that by hand only removes the C loop.

**A flush stall.** `Store.candidate_ids` does not evict, it clears:
`if len(self._admits) >= 64: self._admits.clear()`. One request in every 65 at a fresh as-of
date therefore frees the entire cache — 24 million `int` objects at 500k — on top of the cold
scan it was already paying for:

| corpus | cold scan, empty cache | cold scan that triggers the flush | the flush |
|---|---:|---:|---:|
| real 5 CFR | 5.81 ms | 12.36 ms | +6.5 ms |
| synthetic 13,145 | 6.05 ms | 15.97 ms | +9.9 ms |
| synthetic 50,000 | 30.68 ms | 64.49 ms | +33.8 ms |
| synthetic 150,000 | 112.3 ms | 203.3 ms | +91.0 ms |
| synthetic 500,002 | 358.0 ms | 593.8 ms | **+235.8 ms** |

The key space grows too, but sublinearly: distinct snapshot dates go 66 → 1,384 from 26 to 996
parts, and the calendar bounds it at ~3,500 for this window. 64 entries covers 4.6% of the
as-of dates a 500k corpus would carry, so the flush is not a rare event at scale.

The garbage collector was the first suspect and is innocent: with a full cache resident, warm
predicate latency is 0.0021 ms with GC enabled and 0.0019 ms with it disabled. An earlier
version of this harness reported 3.4 ms for the warm path at 500k; that was the harness
holding the previous query's set alive into the timed region, and freeing 379,000 `int`
objects on the rebind. It is fixed and there is a test for it — the same mistake in the
opposite direction is exactly what a measurement of a cache is prone to.

---

## Exact dense search does not break by 500k

| corpus | dense p50 | of which `isin` | dense matrix |
|---|---:|---:|---:|
| real 5 CFR (quiet machine) | 2.23 ms | 0.22 ms | 20.2 MB |
| synthetic 13,145 | 3.81 ms | 0.27 ms | 20.2 MB |
| synthetic 50,000 | 9.89 ms | 1.24 ms | 76.8 MB |
| synthetic 150,000 | 24.98 ms | 4.07 ms | 230.4 MB |
| synthetic 500,002 | 82.31 ms | 14.02 ms | 768.0 MB |

At 0.164 µs per row, an exhaustive scan reaches 25 ms — today's entire published lexical-only
p50 — at about 150,000 rows, and 131.1 ms at about 800,000. It is never the largest stage at
any size measured: 9.0% of retrieval at 13k, 8.2% at 500k. Even with the lexical stage fixed
to its content-only cost, dense at 500k is 82 ms against lexical's 265 ms.

`retrieve/dense.py` justifies masking scores rather than gathering the admitted rows on a
measurement taken at 13,145, where 76% of rows are admitted. It holds at 38x, and by a wider
margin: at 500k, masking is 52.4 ms against 151.7 ms to gather. The comment is right for the
reason it gives.

It stops being right when the admitted fraction falls. Measured at 150,000 rows, by masking
random subsets:

| admitted | mask | gather |
|---:|---:|---:|
| 100% | 15.0 ms | 56.0 ms |
| 75.8% (today's as-of filter) | 15.1 ms | 43.9 ms |
| 50% | 16.5 ms | 31.3 ms |
| 35% | 15.1 ms | 22.7 ms |
| 25% | 15.1 ms | 16.6 ms |
| 15% | 15.4 ms | 10.2 ms |
| 5% | 14.9 ms | 3.1 ms |

**The crossover is at 23% admitted.** The as-of predicate alone admits 76%, so it never gets
near it — but `sources` and `max_authority` are now in the query path
(`_authority_clause`), and `sources=["usc"]` against a store that is mostly eCFR is a filter
in the single-digit percents. That configuration is *already reachable today* and would make
gathering 5x cheaper than masking. It is a branch on the admitted fraction, not a rewrite.

### What an ANN index would have to buy

An ANN index would have to be worth more than the 82 ms it can save at 500k, at which size the
same pipeline is spending 1,008 ms in lexical retrieval and 20 seconds in generation. It is
not the first thing to fix at 500k; it is not the second; and **at 13,145 chunk versions,
recommending one would be wrong** — it could save at most 2.2 ms of a 24.8 ms retrieval path,
in exchange for a dependency, a build step, and a recall loss that would land squarely on the
metric this project reports.

There is also an architectural cost specific to this system. The predicate is pushed *into*
the search (ARCHITECTURE.md sections 3 and 5): `DenseIndex.search` masks before the top-k, so
nothing inadmissible can reach the result. A graph index cannot do that. It either
post-filters — which requires over-retrieving by a factor that depends on the filter's
selectivity, and the filter here ranges from 76% down to single digits — or it needs native
filtered search, which degrades exactly when the filter is narrow. Adopting one would trade a
hard invariant (ARCHITECTURE.md section 9: every retrieved chunk's validity interval contains
the as-of date) for a probabilistic one.

The condition under which it would become the right call, stated so it can be tested rather
than argued: **the lexical stage is fixed, the corpus is past ~1M rows, and the target is a
sub-200 ms retrieval p50.** Before that, two cheaper levers are unexhausted: gather-then-score
below 23% admitted (measured above), and shrinking the matrix. The second is untested here —
float16 would halve 768 MB, but numpy has no BLAS path for it and the time effect could go
either way, so it is a direction, not a recommendation.

---

## Remedies

### Worth doing at 13,145 rows

**1. Stop ORing high-document-frequency terms in `fts_query`.** This is the only remedy whose
case is made on the *real* corpus at its *real* size, on a quiet machine: lexical p50 is
**24.55 ms with function words and 12.00 ms without**, on the store in `data/`. It is the
largest stage in the pipeline and half of it is spent scoring the postings list for "the".

It is a **ranking change, not a latency change**, and it must be measured as one. Dropping a
term changes BM25's scores, and 48.2% of in-force chunks are under 30 tokens — for those,
function words are a meaningful share of what there is to match on. The gate is the held-out
split and the failure budget, not this document.

**2. Do not add an ANN index.** Stated as a recommendation because it is the conclusion a
reader would otherwise reach from "dense search is exhaustive". See above.

### Worth doing only if the corpus grows

| remedy | pays from | why not now |
|---|---|---|
| `bool` mask instead of `set[int]` for the admitted set | ~50k | saves 50 MB and 0.25 ms today; the change touches `Store.candidate_ids` and `DenseIndex.search` together, and 50 MB is not worth coupling them for |
| LRU eviction instead of `_admits.clear()` | ~50k | costs 6.5 ms on one request in 65 today, against a 20 s generation |
| a covering index for the predicate scan — `(system_to, valid_from, valid_to, id)`; `chunk_asof` leads with `section_id` and cannot serve it | ~150k | the cold scan is 6.6 ms today and the cache hides it |
| gather-then-score in `DenseIndex.search` below 23% admitted | whenever `sources`/`max_authority` narrows the filter, at **any** size | the as-of predicate alone never gets near the crossover |
| an ANN index | not at 500k; see the stated condition | — |

**Nothing about ingest, build time, disk, or fusion needs anything.** Fusion is 0.2 ms at
every size — it consumes two 100-element rank lists and does not know how large the corpus is.

---

## Extrapolated ceiling

With the lexical stage left as it is, retrieval p50 passes one second at ~500k rows and ten
seconds at ~5M — at which point retrieval and generation cost the same, and the whole shape of
the serving argument changes. With the stopword fix and the mask representation, the same
budget reaches ~2M rows before lexical crosses one second, and the binding constraint becomes
the dense matrix's 1,536 bytes per row: 3 GB at 2M rows, co-resident with 6.5 GB of models on
a 31 GB machine. That is comfortable, and it is a memory ceiling rather than a latency one.

None of that is a measurement. It is linear extrapolation from measurements that were linear
across 38x, which is the strongest form the claim can honestly take.

---

## What was not measured

- **Retrieval quality at any size but 13,145.** By construction; see the top of this document.
- **Concurrency.** Every number here is single-threaded and single-request. The pipeline runs
  lexical and dense in a two-thread pool, and several concurrent requests share one SQLite
  file and one dense matrix. `src/warrant/bench/load.py` is the module for that question.
- **Growth in history depth.** The corpus is grown in *breadth* — more parts — because that is
  what "more of the CFR" means. A corpus grown by moving the history floor back from 2017
  would have more versions per paragraph and a *lower* admitted fraction, which moves the
  mask/gather crossover into range and makes the cold predicate scan relatively more
  expensive. Neither is measured.
- **Embedding time and the dense index build.** No encoder runs against synthetic text.
- **The FTS5 index's response to real morphology at scale.** Stemming collapses real English
  in a way independently generated pseudo-words cannot imitate; the synthetic index is 27%
  larger than the real one at the anchor size and that gap is assumed to stay proportional.
- **The 50k–500k points on a quiet machine.** Six other agent processes were running
  throughout the sweep. Two sweeps at identical settings differed by 5–10%, and every p95 in
  the sweep tables is contended. Only the 13,145-row anchor was re-taken quietly, which is
  where a tight absolute number mattered; the larger points carry conclusions that are ratios
  and slopes, and a uniform few per cent moves neither.
