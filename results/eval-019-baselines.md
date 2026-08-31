# Eval 019 — External baselines: is this better than an afternoon of vector DB?

**Date:** 2026-08-30
**Config:** `configs/default.yaml`, hash `dbac692062d7` (final_k 16, candidates_lexical/dense
100, rerank_top_k 50, dense `BAAI/bge-small-en-v1.5`, rerank
`cross-encoder/ms-marco-MiniLM-L-6-v2`). Scored against a read-only copy of
`data/warrant.sqlite3`, taken while other work was in flight against the live file — same
content, zero risk of racing a concurrent write.
**Split:** test only, horizon 2026-08-26. `temporal` 233 items / 47 sections, `scope` 42 / 42,
`scope-exclusion` 42 / 42, `human` **51** / 37 (up from the 29 in
[eval-004](eval-004-held-out.md) — `benchmarks/human.yaml` is being actively grown elsewhere;
read once here, the count above is what was actually scored).
**Reproduce:** no CLI command exists yet for this comparison — see *CLI wiring* at the end.
**Owns:** `src/warrant/eval/baseline.py`, `tests/test_baseline.py`, this file.

Every comparison in this repository up to now has been an **internal ablation**: this
pipeline, with one stage switched off. That answers "does the as-of predicate help *this
system*," not the question a skeptic actually asks first — is any of this better than a vector
DB and an embedding model, built in an afternoon? This report builds that competitor and three
of its closer cousins, and scores all four with the exact same `warrant.eval.run.score` on the
exact same held-out items.

## The four

1. **naive dense** — cosine top-*k* over the whole dense index. No as-of predicate, no scope
   predicate, no fusion, no reranking. Whatever most "RAG in an afternoon" projects ship.
2. **bm25 only** — FTS5 top-*k* over everything believed, both predicates off. The "just use
   search" baseline. Built from the real `Retriever` with `temporal=False` and an empty
   `parts_universe` — the same two flags `cli._paired` already uses to turn a predicate off —
   not a second lexical code path.
3. **dense + post-filter** — cosine top-100, *then* discard anything not in force at `as_of`,
   then cut to final_k. The obvious way to bolt temporality onto a vector baseline, and the
   one this architecture argues against: pushing the predicate into the query means the
   candidate slots a post-filter would reclaim are never spent on dead law to begin with. This
   configuration spends them and then throws the receipt away.
4. **full warrant** — `_retriever(cfg, store)`, unmodified. The reference column.

All four read the same store and the same dense index the real pipeline reads — vectors for
13,212 believed chunk versions across 26 parts, including every superseded version, built once
by `retrieve.dense.build`. No baseline gets its own corpus or its own encoder; the only
variable is which stages run.

## The four-way table

Section-clustered 95% bootstrap intervals throughout — the same instrument
[eval-004](eval-004-held-out.md) and the README use, because an i.i.d. item bootstrap on this
benchmark reports an interval roughly 3.5x too narrow (`eval/stats.py`). `*` marks a wrong-
version rate that is 0% because the distractor was never a candidate for that configuration
(enforced by construction), not because anything was ranked away from it.

### temporal — 233 items, 47 sections

| configuration | sufficiency | 95% CI | wrong version | 95% CI |
|---|---:|:---:|---:|:---:|
| naive dense | 87.1% | 82.4–92.5 | 89.7% | 85.1–93.0 |
| bm25 only | 95.3% | 92.7–98.8 | 97.0% | 93.9–98.5 |
| dense + post-filter | 92.7% | 88.4–97.1 | **0.0%** | 0.0–0.0 |
| full warrant | **96.1%** | 93.4–99.4 | **0.0%*** | 0.0–0.0 |

### human — 51 items, 37 sections

| configuration | sufficiency | 95% CI | wrong version | 95% CI |
|---|---:|:---:|---:|:---:|
| naive dense | 74.5% | 60.9–87.9 | 0.0%* | 0.0–0.0 |
| bm25 only | 56.9% | 41.8–72.5 | 0.0%* | 0.0–0.0 |
| dense + post-filter | 78.4% | 65.2–91.3 | 0.0%* | 0.0–0.0 |
| full warrant | **82.4%** | 71.4–91.3 | 0.0%* | 0.0–0.0 |

### scope — 42 items, 42 sections (governed profile; the section must be retrieved)

| configuration | sufficiency | 95% CI | wrong version | 95% CI |
|---|---:|:---:|---:|:---:|
| naive dense | 100.0% | 100.0–100.0 | 0.0%* | 0.0–0.0 |
| bm25 only | 100.0% | 100.0–100.0 | 0.0%* | 0.0–0.0 |
| dense + post-filter | 100.0% | 100.0–100.0 | 0.0%* | 0.0–0.0 |
| full warrant | 100.0% | 100.0–100.0 | 0.0%* | 0.0–0.0 |

### scope-exclusion — 42 items, 42 sections (non-governed profile; the section must be absent)

n/a in the sufficiency column throughout: there is nothing to retrieve, only an out-of-scope
citation to avoid. See `BucketResult.measures_absence`.

| configuration | wrong version (leaked the out-of-scope citation) | 95% CI |
|---|---:|:---:|
| naive dense | **100.0%** | 91.6–100.0 |
| bm25 only | **100.0%** | 91.6–100.0 |
| dense + post-filter | **100.0%** | 91.6–100.0 |
| full warrant | **0.0%*** | 0.0–0.0 |

**Every baseline cites a part that does not govern the asker, on all 42 of 42 items, every
time.** None of the three implements an applicability predicate, so none of them ever excludes
the wrong part from candidacy — this is not a near-miss, it is total. Full Warrant's
applicability predicate is the only thing standing between a government-wide answer and a
citation to a part-specific rule that does not apply, and this bucket makes it visible in a
way the `scope` bucket (saturated at 100% for all four) cannot.

## Paired deltas against naive dense

`wins` — items only the named configuration got right. `losses` — items only naive dense got
right. Both a nonzero CI and p < 0.05 for `significant`, matching `PairedDelta.significant`.
`scope` is a single row here: all three deltas are exact zero on both measures (42/42 tie),
because the bucket is saturated for every configuration and pairing has nothing to resolve.

| bucket | configuration | measure | delta | 95% CI | won | lost | p | significant |
|---|---|---|---:|:---:|---:|---:|---:|:---|
| temporal | bm25 only | sufficiency | +8.2 | 4.7–12.4 | 19 | 0 | 3.8e-06 | yes |
| temporal | bm25 only | no wrong version | **−7.3** | −12.0–−3.5 | 0 | 17 | 1.5e-05 | yes, and it's a **loss** |
| temporal | dense + post-filter | sufficiency | +5.6 | 2.5–9.7 | 13 | 0 | 2.4e-04 | yes |
| temporal | dense + post-filter | no wrong version | **+89.7** | 84.1–95.1 | 209 | 0 | 2.4e-63 | yes |
| temporal | full warrant | sufficiency | +9.0 | 5.0–12.7 | 23 | 2 | 1.9e-05 | yes |
| temporal | full warrant | no wrong version | **+89.7** | 84.1–95.1 | 209 | 0 | 2.4e-63 | yes |
| human | bm25 only | sufficiency | **−17.6** | −36.8–−1.9 | 2 | 11 | 0.022 | yes, and it's a **loss** |
| human | bm25 only | no wrong version | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | n/a — no distractors in this bucket |
| human | dense + post-filter | sufficiency | +3.9 | 0.0–11.4 | 2 | 0 | 0.5 | not measurable |
| human | dense + post-filter | no wrong version | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | n/a |
| human | full warrant | sufficiency | +7.8 | −4.7–20.4 | 7 | 3 | 0.34 | not measurable |
| human | full warrant | no wrong version | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | n/a |
| scope | any | both measures | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | tied at 100%/0% |
| scope-exclusion | bm25 only | no wrong version | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | tied at 0% (both leak 100%) |
| scope-exclusion | dense + post-filter | no wrong version | +0.0 | 0.0–0.0 | 0 | 0 | 1.0 | tied at 0% (both leak 100%) |
| scope-exclusion | full warrant | no wrong version | **+100.0** | 100.0–100.0 | 42 | 0 | 4.5e-13 | yes |

## The honest headline

**On sufficiency, the whole apparatus buys +9.0 points over naive dense on the bucket built to
show it off (temporal: 87.1% → 96.1%, CI 5.0–12.7, 23 won / 2 lost). On not citing dead law,
it buys +89.7 points (89.7% wrong → 0.0% wrong, CI 84.1–95.1, 209 won / 0 lost).** That is the
same shape the internal ablation in the README reports for the as-of predicate alone (+2.2 /
+96.1) — moderate on sufficiency, overwhelming on correctness — and this is not the same
measurement repeated: it is a different retriever, built from scratch against the same store,
arriving at the same qualitative conclusion. That agreement is worth more than either number
alone, because an internal ablation can only ever prove a flag matters to *this* pipeline; an
external baseline reaching the same shape says the finding is about the *problem*, not about
one codebase's wiring.

**Sufficiency alone would again have been misleading, and worse than in the README's own
ablation.** bm25-only's sufficiency (95.3%) is within 0.8 points of full warrant's (96.1%) —
a reader who stopped at sufficiency would call BM25 nearly as good as the full pipeline. Its
wrong-version rate is 97.0%, the single worst number in this entire report, 7.3 points *worse*
than naive dense's own 89.7% (delta −7.3, CI −12.0 to −3.5, p = 1.5e-05 — bm25-only reliably
loses to naive dense at avoiding superseded text, not just to full warrant). BM25's length
normalisation rewards the shorter pre-amendment paragraph over its longer successor
([eval-014](eval-014-query-terms.md) names this bias directly), and with no as-of predicate to
narrow the candidate set, that bias has the whole corpus to work with instead of one section's
two versions.

## Where a baseline wins

**Naive dense beats bm25-only on the human bucket, and it is not close.** 74.5% vs 56.9%
sufficiency, delta −17.6 points *against* bm25-only (CI −36.8 to −1.9, 2 won / 11 lost,
p = 0.022) — the only baseline-vs-baseline comparison in this report that clears
significance. Human queries are paraphrased, not copied from the regulation; a bag-of-words
match has nothing to grab onto that an embedding does. The "just use search" baseline is worse
than the naive vector-DB baseline on exactly the query style a real user would type.

**Naive dense is also the cost story.** See below — it is ~54x faster than full warrant
(1.5 ms vs 83.4 ms p50) for 90.6% of its sufficiency on temporal (87.1% vs 96.1%). Nobody
should read that as an argument to ship naive dense; a 89.7% wrong-version rate is the reason
not to, and it is a real number that belongs here regardless.

**Dense + post-filter never beats full warrant on any measure in this table.** It is worth
building anyway, because it is the shape a reasonable engineer's first fix would take, and the
report needs its actual number rather than an assumption.

## Where the "obvious fix" breaks

Post-filtering ran dry — fewer than `final_k` (16) survivors after discarding what was not in
force — on **16 of 233 temporal items (6.9%)**, 0 of the other three buckets. Zero on `human`,
`scope`, `scope-exclusion` because those buckets ask about one point in time against a corpus
with little same-section version churn; `temporal` is built from real amendments, so the top
of an unrestricted cosine ranking is disproportionately *both* versions of the same paragraph,
and discarding the wrong one after the fact can leave a candidate pool thinner than what was
asked for. A predicate pushed into the query cannot do this — it can only narrow what the
ranker draws from, never shrink an already-ranked list below the request. This is the concrete
form of the argument [eval-001](eval-001-lexical-baseline.md) made about wasted candidate
slots, now demonstrated on a configuration built specifically to spend them and then discard
the receipt.

Post-filtering also does not implement the scope predicate at all (by design — the task this
baseline models is "add temporality to a vector DB," not "add applicability too"), so its
scope-exclusion leak rate is identical to naive dense's: 100%.

## Cost

Same 60-item sample of temporal test queries, one warm-up query discarded per configuration,
wall clock from `trace.timings["total"]` (all four configurations record it).

| configuration | p50 | p95 | needs a GPU | model(s) loaded |
|---|---:|---:|:---:|---|
| naive dense | **1.5 ms** | 2.1 ms | no | bge-small-en-v1.5 (query only) |
| dense + post-filter | 2.1 ms | 2.5 ms | no | bge-small-en-v1.5 (query only) |
| bm25 only | 32.8 ms | 51.6 ms | no | none — no torch import at all |
| full warrant | 83.4 ms | 102.8 ms | no | bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2 |

None of the four *requires* a GPU — the README documents the same encoder and reranker running
CPU-only. A CUDA GPU was present in this environment and sentence-transformers used it by
default, so the three dense-touching rows understate their CPU-only cost; bm25-only's number
is hardware-independent. The gap between "+ dense" and "+ cross-encoder" in
[eval-004](eval-004-held-out.md)'s own latency table (41.0 ms → 71.0 ms) is the same
reranking cost showing up here as most of full warrant's 83.4 ms.

**On sufficiency alone, naive dense is 90.6% as good as full warrant (87.1%/96.1%) for ~1.9%
of the latency** (1.5 ms vs 83.4 ms, full warrant ~54x slower) — the "90% as good for 10% of
the cost" framing this report
exists to check for, and on that one measure it very nearly holds. It does not hold on
wrong-version rate, which is the measure that matters most here: naive dense's clean rate is
10.3% (100% − 89.7% wrong) against full warrant's 100%. A system that cites superseded law on
9 of 10 dateable questions is not a cheaper version of this one, it is a different, worse one
that happens to share a sufficiency number with the real thing.

BM25-only is the slowest of the three non-reference configurations despite doing the least
work, because it runs the FTS5 `OR` query completely unrestricted — no as-of clause, no
`exclude_parts` clause — against a corpus where that query pattern is already documented to
match 88–90% of all rows at every scale (`retrieve/hybrid.py`). The predicate that full
warrant pushes into the same query is not just a correctness fix; on this baseline's own
lexical stage, running unrestricted is not obviously the cheap option either.

## CLI wiring

Not implemented — `baseline.py`, `test_baseline.py`, and this file are the only files this
task owns. The command this report implies:

```
warrant eval baselines -c CONFIG --split test
```

wired in `cli.py` by extending `_buckets`/`_retriever` the way `_paired` already does:

```python
@eval_app.command("baselines")
def eval_baselines(config: ConfigOpt = None,
                   split: Annotated[str, typer.Option(help="test, dev or all")] = "test") -> None:
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, _ = _buckets(cfg, store)
        buckets = {k: [i for i in v if split == "all" or i.split == split] for k, v in buckets.items()}
        full = _retriever(cfg, store)
        configs = {
            "naive dense": baseline.NaiveDense(store=store, dense_index=full.dense_index,
                                               final_k=cfg.retrieve.final_k),
            "bm25 only": baseline.bm25_only(store, final_k=cfg.retrieve.final_k,
                                            candidates_lexical=cfg.retrieve.candidates_lexical),
            "dense + post-filter": baseline.DensePostFilter(
                store=store, dense_index=full.dense_index,
                candidates_dense=cfg.retrieve.candidates_dense, final_k=cfg.retrieve.final_k),
            "full warrant": full,
        }
        # score() each config x bucket, paired_delta() each against "naive dense", print
        # the tables above, and shortfall_stats() on the post-filter config per bucket.
```

`full.dense_index` is loaded once and shared by three of the four configurations, so the
command pays for the encoder a single time. `naive dense` and `dense + post-filter` never load
the reranker at all; only `full warrant` does.

## Caveats

- This run used a **read-only copy** of `data/warrant.sqlite3`, taken while other agents were
  building against the live file per the session's working agreement — content-identical at
  copy time, not re-synced afterward. `store.read_only()` was set on the connection as a
  second guard.
- `benchmarks/human.yaml` was read exactly once (via a single `_buckets` call); the 51-item
  count above is what was actually scored, not a re-read mid-run. It is nearly double
  eval-004's 29, so the human-bucket numbers here are not directly comparable to that report's
  — a different, larger item set produces a different sufficiency even for an unchanged
  retriever.
- `dense + post-filter` and `full warrant` were not directly paired against each other, only
  each against naive dense — the temporal-bucket gap between them (92.7% vs 96.1% sufficiency,
  0.0% vs 0.0%* wrong version) is a marginal comparison, not a tested one, and is reported as
  such.
- Bootstrap and sign-test intervals are seeded (`samples=1000`, the repo default) and
  deterministic; rerunning this exact config against an unchanged store reproduces every
  number in this report bit-for-bit.
- Latency figures are single-machine, GPU-present, unbatched, and share the exact sampling
  method `cli.eval_latency` already uses (stride to 60 temporal-test items, first query
  discarded as warm-up) — comparable to [eval-004](eval-004-held-out.md)'s latency table by
  construction, not by coincidence.
