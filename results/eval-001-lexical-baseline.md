# Eval 001 — lexical baseline on the temporal bucket

**Date:** 2026-08-30
**Config hash:** see `configs/default.yaml`; lexical only, no dense retrieval, no reranking.
**Reproduce:** `make build && make eval`

251 temporal items over 103 sections, mined from real amendments in 26 parts of 5 CFR
chapter I. Each amendment contributes two items — one dated inside the old version interval,
one inside the new — that share a query and differ only in `as_of`. A system that ignores the
date must get one of every pair wrong.

## Result

| config | k | n | sufficiency | 95% CI | distractor rate | 95% CI |
|---|---:|---:|---:|:---:|---:|:---:|
| as-of ON | 8 | 251 | 35.1% | 29.1–41.0 | **0.0%** | 0.0–0.0 |
| as-of ON | 20 | 251 | 48.6% | 42.6–54.6 | **0.0%** | 0.0–0.0 |
| as-of ON | 50 | 251 | 62.9% | 57.0–68.9 | **0.0%** | 0.0–0.0 |
| as-of ON | 100 | 251 | 75.7% | 70.5–80.9 | **0.0%** | 0.0–0.0 |
| as-of ON | 300 | 251 | 86.5% | 81.7–90.8 | **0.0%** | 0.0–0.0 |
| as-of ON | 1000 | 251 | 90.8% | 86.9–94.4 | **0.0%** | 0.0–0.0 |
| as-of OFF | 8 | 251 | 27.1% | 21.5–32.7 | 59.4% | 53.4–65.7 |
| as-of OFF | 20 | 251 | 36.3% | 30.3–42.2 | 70.5% | 64.5–76.1 |
| as-of OFF | 50 | 251 | 49.8% | 43.4–56.2 | 82.1% | 76.9–86.5 |
| as-of OFF | 100 | 251 | 63.7% | 57.0–69.7 | 86.9% | 82.5–90.4 |
| as-of OFF | 300 | 251 | 82.1% | 77.3–86.9 | 94.0% | 90.8–96.8 |
| as-of OFF | 1000 | 251 | 89.2% | 85.3–93.2 | 94.8% | 92.0–97.6 |

*Sufficiency* — the retrieved set contains a complete minimal sufficient evidence set.
*Distractor rate* — the retrieved set contains the superseded or not-yet-in-force version of
the same paragraph. Intervals are a seeded percentile bootstrap over items.

## What it says

**The as-of predicate eliminates superseded-law citations, and the ablation is what proves
it.** Zero distractors at every depth with the filter on; 59.4% at k=8 without it, rising to
94.8% at k=1000 because deeper candidate lists simply collect more versions of the same
section. Stating that a temporal filter works is an assertion. This is the measurement.

Note the second-order effect: the filter also *raises* sufficiency at every depth (35.1% vs
27.1% at k=8). Superseded versions are near-duplicates of the current text, so without the
predicate they crowd the correct version out of the candidate list. This is the concrete form
of the argument for pushing the predicate into the query rather than post-filtering — the
wasted slots are measurable, not theoretical.

## Where the remaining failure is, and where it is not

The autopsy question, asked of the 64.9% of items that fail at k=8:

| Check | Result |
|---|---|
| Evidence absent from the corpus | **0 of 251** |
| Evidence present but not in force on the item's `as_of` | **0 of 251** |
| Evidence retrievable at all (k=1000) | 90.8% |
| Evidence in the top 8 | 35.1% |

So the failure is **ranking, not ingestion and not recall**. The right paragraph is in the
corpus, correctly dated, and findable — it is just buried. Sufficiency climbing 35.1% → 90.8%
between k=8 and k=1000 is the signature of a weak scorer, not a missing document.

That is a directly actionable diagnosis, and it is what P1 should attack: dense retrieval,
reciprocal rank fusion, and a cross-encoder reranker, in that order, measured on this same
bucket. The failure budget does not yet have a `chunking` or `generation` row because
neither stage exists; it will.

## Caveats

- Lexical retrieval here is a bag-of-words `OR` query over FTS5 with Porter stemming. It is
  a floor, deliberately — a baseline that is already good hides the effect of what comes next.
- The residual 9.2% never retrieved at k=1000 has not been diagnosed. It is the next thing to
  look at, because a hard ceiling there would cap every configuration that follows.
- This bucket measures temporal discrimination and nothing else. Queries are assembled from
  regulatory wording shared by both versions, so they are more literal than real questions.
  The generated and human buckets do not exist yet, and no number here should be read as
  end-to-end answer quality.
- 251 items means differences under roughly six points are inside the noise. Two overlapping
  intervals in this table are not evidence of a difference.
