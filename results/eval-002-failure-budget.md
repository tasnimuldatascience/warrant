# Eval 002 — the failure budget, and what it changed

**Date:** 2026-08-30
**Reproduce:** `make build && make index && make eval && make autopsy`
**Supersedes:** [eval-001](eval-001-lexical-baseline.md), whose temporal number was computed
against a benchmark since found to be invalid — see *Corrections* below.

## Buckets

Four buckets, reported separately and never averaged. They measure different things and have
very different difficulty; a combined score would be a weighted average of incomparable
quantities with weights nobody chose.

| bucket | n | sufficiency | 95% CI | distractor rate | 95% CI |
|---|---:|---:|:---:|---:|:---:|
| temporal | 721 | 76.4% | 73.2–79.3 | **0.0%** | 0.0–0.0 |
| human | 42 | 81.0% | 69.0–92.9 | 0.0% | 0.0–0.0 |
| scope | 60 | 100.0% | 100.0–100.0 | 0.0% | 0.0–0.0 |
| scope-exclusion | 60 | n/a | | **0.0%** | 0.0–0.0 |
| generated | 130 | 100.0% | 100.0–100.0 | 0.0% | 0.0–0.0 |

### Ablations — the predicates, measured rather than asserted

| configuration | bucket | sufficiency | distractor rate |
|---|---|---:|---:|
| as-of predicate **on** | temporal | 76.4% | **0.0%** |
| as-of predicate **off** | temporal | 62.3% | **62.0%** |
| applicability **on** | scope-exclusion | n/a | **0.0%** |
| applicability **off** | scope-exclusion | n/a | **100.0%** |

Both predicates do exactly what they claim, and both ablations are the evidence. Without the
as-of predicate, 62% of answers cite a version of the rule that was not in force on the date
asked. Without the applicability predicate, **every** exclusion item retrieves a part that
does not govern the asker.

The as-of predicate also *raises* sufficiency (76.4% vs 62.3%). Superseded versions are
near-duplicates of the current text, so without the predicate they crowd the correct version
out of the candidate list. That is the concrete, measured cost of post-filtering rather than
pushing the predicate into the query.

## The failure budget

The artifact this repository exists to produce. For every failure, the first stage at which
no sufficient evidence set survives.

**Before** — `rerank_top_k: 30`, `final_k: 8`. 236 failures of 721 (67.3% satisfied):

| stage | failures | share |
|---|---:|---:|
| ingestion | 0 | — |
| applicability | 0 | — |
| temporal | 0 | — |
| retrieval | 25 | 10.6% |
| fusion | 87 | 36.9% |
| rerank | 46 | 19.5% |
| truncation | 78 | 33.1% |

Two thirds of all failures — fusion plus truncation, 165 of 236 — were evidence the system
had already found and then cut. The corpus and both predicates contributed nothing.

**The fix chosen from that table, not in advance:** widen the fused head and the final cut,
`rerank_top_k` 30 → 50 and `final_k` 8 → 16.

**After.** 170 failures of 721 (76.4% satisfied):

| stage | failures | share | change |
|---|---:|---:|---|
| retrieval | 25 | 14.7% | unchanged |
| fusion | 40 | 23.5% | **−47** |
| rerank | 64 | 37.6% | +18 |
| truncation | 41 | 24.1% | **−37** |

66 failures eliminated, +9.1 points. The budget moved where it predicted: fusion and
truncation fell by 84 between them, and `retrieval` did not move at all, which is correct —
widening a downstream window cannot change what was retrieved.

`rerank` rising from 46 to 64 is the instrument working, not a regression. More evidence now
survives long enough to reach the reranker, so more of it can be demoted there. The
bottleneck moved downstream, and the next intervention is named by the new table rather than
by taste.

**Cost.** `final_k: 16` doubles the context budget a generator will eventually be handed.
That trade is not free and is recorded here so it can be revisited when P1 has a token budget
to weigh it against.

## Two bugs the budget found in itself

**The reranker was being blamed for truncation.** The first version of the ladder attributed
a loss to `rerank` whenever a reranker had run, which put 124 failures on the cross-encoder.
Removing the cross-encoder entirely moved the bucket by 0.1 points (67.4% → 67.3%), which
cannot be true of a stage responsible for half the failures. The ladder now blames `rerank`
only when the evidence was inside the fused top-`final_k` and the reranker moved it out;
otherwise plain truncation would have lost it too. That reattributed 78 of the 124.

This is the exact bias the design anticipated — first-loss attribution blaming whatever ran
last — caught by disagreement between the budget and a direct ablation. It is the argument
for running both.

**The interventional label was misleading.** `depth` implied that raising the candidate
budget would fix the item. It does not: raising candidates from 100 to 1000 moves the
temporal bucket by well under a point. What the intervention actually shows is that the
evidence is *reachable* and therefore ranked too low. Renamed to `ranking` / `unreachable`.
Of 60 sampled failures, 60 are `ranking` and 0 `unreachable`: nothing in this bucket fails
because the query and the text do not meet.

## Corrections to eval-001

eval-001 reported temporal sufficiency of 35.1% at k=8. That number is withdrawn. The
benchmark it scored made a whole section's changed paragraphs the evidence for one item, so
**41 of 252 items (16%) required more paragraphs than the pipeline returns** — one needed 56
in a list of 8. They were unsatisfiable by construction and were being counted as retrieval
failures, putting a floor under the bucket that no configuration could beat.

Evidence sets must be minimal: what is needed to answer, not everything that changed on the
same day. The miner now emits one item per amended paragraph, giving 721 items with a
singleton evidence set each. eval-001's ablation conclusion — that the as-of predicate
eliminates superseded citations — was unaffected and is reproduced above on the corrected
benchmark.

## Caveats

- **`generated` at 100% is not an achievement.** Its queries are built from the paragraph
  they retrieve. It measures whether the corpus is reachable at all and should never be read
  as answer quality.
- **`human` at 42 items cannot rank configurations.** Its interval spans 24 points. It
  characterises what a realistic query looks like; it is author-written, not collected from
  users.
- **`scope` at 100% reflects a coarse model.** Applicability is enforced at part level from
  eight parts whose own titles restrict who they govern. Real applicability is often stated
  inside a section's text, and none of that is modelled.
- **The temporal bucket over-represents sections that change often.** Parts 890 and 315
  dominate, and it is retrieval-only: no answer is generated or verified yet.
- 721 items means differences under about three points are inside the noise.
