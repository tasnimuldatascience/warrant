# Eval 005 — abstention: does the system know when it does not know?

> **Superseded serving figures.** This report quotes 21.3 tok/s and a 3-per-minute
> ceiling. Both are wrong: an isolated re-derivation measured 29.2–29.9 tok/s over
> ~205 output tokens, so an answer is 6.6 s and the ceiling is 7.7 req/min. The text
> below is left as it was written, because a results doc is a record of what was
> measured on a day, and editing it to agree with a later number falsifies that
> record. See [eval-010](eval-010-capacity.md).

**Date:** 2026-08-30
**Motivated by:** [eval-004](eval-004-held-out.md), which measured a 1.5%
hallucination rate, 98.5% citation precision, and **6 of 29 held-out human questions answered
from a context that contained no sufficient evidence, with 0 abstentions.**
**Reproduce:** `make build && make index`, then the snippet under
[Reproducing](#reproducing).
**Headline:** an abstention policy is worth shipping. **The learned combiner is not.** It does
not beat a threshold on the raw top-1 fusion score, and on the one comparison where it looks
like a landslide it is measuring threshold transfer rather than discrimination.

---

## What is being predicted, and what is not

The label is **sufficiency of the retrieved set**: did the final 16 chunks contain a complete
minimal sufficient evidence set for the item? It needs no human and no generator — the
benchmark already carries a disjunction of such sets — which is what makes this study
reproducible from a clone with no API key and no model call beyond the ones retrieval already
makes.

It is a proxy, and the gap matters. eval-004's failure was the *generator* writing an answer
from insufficient context. This study predicts whether the context was insufficient. A policy
that abstains perfectly on this label still says nothing about whether the generator would
have hallucinated on the cases it lets through, and nothing here should be read as a
hallucination-rate claim.

Eight features, all computed from the `Trace` and the rows already in hand. No second forward
pass: generation runs at 21.3 tok/s on this machine, so a verifier costing another pass would
double the latency of the stage it guards and be the first thing shed under load — exactly
when the guard is most needed.

## Population

`final_k: 16`, dense and cross-encoder on, horizon 2026-08-26.

| | items | sections | insufficient | rate |
|---|---:|---:|---:|---:|
| dev (fit) | 199 | 104 | 10 | 5.03% |
| **test (report)** | **300** | **98** | **13** | **4.33%** (0.7–8.6) |

Test composition: temporal 229 (6 insufficient), scope 42 (0), human 29 (7).

**`scope-exclusion` is excluded, and that is a decision worth defending.** Those 95 items are
written with an empty acceptable-evidence set — the whole question is whether something is
*absent* — so `is_satisfied_by` returns True for every possible ranked list. Ninety-five items
that cannot be got wrong would have raised coverage and lowered base risk without one of them
being a measurement. `scope` is kept despite being 95/95 sufficient: it is a real measurement
that happened to come out clean, and dropping buckets for being easy is the one-sided
accounting this repo already withdrew a claim over.

Everything below rests on **13 errors**. Every interval is wide for that reason and none of
this would survive a reader who wanted it to.

## The features, measured

Single-feature AUC against sufficiency over all 499 items, and AURC on test (lower is better;
`always answer` is 0.0433, so anything at or above that discriminates nothing):

| feature | AUC | test AURC | range on this corpus |
|---|---:|---:|---|
| `top_score` | **0.828** | **0.0086** | 0.0248–0.0328 |
| `term_coverage` | 0.703 | 0.0260 | 0.40–1.00 |
| `rank_agreement` | 0.563 | 0.0302 | 0.00–0.94 |
| `margin_1_2` | 0.554 | 0.0268 | 0.0000–0.0167 |
| `margin_1_5` | 0.526 | 0.0256 | 0.0004–0.0169 |
| `guidance_top` | 0.500 | 0.0433 | constant 0 |
| `entropy` | 0.360 | 0.0233 *inverted* | 0.9804–0.9999 |
| `log_admitted` | 0.359 | 0.0258 *inverted* | 9.064–9.210 |

Four things in that table are worth saying out loud.

**The prediction about `rank_agreement` was wrong.** It was expected to be the strongest free
signal, on the reasoning that two rankers built on unrelated evidence — term statistics and
embedding geometry — converging is hard to arrange by accident. It is fifth of eight. Both
rankers read the same admitted set through the same `retrieval_text`, and on regulatory prose
they agree about half the time whether or not the answer is in there: mean 0.522 overall
against 0.484 on the items that failed, on a feature ranging 0 to 0.94. The docstring in
`abstain.py` now carries the measured number instead of the reasoning.

**`top_score` wins despite being nearly degenerate.** RRF caps it at `2/(k+1) = 0.0328`, it
takes seventeen distinct values across 300 test items, and it spans a range of 0.008. Being at
the ceiling means both rankers put the same chunk first, which turns out to be most of what
there is to know.

**Three features are constants on this corpus.** `guidance_top` is exactly zero — every row is
eCFR regulation, single-source. `log_admitted` spans 9.06–9.21 because the predicates admit
8,637–9,997 rows on every single item: the "near-empty admitted set means out of corpus" case
this feature exists for **never occurs here**, and it earns its place only once a query can
fall outside the corpus. `entropy` sits at 0.9978 ± 0.0025 because the head it summarises is a
list of RRF weights `1/(60+r)`, which is close to uniform whatever the query — "flat is bad"
cannot discriminate when everything is flat.

**Two features are informative with the sign reversed.** `entropy` and `log_admitted` both
have AUC ≈ 0.36, i.e. ≈ 0.64 inverted. Fitted on ten negatives, that is as likely to be a
quirk of this split as a mechanism, and it is why `entropy` carries the largest coefficient in
the table below while contributing nothing a reader should trust.

Fitted coefficients, standard-deviation units, dev split:

```
entropy         -0.981     term_coverage   +0.190
rank_agreement  +0.497     margin_1_5      -0.178
log_admitted    -0.368     margin_1_2      -0.110
top_score       +0.367     guidance_top    +0.000     intercept +3.461
```

## Calibration

Isotonic (pool-adjacent-violators) fitted on dev, applied to test. Ten equal-width bins,
Wilson intervals on every populated one:

| ECE | Brier |
|---|---|
| raw **0.0329** (0.0154–0.0675) → calibrated **0.0203** (0.0066–0.0615) | 0.0426 → 0.0406 |

**Raw combiner, test:**

| bin | n | confidence | empirical | gap | 95% CI |
|---|---:|---:|---:|---:|:---:|
| 0.0–0.1 | 2 | 0.059 | 1.000 | +0.941 | 34.2–100.0 |
| 0.6–0.7 | 4 | 0.666 | 1.000 | +0.334 | 51.0–100.0 |
| **0.7–0.8** | 7 | 0.765 | **0.429** | **−0.337** | 15.8–75.0 |
| 0.8–0.9 | 12 | 0.857 | 0.750 | −0.107 | 46.8–91.1 |
| 0.9–1.0 | 275 | 0.967 | 0.978 | +0.011 | 95.3–99.0 |

**After isotonic, test:**

| bin | n | confidence | empirical | gap | 95% CI |
|---|---:|---:|---:|---:|:---:|
| 0.4–0.5 | 13 | 0.444 | 0.692 | +0.248 | 42.4–87.3 |
| 0.6–0.7 | 1 | 0.667 | 1.000 | +0.333 | 20.7–100.0 |
| 0.9–1.0 | 286 | 0.977 | 0.969 | −0.009 | 94.1–98.3 |

The 0.7–0.8 row is the failure this measure exists to catch: seven items told they were 77%
safe, of which three were. Isotonic pulls those down into a 0.4–0.5 block and the aggregate
ECE improves by a third. It does not make the model good — the calibrated table now
*under*-confident by 0.248 in its one low block, on thirteen items — but it removes the
region where a reader would have been told to trust a wrong answer.

The recalibration is not free. It pools items into blocks, and pooling destroys the ordering
inside a block: **AURC gets worse, 0.0094 → 0.0105.** Calibration and discrimination are
traded here, not jointly improved, and which one an operator wants depends on whether the
confidence number is shown to anyone.

> A bug found by writing the test: the PAVA fit kept the *lower* bound when it merged two
> blocks, so any test score landing inside a merged block was read off the next block up — the
> one PAVA had just proved it did not belong to. It produced plausible probabilities and
> silently degraded calibration in exactly the region isotonic exists to repair.
> `tests/test_abstain.py::test_isotonic_keeps_the_merged_blocks_upper_bound` pins it.

## Risk against coverage

Test split. AURC integrated over coverage on [0, 1]; intervals are a section-clustered
bootstrap of the whole functional, because two paragraphs of one section are not independent
samples. Operating points use the threshold each policy could have chosen **on dev**.

| policy | AURC | 95% CI | coverage at the shipped threshold | selective risk |
|---|---:|:---:|---:|---:|
| always answer (today) | 0.0433 | 0.0074–0.0865 | 100.0% (98.7–100.0) | 4.33% (0.7–8.6) |
| top-1 fusion score | **0.0086** | 0.0004–0.0211 | **0.0%** (0.0–1.3) | — |
| learned, uncalibrated | 0.0094 | 0.0002–0.0223 | 75.3% | 1.33% |
| **learned + isotonic** | 0.0105 | 0.0002–0.0239 | **74.0%** (68.8–78.6) | **1.35%** (0.0–3.0) |

**The operating point at a ≤2% selective-risk budget: 74.0% coverage (68.8–78.6) at 1.35%
selective risk (0.0–3.0), 3 errors in 222 answers.** Against always-answer that is 13 errors
traded for 3, at the cost of declining 78 of 300 questions.

The learned curve, in full:

| threshold | coverage | selective risk | errors/answered |
|---:|---:|---:|---:|
| 0.9975 | 18.3% | 0.00% | 0/55 |
| 0.9857 | 60.3% | 1.10% | 2/181 |
| **0.9655** | **74.0%** | **1.35%** | **3/222** |
| 0.9512 | 95.3% | 3.15% | 9/286 |
| 0.4444 | 100.0% | 4.33% | 13/300 |

Abstention quality at that point: it catches **10 of 13** insufficient items (76.9%,
49.7–91.8) and needlessly refuses **68 of 287** sufficient ones (23.7%, 19.1–28.9). Roughly
one in four good answers is thrown away to remove three bad ones.

## The null result

**The learned combiner does not beat the single-feature baseline.**

```
AURC(learned) − AURC(top-1 fusion) = +0.0019   95% CI −0.0017 … +0.0070
                                               P(learned better) = 0.26
```

The point estimate favours the baseline, the interval contains zero comfortably, and a paired
section-clustered bootstrap over the same 300 items puts the probability that the combiner has
the better curve at about one in four. Eight features, a ridge fit and an isotonic map bought
nothing over thresholding a number RRF already computed.

**Sweeping the ridge does not rescue it.** `l2` from 0.01 to 10 gives test AURC 0.0121, 0.0122,
0.0105, 0.0101, 0.0094 — monotone toward the baseline and never past it. (`l2 = 0` raises
`LinAlgError`: the constant `guidance_top` column makes the Hessian singular, which is why the
penalty is not optional.) `DEFAULT_L2` stays at 1.0 because no value wins, not because 1.0 was
chosen.

### The one comparison that looks like a win, and why it is not

At the thresholds each policy could actually have shipped, the paired per-item decision delta
is **+72.00 pp (64.8–79.4), 219 wins to 3 losses, p < 1e-4** in favour of the learned
combiner. That number is real and it is not evidence of a better signal. It is this:

> On dev, the best the top-1 fusion score can do is **73.4% coverage at 2.05% selective risk —
> 3 errors in 146.** The budget is 2%. `operating_point` returns None rather than the closest
> miss, `fit_policy` fails closed at threshold 1.0, and on test the baseline therefore answers
> **nothing**. The +72 pp is 74% coverage against 0% coverage.

One dev item is the whole margin: 2 errors in 146 would have been 1.37% and the baseline would
have shipped. A comparison that hinges on a single item in the fitting split is not a finding
about the two signals.

Give the baseline an oracle threshold on test and it is **better** than the learned combiner:

| at its best test threshold | coverage | selective risk |
|---|---:|---:|
| top-1 fusion | 86.3% | 1.54% (4/259) |
| learned + isotonic | 74.0% | 1.35% (3/222) |

Paired decision delta at those points: **−11.67 pp (−19.8 … −4.4), 11 wins to 46 losses,
p < 1e-4 — against the learned combiner.** Relaxing the budget from 2% to 2.5% makes the
baseline operable from dev, and it then lands on test at **73.0% coverage and 0.46%
selective risk (1 error in 219)** — more coverage at a third of the risk.

`Study.beat_baseline` returns **False**, and it requires both the point estimate and the
paired test to agree before it would return True.

## Per bucket, at the shipped threshold

| bucket | n | insufficient | coverage | 95% CI | selective risk | 95% CI |
|---|---:|---:|---:|:---:|---:|:---:|
| temporal | 229 | 6 | 77.7% | 71.9–82.6 | 1.12% | 0.0–2.6 |
| scope | 42 | 0 | 92.9% | 81.0–97.5 | 0.00% | 0.0–9.0 |
| **human** | **29** | **7** | **17.2%** | 7.6–34.5 | **20.00%** | 0.0–60.0 |

**On the bucket that motivated the whole exercise, the policy is close to useless.** It
answers 5 of 29 real questions, and one of those five is still written from insufficient
evidence. It does abstain correctly on 6 of the 7 bad items — the thing eval-004 said never
happened now happens — but at a coverage no operator would accept. The aggregate 74% coverage
is carried by the temporal and scope buckets, which are 91% of the test split and are far
easier: their queries are assembled from wording that is in the corpus by construction.

A single threshold across buckets is the wrong shape for this problem, and this table is the
evidence. Whether a per-bucket or query-type-conditioned threshold helps is not measured here
and should not be assumed.

## Where this could still be fooling you

- **Thirteen errors.** Every risk figure is a ratio with a single-digit numerator. The
  section-clustered intervals are honest about it and they are wide.
- **The threshold is chosen on dev, and dev has ten errors.** The 1.35% test risk is inside
  the 2% budget, but a budget met by a threshold fitted on ten negatives is not a guarantee;
  it is one draw.
- **The label is retrieval sufficiency, not answer correctness.** See the first section.
- **`scope` contributes 42 easy positives and zero negatives.** Keeping it is the conservative
  choice for the honesty of the bucket accounting and the optimistic one for the headline
  coverage. Both readings are in the per-bucket table.
- **The reranker tracks `main`.** `configs/default.yaml` pins no revision, and this run scores
  the human bucket at 22/29 sufficient where eval-004 reported 23/29. One item moved between
  two runs whose config hashes are identical. That is the exact failure the config file warns
  about, and it means the numbers here are reproducible only against the weights that happened
  to be resolved today.
- **The `+72 pp` result will be quoted out of context** if this document is skimmed. It is a
  measurement of threshold transfer on a knife edge and it is refuted two paragraphs later.

## Verdict

1. **Ship abstention.** Always-answer is AURC 0.0433 and 13 wrong answers in 300; every
   selective policy measured here is far better than that. This is the largest reliability
   gain available in the repo and the measurement supports taking it.
2. **Ship the single feature, not the combiner.** Threshold `top_score`, the RRF weight of
   rank 1, at a budget of 2.5% rather than 2%. It matches or beats eight fitted features on
   every comparison that is not an artifact, and it has no model file to keep in sync with a
   feature order.
3. **Keep the combiner only if a calibrated probability is needed** — to show a confidence
   number, or to feed a downstream cost model. That is the one thing thresholding a raw RRF
   weight cannot give you, and it costs 0.002 AURC.
4. **Do not report 74% coverage as a system property.** It is 17% on real questions.

## Reproducing

Deterministic: same input, same number. The two bootstraps take an explicit seed, IRLS has no
step-count knob, and `assign_split` hashes the section id rather than seeding an RNG.

```python
from pathlib import Path
from warrant.cli import _buckets, _retriever
from warrant.config import Config
from warrant.index.store import Store
from warrant.verify.calibrate import collect_examples, study

cfg = Config.load(Path("configs/default.yaml"))
with Store(cfg.store_path) as store:
    buckets, _ = _buckets(cfg, store)
    items = [i for v in buckets.values() for i in v]
    ex = collect_examples(_retriever(cfg, store), items, top_k=cfg.retrieve.final_k)

s = study([e for e in ex if e.split == "dev"],
          [e for e in ex if e.split == "test"], seed=0)
print(s.learned.aurc, s.baseline_top_score.aurc, s.ece_calibrated, s.beat_baseline)
```

`collect_examples` drops `scope-exclusion` itself; nothing else needs excluding by hand.
