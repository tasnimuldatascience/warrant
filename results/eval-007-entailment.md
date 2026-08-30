# Eval 007 — entailment measured against span alignment, and what it buys

> **Stale chunk ids.** Some evidence ids quoted below no longer exist: the parser was
> fixed for a roman-numeral defect that malformed 6.07% of in-force anchors, renaming 810 of
> them. Texts and counts are unchanged — every difference is a rename. Left as written,
> because a results doc records what was measured on a day. See
> [eval-012](eval-012-anchor-correctness.md).

> **Superseded serving figures.** This report quotes 21.3 tok/s and a 3-per-minute
> ceiling. Both are wrong: an isolated re-derivation measured 29.2–29.9 tok/s over
> ~205 output tokens, so an answer is 6.6 s and the ceiling is 7.7 req/min. The text
> below is left as it was written, because a results doc is a record of what was
> measured on a day, and editing it to agree with a later number falsifies that
> record. See [eval-010](eval-010-capacity.md).

**Date:** 2026-08-30
**Module:** `src/warrant/verify/entail.py`
**Model:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — 184M parameters, MIT-licensed,
freely downloadable, 377 MB of VRAM in fp16. No API key and no account.
**Hardware:** RTX 5070 Laptop (8 GB, sm_120), fp16; CPU figures are fp32 on the same machine.

`verify/align.py` answers *where in this chunk is the supporting text*, using lexical
overlap. It therefore says yes to a claim that reuses the paragraph's vocabulary and reverses
its meaning: "an employee **may** schedule restored leave" and "an employee **shall**
schedule restored leave" share every content word the aligner counts. Entailment is the
signal that can, in principle, tell those apart. This document measures whether it does, at
what cost, and where it should not be believed.

**Nothing here is a gate.** Every number below is reported so that an abstention policy can
read it; no arrangement of these signals drops a claim on its own. A verifier that vetoed
answers would substitute a 184M-parameter model trained on newswire for the text of 5 CFR,
and would do it with no record that it had.

---

## The probe set

**182 hand-labelled (claim, evidence) pairs over 91 sections of 22 parts of 5 CFR**, in two
strata that are never pooled into one headline.

| stratum | n | sections | E | N | C | where the claims come from |
|---|---:|---:|---:|---:|---:|---|
| generator | 129 | 74 | 102 | 24 | 3 | Qwen2.5-1.5B-Instruct answering the 56 `human.yaml` questions over real retrieved context |
| adversarial | 53 | 20 | 21 | 10 | 22 | author-written minimal edits of verbatim in-force chunks |

The generator stratum is the real distribution and is the one the headline reports. The
adversarial stratum exists because **the generator emitted three contradictions in 129
pairs** — it copies its premise — and three cases cannot measure the one channel span
alignment structurally cannot do. Its class balance is chosen, not observed, so it is
reported separately and never averaged in.

Premises are verbatim chunk text from `data/warrant.sqlite3`; every pair names its
`version_id` in the appendix, so any label can be checked against the store.

### Labelling rubric

Applied uniformly, and stated because two of the three classes turn on it:

- **E** — the premise, read alone, establishes the claim. A claim that *narrows* a universal
  rule to a subset stays E: a universal entails its instances.
- **N** — the premise neither establishes nor denies it. This includes the case where the
  claim drops a material antecedent and so reads as a broader freestanding rule than the
  premise states (`gen-27`, `gen-43`, `gen-52`). A claim carrying a dangling pronoun — "the
  employing agency shall certify **his or her** leave account" — is read as a fragment of the
  premise's own sentence rather than a freestanding rule, and stays E (`gen-44`).
- **C** — the premise denies the claim: opposite modality, a different number, or an equality
  the premise contradicts.

Labels are the author's, single-annotated; there is no second annotator and no agreement
statistic, which is a real limitation of every number below.

---

## 1. Domain shift: the drop is not where it was expected

The published MNLI-matched accuracy for this checkpoint is around 90%. The expectation going
in was that regulatory prose would collapse it. **It does not — and that is the wrong thing
to have measured.**

| stratum | micro accuracy | 95% CI | macro | entail | neutral | contradict |
|---|---:|:---:|---:|:---:|:---:|:---:|
| generator | **86.8%** | 80.8–92.5 | **60.1%** | 99/102 (97%) | 12/24 (**50%**) | 1/3 (33%) |
| adversarial | 88.7% | 78.4–96.6 | 89.0% | 20/21 (95%) | 9/10 (90%) | 18/22 (82%) |

Intervals are section-clustered bootstraps from `warrant.eval.stats` — 129 pairs come from 74
sections and a single over-cited claim contributed ten of them, so an item-level bootstrap
would report an interval that is too narrow.

**The headline is carried entirely by 102 near-verbatim entailments.** The generator's habit
is to copy its premise, and on that the model is 97% right. On the two classes a grounding
check exists for it is at or near chance: **half the neutrals come back as entailment**, and
the three contradictions it did emit yielded one detection.

So the finding is not "91% on MNLI, 64% here." It is that a 182-pair micro-average hides a
60% macro-average, and that a system reporting only the micro number would have shipped a
verifier that is 50% accurate on exactly the cases it was built to catch. The two classes
that matter are the two the corpus supplies least of, which is why the adversarial stratum
had to be constructed at all.

Pooled confusion, both strata (rows gold, columns predicted):

| | pred E | pred N | pred C |
|---|---:|---:|---:|
| **gold E** (123) | 119 | 4 | 0 |
| **gold N** (34) | 6 | 21 | 7 |
| **gold C** (25) | 4 | 2 | 19 |

Contradiction recall **76%** (19/25), precision **73%** (19/26). Seven of 157
non-contradictions are falsely flagged — a 4.5% false-flag rate, each one a claim a human is
asked to look at rather than a claim that is dropped.

---

## 2. What entailment buys over span alignment

The comparison is binary on both sides — *does the premise support this claim* — because that
is the only question `align` can answer. `align` says supported when it locates a span;
entailment says supported when the calibrated verdict is `supported`. Paired, section-
clustered, McNemar sign test, from `warrant.eval.stats.paired_delta`.

| stratum | n | agreement | align | entailment | delta | 95% CI | won/lost | p | |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---|
| generator | 129 | 91.5% | 89.1% | 91.5% | **+2.3** | −2.5–7.6 | 7 / 4 | 0.55 | **not measurable** |
| adversarial | 53 | 43.4% | 45.3% | 94.3% | **+49.1** | 40.0–56.7 | 28 / 2 | 8.7e-07 | carries its weight |

**On the claims the generator actually emits, entailment buys nothing measurable.** +2.3
points with the interval straddling zero and p = 0.55. This repository has published exactly
this shape of result before: the cross-encoder reranker measured +0.5 points at p = 0.79
while costing 78% of query latency, and it stayed behind a flag with that number written next
to it. The honest accounting here is the same, with one difference — entailment costs 0.1% of
answer latency rather than 78%, so the cost side of the trade is not what decides it.

**On claims that reverse their premise, entailment buys everything.** The aligner locates a
supporting span in **every one of the 22 adversarial contradictions** — the flipped claim
reuses the premise's vocabulary, which is precisely what overlap scores. That is not a
tunable failure. `MIN_OVERLAP` cannot be raised to fix it, because a contradiction has *more*
overlap with its premise than a genuine paraphrase does.

That is the whole case for the module: it is not an accuracy improvement, it is a **new
channel**. `align` has no way to express "the cited text denies this", and a system whose
generator ever produces a modality flip has no other instrument that would notice.

### The 41 disagreements, adjudicated

Every case where the two signals disagree, with the gold label and which signal was right.
**NLI right in 35, align right in 6, neither in 0.**

The 35 NLI wins are **23 contradictions** the aligner confirmed as supported and **12
mis-citations** where the claim shares vocabulary with a paragraph that does not carry it.
One over-cited claim in the appendix — "an individual may attain career tenure only when
employed … in the competitive service" — was aimed by the generator at ten different
paragraphs of parts 315, 330 and 351, and the aligner found a span in all of them.

The six align wins are one failure mode, and it is worth naming: a true entailment stated in
vocabulary the premise does not use. `gen-0` ("determined by the head of each agency" against
"the authority to determine the length … is delegated to the head of each agency") and
`gen-91` ("the time schedule specifies X" against "X is set out as appendices A and B") are
paraphrases the model reads as neutral. Overlap catches them because the content words match;
the model misses them because the relation is lexical, not inferential. Only `gen-0` is
actually called wrong — the other five (`gen-71`, `gen-82`, `gen-91`, `adv-14`, `adv-36`)
land in the uncertain band, which is the band working.

See the appendix for the full table.

---

## 3. Calibration

Temperature scaling, one parameter, fitted by NLL. It cannot change any argmax — so it cannot
move a single accuracy above — and that is the point: it moves only the confidence a
downstream abstention policy reads, which is the quantity the raw head gets wrong under
domain shift.

| | temperature | ECE | Brier | mean confidence | accuracy |
|---|---:|---:|---:|---:|---:|
| raw head | 1.00 | **9.45%** | 0.2178 | 96.2% | 87.4% |
| calibrated | 1.72 | **4.99%** | 0.2095 | 87.6% | 87.4% |

Fitted value 1.717 on all 182 pairs. The ECE above is **out-of-fold**: each pair is scored
with a temperature fitted on the other 90 sections, and those refits range 1.66–1.74, so the
constant is not one section's artefact. The in-sample figure is 3.98%; the honest number is
4.99%.

Temperature > 1 means the raw head is overconfident — it is as sure about "an employee shall
schedule restored leave for use not later than the end of the leave year" as it was about the
image captions it was trained on. That is the expected direction, and it is the part of
domain shift that *does* show up cleanly here.

### Reliability (calibrated, equal-width bins)

| bin | n | mean confidence | accuracy | gap |
|---|---:|---:|---:|---:|
| 0.4–0.5 | 2 | 48.2% | 0.0% | +48.2 |
| 0.5–0.6 | 3 | 54.0% | 33.3% | +20.6 |
| 0.6–0.7 | 7 | 66.4% | 85.7% | −19.3 |
| 0.7–0.8 | 12 | 75.1% | 75.0% | +0.1 |
| 0.8–0.9 | 56 | 85.8% | 91.1% | −5.3 |
| 0.9–1.0 | 102 | 93.3% | 90.2% | +3.1 |

Above 0.7 the model is calibrated to within about five points. Below it, it is not calibrated
at all — and there are only twelve pairs down there, so the two lowest bins are three and two
observations and should be read as "unreliable", not as "0% accurate".

### The abstain band

`DECISION_FLOOR = 0.70`. Below it, `Verdict.report` returns `uncertain` rather than a
direction.

| floor | believed | coverage | accuracy above | accuracy below |
|---:|---:|---:|---:|---:|
| 0.50 | 180 | 98.9% | 88.3% | 0/2 |
| 0.60 | 177 | 97.3% | 89.3% | 20.0% |
| **0.70** | **170** | **93.4%** | **89.4%** | **58.3%** |
| 0.75 | 163 | 89.6% | 90.2% | 63.2% |
| 0.80 | 158 | 86.8% | 90.5% | 66.7% |
| 0.85 | 134 | 73.6% | 91.8% | 75.0% |
| 0.90 | 102 | 56.0% | 90.2% | 83.8% |

The floor is set where the **separation** is widest, not where accuracy above the line is
highest. At 0.70 the model is right on 89.4% of what it keeps and 58.3% of what it sets
aside; by 0.85 the two have converged to 91.8% against 75.0% and coverage has fallen to 74%,
which buys 2.4 points of accuracy for a quarter of the answers. The band is deliberately
wide.

### The contradiction floor

`CONTRADICT_FLOOR = 0.50`, and contradiction is reported at a lower bar than support. The
costs are asymmetric: a missed contradiction ships a claim the regulation denies, a false one
adds a flag a human reads.

| floor | flagged | recall | precision | false flags |
|---:|---:|---:|---:|---:|
| 0.30 | 28 | 84.0% | 75.0% | 7 |
| 0.40 | 26 | 76.0% | 73.1% | 7 |
| **0.50** | **26** | **76.0%** | **73.1%** | **7** |
| 0.60 | 26 | 76.0% | 73.1% | 7 |
| 0.70 | 24 | 68.0% | 70.8% | 7 |
| 0.80 | 23 | 68.0% | 73.9% | 6 |
| 0.90 | 20 | 60.0% | 75.0% | 5 |

Flat from 0.40 to 0.60, and 0.50 is its middle. 0.30 scores two more true flags out of 25
contradictions, which is inside the noise of a set that small and not worth tuning to.

### The two signals combined

`combine(span, support)` over all 182 pairs:

| gold | supported | uncertain | unsupported | contradicted |
|---|---:|---:|---:|---:|
| E (123) | 117 | 6 | 0 | 0 |
| N (34) | 6 | 10 | 11 | 7 |
| C (25) | **2** | 4 | 0 | 19 |

Two contradictions in 25 come back as `supported` — both from the generator stratum
(`gen-96`, where the claim drops a "+20 percent of NA-8" from a wage-schedule formula, and
`adv-51`). That is the residual risk of putting this on the answer path, and it is why the
combination is reported to the abstention policy rather than acted on here.

---

## 4. Latency budget

Retrieval p50 is 18.4 ms (lexical + dense, concurrent). The cross-encoder reranker adds
~54 ms. Generation runs at 21.3 tok/s unbatched, so a 420-token answer is roughly 20 seconds
and the service ceiling is three requests per minute. That is the binding constraint on
everything.

An answer is **2.14 claims × 1.12 citations = 2.39 (claim, chunk) pairs on average**, max 14
in this sample. `score_claim` batches a whole claim in one call.

| | throughput | one answer (2.4 pairs, one call) |
|---|---:|---:|
| GPU, fp16, batch 16 | 458 pairs/s | **24 ms p50 / 28 ms p95** |
| CPU, fp32, batch 16 | 11.3 pairs/s | **~190 ms** |

Per-call latency is almost flat in batch size at these sizes — 1 pair 22.8 ms p50, 2 pairs
24.4, 4 pairs 24.2, 8 pairs 25.2 — so the cost is a fixed ~22 ms forward-pass overhead plus
roughly 0.4 ms per additional pair. An answer with 14 citations costs about as much as one
with two.

| batch | pairs/s | ms/pair |
|---:|---:|---:|
| 1 | 43.5 | 22.97 |
| 4 | 166.0 | 6.02 |
| 8 | 322.8 | 3.10 |
| **16** | **458.5** | **2.18** |
| 32 | 395.2 | 2.53 |
| 64 | 308.7 | 3.24 |

Throughput *peaks* at 16 and falls beyond it on this card, so `DEFAULT_BATCH = 16` is a
measurement rather than a convention. Weights plus batch-16 activations are 918 MB peak,
which co-exists with the 1.5B generator inside 8 GB.

**Verdict: it belongs on the synchronous path.** 24 ms against a ~20 s answer is 0.1%. Even
the CPU path, at ~190 ms, is 1%. There is no latency argument for making this an async audit
behind the answer; if it is deferred it should be for the reason in §2 — that on real
generator output it does not measurably beat the aligner — not for cost.

Cold model load is 1.4 s from a warm disk cache, paid once per process by the module-level
cache in `_load`.

---

## 5. Model choice

Both candidates were already in the local HuggingFace cache; neither was downloaded for this
evaluation.

| model | params | pairs/s | generator micro / macro | adversarial micro / macro | temp | ECE raw → cal |
|---|---:|---:|---:|---:|---:|---|
| **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli** | 184M | 223 | **86.8% / 60.1%** | **88.7% / 89.0%** | 1.72 | 9.45% → 3.98% |
| cross-encoder/nli-deberta-v3-base | 184M | 448 | 80.6% / 58.6% | 77.4% / 76.1% | 2.96 | 18.35% → 9.95% |

The cross-encoder checkpoint is twice as fast at the same parameter count and loses 6 points
on the real stratum, 11 on the adversarial one, and is badly calibrated even after fitting —
its raw head is roughly three times as overconfident. On a stage costing 24 ms per answer the
speed is not worth buying. The two also order their classification heads differently —
`entailment/neutral/contradiction` against `contradiction/entailment/neutral` — which is why
`_label_order` reads `id2label` from the checkpoint and raises rather than guessing. A
hard-coded index 0 reads the second checkpoint's contradictions as support, with every
published number still in range and the pipeline still green.

---

## 6. Reproducibility

- **Determinism.** Seeded, `eval()`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
  Two `Entailer` instances return bit-identical logits on the same input.
- **Device.** fp16 on GPU and fp32 on CPU agreed on **182/182** argmaxes, maximum logit
  difference 0.0055. The verdicts in this document do not depend on having a card.
- **Batch composition.** Batches are formed in input order and never sorted by length, so a
  verdict does not move because a different claim was scored alongside it. This costs 455
  against 560 pairs/s — 19% — on a stage that is 0.1% of answer latency. It buys
  reproducibility at a *fixed* batch size, not independence from batching: re-scoring at
  batch 7 instead of 16 moved logits by up to 0.014 and no argmax at all, so `batch_size`
  belongs in a trace beside the model name.
- **Windowing.** Premises longer than the encoder are split into overlapping sentence windows
  rather than truncated, and the most *decisive* window wins rather than the most entailing
  one. Measured on the 9,961 in-force chunks, this is close to dead code: p99 is 209 tokens
  and **exactly one chunk exceeds 500**. It stays because that one is a long procedural
  section, which is exactly the shape of text where truncation would drop the proviso that
  decides the answer.

---

## 7. What this does not establish

- **One annotator.** No second labeller, no agreement statistic. The rubric above is stated
  so a reader can disagree with a specific label rather than with the number.
- **Three real contradictions.** The contradiction results rest on an author-written
  adversarial set. They measure whether the model can read a flipped modality, not how often
  this generator flips one. The generator's observed rate is 3 in 129 — and eval-004 already
  reports a 1.5% hallucination rate — so the channel is currently guarding against something
  rare. It is guarding against the failure that would be worst.
- **The MNLI comparison is a published figure, not one measured here.** No MNLI validation
  split was downloaded; the test suite does not touch the network.
- **The over-general claim is not caught by either signal.** `gen-27` — "an agency must
  establish a retention register for a competitive level", where §351.404 imposes that duty
  only *when a competing employee is to be released* — gets a span from the aligner and 0.95
  entailment from the model. Both signals agree and both are wrong. Half the generator's
  neutrals are of this shape. Dropping a material antecedent is a distinct failure from
  reversing a rule, and neither instrument here detects it.
- **The corpus is one title.** 5 CFR chapter I. Nothing here says how this transfers to
  another body of regulation.

---

## Integration

`verify/entail.py` is a library and nothing calls it yet. The wiring it needs:

1. **A benchmark file.** The appendix below is the probe set; promoting it to
   `benchmarks/entailment.yaml` (premise by `version_id`, claim, label) and a
   `warrant verify entail` command makes every number here re-runnable rather than
   transcribed. Evidence should be written `section#anchor` and resolved at load time, the
   way `human.yaml` does, so labels do not rot on the next rebuild.
2. **The abstention policy reads it, and nothing else does.** `combine()` returns one of four
   constants and never "verified". `verify/abstain.py` is the only thing that should act on
   them.
3. **Behind a config flag with the §2 number next to it**, exactly as
   `index.rerank.enabled` carries the reranker's +0.5 / p = 0.79. The flag default is a
   judgement call this document does not make: the contradiction channel is real and cheap,
   and on observed generator output it is +2.3 at p = 0.55.
4. **Trace `model_name`, `revision` and `batch_size`.** All three move the verdict.

---

## Appendix — every label

Model column is the raw argmax; **bold** marks a disagreement with the gold label.
Confidence is calibrated. Premises are named by `version_id` and can be read out of
`data/warrant.sqlite3`.

### Generator-emitted pairs (129)

| # | version id | claim | gold | model | conf |
|---:|---|---|:---:|:---:|---:|
| 0 | `315.905#p1@2017-01-01` | The probationary period for a new federal employee is determined by the head of each agency. | **E** | **N** | 0.95 |
| 1 | `315.905#p1@2017-01-01` | An agency may establish different probationary periods for different occupations or a single one for all agency employees. | **E** | E | 0.95 |
| 2 | `315.906#a@2017-01-01` | If the former position was supervisory and the new position managerial, service counts in the manner prescribed by agency regulation. | **E** | E | 0.90 |
| 3 | `315.907#a@2017-01-01` | An employee who, for reasons of supervisory or managerial performance, does not satisfactorily complete the probationary period is entitled to be a... | **E** | E | 0.88 |
| 4 | `315.907#a@2017-01-01` | Satisfactory completion of the prescribed probationary period is a prerequisite to continued service in the position. An employee who, for reasons ... | **E** | E | 0.91 |
| 5 | `316.301#c-1@2026-08-25` | An agency may make a term appointment for a period of more than 1 year but not more than 10 years to a covered position defined in (2) when the nee... | **E** | E | 0.88 |
| 6 | `316.301#c-1@2026-08-25` | No appointment made under this section may last longer than 10 years from the date of the initial appointment. | **E** | E | 0.87 |
| 7 | `432.102#f-1@2026-03-09` | An employee in the competitive service who is serving a probationary or trial period under an initial appointment; | **E** | E | 0.87 |
| 8 | `316.304#a@2025-06-24` | The first year of service of a term employee is a probationary period regardless of the method of appointment. | **E** | E | 0.94 |
| 9 | `316.401#a@2017-01-01` | An agency may make a temporary appointment under part 332 of this chapter, by using competitive procedures, or under part 337 of this chapter, by u... | **N** | N | 0.88 |
| 10 | `316.402#a@2026-08-25` | An agency may make a temporary appointment under part 332 of this chapter, by using competitive procedures, or under part 337 of this chapter, by u... | **E** | E | 0.86 |
| 11 | `316.402#b@2026-08-25` | In accordance with the time limits in § 316.401, an agency may give a noncompetitive temporary appointment, without regard to the requirements of p... | **E** | E | 0.92 |
| 12 | `315.201#b@2026-03-23` | The service requirement for career tenure includes service as described in paragraph (b)(1) of this section. | **E** | E | 0.83 |
| 13 | `315.201#b@2026-03-23` | Total at least 3 years. | **E** | E | 0.89 |
| 14 | `317.703#d-2@2017-01-01` | An agency may voluntarily reinstate a former Presidential appointee without an order from OPM directing such action. | **E** | E | 0.92 |
| 15 | `890.1052#p1@2017-01-01` | OPM shall reinstate a provider without a reinstatement application if: | **E** | E | 0.93 |
| 16 | `315.612#d@2024-02-28` | A spouse is eligible for noncompetitive appointment | **E** | E | 0.92 |
| 17 | `315.612#d-iii-4@2024-02-28` | Any law, Executive order, or regulation that disqualifies an applicant for appointment also disqualifies a spouse for appointment under this section | **E** | E | 0.92 |
| 18 | `335.103#a@2024-08-26` | An agency may make promotions under § 335.102 of this part only to positions for which the agency has adopted and is administering a program design... | **E** | E | 0.90 |
| 19 | `335.103#b@2024-08-26` | Each agency must establish procedures for promoting employees that are based on merit and are available in writing to candidates. | **E** | E | 0.95 |
| 20 | `335.102#a@2017-01-01` | (Agency authority to promote, demote, or reassign) (a) Promote, demote, or reassign a career or career-conditional employee; | **E** | E | 0.83 |
| 21 | `335.102#e@2017-01-01` | (Agency authority to promote, demote, or reassign) (e) Promote, demote, or reassign a term employee serving on a given project to another position ... | **E** | E | 0.80 |
| 22 | `335.102#d@2017-01-01` | (Agency authority to promote, demote, or reassign) (d) Promote, demote, or reassign (1) a status quo employee and (2) an employee serving under an ... | **E** | E | 0.75 |
| 23 | `335.102#c@2017-01-01` | (Agency authority to promote, demote, or reassign) (c) Promote, demote, or reassign an employee serving under an overseas limited appointment of in... | **E** | E | 0.83 |
| 24 | `351.402#b@2026-08-25` | A competitive area must be defined solely in terms of the agency's organizational unit(s) and geographical location. | **E** | E | 0.90 |
| 25 | `351.402#b@2026-08-25` | A competitive area may consist of all or part of an agency. | **E** | E | 0.95 |
| 26 | `351.402#b@2026-08-25` | The minimum competitive area is a subdivision of the agency under separate administration within the local commuting area. | **E** | E | 0.79 |
| 27 | `351.404#a@2017-01-01` | An agency must establish a retention register for a competitive level. | **N** | **E** | 0.96 |
| 28 | `351.404#a@2017-01-01` | The retention register is prepared from the current retention records of employees. | **E** | E | 0.92 |
| 29 | `351.601#a@2017-01-01` | Employees are retained in a RIF in the competitive service in the inverse order of retention standing. | **C** | **N** | 0.71 |
| 30 | `351.601#a@2017-01-01` | An agency may not release a competing employee from a competitive level while retaining in that level an employee with lower retention standing exc... | **E** | E | 0.86 |
| 31 | `351.608#f@2017-01-01` | Service in the former position counts toward completion of the probationary period in the new position. | **N** | N | 0.91 |
| 32 | `351.608#f@2017-01-01` | If the former position was supervisory and the new position managerial, service counts in the manner prescribed by agency regulation. | **N** | N | 0.93 |
| 33 | `351.608#g-1@2017-01-01` | An agency may provide for a cutoff date, a specified number of days prior to the issuance of reduction in force notices after which no new ratings ... | **N** | N | 0.84 |
| 34 | `351.608#g-1@2017-01-01` | When a cutoff date is used, an employee will receive performance credit for the three most recent ratings of record received during the 4-year peri... | **N** | N | 0.93 |
| 35 | `351.804#b@2017-01-01` | An agency may not take the action before the effective date in the notice; | **E** | E | 0.95 |
| 36 | `351.805#c@2017-01-01` | The agency must give an employee an amended written notice and allow the employee to decide whether to accept a better offer of assignment under su... | **E** | E | 0.89 |
| 37 | `351.604#a@2017-01-01` | An agency may furlough a competing employee only when it intends within 1 year to recall the employee to duty in the position from which furloughed. | **E** | E | 0.88 |
| 38 | `351.604#d@2017-01-01` | When an agency recalls employees to duty in the competitive level from which furloughed, it shall recall them in the order of their retention stand... | **E** | E | 0.91 |
| 39 | `353.201#p1@2017-01-01` | The Uniformed Services Employment and Reemployment Rights Act of 1994 revised and strengthened the existing Veterans' Reemployment Rights law | **E** | E | 0.92 |
| 40 | `353.201#p1@2017-01-01` | The new law applies to persons exercising restoration rights on or after December 12, 1994. | **E** | E | 0.95 |
| 41 | `353.207#a@2017-01-01` | An employee returning from the uniformed services following an absence of more than 30 days is entitled to be restored as soon as possible after ma... | **E** | E | 0.89 |
| 42 | `353.205#c@2017-01-01` | (c) If the period of service was for more than 180 days, the employee must submit an application for reemployment not later than 90 days after comp... | **E** | E | 0.88 |
| 43 | `353.207#a@2017-01-01` | An employee is entitled to be restored as soon as possible after making application, but in no event later than 30 days after receipt of the applic... | **N** | **E** | 0.93 |
| 44 | `630.504#a@2017-01-01` | The employing agency shall certify his or her leave account for credit or charge. | **E** | E | 0.90 |
| 45 | `410.309#b-2@2017-01-01` | An employee selected for training subject to an agency continued service agreement must sign an agreement to continue in service after training. | **E** | E | 0.82 |
| 46 | `410.309#c@2017-01-01` | With a signed agreement, the agency has a right to recover training costs, except pay or other compensation, if the employee voluntarily separates ... | **E** | E | 0.94 |
| 47 | `410.308#a@2017-01-01` | An agency may authorize training for an employee to obtain an academic degree under conditions prescribed at 5 U.S.C. 4107(a). | **E** | E | 0.92 |
| 48 | `410.308#e@2017-01-01` | On a periodic basis, OPM may request agency information on the use and effectiveness of training assignments under this section. | **E** | E | 0.91 |
| 49 | `430.206#a@2026-08-06` | An appraisal program shall designate an official appraisal period for which a performance plan shall be prepared, during which performance shall be... | **E** | E | 0.86 |
| 50 | `430.204#b-3-i@2017-01-01` | The length of the appraisal period (as specified in § 430.206(a)). | **E** | E | 0.89 |
| 51 | `430.207#a@2026-08-06` | An appraisal program shall establish a minimum period of performance that must be completed before a performance rating may be prepared. | **E** | E | 0.93 |
| 52 | `430.208#i@2026-08-25` | An agency must not produce or change retroactively a rating of record that covers an earlier appraisal period except that a rating of record may be... | **N** | **E** | 0.90 |
| 53 | `430.208#h@2026-08-25` | Each rating of record must cover a specified appraisal period. | **E** | E | 0.95 |
| 54 | `430.309#c@2017-01-01` | Once the appropriate conditions are met, the agency will then prepare the annual summary rating. | **E** | E | 0.91 |
| 55 | `432.102#a@2026-03-09` | Actions covered include reduction in grade and removal of employees based on unacceptable performance. | **E** | E | 0.93 |
| 56 | `432.102#a@2026-03-09` | This part covers reduction in grade and removal of employees based on unacceptable performance. | **E** | E | 0.92 |
| 57 | `432.105#a-4@2022-12-12` | An employee whose reduction in grade or removal is proposed under this part is entitled to: | **E** | E | 0.93 |
| 58 | `432.105#a-1@2022-12-12` | Once an employee has been afforded a reasonable opportunity to demonstrate acceptable performance pursuant to § 432.104, an agency may propose a re... | **E** | E | 0.87 |
| 59 | `451.103#c@2017-01-01` | An agency award program shall provide for... | **E** | E | 0.91 |
| 60 | `451.103#b@2017-01-01` | Agencies are encouraged to involve employees in developing such programs. | **E** | E | 0.96 |
| 61 | `451.105#b@2017-01-01` | In accordance with 5 U.S.C. 4509, agencies shall not grant cash awards under this subpart to employees appointed by the President with Senate confi... | **E** | E | 0.85 |
| 62 | `451.104#a@2017-01-01` | An agency may grant a cash, honorary, or informal recognition award, or grant time-off without charge to leave or loss of pay consistent with chapt... | **E** | E | 0.85 |
| 63 | `531.610#i@2017-01-01` | Performance-based cash awards under 5 U.S.C. 4505a and 5 CFR part 451, subpart A, when such awards are computed as a percentage of an employee's ra... | **E** | E | 0.84 |
| 64 | `630.1403#b-2@2025-01-16` | As a reward to recognize the performance or contributions of an employee or group of employees (i.e., in lieu of a cash award or a time-off award). | **E** | E | 0.90 |
| 65 | `531.404#c@2017-01-01` | The employee must not have received an equivalent increase during the waiting period. | **E** | E | 0.94 |
| 66 | `531.404#a-2@2017-01-01` | The rating of record used as the basis for an acceptable level of competence determination for a within-grade increase must have been assigned no e... | **E** | E | 0.90 |
| 67 | `531.404#p1@2017-01-01` | An employee paid at less than the maximum rate of the grade of his or her position shall earn advancement in pay to the next higher step of the gra... | **E** | E | 0.90 |
| 68 | `532.417#c@2017-01-01` | Creditable service. The following periods of time shall be considered creditable service for purposes of waiting periods for within-grade increases: | **E** | E | 0.90 |
| 69 | `531.405#b-2@2017-01-01` | The waiting period for within-grade increases varies depending on the position. | **N** | N | 0.93 |
| 70 | `532.417#c@2017-01-01` | The waiting period for within-grade increases varies depending on the position. | **N** | N | 0.93 |
| 71 | `532.267#g@2024-02-23` | For employees with a scheduled tour of duty, the waiting period is 26 weeks between steps 1 and 2 and 78 weeks between steps 2 and 3. | **E** | **N** | 0.53 |
| 72 | `532.417#d@2017-01-01` | A within-grade increase shall be effective at the beginning of the first applicable pay period following the day an employee becomes eligible for t... | **E** | E | 0.91 |
| 73 | `531.414#b@2017-01-01` | An interim within-grade increase granted under paragraph (a) of this section shall become effective on the date of the appellate decision ordering ... | **E** | E | 0.88 |
| 74 | `531.504#p1@2017-01-01` | A quality step increase shall not be required but may be granted only to— | **E** | E | 0.94 |
| 75 | `531.504#a@2017-01-01` | (a) An employee who receives a rating of record at Level 5 (“Outstanding” or equivalent), as defined in part 430, subpart B, of this chapter; | **E** | E | 0.74 |
| 76 | `531.603#a@2023-12-18` | Locality rates of pay under this subpart shall be payable to employees whose official worksites are located in the locality pay areas listed in par... | **E** | E | 0.89 |
| 77 | `531.604#b-2@2017-01-01` | Determination of the locality pay area in which the employee's official worksite is located, consistent with the locality pay areas established in ... | **E** | E | 0.89 |
| 78 | `550.706#a@2017-01-01` | An employee who resigns because he or she expects to be involuntarily separated is considered to have been involuntarily separated if the employee ... | **E** | E | 0.91 |
| 79 | `550.704#a@2026-06-10` | To be eligible for severance pay, an employee must: | **E** | E | 0.93 |
| 80 | `550.704#b-6@2026-06-10` | Occupies a position in Schedule Policy/Career of the excepted service and his or her agency identifies unacceptable performance or misconduct as th... | **E** | E | 0.86 |
| 81 | `550.707#d@2017-01-01` | The severance pay fund is limited to that amount which would provide 52 weeks of severance pay taking into account weeks of severance pay previousl... | **E** | E | 0.87 |
| 82 | `550.707#a@2017-01-01` | The basic severance pay allowance consists of the following: | **E** | E | 0.68 |
| 83 | `550.707#b@2017-01-01` | In the following circumstances, the weekly rate of basic pay used in computing the basic severance pay allowance must be determined based on the we... | **E** | E | 0.83 |
| 84 | `550.707#b-4@2017-01-01` | For positions with seasonal work requirements, compute the weekly average of hours in a pay status (excluding overtime hours) and multiply that ave... | **E** | E | 0.85 |
| 85 | `550.203#e@2017-01-01` | An advance in pay may not be made to the head of an agency or to an employee appointed to a position in the expectation of receiving an appointment... | **E** | E | 0.93 |
| 86 | `300.103#b-2-ii@2026-07-31` | New employees, within a reasonable period of time and in the great majority of cases, can expect to progress to a target position at a higher level. | **E** | E | 0.91 |
| 87 | `550.171#a@2017-01-01` | An employee is entitled to pay at his or her rate of basic pay plus premium pay at a rate equal to 25 percent of his or her rate of basic pay for e... | **E** | E | 0.92 |
| 88 | `532.509#p1@2017-01-01` | A wage employee whose regular work schedule includes a period of service of up to 8 hours which is not overtime work, a part of which is on Sunday,... | **E** | E | 0.91 |
| 89 | `550.183#b@2017-01-01` | For the purpose of this section, regular workday means each day in the criminal investigator's basic workweek during which the investigator works a... | **E** | E | 0.84 |
| 90 | `550.182#c@2017-01-01` | To be considered to be performing work under paragraph (a) of this section, a criminal investigator must be performing work as officially ordered o... | **E** | E | 0.85 |
| 91 | `532.207#f@2021-03-31` | The time schedule for wage surveys specifies the beginning month of appropriated and nonappropriated fund wage surveys and the fiscal year during w... | **E** | **N** | 0.49 |
| 92 | `532.207#a@2021-03-31` | Wage surveys shall be conducted on a 2-year cycle at annual intervals. | **E** | E | 0.93 |
| 93 | `532.203#b@2017-01-01` | Each supervisory regular wage schedule has 19 grades. | **E** | E | 0.80 |
| 94 | `532.203#a@2017-01-01` | Each nonsupervisory and leader regular wage schedule has 15 grades. | **E** | E | 0.82 |
| 95 | `532.203#d@2017-01-01` | The step 2 or payline rate for each grade of an appropriated fund supervisory regular wage schedule is specified. | **E** | E | 0.71 |
| 96 | `532.203#e-1@2017-01-01` | For grades NS-1 through NS-8, the rate for step 2 of the corresponding grade of the nonsupervisory regular wage schedule for the area is used. | **C** | **E** | 0.85 |
| 97 | `317.503#c@2017-01-01` | The probationary period begins on the effective date of the personnel action initially appointing the individual to the SES as a career appointee a... | **E** | E | 0.91 |
| 98 | `575.110#b-3@2026-02-13` | An agency may delay a service agreement commencement date until after the employee completes an initial period of formal training or required proba... | **E** | E | 0.92 |
| 99 | `315.612#d@2024-02-28` | A spouse is eligible for noncompetitive appointment: | **E** | E | 0.92 |
| 100 | `315.612#d-iii-4@2024-02-28` | Any law, Executive order, or regulation that disqualifies an applicant for appointment also disqualifies a spouse for appointment under this section. | **E** | E | 0.93 |
| 101 | `317.301#c@2017-01-01` | Employees excluded from coverage of this subpart and are not entitled to conversion to the Senior Executive Service. | **E** | E | 0.93 |
| 102 | `330.607#i@2017-01-01` | An agency may deny a CTAP eligible future selection priority if the eligible: | **E** | E | 0.94 |
| 103 | `330.607#f@2017-01-01` | If there are two or more CTAP selection priority candidates for a vacancy, the agency may place any of them. | **E** | E | 0.91 |
| 104 | `330.609#b@2026-08-25` | An agency may choose to consider RPL placement priority candidates before other agency permanent competitive service employees under its Career Tra... | **N** | N | 0.66 |
| 105 | `330.609#b@2026-08-25` | In filling vacancies, an agency must give its RPL registrants placement priority for most competitive service vacancies before hiring someone from ... | **N** | N | 0.77 |
| 106 | `315.201#b-1@2026-03-23` | To be creditable, the 3 years of service must begin with one of the following: | **E** | E | 0.88 |
| 107 | `315.201#b-3@2026-03-23` | An employee's creditable service must total at least 3 years, under the following conditions: | **E** | E | 0.92 |
| 108 | `315.201#b-2@2026-03-23` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **E** | E | 0.88 |
| 109 | `351.501#b-1-ii@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.77 |
| 110 | `315.201#b-1-ii@2026-03-23` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **E** | 0.82 |
| 111 | `315.702#c@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.95 |
| 112 | `315.701#e@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.97 |
| 113 | `315.710#d@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.95 |
| 114 | `315.201#b-1-iii@2026-03-23` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **E** | 0.83 |
| 115 | `315.706#c@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.96 |
| 116 | `315.709#c@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | **C** | 0.95 |
| 117 | `330.101#p10@2017-01-01` | An individual may attain career tenure only when employed (or reemployed) in a permanent appointment in the competitive service that provides or le... | **N** | N | 0.53 |
| 118 | `315.201#b-1-iii@2026-03-23` | A person whose employment is converted to career or career-conditional employment under this section acquires a competitive status automatically on... | **N** | N | 0.69 |
| 119 | `315.706#c@2017-01-01` | A person whose employment is converted to career or career-conditional employment under this section acquires a competitive status automatically on... | **N** | **E** | 0.70 |
| 120 | `315.709#c@2017-01-01` | A person whose employment is converted to career or career-conditional employment under this section acquires a competitive status automatically on... | **E** | E | 0.93 |
| 121 | `330.101#p10@2017-01-01` | A person whose employment is converted to career or career-conditional employment under this section acquires a competitive status automatically on... | **N** | N | 0.82 |
| 122 | `550.409#c@2017-01-01` | An agency must terminate evacuation payments under the conditions listed in § 550.407. | **C** | C | 0.85 |
| 123 | `890.502#b@2017-01-01` | The employing office must tell the employee about available health benefits choices as soon as it becomes aware that an employee's premium payments... | **E** | E | 0.85 |
| 124 | `432.101#p1@2017-01-01` | This part applies to reduction in grade and removal of employees covered by the provisions of this part based solely on performance at the unaccept... | **E** | E | 0.91 |
| 125 | `630.306#b-1@2020-08-10` | A full-time employee shall schedule and use excess annual leave of 416 hours or less by the end of the leave year in progress 2 years after the dat... | **E** | E | 0.83 |
| 126 | `630.306#b-1@2020-08-10` | The agency shall extend this period by 1 leave year for each additional 208 hours of excess annual leave or any portion thereof. | **E** | E | 0.88 |
| 127 | `630.306#b-2@2020-08-10` | A part-time employee shall schedule and use excess annual leave in an amount equal to or less than 20 percent of the number of hours in the employe... | **E** | E | 0.74 |
| 128 | `550.805#g-2@2017-01-01` | The agency shall extend this period by 1 leave year for each additional number of hours of excess annual leave, or any portion thereof, equal to 10... | **E** | E | 0.84 |

### Adversarial pairs (53)

| # | version id | claim | gold | model | conf |
|---:|---|---|:---:|:---:|---:|
| 0 | `317.402#c@2017-01-01` | Every qualifications criterion in the standard must be job related. | **E** | E | 0.95 |
| 1 | `317.402#c@2017-01-01` | The standard may emphasize agency-related experience even where doing so precludes well-qualified candidates from outside the agency. | **C** | C | 0.92 |
| 2 | `317.402#c@2017-01-01` | The agency must publish the qualification standard in the vacancy announcement. | **N** | N | 0.95 |
| 3 | `317.703#c@2017-01-01` | OPM will, so far as practicable, direct reinstatement within 45 days of the later of the application's receipt and the separation date. | **E** | E | 0.84 |
| 4 | `317.703#c@2017-01-01` | OPM will direct reinstatement within 30 days of receiving the application. | **C** | C | 0.94 |
| 5 | `317.703#c@2017-01-01` | The reinstated employee returns at the grade held before the Presidential appointment. | **N** | N | 0.95 |
| 6 | `317.703#f@2017-01-01` | An agency has at most 30 calendar days from the date of an OPM reinstatement order to comply with it. | **E** | E | 0.93 |
| 7 | `317.703#f@2017-01-01` | An agency may take up to 30 business days from the date of the order to comply. | **C** | C | 0.87 |
| 8 | `330.213#a@2026-07-31` | An agency may vary its RPL selection method by location. | **E** | E | 0.94 |
| 9 | `330.213#a@2026-07-31` | An agency may vary the selection method it uses for an individual vacancy. | **C** | C | 0.94 |
| 10 | `330.213#a@2026-07-31` | An agency must keep a registrant on the RPL for one year. | **N** | N | 0.94 |
| 11 | `337.204#d-3@2019-05-03` | The agency head must notify OPM within 10 business days of approving the direct hire authority. | **E** | E | 0.93 |
| 12 | `337.204#d-3@2019-05-03` | Notification to OPM need not describe the evidence the determination relied on. | **C** | C | 0.97 |
| 13 | `337.204#d-3@2019-05-03` | OPM may revoke a direct hire authority it has approved. | **N** | N | 0.84 |
| 14 | `351.402#c@2026-08-25` | A competitive area in effect for fewer than 90 days before the RIF takes effect must be described to OPM for approval beforehand. | **E** | **N** | 0.68 |
| 15 | `351.402#c@2026-08-25` | A competitive area in effect for fewer than 90 days before the RIF needs no OPM approval. | **C** | C | 0.97 |
| 16 | `353.209#b@2017-01-01` | An employee reemployed under this subpart may not be discharged except for cause. | **E** | E | 0.90 |
| 17 | `353.209#b@2017-01-01` | An employee reemployed under this subpart may be discharged at the agency's discretion. | **C** | C | 0.62 |
| 18 | `353.209#b@2017-01-01` | A reemployed employee is restored to the position he or she would have held had the service not intervened. | **N** | N | 0.79 |
| 19 | `532.703#b-3@2017-01-01` | To get retroactive relief for a downgrading, the employee must ask for review within 15 calendar days of the effective date of the change. | **E** | E | 0.97 |
| 20 | `532.703#b-3@2017-01-01` | An application involving a downgrading may be filed at any time and still carry entitlement to retroactive corrective action. | **C** | **E** | 0.55 |
| 21 | `534.404#j-3-ii@2024-04-01` | The senior executive gets at least 7 days to respond to the notice. | **E** | E | 0.94 |
| 22 | `534.404#j-3-ii@2024-04-01` | The senior executive must respond within 7 days of the notice. | **C** | **E** | 0.47 |
| 23 | `534.404#j-3-ii@2024-04-01` | The response must be made in writing. | **C** | C | 0.62 |
| 24 | `534.604#b@2024-04-01` | An administrative appeals judge is normally appointed at rate AA-1. | **E** | E | 0.74 |
| 25 | `534.604#b@2024-04-01` | An agency may set an administrative appeals judge's initial rate of basic pay at any rate in the range. | **C** | C | 0.95 |
| 26 | `550.905#b@2017-01-01` | Hours paid availability pay under 550.181 cannot also draw a hazardous duty differential. | **E** | E | 0.94 |
| 27 | `550.905#b@2017-01-01` | An employee receiving availability pay under 550.181 may also be paid a hazardous duty differential for those hours. | **C** | C | 0.95 |
| 28 | `550.905#b@2017-01-01` | The hazardous duty differential is 25 percent of the employee's rate of basic pay. | **N** | N | 0.88 |
| 29 | `575.507#a-1@2017-01-01` | The amount is a quarter of the employee's annual basic pay at the start of the service period, multiplied by the number of years in that period. | **E** | E | 0.94 |
| 30 | `575.507#a-1@2017-01-01` | The amount is 25 percent of the employee's annual rate of basic pay at the end of the service period times the number of years in that period. | **C** | C | 0.95 |
| 31 | `591.239#b@2017-01-01` | A COLA is excluded from basic pay when overtime entitlement is computed. | **E** | E | 0.92 |
| 32 | `591.239#b@2017-01-01` | A post differential counts as part of basic pay when retirement entitlement is computed. | **C** | C | 0.96 |
| 33 | `630.1703#f-3@2020-10-15` | A seasonal employee cannot take paid parental leave in the agency-designated off-season. | **E** | E | 0.97 |
| 34 | `630.1703#f-3@2020-10-15` | A seasonal employee may use paid parental leave during the off-season period designated by the agency. | **C** | C | 0.96 |
| 35 | `630.1703#f-3@2020-10-15` | Paid parental leave must be used within 12 months of the birth or placement. | **N** | N | 0.82 |
| 36 | `630.502#b@2017-01-01` | An employee who returns to Federal employment after December 2, 1994 is generally entitled to have sick leave recredited, whenever the separation occurred. | **E** | E | 0.69 |
| 37 | `630.502#b@2017-01-01` | Sick leave is recredited only when the break in service was shorter than three years. | **C** | **N** | 0.91 |
| 38 | `890.1016#a-3@2017-01-01` | Restitution the provider has paid is left out of OPM's assessment of financial loss. | **E** | E | 0.95 |
| 39 | `890.1016#a-3@2017-01-01` | OPM offsets any restitution the provider has paid against the financial loss it finds. | **C** | C | 0.93 |
| 40 | `890.1040#e@2017-01-01` | Disputed facts are resolved on a preponderance of the evidence. | **E** | E | 0.92 |
| 41 | `890.1040#e@2017-01-01` | The presiding official must issue the written report within 30 days of the hearing opening. | **C** | C | 0.94 |
| 42 | `890.1040#e@2017-01-01` | The suspending official may reject the presiding official's findings. | **N** | N | 0.89 |
| 43 | `530.304#c@2017-01-01` | The special rate supplement is normally a fixed dollar amount or fixed percentage added to every GS rate in the range. | **E** | E | 0.88 |
| 44 | `530.304#c@2017-01-01` | OPM computes the special rate supplement by a fixed dollar amount or fixed percentage and no alternate method is available. | **C** | C | 0.97 |
| 45 | `351.302#b@2017-01-01` | An employee transferred solely for liquidation, whose function was not authorized to run past 60 days, does not compete for positions in the gaining competitive area. | **E** | E | 0.96 |
| 46 | `351.302#b@2017-01-01` | An employee transferred solely for liquidation competes for other positions in the gaining competitive area. | **C** | C | 0.93 |
| 47 | `575.110#b-1@2026-02-13` | The service period ends on the last day of a pay period. | **E** | E | 0.94 |
| 48 | `575.110#b-1@2026-02-13` | The service period may terminate on any day the agency chooses. | **C** | C | 0.94 |
| 49 | `575.110#b-1@2026-02-13` | The agency must repay a terminated incentive on a pro rata basis. | **N** | N | 0.98 |
| 50 | `630.906#b@2017-01-01` | Donated annual leave moves into the leave recipient's annual leave account under the recipient agency's procedures. | **E** | E | 0.95 |
| 51 | `630.906#b@2017-01-01` | The transfer follows procedures established by the leave donor's employing agency. | **C** | **E** | 0.84 |
| 52 | `630.906#b@2017-01-01` | A leave recipient must exhaust his or her own annual leave before using donated leave. | **N** | **C** | 0.84 |

### Disagreements

| # | section | claim (abridged) | gold | align | entailment | right |
|---:|---|---|:---:|:---:|:---:|:---:|
| gen-0 | 315.905 | The probationary period for a new federal employee is determined by the head of each agency. | **E** | span | unsupported | **align** |
| gen-9 | 316.401 | An agency may make a temporary appointment under part 332 of this chapter, by using competitive p... | **N** | span | unsupported | **NLI** |
| gen-29 | 351.601 | Employees are retained in a RIF in the competitive service in the inverse order of retention stan... | **C** | span | unsupported | **NLI** |
| gen-70 | 532.417 | The waiting period for within-grade increases varies depending on the position. | **N** | span | unsupported | **NLI** |
| gen-71 | 532.267 | For employees with a scheduled tour of duty, the waiting period is 26 weeks between steps 1 and 2... | **E** | span | uncertain | **align** |
| gen-82 | 550.707 | The basic severance pay allowance consists of the following: | **E** | span | uncertain | **align** |
| gen-91 | 532.207 | The time schedule for wage surveys specifies the beginning month of appropriated and nonappropria... | **E** | span | uncertain | **align** |
| gen-109 | 351.501 | An individual may attain career tenure only when employed (or reemployed) in a permanent appointm... | **N** | span | contradicted | **NLI** |
| gen-117 | 330.101 | An individual may attain career tenure only when employed (or reemployed) in a permanent appointm... | **N** | span | uncertain | **NLI** |
| gen-118 | 315.201 | A person whose employment is converted to career or career-conditional employment under this sect... | **N** | span | uncertain | **NLI** |
| gen-122 | 550.409 | An agency must terminate evacuation payments under the conditions listed in § 550.407. | **C** | span | contradicted | **NLI** |
| adv-1 | 317.402 | The standard may emphasize agency-related experience even where doing so precludes well-qualified... | **C** | span | contradicted | **NLI** |
| adv-2 | 317.402 | The agency must publish the qualification standard in the vacancy announcement. | **N** | span | unsupported | **NLI** |
| adv-4 | 317.703 | OPM will direct reinstatement within 30 days of receiving the application. | **C** | span | contradicted | **NLI** |
| adv-7 | 317.703 | An agency may take up to 30 business days from the date of the order to comply. | **C** | span | contradicted | **NLI** |
| adv-9 | 330.213 | An agency may vary the selection method it uses for an individual vacancy. | **C** | span | contradicted | **NLI** |
| adv-10 | 330.213 | An agency must keep a registrant on the RPL for one year. | **N** | span | unsupported | **NLI** |
| adv-12 | 337.204 | Notification to OPM need not describe the evidence the determination relied on. | **C** | span | contradicted | **NLI** |
| adv-13 | 337.204 | OPM may revoke a direct hire authority it has approved. | **N** | span | unsupported | **NLI** |
| adv-14 | 351.402 | A competitive area in effect for fewer than 90 days before the RIF takes effect must be described... | **E** | span | uncertain | **align** |
| adv-15 | 351.402 | A competitive area in effect for fewer than 90 days before the RIF needs no OPM approval. | **C** | span | contradicted | **NLI** |
| adv-17 | 353.209 | An employee reemployed under this subpart may be discharged at the agency's discretion. | **C** | span | contradicted | **NLI** |
| adv-20 | 532.703 | An application involving a downgrading may be filed at any time and still carry entitlement to re... | **C** | span | uncertain | **NLI** |
| adv-22 | 534.404 | The senior executive must respond within 7 days of the notice. | **C** | span | uncertain | **NLI** |
| adv-23 | 534.404 | The response must be made in writing. | **C** | span | contradicted | **NLI** |
| adv-25 | 534.604 | An agency may set an administrative appeals judge's initial rate of basic pay at any rate in the ... | **C** | span | contradicted | **NLI** |
| adv-27 | 550.905 | An employee receiving availability pay under 550.181 may also be paid a hazardous duty differenti... | **C** | span | contradicted | **NLI** |
| adv-28 | 550.905 | The hazardous duty differential is 25 percent of the employee's rate of basic pay. | **N** | span | unsupported | **NLI** |
| adv-30 | 575.507 | The amount is 25 percent of the employee's annual rate of basic pay at the end of the service per... | **C** | span | contradicted | **NLI** |
| adv-32 | 591.239 | A post differential counts as part of basic pay when retirement entitlement is computed. | **C** | span | contradicted | **NLI** |
| adv-34 | 630.1703 | A seasonal employee may use paid parental leave during the off-season period designated by the ag... | **C** | span | contradicted | **NLI** |
| adv-35 | 630.1703 | Paid parental leave must be used within 12 months of the birth or placement. | **N** | span | unsupported | **NLI** |
| adv-36 | 630.502 | An employee who returns to Federal employment after December 2, 1994 is generally entitled to hav... | **E** | span | uncertain | **align** |
| adv-37 | 630.502 | Sick leave is recredited only when the break in service was shorter than three years. | **C** | span | unsupported | **NLI** |
| adv-39 | 890.1016 | OPM offsets any restitution the provider has paid against the financial loss it finds. | **C** | span | contradicted | **NLI** |
| adv-41 | 890.1040 | The presiding official must issue the written report within 30 days of the hearing opening. | **C** | span | contradicted | **NLI** |
| adv-42 | 890.1040 | The suspending official may reject the presiding official's findings. | **N** | span | unsupported | **NLI** |
| adv-44 | 530.304 | OPM computes the special rate supplement by a fixed dollar amount or fixed percentage and no alte... | **C** | span | contradicted | **NLI** |
| adv-46 | 351.302 | An employee transferred solely for liquidation competes for other positions in the gaining compet... | **C** | span | contradicted | **NLI** |
| adv-48 | 575.110 | The service period may terminate on any day the agency chooses. | **C** | span | contradicted | **NLI** |
| adv-52 | 630.906 | A leave recipient must exhaust his or her own annual leave before using donated leave. | **N** | span | contradicted | **NLI** |

NLI right 35, align right 6, neither 0

---

## Reproduced — 2026-08-30

The appendix above is now `benchmarks/entailment.yaml`, scored by
`warrant.eval.entailment`. Evidence is written `section#anchor` with the date the premise was
read at and resolved against `data/warrant.sqlite3` at load time; **an anchor that does not
resolve raises rather than being skipped**, because a pair that silently stops being scored
shrinks the set while the reported `n` keeps counting what the file says. All 182 pairs
resolve, over 91 sections, with the class balance this document tabulates.

Re-run against the same checkpoint, from the store rather than from the table:

| number | published here | re-run |
|---|---:|---:|
| generator micro | 86.8% (112/129) | **86.8%** (112/129) |
| generator 95% CI | 80.8–92.5 | **80.8–92.5** |
| generator macro | 60.1% | **60.1%** |
| generator per class | 99/102 · 12/24 · 1/3 | **99/102 · 12/24 · 1/3** |
| adversarial micro | 88.7% (47/53) | **88.7%** (47/53) |
| adversarial 95% CI | 78.4–96.6 | **78.4–96.6** |
| adversarial macro | 89.0% | **89.0%** |
| adversarial per class | 20/21 · 9/10 · 18/22 | **20/21 · 9/10 · 18/22** |
| pooled confusion | 119/4/0 · 6/21/7 · 4/2/19 | **identical** |
| NLI − align, generator | +2.3, CI −2.5–7.6, 7/4, p 0.55 | **+2.3, −2.5–7.6, 7/4, p 0.55** |
| NLI − align, adversarial | +49.1, CI 40.0–56.7, 28/2, p 8.7e-07 | **+49.1, 40.0–56.7, 28/2, p 8.7e-07** |
| disagreements adjudicated | 41 — NLI 35, align 6, neither 0 | **41 — 35 / 6 / 0** |
| leave-one-section-out temperatures | 1.66–1.74 | **1.657–1.737** |

Everything section 1 and section 2 claim reproduces exactly. Sections 3 to 5 — calibration,
latency, the second checkpoint — were not re-run; the benchmark scores accuracy and the two
signals, and nothing else here has been re-measured.

Three things the promotion turned up, none of which moved a number:

- **The appendix could not have rebuilt the set.** 66 of its 182 claim cells are truncated at
  the column width. The file was recovered from the run's own inputs, not retyped from the
  table — which is the concrete form of the problem: a printed table is a record of a
  measurement, not a re-runnable one.
- **Premises are now read from the store, and six of them were not verbatim after all.**
  Six of the twenty adversarial chunks had been transcribed with `§ ` dropped, a trailing
  em dash lost, or an em dash typed as a comma — 16 of the 53 adversarial pairs. The claim
  above that premises are verbatim chunk text was true of the generator stratum and
  approximately true of the adversarial one. Scoring the store's text instead changes no
  argmax and no verdict, which is why every figure in the table above still lands.
- **Section 2 is reported at a temperature fitted on the gold labels.** The verdicts there
  use the leave-one-section-out fit, not the shipped `CALIBRATION_TEMPERATURE = 1.72`. At the
  shipped constant the generator delta is **+3.1 (CI −1.9–8.6, 8/4, p = 0.39)** against +2.3,
  and the adversarial delta is unchanged at +49.1. The conclusion does not move — not
  measurable on generator output either way — but the number a reader quotes depends on which
  temperature was used, and the shipped one is what the serving path will apply.

`pytest -m neural tests/test_entailment_bench.py` asserts the section 1 and section 2 figures
directly, so a checkpoint, corpus or label change fails a test rather than making this
document quietly wrong.
