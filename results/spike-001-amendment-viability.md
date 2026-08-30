# Spike 001 — is the eCFR amendment history clean enough to be a benchmark?

**Date:** 2026-08-30
**Question:** the temporal benchmark is mined from real amendments between point-in-time
snapshots. If those diffs are dominated by restructuring and editorial churn, the benchmark
has no usable ground truth and the whole project needs a different centrepiece. Answer this
before writing any retrieval code.

## Acceptance rule, pre-registered

Written down before any output was inspected:

| | Rule |
|---|---|
| **Primary (go / no-go)** | ≥ 60% of changed sections are *clean substantive*: stable identifier, alignable before/after text, localized semantic change |
| **Secondary (volume)** | ≥ 150 clean substantive changes across the Title 5 HR parts |
| 40–60% | viable only with a documented filter, and the README must state what it discards |
| < 40% | abandon amendment-mining as the centrepiece; rebuild the benchmark around chunk-derived and human-written questions |

## Method

26 Title 5 chapter I parts (parts 331 and 631 have no point-in-time history and were dropped
at survey time). Every distinct snapshot date from 2017 forward, fetched from
`/api/versioner/v1/full/{date}/title-5.xml?part={p}` — 222 snapshots, 50 MB of XML.
Consecutive snapshots aligned by section identifier and classified by
`warrant.corpus.diff`.

## Result

```
  part | subst | whole | edit | appar | renum | added | remvd
   300 |     2 |     0 |    0 |     0 |     0 |     0 |     0
   315 |    15 |     3 |    0 |     9 |     0 |     3 |     6
   316 |    14 |     1 |    1 |    12 |     0 |     0 |     1
   317 |     0 |     1 |    0 |     0 |     0 |     0 |     0
   330 |    10 |     0 |    0 |    18 |     0 |     1 |     0
   335 |     2 |     1 |    0 |     1 |     0 |     1 |     0
   337 |     3 |     1 |    0 |     4 |     0 |     0 |     0
   351 |     8 |     0 |    0 |    21 |     0 |     0 |     0
   353 |     1 |     0 |    0 |     0 |     0 |     0 |     0
   410 |     1 |     0 |    0 |     0 |     0 |     0 |     0
   430 |     9 |     0 |    0 |     6 |     0 |     0 |     0
   432 |     8 |     0 |    2 |     7 |     0 |     1 |     1
   451 |     0 |     0 |    2 |     1 |     0 |     0 |     0
   531 |     7 |     0 |    1 |     8 |     0 |     0 |     0
   532 |    16 |     0 |    5 |    20 |     0 |     0 |     1
   534 |     3 |     0 |    0 |     0 |     0 |     0 |     0
   550 |     4 |     2 |    0 |     4 |     0 |     8 |     0
   575 |    14 |     0 |    1 |     3 |     0 |     0 |     0
   591 |     3 |     1 |    0 |     0 |     0 |     0 |     0
   630 |     8 |     3 |    1 |     6 |     0 |    20 |     1
   890 |    30 |     2 |    2 |    20 |     0 |    37 |     0
 TOTAL |   158 |    15 |   15 |   140 |     0 |    71 |    10
```

| Criterion | Threshold | Measured | |
|---|---|---|---|
| Clean substantive share | ≥ 60% | **84.0%** (158 of 188) | pass |
| Clean substantive volume | ≥ 150 | **158** | pass |

**Decision: proceed.** Amendment-mined temporal questions are the benchmark centrepiece.

## What the spike changed

**Editorial apparatus was contaminating 43% of all detected change.** 140 section-pairs
differed only in material that is not regulatory text — overwhelmingly eCFR pending-amendment
pointers:

```xml
<XREF ID="20200810">Link to an amendment published at 85 FR 48089, Aug. 10, 2020.</XREF>
```

These appear while an amendment is pending and vanish when it publishes. The first run of this
spike did not strip them and reported 77.8% clean substantive; six of its eight sampled
"amendments" were nothing but these pointers appearing or disappearing. Stripping them moves
140 pairs out of the changed set entirely — without it, roughly 43% of the temporal benchmark
would have been questions about publication schedules rather than about the law.

That is why `warrant.corpus.apparatus` is a tested component with fixtures rather than a
`strip()` inside the parser, and why `ingestion` is expected to be a non-zero row of the
failure budget.

**Two API facts, both found by running it rather than reading the documentation:**

- `/versions` returns one row per *section* version, not per snapshot. Part 630 returns 226
  rows that collapse to 8 distinct dates. Counting rows overstates diffable history by more
  than an order of magnitude, and an earlier draft of this project did exactly that.
- Every part advertises a first version date of 2016-12-27 and `/full/` returns 404 for it.
  The usable point-in-time window starts in 2017.

**Renumbering is a non-issue here.** Zero renumbering candidates across 26 parts. The
detection stays in the classifier because it is cheap and its absence is itself a finding, but
it does not need engineering effort.

## Sample of what the benchmark will be built from

Real, localized, machine-checkable amendments:

- **§ 300.301, 2024-06-11** — "Schedule A, Schedule B, or a Veterans Recruitment Appointment"
  becomes "Schedule A, Schedule B, Schedule D, or a Veterans Recruitment Appointment". A
  three-token change with a clean before/after.
- **§ 315.201, 2021-09-17** — a new clause (xvii) adds time-limited post-secondary student
  appointments to the list of service-computation dates.
- **§ 315.612, 2021-10-21** — noncompetitive appointment eligibility for military spouses is
  broadened, and the permanent-change-of-station requirement is removed.
- **§ 315.803** — the strongest item found. A supervisor-notification requirement for ending
  probationary periods is **added on 2020-11-16 and removed on 2022-12-12**. A question about
  it has three different correct answers depending on the date asked, and a system with no
  as-of filtering cannot get all three right.

## Caveats on the number

- 84.0% is the share of *changed* sections that are clean, not the share of the corpus. It
  says the mining is viable; it says nothing about coverage.
- The classifier's thresholds (`wholesale_threshold` 0.50, `min_changed_tokens` 3) are
  judgement calls recorded in `configs/default.yaml`. The stricter `min_changed_tokens`
  is what separates this 84.0% from the scratch script's 90.4%: twelve sub-three-token changes
  are counted as editorial here and were counted as substantive there.
- Snapshot granularity bounds temporal resolution. Benchmark questions are posed away from
  snapshot boundaries for this reason.
- 158 clean changes is enough to build the temporal bucket, not enough to be generous with it.
  If the bucket needs to grow, the next source is more of Title 5 rather than more parts of
  chapter I.
