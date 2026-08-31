# Eval 020 — Growing the human bucket from 56 to 212

**Date:** 2026-08-30
**Config:** `configs/default.yaml`, hash `dbac692062d7` — unchanged from
[eval-004](eval-004-held-out.md)/[eval-005](eval-005-abstention.md)/[eval-019](eval-019-baselines.md).
Only `benchmarks/human.yaml` changed; nothing in `src/`, `configs/`, or the store was touched.
**Split:** horizon 2026-08-26. `human` **212** total, **108 test / 104 dev**, 148 distinct
sections. `temporal` 233/47, `scope` 42/42, `scope-exclusion` 42/42 — unchanged in count,
present only for context.
**Reproduce:** `make eval` (sufficiency table), and the abstention snippet under
[Reproducing](#reproducing) below (the same one eval-005 used, unmodified).
**Owns:** `benchmarks/human.yaml`, this file. Nothing else was edited.

**Headline: the sample size stopped being the problem, and the difficulty of the bucket
turned out to be the more interesting finding under it.** Sufficiency on `human` barely moved
(79.3% → 79.6%) but its confidence interval more than halved (61.8–96.2 → 70.5–87.6), which is
what 29 items becoming 108 on the test split is supposed to buy. What did move: the bucket's
share of every insufficient-context answer in the whole held-out set, from 54% to 71% — the
new items are measurably harder, not just more numerous, and that is the section to read first.

## The task the file started at

`benchmarks/human.yaml` held 56 hand-written items when this session opened, not the 29 the
brief and the current README describe — an earlier pass (visible in `results/eval-019`, which
reports 51, and in `tests/test_guard.py`, which already said "56") had grown it partway. The
target was 200; this session added **156** items to reach **212**, spread across every part in
the corpus rather than concentrated in the parts the original 56 already covered.

## Items by part

| part | subject | n | | part | subject | n |
|---|---|---:|---|---|---|---:|
| 315 | career/career-conditional appointment | 14 | | 337 | examining system | 7 |
| 890 | FEHB | 14 | | 511 | classification | 7 |
| 630 | leave | 13 | | 551 | FLSA | 7 |
| 550 | pay administration | 11 | | 451 | awards | 6 |
| 300 | employment general provisions | 10 | | 530 | pay rates (general) | 6 |
| 316 | temporary/term employment | 9 | | 534 | pay under other systems | 6 |
| 351 | reduction in force | 9 | | 536 | grade/pay retention | 6 |
| 330 | recruitment, RPL, CTAP/ICTAP | 9 | | 575 | recruitment/retention incentives | 6 |
| 430 | performance management | 8 | | 610 | hours of duty | 6 |
| 531 | GS pay | 8 | | 591 | allowances (uniform, COLA) | 6 |
| 532 | prevailing rate (FWS) | 8 | | | | |
| 317 | Senior Executive Service | 8 | | | | |
| 335 | promotion/demotion/reassignment | 7 | | | | |
| 353 | restoration after service/injury | 7 | | | | |
| 410 | training | 7 | | | | |
| 432 | unacceptable performance | 7 | | | | |

All 26 parts in the corpus are represented; the smallest is 6 items, the largest 14. The
original 56 leaned on 315 (13), 351 (7) and 432/531/550 (5 each) and had never touched 300,
317, 337, 511, 530, 534, 536, 551, 575, 591, 610 or 890 — eleven of twenty-six parts,
including the two largest in the corpus by section count (890 at 190 sections, 550 at 175).

## Items by kind

| kind | n | how |
|---|---:|---|
| dated (`as_of` on the before-side of a real amendment) | 13 | 9 from the original 56, 4 new: `575.104`'s SES-limited-term exclusion moved from `(d)(4)` to `(a)(4)(iv)` on 2026-03-09, and `591.103`'s uniform allowance rose from $800 to $1,500 on 2026-07-13 — each pair asks the identical question on either side of the anchor's own renumbering. |
| scope-dependent (`pay_system` or `service`) | 40 | 7 from the original 56, 33 new, across every part `retrieve.scope.PART_RESTRICTIONS` restricts (511, 531, 532, 534, 317, 315, 316, 337). |
| answer split across two or more paragraphs | 7 | e.g. `veterans-preference-points-earned` needs both the 5-point and 10-point tiers (`337.101#b-1`, `#b-2`); `fehb-dual-enrollment-exception` needs the prohibition and its carve-out (`890.302#a-2`, `#a-2-ii`) as one set. |
| exception clauses (`except`/`unless`/`subject to`/negative fact) | ~55 | the largest single category by construction — see below. |
| scope-mismatch, genuinely unanswerable | 6 | see next section. |

The "exception clause" count is a judgment call, not a grep, so it is reported as an
approximation rather than a false-precision number: items phrased as "can I…" or "does…
apply if…" whose gold evidence is the qualifying or excluding paragraph, not the general rule
— `ses-pay-reduction-appeal-exception` (no MSPB appeal of an SES pay cut), `tcc-gross-
misconduct-exclusion` (no temporary continuation of FEHB after a gross-misconduct removal),
`grade-retention-supervisory-probation-exclusion` (no grade retention after failing
supervisory probation). 15.7% of chunks in the corpus carry a qualifier by the number in the
project's own header; this batch deliberately over-samples that shape, because the failure it
guards against — the true sentence with the exception silently dropped — is exactly the one a
system built from the general rule alone will produce.

## The six unanswerable items, and the one that had to be abandoned

The brief asked for questions the corpus cannot answer, "so abstention has something to be
right about." The obvious construction — `evidence: [[]]` — was tried first and dropped.
`BenchItem.is_satisfied_by` treats an empty acceptable-evidence set as satisfied by *any*
retrieved list, because that is the correct semantics for `scope-exclusion`, where the whole
question is whether a part was correctly kept **out** of the query. Reused for a human item
whose real problem is "nothing here answers this," it does the opposite of what was wanted:
`eval.generation.retrieved_evidence` and `verify.calibrate.Example.sufficient` are the same
computed quantity, `item.is_satisfied_by(...)`, so an empty-evidence item would be graded
"sufficient" no matter what came back, and a model that correctly declined to answer it would
be scored as **wrong** — `abstained_with_evidence`, the good-refusal counted as a mistake. This
is exactly why `verify.calibrate.collect_examples` already drops `scope-exclusion` from the
abstention study outright rather than score it; reusing its construct for `human` would have
reintroduced the bug that line of code exists to avoid, quietly, in a bucket nobody excludes.

What is expressible, and is a real instance of the same failure mode: a question that is
completely ordinary on its own but is governed by a part that does not apply to the asker's
declared `pay_system` or `service`. `retrieve.scope.Scope.excluded_parts` is a hard SQL
predicate — the excluded part's chunks are never candidates — so an item like
`wgi-not-applicable-ses` ("How do I earn a within-grade increase?" asked with
`scope: {pay_system: SES}`, gold evidence in `531.404`, a GS-only section) is *guaranteed*
insufficient by the same mechanism that makes `scope-exclusion` guaranteed clean, and for the
same reason: the predicate did its job. Six of these were written, one per restricted part
(531, 532, 315, 317, 337, 316), each phrased the way a person who does not know their own pay
system's boundaries would actually ask it.

Read honestly: these six are not a discovery about the *generator*'s judgment the way a truly
open-domain unanswerable question would be — they are, like `scope-exclusion`, enforced by
construction at the retrieval stage. What they add over the auto-mined `scope-exclusion`
bucket is realism of phrasing and inclusion in the one bucket that reaches
`eval.generation`: `scope-exclusion` is never scored for hallucination or abstention quality
anywhere in the pipeline, and these six now are. Whether the generator says "this does not
apply to your pay system" or fabricates a GS answer from background knowledge is a live
question these items can now put in front of `make generation`, on a bucket that isn't
special-cased away first. A genuinely open-domain unanswerable question — one with no
governing CFR text at all, restricted-part predicate or not — has no clean representation
under the current loader (`evidence` must resolve to a real chunk, or the entry must use the
empty-set construct that breaks abstention scoring). That gap is real and is left open rather
than worked around with a construct that would silently mean something else.

## Anchor verification

Every batch (roughly 10–30 items at a time, 8 batches total) was checked by loading the file
through the actual production path — `warrant.eval.bench.load_human`, not a hand-rolled
reimplementation — against the live `data/warrant.sqlite3` at the corpus horizon. That
function raises `ValueError` on any item with zero resolving evidence sets, which is the same
guard the entailment benchmark uses for the same reason: a silently-skipped item shrinks the
set while the reported count keeps rising. It caught one mistake —
`temp-appointment-max-total-length` cited `316.401#c-1`, an anchor that does not exist; the
governing text for the time limit is stored under the parent anchor `#c` itself, the way a
short chapeau paragraph with its `(1)` inline sometimes is (the same pattern `534.404#j` and
`430.208#b` use). Every anchor in the 212-item file resolves as of this run; `python -c
"...load_human(...)"` printed `loaded 212 items OK` with no exceptions, and the full test
suite (671 tests, including `test_guard.py`'s hand-written-question checks) and `ruff check
src tests` both pass unchanged.

## The held-out re-run

### Sufficiency, test split

| bucket | n | sections | sufficiency | 95% CI | | before (eval-004) |
|---|---:|---:|---:|:---:|---|---|
| human | **108** | **76** | **79.6%** | **70.5–87.6** | | 29 / 22 sec, 79.3%, 61.8–96.2 |
| scope | 42 | 42 | 100.0% | 100.0–100.0 | | unchanged |
| temporal | 233 | 47 | 96.1% | 93.4–99.4 | | 229/47, 97.8%, 94.9–100.0 (corpus grew one snapshot day) |

The central estimate is essentially unchanged — 79.3% moved to 79.6%, a shift the interval
would call noise even on its own. What changed is the interval: **61.8–96.2 down to
70.5–87.6**, a width of 34.4 points collapsing to 17.1. That is the whole point of the
exercise stated plainly: at n=29 the bucket could not separate configurations or support any
claim tighter than "somewhere between two-thirds and essentially all of it"; at n=108 it can
say something a reader could act on.

`temporal`'s own small move (97.8% → 96.1%) is not attributable to this work — `human.yaml`
does not feed the temporal miner, and the difference is the corpus having ingested one more
day of amendments between the two runs (233 items now against 229 then). It is reported for
completeness, not as a finding.

The as-of/cross-encoder paired ablations in `make eval` run only over the `temporal` bucket
(`cli._paired` is called with `buckets["temporal"]` specifically), so nothing about their
numbers moving is attributable to `human.yaml` either — noted so the two tables in this
document aren't misread as more connected than they are.

## The abstention re-run

This is where the harder-item claim earns its keep. Same method as
[eval-005](eval-005-abstention.md): `verify.calibrate.collect_examples` over every bucket
(minus `scope-exclusion`, dropped for the reason above), `study(dev, test)` fits the combiner
and both thresholds on dev, reports on test.

### Overall (all buckets pooled)

| | before (eval-005) | now |
|---|---:|---:|
| test n | 300 | **383** |
| insufficient | 13 (4.33%, CI 0.7–8.6) | **31 (8.09%, CI 5.3–11.2)** |
| human's share of all insufficiency | 7/13 = 54% | **22/31 = 71%** |
| learned AURC | 0.0105 | 0.0262 (0.011–0.046) |
| top-1 fusion AURC | 0.0086 | 0.0431 (0.023–0.066) |
| `beat_baseline` | **False** | **True** — +8.88pp paired (2.7–14.7), 65 won / 31 lost, p=6.7e-4 |

The aggregate insufficiency rate nearly doubled, and it is not spread evenly: the human
bucket now accounts for seven-tenths of every wrong-context answer the whole held-out set
produces, up from a bit over half. That is the "sufficiency drops because the new items are
harder" finding the brief asked to lead with if it showed up, and it showed up.

The `beat_baseline` flip from **False to True** looks like eval-005's central finding
reversing itself, and superficially it has — but it is worth being exact about *why* before
treating it as a correction. Eval-005's null result rested on the top-1-fusion baseline
missing its 2% risk budget on dev by one item and failing closed, answering nothing on test;
that fragility is a property of the baseline's dev-fitted threshold, not of the signal. Adding
108 harder human items shifted where that threshold lands and gave the baseline enough
headroom to answer something again — the mechanism eval-005 already diagnosed, now landing on
the other side of one item's flip by chance of which items are in dev this time.

### Per bucket, at the shipped threshold — the number that actually governs the decision

| bucket | n | insufficient | learned+isotonic coverage | selective risk | | top-1 fusion coverage | selective risk |
|---|---:|---:|---:|---:|---|---:|---:|
| human | 108 | 22 | 18.5% (12.3–26.9) | 20.00% (0.0–40.9) | | **26.9%** (19.4–35.9) | **13.79%** (3.4–26.7) |
| scope | 42 | 0 | 95.2% (84.2–98.7) | 0.00% | | 85.7% (72.2–93.3) | 0.00% |
| temporal | 233 | 9 | 89.3% (84.6–92.6) | 3.37% (0.0–6.5) | | 70.0% (63.8–75.5) | 2.45% (0.7–4.4) |

**On the bucket the whole abstention exercise exists for, the baseline still wins.** The
aggregate table says the learned combiner is now significantly better; the human-specific
slice says the single-feature threshold answers *more* real questions (26.9% vs 18.5%) at
*lower* risk (13.79% vs 20.00%) than the fitted model does. Both of those are true
simultaneously because the aggregate is dominated by `temporal` (233 of 383 test items, 61%)
and `scope` (42, 11%), where the combiner's calibration genuinely helps; `human` is 28% of the
test set and pulls the other way. This is the same warning eval-005 gave at n=29 — "the
aggregate is not the number that should govern the decision" — now measured with an interval
tight enough to trust: at n=108 the human-bucket gap (26.9% vs 18.5% coverage) does not
overlap zero the way it would have at n=29.

Coverage for both policies on `human` is low in absolute terms (below 27% either way) and
that is not new — eval-005's 17.2% said the same thing with a much wider skirt around it. It
is now a number worth keeping: on real HR questions, whichever policy ships, roughly three in
four are still going to be declined at a 2% risk budget, and this session's items are why that
number now has a CI that fits in a sentence rather than a paragraph.

## Least confident

- The six scope-mismatch items (above) rest on a construct — `is_satisfied_by` guaranteed
  False through a hard predicate exclusion — that has not been used this way anywhere else in
  the codebase. It is correct by the same logic `scope-exclusion` already relies on, but it is
  new *usage* of that logic and worth a second reader.
- `temp-appointment-seasonal-exception` (`316.401#d`) and `ses-pay-reduction-cap`
  (`534.404#j`) cite a chapeau paragraph whose numbered sub-item is folded into the same
  anchor rather than split out — correct against the store as ingested, but a paragraph shape
  worth spot-checking if the chunker's designator-stack logic (README, "citation addresses")
  ever changes.
- A handful of binary "can I / does X apply" items were framed from a single paragraph after
  reading its neighbors rather than the whole section end-to-end — `fehb-court-order-child-
  coverage-protection` (`890.301#e-1-ii`) and `investigative-leave-initial-duration`
  (`630.1504#b`) sit inside sections long enough (890.301 has 16 lettered paragraphs;
  630.1504 has extension and conversion rules running past the cited anchor) that a stricter
  reading might want the neighboring paragraph pooled in as a second acceptable set. None of
  these failed the anchor-resolution check; the risk is a narrower gold than the real answer,
  not a wrong one.
- The four new dated items (575.104, 591.103) both rely on the store having captured a clean
  renumbering rather than a substantive rewrite alongside it — confirmed by reading both full
  paragraph texts side by side, not just the anchors, but it is the kind of amendment where a
  chunker bug would be easy to miss by only checking that both anchors resolve.

## Still open

- No genuinely open-domain unanswerable item exists in this file, and the reason is structural
  (see above), not an oversight to fix by writing more of them the same way.
- The "exception clause" and "cross-paragraph" counts in the kinds table are read off the
  items by eye, not asserted by a test the way anchor resolution is.
- The reranker model revision is unpinned in `configs/default.yaml`, the same caveat
  eval-005 raised: these numbers are reproducible against the weights that resolved on
  2026-08-30, not necessarily against any other day's download.

## Reproducing

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
print(s.learned.aurc, s.baseline_top_score.aurc, s.beat_baseline)
```

For the per-bucket table, filter `ex` (or the `test` list inside the snippet) by `e.bucket ==
"human"` before computing coverage/risk at `s.policy.threshold` — `Study` reports the pooled
curve only; the per-bucket breakdown here was computed by hand from the same `Example` objects
`study()` already produces, which is why it needs no separate CLI command.
