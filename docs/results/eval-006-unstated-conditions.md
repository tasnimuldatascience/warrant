# Eval 006 — the true sentence with the exception dropped

**Date:** 2026-08-30
**Modules:** `verify/qualifier.py` (new), `verify/xref.py` (rewritten in part)
**Reproduce:** `make build && make index`, then the scripts described under
[How each number was produced](#how-each-number-was-produced). No network, no paid API, no
model in the shipped path.

Hallucination on the held-out human items is 1.5% and citation precision 98.5%
([eval-004](../../results/eval-004-held-out.md)). Neither number can see the failure this
document is about:

> **Restored annual leave must be scheduled and used within 2 years.**

That is a faithful reading of §630.306(a). It is also wrong, because (a) opens *"Except as
provided in paragraph (b) of this section"*. Span alignment passes it — the cited text does
say what the claim says. Entailment passes it — the premise entails the claim. Citation
precision passes it — the citation is real. The claim is simply not the whole rule.

Three numbers, in order of how much they should change what gets built next.

---

## 1. Retrieval leaves the conditional chain broken more often than not

For every benchmark item, retrieval was run at its own `as_of` and scope, and the evidence set
it produced was checked for references whose target the corpus holds and the evidence set does
not. 594 items across all four buckets.

| `final_k` | evidence sets with ≥1 **missing** reference | mean missing per set | missing per cited chunk | sets carrying ≥1 qualifier |
|---:|---:|---:|---:|---:|
| 4 | **51.7%** | 1.32 | 0.330 | 70.2% |
| 8 | **63.6%** | 2.39 | 0.298 | 87.0% |
| 16 (shipped) | **77.3%** | 4.14 | 0.259 | 95.3% |

**Widening retrieval makes this worse, not better.** Going from `final_k: 4` to the shipped
16 quadruples the number of unsatisfied references per answer. Each chunk admitted brings in
more references than the chunks already present satisfy — the per-chunk rate falls from 0.330
to 0.259, and the per-answer rate rises from 1.32 to 4.14. There is no `final_k` in the
plausible range at which the chain closes; the fix is not a larger `k`, it is reference-aware
expansion, and that is now a measurement rather than an intuition.

At the shipped configuration, per bucket:

| bucket | n | with a missing reference | with a qualifier |
|---|---:|---:|---:|
| human | 56 | 87.5% | 100.0% |
| scope | 95 | 82.1% | 95.8% |
| scope-exclusion | 95 | 91.6% | 95.8% |
| temporal | 348 | 70.4% | 94.3% |

The three reference outcomes are kept apart, because collapsing them would let the corpus
boundary inflate a number that is supposed to be about retrieval. At `final_k: 16`, over
9,504 cited chunks:

| status | count | meaning |
|---|---:|---|
| `missing` | 2,458 | the target is in the corpus and was not retrieved — **a retrieval gap** |
| `outside` | 2,047 | a U.S.C. section, another CFR title, a Public Law — a corpus-scope limit |
| `unscoped` | 2,962 | "this subpart" names no single chunk — not actionable |

Only the first column is a defect of this system. The second is the honest cost of a corpus
that stops at 5 CFR chapter I: 10.3% of in-force chunks cite the U.S.C., and no amount of
retrieval tuning will satisfy those.

The most-missed targets are not random. §630.1203(a) — the FMLA entitlement paragraph that
half of subpart L hangs off — is referenced by 46 retrieved chunks that do not have it.

---

## 2. Qualifier detection: 90.6% precision, and a recall gap that is larger than it

`qualifiers()` returns `Qualifier(kind, span, text, refers_to)` objects over one chunk's text.
Character spans are correct here and only here: citations in this repository are evidence ids
throughout because the generator cannot count characters (ARCHITECTURE.md §5), but nothing in
this module asks a model for an offset — the spans come from a regex over the chunk's own
text, which is arithmetic.

Over the 9,961 in-force chunks:

| kind | chunks | % | occurrences |
|---|---:|---:|---:|
| `chapeau` | 979 | 9.8% | 979 |
| `except` | 504 | 5.1% | 532 |
| `prohibition` | 484 | 4.9% | 519 |
| `subject_to` | 389 | 3.9% | 415 |
| `unless` | 279 | 2.8% | 288 |
| `only_if` | 117 | 1.2% | 119 |
| `other_than` | 113 | 1.1% | 115 |
| `proviso` | 69 | 0.7% | 71 |
| `bound` | 69 | 0.7% | 70 |
| `notwithstanding` | 56 | 0.6% | 56 |
| **any** | **2,562** | **25.7%** | 3,164 |

### The labelled sample

104 detected instances were drawn with a fixed seed and hand-labelled: 8 per kind, plus 24
extra for `subject_to`, held out and drawn after the first 8 had already been read. A positive
is *a clause that bounds what the cited rule requires* — an answer quoting the rule without it
would be incomplete or wrong. Every one is listed in
[the audit table](#the-labelled-instances) so a reader can disagree with a specific call
rather than with the total.

| kind | correct / labelled | precision |
|---|---:|---:|
| `chapeau` | 8/8 | 100% |
| `except` | 8/8 | 100% |
| `unless` | 8/8 | 100% |
| `notwithstanding` | 8/8 | 100% |
| `only_if` | 8/8 | 100% |
| `bound` | 8/8 | 100% |
| `prohibition` | 7/8 | 88% |
| `proviso` | 7/8 | 88% |
| `other_than` | 6/8 | 75% |
| **`subject_to`** | **17/32** | **53%** (95% CI 36–69) |

Weighted by how often each kind actually occurs — the sample is stratified, so an unweighted
mean over kinds would be a different quantity from precision on the corpus:

| | precision | 95% CI | occurrences covered |
|---|---:|:---:|---:|
| all kinds | **90.6%** | 85.6–94.6 | 3,164 |
| conditional kinds only (what the check consumes) | **90.4%** | 85.3–94.5 | 3,094 |
| lexical kinds only (excluding the structural `chapeau`) | **86.0%** | 78.4–91.9 | 2,115 |

Intervals are a stratified bootstrap, resampling within each kind and reweighting.

### `subject_to` is the failure, and the discriminator is not lexical

Half of all `subject to` matches are not conditions. The distinguishing feature is
**grammatical**: what the phrase attaches to.

```
condition      "OPM shall debar him for 3 years, subject to adjustment on the basis of
                the aggravating and mitigating circumstances listed in § 890.1016."
condition      "Subject to the provisions of paragraph (d)(iii) of this section, an
                enrollee ... may cancel his or her enrollment at any time."

not            "Covered positions are subject to a performance appraisal system
                established under 5 U.S.C. chapter 43."             (an operative statement)
not            "The head of an agency having employees subject to this subpart is
                responsible for ..."                                (a restrictive modifier)
not            "the employee will not be subject to time limits on usage of any
                restored leave"                                     (the rule itself)
```

The false positives all attach "subject to" to a *person or position* and describe its status;
the true positives attach it to a *rule or an action* and condition it. That is a dependency
relation, not a lexical one. Three lexical guards were written and run against the 32 labels:

| guard | kept | precision | true conditions dropped |
|---|---:|---:|---:|
| none (shipped) | 32/32 | 17/32 = **53%** | 0 |
| require a normative object (§-reference, "requirement", "provisions", "limitation") | 28/32 | 16/28 = 57% | 1 |
| reject when preceded by a person noun | 16/32 | 11/16 = 69% | 6 |
| reject after a copula (`is`/`are`/`becomes`/`remains`) | 14/32 | 11/14 = 79% | 6 |

The normative-object guard does nothing: the false positives name norms too. The other two
buy precision by discarding a third of the real conditions — and both were written *after*
reading these 32 instances, so their numbers are fitted to the set that measures them.

**No guard is shipped.** A 53%-precision kind is reported at 53% rather than tuned against the
32 instances that would then no longer be evidence of anything. This is the place where a
model would plainly earn its keep, and §3 measures whether the one already in this repository
does — on a different question.

### Two false positives the brief predicted, and what was actually there

*"'unless otherwise noted' in a table caption is not a legal condition."* In this corpus it
never occurs. All 13 chunks containing "unless otherwise" name an authority that can displace
the rule — *"unless otherwise approved by OPM"*, *"unless otherwise provided by the Office"* —
and all 13 are kept, correctly.

*"'may not exceed' as a numeric bound already stated in the answer is not an unstated
condition."* This one is real: 57 of the 391 "may not" chunks are "may not exceed". They are
detected as a separate kind, `bound`, whose `conditional` property is `False`, so the
unstated-condition check never sees them. Hand-labelling found one more shape of the same
thing — §591.238(b) pays "so much of the post differential as **will not cause** the combined
total **to exceed** 25 percent", a ceiling written as a prohibition — and `_BOUND` was widened
to catch it. That fix came from this sample, so the `prohibition` row above is reported at the
7/8 it scored *before* the fix, not the 8/8 it scores now.

### The hardest calls

Three, all argued in the audit table and all defensible the other way:

- **§551.601(a)** *"applies to all employment subject to its child labor provisions"* —
  labelled a condition, because it limits which employment the 16-year minimum reaches.
  §550.401(d) *"employees subject to this subpart"* was labelled not a condition, because the
  subpart's scope conditioning the subpart is tautological. The line between them is whether
  the qualifier is informative, which is a judgement.
- **§451.302(b)(1)** *"a position that is subject to OPM position allocations under part
  319"* — an eligibility criterion written as a restrictive relative clause. Labelled not a
  condition; a reader who says an answer omitting it is incomplete would be right too.
- **§550.807(a)** *"the appropriate authority ... other than the employing agency"* —
  labelled not an exclusion, because it identifies which authority rather than excepting
  anything. It is the closest call in the `other_than` stratum.

### Recall is the bigger hole, and the trigger list is why

20 chunks the detector reports as carrying **no** condition were drawn and read. **11 of them
carry one** (95% CI 34–74%):

| chunk | the condition the detector does not see |
|---|---|
| 335.107#b | "**when** the agency is accepting applications from individuals within the agency's workforce" |
| 630.1113#a | "may be used **only for** purposes related to the disaster" |
| 330.601#c | "**With prior OPM approval**, an agency may operate an alternate placement program" |
| 534.505#a-2 | "**if** the applicable agency performance appraisal system has been certified" |
| 330.212#e | "**If** an agency provides this flexibility in its RPL policies" |
| 890.502#d-ii-4 | "**If** the annuitant ... is prevented by circumstances beyond his or her control" |
| 530.302#p17 | "but **excluding** additional pay of any other kind" |
| 550.172#a | "**is not included** in the rate of basic pay used to compute ..." |
| 531.212#a-ii-5-vii | "**when** the appointment must be cleared through the White House Office" |
| 430.405#h | "**When** OPM determines that an agency's certified appraisal system is no longer in compliance" |
| 550.807#e | "**When** a determination ... is based on a finding of discrimination" |

Almost all are the bare `if` / `when` antecedent, which appears in 13.0% of in-force chunks and
is not in the commissioned trigger list. **The shipped detector finds the explicit exception
and misses the ordinary conditional.** Two caveats on the 55%: the sample was drawn from
chunks over 250 characters (2,327 of the 7,433 the detector calls unconditioned), so it is
biased toward the longer chunks that are likelier to carry a condition; and "condition" is
being read broadly, more broadly than the precision labels above.

`if`/`when` was **not** added. Adding a trigger with a 13% base rate and no measured precision
is exactly the untested expansion the corrected-statistics writeup was about
([eval-003](../../results/eval-003-corrected-statistics.md)). It is the next measurement, not
this one.

---

## 3. The unstated-condition check, without a model

Two signals, both required:

- **cue** — the answer carries a connective of the same kind: *unless*, *except*, *only if*,
  *notwithstanding*, *provided*, *subject to*, *other than*, a negation.
- **overlap** — the answer repeats at least `MIN_ACKNOWLEDGEMENT = 0.25` of the qualifier
  clause's content words, scored on **the clause**, not the paragraph. The words a qualifier
  shares with the rule it qualifies are not evidence either way, which is what the character
  span is for.

Neither alone works, and the sets below are constructed to show which one fails where.

| set | n | what it is |
|---|---:|---|
| **A** matched pairs | 32 | 16 real chunks × (an answer that acknowledges the condition, an answer that drops it), hand-written in the register `generate/answer.py` produces |
| **B** cross-paired | 480 | every answer in A against every *other* chunk's qualifier. Gold: not acknowledged. |
| **C** corpus ablation | 4,172 | every conditional qualifier in force, as (chunk with the clause cut out → unstated) and (chunk entire → acknowledged) |

Positive class = *correctly flagged as unstated*.

| set | signal | P | R | F1 | false alarms | missed |
|---|---|---:|---:|---:|---:|---:|
| A | cue only | 1.000 | 1.000 | 1.000 | 0 | 0 |
| A | overlap only @0.25 | 1.000 | 0.562 | 0.720 | 0 | 7 |
| A | **both @0.25** | **1.000** | **1.000** | **1.000** | **0** | **0** |
| A | both @0.35 | 0.889 | 1.000 | 0.941 | 2 | 0 |
| B | cue only | — | 0.794 | — | — | 99 |
| B | overlap only @0.25 | — | 0.881 | — | — | 57 |
| B | **both @0.25** | — | **0.979** | — | — | **10** |
| C | cue only | — | 0.771 | — | — | 478 |
| C | overlap only @0.25 | — | 0.587 | — | — | 861 |
| C | **both @0.25** | — | **0.834** | — | — | 346 |

Precision is not defined on B and C: every case in B has gold "not acknowledged", and C's
acknowledged side is the chunk quoting its own clause verbatim, which nothing could get
wrong. Only recall is informative there, and it is reported as such.

**Set A cannot honestly be used to argue the cue works.** An answer acknowledges a condition
by using a conditional connective, so the cue signal is close to definitional on hand-written
pairs, and the 1.000 is a check that the cue lists are complete for the connectives an author
reaches for — nothing more. **Set B is the measurement that is not circular:** on 480 cases
where the answer discusses some *other* chunk's condition, the cue alone falsely reports
acknowledgement 20.6% of the time — both answers say "unless" — and adding the overlap signal
cuts that to 2.1%. That is what the second signal buys.

0.25 is the knee, chosen from the sweep rather than by feel: it is the largest threshold at
which no acknowledging answer in A is falsely flagged, and it already recovers 97.9% of B.
Raising it to 0.35 buys 0.9 points on B and costs 2 of 16 false alarms on A.

**The weakest number here is the false-alarm rate**, which rests on 16 hand-written
acknowledging answers. C's acknowledged side is trivial by construction and B has none. An
over-flagging rate measured against real generator output is not available, because no
generated answers are on disk — see §5.

### Does the model earn its latency?

The repository already carries an NLI model (`verify/entail.py`,
DeBERTa-v3-base-mnli-fever-anli). It was run on the same 512 cases, premise = the answer,
hypothesis = the qualifier clause, acknowledged when P(entail) ≥ threshold:

| | A: F1 | A: false alarms | B: recall | ms/pair |
|---|---:|---:|---:|---:|
| lexical @0.25 | **1.000** | **0** | 0.979 | **0.010** |
| NLI @0.30 | 1.000 | 0 | 0.981 | 1.71 |
| NLI @0.50 | 0.970 | 1 | 0.994 | 1.71 |
| NLI @0.70 | 0.865 | 5 | 0.998 | 1.71 |
| NLI @0.90 | 0.727 | 12 | 1.000 | 1.71 |

**There is no threshold at which the model wins on both sets.** At its best joint operating
point it is 0.981 against 0.979 on B — two cases in 480 — and identical on A. Every threshold
that improves B degrades A faster.

Latency is not the argument against it: 1.71 ms/pair warm on an RTX 5070 (586 pairs/s) is
nothing beside generation at 21.3 tok/s. The arguments against it are that it does not do
better, that it is ~200 ms/pair on CPU where the `neural` extra is not installed, and that it
needs a 700 MB checkpoint the base install does not have. The lexical check is 10.3 µs/pair;
`unstated_conditions` over a 16-chunk evidence set is **1.09 ms** with a precomputed corpus
index and 3.51 ms without.

The `subject_to` problem in §2 is a different question and the model was **not** tested on it.
Deciding whether "subject to" conditions a rule or describes a status is a parsing task, and
whether this NLI checkpoint can do it is unmeasured.

---

## 4. What `xref.py` was doing wrong

The module existed before this work and had never been run against the corpus. Six defects,
each found by measuring rather than by reading:

| defect | symptom | measured effect |
|---|---|---|
| `_DES_RUN` consumes trailing whitespace, so `\s+of` could never match | "paragraph (b) **of this section**" never attached; the reference resolved against the citing section and "this section" reappeared as a scope reference of its own | 1,232 of the 1,335 paragraph references carry an "of …" tail — 92.3% |
| lower-priority matches discarded whole on overlap | "paragraph (d) of § 630.309 **and § 630.310(a)**" — the section pattern matches both numbers as one phrase, the first of which the paragraph reference owns, so § 630.310 vanished with it | patterns now run over a masked copy |
| elided prefixes | "paragraphs (a)(1) through (4)" resolved to `351.403#a-1` and `351.403#4` | fabricated targets, reported as dangling |
| ranges not expanded | "through" emitted only its endpoints | 30+ distinct range phrases, interiors unchecked |
| `_USC_REF` compiled without `IGNORECASE` | "5 U.S.C. **Chapter** 43" was not a reference | 13 chunks |
| `_SEC` required a 3-digit part | "§ 6.7 of this chapter" (a Civil Service Rule) was not a reference | 5 chunks, all correctly `outside` once matched |

Resolution against the store was then measured, which it had not been. Of the 4,281 targets
emitted for sections the corpus holds:

| resolves to | share |
|---|---:|
| that exact chunk id | 94.7% |
| an ancestor of it | 4.6% |
| only the section | 0.8% |
| nothing | 0.0% |

Before the fixes, **11.9% of in-corpus targets resolved to a chunk id present nowhere in the
store** and were being charged to the corpus boundary as `outside` — the column that is
supposed to mean "not 5 CFR chapter I".

**That residue is a chunker finding, not an xref finding.** `corpus/parse.py` runs a
paragraph's first item together with its chapeau, so:

- §300.201 is written "(a) ... The Office does not release the following: (1) ..." and the
  store holds `300.201#a` with no `#a-1` beneath it. A reference to (a)(1) is answered by (a).
- §890.102 holds `j-1` .. `j-5` and no bare `#j`.
- §890.301 holds `ii-7-n` where the regulation has paragraph (n), and §630.1503 holds `c-i`
  where it has (c)(1) — the designator stack mistook a roman numeral for a letter.

`resolve()` walks up to the deepest address the store actually holds, and `nameable_ids()`
materialises the ancestors those imply, so a reference to a paragraph whose chapeau was
inlined resolves rather than being called `outside`. The first two are addressing artefacts
and harmless once resolution accounts for them. **The last two are wrong anchors and belong
to whoever owns `corpus/parse.py`** — `890.301#ii-7-n` is not an address any reader would
write down, and it is what a citation to §890.301(n) currently renders as.

Finally, "this section" now resolves to the citing section. Left unresolved it was the single
largest entry in the `unscoped` column — 28.0% of all chunks — and none of it was actionable,
since the evidence set contains the citing section by construction.

---

## 5. What is not measured here

- **No generated answers exist on disk.** `data/traces.sqlite3` holds 281 traces and every
  one has `answer IS NULL`; the eval harness records candidates, not generations. So §3's
  false-alarm rate rests on 16 hand-written answers, and the prevalence of *actually* dropped
  conditions in *actually* generated answers is unknown. Everything in §1 and §2 is measured
  on the real corpus and the real retrieval output and does not depend on this.
- **The `if` / `when` conditional is not detected** (§2), and on the evidence of a 20-chunk
  probe that is the majority of conditions in this corpus.
- **`subject_to` at 53%** drags the weighted precision down by about 4 points and is shipped
  untuned (§2).
- **A qualifier nested inside a governing clause is counted twice.** 78 of 2,185 non-chapeau
  qualifiers (3.6%) sit inside an `except` / `unless` / `notwithstanding` / `only_if` /
  `proviso` clause. Some are genuinely two conditions; some are one condition counted twice.
  They are not separated.
- **Clause spans run long.** Median 91 characters, p90 243, max 912. A comma ends a
  sentence-initial qualifier but not a trailing one, because cutting "only if OPM determines
  that the agency has, for a period of no less than 90 days, ..." at the first comma left the
  acknowledgement check scoring against four words. The failure in the other direction —
  §630.305 where "may not be used" runs on into the main clause — is unfixed and dilutes
  overlap.
- **No abstention policy is proposed.** This module reports; what to do with an unstated
  condition belongs to `verify/abstain.py`, and the risk–coverage curve that would justify a
  threshold has not been run ([eval-005](eval-005-abstention.md), outstanding).

---

## How each number was produced

Everything below runs offline against `data/warrant.sqlite3` and the cached model weights.

| § | what | how |
|---|---|---|
| 1 | dangling references over the benchmark | `mine_all` at horizon = `MAX(valid_from)`, retrieve each item at its own `as_of` and scope, then `dangling_references(evidence, in_corpus=nameable_ids(...), include_outside=True)` at `final_k` ∈ {4, 8, 16} |
| 2 | prevalence | `qualifiers()` over `store.as_of("2026-08-30")`, 9,961 chunks |
| 2 | precision | 8 instances per kind at seed 2026, plus 24 more `subject_to` at seed 7 drawn disjointly from the first 8; labelled by reading the whole chunk |
| 2 | recall probe | 20 chunks over 250 characters with no conditional qualifier, seed 7 |
| 3 | sets A/B/C | 16 hand-written answer pairs; the 480 cross-products; the 4,172 corpus ablations |
| 4 | resolution depth | `resolve()` over every target for a section the corpus holds |

Corpus as of this run: 13,145 chunk versions, 9,961 in force, 26 parts.

## The labelled instances

`Y` = a clause that bounds what the cited rule requires. `N` = a false positive. The seed and
the sampling rule are above; these are all 104 drawn, none discarded.

### bound — 8/8

| # | chunk | clause | label |
|---|---|---|:--:|
| 01 | 531.606#a | may not exceed the rate of basic pay payable for level IV of the Executive Schedule | Y |
| 02 | 575.109#c-1 | may not exceed 50 percent of the employee's annual rate of basic pay | Y |
| 03 | 550.112#m-3 | may not exceed 8 hours in any 24-hour period | Y |
| 04 | 630.401#c | may not exceed a total of 480 hours | Y |
| 05 | 630.606#p6 | may not be less than the tour of duty prescribed for the employee's post | Y |
| 06 | 575.110#a | may not exceed 4 years | Y |
| 07 | 630.1203#j-3 | may not exceed 26 administrative workweeks | Y |
| 08 | 451.303#a-1 | may not exceed 5 percent of the career SES | Y |

### chapeau — 8/8

| # | chunk | ends | label |
|---|---|---|:--:|
| 09 | 890.1410#c-1 | "at midnight of the earlier of the following dates:" | Y |
| 10 | 430.309#a | "each agency must—" | Y |
| 11 | 890.1604#d-1 | "To a Postal Service Medicare covered annuitant who—" | Y |
| 12 | 575.109#c-2 | "must be made in writing and include—" | Y |
| 13 | 575.305#b | "when the agency determines that—" | Y |
| 14 | 551.211#d | "unless both of the following conditions are met:" | Y |
| 15 | 536.303#b | "must apply the following rules ...:" | Y |
| 16 | 630.302#a | "is the:" | Y |

### except — 8/8

| # | chunk | clause | label |
|---|---|---|:--:|
| 17 | 890.304#b | Except as provided in paragraph (b)(3) of this section | Y |
| 18 | 890.1108#a | Except as otherwise provided | Y |
| 19 | 550.1206#a | except as provided in paragraphs (b) and (c) of this section | Y |
| 20 | 300.104#c-2 | Except as provided in paragraph (c)(1) of this section | Y |
| 21 | 353.209#a | except for cause | Y |
| 22 | 530.304#c | except that an alternate method may be used— | Y |
| 23 | 550.184#e | except for special agents in the Foreign Service | Y |
| 24 | 511.201#p1 | except those specifically excluded by section 5102 of title 5 | Y |

### notwithstanding — 8/8

| # | chunk | clause | label |
|---|---|---|:--:|
| 25 | 551.211#f | Notwithstanding any other provision of this section | Y |
| 26 | 530.203#h | notwithstanding § 530.204 | Y |
| 27 | 890.1043#b | notwithstanding the OPM debarment | Y |
| 28 | 630.205#b | Notwithstanding 5 U.S.C. 6303(a) | Y |
| 29 | 575.204#b | Notwithstanding any other provision in this subpart | Y |
| 30 | 890.106#h | notwithstanding any state or local law | Y |
| 31 | 550.162#g | Notwithstanding paragraph (c)(1) of this section | Y |
| 32 | 550.1611#f-5 | notwithstanding any other provision of law or this subpart | Y |

### only_if — 8/8

| # | chunk | clause | label |
|---|---|---|:--:|
| 33 | 534.507#e | only if OPM determines that the agency has ... consistently applied | Y |
| 34 | 430.304#b-5 | only if the agency determines there is an adequate basis | Y |
| 35 | 532.703#b-8 | only to the extent of restoration to the grade immediately preceding | Y |
| 36 | 534.203#b | only if OPM has determined that a higher maximum stipend is warranted | Y |
| 37 | 630.606#a | only when he has completed a basic service period of 24 months | Y |
| 38 | 534.507#b-1 | only upon a determination by the authorized agency official | Y |
| 39 | 630.1504#g | only if the agency makes a written determination | Y |
| 40 | 550.1302#2-2-ii | only if application of 5 U.S.C. 5545b has not been waived | Y |

### other_than — 6/8

| # | chunk | clause | label | why |
|---|---|---|:--:|---|
| 41 | 630.1202#… | other than migraines) | Y | excludes a class from the definition |
| 42 | 315.610#a-1 | other than by removal for cause on charges of misconduct | Y | |
| 43 | 890.102#c-1 | other than an acting postmaster | Y | |
| 44 | 531.609#e | other than a general pay adjustment) | Y | |
| 45 | 315.906#c | other than for compensable injury or military duty) | Y | |
| 46 | 551.104#p86 | to someone other than the agency | **N** | a component of a definition, not an exception |
| 47 | 550.807#a | other than the employing agency | **N** | identifies which authority; hardest call in this stratum |
| 48 | 530.321#a | other than a general pay adjustment) | Y | |

### prohibition — 7/8

| # | chunk | clause | label | why |
|---|---|---|:--:|---|
| 49 | 532.504#c | may not directly or indirectly intimidate | Y | |
| 50 | 550.182#f | will not be payable during the designated period | Y | |
| 51 | 351.705#b-4 | May not provide for the assignment of a full-time employee | Y | |
| 52 | 890.605#p1 | may not be limited for persons who ... | Y | |
| 53 | 630.1208#c | may not require any personal or confidential information | Y | |
| 54 | 630.305#p1 | may not be used by employees to avoid forfeiture | Y | span runs into the main clause |
| 55 | 630.407#p1 | may not thereafter be used | Y | |
| 56 | 591.238#b | will not cause the combined total to exceed 25 percent | **N** | a ceiling; now classified `bound` |

### proviso — 7/8

| # | chunk | clause | label | why |
|---|---|---|:--:|---|
| 57 | 551.424#b | provided an event arises incident to representational functions | Y | |
| 58 | 315.201#b-1-vii | provided the appointment is converted to a career appointment | Y | |
| 59 | 630.1204#a-8 | provided that the agency and employee agree | Y | |
| 60 | 630.1204#a-5 | provided that the need for counseling arises | Y | |
| 61 | 890.303#i | provided that the employee continues to be entitled | Y | |
| 62 | 890.1604#d-2-ii | provided that the individual demonstrates such residency | Y | |
| 63 | 315.201#b-3-ii-F | provided the person is employed in the competitive service | Y | |
| 64 | 430.206#a-2 | employees are provided a rating of record on an annual basis | **N** | past participle |

### unless — 8/8

| # | chunk | clause | label |
|---|---|---|:--:|
| 73 | 890.204#a-2 | unless it is waived in writing by the carrier | Y |
| 74 | 630.206#a | Unless an agency establishes a minimum charge of less than one hour | Y |
| 75 | 890.306#f-i | unless the annuitant provides documentation | Y |
| 76 | 550.1616#b | unless CBP determines there exists ... evidence of fraud | Y |
| 77 | 890.1035#a | unless rescinded by the suspending official | Y |
| 78 | 890.1610#a-1 | unless otherwise stated in this subpart | Y |
| 79 | 890.304#b-4-i | unless he is eligible for continued enrollment as an employee | Y |
| 80 | 890.304#d-iii | unless the employee or annuitant provides documentation | Y |

### subject_to — 17/32

First 8 (drawn with the rest):

| # | chunk | clause | label | why |
|---|---|---|:--:|---|
| 65 | 630.302#a | becomes subject to section 6304(b) of title 5 | **N** | the rule is about the date; the phrase is its subject matter |
| 66 | 630.1506#b-iv-2 | the requirement ... is subject to applicable laws | Y | conditions a requirement |
| 67 | 630.101#p1 | having employees subject to this part | **N** | tautological self-scope |
| 68 | 530.308#c | subject to the requirement that ... | Y | |
| 69 | 551.601#a | applies to all employment subject to its child labor provisions | Y | limits the reach of the rule; a close call |
| 70 | 630.1015#a | shall become subject to the policies and procedures of the new agency | **N** | the operative rule itself |
| 71 | 890.304#d-ii | Subject to the provisions of paragraph (d)(iii) of this section | Y | |
| 72 | 630.310#h-2 | will not be subject to time limits on usage | **N** | the operative rule itself |

Held out, drawn after the above were labelled:

| # | chunk | clause | label | why |
|---|---|---|:--:|---|
| S01 | 550.1207#d | subject to reduction in the same manner as provided in 5 U.S.C. 6304(c) | Y | |
| S02 | 532.254#c | shall be subject to the general provisions of this part | **N** | operative statement |
| S03 | 550.401#d | having employees subject to this subpart | **N** | tautological self-scope |
| S04 | 890.1019#c | subject to adjustment on the basis of the ... circumstances in § 890.1016 | Y | |
| S05 | 351.203#p12 | For an employee not subject to 5 U.S.C. Chapter 43 | Y | bounds which definition applies |
| S06 | 430.208#k | Subject to 5 U.S.C. 7116(a)(7) | Y | |
| S07 | 630.1203#j-5 | are subject to the same notification and scheduling requirements | Y | imposes requirements the answer must carry |
| S08 | 451.302#b-1 | a position that is subject to OPM position allocations under part 319 | **N** | restrictive relative clause; a close call |
| S09 | 550.162#b | only during the period he is subject to these conditions | Y | the temporal condition on payment |
| S10 | 630.301#b-3 | Covered positions are subject to a performance appraisal system | **N** | operative statement |
| S11 | 353.203#d | The Ready Reserve as a whole is subject to ... active duty | **N** | background |
| S12 | 591.306#a | or is subject to prescribed minimum inconvenience or hardship factors | Y | one arm of the "only (1)…(2)" test |
| S13 | 534.508#c | subject to part 752, subpart D of this chapter | Y | |
| S14 | 330.211#a | subject to the requirements of subpart F of this part | Y | |
| S15 | 432.106#a-2 | an appointment which is not subject to a probationary period | **N** | restrictive relative clause |
| S16 | 551.501#b | For employees subject to part 610 of this chapter | Y | applicability condition |
| S17 | 551.204#b | unless the employees are subject to § 551.211 or § 551.212 | Y | real, though the `unless` names it too |
| S18 | 410.402#b-8 | subject to the limitation in 5 U.S.C. 5550(b)(2)(G) | Y | |
| S19 | 536.104#c-5 | unless the employee is subject to a mobility agreement | **N** | status; the `unless` is the condition |
| S20 | 451.104#c | An award is subject to applicable tax rules | **N** | the operative rule itself |
| S21 | 630.1302#p1 | employees ... who are subject to regulations issued by the Postmaster General | **N** | restrictive relative clause |
| S22 | 551.212#b | the employee is subject to the foreign exemption of the Act | **N** | operative consequence |
| S23 | 353.205#e | is subject to whatever policy and disciplinary action the agency would apply | **N** | operative consequence |
| S24 | 630.1504#g | (subject to § 630.1506(b)) | Y | |
