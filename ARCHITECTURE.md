# Architecture

## The claim

Warrant answers questions about US federal HR regulation **correctly for a given scope and a
given date**, and — when it is wrong — localizes the failure to the pipeline stage responsible.

The second half is the point. Most RAG evaluations report that a system failed. This one
reports *where the evidence was lost*, and then tests whether repairing that stage actually
fixes the answer.

## Why the usual diagram is not enough

The standard production-RAG architecture is a forward pipeline: ingest, chunk, embed, index,
retrieve, rerank, prompt, generate, guard. It is correct as far as it goes, and every arrow
points one way. It has no answer to the only question that matters once a system is live:
*this answer is wrong — which stage made it wrong?*

Warrant is that forward pipeline plus a **backward attribution path**, and the backward path
is the part with novelty in it.

---

## How to read this document

Sections describe the design. Not all of the design is built, and a document that reads as
present tense throughout is a document a reviewer cannot trust. Every section below is marked:

| Marker | Meaning |
|---|---|
| **[built]** | in `src/`, covered by tests, and exercised by the published numbers |
| **[partial]** | some of it exists; the section says exactly which part |
| **[designed]** | decided and specified, no implementation yet; the phase table in section 12 says when |
| **[built, off]** | built and tested, disabled in the shipped config, with the measurement that decided it written beside the flag |

If a claim is unmarked, treat it as designed rather than built. An earlier revision of this
document described query classification, parent expansion, context assembly, TREC pooling,
three oracle interventions, two replay modes and admission control in the present
indicative. None of those existed. That is a worse failure than any of them being missing,
because it makes every other claim on the page unverifiable by inspection.

---

## 1. Corpus **[built]**

Point-in-time US federal regulation from the [eCFR versioner
API](https://www.ecfr.gov/developers/documentation/api/v1), Title 5 (Administrative Personnel),
chapter I — the OPM parts covering leave, pay administration, hours of duty, staffing,
performance management, and benefits.

Two endpoints carry the whole corpus:

| Endpoint | Gives |
|---|---|
| `/versions/title-5.json?part=630` | every section-level version record, with amendment and issue dates |
| `/full/{date}/title-5.xml?part=630` | the full text of that part **as it stood on `{date}`** |

That second endpoint is the unlock. It means "what was the rule on 2019-06-01?" has a
machine-checkable answer drawn from the primary source, not from a synthetic dataset.

### What the corpus is not

Stated plainly here because it is easy to overclaim:

- **The history floor is 2017.** Every part reports a first version date of 2016-12-27, but
  `/full/` returns 404 there. The usable point-in-time span is 2017 to present.
- **Version records are per-section, not per-snapshot.** `/versions` returns hundreds of rows
  for a single part; they collapse to a much smaller number of distinct snapshot dates. Part 630
  has 8. Count distinct dates, never rows.
- **This is public law.** Nothing in it is confidential. See section 3.

---

## 2. Ingestion **[built]**

### Editorial apparatus is the hard problem

Regulatory XML interleaves regulatory text with editorial apparatus: authority notes
(`AUTH`), source notes (`SOURCE`), editorial notes (`EDNOTE`), footnotes (`FTNT`), and —
the one that actually bites — pending-amendment pointers:

```xml
<XREF ID="20200810" REFID="1">Link to an amendment published at 85 FR 48089, Aug. 10, 2020.</XREF>
```

These appear when an amendment is pending and vanish when it publishes. A naive differ reads
every appearance and every disappearance as a substantive amendment.

Measured across 26 Title 5 parts and 222 snapshots: **140 section-pairs differ only in
apparatus, against 188 whose regulatory text genuinely changed.** Without stripping, roughly
43% of detected "amendments" would have been publication schedules rather than law — and in
the first run against Part 630, six of eight sampled diffs were nothing but these pointers.
Full run record in [`results/spike-001-amendment-viability.md`](results/spike-001-amendment-viability.md).

So apparatus stripping is a first-class, unit-tested component with its own fixtures — not a
`strip()` inside a parser. It is also the reason the `ingestion` row of the failure budget is
expected to be non-zero: it is the row everyone assumes is free.

### Two ways the corpus silently loses coverage

Both were found by ingesting and then querying, not by reading code. Both fail without an
error: the corpus simply holds no evidence for some dates, and a question about those dates
gets an honest-looking "I do not know" that is actually a bug.

**Parts never amended since 2017 ingest to nothing.** Such a part advertises exactly one
version date — the 2016-12-27 floor — which `/full/` refuses to serve. Parts 511, 530, 536
and 610 all behaved this way, so hours-of-duty questions had no evidence at all. The fix is
to always ingest the title's `latest_issue_date` as a snapshot. Not *today*: `/full/` 404s on
any date after the issue date, and the gap is days wide, so ingestion reads the API's own
statement of currency rather than the wall clock.

**Sections are dated from the floor, not from first observation.** Part 315 has no post-floor
amendment until 2020-10-16. Dating its text from that snapshot would make three years of
answerable questions unanswerable. eCFR records a version date whenever a part is amended, so
the absence of one between the floor and the first snapshot is *positive evidence* that the
text did not change — the backfill is licensed by the source, not assumed. Sections that
first appear in a later snapshot are never backfilled; they did not exist, and dating them
earlier would invent law.

### Chunking

Hierarchical, following the regulation's own structure: title, chapter, part, subpart,
section, paragraph. Sections are the retrieval unit; paragraphs are the citation unit; the
parent section is available for expansion at context-assembly time. Tables are never split.

### Diffing

Consecutive snapshots are aligned by section identifier and classified:

| Class | Meaning |
|---|---|
| `substantive_localized` | stable identifier, alignable text, localized semantic change |
| `wholesale_rewrite` | section replaced outright; before/after not alignable at paragraph level |
| `editorial` | punctuation, case, or whitespace only |
| `apparatus_only` | changed only in stripped apparatus; suppressed |
| `renumbered` | text preserved under a new identifier |
| `added` / `removed` | section appears or disappears |

Only `substantive_localized` feeds the temporal benchmark. The rest are counted and reported,
because the discard rate is itself a finding.

Measured: **158 of 188 changed section-pairs (84.0%) are clean substantive**, which cleared
the pre-registered viability threshold of 60% and made amendment mining the benchmark
centrepiece. Zero renumbering was found across all 26 parts.

---

## 3. Scope and applicability — and what this is *not* **[partial]**

Warrant filters retrieval by **applicability**: which regulation governs this person, in this
agency, in this role, on this date. Government-wide OPM rules apply to everyone; agency
supplements do not; some subparts reach only certain pay systems or bargaining units.

**This is not access control, and the repository does not claim it is.** eCFR is published law.
Nothing here is confidential, nothing can leak, and no "zero leak rate" claim is made or
measured. The metric is **applicability error rate** — citing a rule that does not govern the
asker — which is a correctness failure, not a security breach.

The architectural decision that *does* carry over from access-controlled systems is where the
predicate lives: **inside the retrieval query, not as a post-filter.** The justification here is
correctness and cost — inapplicable text should never compete for reranker budget or context
window — rather than confidentiality, which would be the justification in a system with real
ACLs. Confidentiality-aware retrieval is a different project against a corpus with real ACL
ground truth.

---

## 4. Bitemporal store **[built]**

Two independent time axes, because there are two independent questions.

```
valid_from  / valid_to    when this text was the law in the real world
system_from / system_to   when Warrant believed this text was the law
```

Rows are **append-only**. A correction never updates a row; it closes the old row's
`system_to` and inserts a new one. `ingested_at` alone cannot support this — a single arrival
timestamp records when a row appeared but not when it stopped being believed, so a re-ingest
silently destroys the state you would need to reconstruct.

With both axes closed you can ask the question that matters for audit:

> Reproduce the answer the system would have given on 14 March, using only what it believed
> on 14 March.

Storage is SQLite (WAL) with FTS5 for lexical retrieval and `sqlite-vec` for dense vectors.
No server, no Docker: `git clone && make` has to work.

**Honest limit.** System-time versioning covers the *text*. It does not reconstruct historical
*indexes* — embeddings, chunker version, BM25 parameters. Index state is recorded by config
hash, not replayed. Section 8 says what each replay mode therefore guarantees.

---

## 5. Request path **[partial]**

```
query
  -> scope resolution: (pay system, service)                      [built]
  -> applicability + as-of predicate, pushed into the query       [built]
  -> BM25  ||  dense (concurrent; 31.1 -> 18.4 ms p50)            [built]
  -> reciprocal rank fusion                                       [built]
  -> cross-encoder rerank                                         [built]
  -> context assembly (top context_k excerpts, numbered)          [partial]
  -> generation -> claims + evidence ids                          [built]
  -> deterministic span alignment                                 [built]
  -> answer | abstain                                             [built]

  -> source + authority filter, authority tie-break in fusion     [built]
  -> query classification (semantic / structured / comparative)   [designed]
  -> parent expansion + dedup                                     [designed]
  -> token-budgeted assembly, table preservation                  [designed]
  -> entailment + contradiction                                   [built, off]
  -> calibrated confidence -> qualified answer | flag conflict     [built, off]
```

Two stages are marked **[built, off]** rather than built. Both exist, are tested, and are
disabled in `configs/default.yaml` with the number that decided it beside them: entailment
buys +2.3 points over span alignment at p=0.55 on real generator output, and the calibrated
combiner does not beat a threshold on the single top-1 fusion score (AURC delta +0.0019, CI
-0.0017 to +0.0070). Shipping them on would be asserting a benefit the measurement does not
support; deleting them would discard the one thing they demonstrably do buy, which for
entailment is a second channel -- +49.1 points on claims flipped to contradict their
evidence, where the span aligner finds a supporting span every single time.

**Scope facets are `pay_system` and `service` only.** Agency, role and bargaining unit are
discussed in section 3 as the shape of the problem; `Scope.of` rejects them. Context assembly
is a slice at `retrieve.context_k`, not a token budget: no tokenizer is consulted and no
table logic exists, which is currently unfalsifiable because the corpus contains no
`GPOTABLE` elements at all.

### Grounding

The generator emits **claims plus evidence chunk IDs** under a constrained JSON schema. It does
**not** emit character offsets — asking a 4B model to count characters produces confidently
wrong indices. Spans are computed afterwards by a deterministic aligner.

The aligner is allowed to return `span = null`. That is not a bug; it means *the model claimed
support from this chunk and no supporting span can be located in it*, which is a grounding
failure and is recorded as one.

### Verification

No single signal is the authority. NLI models are trained on short premise/hypothesis pairs
and degrade on long regulatory prose. Signals — lexical/semantic alignment, entailment,
contradiction, citation coverage — feed a **calibrated** combiner (logistic regression first;
interpretable beats clever) fitted on a labeled dev set.

Reported as calibration (Brier, ECE) and a **risk–coverage curve**, not as a threshold —
the published form would read like this, with numbers this system has not yet measured:

> at 90% coverage, error X%; at 75% coverage, error Y%

That turns abstention into an engineering trade-off instead of a checkbox. **[designed]** —
only `verify/align.py` exists today: a lexical-overlap span aligner. There is no entailment
model, no contradiction detection, no combiner and no calibration. An earlier revision of
this document carried illustrative figures in that blockquote, formatted exactly like a
result, which is the single most misreadable thing a design document can do.

### Temporal conflict, expected and unexpected

Two versions of one section genuinely contradict each other, and surfacing that is correct for
*"compare the 2019 and 2026 rules."* It is a **failure** for *"what is the rule today."*
The two cases are labeled separately; only the second enters the failure budget.

---

## 6. Evaluation **[partial]**

Three buckets, **reported separately and never averaged into one number**:

| Bucket | Source | Measures |
|---|---|---|
| Temporal | mined from real amendments; ground truth from the diff itself | dating correctness |
| Scope | part-level applicability, read off the CFR part titles | not over-excluding |
| Scope-exclusion | the same, inverted | not over-including |
| Generated | in-force paragraphs on a deterministic stride | corpus reachability |
| Human | hand-written, `benchmarks/human.yaml` | realistic query distribution |

Each temporal item covers **one amended paragraph**, not a section's whole changed set. An
earlier miner used the section's changed set, which made 41 of 252 items require more
paragraphs than the pipeline returns — one needed 56 in a list of 8. They were unsatisfiable
by construction and were being reported as retrieval failures, putting a floor under the
bucket no configuration could beat. Minimality is not a nicety; a non-minimal evidence set is
a silently broken benchmark.

The human bucket will be small, and at that size confidence intervals will swallow the
differences between configurations. It exists to characterize the query distribution, not to
rank systems, and the README says so.

### Minimal sufficient evidence sets

A question does not have one gold chunk. It has a disjunction of sufficient sets:

```json
{
  "question": "How much unpaid FMLA leave is available in a 12-month period?",
  "acceptable_evidence": [["630.1203#a"], ["630.1202#def", "630.1203#a"]]
}
```

Each set must be **minimal** — no proper subset is also sufficient — or supersets creep in and
everything becomes "sufficient."

Sets are to be discovered rather than enumerated up front, using **TREC-style pooling**: run
diverse retrieval configurations, pool their unseen high-ranked evidence, judge it, and
promote genuinely sufficient alternatives into the accepted sets. This carries pooling's
known bias — systems outside the pool are penalized for finding valid unjudged evidence.

**[designed].** No pooling is implemented. Every mined item carries exactly one evidence set
of exactly one chunk; only `benchmarks/human.yaml` can express a disjunction, and those are
hand-written. So for 809 of 851 non-human items the distinction below between *"did any
sufficient set survive"* and *"was the gold chunk retrieved"* is currently a distinction
without a difference, and the machinery is in place for when pooling lands rather than
earning its keep today.

The autopsy therefore asks *"did any sufficient set survive this stage?"*, never *"was the gold
chunk retrieved?"*

---

## 7. Failure localization **[built]**

### Observational budget — every failure, cheap

Walk the stages and record where the last sufficient evidence set was lost:

```
in corpus -> chunked intact -> BM25@k -> dense@k -> fusion -> rerank
          -> applicability/as-of filter -> context -> generation -> grounding -> verifier
```

This answers **where the evidence visibly disappeared**. It is labeled the *observational*
failure budget, and it is biased: it attributes to the first stage where evidence was lost,
which systematically under-counts upstream causes. A section split badly by the chunker is
retrieved "successfully" and blamed on generation.

### Interventional localization — a sample, expensive

Replace one stage's output with an oracle and re-run:

| Intervention | If the answer becomes correct | Status |
|---|---|---|
| unbounded candidate depth, reranker off | evidence is reachable, so it was ranked too low | **[built]** |
| oracle evidence straight into context | fault is upstream of generation | **[designed]** |
| oracle chunking, original retrieval downstream | chunking implicated | **[designed]** |

Only the first is implemented, and it is a depth sweep rather than an oracle substitution.
It yields two labels, `ranking` and `unreachable`.

This is **fault localization, not causal proof**. Oracle substitution shows that repairing a
stage repairs the answer; it does not establish that the stage was the unique cause, and stage
interactions are real. The docs call it repair attribution and claim nothing stronger.

Attribution is **multi-label**. A failure can implicate chunking *and* generation. The
interventional totals therefore **do not sum to the failure count**, and that is more honest
than forcing them to.

### Distinguishing a demotion from a cut

`rerank` is charged only when the evidence was inside the fused top-`final_k` and the
reranker moved it out. If it sat below `final_k` in the fused order, plain truncation would
have lost it too, and the loss is `truncation`.

This is not a detail. The first version of the ladder blamed `rerank` whenever a reranker had
run, putting 124 failures on the cross-encoder — and removing the cross-encoder entirely then
moved the bucket by 0.1 points, which cannot be true of a stage responsible for half the
failures. Exactly the first-loss bias described above, caught by disagreement between the
budget and a direct ablation, which is the argument for running both.

### The artifact

Measured, not illustrative. Full run in
[`results/eval-002-failure-budget.md`](results/eval-002-failure-budget.md).

```
721 temporal items          before              after widening the head and cut
                            236 failures        170 failures  (67.3% -> 76.4%)
  ingestion                       0                   0
  applicability                   0                   0
  temporal                        0                   0
  retrieval                      25                  25   unchanged, correctly
  fusion                         87                  40   -47
  rerank                         46                  64   +18, bottleneck moved on
  truncation                     78                  41   -37
```

The fix was chosen from the table rather than in advance, and the table moved where it
predicted. `rerank` rising is the instrument working: more evidence survives long enough to
reach the reranker, so more of it can be demoted there.

The budget must be shown to **move** after a targeted fix. A budget that is only ever printed
once is decoration; a budget that redirects the next commit is an instrument. Which row gets
fixed first is decided by the P0 measurement, not in advance.

---

## 8. Replay **[built]**

Two modes, two different guarantees.

**Artifact replay** — *what exactly happened on request X?* The full trace is stored: query,
scope, as-of date, candidate IDs with per-stage scores, fusion and rerank orders, assembled
context, prompt, model and config hashes, answer, verification verdicts. Inspecting a past
decision needs no historical index.

**Counterfactual replay** — *what would today's system do with that request?* Re-runs the
original query through the current pipeline and diffs: which stage's membership changed,
which reordered, what entered and left the final k, and the **first** stage that diverged.
This is the regression harness over production traffic, and it is what gates a config change.

Both are shipped: `warrant replay show` and `warrant replay diff`, over traces the API records
on every request. Every stage stores the score it ranked by — BM25 out of the SQL, the RRF
weight, the cross-encoder logit — and its wall-clock time; all of those used to be computed and
dropped on the next line, which left "the reranker demoted it" and "the reranker barely
preferred anything" indistinguishable after the fact.

Splitting them is what makes the honest limit in section 4 a non-issue: neither mode pretends
to rebuild a historical vector index.

---

## 9. Invariants **[built]**

Deterministic assertions in CI — correctness moved out of probabilistic evaluation:

All of these run in CI as a separate job (`make invariants`), so a failure reads as *the
system is wrong* rather than *a test broke*.

- For an as-of-dated query, **at most one version of any section is in force** — all rows for
  a section share a `valid_from`, and no paragraph address appears twice. Two versions of
  630.1203 reaching one prompt is a filter bug, detectable before the model runs.
- Every retrieved chunk's validity interval contains the as-of date.
- Every retrieved chunk is applicable to the resolved scope.
- **Every citation address is unambiguous.** 13% were not, before paragraph designators were
  tracked hierarchically: eCFR flattens the CFR's `(a) -> (1) -> (i) -> (A)` nesting into
  sibling `<P>` elements, so a section with several sub-lists restarts at `(a)` repeatedly and
  `550.703#a` matched four different paragraphs. Levels are now recovered by designator
  sequence continuity — `(ii)` after `(b)(1)(i)` is the second roman numeral, not the ninth
  letter, and only what came before can tell those apart.
- Validity intervals for a paragraph never overlap.
- Apparatus stripping is idempotent, and fixtures assert the known pointer forms are removed.
- *(P1)* Every claim in an emitted answer carries at least one evidence ID.

## 10. Load policy **[partial]**

Under pressure, **admission control before degradation**. Shedding the verifier to save
latency trades a slow answer for a wrong answer about someone's leave entitlement, which is the
wrong trade in this domain.

Which stages *may* be shed is decided by measurement, not by declaration, and `make latency`
now produces the frontier to decide from. Measured on the temporal bucket: lexical-only runs at
23.8 ms p50 with the same sufficiency as the full pipeline at 71.0 ms, so on this corpus the
cross-encoder is a sheddable stage and the as-of predicate — which the paired test shows moves
the wrong-version rate by 96 points — is not.

Admission control is **[partial]** for generation, which is where the real ceiling is: 29.2
tokens/s unbatched over ~205 tokens is 7.7 requests per minute, so the API admits under a
semaphore
and returns `503` with `Retry-After` rather than queueing a client for the 33 minutes that 100
concurrent requests would actually take. What is still **[designed]**: shedding individual
retrieval stages under load, and a load test that produces the condition rather than reasoning
about it. Untested admission control is aspirational config, and this half is tested only by
unit tests that fake the contention.

---

## 11. Non-goals

- Not an access-control system. See section 3.
- Not legal advice, and the UI says so.
- Not a general web-scale RAG. The corpus is bounded, versioned, and structured on purpose.
- Not a claim to beat published RAG benchmarks. The benchmark here is constructed, its
  construction is documented, and its biases are stated.

## 12. Phases

Each ships something demonstrable on its own.

| | Ships | Rationale |
|---|---|---|
| **P0** ✅ | eCFR point-in-time ingest, apparatus stripping, structural diff, bitemporal store, lexical + dense + RRF + cross-encoder, applicability and as-of predicates, four benchmark buckets with a section-level dev/test split, observational and interventional failure localization, seven CI invariants | The headline artifact needs no LLM |
| **P1** ✅ | Generation, evidence ids, deterministic span alignment, and the measurement of all three — hallucination rate, citation precision, abstention quality | A generator nothing scores is a generator nobody can trust |
| **P2** | UI: time slider, scope selector, citation highlighting, trace viewer | Designed; the API it runs on is built, and `/api/ask/stream` sends the evidence at 18 ms rather than making the UI wait 19 s for the prose |
| **P3** ✅ | Calibrated verifier, abstention tiers, risk–coverage | Built and measured. The finding is a null: eight features do not beat one. Both stages ship behind flags carrying their own p-values |
| **P4** ✅ | Artifact and counterfactual replay over persisted traces | |
| **P5** | Retrieval-stage shedding and a real load test | The latency frontier is measured; the shedding policy is not tested under load |
| **P6** ✅ | Five-tier source hierarchy: statute, regulation, notices, guidance, archival scans, with authority pushed into retrieval | Reading only the regulation is reading the middle of an argument |
| **P7** ✅ | Operations: Prometheus metrics, correlated structured logs, measured serving guardrails, a quality gate whose floor is a bootstrap lower bound | A number nobody reads back is a number that silently stops being true |

P1 did not start until P0 showed the benchmark was viable, and the benchmark was
rebuilt twice after that when the instrument found it measuring the wrong thing.

## 13. Hardware envelope

Developed on an RTX 5070 Laptop (8 GB, `sm_120`, torch 2.11+cu128), 31 GB RAM, no Docker.
Co-resident budget: embedder ~2 GB + cross-encoder reranker ~2 GB + a 4B generator at Q4
~2.5 GB; or a 7B at Q4 with the reranker on CPU. Everything is free and offline after the
first ingest. A reviewer without a GPU can run the full lexical path and the entire P0
failure budget.
