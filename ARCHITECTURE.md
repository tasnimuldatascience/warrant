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

## 1. Corpus

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

## 2. Ingestion

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

## 3. Scope and applicability — and what this is *not*

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

## 4. Bitemporal store

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

## 5. Request path

```
query
  -> query classification (semantic / structured / comparative / temporal)
  -> scope resolution: (agency, role, pay system, bargaining unit, as_of date)
  -> applicability + as-of predicate  ......... pushed into the retrieval query
  -> BM25  ||  dense                  ......... run concurrently
  -> reciprocal rank fusion
  -> cross-encoder rerank
  -> parent expansion + dedup
  -> context assembly (token budget, table preservation)
  -> generation
  -> claim decomposition
  -> evidence alignment
  -> entailment + contradiction
  -> calibrated confidence -> answer | qualified answer | abstain | flag conflict
```

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

Reported as calibration (Brier, ECE) and a **risk–coverage curve**, not as a threshold:

> at 90% coverage, error 2.8%; at 75% coverage, error 0.7%

That turns abstention into an engineering trade-off instead of a checkbox.

### Temporal conflict, expected and unexpected

Two versions of one section genuinely contradict each other, and surfacing that is correct for
*"compare the 2019 and 2026 rules."* It is a **failure** for *"what is the rule today."*
The two cases are labeled separately; only the second enters the failure budget.

---

## 6. Evaluation

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

Sets are discovered rather than enumerated up front, using **TREC-style pooling**: run diverse
retrieval configurations, pool their unseen high-ranked evidence, judge it, and promote
genuinely sufficient alternatives into the accepted sets. This carries pooling's known bias —
systems outside the pool are penalized for finding valid unjudged evidence — and the bias is
documented rather than hidden.

The autopsy therefore asks *"did any sufficient set survive this stage?"*, never *"was the gold
chunk retrieved?"*

---

## 7. Failure localization

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

| Intervention | If the answer becomes correct |
|---|---|
| oracle evidence straight into context | fault is upstream of generation — not yet which stage |
| oracle retrieval, original chunking downstream | retrieval implicated |
| oracle chunking, original retrieval downstream | chunking implicated |

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

## 8. Replay

Two modes, two different guarantees.

**Artifact replay** — *what exactly happened on request X?* The full trace is stored: query,
scope, as-of date, candidate IDs with per-stage scores, fusion and rerank orders, assembled
context, prompt, model and config hashes, answer, verification verdicts. Inspecting a past
decision needs no historical index.

**Counterfactual replay** — *what would today's system do with that request?* Re-runs the
original query through the current pipeline and diffs: retrieval changed, ranking changed,
context changed, answer changed, verdict changed. This is the regression harness over
production traffic, and it is what gates a config change.

Splitting them is what makes the honest limit in section 4 a non-issue: neither mode pretends
to rebuild a historical vector index.

---

## 9. Invariants

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

## 10. Load policy

Under pressure, **admission control before degradation**. Shedding the verifier to save
latency trades a slow answer for a wrong answer about someone's leave entitlement, which is the
wrong trade in this domain.

Which stages *may* be shed is decided by measurement, not by declaration: the latency/quality
Pareto frontier shows which stages are inside the noise. Stages measured to matter are never
shed; the system queues and returns `429` with `Retry-After` instead. Any shedding policy that
ships must come with a load test that actually produces the condition — untested admission
control is aspirational config.

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
| **P0** ✅ | eCFR point-in-time ingest, apparatus stripping, structural diff, bitemporal store, lexical + dense retrieval + RRF + cross-encoder, applicability and as-of predicates, four benchmark buckets, observational and interventional failure localization, one before/after shift, five CI invariants | The headline artifact needs no LLM: 6 of the 8 stages are pre-generation |
| **P1** | Generation, evidence IDs, deterministic span alignment | Makes it a RAG system rather than a search system |
| **P2** | UI: time slider, scope selector, citation highlighting, trace viewer | The shareable artifact |
| **P3** | Calibrated verifier, abstention tiers, risk–coverage | Depth; the repo stands without it |
| **P4** | Artifact + counterfactual replay | |
| **P5** | Latency Pareto, measured shedding, admission control, load test | Only if the frontier is real |

P1 does not start until P0 shows the benchmark is viable.

## 13. Hardware envelope

Developed on an RTX 5070 Laptop (8 GB, `sm_120`, torch 2.11+cu128), 31 GB RAM, no Docker.
Co-resident budget: embedder ~2 GB + cross-encoder reranker ~2 GB + a 4B generator at Q4
~2.5 GB; or a 7B at Q4 with the reranker on CPU. Everything is free and offline after the
first ingest. A reviewer without a GPU can run the full lexical path and the entire P0
failure budget.
