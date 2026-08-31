# What this deliberately does not do

A system is judged as much by what it refuses to build as by what it ships. Everything below
is absent on purpose, and each entry says what would change the decision. Where an absence is
a gap rather than a choice, it says that instead — the two are not the same thing, and
blurring them is how a design document becomes marketing.

---

## No chat interface

Single-shot: question, as-of date, scope in; evidence and cited claims out.

**Why.** A conversation that carries state across an as-of change is a wrong-version bug
waiting to happen. Ask *"by when must restored annual leave be scheduled?"* as of 2019, then
follow up with *"and what's the exception?"* — carry 2019 forward silently and the reader
does not know which law they are reading; re-resolve to today and two turns answer from
different regulations without saying so. That is precisely the failure the as-of predicate
prevents, measured at **+96.1 points of wrong-version rate, 220 wins / 0 losses, p=1.2e-66**.
A chat window makes it the default behaviour.

**What exists instead.** Follow-ups scoped to the evidence already retrieved: the as-of and
the evidence set are pinned for the exchange and shown on every turn, a follow-up that needs
evidence the set does not hold says so rather than silently re-retrieving, and changing the
date starts a new exchange rather than reinterpreting the old one.

**What would change it.** A corpus where the answer does not depend on when you ask.

---

## No memory

No conversation history, no user profile, no cross-session state. Every exchange is
independent.

**Why.** Memory in a legal-lookup system is a correctness liability before it is a
convenience. "You asked about GS pay last week" is a scope assumption; applying it silently
is how a Federal Wage System employee gets an answer from part 531 when part 532 governs
them. Scope is an *input*, visible on every request, and the applicability predicate is
scored on getting it right — all three external baselines cite an out-of-scope part on
**42 of 42** scope-exclusion items because none of them has one.

**What would change it.** A product where the same person asks many questions under a stable
scope, and where the cost of a silently wrong assumption is low. Neither holds here.

---

## No model routing

One encoder, one reranker, one generator. Nothing routes by query type, difficulty or cost.

**Why.** Routing needs a spread of difficulty and cost worth arbitrating, and this corpus does
not have one. Every query is the same shape against 13,212 chunks of one CFR title, retrieval
is 18.4 ms flat, and the measured spread across buckets is a property of the *questions*, not
of a model choice. A router here would be tuning a knob nothing is attached to.

**The nearest thing that does exist,** and is measured: the abstention policy decides whether
to answer at all — 74.0% coverage at 1.35% selective risk against 4.33% for always answering
([eval-005](../results/eval-005-abstention.md)). It ships **off**, because the learned
combiner does not beat a threshold on the single top-1 fusion score (AURC delta +0.0019, CI
−0.0017 to +0.0070).

**What would change it.** Two generators with genuinely different cost/quality points, and a
query distribution where the cheap one is sufficient often enough to matter. Both are corpus
decisions, not architecture ones.

---

## No agents

The pipeline is a fixed sequence, and the one multi-step behaviour — following the
cross-references a retrieved paragraph makes — is a **deterministic graph walk**, not an agent
loop.

**Why.** The edges are written in the text by the drafter. "As authorized under §630.309" is
not a decision to be reasoned about; it is an address. Traversing it needs no inference, and
putting a model in that position adds latency, non-determinism and a new failure mode to a
step that already has a correct answer.

That walk is measured: at budget 8, depth 3 it takes evidence sets carrying an unsatisfied
reference from **70.8% to 32.2%** for 1.4 ms
([eval-013](../results/eval-013-multihop.md)) — and it ships **off**, because sufficiency
costs −0.88 points (CI −2.09 to 0.00) and was never once positive at any budget or depth. The
benefit is on an intermediate metric and the cost is on the outcome metric.

**What would change it.** A task where the next step genuinely is not determined by the
previous one. Reference-following is not that task.

---

## Governance: half covered, and here is which half

**Covered.** Provenance is first-class. Five source tiers with an ordered `authority` column
that retrieval filters and sorts on; a `kind` column recording whether text came from parsed
XML or OCR, because a citation to a misread digit is weaker evidence than a citation to
markup and a verifier cannot weigh that unless ingestion wrote it down. The store is
bitemporal and append-only, so what the system believed on any past date is recoverable and a
past answer stays reproducible. Every claim cites by evidence id, validated against the set
actually retrieved. Every request leaves a replayable trace.

**Not covered.** No access control, no authentication, no per-tenant isolation. No data
retention or deletion policy. No PII handling — the corpus is published federal regulation and
contains none, but a query log does, and there is no policy for it. No model card. No
dependency or supply-chain attestation beyond pinned model revisions and a digest-pinned base
image.

**This is a gap, not a decision.** It is absent because the corpus is public law and nothing
here has served a real user, not because it was reasoned away. A deployment carrying private
documents needs all of it before anything else in this repository matters.

---

## Not a security system

The corpus is published law. Nothing in it is confidential and nothing can leak.

An earlier version of this project imported a security framing from a different problem and
claimed a "zero leak rate". That claim was withdrawn and the vocabulary renamed to
**applicability** — whether a rule governs the asker — because that is what is actually being
measured. Citing a rule that does not apply to you is a correctness failure, not a breach.

The guardrails that do exist defend serving integrity, and they are measured: a 2,600-token
repetition cost **23,004 ms** and is refused in 0.02 ms; an unbalanced quote was a 500 and is
now a 3 ms answer; a Cyrillic homoglyph matched nothing and now matches 100 chunks
([eval-008](../results/eval-008-serving-guardrails.md)).

---

## Never served real traffic

Every capacity number in this repository came from a synthetic load generator against
localhost. The load harness runs open-loop as well as closed-loop, because a closed-loop
harness with fixed workers sends *less* traffic to a slowing server and hides the collapse it
is measuring — but a harness is still not users.

Known and unfixed: the read path degrades before the expensive path does. Under 100 concurrent
asks a retrieval-only probe completed **0 of 64** requests before `/api/ask` was made async
([eval-010](../results/eval-010-capacity.md)).

---

## One CFR title

26 parts, 13,212 chunk versions, 2017 to 2026. The scale study extrapolates to 500k with a
*synthetic* corpus matched to the real one's statistical shape, and says plainly that this
bounds mechanism cost — index size, query time, memory — and says nothing about retrieval
*quality* at scale, which needs real text and real questions
([eval-011](../results/eval-011-scale.md)).

The first thing to break is the lexical stage at roughly 3×, and the fix is measured and
waiting behind a config flag it has not earned yet
([eval-014](../results/eval-014-query-terms.md)).
