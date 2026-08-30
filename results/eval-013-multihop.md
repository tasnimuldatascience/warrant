# Eval 013 — following the reference instead of widening the beam

> **Superseded serving figures.** This report quotes 21.3 tok/s and a 3-per-minute
> ceiling. Both are wrong: an isolated re-derivation measured 29.2–29.9 tok/s over
> ~205 output tokens, so an answer is 6.6 s and the ceiling is 7.7 req/min. The text
> below is left as it was written, because a results doc is a record of what was
> measured on a day, and editing it to agree with a later number falsifies that
> record. See [eval-010](eval-010-capacity.md).

**Date:** 2026-08-30
**Module:** `retrieve/multihop.py` (new), `tests/test_multihop.py`
**Reproduce:** `make build && make index`, then the scripts under
[How each number was produced](#how-each-number-was-produced). No network, no paid API, no
model in the second hop — it is a regex and two SQL queries.

**Caveat on every number below.** They were taken against the store as it stood at schema v3,
**before** the `corpus/parse.py` citation-anchor rebuild that re-addresses 125 anchors in 45
sections. This module is written against `xref.resolve()` rather than against any anchor
shape, so the rebuild changes which chunk id an admission lands on, not whether the walk
finds it — but the counts here will move and must be re-run after it lands.

---

[eval-006 §1](eval-006-unstated-conditions.md) measured that **77.3% of shipped evidence sets
are missing at least one reference whose target the corpus holds**, 4.14 of them per answer,
and that widening `final_k` from 4 to 16 makes it worse rather than better because each
admitted chunk brings in more references than the set it joins satisfies. There is no `k` at
which the chain closes. The missing paragraph is usually not *similar* to the query —
§630.306(a) opens "Except as provided in paragraph (b) of this section", and (b) is reached by
citation, not by resemblance — so a wider similarity ranking cannot find it.

This is that experiment: after the first hop, parse the retrieved text's outbound references,
resolve them, and admit the targets the evidence set does not already hold, within a slot
budget taken off the tail of the same `final_k`.

**The result in one line.** The second hop closes **more than half of the dangling
references** — mean unsatisfied references per evidence set 3.65 → 1.67 at budget 4 / depth 2
and → 1.13 at budget 8 / depth 4, the share of sets carrying one 70.8% → 42.1% → 27.8% — for a
sufficiency cost of **−0.88 points (95% CI −2.09 to 0.00, 0 items won, 3 lost, p = 0.25)** —
not distinguishable from zero, and never once positive. The benefit is measured on an
intermediate metric; the cost is measured on the outcome metric. **It is therefore shipped
off**, with the numbers beside the flag, on the same discipline as `index.rerank` and
`entail.enabled`.

---

## 1. What it closes

342 held-out test items (of 594; the split is by section — temporal 229, scope 42,
scope-exclusion 42, human 29). Shipped configuration: `final_k: 16`, lexical + dense +
cross-encoder. The first hop is run **once per item** and every configuration below re-cuts
the same ranking, so nothing in these comparisons is retrieval noise.

`budget` is the number of the 16 slots reference-directed candidates may take;
`depth` 2 is one round of expansion from the first-hop set, 3 also follows what that round
admitted.

| budget | depth | sufficiency | 95% CI | sets with ≥1 missing ref | 95% CI | mean missing | admitted |
|---:|---:|---:|:---:|---:|:---:|---:|---:|
| **0 (shipped)** | — | **96.2%** | 92.8–99.4 | **70.8%** | 56.7–84.6 | **3.65** | 0.00 |
| 2 | 2 | 95.9% | 92.4–99.1 | 48.5% | 36.2–61.6 | 2.16 | 1.27 |
| 4 | 2 | 95.3% | 91.8–98.6 | 42.1% | 31.3–54.1 | 1.67 | 2.05 |
| 6 | 2 | 95.3% | 91.8–98.6 | 42.7% | 31.7–55.0 | 1.48 | 2.48 |
| 8 | 2 | 95.3% | 91.8–98.6 | 42.7% | 31.7–55.0 | 1.51 | 2.70 |
| 4 | 3 | 95.3% | 91.8–98.6 | 38.0% | 27.4–49.5 | 1.57 | 2.22 |
| 8 | 3 | 95.3% | 91.8–98.6 | 32.2% | 21.6–42.9 | 1.20 | 3.25 |
| 8 | 4 | 95.3% | 91.8–98.6 | **27.8%** | 17.7–39.2 | **1.13** | 3.35 |
| 10 | 4 | 95.3% | 91.8–98.6 | 26.6% | 17.0–37.5 | 1.13 | 3.70 |

Intervals are section-clustered bootstraps over 2,000 resamples (`eval.stats`). The
`budget = 0` row is the shipped pipeline and is byte-identical to it: the walk hands unfilled
budget back rather than shortening the answer, so an evidence set whose references are already
satisfied retrieves exactly what it retrieves today.

The baseline 70.8% here is the same quantity eval-006 reported as 77.3% over all 594 items;
this is the test half, and the per-bucket rates line up (human 86.2% against 87.5%,
scope-exclusion 95.2% against 91.6%).

Per bucket, at `budget 4, depth 2`:

| bucket | n | sufficiency | sets with a missing ref | mean missing |
|---|---:|---|---|---|
| human | 29 | 75.9% → **69.0%** | 86.2% → 37.9% | 2.66 → 0.66 |
| scope | 42 | 100.0% → 100.0% | 78.6% → 47.6% | 4.40 → 2.05 |
| scope-exclusion | 42 | 100.0% → 100.0% | 95.2% → 64.3% | 4.64 → 2.48 |
| temporal | 229 | 97.4% → 96.9% | 62.9% → 37.6% | 3.45 → 1.59 |

**The entire sufficiency loss is in the human bucket**, which is 29 items and the one the
README already says is too small to rank configurations on. That is the reading that governs
the decision in §7, not the aggregate.

### It reduces the chain and cannot close it

At `budget 8, depth 4` the walk was offered 3.73 dangling targets per item and admitted 3.35 —
the budget is nearly saturated and the residual is not a slot shortage. 27.8% of sets still
carry a missing reference because **the admitted paragraphs make references of their own**,
which is precisely the amplification eval-006 found when `final_k` was widened. Following the
edges does not escape it; it moves the fixed point. Going from depth 2 to 4 at budget 8 buys
42.7% → 27.8% and then flattens.

Worked examples, human bucket, `budget 4, depth 2`:

```
term-probation          + 315.201#c  <- 315.201#a  rank  5  "paragraph (c) of this section"
                        + 315.201#b  <- 315.201#a  rank  5  "paragraph (b) of this section"
military-spouse         + 315.612#b-4-ii   <- 315.612#d-iii-3  rank 6
                        + 315.612#b-4-iii     "paragraph (b)(4)(ii) or (iii) of this section"
rif-performance-credit  + 430.201#c  <- 351.504#a-3  rank 2  "§ 430.201(c)"
                        + 351.203#p1 <- 351.504#a    rank 3  "§ 351.203"
```

Of 702 admissions at `budget 4, depth 2`: **450 section-level** ("as provided in § 351.703"),
221 paragraph-level, 31 same-title CFR. A section-level reference names no paragraph, so the
walk admits the head of the section — one chunk per reference, never the subtree. That rule is
doing most of the work here and is a real approximation: it satisfies the check without
necessarily showing the paragraph the citing text meant.

---

## 2. What it costs, and exactly where

Paired against the shipped configuration on the same items, section-clustered:

| configuration | Δ sufficiency | 95% CI | won | lost | p |
|---|---:|:---:|---:|---:|---:|
| budget 2, depth 2 | −0.29 | −1.07 to 0.00 | 0 | 1 | 1.0 |
| budget 4, depth 2 | −0.88 | −2.09 to 0.00 | 0 | 3 | 0.25 |
| budget 8, depth 3 | −0.88 | −2.09 to 0.00 | 0 | 3 | 0.25 |
| budget 16, depth 3 | −4.09 | — | 0 | 14 | — |

Zero is inside every interval and the sign test never reaches significance. But the sign never
reverses either: **across every budget and depth measured, the second hop did not win a single
item.** A 342-item split cannot separate 95.3% from 96.2%, and reporting "no measurable cost"
would be reporting the resolution of the benchmark rather than a property of the change.

All three lost items are the same mechanism. The gold chunk was at rank 13, 14 or 15 of 16 and
the budget displaced it:

| item | gold | its rank | displaced by |
|---|---|---:|---|
| `890.1604#c@2024-10-24:after` | `890.1604#c` | 14 | four sibling paragraphs of §890.1604 |
| `rif-competitive-area:human` | `351.402#a` | 13 | §351.504(e), §351.606, §351.806, §351.701(b) |
| `severance-computation:human` | `550.707#a` | 15 | §550.103, §532.401, §550.712(a), §550.1302 |

`severance-computation` is the sharpest of the three: every admission was caused by a
reference in `550.707#b-2`, `#b-3`, `#d` or `#b-5` — sibling paragraphs of the very section
whose paragraph (a) got displaced. The walk has no notion that the chapeau of the section it
is walking around might be worth more than the paragraph that section points at.

The obvious mitigation is to protect the citing chunks' own section from displacement. **It is
not implemented**, because it was designed after reading these three items and would then be
fitted to the three items that measure it — the same reason eval-006 shipped `subject_to` at
53% rather than tuning it against its own 32 labels. It is the next measurement, on the dev
split, not this one.

---

## 3. The predicates hold on hop 2, and it matters by 34.5 points

A reference names a **chunk id**, never a version id: the 2017 text of §630.306 cites §630.310,
not any particular version of it. Choosing the version is the as-of predicate's job. So every
hop-2 candidate is drawn by a query carrying the same valid-time, system-time, applicability,
source and authority clauses the first hop ran under, and the resolution set is built from what
*that* query returned — a reference into a superseded version resolves to nothing rather than
being filtered out after the fact.

The counterfactual, at `budget 4, depth 2` over 342 items and 702 admissions:

| where the predicate is applied | superseded chunks admitted | items affected | out-of-scope admissions | items |
|---|---:|---:|---:|---:|
| **inside the query (shipped)** | **0** | **0 / 342** (0.0–1.1) | **0** | **0 / 342** |
| as-of predicate off | 199 | **118 / 342 = 34.5%** (29.7–39.7) | 0 | 0 / 342 |
| applicability off | 0 | 0 / 342 | 9 | 7 / 342 = 2.0% |
| both off | 202 | 121 / 342 = 35.4% | 9 | 7 / 342 |

A second hop that resolved references against the whole corpus and filtered afterwards would
put superseded regulation into **more than a third of all answers** — through a code path the
as-of ablation does not test, in a system whose headline number is that the as-of predicate is
worth +96.1 points on wrong-version rate. The exact count of 199 depends on which version the
lookup happens to take first; that it takes one at all is the point.

The benchmark's own distractor rate stays at **0.0%** at every budget and depth. That is
enforced by construction — the predicates never admit a distractor, so 0% restates the WHERE
clause — but it is worth reporting here for one reason: the second hop is a **new admission
path**, and a new admission path is exactly how a filter that was enforced by construction
stops being.

Two of the three interesting unit tests are these cases (`tests/test_multihop.py`): a
reference followed into a superseded version, and a reference into a part the asker's scope
excludes. The third is §4.

---

## 4. Termination, cycles and depth

Regulations cite in loops: §630.306(a) excepts (b), (b) points on to §630.310, and §630.310(b)
points back at §630.306(a).

- **Hop limit** — `depth`. 1 disables the walk, 2 is one round from the first-hop set, 3 also
  follows what that round admitted. Measured, not assumed: the design note guessed depth 2
  would be enough and it is wrong. Depth 3 buys 42.7% → 32.2% at budget 8, and depth 4 another
  4.4 points, all at **identical sufficiency and the same three lost items**.
- **Dedup** — a visited set of chunk ids, seeded with the whole `final_k` evidence set and
  grown as the round admits. Two chunks citing the same paragraph spend one slot.
- **Cycles** — terminate on the visited set, not on the depth cap. A target the evidence set
  already contains is not a dangling reference, so it is never followed; the loop closes one
  hop after it opens, at any budget and any depth. The corpus is finite and every round admits
  only chunk ids not already visited, so the walk terminates even with the depth cap removed.
- **Coverage semantics** are `xref`'s: a section-level target is covered by any paragraph of
  that section, a paragraph-level one by itself or a descendant, never by an ancestor —
  holding the chapeau of (b) says nothing about what (b)(2) requires.

Determinism: references are followed in (rank of the citing chunk, the drafter's own order
within it) order. No score is invented for a hop-2 candidate, because a reference's relevance
is not a similarity — it is that the text the reader was shown says to go and read it. Same
query, same expansion, same order.

---

## 5. Latency

The second hop is one regex pass over text already in the store, one indexed SQL lookup over
the sections those references name, and one lookup for the admitted rows.

| | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| expansion alone, budget 4 depth 2, warm (n = 1,710) | **0.92 ms** | 1.69 ms | 2.35 ms | 3.38 ms |
| expansion alone, budget 8 depth 3, single cold pass | 1.40 ms | 2.59 ms | — | — |
| expansion alone, budget 10 depth 4, single cold pass | 1.54 ms | 3.08 ms | — | — |

For scale, measured on the same machine in the same run — **contended, with other work
running, so read the deltas and not the absolutes**:

| | p50 | p95 |
|---|---:|---:|
| lexical + dense, `final_k: 16` | 25.9 ms | 38.8 ms |
| + cross-encoder (shipped) | 71.2 ms | 98.1 ms |

So the hop adds about **+0.9 to +1.5 ms p50**: ~4% of the lexical+dense path, ~1.5% of the
shipped path, and ~5% of the 18.4 ms this repo publishes for the uncontended lexical+dense
configuration. Against generation at 21.3 tok/s it is not measurable at all.

Two choices in the implementation are the reason it is that cheap, and both were made for it:
the resolution set is scoped to the ~40 sections the references actually name rather than
being a corpus-wide membership set (which would have to be cached per `(as_of, scope)` pair,
and serving sees a different `as_of` per request, so the cache would miss on exactly the path
it exists for); and each round re-reads only what the previous round admitted, never the
frontier it came from.

---

## 6. Trace and attribution

Every admission is recorded on the trace as `trace.admissions`, carrying the version id, the
depth, the citing chunk, its rank, and **the reference phrase as the drafter wrote it**. A
`Candidate` can carry an id, a score and a rank, and cannot carry the reason — so without this
a hop-2 chunk in the evidence set is indistinguishable from a hop-1 chunk that ranked badly,
and the failure budget cannot charge anything to the stage. `trace.timings["expanded"]` carries
the cost.

The stage's ranking is also offered to `Trace.record("expanded", …)`, which is refused today
because `hybrid.STAGES` does not carry the name — see the wiring note in §7. `trace.admissions`
is the record until it does.

---

## 7. The recommendation

**Ship it off.** `retrieve.hop_budget: 0` in `configs/default.yaml`, with these
numbers written beside the flag, exactly as `index.rerank` and `entail.enabled` are.

The case for turning it on is genuine: 70.8% → 27.8% of evidence sets carrying an unsatisfied
reference, 3.65 → 1.13 per answer, for 1.5 ms and no measurable sufficiency loss. The case
against it is the one that decides:

1. **The benefit is on an intermediate metric and the cost is on the outcome metric.** Nothing
   in this repository has yet shown that closing a dangling reference produces a better
   *answer*. It cannot: `data/traces.sqlite3` holds no generations (eval-006 §5), so the
   experiment that would settle it — generate with and without the hop, score for the dropped
   condition eval-006 §3 detects — has not been run.
2. **Sufficiency never went up.** 0 items won across every configuration measured. The
   aggregate says "not distinguishable from zero"; the sign says "always the same direction".
3. **The loss lands on the human bucket**, 75.9% → 69.0%, which is the bucket that represents
   real questions and the one that is too small to settle anything.

If it is turned on anyway, `hop_budget: 8, hop_depth: 3` is the operating point: it is where
the dangling rate has taken most of its fall (32.2%) and where more budget stops buying
anything, and it costs exactly what `budget 4` costs.

### Wiring, for whoever owns the files

`config.py` — in `RetrieveConfig`:

```python
    #: Slots of final_k that reference-directed second-hop candidates may take. 0 disables
    #: the hop. See results/eval-013: at 8/depth 3 the share of evidence sets carrying an
    #: unsatisfied reference falls 70.8% -> 32.2% for -0.88 points of sufficiency (CI -2.09
    #: to 0.00, 0 won / 3 lost) -- a benefit on an intermediate metric against a cost on the
    #: outcome metric, so it is off until generated answers can settle it.
    hop_budget: int = 0
    hop_depth: int = 2
```

`configs/default.yaml` — under `retrieve:`, `hop_budget: 0` and `hop_depth: 2` with the same
note.

`cli.py` — in `_retriever`, swap the constructor when the budget is non-zero:

```python
    cls = MultiHopRetriever if cfg.retrieve.hop_budget > 0 else Retriever
    extra = ({"hop_budget": cfg.retrieve.hop_budget, "hop_depth": cfg.retrieve.hop_depth}
             if cfg.retrieve.hop_budget > 0 else {})
    return cls(store=store, dense_index=index, reranker=reranker, ..., **extra)
```

`MultiHopRetriever` subclasses `Retriever` and overrides only `retrieve`, so `eval.run.score`,
the autopsy, the API and replay need no change at all.

`hybrid.py` — optional, and the only edit that makes the stage first-class. Add `"expanded"`
to `STAGES` between `"reranked"` and `"final"`, add the matching keyword to `Trace.__init__`
and to the tuple it zips (the zip is `strict=True`, so both have to move together), and add
the derived property. `multihop.Expansion.record` already writes the stage and swallows the
`KeyError` it currently gets, so it starts working with no change here. Do **not** name the
new `Trace` attribute `expanded` — `multihop` writes provenance to `trace.admissions` for
exactly this reason.

---

## 8. What is not measured here

- **Whether a closed reference produces a better answer.** The whole of §1 is an intermediate
  metric. See §7.1.
- **Anchors move under this.** Every number predates the `corpus/parse.py` roman-numeral
  rebuild (125 anchors, 45 sections). §890.301 currently addresses paragraph (n) as
  `890.301#ii-7-n`; a reference to it resolves through `xref.resolve`'s walk today and will
  resolve exactly after the rebuild, but the counts will move.
- **The section-level rule.** 64% of admissions come from "as provided in § X", and the walk
  answers those with the head of that section. Whether that is the paragraph the citing text
  meant is not measured, and could be: the human items have hand-written evidence sets.
- **The displacement rule is untuned and known to be wrong in at least one shape** (§2). No
  guard is shipped, for the reason given there.
- **`outside` and `unscoped` references are untouched.** 2,047 and 2,962 of them at
  `final_k: 16` in eval-006. The first is the honest cost of a corpus that stops at 5 CFR
  chapter I and no retrieval change can close it; the second names no single chunk.
- **Nothing here runs on the generator.** `context_chunks` defaults to `final_k`, so the
  admitted paragraphs would reach the prompt, but no generation was scored.
- **Latency was measured on a contended machine.** The deltas are stable; the absolutes are
  not, and the resume note asking for a quiet re-measurement still applies.

---

## How each number was produced

Everything runs offline against a read-only copy of `data/warrant.sqlite3` (13,145 chunk
versions, schema v3, horizon `2026-08-25`). The benchmark is `mine_all` at
`horizon = MAX(valid_from)`, filtered to `split == "test"` — 342 of 594 items.

| § | what | how |
|---|---|---|
| 1 | budget × depth grid | first hop run once per item at its own `as_of` and scope; `ReferenceExpander.expand` re-cuts the same `reranked` ranking at each setting. Sufficiency by `BenchItem.is_satisfied_by`; intervals `eval.stats.cluster_bootstrap_ci`, keyed on `section_id` |
| 1 | dangling rate | `xref.dangling_references(evidence, in_corpus=nameable_ids(store.as_of(item.as_of)))`, counting `status == "missing"` — corpus-wide and scope-free, the same definition eval-006 §1 used, so before and after are comparable |
| 2 | paired deltas | `eval.stats.paired_delta`, section-clustered, 2,000 resamples, with an exact McNemar sign test |
| 2 | the three lost items | the gold's rank in the first-hop `reranked` list against the displaced tail |
| 3 | predicate counterfactual | the same walk with `temporal=False` and/or `exclude_parts=()`; an admission is "superseded" when its `version_id` is not in force on the item's `as_of` |
| 5 | expansion latency | `perf_counter` around `expand` alone, 5 passes over the 342 items (n = 1,710); base retrieval from `trace.timings["total"]` on the same run |
