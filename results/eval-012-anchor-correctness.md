# Eval 012 — the citation address, measured

**Date:** 2026-08-30
**Module:** `src/warrant/corpus/parse.py` (designator stack rewritten)
**Reproduce:** offline, against the cached snapshots in `data/ecfr/`. `python -m pytest
tests/test_parse.py` runs both instruments over the whole cache as gates; the method behind
each table is under [How each number was produced](#how-each-number-was-produced). No network,
no model, no paid API.

Every citation in this system is an evidence id. `ARCHITECTURE.md` §5 explains why the
generator is not asked for character offsets, and that choice only pays if the ids are right.
A wrong-but-well-formed anchor is worse than a missing one: it looks checkable and is not.

> `890.301#ii-7-n` is what a citation to 5 CFR 890.301(n) rendered as. It addresses a
> paragraph that does not exist in the law, and `890.301#n` addressed nothing.

---

## 1. What was wrong

`(i)` is the ninth lowercase letter and the first roman numeral, and the CFR uses both at
different depths. The parser has to decide from context — what level it is currently at, and
what preceded — because the token cannot decide. The old stack decided by scanning for the
first level the token continued and taking it, which loses at the first ambiguity:

```
5 CFR 890.301, as stored          as the regulation reads
  h                                 (h)
  h-1                               (h)(1)
  i          <- (h)(1)(i) read as   (h)(1)(i)
  ii            the letter after    (h)(1)(ii)
  ii-2          (h), which it       (h)(2)
  ...           equally is          ...
  ii-7                              (h)(7)
  ii-7-i                            (i)          <- top level, and this time it *is* the letter
  ...                               ...
  ii-7-j                            (j)
  ii-7-n                            (n)
```

The top level was lost at (h)(1) and never recovered, so the last seven subsections of a
52-paragraph FEHB section were filed two levels beneath a roman numeral that is not in the
regulation at all.

**eCFR gives no nesting to fall back on.** `sources/usc.py` found that where USLM expresses
nesting, nesting outranks sequence continuity — its `_place` defers to the element tree and
agreed with OLRC's own identifiers 20,217 times out of 20,217. eCFR is the opposite case, and
this was checked rather than assumed: across all 226 cached snapshots there are 122,467 `<P>`
elements inside a `DIV8` section and **122,273 of them — 99.84% — are direct children of it**.
The 194 that are not sit inside an `EXTRACT` or a table cell, which is a quotation boundary
and not a paragraph level. The only attributes any of them carry are `ecfr-split-paragraph`
(7) and `class` (22); no element carries a designator, a level, or a path. The hierarchy
exists only in the designators themselves, so it has to be inferred, and the only question is
how well.

That is the reported defect. Four more surfaced once there was an instrument pointed at the
problem, and all five produce the same pathology — an address that parses as a citation and is
not one:

| | Defect | What it minted |
|---|---|---|
| 1 | roman/letter decided greedily and locally | `890.301#ii-7-n` for (n); `630.1503#c-i` for (c)(1) |
| 2 | a level's kind read from its absolute depth | `330.602#1-1-i` for a paragraph the section calls (1)(i) |
| 3 | a list read as a child of the last designator, across the undesignated headword it belongs to | `551.104#6-1` and `630.201#b-7-1`, in definitions sections that have no such paragraphs |
| 4 | letters past (z) counted in base 26 | §330.609 runs (a)…(z), (aa)…(ee); (cc) stopped being (bb)'s successor, giving `330.609#bb-1-cc` |
| 5 | only one level allowed to be buried in a chapeau | §315.612(e) buries both (1) and (i) before the standalone (A), which became `315.612#e-A` |

---

## 2. The instruments

The 1.26% in the bug report — anchors whose *root* is a roman numeral — is a lower bound, and
a check of that shape also fires on addresses that are right: §330.609 and §330.707 number
their subsections past (u), so `330.609#v` and `330.707#x` read as roman numerals and are
letters. Two better instruments, both offline, both independent of the fix:

**A. Structural well-formedness.** The CFR numbers its levels in a fixed chain: (a) holds (1)
holds (i) holds (A), and below that (1) and (i) again. An anchor whose component *kinds*
cannot form such a chain is wrong without anyone needing to know the right answer. `ii-7-j` is
a letter under a number under a roman numeral, which the chain does not allow. This is still a
lower bound — an anchor can be well-formed and address the wrong paragraph — but it is
exhaustive over the corpus and needs no ground truth.

**B. The drafters' own paragraph references.** The regulation cites itself: *"as discussed in
paragraph (g)(3) of this section"*. Those are full addresses written by the person who wrote
the numbering, and they are not derived from this parser. 1,517 of them in the in-force
corpus, 17,431 across all snapshots. A reference **resolves** if its address is an anchor, or
is a hyphen-prefix of one — the prefix case being a paragraph whose chapeau ran its first
child into its own sentence, so the subtree exists and the bare address does not.

Elisions are expanded before matching: *"paragraphs (a)(1) through (4)"* names (a)(4), not a
top-level (4), and *"paragraph (a)(5) or (b)"* names (b), not (a)(b). `verify/xref.py` had the
same defect ([eval-006 §4](eval-006-unstated-conditions.md)); charging it to the parser here
would be charging the parser for a fault in the instrument.

---

## 3. Before and after

Newest cached snapshot of each of the 26 parts — the in-force corpus, 1,310 sections:

| | before | after |
|---|---:|---:|
| paragraph anchors | 9,955 | 9,955 |
| **A.** malformed | **604 (6.07%)** | **0** |
| sections containing one | 66 | 0 |
| **B.** references | 1,517 | 1,517 |
| resolve to that exact anchor | 1,318 (86.9%) | **1,356 (89.4%)** |
| resolve to an anchor beneath it | 70 (4.6%) | 86 (5.7%) |
| do not resolve | **129 (8.5%)** | **75 (4.9%)** |

All 226 cached snapshots, 16,024 section-versions:

| | before | after |
|---|---:|---:|
| paragraph anchors | 127,402 | 127,402 |
| **A.** malformed | **6,845 (5.37%)** | **0** |
| section-versions containing one | 767 | 0 |
| **B.** references | 17,431 | 17,431 |
| resolve to that exact anchor | 14,928 (85.6%) | **15,448 (88.6%)** |
| resolve to an anchor beneath it | 980 (5.6%) | 1,115 (6.4%) |
| do not resolve | **1,523 (8.7%)** | **868 (5.0%)** |

**So the true error rate was 6.07% of in-force anchors, not 1.26%** — five times the crude
count, and part of the crude count was not error at all. Instrument A now reads zero and is
kept as a test, so it reads zero by assertion rather than by luck.

Two hand-checked sections, against the printed regulation:

```
890.301  (52)  a b c d e e-1-i e-1-ii e-2 f f-2 f-3 f-4-i f-4-ii f-5 g g-2 g-3-i
               g-3-ii g-4 g-4-i g-4-ii h h-1 h-1-i h-1-ii h-2 h-3 h-4-i h-4-ii
               h-5 h-6 h-7 i i-1 i-2 i-3 i-4 i-4-i i-4-ii i-4-iii i-4-iv i-4-v
               i-5 i-6 i-7 j k l m n o p
630.1503 (45)  a a-1 a-2 a-2-i a-2-ii a-2-ii-A a-2-ii-B b b-1 b-1-i b-1-ii
               b-1-iii b-1-iv b-2 b-2-i b-2-ii b-2-iii b-2-iv b-3 c c-1-i c-1-ii
               c-1-iii c-1-iv c-2 c-3 c-4 c-5 c-5-i c-5-ii d d-2 d-3 d-4 e e-1
               e-2 e-3 e-3-i e-3-ii e-3-iii e-3-iv e-3-v f g
```

`630.1503#c-i` — a roman numeral read as a letter — is now `630.1503#c-1-i`, and the store
holds no `#c-i`.

### The residual, and what it is

75 references still do not resolve in the in-force corpus. They are not one thing, and all 75
were classified:

| | count | |
|---|---:|---|
| chapeau ran its first child inline | 48 | §351.504 is written "(a) *Ratings used.* (1) Only ratings of record …", so (a)(1) has no `<P>` of its own and no address of its own. `resolve()` already walks up to the deepest address the store holds ([eval-006 §4](eval-006-unstated-conditions.md)); splitting the paragraph is a chunker change, not a parser one, and would change what a citation quotes. |
| a reference relative to its own list | 20 | "paragraphs (i) and (ii) of this section", written from inside the list it is pointing at, gives the instrument no parent to attach them to. Not a parser fault and not fixable in the parser. |
| inspected one by one | 7 | 3 (§300.201, §317.503, §630.1006) are the inline-chapeau class again, missed by the classifier because the parent's designator run is far into a long paragraph. 2 are the CFR's own slips: §890.304 says "paragraph (d)(iii)" for a paragraph it numbers (d)(1)(iii), and §315.701 cites a "(b)(2)" that (b) does not have. 2 are reference-extractor artefacts in §550.1104. |

No designator-stack failure has been identified in the residual. Instrument A reading zero is
the stronger of the two statements, since it is exhaustive over every anchor rather than over
the paragraphs the drafters happened to cite.

---

## 4. How the fix works

Kind — alpha, digit, roman, upper — is what separates the ninth letter from the first roman
numeral, and a level's kind is fixed by **its parent's** kind, not by its depth: (a) holds
(1) holds (i) holds (A). Keyed on the parent because a section is free to start partway down
the chain, which is what defect 2 above was.

That is still not enough, and this is the part worth stating plainly: **after `(h)(1)`, a
`(i)` is equally the roman numeral opening (h)(1)(i) and the letter opening a top-level (i),
and no amount of care at that paragraph decides it.** §890.301 contains both readings nine
paragraphs apart. What separates them is what comes *next* — `(ii)` continues a roman run,
`(1)` cannot follow a roman numeral without a level being skipped — so the decision has to be
deferred.

`_placements` therefore enumerates every level a designator could sit at with a cost, and
`_resolve` runs a beam over the section's whole designator stream and takes the cheapest
reading of the sequence. The costs are an ordering, not tuned thresholds: continuing an open
level and opening the next one with its own first designator cost nothing; a chapeau that ran
its first child inline beats a level going unwritten, which beats a designator missing from a
level, which beats a level numbered off the chain, which beats a level restarting under one
parent.

An undesignated paragraph is not a passenger in that stream. It is offered as an alternative
root, which is defect 3: a definitions section is a run of headwords each carrying its own
(1), (2), (3), and reading those as children of the last designator seen is where
`551.104#6-1` came from. A list that *continues* across an undesignated paragraph — the flush
sentence that closes a list — keeps its addresses; only a list that *starts* across one is
re-rooted. Those numbered items have no CFR designator path at all, so they are addressed
under the paragraph they belong to: `551.104#p19-1`, positional and honest, rather than
`551.104#6-1`, which reads like a citation.

**Determinism.** The beam is ordered by cost and then by the stack itself, so dict iteration
order cannot reach the output. SHA-256 over every anchor in the cache is identical at
`PYTHONHASHSEED` 0, 7 and 999.

**Beam width.** Over all 226 snapshots, widths 4, 8, 16 and 32 give byte-identical anchors and
widths 2 and 3 do not. Shipped at 8 — double the width that converged. Parsing the whole cache
costs 11.6 s against 6.2 s for the greedy push it replaces, five seconds spread over a build
that also reads 226 XML files and writes 13,212 rows.

---

## 5. What a rebuild changes

Anchors are chunk ids, so fixing them changes ids. Measured by parsing every cached snapshot
under both parsers and diffing position by position — the paragraph *texts* are untouched and
the anchor counts are identical, so every difference is a rename:

| | in-force | all snapshots |
|---|---:|---:|
| anchors | 9,955 | 127,402 |
| unchanged | 9,145 | 118,219 |
| **renamed** | **810 (8.14%)** | **9,183 (7.21%)** |
| distinct sections touched | 83, in 20 parts | 86 |

Distinct chunk ids: **851 disappear, 858 appear.** By shape, in force:

| | |
|---:|---|
| 380 | re-rooted at the undesignated paragraph they belong to (`6-1` → `p19-1`) |
| 225 | a level closed that should have closed (`ii-7-n` → `n`, `c-ii-C-2` → `c-2`) |
| 185 | a level filled in that the markup buried (`c-i` → `c-1-i`, `e-A` → `e-1-i-A`) |
| 20 | same depth, different address |

A rebuild into a scratch store — a session scratchpad path, never `data/warrant.sqlite3` —
completes offline and clean, twice, with the same result: 13,212 chunk versions over 26 parts
and 81 distinct snapshots. **No anchor in the corpus needs the collision suffix.** `550.703#a`
matched four paragraphs before designators were tracked at all; over all 127,402 anchors the
`.2` backstop now stays unused, which a test asserts.

### Recorded artifacts that go stale

`data/warrant.sqlite3` was rebuilt during this session by another hand — it now holds
`890.301#n` and `315.612#c-1-i`, addresses only this parser produces — so the list below is
already stale rather than prospectively stale. None of it was re-recorded here.

| Artifact | What is stale |
|---|---|
| `benchmarks/entailment.yaml` | pair `gen-17` cited `315.612#d-iii-4`, which the rebuild renamed to `315.612#d-4` — the same paragraph, same claim text. `tests/test_entailment_bench.py` caught it, which is the behaviour that test was written for, and its owner re-anchored the pair while this was being measured. **Already fixed; listed because it is the shape of what else will break.** |
| **`results/eval-007-entailment.md`** | still quotes `315.612#d-iii-4` in two adjudication rows. |
| **`results/eval-006-unstated-conditions.md`** | quotes 8 stale ids, including the two it correctly attributed to this parser. §4's resolution table (94.7% exact, 4.6% ancestor) was measured *against these anchors* and has to be re-measured; it should improve, since two of the four defects it lists as chunker artefacts are now fixed at the source. |
| **`results/eval-floor.json`** | the recorded quality floor is a bootstrap lower bound over mined items whose evidence ids come from the store. The item sets change, so the floor is not comparable across the rebuild and must be re-recorded before `make gate` means anything. |
| **`results/failure-budget.json`** and `results/eval-002` | same reason: every row is keyed on evidence ids. |
| **`data/traces.sqlite3`** | 5 of 3,709 candidate rows across 990 traces name 2 stale ids — `550.1203#i` → `550.1203#h-6-i` and `630.301#i` → `630.301#h-1-i`, both the roman/letter bug. Artifact replay of those traces will show a candidate the store no longer holds. Counterfactual replay re-runs retrieval and is unaffected. |
| `benchmarks/human.yaml` | **not stale.** All 53 hand-written evidence ids survive the rename. |

`ARCHITECTURE.md` §9 states the unambiguous-address invariant as recovered "by designator
sequence continuity". That is now half the story — sequence continuity alone is what produced
`890.301#ii-7-n` — and the sentence needs the level-kind chain and the deferred decision added
to it. That file is not this change's to edit.

---

## How each number was produced

| § | what | how |
|---|---|---|
| 1 | eCFR carries no paragraph nesting | every `DIV8[@TYPE="SECTION"]` in the 226 cached snapshots; `P` elements counted by depth below the section, and their attribute names tallied |
| 2, 3 | instrument A | every anchor's components classified by kind; an anchor passes if some contiguous walk of (a)→(1)→(i)→(A)→(1)→(i) admits all of them. Positional fallbacks (`p3`, `t1`) are not charged |
| 2, 3 | instrument B | `paragraphs? … of this section` spans over each section's own text, elisions expanded, matched against that section's anchor set and its prefixes |
| 3 | hand check | §890.301 and §630.1503 read against the eCFR text of the 2026-08-26 snapshot |
| 4 | determinism | SHA-256 over `section#anchor` for all 226 snapshots at three `PYTHONHASHSEED` values |
| 4 | beam width | the same digest at widths 2, 3, 4, 8, 16, 32 |
| 5 | rename table | position-by-position diff of both parsers over all 226 snapshots |
| 5 | stale artifacts | the disappearing-id set matched against `benchmarks/`, `results/`, `configs/`, `docs/` and every text column of `data/traces.sqlite3` |
