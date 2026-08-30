# Eval 009 — dating the statute layer, and a source that was invisible to every dated query

**Date:** 2026-08-30
**Module:** `src/warrant/sources/usc.py`
**Reproduce:** `warrant corpus ingest --source usc` with `sources.usc.enabled: true`, against a
store built by `make build`. The measurements below read release point 119-102 of title 5
(`xml_usc05@119-102.zip`, 2.9 MB, cached) and touch no network.

## The failure

`warrant corpus ingest --source usc` reported 38 units of 5 U.S.C. 6304 at authority 1 and
exited zero. The store agreed:

```
source  valid_from range              open   n
ecfr    2017-01-01 .. 2026-08-25      9961   13145
usc     2026-07-12 .. 2026-07-12        38      38
```

`2026-07-12` is the date of Public Law 119-102, the last statute folded into the OLRC
**edition** this text was read from. It is not the date 5 U.S.C. 6304 became the law — that
was 1966 — and it is not the date it last changed, which was 2010. The as-of predicate is
`valid_from <= :v` pushed into SQL (`Store.candidate_ids`, `Store.search`), so the whole
source sat above the top of the corpus window:

| as_of | admitted, `sources=["usc"]` |
|---|---:|
| 2017-01-01 | **0** |
| 2020-06-30 | **0** |
| 2024-01-01 | **0** |
| 2026-08-25 | 38 |

The eCFR corpus runs 2017-01-01 (`corpus.history_floor`) to 2026-08-25. Every dated question
anyone would ask of this system returned nothing from the statute layer, and an empty result
is indistinguishable from a question with no statutory answer. The ingest reported success at
every step.

The module knew the hazard. Its own comment said a statute's `valid_from` is an edition date
and warned that a reader who treats it as an amendment date "will conclude 5 U.S.C. 6304
changed in July 2026". That was correct, and it was written next to code that then handed
that date to the predicate anyway.

## The fix, and what it does and does not claim

USLM sections carry a `<sourceCredit>` — the enactment-and-amendment chain — and OLRC marks
each date inside it up machine-readably:

```xml
<sourceCredit>(<ref href="/us/pl/89/554">Pub. L. 89–554</ref>,
  <date date="1966-09-06">Sept. 6, 1966</date>, <ref href="/us/stat/80/519">80 Stat. 519</ref>;
  … <ref href="/us/pl/111/282/s2/b">Pub. L. 111–282, § 2(b)</ref>,
  <date date="2010-10-15">Oct. 15, 2010</date>, <ref href="/us/stat/124/3038">124 Stat. 3038</ref>.)
</sourceCredit>
```

`valid_from` is now the **latest** date in that chain. The claim is *in force at least since*:
the text as it currently reads has stood since Congress last amended the section. It is a
lower bound and it is deliberately loose in the safe direction —

- It does **not** claim the text was published in this exact form that day. A credit records
  that the section was amended, not which of its twenty subsections the amendment touched.
- Every unit of a section gets the section's bound, so a subsection untouched since 1966 in a
  section amended in 2010 reads `2010-10-15`. That understates how long that paragraph has
  held, which is the direction that costs recall and never invents currency.
- `valid_to` stays open. A release point is a snapshot of the Code as it stands; it carries no
  forward history, so this source can never close an interval. A section repealed in a later
  edition will not be closed by re-ingesting a newer one.
- The lower bound is what the predicate needs. `valid_from <= :v` asks whether the text was
  already law on the asking date; the exact date is not recoverable from an edition at all.

The edition remains on every row. `meta["release_point"]` (`119-102`),
`meta["release_point_date"]` and `meta["snapshot"]` — which `corpus/ingest.py` writes into each
chunk's `source_snapshot` — say which OLRC file a quotation can be checked against.
`meta["valid_from_basis"] = "source_credit"` states in the store itself what the column means
for these rows, because it does **not** mean what it means for eCFR rows, where `valid_from`
is the date an amendment took effect and `valid_to` closes at the next one.

## Coverage across the whole title

Not one section. All 1,163 operative sections of title 5 at 119-102 (`<section>` elements not
nested in a note or `<quotedContent>`):

| | sections |
|---|---:|
| operative sections in title 5 | 1,163 |
| carrying a `<sourceCredit>` | 1,129 |
| **yielding at least one date** | **1,129** |
| carrying no `<sourceCredit>` | 34 |

**100.0% of ingestible sections (1,129/1,129) yield a date.** The 34 without a credit are
exactly the 27 repealed, 3 omitted, 3 renumbered and 1 transferred sections. Every one of them
parses to **zero units** and was already skipped for having no operative text, so the undated
policy below never fires on this title: the set of sections it would drop and the set already
dropped for emptiness are the same set.

Two independent paths read the credit, and the second is a backstop rather than a second
opinion:

| | sections |
|---|---:|
| `<date date="…">` attribute path answered | 1,129 |
| prose path (`"Oct. 15, 2010"`) answered | 1,129 |
| the two disagreed on the latest date | **0** |
| only one of the two answered | 0 |

The prose path does no work at all on title 5 — the attribute path carries every section — and
it exists for titles whose conversion predates the `<date>` markup. It is exercised by test,
not by this corpus, and this table is what says so.

No credit date is later than the edition it came from: the latest across the title is
2026-05-29 against an edition of 2026-07-12. Sections carry exactly one `<sourceCredit>` each
(0 with more than one), and no `<date>` inside a credit is nested in a note.

### When title 5 was last amended

Latest credit date per section, 1,129 sections:

| last amended | sections | share |
|---|---:|---:|
| before 2017-01-01 (the whole corpus window) | **911** | 80.7% |
| 2017-01-01 or later | 218 | 19.3% |
| still as enacted, 1966-09-06 | 110 | 9.7% |

That first row is the point. For four sections in five the honest claim is *in force for the
entire eCFR point-in-time window* — a **stronger** statement than the edition date made, not a
weaker one. The old behaviour did not merely mis-date those sections; it inverted them, taking
the most settled law in the title and marking it as the newest thing in the store.

## Policy for a credit that cannot be dated

**A section whose credit yields no date is not ingested.** It is appended to
`UscSource.undated`, logged by citation at WARNING, and counted in the run's summary line.

The two alternatives are both wrong and, worse, both silent:

- **Fall back to the edition date.** This is the bug wearing a new hat. The section vanishes
  from every dated query inside the corpus window and the ingest still reports success.
- **Invent an earlier bound** — the history floor, or the section's enactment date. This trades
  silence for a wrong-version answer, which is the failure this system exists to measure: on
  the held-out split, dropping the as-of predicate moves the wrong-version rate by **+96.1
  points** (92.8–99.5, 220/0). Text that cannot be placed in time must not be allowed to answer
  as though it were current.

Dropping the section is also a loss, but it is a *counted* one. The measurement above is what
makes that acceptable: it costs 0 sections of 1,129 on title 5, so the fallback is not quietly
doing the work the primary path is being credited for.

## Before and after, end to end

A scratch copy of `data/warrant.sqlite3` (13,145 chunk versions), `usc` rows cleared, then
ingested twice from the same cached release point — once with `valid_from` forced to the
edition date, which is what the old code did, and once as the module now behaves. Admitted
counts from `Store.candidate_ids(valid_date=…, sources=["usc"])`, the same call retrieval makes.

**5 U.S.C. 6304 alone** — the section from the live report. 38 units both times; ingest
identical; `undated` 0.

| as_of | before | after |
|---|---:|---:|
| 2017-01-01 | 0 | **38** |
| 2020-06-30 | 0 | **38** |
| 2024-01-01 | **0** | **38** |
| 2026-08-25 | 38 | 38 |

`valid_from` moves from `2026-07-12 .. 2026-07-12` to `2010-10-15`. A lexical search for
"annual leave accumulates" restricted to `sources=["usc"]` at `as_of="2024-01-01"` returns 0
hits before and 38 after.

**Chapter 63, the configured slice** — the leave statute behind 5 CFR 630. 55 sections, 619
units, 0 undated, `valid_from` spanning 1966-09-06 to 2024-12-23; 48 of the 55 were last
amended before the corpus floor.

| as_of | before | after |
|---|---:|---:|
| 2017-01-01 | 0 | **487** |
| 2024-01-01 | 0 | **607** |
| 2026-08-25 | 619 | 619 |

The after column is not flat, and that is the second thing worth seeing. 487 of 619 units at
the 2017 floor, 607 at 2024, 619 at the top: the seven sections amended after the floor
correctly do not appear before the date Congress amended them. The statute layer is now
point-in-time in the same sense the regulation layer is, rather than being uniformly present
or uniformly absent.

## Limitations

- `valid_from` is a lower bound on validity, not a publication date, and the "at least since"
  reading is the only one it supports. See above.
- Coverage is measured on title 5 at one release point. It is the title this project ingests,
  and it is the only one the number covers.
- The prose fallback is validated against the attribute path on 1,129 sections and by unit
  test, but it has never been the answering path on real input. If it ever becomes one, that
  is a fact worth re-measuring rather than assuming.
- Nothing here detects repeal. `valid_to` is open on every statute row, so a section removed
  in a later edition stays in force in the store until something else closes it.
