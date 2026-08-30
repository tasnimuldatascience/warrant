# Resume point — session limit hit 2026-08-30 ~07:50 ET

Seven agents were terminated mid-task by the session limit. Nothing is broken:
`ruff check src tests` is clean and the full suite passes. But four modules landed
without the tests and results docs they were commissioned with, and they must not be
treated as finished.

## Complete and tested

| Module | State |
|---|---|
| `sources/federal_register.py` | 26 tests. Notices joined to CFR parts. |
| `sources/usc.py` | 34 tests. Found that govinfo soft-404s USLM with HTTP 200; uses OLRC release points instead. |
| `sources/html.py` | 33 tests. OPM fact sheets; 45.1% boilerplate dropped. |
| `sources/pdf.py` | 34 tests. Layout, tables, OCR; column detection by character-weighted projection. |
| `serve/cache.py` | 25 tests. Bitemporal answer cache, invalidated by content hash not version id. |
| `retrieve/query.py`, `corpus/chunking.py`, `train/*` | tests present |
| `index/store.py` schema v3, concurrent lexical+dense, cached admitted set | tests added |

## Incomplete — finish these first

| Module | What is missing |
|---|---|
| `verify/abstain.py` | no `tests/test_abstain.py`, no `docs/results/eval-005-abstention.md`. The risk–coverage curve and ECE were never measured. |
| `verify/calibrate.py` | same commission; untested. |
| `verify/entail.py` | no tests, no `eval-007`. The domain-shift measurement (MNLI → regulatory prose) is the whole point and has not been run. |
| `verify/xref.py` | no tests; the dangling-reference rate was never measured. |
| `verify/qualifier.py` | **never written.** 15.7% of in-force chunks carry a qualifier (763 except/unless, 389 subject to, 487 may not/shall not of 9,961). |
| `serve/guard.py` | **never written.** |

## Then, in order

1. Wire the four sources into the CLI as `warrant corpus ingest --source {federal_register,usc,opm,govinfo}`, using the existing `corpus/ingest.py`. A `sources:` block in `configs/default.yaml` is the missing piece; source constructors are all dataclasses and their fields are already documented in their modules.
2. Source-aware retrieval: `sources` filter and `max_authority` on `Retriever`/`Store`. Ingesting an authority hierarchy is pointless while ranking cannot see it.
3. Re-measure latency on a quiet machine. The numbers below were taken with seven agents running and the p95s are contention, not the system.

## Latency, measured this session (60 queries, contended machine)

```
                    before     after
predicates          9.74ms  →   0.02ms
lexical + dense    31.14ms  →  18.39ms   (concurrent, not sequential)
```

Retrieval is not the bottleneck. Generation at 21.3 tok/s → ~3 requests/minute is.
