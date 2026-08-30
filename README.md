<div align="center">

# warrant

**Most RAG evaluations tell you the system was wrong. This one tells you which stage made it wrong.**

[![license](https://img.shields.io/badge/license-MIT-22863a)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%20|%203.13-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![tests](https://img.shields.io/badge/tests-0%20passing-22863a)](tests/)
[![corpus](https://img.shields.io/badge/corpus-12,858%20chunk%20versions%20|%2026%20CFR%20parts-5b8cff)](results/spike-001-amendment-viability.md)

</div>

---

## What is this?

A retrieval system over US federal HR regulation that answers **for a given scope, as of a
given date** — and that localizes every wrong answer to the pipeline stage responsible.

Ask *"how long must restored annual leave be scheduled within?"* and the correct answer
depends on when you are asking, because the regulation changed. Ask *"how is my within-grade
increase determined?"* and it depends on whether you are paid under the General Schedule or
the Federal Wage System, because different parts of the CFR govern each.

Built on the [eCFR](https://www.ecfr.gov) versioner API, which serves the text of any part
**as it stood on any date**. Nothing here is synthetic: every benchmark question is grounded
in an amendment that actually happened.

## The failure budget

Every failure attributed to the first stage at which no sufficient evidence survives, and
what happened when the largest rows were fixed.

| stage | before | after | |
|---|---:|---:|---|
| ingestion | 0 | 0 | |
| applicability | 0 | 0 | |
| temporal | 0 | 0 | |
| retrieval | 25 | 25 | unchanged — correct, the fix was downstream |
| fusion | 87 | **40** | −47 |
| rerank | 46 | 64 | +18 — bottleneck moved downstream |
| truncation | 78 | **41** | −37 |
| **total failures** | **236** | **170** | 67.3% → **76.4%** satisfied |

Two thirds of failures were evidence the system had already found and then cut. Widening the
fused head and the final cut — a fix chosen *from that table*, not in advance — removed 66 of
them. `retrieval` did not move, which is right: a downstream window cannot change what was
retrieved.

[**results/eval-002**](results/eval-002-failure-budget.md) has the full run, including two
bugs the budget found in itself.

## Do the predicates actually work?

Asserting that a temporal filter works is easy. This is the measurement.

| | sufficiency | cites the wrong version |
|---|---:|---:|
| as-of predicate **on** | 76.4% | **0.0%** |
| as-of predicate **off** | 62.3% | **62.0%** |
| applicability **on** | — | **0.0%** |
| applicability **off** | — | **100.0%** |

Without the as-of predicate, 62% of answers cite a rule that was not in force on the date
asked. The predicate also *raises* sufficiency, because superseded near-duplicates otherwise
crowd the correct version out of the candidate list — the measured cost of post-filtering
instead of pushing the predicate into the query.

## Buckets, reported separately

| bucket | n | sufficiency | 95% CI | what it measures |
|---|---:|---:|:---:|---|
| temporal | 721 | 76.4% | 73.2–79.3 | dating correctness |
| human | 42 | 81.0% | 69.0–92.9 | realistic queries |
| scope | 60 | 100.0% | 100.0–100.0 | not over-excluding |
| scope-exclusion | 60 | n/a | | not over-including |
| generated | 130 | 100.0% | 100.0–100.0 | corpus reachability |

Never averaged into one number. `generated` at 100% is not an achievement — its queries are
built from the paragraph they retrieve. `human` at 42 items has a 24-point interval and
cannot rank configurations; it exists to characterise what a real query looks like.

## What this is not

**It is not access control.** eCFR is published law. Nothing here is confidential, nothing
can leak, and no leak-rate claim is made. Filtering by who is asking is an *applicability*
question — citing a rule that does not govern you is a correctness failure, not a security
breach. [ARCHITECTURE.md](ARCHITECTURE.md) section 3 says this at length, because it is the
easiest thing in this project to overclaim.

It is also not legal advice, not a general web-scale RAG, and not yet a generation system:
everything above is retrieval-only. See the phase table in
[ARCHITECTURE.md](ARCHITECTURE.md) section 12.

## Run it

```bash
git clone https://github.com/tasnimuldatascience/warrant
cd warrant && make install

make survey     # how much amendment history each eCFR part actually has
make fetch      # download point-in-time snapshots (~220 files, cached, ~10 min once)
make build      # parse into the bitemporal store
make index      # embed for dense retrieval (~10 s on a laptop GPU)
make eval       # score every bucket, with ablations
make autopsy    # localize failures; print the failure budget
```

**No graphics card?** Set `index.dense.enabled: false` and `index.rerank.enabled: false`.
The lexical path, both predicates and the whole failure budget run without torch.

## How it works

Point-in-time snapshots are ingested into a **bitemporal** SQLite store — `valid_from/valid_to`
for when the text was the law, `system_from/system_to` for when Warrant believed it was, so a
past answer can be reproduced from what was known at the time. Retrieval is BM25 over FTS5
plus dense cosine, fused by reciprocal rank, with the as-of and applicability predicates
pushed **into** the query rather than applied to the results. Every stage writes what it saw
into a trace, and the autopsy reads the trace rather than re-running retrieval.

[ARCHITECTURE.md](ARCHITECTURE.md) is the full design, including what it deliberately does
not claim.

## License

MIT
