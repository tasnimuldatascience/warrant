<div align="center">

# warrant

**Most RAG evaluations tell you the system was wrong. This one tells you which stage made it wrong.**

[![license](https://img.shields.io/badge/license-MIT-22863a)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%20|%203.13-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![tests](https://img.shields.io/badge/tests-246%20passing-22863a)](tests/)
[![corpus](https://img.shields.io/badge/corpus-13,145%20chunk%20versions%20|%2026%20CFR%20parts-5b8cff)](results/eval-004-held-out.md)

</div>

---

## What is this?

A question-answering system over US federal HR regulation that answers **for a given scope, as
of a given date** — and that localizes every wrong answer to the pipeline stage responsible.

Ask *"by when must restored annual leave be scheduled?"* and the correct answer depends on
when you are asking, because the regulation changed. Ask *"how is my within-grade increase
determined?"* and it depends on whether you are paid under the General Schedule or the Federal
Wage System, because different parts of the CFR govern each.

Built on the [eCFR](https://www.ecfr.gov) versioner API, which serves the text of any part
**as it stood on any date**. Nothing is synthetic: every temporal benchmark item is grounded
in an amendment that actually happened.

## Do the predicates work?

Every number below is on a **held-out test split**, split by section. Retrieval parameters
were chosen by reading the failure budget, so the budget runs on `dev` and results are
reported from `test` — the two commands default to opposite sides.

| removing | measure | delta | 95% CI | won / lost | p |
|---|---|---:|:---:|---:|---:|
| the as-of predicate | sufficiency | +2.2 | 0.0–5.5 | 5 / 0 | 0.06 |
| the as-of predicate | **wrong-version rate** | **+96.1** | 92.8–99.5 | **220 / 0** | 1.2e-66 |
| the cross-encoder | sufficiency | +1.3 | 0.0–4.0 | 4 / 1 | 0.38 |

Paired and section-clustered — 229 temporal items come from 47 sections, so items are not
independent trials and an item-level bootstrap reports an interval roughly 3.5× too narrow.

**Sufficiency alone would have called the as-of predicate useless.** With `final_k: 16`
several versions of a section fit in the result list at once, so removing the predicate barely
changes whether the right paragraph is present — it changes whether the *wrong* one is present
beside it. One measure would have dismissed the predicate on the very bucket built to prove it
works. The cross-encoder, on the same test, does nothing measurable.

## The generator

Retrieval quality and answer quality are different questions, and only the first used to be
measured. Held-out human items:

| measure | value | 95% CI |
|---|---:|:---:|
| hallucination rate | **1.5%** | 0.3–8.0 |
| citation precision | **98.5%** | 92.1–99.7 |
| answered with evidence present | 23 | correct |
| **answered with evidence absent** | **6** | **wrong — answered anyway** |
| abstained, either way | 0 | |

**The model never abstains.** A low hallucination rate only means something if the system also
declines when it should. That is the next thing to fix, and it is now a number rather than an
assumption.

## Latency against quality

| configuration | p50 | p95 | sufficiency |
|---|---:|---:|---:|
| lexical only | **23.8 ms** | 31.5 ms | 96.7% |
| + dense | 41.0 ms | 51.2 ms | 96.7% |
| + cross-encoder | 71.0 ms | 131.1 ms | 98.3% |

Lexical-only is three times faster at the same quality on this bucket. A stage may be shed
under load only where this table shows it inside the noise — declaring which stages are
optional without measuring them is how a load-shedding policy trades a slow answer for a wrong
one.

Generation is a different order: 21.3 tokens/s unbatched, so the serving ceiling is **three
requests per minute**. The API admits under a semaphore and returns 503 with `Retry-After`
rather than queueing silently for 33 minutes.

## The failure budget

Every failure attributed to the first stage at which no sufficient evidence survives — and the
ladder runs through `generation` and `grounding`, so a model failure is attributable rather
than invisible. On dev: 119 items, 3 failures, `ingestion` / `applicability` / `temporal` all
zero.

The instrument found four of its own bugs, each written up in [results/](results/): a benchmark
whose items were unsatisfiable by construction, a reranker blamed for truncation, an
intervention label that implied the wrong fix, and a chunker dropping 4.5% of the corpus that
no instrument could detect — because the gold chunks came from the same parser.

## What this is not

**It is not access control.** eCFR is published law. Nothing here is confidential, nothing can
leak, and no leak-rate claim is made. Filtering by who is asking is an *applicability*
question — citing a rule that does not govern you is a correctness failure, not a security
breach. [ARCHITECTURE.md](ARCHITECTURE.md) section 3 says this at length, because it is the
easiest thing in this project to overclaim.

Not legal advice. Not a general web-scale RAG. Every architecture section is marked
`[built]`, `[partial]` or `[designed]` so no claim has to be taken on trust.

## Run it

```bash
git clone https://github.com/tasnimuldatascience/warrant
cd warrant && make install

make fetch        # eCFR point-in-time snapshots (cached, ~10 min once)
make build        # parse into the bitemporal store
make index        # embed for dense retrieval (~10 s on a laptop GPU)
make eval         # score the held-out split, with paired ablations
make autopsy      # localize failures on dev; print the failure budget
make generation   # hallucination, citation precision, abstention
make latency      # latency vs quality per configuration
make serve        # the API on :8000
```

**No graphics card?** Set `index.dense.enabled: false` and `index.rerank.enabled: false`. The
lexical path, both predicates and the whole failure budget run without torch — and on this
bucket lexical-only matches the full pipeline anyway.

## How it works

Point-in-time snapshots — including the flush paragraphs and tables an earlier chunker dropped
— are ingested into a **bitemporal** SQLite store: `valid_from/valid_to` for when the text was
the law, `system_from/system_to` for when Warrant believed it was, so a past answer can be
reproduced from what was known at the time. Retrieval is BM25 over FTS5 plus dense cosine,
fused by reciprocal rank, with the as-of and applicability predicates pushed **into** the query
rather than applied to the results.

Every stage records what it saw, the score it ranked by, and how long it took. Traces persist,
so `warrant replay show` reconstructs a past request without re-running retrieval, and
`warrant replay diff` re-runs it through today's pipeline and reports the first stage that
moved. The failure budget reads those traces.

[ARCHITECTURE.md](ARCHITECTURE.md) is the full design, including what it deliberately does not
claim.

## License

MIT
