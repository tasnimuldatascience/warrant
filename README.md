<div align="center">

# warrant

**Most RAG evaluations tell you the system was wrong. This one tells you which stage made it wrong.**

[![license](https://img.shields.io/badge/license-MIT-22863a)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%20|%203.13-3776ab?logo=python&logoColor=white)](pyproject.toml)

</div>

---

> **Status: in development.** The failure budget below is the artifact this repository exists
> to produce. It is not published until it is measured, and no number appears here that has
> not been produced by `make autopsy` on a clean checkout.

## What is this?

A question-answering system over US federal HR regulation that answers **for a given scope, as
of a given date** — and that localizes every wrong answer to the pipeline stage responsible.

Ask *"how much restored annual leave can I carry, and by when must I use it?"* and the answer
depends on your agency, your pay system, and the date you are asking about. The regulation
genuinely changed; Warrant retrieves the version that was in force.

## How it works

Ingest point-in-time snapshots from the [eCFR](https://www.ecfr.gov) versioner API into a
bitemporal store, retrieve with a scope-and-date predicate pushed into the query, generate
answers whose every claim carries evidence, and then — when an answer is wrong — walk the
pipeline backwards to find where the evidence was lost.

[**ARCHITECTURE.md**](ARCHITECTURE.md) is the full design, including what this system
deliberately does *not* claim.

## Run it

```bash
git clone https://github.com/tasnimuldatascience/warrant
cd warrant && make install

make survey     # how much amendment history each eCFR part actually has
make fetch      # download point-in-time snapshots (cached)
make diff       # classify what changed, and report the discard rate
```

Later phases (bitemporal store, retrieval, benchmark, failure budget) add their targets to
`make help` as they land. Nothing is advertised before it works.

**No graphics card?** Set `index.dense.enabled: false`. The lexical path and the entire
failure budget run without torch.

## License

MIT
