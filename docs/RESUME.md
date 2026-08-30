# Resume point — paused 2026-08-30, four agents stopped mid-task

Everything committed and pushed is green: `ruff check src tests` clean, full suite passing,
clone-verified. **https://github.com/tasnimuldatascience/warrant**, branch `main`.

Three partial files are on disk, uncommitted and unfinished. They are the leading edge of
four tasks that were stopped, not abandoned:

| path | state |
|---|---|
| `ui/` | Scaffolding plus some screens. The agent was mid-way through the Ask screen. **No `ui/dist` yet**, so the server mounts nothing. |
| `configs/multi.yaml` | Written; the live ingest had not run. |
| `src/warrant/eval/hopstudy.py` | Written; tests and the generation run had not. |

Nothing on the shipped path imports any of them, which is why the suite is green with them
present. Decide whether to finish or delete them before committing.

## The four tasks, in the order they are worth resuming

**1. Serving fixes** — `src/warrant/serve/api.py`, `tests/test_api.py`.
`/api/ask` is `def`, so the real queue is Starlette's unbounded four-thread pool in *front*
of the generation semaphore: a 503's floor is 20.1 s and its p50 at 3x load is 65 s, and
`GENERATE_DEADLINE_S` is dead code (its counter has never incremented across 14,628
requests). `results/eval-010-capacity.md` §7 specifies the fix precisely: `async def` with an
explicit thread hop, an `anyio.Semaphore` taken before that hop, deadline started at arrival
in middleware. Same task also wires `serve/cache.py`, which has 25 tests and is imported by
nothing — and should resolve an omitted `as_of` to the corpus's latest snapshot date rather
than to today, which is a one-line choice worth 10-15 points of hit rate.

**2. The interface** — everything under `ui/`.
`serve/api.py` mounts `<repo>/ui/dist` automatically. Four screens: Ask (use
`GET /api/ask/stream`, and render evidence the moment it arrives — retrieval is 18 ms and
generation is 7 s, and hiding that behind one spinner throws away the most distinctive thing
about the serving design), Timeline, Diff, Trace. Design direction is an archival instrument,
not a chat app. Commit `ui/dist`; a clone must get a working interface with no node toolchain.

**3. Multi-source corpus** — `configs/multi.yaml`, `results/eval-017`, `tests/`.
All four non-eCFR sources are built and tested and the shipped store still holds **only
eCFR**. Build at a separate path (`data/warrant-multi.sqlite3`), scoped to 5 CFR 630 + 5 USC
ch. 63 + the FR notices amending 630 + the OPM leave fact sheets, so all five tiers speak to
one subject. The measurement that matters: does a guidance page outrank the law it
summarises, and does the authority tie-break in fusion ever actually fire? If it almost never
fires, that is a finding about a design decision of mine that should not survive unexamined.

**4. Does multi-hop improve *answers*** — `src/warrant/eval/hopstudy.py`.
Both multi-hop and the qualifier check were deferred on the same sentence: nothing has shown
that closing a dangling reference produces a better answer, because no generations were on
disk. There are now 134. Generate with `hop_budget: 8, hop_depth: 3` and without, on
identical items, and score hallucination, citation precision, unstated conditions and the
abstention quadrant. eval-013 measured sufficiency at -0.88 with the hop on, never once
positive at any budget or depth — so the likely answer is "leave it off", and 29 human items
means a null reads as "cannot tell", not "no effect".

## Standing facts worth not re-deriving

- Corpus 13,212 chunks, schema **v4**, eCFR only. The store refuses a version mismatch by
  design; a rebuild is ~6 s plus ~30 s to re-index.
- Retrieval p50 **18.4 ms**. Generation **29.2-29.9 tok/s** over ~205 tokens = **6.6 s**,
  ceiling **7.7 req/min**, stable band 6. The 21.3 / 19.7 s / 3-per-minute figures in older
  results docs are superseded and carry pointers saying so.
- Four stages ship **off** with their own p-value beside the flag: rerank (+0.5, p=0.79),
  entailment (+2.3, p=0.55), the calibrated combiner (AURC +0.0019), multi-hop (-0.88,
  p=0.25). That consistency is the point; do not quietly turn one on.
- Quality floors are recorded per *what ran*, not per config file — `results/eval-floor.json`
  (dense+rerank) and `results/eval-floor-lexical.json` (CI). `warrant eval gate` reports
  *incomparable* across a config or model-set change rather than passing.
- Rebuilding changes chunk ids. `results/eval-floor*.json`, `results/failure-budget.json` and
  `benchmarks/entailment.yaml` are keyed on them; `benchmarks/human.yaml` is not.
