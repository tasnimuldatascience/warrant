# Eval 010 — the capacity envelope, and what admission control actually does

**Date:** 2026-08-30
**Reproduce:** `python -m warrant.bench.load --url http://127.0.0.1:8000 --mode open --rate 0.1
--duration 180` against a running instance. Every table below names the loop shape it came
from.
**Supersedes:** the serving numbers asserted in `serve/api.py`, `serve/guard.py`,
`ARCHITECTURE.md` §10 and the README. Those were inferences from one token-rate measurement.
Three of them are wrong, and this is the run that produced the condition rather than reasoning
about it.

## Summary

| claim, before | measured | verdict |
|---|---|---|
| generation 21.3 tok/s | **29.2–29.9 tok/s** isolated | 37% low |
| ~420 tokens per answer → 19.7 s | **7.06 s** mean over 223 generations, ~205 tokens | 2.8× high |
| **3 requests/minute** | **7.7 req/min** (6.7–9.4), stable band ≤ 6 | 2.5× low |
| retrieval peaks at 4 threads, 66 QPS | peaks at 4, **23.9 QPS** end-to-end with rerank | different measurement |
| retrieval p50 18.4 ms | 78.6 ms over HTTP; 22.4 ms lexical + 16.5 ms dense confirmed | both true, different scopes |
| predicates 0.02 ms | **0.02 ms** over 320 calls | holds exactly |
| admission control refuses rather than queueing | **it queues, then refuses after 20–120 s** | does not hold |

The one-line version: **the semaphore bounds the GPU, not the queue in front of it.** Under a
100-way overload the GPU stayed inside 6,071 MiB of 8,151 and nothing leaked — and the
retrieval path, which uses no GPU and costs 82 ms, went to **zero successful requests**.

## What was measured, and against what

Two revisions, because the repo moved during the run and pretending otherwise would be the
kind of one-sided accounting this project has already withdrawn a claim for:

- **R1** — `serve/api.py` with the generation semaphore only, before `serve/guard.py` was
  wired in. Store schema v3, 13,145 chunk versions. All generation-path, overload and
  starvation numbers.
- **R2** — `serve/api.py` with `guard.RateLimitMiddleware` wired (store schema v4, 13,212
  chunk versions). The retrieval envelope re-verified; the limiter measured.

Hardware: RTX 5070 Laptop (8,151 MiB), 16 logical CPUs, Windows 11, Python 3.13.5,
torch 2.11+cu128. Config as `configs/default.yaml`: dense on, cross-encoder on,
Qwen2.5-1.5B-Instruct, `final_k: 16`, `THREAD_LIMIT = 4`.

**The machine was not quiet.** Other agents were building and testing this repo throughout;
one 1.7 GB torch process and a full pytest run overlapped parts of the sweep. Where repeats
disagree, both are reported and the range is given. Two of the retrieval reps are visibly
contaminated; they are in the table rather than dropped, because dropping the slow runs of a
load test is how a capacity number becomes a best case wearing a median's name.

Corpus was a scratch copy; `data/warrant.sqlite3` was never written to.

## Why both loop shapes, and which number came from which

A **closed-loop** harness runs N workers, each waiting for its response before sending again.
Its offered load is a function of the server's speed: when the server slows down it receives
*less* traffic, so the queue never grows and every latency it reports is a service time
measured from a moment the harness itself chose. That is **coordinated omission**. It is not a
rounding error. Measured here, on the same server, in the same hour:

| shape | what it offered | what it reported |
|---|---|---|
| closed loop, 1 worker | 8.4 req/min (its own achieved rate) | p99 **11.9 s** — reads as healthy |
| open loop, 12 req/min | 12 req/min regardless | p99 **47.4 s**, ramp +27.8 s, 14% refused |
| closed loop, 100 workers | 100 sockets, 7.9 req/min drained | p50 **561.5 s** — reads as catastrophic |

Asking for 40% more than the closed-loop harness happened to ask for makes the tail four times
worse, and a capacity plan built on the first row would have set the answer SLO at 12 seconds.

The general statement is stronger than the example: a closed-loop harness with N workers
offers exactly `min(N / service_time, capacity)`. It **cannot ask for more than the server can
do**, so it can never find a knee. Every knee in this document came from open-loop.

Two honesty checks are built into the harness rather than assumed:

- `wait_p99` — the gap between intended arrival and actual send. If the harness is late, an
  open-loop run has quietly degenerated into a closed-loop one. **Maximum observed across
  every run below: 10.7 ms**, against latencies of hundreds of milliseconds to minutes. No run
  degenerated.
- `latency_ramp` — median latency of the last fifth of arrivals minus the first fifth. A
  server *at* capacity has a flat ramp however slow it is; a server *over* capacity has a
  queue that grows for as long as the run lasts, which means its p99 is a function of how long
  you ran and not a property of the system at all. This is what locates the knee below.

Note that in an open-loop HTTP run the `omission` ratio (latency p99 ÷ service p99) stays near
1.0, because the client sends on time and the queue lives in the server. That ratio's job is
to catch a harness that *could not* send on time. The coordinated omission itself is only
visible by comparing the two shapes, which is why both are implemented.

## 1. Retrieval-only — `/api/ask?generate=false`

### Closed loop: service time against concurrency

Two revisions, ranges across both.

| workers | QPS | p50 ms | p90 ms | p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 12.2–13.2 | 75.5–78.6 | 89–104 | 109–140 |
| 2 | 18.0–18.2 | 108–111 | 143 | 158–177 |
| **4** | **23.5–23.9** | 158–161 | 224–237 | 253–315 |
| 8 | 22.0–23.2 | 327–354 | 457–564 | 524–627 |
| 16 | 21.0–23.0 | 645–710 | 963–1152 | 1143–1474 |
| 32 | 15.5–19.7 | 1495–2107 | 1916–2761 | 3179–3276 |
| 64 | 15.3 | 4246 | 5431 | 7788 |

Throughput peaks at 4 and falls, which is the shape `api.py` claims. The magnitude is not:
**23.9 QPS, not 66.** The docstring's number is an in-process retrieval measurement; this one
is what a client sees, and the difference is one stage.

### Where it goes, per stage

Differenced from `warrant_stage_duration_ms` across the run, so these are the service's own
numbers rather than a second stopwatch:

| stage | mean ms @ 1 | mean ms @ 4 |
|---|---:|---:|
| predicates | **0.02** | **0.02** |
| lexical | 22.4 | 25.2 |
| dense (concurrent with lexical) | 16.5 | 19.5 |
| fusion | 1.4 | 2.8 |
| **rerank** | **40.1** | **98.6** |

The predicate cache holds exactly as claimed. Lexical and dense hold. **The cross-encoder is
the stage that does not scale**: it is half of one request's latency at concurrency 1 and it
*doubles* at concurrency 4 while everything else moves by 15%, because it is a GPU stage being
entered by four Python threads. §10 of ARCHITECTURE.md says the shed list is decided by
measurement; this is the throughput half of that measurement, and it says the same thing the
quality half did.

### Open loop: response time against offered rate

Poisson arrivals, seed 11, 20 s per point, median [range] over reps.

| offered req/s | achieved | p50 ms | p99 ms | ramp s | refused |
|---:|---:|---:|---:|---:|---:|
| 5 | 4.9 | 124 | 312 | −0.0 | 0 |
| 8 | 8.0 | 110 [107–228] | 346 [281–1135] | +0.1 | 0 |
| 10 | 9.4 | 143 [112–269] | 668 [292–1175] | −0.0 | 0 |
| 12 | 11.3 | 125 [121–138] | 341 [332–1069] | −0.0 | 0 |
| 14 | 13.5 | 142 [133–1262] | 510 [433–2297] | +0.0 | 0 |
| 16 | 15.1 | 221 [164–5716] | 1329 [644–8232] | +0.6 | 0 |
| **18** | 17.0 | 339 [226–7953] | 940 [784–9501] | −0.0 | 0 |
| 20 | 15.5 | 3882 [1048–4362] | 7644 [1993–13014] | **+5.9** | 0 |
| 22 | 18.7 | 2161 [942–17884] | 4569 [2322–23539] | +3.5 | 0 |
| 25 | 11.5 | 23273 | 29293 | **+17.4** | 0 |

**The knee is 18 req/s on a quiet machine, 12–14 with an unrelated build running.** Below it
the ramp is flat and the p99 is under a second; at 20 the queue starts growing and never
stops.

Two things in that table matter more than the knee:

**Nothing is ever refused.** Retrieval has no admission control at all. At 25 req/s the median
caller waits 23 seconds for an 80 ms operation, and the service returns 200 to every one of
them.

**Throughput collapses rather than plateauing.** 23.9 QPS at concurrency 4, 11.5 QPS at 25
offered — the server does *half* as much work when asked for more. That is congestion
collapse, and it is the specific outcome admission control exists to prevent.

## 2. The answer path — `/api/ask`

### The ceiling, and the token rate under it

Single stream, closed loop, one worker, four runs across both revisions:

| | run 1 | run 2 | run 3 | run 4 | server-side |
|---|---:|---:|---:|---:|---:|
| req/min | 9.40 | 8.41 | 6.67 | 7.67 | — |
| p50 s | 5.90 | 6.22 | 7.71 | 6.89 | — |
| p99 s | 11.23 | 11.94 | 12.85 | 12.31 | — |
| mean generation s | | | | | **7.06** (223 calls, `warrant_generate_duration_s`) |

**7.7 req/min (6.7–9.4).** The spread is machine contention and answer length; runs 3 and 4
are the current revision, which also does prompt bounding and output validation, and the run
is not clean enough to attribute the difference to that rather than to the box.

Re-derived in isolation with nothing else on the GPU, 1,015-token prompt, forced lengths:

| new tokens | wall | tok/s |
|---:|---:|---:|
| 32 | 1.56 s | 20.5 |
| 256 | 8.67 s | 29.5 |
| 256 | 8.57 s | 29.9 |
| 420 | 14.37 s | 29.2 |

**29.2–29.9 tok/s, not 21.3.** The 32-token row is the likely origin of the old figure: at
short outputs the fixed prefill and launch cost dominates and the apparent rate collapses to
20.5. Weights peak at 3,131 MiB allocated.

And the second half of the arithmetic is wrong too. 7.06 s mean at 29.5 tok/s is **~205 output
tokens**, not the 420 `MAX_NEW_TOKENS` ceiling the estimate used. The published 19.7 s per
answer is the product of two errors in the same direction: **19.7 s predicted, 6.6 s
measured.**

Everything downstream inherits it — `GENERATE_QUEUE_WAIT_S`, `GENERATE_FLOOR_S`,
`ANSWER_RATE_PER_S`, and the README's "three requests per minute".

### Open loop: the envelope

Uniform arrivals, 180 s per point.

| offered req/min | n | goodput/min | p50 s | p90 s | p99 s | ramp s | 503 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **6** | 18 | 6.21 | 5.85 | 11.38 | 12.35 | **−0.48** | 0 |
| 9 | 27 | 8.23 | 10.67 | 20.09 | 28.52 | +4.82 | 1 |
| 12 | 36 | 8.68 | 24.57 | 39.35 | 47.43 | +27.80 | 5 |
| 18 | 54 | 8.63 | 101.28 | 166.31 | 176.86 | **+114.17** | 11 |

**Goodput saturates at 8.6 req/min. The knee is between 6 and 9.** At 6/min the ramp is
negative — the queue drains as fast as it fills, and the worst request in three minutes took
12.35 s. At 9/min the ramp is already +4.8 s and the run's p99 has started depending on how
long the run was.

At 18/min — three times the stable rate — the median caller waits **101 seconds** for an
answer that costs 6.6.

### Does admission control refuse before latency degrades?

**No.** This is the load test's most useful result, and it is the opposite of what
`api.py:_generate_answer` documents.

Latency split by status code, same runs:

| offered req/min | 200 n | 200 p50 | 503 n | 503 p50 | 503 min | 503 max |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 26 | 9.4 s | 1 | 20.1 s | 20.1 s | 20.1 s |
| 12 | 31 | 25.2 s | 5 | 23.7 s | 20.1 s | 29.0 s |
| 18 | 43 | 121.0 s | 11 | **65.1 s** | 24.8 s | **120.2 s** |

A refusal is never prompt. Its **floor is 20.1 s** — the semaphore timeout, paid in full before
the answer "come back in 30 seconds" is even reached — and under 3× load the median rejected
caller is held for **65 seconds** and the worst for **120**, on a service whose entire answer
takes seven.

The mechanism, and it is worth stating precisely because it makes the fix obvious. `/api/ask`
is a **sync** endpoint, so it runs in anyio's threadpool, capped at `THREAD_LIMIT = 4`. A
request must first obtain one of those four tokens, and that queue is unbounded, untimed and
FIFO. Only after it holds a thread does it reach `_GENERATION_SLOT.acquire(timeout=20)`. So
the 20-second bound governs the last four requests in the system and nothing else; the other
fifty wait in a queue with no bound at all.

The same ordering makes `GENERATE_DEADLINE_S` dead code as wired. `deadline` is computed at
the top of `_generate_answer` — *after* the thread hop — so the 90 s budget always has 90 s
remaining when it is measured, and the `GENERATE_FLOOR_S` check can never fire.
Confirmed from the service's own metrics: after **14,628 requests including a 100-way
overload**, `warrant_admission_rejected_total{reason="queue_full"}` is 42 and
`{reason="deadline"}` **has never been incremented and is absent from the exposition
entirely.**

## 3. What breaks first under overload

100 concurrent `/api/ask`, closed loop, with two probes running beside it on separate
connections.

| | result |
|---|---|
| generation blast | 570.5 s to drain, 75 × 200, 25 × 503 (`Retry-After: 30`), goodput 7.89/min, service p50 561.5 s |
| **retrieval probe, 1 req/s** | **0 of 64 succeeded — every one a ReadTimeout at 300 s** |
| `/health` probe, 2 req/s | 40 × 200, p50 **188.6 s**, p99 198.6 s |
| GPU | peak 6,071 MiB of 8,151; no OOM |
| host RSS | 2,508 → 2,522 MB |
| threads | 67 → 70 |
| TCP connections | 4 → 4 |
| OS handles | 448 → 435 |
| SQLite | zero errors; the only 5xx in 14,628 requests were the 42 admission 503s |

**The read path breaks first, and for a reason that has nothing to do with its own cost.**
Retrieval uses no GPU, costs 82 ms, and is completely unavailable during a generation overload
— because both paths draw from the same four-token threadpool and a generation waiter holds
its token for up to 20 seconds doing nothing.

**Liveness breaks second.** `/health` touches nothing and answers in 1.1 ms; it is also a sync
endpoint, so it queued behind the same pool and took over three minutes. Any orchestrator with
a sane probe timeout restarts the process into the same load, forever — which is precisely the
failure `ReadyResponse`'s docstring warns about, arriving through a door nobody had checked.

For contrast, under *retrieval* saturation (64 concurrent, no generation) `/health` degrades to
2.57 s p50 and stays up. The severity ladder is unambiguous: generation overload is an order
of magnitude worse for everything else in the process.

**Nothing leaks.** Over ~14,600 requests and a 9.5-minute overload: RSS +17 MB, connections
back to baseline, handles down, GPU released, no SQLite errors, thread count +3. The
thread-local connection design holds with the pool saturated. That is the part of the serving
story that survives this document intact.

**What could not be falsified:** the claim that the GPU OOMs past ~35 in flight. With the
semaphore in place only one generation exists at a time, so 100-way concurrency peaked at
6,071 MiB and nothing went wrong. The semaphore demonstrably protects GPU memory. It
demonstrably does not protect latency, the read path, or liveness.

## 4. The rate limiter, as wired

`guard.RateLimitMiddleware` landed mid-run. Measured on R2 with a single client:

| endpoint | served | refused | Retry-After | cost of a refusal |
|---|---:|---:|---:|---:|
| `/api/ask?generate=false` × 30 | 3 | 27 × 429 | 20 s | 0.8 ms |
| `/api/meta` × 60 | 20 | 40 × 429 | 1 s | 0.8 ms |
| `/health` × 300 | 300 | 0 (exempt) | — | — |

The token bucket works exactly as designed and refusing really does cost a dict lookup. Two
of its constants do not survive this measurement:

**`ANSWER_RATE_PER_S = 3/60` is derived from 21.3 tok/s.** The measured server ceiling is
7.7 req/min and the stable band is 6, so one client is capped at 39% of what one instance can
actually serve — and the docstring's justification ("a client inside the ceiling is never
limited") no longer holds, because the ceiling moved.

**`/api/ask?generate=false` is charged against the answer bucket.** It is in `answer_paths`
regardless of the `generate` flag, so the 82 ms GPU-free path is metered as if it cost a
generation: three requests and a browsing client is cut off for 20 seconds. Any UI that uses
`generate=false` to populate a result list hits this on its fourth interaction.

## 5. Cost per query

Concurrency 1, GPU busy sampled from `nvidia-smi utilization.gpu` at 5 Hz.

| | wall | GPU-seconds | GPU memory |
|---|---:|---:|---|
| retrieval-only | **82.5 ms** | **26.7 ms** | 4,994 MiB resident, no transient |
| full answer | **7.14 s** (6.6 s median across runs) | **3.7–4.5 s** | 4,994 resident + up to 1,423 transient (peak 6,417 MiB) |
| idle | — | 0.0% util | 4,994 MiB resident |

The 4,994 MiB is the price of admission and it is paid whether or not anyone asks anything:
bge-small (127 MB) + MiniLM cross-encoder (87 MB) + Qwen2.5-1.5B fp16 (3,131 MiB allocated) +
the CUDA context. The generation path adds up to 1.4 GB of transient KV cache and activations.

**A thousand queries a day:**

| | wall | GPU-hours | share of one card-day |
|---|---:|---:|---:|
| retrieval-only | 82.5 s | 0.0074 | 0.03% |
| full answer | 1.98 h | 1.02–1.24 | 4.3–5.2% |

So a thousand answers a day fits on this laptop GPU with room to spare *on average* — 0.69
req/min against a 7.7 req/min ceiling. The average is not the constraint. A thousand queries
concentrated into an 8-hour workday is 2.1 req/min mean, and an ordinary 3× Poisson peak
reaches 6 req/min, which is exactly the top of the stable band. **The capacity plan for 1,000
queries a day is not "buy 5% of a GPU", it is "the peak minute is at the knee".**

No dollar figure is given. What it costs depends entirely on where it runs, and this repo has
no pricing to reproduce from.

## 6. Proposed SLOs

Three classes, because the paths differ by ~90× in cost and an SLO that averages them is an
SLO about nothing. Each objective is set where the measurement is, with the margin named.

### Class A — metadata and read (`/api/meta`, `/api/section*`, `/api/diff`)

> **p99 ≤ 250 ms at offered ≤ 50 req/s per instance, measured over a rolling 5 minutes.**

Measured: 1.13 ms p50 at 817–879 QPS single-stream. The margin is ~200× and that is
deliberate — this is the class that must still work when the others do not.

**It does not hold today.** During a generation overload `/health` reached 188 s. This
objective is meetable only once the read path stops sharing a threadpool with generation
waiters; until then it is a target, not a commitment.

### Class B — retrieval (`/api/ask?generate=false`)

> **p99 ≤ 1.5 s at offered ≤ 12 req/s, measured over a rolling 5 minutes.**
> Error budget: 1% — 36 requests per window.

Measured at 12 req/s: p99 341 ms median over three repeats, worst 1,069 ms on a contended
machine. The 1.5 s figure is not the pipeline's number, it is the contention headroom: the
pipeline does this in 341 ms and the extra second buys tolerance for whatever else is on the
box. Headroom to the measured knee (18 req/s) is 1.5×.

### Class C — the answer path (`/api/ask`)

> **C1. 99% of admitted requests answered within 25 s, rolling 1 hour, while offered load is
> ≤ 6 req/min.**

Measured at 6 req/min over three minutes: p99 12.35 s, max 12.35 s, zero rejections, ramp
−0.48 s. The 2× margin covers a full 420-token answer (14.4 s of decode at 29.2 tok/s) plus
retrieval, which is the worst case the generator can produce.

> **C2. 99% of refusals returned within 2 s.**

**This objective fails today, and it is stated anyway because it is the one worth fixing.**
Measured floor 20.1 s; p50 65 s and max 120 s at 3× load. A refusal that takes 65 seconds is
not admission control, it is a timeout with a nicer status code. The fix is in §7.

> **C3. Above 6 req/min there is no availability objective.**

Stated plainly rather than hidden in a percentile: **this is a seven-requests-per-minute
service, and past six of them there is a queue.** An SLO the system provably cannot meet is
worse than none, and "99.9% under 30 s at any load" is a sentence this architecture cannot
back with a single unbatched 1.5B model on one GPU.

### What the error budget buys

At C1's own ceiling a day is 6 × 60 × 24 = **8,640 requests**, so 1% is **86 requests**.

At 18 req/min *every* request exceeds 25 s — measured p50 was 101 s. So the entire daily error
budget is consumed by **4.8 minutes of 3× overload**. That is the number to put in front of
anyone proposing to point a dashboard, a crawler or a demo audience at this endpoint: five
minutes of triple traffic, once, and the day's budget is gone.

Measured the other way: at 12 req/min the 503 rate is 14% and the p99 is 47 s, so the budget
survives roughly **9 minutes** of 2× load per day.

## 7. Wiring this measurement implies

Not applied here — these files belong to other work — but each one is a direct consequence of
a number above.

1. **Make `/api/ask` `async def`** and hop to the threadpool explicitly around retrieval, so a
   request waiting for the generation slot does not hold one of four pool tokens. This is the
   single change behind the starved read path, the 188 s `/health`, and the 20–120 s refusals.
2. **Acquire the generation slot with an `anyio.Semaphore` in the async context**, before the
   thread hop. Then `GENERATE_QUEUE_WAIT_S` bounds the real queue instead of its last four
   entries, and the 503 becomes prompt.
3. **Start `GENERATE_DEADLINE_S` at request arrival**, in middleware, not after the thread is
   acquired. As wired, `reason="deadline"` cannot fire and never has.
4. **Make `/health` and `/ready` `async def`.** They touch nothing and answer in 1.1 ms; being
   sync is the only reason liveness can queue for three minutes.
5. **Take `/api/ask?generate=false` out of `guard.answer_paths`**, or key the bucket on the
   resolved cost. An 82 ms GPU-free request should not spend an answer token.
6. **Raise `ANSWER_RATE_PER_S` from 3/min toward 6/min**, and update the comment that derives
   it from 21.3 tok/s.
7. **Correct 21.3 tok/s → 29.2–29.9 tok/s and 3 req/min → 7.7 req/min** in `serve/api.py`,
   `serve/guard.py`, `serve/metrics.py`, `ARCHITECTURE.md` §10 and the README. The
   `GENERATE_BUCKETS_S` histogram edges were chosen against the old number and now put the
   entire distribution in the first three buckets.
8. **A CLI entry point** — `warrant bench load` — if the load generator should be reachable the
   way `make latency` is. It is deliberately not wired: this is the one command that can make
   the service look broken to everyone else on the machine.

## 8. What this does not measure

- **Tokens per request, inside the server.** Nothing exports an output-token count, so the
  ~205 tokens/answer figure is `7.06 s ÷ 29.5 tok/s` and inherits every assumption of the
  isolated measurement.
- **GPU work.** `utilization.gpu` at 5 Hz is the fraction of sampled instants with at least
  one kernel resident. It is an upper bound on "busy", not occupancy and not energy. The
  GPU-seconds figures are that bound, not a measure of work done.
- **Batching.** Every number here is unbatched. Continuous batching is the one change that
  would move the ceiling, and none of it is tested.
- **The answer cache.** `serve/cache.py` exists and is not wired into `api.py` at either
  revision measured, so every request paid full generation. A wired cache changes the answer
  path envelope completely and this document says nothing about it.
- **Multiple clients against the rate limiter.** All limiter numbers are one client key
  (127.0.0.1). N clients each inside their own bucket still saturate one instance.
- **More than one instance.** No horizontal scaling, no shared-nothing test, no leader.
- **The 33-minute and GPU-OOM claims** for the un-semaphored path. Reproducing them means
  removing the semaphore, which was not done.
- **A quiet machine.** Nothing here was measured on an idle box, and the spread in the
  retrieval tables is the price of that. The generation numbers are less affected — a
  serialised GPU stage does not care much about spare CPU — but they are not immune either.
