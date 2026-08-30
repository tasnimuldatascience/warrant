"""Load generation in both shapes, because only one of them can see the collapse.

A **closed-loop** harness runs N workers, each of which sends a request and waits for the
response before sending the next. Its offered load is therefore a function of the server's
speed: when the server slows down it receives *less* traffic, the queue in front of it never
grows, and every latency it reports was measured from a moment the harness itself chose. That
is **coordinated omission** — the harness and the server have coordinated to omit exactly the
requests that would have been slow. It is not a subtle bias. Against this service's
generation path, one closed-loop worker reports a p99 of about 20 s and calls that healthy,
because 20 s is what one generation costs; the fact that a caller who wanted to ask at
10:00:00 was not served until 10:04:30 is invisible, because that caller was never sent.

An **open-loop** harness fixes the arrival schedule before the run starts and sends at those
times regardless of what has come back. Latency is then measured from the *intended* arrival,
not from the send, so a request that could not be issued on time carries its own queueing in
its number. That is the load a real client population applies: users do not slow down because
the server did.

Both are implemented here and every number this repo publishes says which one produced it.
The two are not competing estimates of one quantity — they answer different questions:

    closed-loop   what does one request cost when N are in flight?  (service time)
    open-loop     what does a caller experience at an offered rate?  (response time)

For a system whose ceiling is 0.051 req/s, the second is the only one that can say anything
about capacity at all, and it is the one most homemade harnesses do not implement.

Three implementation details carry the honesty of the result:

**The arrival schedule is precomputed from t0.** Offset *i* is ``i / rate`` — never
"previous send plus one interval". An accumulated schedule inherits every scheduling delay
into the next arrival, so a harness that stalls for 200 ms silently reduces its own offered
rate and then reports the reduced rate as if it had been asked for.

**Client-side queueing is measured and reported separately.** ``Sample.wait`` is the gap
between the intended arrival and the actual send. If it is not near zero the *harness* was
the bottleneck, and the run says so rather than attributing its own thread starvation to the
server.

**Shed arrivals are counted, never dropped.** When ``max_in_flight`` is reached the arrival
is recorded as shed. A harness that quietly stops generating load under saturation is telling
the same lie as a closed-loop one, just later.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import random
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

#: A request attempt: performs one call and returns ``(status, response_bytes)``. Anything
#: raised is caught by the runner and recorded as an error sample, because a harness that
#: dies on the first connection reset cannot measure an overload.
Request = Callable[[], tuple[int, int]]


# -- clock ------------------------------------------------------------------------------


class Clock(Protocol):
    """Wall time, injectable.

    ``sleep_until`` rather than ``sleep(seconds)`` deliberately. Sleeping for a computed
    duration accumulates error into the next arrival; sleeping until an absolute deadline
    derived from t0 cannot. It also makes a fake clock safe to share between the scheduler
    and the worker threads: advancing *to* a deadline is idempotent, where adding a duration
    is not, so two threads waiting on the same instant cannot advance virtual time twice.
    """

    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class WallClock:
    """Real time. ``perf_counter`` because the schedule needs a monotonic source."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, deadline: float) -> None:
        delay = deadline - self.now()
        if delay > 0:
            time.sleep(delay)


@dataclass
class ManualClock:
    """Virtual time, advanced explicitly. The reason the unit tests contain no sleeps.

    Not a testing afterthought: a load harness whose own tests are timing-dependent is a
    harness whose failures are indistinguishable from the flakiness it exists to measure.
    """

    t: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def now(self) -> float:
        with self._lock:
            return self.t

    def sleep_until(self, deadline: float) -> None:
        with self._lock:
            self.t = max(self.t, deadline)

    def advance(self, seconds: float) -> None:
        """Move time forward, as a served request does."""
        with self._lock:
            self.t += seconds


# -- samples and summary ----------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One request attempt, timed against both clocks that matter."""

    #: When the load model says this request should have been sent. In closed-loop mode this
    #: is the send time by definition — see `closed_loop`.
    intended: float
    sent: float
    done: float
    status: int | None = None
    error: str | None = None
    size: int = 0

    @property
    def service(self) -> float:
        """Send to response. What a closed-loop harness can see."""
        return self.done - self.sent

    @property
    def latency(self) -> float:
        """Intended arrival to response. What the caller experiences."""
        return self.done - self.intended

    @property
    def wait(self) -> float:
        """Intended arrival to send. Non-zero means the harness was late, not the server."""
        return self.sent - self.intended

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 400


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest rank, not linear interpolation.

    At the sample counts a three-requests-per-minute service produces in an afternoon, an
    interpolated p99 is a number no request experienced: it sits between the two slowest
    observations and is therefore smaller than the second-slowest thing that actually
    happened. Nearest rank always returns an observation.
    """
    if not values:
        raise ValueError("percentile of no samples")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    return ordered[max(1, rank) - 1]


def _spread(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    return {"p50": percentile(values, 50), "p90": percentile(values, 90),
            "p99": percentile(values, 99), "max": max(values)}


@dataclass(frozen=True)
class Summary:
    """What a run measured. Every field says which loop shape it came from."""

    mode: str
    n: int
    ok: int
    shed: int
    wall: float
    offered_rate: float | None
    statuses: dict[str, int]
    #: Send-to-response, in seconds.
    service: dict[str, float]
    #: Intended-arrival-to-response, in seconds. Identical to `service` in closed-loop mode,
    #: which is the point rather than a defect.
    latency: dict[str, float]
    #: p99 of the harness's own lateness. Above ~1% of the latency p99 the run is suspect.
    wait_p99: float
    #: Median latency of the last fifth of arrivals minus that of the first fifth, seconds.
    #: The within-run saturation signal: a server at capacity has a flat ramp however slow it
    #: is, and a server *over* capacity has a queue that grows for as long as the run lasts,
    #: which means its p99 is a function of how long you ran rather than a property at all.
    latency_ramp: float

    @property
    def achieved_rate(self) -> float:
        return self.n / self.wall if self.wall > 0 else math.nan

    @property
    def goodput(self) -> float:
        """Successful responses per second. 503s are throughput, not goodput."""
        return self.ok / self.wall if self.wall > 0 else math.nan

    @property
    def omission(self) -> float:
        """latency p99 / service p99 — how much a closed-loop reading would have hidden.

        1.0 means the server kept up with the schedule. 10 means a closed-loop harness would
        have reported a tail ten times better than the one callers actually got.
        """
        s = self.service["p99"]
        return self.latency["p99"] / s if s else math.nan

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d |= {"achieved_rate": self.achieved_rate, "goodput": self.goodput,
              "omission": self.omission}
        return d


def summarize(samples: Sequence[Sample], *, mode: str, wall: float,
              offered_rate: float | None = None, shed: int = 0) -> Summary:
    statuses: dict[str, int] = {}
    for s in samples:
        key = s.error.split(":")[0] if s.error else str(s.status)
        statuses[key] = statuses.get(key, 0) + 1
    return Summary(
        mode=mode, n=len(samples), ok=sum(1 for s in samples if s.ok), shed=shed,
        wall=wall, offered_rate=offered_rate, statuses=dict(sorted(statuses.items())),
        service=_spread([s.service for s in samples]),
        latency=_spread([s.latency for s in samples]),
        wait_p99=percentile([s.wait for s in samples], 99) if samples else math.nan,
        latency_ramp=_ramp(samples),
    )


def _ramp(samples: Sequence[Sample]) -> float:
    """How much worse the end of the run was than the start, by intended arrival order.

    Ordered by *intended* time rather than completion: sorting by completion puts the fast
    responses first no matter when they were asked for, which flattens exactly the trend
    this is looking for.
    """
    if len(samples) < 2:
        return math.nan
    ordered = sorted(samples, key=lambda s: s.intended)
    fifth = max(1, len(ordered) // 5)
    return (percentile([s.latency for s in ordered[-fifth:]], 50)
            - percentile([s.latency for s in ordered[:fifth]], 50))


@dataclass
class Run:
    mode: str
    samples: list[Sample]
    shed: list[float]
    wall: float
    offered_rate: float | None
    params: dict[str, Any]

    def summary(self) -> Summary:
        return summarize(self.samples, mode=self.mode, wall=self.wall,
                         offered_rate=self.offered_rate, shed=len(self.shed))


# -- arrival schedule -------------------------------------------------------------------


def arrival_offsets(rate: float, *, count: int | None = None, duration: float | None = None,
                    arrivals: str = "uniform", seed: int = 0) -> list[float]:
    """Offsets from t0 at which requests are due, computed before the run starts.

    Deterministic once ``seed`` is fixed, and independent of anything the server does — that
    independence is the whole definition of open-loop. ``poisson`` draws exponential
    inter-arrival gaps, which is the arrival process a large independent user population
    actually produces; ``uniform`` is an evenly spaced schedule, which is easier to reason
    about and slightly kinder to the server because it never bursts.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if (count is None) == (duration is None):
        raise ValueError("give exactly one of count or duration")
    if arrivals == "uniform":
        n = count if count is not None else math.ceil(rate * float(duration))
        # i / rate, never a running total: an accumulated schedule quietly lowers its own
        # offered rate by every millisecond the scheduler was late.
        return [i / rate for i in range(n)]
    if arrivals != "poisson":
        raise ValueError(f"unknown arrival process {arrivals!r}")
    rng = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    while (count is not None and len(offsets) < count) or (
            duration is not None and t < duration):
        offsets.append(t)
        t += rng.expovariate(rate)
    return offsets


# -- the two loops ----------------------------------------------------------------------


class _Pool(Protocol):
    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future: ...


def open_loop(request: Request, *, rate: float, count: int | None = None,
              duration: float | None = None, arrivals: str = "uniform", seed: int = 0,
              max_in_flight: int = 256, clock: Clock | None = None,
              pool: _Pool | None = None) -> Run:
    """Send at a fixed rate whatever the server is doing, and time from the intended arrival.

    ``max_in_flight`` bounds the harness, not the server. It exists because a client that
    opens an unbounded number of sockets against a service with a one-slot generator stops
    measuring the service and starts measuring its own thread scheduler; arrivals refused by
    that bound are counted as shed rather than dropped, so the offered load stays auditable.
    """
    clock = clock or WallClock()
    offsets = arrival_offsets(rate, count=count, duration=duration, arrivals=arrivals,
                              seed=seed)
    owned = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=max_in_flight,
                                      thread_name_prefix="warrant-load")
    samples: list[Sample] = []
    shed: list[float] = []
    lock = threading.Lock()
    in_flight = 0

    def _one(due: float) -> None:
        nonlocal in_flight
        sent = clock.now()
        status: int | None = None
        size = 0
        error: str | None = None
        try:
            status, size = request()
        except Exception as exc:  # noqa: BLE001 - a refused connection is a measurement
            error = f"{type(exc).__name__}: {exc}"
        done = clock.now()
        with lock:
            samples.append(Sample(intended=due, sent=sent, done=done, status=status,
                                  error=error, size=size))
            in_flight -= 1

    futures: list[Future] = []
    t0 = clock.now()
    try:
        for offset in offsets:
            due = t0 + offset
            clock.sleep_until(due)
            with lock:
                admitted = in_flight < max_in_flight
                if admitted:
                    in_flight += 1
            if not admitted:
                shed.append(due)
                continue
            futures.append(pool.submit(_one, due))
        wait(futures)
    finally:
        if owned:
            pool.shutdown(wait=True)
    return Run(mode="open", samples=samples, shed=shed, wall=clock.now() - t0,
               offered_rate=rate,
               params={"rate": rate, "count": count, "duration": duration,
                       "arrivals": arrivals, "seed": seed, "max_in_flight": max_in_flight})


def closed_loop(request: Request, *, workers: int, count: int | None = None,
                duration: float | None = None, clock: Clock | None = None,
                pool: _Pool | None = None) -> Run:
    """N workers, each waiting for its response before sending again.

    ``Sample.intended`` is the send time here, and that is not an approximation — a
    closed-loop harness has no arrival schedule to be late against, so its latency *is* a
    service time and its throughput *is* the server's. Reported as such: `Summary.omission`
    is 1.0 for every closed-loop run by construction, which is the reading to distrust rather
    than the reading to celebrate.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if (count is None) == (duration is None):
        raise ValueError("give exactly one of count or duration")
    clock = clock or WallClock()
    owned = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=workers,
                                      thread_name_prefix="warrant-load")
    samples: list[Sample] = []
    lock = threading.Lock()
    issued = 0
    t0 = clock.now()

    def _worker() -> None:
        nonlocal issued
        while True:
            if duration is not None and clock.now() - t0 >= duration:
                return
            with lock:
                if count is not None and issued >= count:
                    return
                issued += 1
            sent = clock.now()
            status: int | None = None
            size = 0
            error: str | None = None
            try:
                status, size = request()
            except Exception as exc:  # noqa: BLE001 - a refused connection is a measurement
                error = f"{type(exc).__name__}: {exc}"
            done = clock.now()
            with lock:
                samples.append(Sample(intended=sent, sent=sent, done=done, status=status,
                                      error=error, size=size))

    try:
        futures = [pool.submit(_worker) for _ in range(workers)]
        wait(futures)
        for f in futures:
            f.result()
    finally:
        if owned:
            pool.shutdown(wait=True)
    return Run(mode="closed", samples=samples, shed=[], wall=clock.now() - t0,
               offered_rate=None,
               params={"workers": workers, "count": count, "duration": duration})


# -- HTTP target ------------------------------------------------------------------------


def http_requester(client: httpx.Client, path: str,
                   params: Sequence[dict[str, Any]]) -> Request:
    """A `Request` that rotates through a fixed parameter list.

    Rotation matters more here than in most harnesses. `/api/ask` caches the admitted set
    per (as_of, scope) and FastAPI caches `/api/meta` outright, so hammering one query
    measures a warm path nobody's users are on. ``itertools.count`` is used unlocked
    deliberately — ``next`` on a count object is atomic under CPython, and a lock around a
    counter would put harness contention on the send path.
    """
    if not params:
        raise ValueError("no request parameters")
    turn = itertools.count()

    def _go() -> tuple[int, int]:
        response = client.get(path, params=params[next(turn) % len(params)])
        return response.status_code, len(response.content)

    return _go


DEFAULT_QUERIES: tuple[str, ...] = (
    "How long is the probationary period for a new federal employee?",
    "By when must restored annual leave be scheduled?",
    "How is a within-grade increase determined?",
    "How much service is required to gain career tenure?",
    "When may an agency make a temporary limited appointment?",
    "What happens if an employee does not satisfactorily complete probation?",
    "Can a former federal employee be reinstated without competing again?",
    "How is severance pay computed?",
)

#: Dates spread across the corpus window rather than all "today". One date is one cached
#: admitted set, so a single as_of would measure the predicate cache instead of the predicate.
DEFAULT_DATES: tuple[str, ...] = ("2018-06-01", "2020-11-16", "2022-03-01", "2024-01-01",
                                  "2025-09-01")


def load_queries(path: Path) -> list[str]:
    """Queries from a YAML benchmark (``query:`` per item) or a plain one-per-line file."""
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        return [i["query"] for i in items if isinstance(i, dict) and i.get("query")]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def ask_params(queries: Sequence[str], dates: Sequence[str] = DEFAULT_DATES, *,
               generate: bool = True) -> list[dict[str, Any]]:
    return [{"q": q, "as_of": d, "generate": "true" if generate else "false"}
            for q in queries for d in dates]


# -- GPU sampling -----------------------------------------------------------------------


@dataclass
class GpuSampler:
    """Poll `nvidia-smi` in the background so cost-per-query has a memory number.

    Sampled rather than instrumented: `torch.cuda.max_memory_allocated` reports the
    allocator's view inside one process and misses both the CUDA context and anything else
    on the card, and the question "does a thousand queries a day fit on this GPU" is about
    the card, not about one allocator.
    """

    interval: float = 0.25
    used_mb: list[float] = field(default_factory=list)
    util_pct: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def _poll(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
               "--format=csv,noheader,nounits"]
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                     check=True).stdout.strip().splitlines()[0]
                util, used = (float(x) for x in out.split(","))
            except Exception:  # noqa: BLE001 - no GPU, or nvidia-smi busy; not fatal
                continue
            self.util_pct.append(util)
            self.used_mb.append(used)

    def __enter__(self) -> GpuSampler:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def report(self) -> dict[str, float]:
        if not self.used_mb:
            return {}
        return {"gpu_mem_mean_mb": sum(self.used_mb) / len(self.used_mb),
                "gpu_mem_peak_mb": max(self.used_mb),
                "gpu_util_mean_pct": sum(self.util_pct) / len(self.util_pct),
                "samples": float(len(self.used_mb))}


# -- runner -----------------------------------------------------------------------------


def _print(summary: Summary, gpu: dict[str, float]) -> None:
    def ms(d: dict[str, float], k: str) -> str:
        return f"{d[k] * 1000:.1f}"

    print(f"\nmode={summary.mode} n={summary.n} ok={summary.ok} shed={summary.shed} "
          f"wall={summary.wall:.1f}s")
    if summary.offered_rate:
        print(f"offered {summary.offered_rate:.3f} req/s   "
              f"achieved {summary.achieved_rate:.3f} req/s   "
              f"goodput {summary.goodput:.3f} req/s")
    else:
        print(f"achieved {summary.achieved_rate:.3f} req/s   "
              f"goodput {summary.goodput:.3f} req/s")
    print(f"service ms   p50 {ms(summary.service, 'p50')}  p90 {ms(summary.service, 'p90')}  "
          f"p99 {ms(summary.service, 'p99')}  max {ms(summary.service, 'max')}")
    print(f"latency ms   p50 {ms(summary.latency, 'p50')}  p90 {ms(summary.latency, 'p90')}  "
          f"p99 {ms(summary.latency, 'p99')}  max {ms(summary.latency, 'max')}")
    print(f"harness wait p99 {summary.wait_p99 * 1000:.1f} ms   "
          f"omission x{summary.omission:.2f}   ramp {summary.latency_ramp:+.2f} s")
    print(f"statuses {summary.statuses}")
    if gpu:
        print(f"gpu mean {gpu['gpu_mem_mean_mb']:.0f} MB  peak {gpu['gpu_mem_peak_mb']:.0f} MB"
              f"  util {gpu['gpu_util_mean_pct']:.0f}%")


def main(argv: Sequence[str] | None = None) -> int:
    """Drive one load run against a running instance.

    Not wired into `warrant.cli` on purpose while this module is new: a load generator is the
    one command that can make a service look broken to everyone else using it, and it should
    be harder to reach than `--help`.
    """
    p = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--path", default="/api/ask")
    p.add_argument("--mode", choices=("open", "closed"), default="open")
    p.add_argument("--rate", type=float, default=1.0, help="open-loop arrivals per second")
    p.add_argument("--workers", type=int, default=1, help="closed-loop concurrency")
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--arrivals", choices=("uniform", "poisson"), default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-in-flight", type=int, default=256)
    p.add_argument("--no-generate", action="store_true", help="retrieval-only /api/ask")
    p.add_argument("--queries", type=Path, default=None)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=1,
                   help="requests to issue and discard; the first pays model load")
    p.add_argument("--gpu", action="store_true", help="poll nvidia-smi during the run")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    if args.count is None and args.duration is None:
        args.count = 30
    queries = load_queries(args.queries) if args.queries else list(DEFAULT_QUERIES)
    params = ([{}] if args.path in {"/health", "/ready", "/api/meta", "/metrics"}
              else ask_params(queries, generate=not args.no_generate))

    limits = httpx.Limits(max_connections=max(args.max_in_flight, args.workers) + 8,
                          max_keepalive_connections=64)
    with httpx.Client(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        request = http_requester(client, args.path, params)
        for _ in range(args.warmup):
            try:
                request()
            except Exception as exc:  # noqa: BLE001 - warm-up failure is worth seeing, not fatal
                print(f"warm-up: {type(exc).__name__}: {exc}")
        sampler = GpuSampler() if args.gpu else None
        with sampler or contextlib.nullcontext():
            if args.mode == "open":
                run = open_loop(request, rate=args.rate, count=args.count,
                                duration=args.duration, arrivals=args.arrivals,
                                seed=args.seed, max_in_flight=args.max_in_flight)
            else:
                run = closed_loop(request, workers=args.workers, count=args.count,
                                  duration=args.duration)
    gpu = sampler.report() if sampler else {}
    summary = run.summary()
    _print(summary, gpu)
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": summary.as_dict(), "params": run.params, "gpu": gpu,
             "target": {"url": args.url, "path": args.path, "generate": not args.no_generate},
             "samples": [asdict(s) for s in run.samples]}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
