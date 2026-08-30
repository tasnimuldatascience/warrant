"""Prometheus metrics, emitted by hand.

No client library. The exposition format is four lines of text per series and has been
stable for a decade, and `prometheus_client` brings a process-wide default registry, a
multiprocess mode with its own file-backed collector, and a signal-handler-based HTTP server
-- none of which this service uses, all of which would have to be understood by anyone
reading it. A hundred lines that can be read in full is the better trade at this size.

Two decisions carry most of the value here:

**Labels are bounded by construction.** Every label value in this module comes from a fixed
set -- an endpoint from the route table, a status class, a stage name from the retrieval
pipeline. Nothing derived from a request body ever becomes a label. A metric labelled by
query text or request id is not a metric, it is a log line with a counter attached, and it
takes a Prometheus server down by cardinality explosion rather than by load. This is the
single most common way self-instrumented services fail, so the registry refuses a label
value it has not been told about rather than trusting call sites to remember.

**Histograms use explicit buckets chosen against measured latency.** The default buckets
most libraries ship start at 5ms and double; this service's p50 retrieval is 18.4ms and its
p95 generation is minutes, so those buckets would put almost every retrieval in one bucket
and every generation in +Inf, and the quantiles computed from them would be worthless in
exactly the two places anyone would look.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

#: Retrieval stages run in tens of milliseconds; generation runs in tens of seconds. One
#: bucket set cannot serve both without wasting resolution where it matters, so there are
#: two, and each is anchored on measured numbers rather than on a doubling sequence:
#: retrieval p50 is 18.4ms and p95 is around 40ms, so the buckets are dense there and thin
#: out past 250ms where the only question left is "how bad".
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1, 2.5, 5, 10, 20, 25, 40, 60, 100, 150, 250, 500, 1000, 2500, 5000,
)
#: Generation is serialised at a measured 29.2-29.9 tok/s over ~205 output tokens, so a
#: typical answer is 6.6s and the queue in front of it dominates. Buckets stay dense below
#: 10s where answers land and run to five minutes because the deadline does -- and because
#: overload was measured putting a 503's own p50 at 65s, which a bucket set ending at the
#: happy path would report as a single saturated +Inf.
GENERATE_BUCKETS_S: tuple[float, ...] = (1, 2.5, 5, 10, 20, 30, 45, 60, 90, 120, 300)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: Mapping[str, str]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(pairs.items()))
    return "{" + inner + "}"


@dataclass
class _Series:
    name: str
    help: str
    kind: str                       # counter | gauge | histogram
    label_names: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)
    #: histogram only: per-label-set bucket counts and sum
    counts: dict[tuple[str, ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[str, ...], float] = field(default_factory=dict)


class Registry:
    """A small, explicit metric registry.

    Series are declared up front. A call site that observes an undeclared series raises
    rather than creating one, because a metric that springs into existence on first use is a
    metric nobody wrote a dashboard for, and a typo in its name is invisible.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._series: dict[str, _Series] = {}

    # -- declaration -----------------------------------------------------------------

    def counter(self, name: str, help: str, labels: Sequence[str] = ()) -> None:
        self._declare(name, help, "counter", labels)

    def gauge(self, name: str, help: str, labels: Sequence[str] = ()) -> None:
        self._declare(name, help, "gauge", labels)

    def histogram(self, name: str, help: str, buckets: Sequence[float],
                  labels: Sequence[str] = ()) -> None:
        self._declare(name, help, "histogram", labels, tuple(buckets))

    def _declare(self, name: str, help: str, kind: str, labels: Sequence[str],
                 buckets: tuple[float, ...] = ()) -> None:
        with self._lock:
            if name in self._series:
                raise ValueError(f"{name} is already declared")
            self._series[name] = _Series(name=name, help=help, kind=kind,
                                         label_names=tuple(labels), buckets=buckets)

    # -- observation -----------------------------------------------------------------

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        series, key = self._resolve(name, labels)
        with self._lock:
            series.values[key] = series.values.get(key, 0.0) + amount

    def set(self, name: str, value: float, **labels: str) -> None:
        series, key = self._resolve(name, labels)
        with self._lock:
            series.values[key] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        series, key = self._resolve(name, labels)
        with self._lock:
            counts = series.counts.setdefault(key, [0] * (len(series.buckets) + 1))
            # Cumulative buckets are produced at render time; here each observation lands in
            # exactly one slot, with the last standing for +Inf. Counting cumulatively on
            # every observation would cost O(buckets) per call on the request path.
            slot = len(series.buckets)
            for i, edge in enumerate(series.buckets):
                if value <= edge:
                    slot = i
                    break
            counts[slot] += 1
            series.sums[key] = series.sums.get(key, 0.0) + value

    @contextmanager
    def timed(self, name: str, *, scale: float = 1.0, **labels: str) -> Iterator[None]:
        """Observe how long the block took, in whatever unit the histogram is declared in.

        ``scale`` converts from seconds: 1000 for a millisecond histogram, 1 for seconds.
        It is explicit because a histogram whose buckets are milliseconds and whose
        observations are seconds looks completely healthy and is completely wrong.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * scale, **labels)

    def _resolve(self, name: str, labels: Mapping[str, str]) -> tuple[_Series, tuple[str, ...]]:
        series = self._series.get(name)
        if series is None:
            raise KeyError(f"{name} was never declared; declare it at start-up")
        if tuple(sorted(labels)) != tuple(sorted(series.label_names)):
            raise ValueError(
                f"{name} expects labels {series.label_names}, got {tuple(sorted(labels))}")
        return series, tuple(labels[n] for n in series.label_names)

    # -- exposition ------------------------------------------------------------------

    def render(self) -> str:
        """The Prometheus text exposition format, version 0.0.4."""
        out: list[str] = []
        with self._lock:
            for series in self._series.values():
                out.append(f"# HELP {series.name} {series.help}")
                out.append(f"# TYPE {series.name} {series.kind}")
                if series.kind == "histogram":
                    for key, counts in sorted(series.counts.items()):
                        pairs = dict(zip(series.label_names, key, strict=True))
                        running = 0
                        for edge, count in zip(series.buckets, counts, strict=False):
                            running += count
                            le = {**pairs, "le": _format(edge)}
                            out.append(f"{series.name}_bucket{_labels(le)} {running}")
                        running += counts[-1]
                        le = {**pairs, "le": "+Inf"}
                        out.append(f"{series.name}_bucket{_labels(le)} {running}")
                        out.append(f"{series.name}_sum{_labels(pairs)} "
                                   f"{series.sums.get(key, 0.0):g}")
                        out.append(f"{series.name}_count{_labels(pairs)} {running}")
                    continue
                for key, value in sorted(series.values.items()):
                    pairs = dict(zip(series.label_names, key, strict=True))
                    out.append(f"{series.name}{_labels(pairs)} {value:g}")
        return "\n".join(out) + "\n"


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(value)


#: The service's metrics, declared once. Kept in one place rather than beside each call site
#: so the full set a dashboard can rely on is readable without grepping, and so a rename
#: cannot half-happen.
def build() -> Registry:
    r = Registry()
    r.counter("warrant_requests_total", "HTTP requests served.", ["endpoint", "status"])
    r.histogram("warrant_request_duration_ms", "Wall time per HTTP request.",
                LATENCY_BUCKETS_MS, ["endpoint"])
    r.histogram("warrant_stage_duration_ms",
                "Wall time per retrieval stage, as recorded on the trace.",
                LATENCY_BUCKETS_MS, ["stage"])
    r.histogram("warrant_generate_duration_s", "Wall time per generation call.",
                GENERATE_BUCKETS_S)
    r.counter("warrant_admission_rejected_total",
              "Requests refused by admission control rather than queued.", ["reason"])
    r.counter("warrant_cache_total", "Answer cache lookups by outcome.", ["outcome"])
    r.gauge("warrant_corpus_chunks", "Chunk versions the store currently believes.")
    r.gauge("warrant_uncovered_chunks",
            "Believed chunks the dense index has no vector for.")
    r.gauge("warrant_ready", "1 when this process can answer, 0 otherwise.")
    return r
