"""Metrics, and the two ways self-instrumentation usually goes wrong.

The interesting tests here are not "does the counter count". They are the cardinality
guard and the bucket-unit guard: a metric labelled by request path takes down the collector
scraping it, and a histogram whose buckets are milliseconds fed observations in seconds
looks perfectly healthy while being wrong by a factor of a thousand.
"""

from __future__ import annotations

import pytest

from warrant.serve.metrics import (
    GENERATE_BUCKETS_S,
    LATENCY_BUCKETS_MS,
    Registry,
    build,
)


@pytest.fixture
def reg() -> Registry:
    r = Registry()
    r.counter("reqs", "requests", ["endpoint", "status"])
    r.gauge("chunks", "corpus size")
    r.histogram("dur_ms", "duration", (1, 10, 100), ["stage"])
    return r


def test_an_undeclared_series_raises_rather_than_springing_into_existence(reg: Registry):
    """A metric created on first use is one nobody wrote a dashboard for, and a typo in its
    name is invisible -- it just quietly becomes a second, empty series."""
    with pytest.raises(KeyError):
        reg.inc("reqs_total", endpoint="/x", status="2xx")


def test_the_wrong_label_set_raises(reg: Registry):
    """Two series with the same name and different labels are not comparable and cannot be
    summed. Prometheus will accept them and every query over them will be wrong."""
    with pytest.raises(ValueError, match="expects labels"):
        reg.inc("reqs", endpoint="/x")


def test_histogram_buckets_are_cumulative_and_close_at_inf(reg: Registry):
    for value in (0.5, 5, 50, 5000):
        reg.observe("dur_ms", value, stage="lexical")
    text = reg.render()
    assert 'dur_ms_bucket{le="1",stage="lexical"} 1' in text
    assert 'dur_ms_bucket{le="10",stage="lexical"} 2' in text
    assert 'dur_ms_bucket{le="100",stage="lexical"} 3' in text
    assert 'dur_ms_bucket{le="+Inf",stage="lexical"} 4' in text
    assert "dur_ms_count{stage=\"lexical\"} 4" in text
    assert "dur_ms_sum{stage=\"lexical\"} 5055.5" in text


def test_label_values_are_escaped(reg: Registry):
    """An unescaped quote in a label value produces a line Prometheus rejects, and it
    rejects the whole scrape, not the one line."""
    reg.inc("reqs", endpoint='/a"b', status="2xx")
    assert '/a\\"b' in reg.render()


def test_timed_scales_to_the_declared_unit(reg: Registry):
    """A millisecond histogram fed seconds looks healthy and is wrong by 1000x, which is
    why the conversion is a required argument rather than an assumed one."""
    with reg.timed("dur_ms", scale=1000.0, stage="dense"):
        pass
    text = reg.render()
    assert 'dur_ms_count{stage="dense"} 1' in text
    # Anything under a millisecond lands in the first bucket; seconds would not.
    assert 'dur_ms_bucket{le="1",stage="dense"} 1' in text


def test_declaring_a_name_twice_raises(reg: Registry):
    with pytest.raises(ValueError, match="already declared"):
        reg.counter("reqs", "requests again", ["endpoint", "status"])


def test_the_service_buckets_straddle_the_measured_latencies():
    """Bucket edges are a measurement decision, not a default. Retrieval p50 is 18.4ms and
    generation is tens of seconds; a doubling sequence starting at 5ms puts nearly every
    retrieval in one bucket and every generation in +Inf, and the quantiles read off them
    are then worthless in exactly the two places anyone looks."""
    assert min(LATENCY_BUCKETS_MS) <= 18.4 <= max(LATENCY_BUCKETS_MS)
    below = [b for b in LATENCY_BUCKETS_MS if b <= 40]
    assert len(below) >= 6, "not enough resolution around the measured retrieval range"
    # 21.3 tok/s means a 400-token answer is ~19s; the buckets must not end before that.
    assert max(GENERATE_BUCKETS_S) >= 120


def test_the_service_registry_declares_no_unbounded_label():
    """Every label in this service comes from a fixed set -- a route template, a status
    class, a stage name. Nothing derived from a request body may become one, because that
    is a log line with a counter attached, and it fails by cardinality rather than by load.
    """
    allowed = {"endpoint", "status", "stage", "reason", "outcome"}
    registry = build()
    for series in registry._series.values():
        assert set(series.label_names) <= allowed, series.name
