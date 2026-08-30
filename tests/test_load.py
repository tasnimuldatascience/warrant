"""The load generator's own logic. Nothing here loads anything.

A harness that is only exercised by pointing it at a server is a harness whose bugs look
exactly like server behaviour -- which is the failure this file exists to prevent. Every test
below runs on a virtual clock with a fake server, so the suite is deterministic and contains
no sleeps.

The load-bearing test is `test_slow_server_shows_what_closed_loop_hides`: the same fake server
is measured both ways and the two readings differ by 18x. If that ever stops holding, the
open-loop scheduler has quietly started waiting for the server, and every capacity number in
docs/results/eval-010-capacity.md would become a service time wearing a response time's name.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import Future
from typing import Any

import pytest

from warrant.bench.load import (
    ManualClock,
    Sample,
    arrival_offsets,
    ask_params,
    closed_loop,
    http_requester,
    open_loop,
    percentile,
    summarize,
)


class InlineExecutor:
    """Runs work on the calling thread.

    Paired with `ManualClock` this models a server with exactly one slot serving in arrival
    order: request k+1 cannot start until request k has finished advancing virtual time. It
    is the whole reason the assertions below are equalities rather than tolerances.
    """

    def submit(self, fn, /, *args) -> Future:
        f: Future = Future()
        try:
            f.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 - mirror what a real pool does
            f.set_exception(exc)
        return f


class NeverFinishesExecutor:
    """Accepts every submission and completes none, so in-flight only ever grows.

    From a client's side that is what a saturated server looks like: the connection was
    accepted and the answer never came. Held work is run by `drain` afterwards.
    """

    def __init__(self) -> None:
        self.held: list[tuple[Any, tuple[Any, ...]]] = []

    def submit(self, fn, /, *args) -> Future:
        self.held.append((fn, args))
        # Already done, so the scheduler's join returns; the work itself has not run, so the
        # in-flight counter it decrements never drops.
        f: Future = Future()
        f.set_result(None)
        return f

    def drain(self) -> None:
        for fn, args in self.held:
            fn(*args)


def slow_server(clock: ManualClock, seconds: float, status: int = 200):
    """A server that costs a fixed amount of virtual time per request."""

    def _request() -> tuple[int, int]:
        clock.advance(seconds)
        return status, 0

    return _request


# -- percentiles ------------------------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 90) == 9
    assert percentile(values, 99) == 10
    assert percentile(values, 100) == 10
    assert percentile(values, 0) == 1


def test_percentile_returns_an_observation_never_an_interpolation():
    # Linear interpolation would report 990 here -- a latency no request experienced, and
    # smaller than the slowest thing that actually happened.
    assert percentile([10.0, 1000.0], 99) == 1000.0


def test_percentile_of_one_sample_is_that_sample():
    assert percentile([7.5], 50) == percentile([7.5], 99) == 7.5


def test_percentile_is_order_independent():
    assert percentile([5, 1, 4, 2, 3], 50) == 3


def test_percentile_refuses_an_empty_series():
    with pytest.raises(ValueError, match="no samples"):
        percentile([], 50)


@pytest.mark.parametrize("q", [-1, 101])
def test_percentile_refuses_a_quantile_outside_the_range(q):
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1.0], q)


# -- the arrival schedule ---------------------------------------------------------------


def test_uniform_offsets_are_evenly_spaced():
    assert arrival_offsets(4, count=5) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_uniform_schedule_does_not_drift():
    """i/rate, not a running total.

    Accumulating 1/3 a thousand times lands ~5e-14 away from 333.0. That error is harmless on
    its own; what is not harmless is the habit -- an accumulated schedule also inherits every
    millisecond the scheduler was late, which silently lowers the offered rate the run then
    reports as if it had been asked for.
    """
    offsets = arrival_offsets(3, count=1000)
    assert offsets[-1] == 333.0

    accumulated = 0.0
    for _ in range(999):
        accumulated += 1 / 3
    assert accumulated != 333.0


def test_duration_form_covers_the_window():
    assert arrival_offsets(2, duration=5) == [i / 2 for i in range(10)]


def test_poisson_arrivals_are_seeded_and_increasing():
    a = arrival_offsets(10, count=200, arrivals="poisson", seed=7)
    b = arrival_offsets(10, count=200, arrivals="poisson", seed=7)
    c = arrival_offsets(10, count=200, arrivals="poisson", seed=8)
    assert a == b != c
    assert a == sorted(a)
    assert a[0] == 0.0


def test_schedule_refuses_an_ambiguous_budget():
    with pytest.raises(ValueError, match="exactly one"):
        arrival_offsets(1, count=10, duration=10)
    with pytest.raises(ValueError, match="exactly one"):
        arrival_offsets(1)


def test_schedule_refuses_a_nonpositive_rate():
    with pytest.raises(ValueError, match="positive"):
        arrival_offsets(0, count=1)


def test_unknown_arrival_process_is_rejected():
    with pytest.raises(ValueError, match="deterministic"):
        arrival_offsets(1, count=1, arrivals="deterministic")


# -- the clock seam ---------------------------------------------------------------------


def test_sleep_until_is_idempotent():
    """Two waiters on the same instant must not advance virtual time twice.

    This is why the Clock protocol is `sleep_until(deadline)` and not `sleep(seconds)`: with
    a duration, the scheduler and a worker both waiting for t=1.0 would leave the clock at
    2.0, and every latency measured afterwards would be wrong by a factor nobody could see.
    """
    clock = ManualClock()
    clock.sleep_until(1.0)
    clock.sleep_until(1.0)
    clock.sleep_until(0.5)
    assert clock.now() == 1.0


# -- open loop --------------------------------------------------------------------------


def test_open_loop_intended_times_ignore_the_server():
    """The schedule is what was asked for, not what was achieved."""
    clock = ManualClock()
    run = open_loop(slow_server(clock, 1.0), rate=10, count=5, clock=clock,
                    pool=InlineExecutor())
    assert [s.intended for s in run.samples] == [0.0, 0.1, 0.2, 0.3, 0.4]
    # Every request took the same 1.0 s to serve; only the queue in front of it grew.
    assert all(s.service == 1.0 for s in run.samples)


def test_open_loop_records_latency_from_the_intended_arrival():
    clock = ManualClock()
    run = open_loop(slow_server(clock, 1.0), rate=10, count=5, clock=clock,
                    pool=InlineExecutor())
    # request k is sent at k s and answered at k+1 s, having been due at 0.1k.
    assert [s.latency for s in run.samples] == pytest.approx(
        [1.0, 1.9, 2.8, 3.7, 4.6])


def test_slow_server_shows_what_closed_loop_hides():
    """The same server, measured twice. This is coordinated omission, quantified.

    A one-second-per-request server is offered 10 req/s. Closed-loop with one worker calls it
    healthy at a 1.0 s p99, because 1.0 s is what one request costs and it never asks for a
    second one until the first is back. Open-loop keeps its schedule and reports 18.1 s,
    because that is what the twentieth caller actually waited. Both are correct answers to
    different questions, and only one of them is a capacity measurement.
    """
    closed_clock = ManualClock()
    closed = closed_loop(slow_server(closed_clock, 1.0), workers=1, count=20,
                         clock=closed_clock, pool=InlineExecutor()).summary()

    open_clock = ManualClock()
    opened = open_loop(slow_server(open_clock, 1.0), rate=10, count=20, clock=open_clock,
                       pool=InlineExecutor()).summary()

    assert closed.latency["p99"] == 1.0
    assert opened.latency["p99"] == pytest.approx(18.1)
    assert opened.latency["p99"] > 18 * closed.latency["p99"]

    # Both drained the same 20 requests in the same 20 s. Throughput is not where the two
    # disagree -- the tail is, and throughput alone would have shown nothing.
    assert closed.achieved_rate == opened.achieved_rate == pytest.approx(1.0)
    assert opened.offered_rate == 10.0

    # The queue is charged to the harness here, because an inline pool is a client with one
    # connection. Same seconds, same conclusion: closed-loop simply never recorded them.
    assert opened.wait_p99 == pytest.approx(17.1)
    assert opened.omission > 18


def test_open_loop_latency_ramps_when_the_server_is_over_capacity():
    """A growing queue is visible within one run, without needing a second one to compare."""
    clock = ManualClock()
    fast = open_loop(slow_server(clock, 0.01), rate=10, count=40, clock=clock,
                     pool=InlineExecutor()).summary()
    clock = ManualClock()
    slow = open_loop(slow_server(clock, 1.0), rate=10, count=40, clock=clock,
                     pool=InlineExecutor()).summary()
    assert fast.latency_ramp == pytest.approx(0.0, abs=1e-9)
    assert slow.latency_ramp > 20


def test_closed_loop_cannot_measure_omission_by_construction():
    clock = ManualClock()
    summary = closed_loop(slow_server(clock, 1.0), workers=1, count=5, clock=clock,
                          pool=InlineExecutor()).summary()
    assert summary.omission == 1.0
    assert summary.wait_p99 == 0.0
    assert summary.service == summary.latency


def test_closed_loop_stops_on_the_duration_budget():
    clock = ManualClock()
    run = closed_loop(slow_server(clock, 1.0), workers=1, duration=5.0, clock=clock,
                      pool=InlineExecutor())
    assert len(run.samples) == 5
    assert run.wall == 5.0


def test_closed_loop_refuses_an_ambiguous_budget():
    with pytest.raises(ValueError, match="exactly one"):
        closed_loop(lambda: (200, 0), workers=1)
    with pytest.raises(ValueError, match="workers"):
        closed_loop(lambda: (200, 0), workers=0, count=1)


# -- shedding and errors ----------------------------------------------------------------


def test_shed_arrivals_are_counted_and_keep_their_intended_times():
    """A saturated harness must say so rather than silently lowering the offered load."""
    clock = ManualClock()
    pool = NeverFinishesExecutor()
    run = open_loop(lambda: (200, 0), rate=10, count=10, max_in_flight=3, clock=clock,
                    pool=pool)
    assert len(pool.held) == 3
    assert run.shed == pytest.approx([i / 10 for i in range(3, 10)])
    assert run.summary().shed == 7
    assert run.summary().n == 0

    pool.drain()
    assert len(run.samples) == 3


def test_an_exception_is_a_measurement_not_a_crash():
    def broken() -> tuple[int, int]:
        raise ConnectionResetError("peer went away")

    clock = ManualClock()
    summary = closed_loop(broken, workers=1, count=3, clock=clock,
                          pool=InlineExecutor()).summary()
    assert summary.n == 3
    assert summary.ok == 0
    assert summary.statuses == {"ConnectionResetError": 3}


def test_a_503_is_throughput_but_not_goodput():
    clock = ManualClock()
    summary = closed_loop(slow_server(clock, 1.0, status=503), workers=1, count=4,
                          clock=clock, pool=InlineExecutor()).summary()
    assert summary.n == 4
    assert summary.ok == 0
    assert summary.achieved_rate == pytest.approx(1.0)
    assert summary.goodput == 0.0
    assert summary.statuses == {"503": 4}


def test_summary_of_nothing_is_nan_rather_than_zero():
    """Zero would read as "instant". An empty run has no latency, and says so."""
    summary = summarize([], mode="open", wall=10.0, offered_rate=1.0)
    assert summary.n == 0
    assert math.isnan(summary.latency["p99"])
    assert math.isnan(summary.wait_p99)
    assert summary.goodput == 0.0


def test_sample_classifies_status_codes():
    ok = Sample(intended=0, sent=0, done=1, status=200)
    refused = Sample(intended=0, sent=0, done=1, status=503)
    broken = Sample(intended=0, sent=0, done=1, error="ReadTimeout: too slow")
    assert ok.ok and not refused.ok and not broken.ok
    assert ok.service == ok.latency == 1.0


# -- request construction ---------------------------------------------------------------


class _StubResponse:
    status_code = 200
    content = b"x" * 12


class _StubClient:
    def __init__(self) -> None:
        self.seen: list[dict] = []

    def get(self, path: str, params: dict) -> _StubResponse:
        self.seen.append(params)
        return _StubResponse()


def test_requester_rotates_through_its_parameters():
    """One query would measure the predicate cache instead of the predicate."""
    client = _StubClient()
    request = http_requester(client, "/api/ask", [{"q": "a"}, {"q": "b"}])
    assert [request() for _ in range(3)] == [(200, 12)] * 3
    assert [p["q"] for p in client.seen] == ["a", "b", "a"]


def test_requester_refuses_an_empty_parameter_list():
    with pytest.raises(ValueError, match="no request parameters"):
        http_requester(_StubClient(), "/api/ask", [])


def test_ask_params_crosses_queries_with_dates():
    params = ask_params(["one", "two"], ["2020-01-01", "2021-01-01"], generate=False)
    assert len(params) == 4
    assert {p["generate"] for p in params} == {"false"}
    assert {p["as_of"] for p in params} == {"2020-01-01", "2021-01-01"}


# -- thread safety of the recording path ------------------------------------------------


def test_samples_from_many_threads_are_all_recorded():
    """The results list is shared by every worker; losing samples under load would bias the
    result toward whatever finished first."""
    clock = ManualClock()
    counted = threading.Semaphore(0)

    def request() -> tuple[int, int]:
        counted.release()
        return 200, 0

    run = closed_loop(request, workers=8, count=400, clock=clock)
    assert len(run.samples) == 400
    assert sum(1 for _ in iter(lambda: counted.acquire(blocking=False), False)) == 400
