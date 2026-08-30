"""Structured logging, and the failures that make a log useless exactly when it is needed.

Every test here is about a way logs go wrong under pressure: a line that raises inside the
handler, a line that loses its correlation id on the error path, a duplicate handler that
doubles an error rate a dashboard is counting.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from warrant.observe.logging import (
    HumanFormatter,
    JsonFormatter,
    configure,
    request_id,
    trace_id,
)


@pytest.fixture
def stream() -> io.StringIO:
    buf = io.StringIO()
    configure(json_output=True, stream=buf)
    yield buf
    logging.getLogger().handlers.clear()


def lines(buf: io.StringIO) -> list[dict]:
    return [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.strip()]


def test_extra_fields_become_queryable_fields(stream: io.StringIO):
    """`log.info("scored %s", query)` produces a message no aggregator can group and a
    field no query can filter on. The message stays constant; the variables are fields."""
    logging.getLogger("t").info("retrieved", extra={"admitted": 9627, "stage": "lexical"})
    record = lines(stream)[0]
    assert record["msg"] == "retrieved"
    assert record["admitted"] == 9627
    assert record["stage"] == "lexical"


def test_the_correlation_ids_are_attached_without_being_passed(stream: io.StringIO):
    token = request_id.set("abc123")
    try:
        logging.getLogger("t").info("inside a request")
    finally:
        request_id.reset(token)
    assert lines(stream)[0]["request_id"] == "abc123"


def test_a_request_id_without_a_trace_id_is_a_meaningful_state(stream: io.StringIO):
    """A request rejected by admission control has a request id and no trace id. That
    asymmetry is the difference between 'we answered wrongly' and 'we never got to answer',
    so an empty trace id must be absent rather than emitted as an empty string."""
    token = request_id.set("abc123")
    try:
        logging.getLogger("t").warning("rejected", extra={"reason": "queue_full"})
    finally:
        request_id.reset(token)
    record = lines(stream)[0]
    assert record["request_id"] == "abc123"
    assert "trace_id" not in record


def test_an_unserialisable_value_does_not_lose_the_line(stream: io.StringIO):
    """A TypeError inside the handler is swallowed by `logging` and the line disappears --
    and the lines carrying odd objects are the ones written while something is already
    going wrong."""
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    logging.getLogger("t").info("odd", extra={"thing": Opaque()})
    assert lines(stream)[0]["thing"] == "<opaque>"


def test_exceptions_carry_their_traceback(stream: io.StringIO):
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("t").exception("failed")
    record = lines(stream)[0]
    assert "ValueError: boom" in record["exc"]


def test_configure_replaces_handlers_rather_than_adding(stream: io.StringIO):
    """Two handlers on one logger emit every line twice, which does not merely add noise:
    it doubles the apparent rate of any error a dashboard counts."""
    second = io.StringIO()
    configure(json_output=True, stream=second)
    logging.getLogger("t").info("once")
    assert len(lines(second)) == 1
    assert lines(stream) == []


def test_the_human_formatter_shows_the_same_fields():
    """A developer who has to pipe every local run through jq stops reading logs. Nothing
    may be visible in one mode and absent in the other."""
    record = logging.LogRecord("t", logging.INFO, "f", 1, "retrieved", None, None)
    record.admitted = 9627
    token = request_id.set("abc123")
    try:
        rendered = HumanFormatter().format(record)
    finally:
        request_id.reset(token)
    assert "retrieved" in rendered
    assert "admitted=9627" in rendered
    assert "request_id=abc123" in rendered


def test_json_and_human_agree_on_which_fields_exist():
    record = logging.LogRecord("t", logging.INFO, "f", 1, "m", None, None)
    record.stage = "fusion"
    token = trace_id.set("t-1")
    try:
        as_json = json.loads(JsonFormatter().format(record))
        as_text = HumanFormatter().format(record)
    finally:
        trace_id.reset(token)
    assert as_json["stage"] == "fusion" and "stage=fusion" in as_text
    assert as_json["trace_id"] == "t-1" and "trace_id=t-1" in as_text
