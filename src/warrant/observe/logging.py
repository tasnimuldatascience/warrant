"""Structured logs, correlated to the request and to the replayable trace.

The question this module exists to answer is the one a maintainer actually gets: *a user
says the answer they got on Tuesday was wrong, and quotes an id — show me everything that
happened.* Answering it needs two ids on every line and one format a machine can read.

**Two ids, not one, because they answer different questions.** ``request_id`` is minted per
HTTP request and is what the caller sees in the ``X-Request-ID`` header; it groups the log
lines of one call, including the ones written before anything was retrieved and the ones
written when the request failed. ``trace_id`` names a *persisted artifact* that can be
replayed. A request that was rejected by admission control has a request id and no trace
id, and that asymmetry is information: it is precisely the difference between "we did work
and it was wrong" and "we never got to do the work".

**Ids arrive by contextvar, not by argument.** Threading an id through every function that
might log is the kind of change that is 95% done forever, and the 5% that never gets done is
always the error path. A contextvar is copied into the thread Starlette runs synchronous
endpoints on, so a log line written deep inside retrieval carries the id without retrieval
knowing an HTTP layer exists.

**JSON, and no interpolation into the message.** ``log.info("scored %s", query)`` produces a
message no aggregator can group and a field no query can filter on. The message stays a
constant string and everything variable goes in ``extra``, which is what makes "show me
every request where admission control rejected" a query rather than a grep.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

#: Set by the API middleware for the lifetime of one request. Empty outside one -- the CLI
#: and the eval harness log through the same handler and simply have no request to name.
request_id: ContextVar[str] = ContextVar("request_id", default="")
trace_id: ContextVar[str] = ContextVar("trace_id", default="")

#: LogRecord attributes that are not ours. Anything else a caller passes in `extra` is
#: emitted as a field, which is what makes the log queryable; this set is what keeps
#: Python's own record bookkeeping from being emitted alongside it.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation ids folded in.

    ``default=str`` on the dump is deliberate: a log line must never be the thing that
    raises. An un-serialisable value in ``extra`` becomes its repr rather than a
    ``TypeError`` inside the handler, which would be swallowed by ``logging`` and lose the
    line entirely -- and the lines most likely to carry an odd object are the ones written
    while something is already going wrong.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if rid := request_id.get():
            payload["request_id"] = rid
        if tid := trace_id.get():
            payload["trace_id"] = tid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """The same fields, for a terminal.

    A developer reading a build is not served by JSON, and a developer who has to pipe every
    local run through `jq` stops reading logs. The fields are the same ones; only the
    rendering differs, so nothing is visible in one mode and absent in the other.
    """

    def format(self, record: logging.LogRecord) -> str:
        head = f"{record.levelname:<7} {record.name} {record.getMessage()}"
        rid = request_id.get()
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in _RESERVED and not k.startswith("_")}
        if rid:
            extras = {"request_id": rid, **extras}
        if tid := trace_id.get():
            extras = {"trace_id": tid, **extras}
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        line = f"{head}  {tail}" if tail else head
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure(*, level: str = "INFO", json_output: bool = True,
              stream: Any = None) -> None:
    """Install one handler on the root logger, replacing whatever was there.

    Replacing rather than adding: uvicorn installs its own handlers, and a second one on the
    same logger produces every line twice -- which is not merely noisy, it doubles the
    apparent rate of any error a dashboard counts. Uvicorn's own loggers are made to
    propagate to root so its access lines get the same format and the same correlation ids
    as everything else.
    """
    formatter: logging.Formatter = JsonFormatter() if json_output else HumanFormatter()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True

    # httpx logs a full INFO line per request. During a corpus fetch that is one line per
    # snapshot for thousands of snapshots, which buries everything this project logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
