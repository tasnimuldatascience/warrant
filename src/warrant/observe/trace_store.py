"""Durable traces: one row per request, plus every candidate each stage ranked.

A trace that lives only in the process that produced it can explain a request while that
request is still on screen and nothing afterwards. Persisting it is what turns the trace from
a debugging aid into the primary artifact ARCHITECTURE.md section 8 claims it is -- inspecting
a past decision, and diffing today's pipeline against real traffic, both start by reading a
trace back whole, long after the retriever that produced it has been garbage collected and the
index it ran against has been rebuilt.

**A separate database file from the corpus.** The chunk store is append-only law, rebuilt by
``make build`` and copied between machines; traces are request telemetry that grows without
bound and is safe to delete. Putting them in one file would mean either shipping request logs
with the corpus or truncating the corpus to drop logs. The path is an explicit argument rather
than a config field for the same reason: what to keep and where is the caller's policy.

**Model names and the config hash are recorded, not derived.** A trace that cannot say what
produced it cannot be replayed against anything -- a diff between it and today's pipeline
would report that the answer changed without being able to say whether the config, the
encoder, or the corpus moved. `warrant.retrieve.hybrid.Retriever` stamps both onto every
trace it produces.

The generated columns -- prompt, context, answer -- are handed in by the caller as plain JSON,
and nothing here imports the generator. Retrieval-only installs have no torch, and a trace
store that could not be opened without it would be unusable in exactly the environment where
most traces are recorded.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..index.store import SchemaMismatch, now
from ..retrieve.hybrid import STAGES, Candidate, Trace

#: Bump when a stored trace can no longer be read by this build. Same reasoning as the chunk
#: store's version: every statement below is IF NOT EXISTS, so an older file would otherwise
#: open cleanly and fail much later inside a query that names neither cause nor cure.
TRACE_SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trace (
    trace_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,      -- ISO timestamp, when the request was recorded

    query           TEXT NOT NULL,
    as_of           TEXT NOT NULL,      -- valid time the question was asked about
    system_time     TEXT,               -- belief time; NULL = "as believed when it ran"
    scope           TEXT NOT NULL,      -- the human description
    scope_facets    TEXT NOT NULL,      -- JSON; the machine form replay rebuilds from
    excluded_parts  TEXT NOT NULL,      -- JSON
    admitted        INTEGER NOT NULL,   -- rows surviving the predicates

    config_hash     TEXT NOT NULL,
    models          TEXT NOT NULL,      -- JSON {stage: model name}
    timings         TEXT NOT NULL,      -- JSON {stage: milliseconds}

    prompt          TEXT,               -- generation, supplied by the caller
    context         TEXT,               -- JSON
    answer          TEXT                -- JSON: claims, citations, verification verdicts
);

CREATE TABLE IF NOT EXISTS trace_candidate (
    trace_id        TEXT NOT NULL REFERENCES trace(trace_id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,
    rank            INTEGER NOT NULL,   -- 1-based, within its stage
    version_id      TEXT NOT NULL,
    -- NULL where the stage ordered without scoring. Distinct from 0.0, which several
    -- scorers can legitimately produce.
    score           REAL,
    PRIMARY KEY (trace_id, stage, rank)
);

CREATE INDEX IF NOT EXISTS trace_created ON trace (created_at);
CREATE INDEX IF NOT EXISTS trace_candidate_version ON trace_candidate (version_id);
"""


@dataclass(frozen=True)
class StoredTrace:
    """A trace read back whole, with the scores and timings it was recorded at.

    Everything needed to describe what happened on one request is here, and nothing here
    requires the corpus, the index, or the retriever to still exist.
    """

    trace_id: str
    created_at: str
    query: str
    as_of: str
    scope: str
    scope_facets: dict[str, str] = field(default_factory=dict)
    system_time: str | None = None
    excluded_parts: list[str] = field(default_factory=list)
    admitted: int = 0
    config_hash: str = ""
    models: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    stages: dict[str, list[Candidate]] = field(default_factory=dict)
    prompt: str | None = None
    context: Any = None
    answer: Any = None

    def candidates(self, stage: str) -> list[Candidate]:
        return list(self.stages.get(stage, ()))

    def ids(self, stage: str) -> list[str]:
        return [c.version_id for c in self.stages.get(stage, ())]

    @property
    def final(self) -> list[str]:
        return self.ids("final")

    def to_trace(self) -> Trace:
        """Rebuild the in-memory `Trace`, so a stored request can be re-examined.

        This is what makes a stored trace worth more than a log line: every consumer that
        takes a live trace -- the failure autopsy above all -- takes this one unchanged, and
        can localise a stage failure on a request that ran last month without re-running a
        single query against the index.
        """
        return Trace(
            query=self.query, as_of=self.as_of, scope=self.scope,
            scope_facets=dict(self.scope_facets), system_time=self.system_time,
            excluded_parts=list(self.excluded_parts), admitted=self.admitted,
            config_hash=self.config_hash, models=dict(self.models),
            timings=dict(self.timings),
            **{stage: self.candidates(stage) for stage in STAGES},
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for an API response or a written artifact."""
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "query": self.query,
            "as_of": self.as_of,
            "system_time": self.system_time,
            "scope": self.scope,
            "scope_facets": dict(self.scope_facets),
            "excluded_parts": list(self.excluded_parts),
            "admitted": self.admitted,
            "config_hash": self.config_hash,
            "models": dict(self.models),
            "timings": dict(self.timings),
            "stages": {
                stage: [{"version_id": c.version_id, "score": c.score, "rank": c.rank}
                        for c in self.candidates(stage)]
                for stage in STAGES if self.stages.get(stage)
            },
            "prompt": self.prompt,
            "context": self.context,
            "answer": self.answer,
        }


class TraceStore:
    """SQLite store for traces, in its own file.

    Connections are thread-local for the same reason `warrant.index.store.Store` makes them
    thread-local: a sqlite3 connection may only be used from the thread that created it, and
    the API records traces from a threadpool. An in-memory store is again the exception --
    each ``:memory:`` connection is its own empty database -- so that connection is shared.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._memory = str(path) == ":memory:"
        if not self._memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = self._connect() if self._memory else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path) if not self._memory else ":memory:")
        conn.row_factory = sqlite3.Row
        existing = conn.execute("PRAGMA user_version").fetchone()[0]
        written = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trace'"
        ).fetchone() is not None
        if written and existing != TRACE_SCHEMA_VERSION:
            conn.close()
            raise SchemaMismatch(
                f"{self.path} holds traces written by schema v{existing}, this build expects "
                f"v{TRACE_SCHEMA_VERSION}. Traces are telemetry, not corpus: delete the file "
                f"if the history is not worth migrating.")
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {TRACE_SCHEMA_VERSION}")
        return conn

    @property
    def db(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def close(self) -> None:
        conn = self._shared or getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._shared = None
        self._local = threading.local()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------------

    def record(self, trace: Trace, *, trace_id: str | None = None,
               created_at: str | None = None, prompt: str | None = None,
               context: Any = None, answer: Any = None) -> str:
        """Persist one request and return its trace id.

        ``prompt``, ``context`` and ``answer`` are whatever the caller has: this module does
        not reach into the generator to find them, so a retrieval-only run records a complete
        retrieval trace and leaves them null rather than failing to import torch.

        The id is random rather than derived from the query. Two identical questions asked a
        month apart are two requests with two different answers, and collapsing them onto one
        row would destroy the second the moment it mattered -- so a collision raises
        ``IntegrityError`` from the PRIMARY KEY instead of silently overwriting.
        """
        tid = trace_id or uuid.uuid4().hex[:16]
        rows = [(tid, stage, c.rank, c.version_id, c.score)
                for stage in STAGES for c in trace.candidates(stage)]
        with self.db as db:
            db.execute(
                "INSERT INTO trace (trace_id, created_at, query, as_of, system_time, scope, "
                "scope_facets, excluded_parts, admitted, config_hash, models, timings, "
                "prompt, context, answer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, created_at or now(), trace.query, trace.as_of, trace.system_time,
                 trace.scope, json.dumps(trace.scope_facets),
                 json.dumps(trace.excluded_parts), trace.admitted, trace.config_hash,
                 json.dumps(trace.models), json.dumps(trace.timings), prompt,
                 None if context is None else json.dumps(context),
                 None if answer is None else json.dumps(answer)),
            )
            db.executemany(
                "INSERT INTO trace_candidate (trace_id, stage, rank, version_id, score) "
                "VALUES (?,?,?,?,?)", rows)
        return tid

    def delete(self, trace_id: str) -> int:
        """Forget one trace. Telemetry is deletable; that is half of why it lives here."""
        with self.db as db:
            db.execute("DELETE FROM trace_candidate WHERE trace_id = ?", (trace_id,))
            cur = db.execute("DELETE FROM trace WHERE trace_id = ?", (trace_id,))
        return cur.rowcount

    # -- reading -----------------------------------------------------------------

    def load(self, trace_id: str) -> StoredTrace:
        """One trace, whole. Raises ``KeyError`` if it was never recorded or was deleted."""
        row = self.db.execute("SELECT * FROM trace WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(trace_id)
        return _stored(row, self._candidates([trace_id]).get(trace_id, {}))

    def recent(self, limit: int = 20) -> list[StoredTrace]:
        """The newest traces, newest first, loaded whole.

        Whole rather than as headers: a trace is a few hundred candidate rows, and a listing
        that had to be re-queried per row to say anything useful is a listing that will be.
        """
        rows = self.db.execute(
            "SELECT * FROM trace ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,)).fetchall()
        stages = self._candidates([r["trace_id"] for r in rows])
        return [_stored(r, stages.get(r["trace_id"], {})) for r in rows]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM trace").fetchone()[0]

    def _candidates(self, trace_ids: Sequence[str]) -> dict[str, dict[str, list[Candidate]]]:
        """Every stage list for several traces in one query, rather than one query per row."""
        if not trace_ids:
            return {}
        marks = ",".join("?" * len(trace_ids))
        out: dict[str, dict[str, list[Candidate]]] = {}
        for r in self.db.execute(
            f"SELECT trace_id, stage, rank, version_id, score FROM trace_candidate "
            f"WHERE trace_id IN ({marks}) ORDER BY stage, rank", list(trace_ids)
        ):
            stage = out.setdefault(r["trace_id"], {}).setdefault(r["stage"], [])
            stage.append(Candidate(r["version_id"], r["score"], r["rank"]))
        return out


def _stored(row: sqlite3.Row, stages: dict[str, list[Candidate]]) -> StoredTrace:
    return StoredTrace(
        trace_id=row["trace_id"],
        created_at=row["created_at"],
        query=row["query"],
        as_of=row["as_of"],
        scope=row["scope"],
        scope_facets=json.loads(row["scope_facets"]),
        system_time=row["system_time"],
        excluded_parts=json.loads(row["excluded_parts"]),
        admitted=row["admitted"],
        config_hash=row["config_hash"],
        models=json.loads(row["models"]),
        timings=json.loads(row["timings"]),
        stages={stage: list(cands) for stage, cands in stages.items()},
        prompt=row["prompt"],
        context=None if row["context"] is None else json.loads(row["context"]),
        answer=None if row["answer"] is None else json.loads(row["answer"]),
    )
