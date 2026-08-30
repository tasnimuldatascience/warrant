"""Cached answers that expire when the store says they stop being true.

Every RAG system caches answers and every one of them guesses at a TTL, because nothing in
the system knows when a cached answer stops being correct. Five minutes is picked because it
felt safe; an hour because five minutes was not paying; and either way the number is a bet
against a corpus nobody is watching. Warrant does not have to guess.

An answer here is a function of ``(query, scope, as_of, config)`` over a specific set of
evidence *versions*, and `warrant.index.store` records exactly when each of those versions
ceases to be in force (``valid_to``) and when the system stops believing it (``system_to``).
So the expiry is **computed, not configured**:

    expires_at = min(valid_to) over the cited versions, NULL where they are all open

and, independently of any date, an entry is dead the moment any cited version is no longer
believed -- a retraction, or a corrected parse -- because the answer was assembled from text
the system now says was wrong.

Cache invalidation stops being the hard problem when your store knows when its facts stop
being true.

**One correction to the naive rule.** A cached answer answers a question about a *date*, and
valid time does not move: the versions in force on 2019-03-01 are in force on 2019-03-01
forever. So a bound that has already passed at write time -- a historical question, whose
cited text was superseded years ago -- is not an expiry at all; it describes a closed interval
that nothing can reopen, and such an entry has no valid-time expiry. A bound in the *future*
is kept even for a historical question, where it is merely conservative: over-expiry costs one
recomputation, under-expiry serves a wrong answer, and that asymmetry decides which way to
round.

**Only for the live serving path.** Entries are valid "as believed now". A replay pinned to a
past ``system_time`` (`warrant.retrieve.hybrid.Retriever.retrieve`'s ``system_time``) must not
read this cache: belief invalidation below is defined against *current* belief, so an entry
that is correctly dead for a live request may be exactly what a replay is asking for.

**Its own SQLite file**, neither the corpus store nor the trace store. A cache is disposable
by definition -- deleting it costs latency and nothing else -- while the corpus is rebuilt
wholesale by ``make build`` and traces are the primary artifact ARCHITECTURE.md section 8 is
built on. Sharing a file would mean a cache flush could not be a ``rm``, and that a schema
bump here forced a decision about data that deserves better.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..index.store import Store
from ..index.store import now as utc_now
from ..retrieve.hybrid import fts_query
from ..retrieve.scope import Scope

#: Bump when a stored entry can no longer be read by this build. Unlike the corpus and trace
#: stores, a mismatch here is *not* an error: those refuse to open and tell an operator to
#: delete the file, because their contents are worth a deliberate decision. A cache's contents
#: are worth nothing, so a mismatch drops the tables and starts empty -- the one thing a cache
#: must never do is make a deployment fail.
CACHE_SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entry (
    key             TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,      -- ISO instant
    -- LRU order is a sequence, not a clock. Two hits inside the same second tie on any
    -- timestamp this codebase writes (store.now() is second-resolution), and an LRU that
    -- breaks ties arbitrarily evicts the wrong one of the two entries it was asked about.
    used_seq        INTEGER NOT NULL,
    -- Computed on write; NULL means "no valid-time bound", using the same open-interval
    -- convention as chunk.valid_to rather than a sentinel date (see index.store.OPEN).
    expires_at      TEXT,

    -- Descriptive only: the key above already commits to all four. Kept so that a human
    -- reading this file can tell what an opaque key stands for.
    query           TEXT NOT NULL DEFAULT '',
    as_of           TEXT NOT NULL DEFAULT '',
    config_hash     TEXT NOT NULL DEFAULT '',

    payload         TEXT NOT NULL       -- JSON, whatever the caller stored
);

CREATE TABLE IF NOT EXISTS entry_citation (
    key             TEXT NOT NULL REFERENCES entry(key) ON DELETE CASCADE,
    version_id      TEXT NOT NULL,
    -- The text as it was when the answer was written. A retraction followed by a corrected
    -- parse re-inserts the *same* version_id with different text, so a version_id that is
    -- believed again is not evidence that the answer still stands -- the hash is.
    content_hash    TEXT NOT NULL,
    doc_id          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (key, version_id)
);

CREATE TABLE IF NOT EXISTS counter (
    name            TEXT PRIMARY KEY,
    n               INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS entry_lru     ON entry (used_seq);
CREATE INDEX IF NOT EXISTS entry_expiry  ON entry (expires_at);
CREATE INDEX IF NOT EXISTS citation_doc  ON entry_citation (doc_id);
CREATE INDEX IF NOT EXISTS citation_ver  ON entry_citation (version_id);
"""

#: Four ways a lookup can fail to serve, plus what the writer did. Deliberately not collapsed
#: into a hit rate: "the corpus was corrected under us", "the amendment landed", "nobody has
#: asked this before" and "we threw it away to stay inside the cap" are four different
#: problems with four different fixes, and one ratio hides which is happening.
COUNTERS = ("hits", "misses", "expired", "invalidated", "evicted", "writes", "refused")

#: Entries, not bytes: an AskResponse with eight evidence rows is 8-16 KB of JSON, so this cap
#: bounds the file at roughly 16-32 MB. The number matters because this project already had
#: one unbounded cache keyed on raw user text -- ~1.8 KB per entry, 1.7 GiB per million
#: distinct queries, held for the lifetime of the process (see hybrid.Retriever._dense). An
#: endpoint that anyone can call with arbitrary text must not be able to grow storage without
#: limit, and "we will watch it" is not a bound.
DEFAULT_MAX_ENTRIES = 2048

#: The cap that exists because the store cannot see everything. A guidance page can be
#: rewritten at its source without any version in this corpus closing, and an abstention
#: ("the excerpts do not answer this") is caused by the *absence* of evidence, which has no
#: valid_to at all. Those are the cases with no computable bound, and this is the only honest
#: thing to do about them. It is a ceiling, never a floor: the computed bound always wins when
#: it is sooner.
DEFAULT_MAX_TTL_SECONDS = 24 * 60 * 60.0


def _instant(value: str) -> str:
    """Canonical UTC ISO-8601 second, so that string comparison *is* time comparison.

    Every instant in this module is written through here, which is what makes the plain
    ``<`` in the expiry checks correct: the strings are fixed-width, UTC, and zero-padded, so
    lexicographic order is chronological order. Mixing ``valid_to`` (a bare date) with
    ``store.now()`` (a timestamp) without this would compare "2026-08-31" against
    "2026-08-31T00:00:00+00:00" as *less than*, and expire the entry a day early.

    A bare date becomes midnight UTC, which is the right reading of ``valid_to``: the
    interval is exclusive at that end, so the old text stops being law at the first instant
    of that day, not the last.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _after(instant: str, seconds: float) -> str:
    return _instant((datetime.fromisoformat(instant) + timedelta(seconds=seconds)).isoformat())


def normalise_query(query: str) -> str:
    """The cache's normal form for a query: exactly what the lexical stage runs.

    This is `warrant.retrieve.hybrid.fts_query` and nothing else, imported rather than
    reimplemented. A second normalisation written here would start identical and drift on the
    first change to either -- and a drifted key is not a bug that shows up as an exception, it
    is a cache that quietly stops hitting, or worse, one that merges two questions retrieval
    would answer differently.

    What that collapses is the same thing retrieval collapses: punctuation, repeated terms,
    and everything past the 64-token cap. Two queries with one normal form get one candidate
    set out of FTS5 and are the same question by the only definition this pipeline has.
    """
    return fts_query(query)


def cache_key(query: str, *, as_of: str, scope: Scope | Mapping[str, str],
              config_hash: str) -> str:
    """The identity of a cached answer: normalised query, scope facets, as_of, config hash.

    ``config_hash`` is in the key and is not optional. An answer produced under ``final_k=8``
    is not the answer a ``final_k=16`` build would give, and serving it because the query text
    matched is a silent regression with no symptom -- the endpoint returns 200, the claims are
    plausible, and the tuning result the failure budget just paid for is invisibly reverted
    for every repeat visitor. `warrant.config.Config.hash` is the value to pass.

    128 bits of the digest, not the 12 hex characters `Config.hash` uses. That hash labels a
    configuration for a human reading a trace; this one decides whether a stored answer is
    served in place of a computed one, and identity needs more room than a label.
    """
    facets = scope.facets if isinstance(scope, Scope) else dict(scope)
    material = json.dumps(
        {"q": normalise_query(query), "as_of": as_of,
         "scope": sorted(facets.items()), "config": config_hash},
        separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CacheEntry:
    """A stored answer, with the evidence it stands on and the date it stops standing."""

    key: str
    payload: Any
    cited: list[str]
    created_at: str
    expires_at: str | None = None
    query: str = ""
    as_of: str = ""
    config_hash: str = ""


@dataclass(frozen=True)
class CacheStats:
    """Counts since the file was created. Four failures, kept apart deliberately."""

    entries: int = 0
    hits: int = 0
    misses: int = 0
    expired: int = 0
    invalidated: int = 0
    evicted: int = 0
    writes: int = 0
    refused: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses + self.expired + self.invalidated

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"entries": self.entries, "hits": self.hits, "misses": self.misses,
                "expired": self.expired, "invalidated": self.invalidated,
                "evicted": self.evicted, "writes": self.writes, "refused": self.refused,
                "lookups": self.lookups, "hit_rate": round(self.hit_rate, 4)}


class AnswerCache:
    """Answers keyed on what produced them, expiring when the corpus says they are wrong.

    Holds a reference to the corpus `warrant.index.store.Store` because the entire premise
    requires it: the expiry is read off ``valid_to`` at write time and belief is re-checked
    against ``system_to`` on every read. A cache that could not see the store would be back to
    guessing a TTL.

    Connections are thread-local for the reason `warrant.index.store.Store` gives: a sqlite3
    connection may only be used from the thread that created it, and FastAPI runs synchronous
    endpoints in a threadpool. An in-memory cache is the exception -- each ``:memory:``
    connection is its own empty database -- so that one is shared.
    """

    def __init__(self, path: str | Path, store: Store, *,
                 max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_ttl_seconds: float | None = DEFAULT_MAX_TTL_SECONDS):
        self.path = Path(path)
        self.store = store
        self.max_entries = max_entries
        self.max_ttl_seconds = max_ttl_seconds
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
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry'"
        ).fetchone() is not None
        if written and existing != CACHE_SCHEMA_VERSION:
            # Discarded rather than refused. Every statement in SCHEMA is IF NOT EXISTS, so an
            # old file would otherwise open cleanly and fail inside a query much later; and
            # unlike a corpus or a trace file there is nothing here to migrate or mourn.
            conn.executescript(
                "DROP TABLE IF EXISTS entry_citation;"
                "DROP TABLE IF EXISTS entry;"
                "DROP TABLE IF EXISTS counter;")
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
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

    def __enter__(self) -> AnswerCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the store's opinion of the evidence ---------------------------------------

    def _believed(self, version_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        """Currently-believed rows for these versions: ``system_to IS NULL``, by version id.

        A version absent from the result is one the system does not stand behind right now --
        retracted, or gone with a rebuilt corpus. Both mean the same thing to a cached answer
        that cites it.
        """
        if not version_ids:
            return {}
        marks = ",".join("?" * len(version_ids))
        rows = self.store.db.execute(
            f"SELECT version_id, doc_id, content_hash, valid_to FROM chunk "
            f"WHERE version_id IN ({marks}) AND system_to IS NULL", list(version_ids))
        return {r["version_id"]: r for r in rows}

    def _expiry(self, rows: Iterable[sqlite3.Row], *, now: str) -> str | None:
        """The date the answer becomes wrong: earliest cited ``valid_to``, capped by the TTL.

        NULL ``valid_to`` is treated as infinity rather than as a sentinel date, so a set of
        still-in-force citations contributes no bound at all and the answer is held only by
        the TTL. A bound already in the past is dropped: see the module docstring -- valid
        time does not move, so a superseded interval describes a historical answer that cannot
        become wrong, and expiring it would make exactly the most cacheable queries in the
        corpus uncacheable.
        """
        closes = [_instant(r["valid_to"]) for r in rows if r["valid_to"] is not None]
        computed = min(closes) if closes else None
        if computed is not None and computed <= now:
            computed = None
        if self.max_ttl_seconds is None:
            return computed
        ttl_bound = _after(now, self.max_ttl_seconds)
        return ttl_bound if computed is None else min(computed, ttl_bound)

    # -- counters -------------------------------------------------------------------

    def _bump(self, db: sqlite3.Connection, name: str, n: int = 1) -> None:
        if n:
            db.execute("INSERT INTO counter (name, n) VALUES (?, ?) "
                       "ON CONFLICT(name) DO UPDATE SET n = n + excluded.n", (name, n))

    def stats(self) -> CacheStats:
        counts = {r["name"]: r["n"] for r in self.db.execute("SELECT name, n FROM counter")}
        return CacheStats(entries=self.count(),
                          **{c: counts.get(c, 0) for c in COUNTERS})

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM entry").fetchone()[0]

    # -- reading ---------------------------------------------------------------------

    def get(self, key: str, *, now: str | None = None) -> CacheEntry | None:
        """The stored answer for ``key``, or None -- recording *why* it was not served.

        Three checks in cost order: present, in date, still believed. The belief check is last
        because it is the only one that touches the corpus, and it is done on every read
        because it is one indexed query over at most a handful of version ids. Skipping it and
        trusting ``expires_at`` is the difference between a cache and a liability: a corrected
        parse lands with no date attached, and every entry assembled from the bad text would
        keep serving until its unrelated expiry.

        A failed entry is deleted on the way out. Leaving it costs the same check on every
        subsequent lookup and holds a slot against the cap for an answer that can never be
        served again.
        """
        instant = _instant(now) if now else utc_now()
        with self.db as db:
            row = db.execute("SELECT * FROM entry WHERE key = ?", (key,)).fetchone()
            if row is None:
                self._bump(db, "misses")
                return None
            if row["expires_at"] is not None and instant >= row["expires_at"]:
                self._drop(db, [key])
                self._bump(db, "expired")
                return None

            cites = db.execute(
                "SELECT version_id, content_hash FROM entry_citation WHERE key = ?",
                (key,)).fetchall()
            believed = self._believed([c["version_id"] for c in cites])
            if any(c["version_id"] not in believed
                   or believed[c["version_id"]]["content_hash"] != c["content_hash"]
                   for c in cites):
                self._drop(db, [key])
                self._bump(db, "invalidated")
                return None

            db.execute("UPDATE entry SET used_seq = "
                       "(SELECT COALESCE(MAX(used_seq), 0) + 1 FROM entry) WHERE key = ?",
                       (key,))
            self._bump(db, "hits")
            return _entry(row, [c["version_id"] for c in cites])

    # -- writing ---------------------------------------------------------------------

    def put(self, key: str, payload: Any, *, cited: Sequence[str],
            query: str = "", as_of: str = "", config_hash: str = "",
            now: str | None = None) -> CacheEntry | None:
        """Store one answer under ``key``, computing its expiry from the evidence it cites.

        ``payload`` is anything ``json.dumps`` accepts -- a pydantic ``model_dump()`` from the
        API, a plain dict from a test. Nothing here imports the response models or the
        generator, for the reason `warrant.observe.trace_store` gives: a retrieval-only
        install has no torch, and a cache that could not be opened without it would be useless
        in exactly the deployment that most needs one.

        ``cited`` is every version id the payload *depends on*, which is usually wider than
        the claims' evidence: an ``/api/ask`` response also displays the retrieved evidence
        rows, and a response showing superseded text is stale whether or not a claim cited it.

        Returns None without storing anything when a cited version is not currently believed.
        That is not a cache miss to paper over -- it means the answer was assembled from text
        the store has already disowned, and writing it would burn a slot on an entry the very
        next `get` is obliged to throw away.
        """
        instant = _instant(now) if now else utc_now()
        keys = list(dict.fromkeys(cited))
        believed = self._believed(keys)
        with self.db as db:
            if any(k not in believed for k in keys):
                self._bump(db, "refused")
                return None

            expires_at = self._expiry([believed[k] for k in keys], now=instant)
            self._drop(db, [key])
            db.execute(
                "INSERT INTO entry (key, created_at, used_seq, expires_at, query, as_of, "
                "config_hash, payload) VALUES (?, ?, "
                "(SELECT COALESCE(MAX(used_seq), 0) + 1 FROM entry), ?, ?, ?, ?, ?)",
                (key, instant, expires_at, query, as_of, config_hash, json.dumps(payload)))
            db.executemany(
                "INSERT INTO entry_citation (key, version_id, content_hash, doc_id) "
                "VALUES (?, ?, ?, ?)",
                [(key, k, believed[k]["content_hash"], believed[k]["doc_id"]) for k in keys])
            self._bump(db, "writes")
            self._bump(db, "evicted", self._evict(db))
        return CacheEntry(key=key, payload=payload, cited=keys, created_at=instant,
                          expires_at=expires_at, query=query, as_of=as_of,
                          config_hash=config_hash)

    def _evict(self, db: sqlite3.Connection) -> int:
        """Drop least-recently-used entries down to the cap. Returns how many went."""
        over = db.execute("SELECT COUNT(*) FROM entry").fetchone()[0] - self.max_entries
        if over <= 0:
            return 0
        victims = [r["key"] for r in db.execute(
            "SELECT key FROM entry ORDER BY used_seq LIMIT ?", (over,))]
        return self._drop(db, victims)

    def _drop(self, db: sqlite3.Connection, keys: Sequence[str]) -> int:
        """Delete entries and their citations. Explicit rather than by cascade: ``PRAGMA
        foreign_keys`` is per-connection, and an orphaned citation row would keep matching
        `invalidate_document` forever."""
        if not keys:
            return 0
        marks = ",".join("?" * len(keys))
        db.execute(f"DELETE FROM entry_citation WHERE key IN ({marks})", list(keys))
        return db.execute(f"DELETE FROM entry WHERE key IN ({marks})", list(keys)).rowcount

    # -- invalidation ------------------------------------------------------------------

    def invalidate_document(self, doc_id: str) -> int:
        """Forget every answer citing any version of one document. Returns entries removed.

        The hook for a re-ingest: ingestion knows which document it just rewrote long before
        any reader would notice, and this turns that into an immediate flush of the affected
        answers instead of a wait for their expiry. Keyed on ``doc_id`` rather than
        ``version_id`` because a re-ingest closes and reopens several versions at once, and
        the caller should not have to enumerate them to be correct.
        """
        with self.db as db:
            victims = [r["key"] for r in db.execute(
                "SELECT DISTINCT key FROM entry_citation WHERE doc_id = ?", (doc_id,))]
            return self._drop(db, victims)

    def clear(self) -> int:
        """Empty the cache, keeping the counters. Returns entries removed."""
        with self.db as db:
            n = db.execute("SELECT COUNT(*) FROM entry").fetchone()[0]
            db.execute("DELETE FROM entry_citation")
            db.execute("DELETE FROM entry")
        return n

    def sweep(self, now: str | None = None) -> int:
        """Drop everything already dead, without waiting for a lookup to notice.

        `get` removes a dead entry when it is asked for, which is enough for correctness and
        not enough for the cap: an entry nobody asks for again holds a slot until LRU reaches
        it. Both failures are counted the same way they would have been on read, so a sweep
        does not launder an expiry into an eviction.
        """
        instant = _instant(now) if now else utc_now()
        with self.db as db:
            stale = [r["key"] for r in db.execute(
                "SELECT key FROM entry WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (instant,))]
            self._bump(db, "expired", self._drop(db, stale))

            cites = db.execute(
                "SELECT c.key, c.version_id, c.content_hash FROM entry_citation c").fetchall()
            believed = self._believed(sorted({c["version_id"] for c in cites}))
            doubted = sorted({
                c["key"] for c in cites
                if c["version_id"] not in believed
                or believed[c["version_id"]]["content_hash"] != c["content_hash"]})
            self._bump(db, "invalidated", self._drop(db, doubted))
        return len(stale) + len(doubted)


def _entry(row: sqlite3.Row, cited: list[str]) -> CacheEntry:
    return CacheEntry(
        key=row["key"],
        payload=json.loads(row["payload"]),
        cited=cited,
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        query=row["query"],
        as_of=row["as_of"],
        config_hash=row["config_hash"],
    )
