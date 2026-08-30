"""The bitemporal chunk store.

Two independent time axes, because there are two independent questions (ARCHITECTURE.md
section 4):

    valid_from  / valid_to     when this text was the law in the real world
    system_from / system_to    when Warrant believed this text was the law

The store is **append-only**. Correcting a bad parse does not update a row; it closes the
old row system_to and inserts a new one. A single ``ingested_at`` column cannot support
this: it records when a row arrived but not when it stopped being believed, so a re-ingest
silently destroys the state needed to answer

    "reproduce the answer the system would have given on 14 March,
     using only what it believed on 14 March."

Storage is SQLite with FTS5. No server and no Docker: ``git clone && make`` has to work on a
reviewer laptop, and a corpus of this size does not need anything larger.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: Open-ended intervals use NULL rather than a sentinel date. A sentinel invites the bug
#: where "9999-12-31" sorts correctly but compares wrong against a NULL from another path.
OPEN = None


class SchemaMismatch(RuntimeError):
    """The store on disk was written by a different schema version than this build."""

#: Bump whenever the schema changes in a way an existing store cannot satisfy. Every DDL
#: statement below is IF NOT EXISTS, so an old store connects without error and then fails
#: much later inside a query with "no such column", naming neither the cause nor the cure.
#: Worse are the silent drifts -- an edited FTS tokenizer or trigger body is simply never
#: applied to an existing store, and the only symptom is a few points of retrieval quality.
SCHEMA_VERSION = 2

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS chunk (
    id              INTEGER PRIMARY KEY,
    -- The addressable unit of evidence: 630.1203#a@2020-08-10. Version-qualified, because
    -- chunk_id alone repeats across versions, and in a system whose entire subject is which
    -- version was in force, a citation that cannot name the version is not a citation.
    -- Not UNIQUE: one valid-time version can be believed more than once over system time,
    -- which is exactly what a corrected parse produces.
    version_id      TEXT    NOT NULL,
    chunk_id        TEXT    NOT NULL,   -- 630.1203#a, stable across versions

    -- Multi-source provenance. Federal HR law is a hierarchy of documents, not one
    -- document: a statute, the regulation implementing it, the notice explaining the
    -- amendment, and the guidance interpreting it. Retrieval that mixes them without
    -- recording which is which will cite a fact sheet over the law it summarises, and
    -- nothing downstream can detect that. authority is an int because the ordering is the
    -- semantics: 1 statute, 2 regulation, 3 notice, 4 guidance, 5 archival.
    source          TEXT    NOT NULL DEFAULT 'ecfr',
    doc_id          TEXT    NOT NULL DEFAULT '',
    authority       INTEGER NOT NULL DEFAULT 2,
    -- How the text was recovered: prose, table, heading, ocr, caption. A citation to OCR of
    -- a scanned page is weaker evidence than one to parsed XML, and a verifier can only
    -- weigh that if ingestion wrote it down.
    kind            TEXT    NOT NULL DEFAULT 'prose',
    locator         TEXT    NOT NULL DEFAULT '',

    -- CFR-shaped fields. Non-CFR sources set section_id = doc_id so that every grouping,
    -- clustering and invariant keyed on section_id keeps working unchanged across sources.
    section_id      TEXT    NOT NULL,   -- 630.1203
    title           INTEGER NOT NULL,
    part            TEXT    NOT NULL,
    subpart         TEXT,
    anchor          TEXT,
    heading         TEXT,
    text            TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,

    valid_from      TEXT    NOT NULL,   -- ISO date; in force from
    valid_to        TEXT,               -- ISO date, exclusive; NULL = still in force
    system_from     TEXT    NOT NULL,   -- ISO timestamp; believed from
    system_to       TEXT,               -- ISO timestamp, exclusive; NULL = still believed

    source_snapshot TEXT    NOT NULL,   -- the /full/ date this text was read from
    config_hash     TEXT    NOT NULL    -- ingestion settings that produced it
);

CREATE INDEX IF NOT EXISTS chunk_asof
    ON chunk (section_id, valid_from, valid_to, system_from, system_to);
CREATE INDEX IF NOT EXISTS chunk_lookup ON chunk (chunk_id);
CREATE INDEX IF NOT EXISTS chunk_version ON chunk (version_id);
CREATE INDEX IF NOT EXISTS chunk_part   ON chunk (part, subpart);
CREATE INDEX IF NOT EXISTS chunk_source ON chunk (source, authority);
CREATE INDEX IF NOT EXISTS chunk_doc    ON chunk (doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text, heading, version_id UNINDEXED,
    content='chunk', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts(rowid, text, heading, version_id)
    VALUES (new.id, new.text, new.heading, new.version_id);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text, heading, version_id)
    VALUES ('delete', old.id, old.text, old.heading, old.version_id);
END;
"""


def _exclusion_clause(parts: Sequence[str], params: dict[str, object],
                      alias: str = "") -> str:
    """Bind an applicability exclusion into a query, or return nothing if there is none."""
    if not parts:
        return ""
    names = []
    for i, p in enumerate(parts):
        key = f"xp{i}"
        params[key] = p
        names.append(f":{key}")
    return f"AND {alias}part NOT IN ({', '.join(names)})"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str
    title: int
    part: str
    text: str
    anchor: str | None = None
    subpart: str | None = None
    heading: str | None = None
    source: str = "ecfr"
    doc_id: str = ""
    authority: int = 2
    kind: str = "prose"
    locator: str = ""
    valid_from: str = "1970-01-01"
    valid_to: str | None = OPEN
    source_snapshot: str = ""
    config_hash: str = ""

    @property
    def version_id(self) -> str:
        """Version-qualified address: ``630.1203#a@2020-08-10``."""
        return f"{self.chunk_id}@{self.valid_from}"


class Store:
    """Append-only bitemporal store over SQLite.

    Connections are **thread-local**. A sqlite3 connection may only be used from the thread
    that created it, and FastAPI runs synchronous endpoints in a threadpool, so a single
    shared connection raises ``ProgrammingError`` on every request that does not happen to
    land on the creating thread -- which, measured, was 8 of 8. Passing
    ``check_same_thread=False`` would silence the exception while leaving one connection and
    its GIL-bound cursor contended by every worker; a connection per thread is cheap against
    a local file and is correct for writes as well as reads.

    An in-memory store is the exception: each connection to ``:memory:`` is its own empty
    database, so a thread-local one would silently see nothing. Tests are single-threaded, so
    that connection is shared deliberately.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._memory = str(path) == ":memory:"
        if not self._memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None
        if self._memory:
            self._shared = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path) if not self._memory else ":memory:")
        conn.row_factory = sqlite3.Row
        existing = conn.execute("PRAGMA user_version").fetchone()[0]
        has_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk'"
        ).fetchone() is not None
        if has_rows and existing != SCHEMA_VERSION:
            conn.close()
            raise SchemaMismatch(
                f"{self.path} was written by schema v{existing}, this build expects "
                f"v{SCHEMA_VERSION}. Delete it and re-run `make build` "
                f"(about 6 seconds from the cached snapshots).")
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return conn

    @property
    def db(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def read_only(self) -> None:
        """Refuse writes on this connection. Serving paths should call it."""
        self.db.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        conn = self._shared or getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._shared = None
        self._local = threading.local()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """A transaction that nests.

        sqlite3 connection context managers do not nest: an inner ``with connection`` block
        commits the *outer* transaction when it exits. So a caller wrapping several writes to
        make them atomic got no atomicity at all, because ``add`` and ``close_valid`` each
        open their own. Ingest hit exactly that -- a crash between closing a section and
        inserting its replacement left the store believing the law had simply been repealed,
        with the one-version-in-force invariant still passing, because zero is not two.

        Depth is tracked per thread, alongside the connection, so a nested call joins the
        outer transaction instead of ending it.
        """
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield self.db
            finally:
                self._local.depth -= 1
            return
        self._local.depth = 1
        try:
            with self.db as conn:
                yield conn
        finally:
            self._local.depth = 0

    # -- writing -----------------------------------------------------------------

    def add(self, chunks: Iterable[Chunk], *, system_from: str | None = None) -> int:
        """Insert new chunk versions. Never updates the text of an existing row."""
        ts = system_from or now()
        rows = [
            (c.version_id, c.chunk_id, c.source, c.doc_id or c.section_id, c.authority,
             c.kind, c.locator, c.section_id, c.title, c.part, c.subpart, c.anchor,
             c.heading, c.text, content_hash(c.text), c.valid_from, c.valid_to, ts, None,
             c.source_snapshot, c.config_hash)
            for c in chunks
        ]
        with self.tx() as db:
            db.executemany(
                "INSERT INTO chunk (version_id, chunk_id, source, doc_id, authority, kind, "
                "locator, section_id, title, part, subpart, anchor, heading, text, "
                "content_hash, valid_from, valid_to, system_from, system_to, "
                "source_snapshot, config_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def close_valid(self, section_id: str, valid_to: str) -> int:
        """Mark the currently-in-force version of a section as superseded on ``valid_to``.

        ``valid_from < :valid_to`` is load-bearing. Without it, closing a section at a date it
        was also opened on produces a **zero-width interval**, and since ``as_of`` asks for
        ``valid_from <= d AND valid_to > d``, no date can ever satisfy it: that version of the
        law becomes permanently unretrievable. Re-running an ingest into a non-empty store did
        exactly that -- it deleted in-force law from the answerable range while duplicating
        the superseded law beside it, with no error.
        """
        with self.tx() as db:
            cur = db.execute(
                "UPDATE chunk SET valid_to = ? "
                "WHERE section_id = ? AND valid_to IS NULL AND system_to IS NULL "
                "AND valid_from < ?",
                (valid_to, section_id, valid_to),
            )
        return cur.rowcount

    def is_empty(self) -> bool:
        return self.count() == 0

    def retract(self, version_id: str, *, system_to: str | None = None) -> int:
        """Stop believing one valid-time version, without deleting it.

        This is how a corrected parse is recorded: close system time on the old row and
        insert the replacement. The old row stays readable at its own system time, which is
        the entire point of the second axis.

        Keyed on ``version_id``, not ``chunk_id``: retracting by chunk_id would retract every
        historical version of that paragraph at once, so fixing a parse error in the 2020
        text would also stop the system believing the 2018 text.
        """
        ts = system_to or now()
        with self.tx() as db:
            cur = db.execute(
                "UPDATE chunk SET system_to = ? WHERE version_id = ? AND system_to IS NULL",
                (ts, version_id),
            )
        return cur.rowcount

    # -- reading -----------------------------------------------------------------

    def as_of(self, valid_date: str, *, system_time: str | None = None,
              part: str | None = None,
              exclude_parts: Sequence[str] = ()) -> list[sqlite3.Row]:
        """Every chunk in force on ``valid_date``, as the system believed at ``system_time``.

        Omitting ``system_time`` means "as the system believes now", which is the ordinary
        query path. Supplying it is what makes audit replay possible.

        ``exclude_parts`` carries the applicability predicate, computed by
        ``warrant.retrieve.scope``. It is a correctness filter, not an access control: see
        that module and ARCHITECTURE.md section 3.
        """
        sys_t = system_time or now()
        sql = [
            "SELECT * FROM chunk WHERE valid_from <= :v "
            "AND (valid_to IS NULL OR valid_to > :v) "
            "AND system_from <= :s AND (system_to IS NULL OR system_to > :s)"
        ]
        params: dict[str, object] = {"v": valid_date, "s": sys_t}
        if part is not None:
            sql.append("AND part = :p")
            params["p"] = part
        sql.append(_exclusion_clause(exclude_parts, params))
        return self.db.execute(" ".join(sql), params).fetchall()

    def versions_of(self, section_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM chunk WHERE section_id = ? "
            "ORDER BY valid_from, system_from", (section_id,)
        ).fetchall()

    def search(self, query: str, *, valid_date: str, system_time: str | None = None,
               limit: int = 100, temporal: bool = True,
               exclude_parts: Sequence[str] = ()) -> list[sqlite3.Row]:
        """Lexical search with the as-of predicate pushed into the query.

        The predicate lives inside the SQL, not applied to results afterwards. Filtering
        after the fact would let superseded text consume candidate slots and rerank budget
        before being discarded, and would break the invariant in ARCHITECTURE.md section 9
        that a dated query sees at most one version of any section.

        ``temporal=False`` drops the valid-time predicate. It exists solely as an ablation:
        it is how the temporal bucket demonstrates that the filter is doing the work, rather
        than being asserted to. It is never a serving mode -- without it the store will
        happily return four versions of one section and let the model pick.
        """
        sys_t = system_time or now()
        valid_clause = ("AND c.valid_from <= :v AND (c.valid_to IS NULL OR c.valid_to > :v) "
                        if temporal else "")
        params: dict[str, object] = {"q": query, "v": valid_date, "s": sys_t, "k": limit}
        excl = _exclusion_clause(exclude_parts, params, alias="c.")
        return self.db.execute(
            "SELECT c.*, bm25(chunk_fts) AS score "
            "FROM chunk_fts JOIN chunk c ON c.id = chunk_fts.rowid "
            "WHERE chunk_fts MATCH :q "
            f"{valid_clause}"
            "AND c.system_from <= :s AND (c.system_to IS NULL OR c.system_to > :s) "
            f"{excl} "
            "ORDER BY score LIMIT :k",
            params,
        ).fetchall()

    def candidate_ids(self, *, valid_date: str, system_time: str | None = None,
                      temporal: bool = True,
                      exclude_parts: Sequence[str] = ()) -> set[int]:
        """Row ids admitted by the predicates, for restricting a dense search.

        Dense retrieval cannot express the predicate in SQL, so it restricts the search space
        to these ids *before* scoring. That is still pushing the predicate into the query --
        the candidate set is narrowed first -- rather than filtering a ranked list afterwards.
        """
        sys_t = system_time or now()
        valid_clause = ("AND valid_from <= :v AND (valid_to IS NULL OR valid_to > :v) "
                        if temporal else "")
        params: dict[str, object] = {"v": valid_date, "s": sys_t}
        excl = _exclusion_clause(exclude_parts, params)
        rows = self.db.execute(
            "SELECT id FROM chunk WHERE system_from <= :s "
            "AND (system_to IS NULL OR system_to > :s) "
            f"{valid_clause}{excl}",
            params,
        ).fetchall()
        return {r["id"] for r in rows}

    def rows_by_id(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        return {r["id"]: r for r in
                self.db.execute(f"SELECT * FROM chunk WHERE id IN ({marks})", list(ids))}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
