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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: Open-ended intervals use NULL rather than a sentinel date. A sentinel invites the bug
#: where "9999-12-31" sorts correctly but compares wrong against a NULL from another path.
OPEN = None

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS chunk (
    id              INTEGER PRIMARY KEY,
    chunk_id        TEXT    NOT NULL,   -- 630.1203#a
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
CREATE INDEX IF NOT EXISTS chunk_part   ON chunk (part, subpart);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text, heading, chunk_id UNINDEXED,
    content='chunk', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts(rowid, text, heading, chunk_id)
    VALUES (new.id, new.text, new.heading, new.chunk_id);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text, heading, chunk_id)
    VALUES ('delete', old.id, old.text, old.heading, old.chunk_id);
END;
"""


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
    valid_from: str = "1970-01-01"
    valid_to: str | None = OPEN
    source_snapshot: str = ""
    config_hash: str = ""


class Store:
    """Append-only bitemporal store over SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.db:
            yield self.db

    # -- writing -----------------------------------------------------------------

    def add(self, chunks: Iterable[Chunk], *, system_from: str | None = None) -> int:
        """Insert new chunk versions. Never updates the text of an existing row."""
        ts = system_from or now()
        rows = [
            (c.chunk_id, c.section_id, c.title, c.part, c.subpart, c.anchor, c.heading,
             c.text, content_hash(c.text), c.valid_from, c.valid_to, ts, None,
             c.source_snapshot, c.config_hash)
            for c in chunks
        ]
        with self.tx() as db:
            db.executemany(
                "INSERT INTO chunk (chunk_id, section_id, title, part, subpart, anchor, "
                "heading, text, content_hash, valid_from, valid_to, system_from, system_to, "
                "source_snapshot, config_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def close_valid(self, section_id: str, valid_to: str) -> int:
        """Mark the currently-in-force version of a section as superseded on ``valid_to``."""
        with self.tx() as db:
            cur = db.execute(
                "UPDATE chunk SET valid_to = ? "
                "WHERE section_id = ? AND valid_to IS NULL AND system_to IS NULL",
                (valid_to, section_id),
            )
        return cur.rowcount

    def retract(self, chunk_id: str, *, system_to: str | None = None) -> int:
        """Stop believing a chunk version, without deleting it.

        This is how a corrected parse is recorded: close system time on the old row and
        insert the replacement. The old row stays readable at its own system time, which is
        the entire point of the second axis.
        """
        ts = system_to or now()
        with self.tx() as db:
            cur = db.execute(
                "UPDATE chunk SET system_to = ? WHERE chunk_id = ? AND system_to IS NULL",
                (ts, chunk_id),
            )
        return cur.rowcount

    # -- reading -----------------------------------------------------------------

    def as_of(self, valid_date: str, *, system_time: str | None = None,
              part: str | None = None) -> list[sqlite3.Row]:
        """Every chunk in force on ``valid_date``, as the system believed at ``system_time``.

        Omitting ``system_time`` means "as the system believes now", which is the ordinary
        query path. Supplying it is what makes audit replay possible.
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
        return self.db.execute(" ".join(sql), params).fetchall()

    def versions_of(self, section_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM chunk WHERE section_id = ? "
            "ORDER BY valid_from, system_from", (section_id,)
        ).fetchall()

    def search(self, query: str, *, valid_date: str, system_time: str | None = None,
               limit: int = 100) -> list[sqlite3.Row]:
        """Lexical search with the as-of predicate pushed into the query.

        The predicate lives inside the SQL, not applied to results afterwards. Filtering
        after the fact would let superseded text consume candidate slots and rerank budget
        before being discarded, and would break the invariant in ARCHITECTURE.md section 9
        that a dated query sees at most one version of any section.
        """
        sys_t = system_time or now()
        return self.db.execute(
            "SELECT c.*, bm25(chunk_fts) AS score "
            "FROM chunk_fts JOIN chunk c ON c.id = chunk_fts.rowid "
            "WHERE chunk_fts MATCH :q "
            "AND c.valid_from <= :v AND (c.valid_to IS NULL OR c.valid_to > :v) "
            "AND c.system_from <= :s AND (c.system_to IS NULL OR c.system_to > :s) "
            "ORDER BY score LIMIT :k",
            {"q": query, "v": valid_date, "s": sys_t, "k": limit},
        ).fetchall()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
