"""Writing any source into the bitemporal store.

One ingester for every format. The eCFR path stays separate because it is genuinely
different -- it walks point-in-time snapshots and derives validity intervals by diffing
consecutive ones, which only works for a source that publishes the same document repeatedly.
Every other source hands over documents that already know their own dates, and they all land
here.

The reason this is one function rather than four is the failure budget. Its ``ingestion`` row
has to mean the same thing whether the text arrived as USLM XML, a Federal Register JSON
body, an OPM page, or OCR from a scanned 1994 notice. If each source wrote its own rows its
own way, "ingestion lost the evidence" would be four different claims wearing one label, and
the one number a reader looks at first would be the least trustworthy on the page.

Change detection is by content hash, not by re-reading. A guidance page re-fetched daily is
usually byte-identical; inserting a new version each time would inflate the store, break the
one-version-in-force invariant, and make every citation ambiguous. So a document whose units
hash the same as the believed ones is a no-op, and one that differs closes the old interval
and opens a new one -- the same discipline the CFR path uses, applied to sources that do not
hand us a version history.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..index.store import Chunk, Store, content_hash
from ..sources.base import SourceDoc

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    source: str
    documents: int = 0
    documents_unchanged: int = 0
    units_inserted: int = 0
    versions_closed: int = 0
    documents_failed: int = 0
    #: Documents that yielded no units at all. Counted separately from failures because a
    #: page that parses cleanly to nothing is a different bug from one that raised, and the
    #: silent version is the one that reaches production.
    documents_empty: int = 0
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.source}: {self.documents} documents, "
                f"{self.units_inserted} units, {self.documents_unchanged} unchanged, "
                f"{self.documents_empty} empty, {self.documents_failed} failed")


def doc_chunks(doc: SourceDoc, *, config_hash: str) -> list[Chunk]:
    """One chunk per unit, carrying the document's provenance onto every row.

    ``section_id`` is set to the document id for non-CFR sources. Every grouping, clustering
    and invariant in this codebase keys on ``section_id`` -- the section-clustered bootstrap,
    the one-version-in-force gate, the dev/test split -- and giving a Federal Register notice
    a ``section_id`` means all of that keeps working across sources without a second code
    path. A notice is the unit of authorship the way a CFR section is.
    """
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{unit.anchor}",
            section_id=doc.doc_id,
            doc_id=doc.doc_id,
            source=doc.source,
            authority=doc.authority,
            kind=unit.kind,
            locator=unit.locator,
            title=0,
            part=doc.meta.get("part", ""),
            subpart=doc.meta.get("subpart") or None,
            anchor=unit.anchor,
            heading=unit.heading or doc.title,
            text=unit.text,
            valid_from=doc.valid_from,
            valid_to=doc.valid_to,
            source_snapshot=doc.meta.get("snapshot", doc.valid_from),
            config_hash=config_hash,
        )
        for unit in doc.units
    ]


def _believed_hashes(store: Store, doc_id: str) -> dict[str, str]:
    rows = store.db.execute(
        "SELECT chunk_id, content_hash FROM chunk "
        "WHERE doc_id = ? AND system_to IS NULL AND valid_to IS NULL", (doc_id,)).fetchall()
    return {r["chunk_id"]: r["content_hash"] for r in rows}


def ingest(store: Store, documents: Iterable[SourceDoc], *, source: str,
           config_hash: str, system_from: str | None = None) -> IngestStats:
    """Write documents into the store, one transaction per document.

    Per document, not per batch: a source that fails on its 400th page must leave the first
    399 readable rather than rolling back an afternoon of fetching. The CFR path makes the
    same choice at snapshot granularity, for the same reason -- a half-applied document is
    indistinguishable from a document that said less than it does.
    """
    stats = IngestStats(source=source)

    for doc in documents:
        try:
            chunks = doc_chunks(doc, config_hash=config_hash)
            if not chunks:
                stats.documents_empty += 1
                log.warning("%s: %s parsed to zero units", source, doc.doc_id)
                continue

            with store.tx():
                believed = _believed_hashes(store, doc.doc_id)
                incoming = {c.chunk_id: content_hash(c.text) for c in chunks}
                if believed and believed == incoming:
                    stats.documents_unchanged += 1
                    stats.documents += 1
                    continue
                if believed:
                    # The document changed. Close what was in force rather than editing it:
                    # the store is append-only so that a past answer stays reproducible, and
                    # an in-place update would silently rewrite history a citation points at.
                    stats.versions_closed += store.close_valid(doc.doc_id, doc.valid_from)
                stats.units_inserted += store.add(chunks, system_from=system_from)
            stats.documents += 1
            log.info("%s: ingested %s (%d units)", source, doc.doc_id, len(chunks))

        except Exception as exc:  # noqa: BLE001 - one bad document must not end the run
            stats.documents_failed += 1
            stats.failures.append(f"{doc.doc_id}: {exc}")
            log.exception("%s: failed on %s", source, doc.doc_id)

    return stats
