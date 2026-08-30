"""Turning point-in-time snapshots into bitemporal chunk versions.

Walk a part forward through its snapshots. A section whose text is unchanged keeps its open
validity interval; a section whose text changed has its interval closed on the new snapshot
date and a fresh version inserted. Nothing is ever updated in place, so the store can always
be read as of an earlier system time.

Two deliberate choices, both of which bound what the benchmark may claim:

**Versioning is at section level, not paragraph level.** Paragraph anchors move when a
paragraph is inserted: adding a new ``(b)`` renumbers everything below it, and paragraph-keyed
versioning would report a dozen amendments where the law changed once. Sections are the unit
the regulation itself is amended in. Paragraphs within a section version share that version
interval and remain the citation unit.

**Temporal resolution is snapshot granularity.** The eCFR versions endpoint carries a
per-section ``amendment_date`` that can precede the snapshot in which the text first appears.
Warrant dates a change to the snapshot where the text demonstrably differs, because that is
what it can prove from the primary source. Questions in the temporal benchmark are therefore
posed away from snapshot boundaries.

Below its history floor the store holds nothing, and Warrant makes no claim about dates
before it -- see ARCHITECTURE.md section 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..index.store import Chunk, Store
from .chunking import DEFAULT_POLICY, ChunkPolicy, units
from .ecfr import ECFRClient
from .parse import Section, parse_sections

log = logging.getLogger(__name__)


@dataclass
class BuildStats:
    part: str
    snapshots: int = 0
    sections_seen: int = 0
    versions_inserted: int = 0
    chunks_inserted: int = 0
    sections_closed: int = 0
    unchanged: int = 0
    dates: list[str] = field(default_factory=list)
    #: Advertised dates whose text the API would not serve. Carried on the stats because a
    #: part that ingested three of its eight snapshots is not a part that ingested cleanly,
    #: and the caller has no other way to find out.
    snapshots_skipped: int = 0
    skipped_dates: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        skipped = f", {self.snapshots_skipped} dates unavailable" if self.snapshots_skipped else ""
        return (f"part {self.part}: {self.snapshots} snapshots, "
                f"{self.versions_inserted} section versions, "
                f"{self.chunks_inserted} chunks, {self.sections_closed} closures{skipped}")


def section_chunks(section: Section, *, title: int, part: str, valid_from: str,
                   snapshot: str, config_hash: str,
                   policy: ChunkPolicy = DEFAULT_POLICY) -> list[Chunk]:
    """Chunks for one section version, under a chunking policy.

    Ingestion decides *when* a version exists; ``warrant.corpus.chunking`` decides what a
    retrievable unit is. This function is the seam, and it is the only place the two meet:
    pass ``ChunkPolicy.legacy()`` and the store is byte-identical to the one built before
    contextual augmentation existed, which is what makes the two measurable against each
    other rather than merely describable.

    A section with no parsed paragraphs still yields one chunk carrying the whole section
    text; dropping it would make the section unretrievable, which is a silent corpus hole
    of exactly the kind the ingestion row of the failure budget is meant to expose.
    """
    return [
        Chunk(chunk_id=f"{section.identifier}#{u.anchor or 'full'}",
              section_id=section.identifier, title=title, part=part,
              subpart=section.subpart, anchor=u.anchor, heading=section.heading,
              text=u.text, context=u.context, parent_id=u.parent_id, kind=u.kind,
              valid_from=valid_from, source_snapshot=snapshot, config_hash=config_hash)
        for u in units(section, policy)
    ]


def build_part(store: Store, client: ECFRClient, *, title: int, part: str,
               floor: str, config_hash: str, system_from: str | None = None,
               policy: ChunkPolicy = DEFAULT_POLICY) -> BuildStats:
    """Ingest every retrievable snapshot of one part into the store.

    Sections present in a part's *first* snapshot are dated from the history floor rather
    than from that snapshot. This is licensed by the API rather than assumed: eCFR records a
    version date whenever a part is amended, so the absence of any version date between the
    floor and the first snapshot is positive evidence that the text did not change in that
    window. Part 315 has no post-floor amendment until 2020-10-16, so its text at that
    snapshot was already in force in 2017, and dating it from 2020 would make three years of
    answerable questions silently unanswerable.

    Sections that first appear in a *later* snapshot are not backfilled -- they did not exist
    before, and claiming otherwise would invent law.

    ``policy`` is passed straight through to ``section_chunks``, so a whole store can be
    rebuilt under ``ChunkPolicy.legacy()`` and scored against one built under the default.
    That is the only honest way to report a chunking change: the two stores differ in the
    chunker and in nothing else, not in the corpus, the dates or the ingest code.

    Each snapshot is applied in one transaction, so a part always advances a whole snapshot
    at a time. A half-applied snapshot is worse than no snapshot: it is indistinguishable
    from the law having been repealed. ``Store.tx`` nests, so the per-write transactions
    inside ``add`` and ``close_valid`` join this one rather than committing it early.
    """
    stats = BuildStats(part=part)
    previous: dict[str, Section] = {}
    first_snapshot = True

    for date, xml in client.snapshots(title, part, floor=floor):
        current = {s.identifier: s for s in
                   parse_sections(xml, source=f"title {title} part {part} as of {date}")}
        valid_from = floor if first_snapshot else date

        with store.tx():
            new_chunks: list[Chunk] = []
            closed = 0
            for ident, section in current.items():
                prior = previous.get(ident)
                if prior is not None and prior.text == section.text:
                    stats.unchanged += 1
                    continue
                if prior is not None:
                    # The prior version stopped being the law on this date, exclusive.
                    closed += store.close_valid(ident, date)
                new_chunks.extend(section_chunks(section, title=title, part=part,
                                                 valid_from=valid_from, snapshot=date,
                                                 config_hash=config_hash, policy=policy))
                stats.versions_inserted += 1

            for ident in set(previous) - set(current):
                closed += store.close_valid(ident, date)

            if new_chunks:
                stats.chunks_inserted += store.add(new_chunks, system_from=system_from)

        stats.sections_closed += closed
        stats.snapshots += 1
        stats.dates.append(date)
        stats.sections_seen = max(stats.sections_seen, len(current))
        first_snapshot = False
        previous = current
        log.info("part %s %s: %d sections, %d new chunks, %d closures",
                 part, date, len(current), len(new_chunks), closed)

    # Duck-typed: build_part accepts anything with a ``snapshots`` method, and the fixtures
    # in the ingestion tests deliberately have only that.
    report = getattr(client, "skipped_dates", None)
    if report is not None:
        stats.skipped_dates = list(report(title, part))
        stats.snapshots_skipped = len(stats.skipped_dates)
    log.info("%s", stats)
    return stats
