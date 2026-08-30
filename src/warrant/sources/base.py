"""The contract every data source implements.

Federal HR law is not one document, it is a hierarchy of them, and a system that reads only
the regulation is reading the middle of an argument:

    statute      5 U.S.C. 6304(d)      what Congress required
    regulation   5 CFR 630.306         how OPM implemented it
    notice       85 FR 48089           why it changed, and on what reasoning
    guidance     OPM fact sheets       how OPM says it should be read
    archival     govinfo PDF           the printed record, sometimes only as scanned images

Answering "by when must restored leave be used, and why did that change in 2020" needs three
of those five, and they arrive as XML, JSON, HTML and PDF respectively. So the ingestion
contract has to be format-agnostic without becoming so abstract it stops asserting anything.

Two rules hold everything together:

**Authority is ordered and explicit.** When sources disagree, a statute outranks a regulation
outranks guidance. Retrieval that mixes them without recording which is which will
confidently cite an OPM fact sheet over the law it summarises, and there is no way to detect
that after the fact. ``authority`` is on every unit, and it is an int so it sorts.

**Every unit keeps its provenance.** A chunk knows its source, its document, and how it was
extracted -- prose, a table row, or OCR from a scan. A citation to OCR text carries different
weight than a citation to parsed XML, and a system that cannot tell them apart cannot be
honest about its own confidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Lower binds harder. Ints rather than an enum because retrieval sorts and filters on this,
#: and the ordering *is* the semantics -- an enum would hide the comparison behind a lookup.
AUTHORITY_STATUTE = 1
AUTHORITY_REGULATION = 2
AUTHORITY_NOTICE = 3
AUTHORITY_GUIDANCE = 4
AUTHORITY_ARCHIVAL = 5

AUTHORITY_NAMES = {
    AUTHORITY_STATUTE: "statute",
    AUTHORITY_REGULATION: "regulation",
    AUTHORITY_NOTICE: "notice",
    AUTHORITY_GUIDANCE: "guidance",
    AUTHORITY_ARCHIVAL: "archival",
}

#: How a unit's text was recovered. Recorded because it bounds what a citation to it can
#: claim: OCR of a 1994 scan is evidence, and it is weaker evidence than parsed XML, and the
#: only way a verifier can weigh that is if ingestion wrote it down.
KIND_PROSE = "prose"
KIND_TABLE = "table"
KIND_HEADING = "heading"
KIND_OCR = "ocr"
KIND_CAPTION = "caption"


@dataclass(frozen=True)
class Unit:
    """One retrievable, citable piece of a document.

    The unit, not the document, is what gets embedded and cited, because a reader acts on a
    paragraph and not on a 40-page notice. ``anchor`` must be unique within its document: a
    citation that matches two units is not a citation.
    """

    anchor: str
    text: str
    heading: str = ""
    kind: str = KIND_PROSE
    #: Where this text came from inside the document -- a page number, an XML path, a table
    #: cell range. Free-form because it means something different per format, and useful
    #: precisely when someone disputes a quote.
    locator: str = ""

    def __post_init__(self) -> None:
        if not self.anchor:
            raise ValueError("a unit without an anchor cannot be cited")


@dataclass(frozen=True)
class SourceDoc:
    """One document from one source, already split into citable units.

    Carries validity dates because the whole system is bitemporal: a Federal Register notice
    is valid from its publication date forever (it is a historical fact), while a regulation
    section is valid only until the next amendment. A source that cannot say when its content
    was true has to say so by leaving ``valid_to`` open rather than by guessing.
    """

    source: str
    doc_id: str
    title: str
    authority: int
    units: list[Unit]
    valid_from: str
    valid_to: str | None = None
    #: Identifiers this document points at, as the source states them: a notice naming the
    #: CFR sections it amends, a regulation citing its authorising statute. Kept as raw
    #: strings; resolving them to internal ids is a later, separately-testable step.
    references: list[str] = field(default_factory=list)
    url: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.authority not in AUTHORITY_NAMES:
            raise ValueError(f"unknown authority {self.authority!r}")
        anchors = [u.anchor for u in self.units]
        if len(anchors) != len(set(anchors)):
            dupes = sorted({a for a in anchors if anchors.count(a) > 1})
            raise ValueError(f"{self.doc_id}: duplicate unit anchors {dupes[:5]}")

    @property
    def authority_name(self) -> str:
        return AUTHORITY_NAMES[self.authority]

    @property
    def words(self) -> int:
        return sum(len(u.text.split()) for u in self.units)


@runtime_checkable
class Source(Protocol):
    """What every ingester provides.

    Deliberately small. A source fetches and parses; it does not decide chunk boundaries
    beyond its own document structure, does not embed, and does not write to the store --
    those belong to stages that must behave identically whatever the format was, or the
    failure budget cannot compare an ingestion failure in PDF against one in XML.
    """

    #: Stable short name, used as the ``source`` column and in citations.
    name: str
    authority: int

    def documents(self) -> Iterator[SourceDoc]:
        """Yield every document this source offers, already parsed into units."""
        ...


def merge_anchors(units: list[Unit]) -> list[Unit]:
    """Make anchors unique by suffixing repeats, preserving order.

    A backstop, not a strategy. Every parser should produce unique anchors on its own -- the
    CFR parser earned this the hard way, where 13% of citation addresses collided until
    paragraph designators were tracked hierarchically. This exists so that a new source with
    an imperfect parser degrades to an ugly citation rather than to a silently ambiguous one.
    """
    seen: dict[str, int] = {}
    out: list[Unit] = []
    for unit in units:
        anchor = unit.anchor
        if anchor in seen:
            seen[anchor] += 1
            anchor = f"{anchor}.{seen[anchor]}"
        else:
            seen[anchor] = 1
        out.append(unit if anchor == unit.anchor else
                   Unit(anchor=anchor, text=unit.text, heading=unit.heading,
                        kind=unit.kind, locator=unit.locator))
    return out
