"""eCFR XML into the units Warrant retrieves and cites.

The regulation's own hierarchy is the chunking strategy. Title -> chapter -> part ->
subpart -> section -> paragraph is not an arbitrary segmentation someone tuned; it is how
the document is written, cross-referenced and amended. Inventing a fixed-token window on
top of it would throw away the one structure that makes citation and applicability
reasoning possible.

  section    the retrieval unit    <DIV8 TYPE="SECTION" N="630.1203">
  paragraph  the citation unit     <P>(a) An employee shall be entitled to ...

A citation therefore reads ``630.1203#a`` -- addressable, stable across snapshots when the
text is only amended, and exactly what a reader would write down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from .apparatus import strip_apparatus, text_of

#: Leading paragraph designator: (a), (a)(1), (b)(2)(i) ...
_LABEL = re.compile(r"^\s*\(([a-zA-Z0-9]{1,4})\)((?:\s*\([a-zA-Z0-9]{1,4}\))*)")
_WS = re.compile(r"\s+")
#: eCFR heads read "§ 630.1203 Leave entitlement." -- keep the title, drop the number.
_HEAD_NUM = re.compile(r"^\s*(?:&#167;|§)?\s*[\d.\-]+\s*")


@dataclass(frozen=True)
class Paragraph:
    anchor: str  # "a", "a-1", or "p3" when the paragraph carries no designator
    text: str


@dataclass(frozen=True)
class Section:
    identifier: str  # "630.1203"
    heading: str
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    subpart: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.identifier}"


def _anchor(raw_label: str, extra: str, ordinal: int) -> str:
    if not raw_label:
        return f"p{ordinal}"
    tail = re.findall(r"\(([a-zA-Z0-9]{1,4})\)", extra or "")
    return "-".join([raw_label, *tail])


def _paragraphs(node: etree._Element) -> list[Paragraph]:
    out: list[Paragraph] = []
    for i, p in enumerate(node.iter("P"), start=1):
        text = _WS.sub(" ", "".join(strip_apparatus(p).itertext())).strip()
        if not text:
            continue
        m = _LABEL.match(text)
        out.append(Paragraph(anchor=_anchor(m.group(1) if m else "",
                                            m.group(2) if m else "", i),
                             text=text))
    return out


def _subpart_of(node: etree._Element) -> str | None:
    for anc in node.iterancestors("DIV6"):
        if anc.get("TYPE") == "SUBPART":
            return anc.get("N")
    return None


def parse_sections(xml: bytes) -> list[Section]:
    """Every section in a part snapshot, apparatus already removed."""
    root = etree.fromstring(xml)
    sections: list[Section] = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        ident = (div.get("N") or "").strip()
        if not ident:
            continue
        head_el = div.find("HEAD")
        heading = ""
        if head_el is not None:
            heading = _HEAD_NUM.sub("", _WS.sub(" ", "".join(head_el.itertext())).strip())
        sections.append(
            Section(
                identifier=ident,
                heading=heading.strip(" .§"),
                text=text_of(div),
                paragraphs=_paragraphs(div),
                subpart=_subpart_of(div),
            )
        )
    return sections


def section_index(xml: bytes) -> dict[str, Section]:
    return {s.identifier: s for s in parse_sections(xml)}
