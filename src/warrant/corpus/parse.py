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

from .apparatus import APPARATUS_TAGS, strip_apparatus, text_of

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


_ROMAN = re.compile(r"^[ivxlcdm]+$")
_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(token: str) -> int | None:
    if not _ROMAN.match(token):
        return None
    total = 0
    for i, ch in enumerate(token):
        v = _ROMAN_VALUE[ch]
        nxt = _ROMAN_VALUE.get(token[i + 1]) if i + 1 < len(token) else None
        total += -v if nxt and nxt > v else v
    return total


def _alpha_to_int(token: str) -> int | None:
    """Spreadsheet-style ordinal: a=1 ... z=26, aa=27. CFR rarely goes past one letter."""
    low = token.lower()
    if not low.isalpha():
        return None
    n = 0
    for ch in low:
        n = n * 26 + (ord(ch) - 96)
    return n


def _is_successor(token: str, previous: str) -> bool:
    """Is ``token`` the next designator after ``previous`` in the same numbering system?

    Sequence continuity, not character class, is what places a designator. Classifying by
    type alone gets ``(ii)`` wrong after ``(b)(1)(i)``: ``ii`` looks like a plain letter, and
    the ambiguity between the ninth letter and the second roman numeral cannot be resolved
    without knowing what came before.
    """
    if token.isdigit() and previous.isdigit():
        return int(token) == int(previous) + 1
    if token.isupper() != previous.isupper():
        return False
    tr, pr = _roman_to_int(token), _roman_to_int(previous)
    if tr is not None and pr is not None and tr == pr + 1:
        return True
    ta, pa = _alpha_to_int(token), _alpha_to_int(previous)
    return ta is not None and pa is not None and ta == pa + 1


#: The opening designator of each level of the CFR hierarchy: (a) (1) (i) (A).
_FIRST = frozenset({"a", "1", "i", "A"})


def _push(stack: list[str], token: str) -> None:
    """Place a designator at its level in the hierarchy.

    A designator that continues an existing level replaces it and closes everything below.
    A designator that opens a level starts a new one. Anything else -- most often a level
    whose first item was written inline with its parent, as in ``(d) ... (1) ...`` followed
    by a standalone ``(2)`` -- opens a new level too, which keeps the address unique and in
    document order even when the hierarchy cannot be recovered exactly.
    """
    for i in range(len(stack) - 1, -1, -1):
        if _is_successor(token, stack[i]):
            del stack[i:]
            stack.append(token)
            return
    if token in _FIRST:
        stack.append(token)
        return
    stack.append(token)


#: Body elements that carry regulatory prose. ``P`` is the ordinary paragraph; the ``FP``
#: family is a *flush paragraph* -- unindented continuation text, used for the closing
#: sentence of a list and for the notes under a table. Reading only ``P`` silently dropped
#: 18,705 words, 4.5% of the corpus, concentrated in the Federal Wage System parts the
#: applicability story is built on: 88% of §532.313 and 46% of §531.214 were simply absent.
#:
#: That loss was invisible to the failure budget by construction. Its ``ingestion`` row asks
#: whether a gold chunk is in the store, and gold chunks are minted by this same function --
#: so text this parser never emitted could never be missed. A row that can only read zero is
#: not measuring anything, which is why the coverage assertion in tests/invariants exists.
_PROSE_TAGS = frozenset({"P", "FP", "FP-1", "FP-2", "FP1-2", "FP-DASH", "PSPACE"})
_TABLE_TAGS = frozenset({"TABLE", "GPOTABLE"})


def _table_text(node: etree._Element) -> str:
    """Flatten a table to one line per row, cells separated by ' | '.

    Serialised rather than skipped or split. A regulatory table is a single semantic unit --
    a wage schedule, a step progression -- and splitting it per cell destroys the row
    relationship that makes it answerable at all.
    """
    rows: list[str] = []
    for tr in node.iter("TR"):
        cells = [_WS.sub(" ", "".join(td.itertext())).strip()
                 for td in tr.iter("TD", "TH", "ENT")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        rows = [_WS.sub(" ", "".join(node.itertext())).strip()]
    return "\n".join(r for r in rows if r)


def _body_elements(section: etree._Element):
    """Prose and table elements in document order, without descending into them.

    A plain ``iter("P")`` would miss flush paragraphs and tables; iterating several tags
    naively would double-count the paragraphs nested inside an ``EXTRACT``. Walking and
    pruning at each body element gives each piece of text exactly once.
    """
    stack = list(reversed(list(section)))
    while stack:
        el = stack.pop()
        if el.tag in _PROSE_TAGS or el.tag in _TABLE_TAGS:
            yield el
            continue
        if el.tag in ("HEAD", *APPARATUS_TAGS):
            continue
        stack.extend(reversed(list(el)))


def _paragraphs(node: etree._Element) -> list[Paragraph]:
    """Paragraphs with hierarchical, section-unique anchors.

    Anchors must be unique inside a section version or a citation does not identify
    anything. Before the designator stack was tracked, 13% of addresses in the corpus were
    ambiguous -- ``550.703#a`` matched four different paragraphs, because a section with
    several sub-lists restarts at ``(a)`` and ``(1)`` repeatedly. A collision suffix is kept
    as a backstop for markup the stack cannot resolve; it should stay unused, and a test
    asserts uniqueness over the real corpus.
    """
    out: list[Paragraph] = []
    stack: list[str] = []
    used: dict[str, int] = {}
    tables = 0
    for i, p in enumerate(_body_elements(node), start=1):
        if p.tag in _TABLE_TAGS:
            text = _table_text(strip_apparatus(p))
            if not text:
                continue
            tables += 1
            anchor = f"t{tables}"
            if anchor in used:
                used[anchor] += 1
                anchor = f"{anchor}.{used[anchor]}"
            else:
                used[anchor] = 1
            out.append(Paragraph(anchor=anchor, text=text))
            continue
        text = _WS.sub(" ", "".join(strip_apparatus(p).itertext())).strip()
        if not text:
            continue
        m = _LABEL.match(text)
        if m:
            _push(stack, m.group(1))
            for extra in re.findall(r"\(([a-zA-Z0-9]{1,4})\)", m.group(2) or ""):
                _push(stack, extra)
            anchor = "-".join(stack)
        else:
            # Flush text with no designator: an introductory or concluding paragraph.
            anchor = f"p{i}"
        if anchor in used:
            used[anchor] += 1
            anchor = f"{anchor}.{used[anchor]}"
        else:
            used[anchor] = 1
        out.append(Paragraph(anchor=anchor, text=text))
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
