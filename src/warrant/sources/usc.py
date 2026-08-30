"""The United States Code: the statute layer, and the top of the authority hierarchy.

5 CFR 630.306 exists because 5 U.S.C. 6304(d) told OPM to make it exist, and the regulation
says so in its own text. A corpus that holds only the regulation can quote the rule but
cannot show the grant of authority behind it, which is the half of the argument a reader
actually disputes.

**Where the XML comes from, and why not from the obvious place.** govinfo publishes the USC
as an annual edition, and its granule URLs look like they should serve USLM::

    https://www.govinfo.gov/content/pkg/USCODE-2023-title5/xml/USCODE-2023-title5-...-sec6304.xml

That URL returns **HTTP 200 and 44 KB of Drupal**: a rendered "Page Not Found" page, for
every title, every year and every section. ``getContentDetail`` for the same granule lists
exactly four formats -- pdf, htm, mods and a whole-title zip -- and no USLM at all. A client
that trusted the status code would have cached the error page as a statute and parsed it into
zero units with no failure anywhere. Hence ``_looks_like_zip`` below: this source checks what
it got, not what the server said about it.

The Office of Law Revision Counsel, which *authors* USLM, publishes it at release points, one
zip per title::

    https://uscode.house.gov/download/releasepoints/us/pl/119/102/xml_usc05@119-102.zip

Title 5 is 2.9 MB compressed and 19 MB of XML -- one request, cached, for every section this
project needs. The all-titles zip is the 107 MB one, and nothing here wants it.
"""

from __future__ import annotations

import logging
import os
import re
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
from lxml import etree
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    AUTHORITY_STATUTE,
    KIND_PROSE,
    KIND_TABLE,
    SourceDoc,
    Unit,
    merge_anchors,
)

log = logging.getLogger(__name__)

USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

#: USLM 1.0, the schema OLRC ships. Declared as the *default* namespace in every release
#: point file, so there is no prefix to hard-code and ``root.iter("section")`` finds nothing.
#: Everything below resolves by local name against whatever the document declares.
USLM_NS = "http://xml.house.gov/schemas/uslm/1.0"

#: Where the current release point is announced. This is the only endpoint that changes:
#: a new public law becomes visible here first, so it is cached with a TTL rather than
#: pinned, for the reason ``corpus/ecfr.py`` pins snapshots but expires its indexes.
RELEASE_POINT_INDEX = "https://uscode.house.gov/download/download.shtml"

#: One zip per title per release point. ``{title}`` is zero-padded to two digits with any
#: appendix letter kept: title 5 is ``usc05``, title 5 appendix is ``usc05a``, title 42 is
#: ``usc42``.
TITLE_XML_URL = ("https://uscode.house.gov/download/releasepoints/us/pl/{congress}/{law}/"
                 "xml_usc{title}@{congress}-{law}.zip")

#: Human-readable address for a section, for the ``url`` a citation shows a reader. OLRC's
#: viewer, not govinfo's, because it serves the same release point this parser read.
SECTION_URL = ("https://uscode.house.gov/view.xhtml?req=granuleid:"
               "USC-prelim-title{title}-section{num}&num=0&edition=prelim")

#: How long the release-point index stays authoritative. A day, matching the corpus config:
#: OLRC publishes a release point every few weeks, and a frozen index is invisible -- it
#: reports no new law, which is indistinguishable from there being none.
INDEX_TTL_HOURS = 24.0


class UscParseError(ValueError):
    """A USLM document could not be read, naming the section.

    lxml reports a line and column in a byte string nobody can find again. Ingestion walks a
    thousand sections out of one 19 MB file, so a failure that does not name the section is a
    failure nobody can reproduce.
    """


class SectionUnavailable(LookupError):
    """The configured section is not in this edition of this title.

    Not an error. Sections are repealed, renumbered and transferred between editions -- title
    5 alone carries 27 repealed and 3 renumbered sections at release point 119-102 -- and a
    config that names one is out of date, not broken.
    """


# ---------------------------------------------------------------------------- parsing

_WS = re.compile(r"\s+")
#: A leading paragraph designator written into prose rather than marked up: "(a)", "(a)(1)".
_INLINE_LABEL = re.compile(r"^\s*\(([A-Za-z0-9]{1,6})\)((?:\s*\([A-Za-z0-9]{1,6}\))*)\s*")
_PAREN = re.compile(r"\(([A-Za-z0-9]{1,6})\)")
#: ``<ref href="/us/usc/t5/s6303">``, and its deeper form ``/us/usc/t5/s5562/a``.
_USC_HREF = re.compile(r"^/us/usc/t([0-9]+[a-z]?)/s([0-9]+[A-Za-z]?(?:[-–][0-9]+)?)")
#: ``identifier="/us/usc/t5/s6304"`` on the section element itself. Unanchored at the end
#: because a section repealed jointly with its neighbour carries *both* paths in one
#: attribute -- ``identifier="/us/usc/t5/s1207 /us/usc/t5/s1208"`` -- and the first is the
#: one the section is filed under. Those combined entries are always repealed and carry no
#: operative text, so the second number is not lost from anything the store would hold.
_SECTION_ID = re.compile(r"^/us/usc/t([0-9]+[a-z]?)/s([^/\s]+)")

#: Every USLM element that addresses a level of the hierarchy. ``level`` is the generic one
#: the converter emits where it could not classify the rung -- four of them in title 5, all
#: inside 5 U.S.C. 13103(h)(2)(i) -- and dropping it would silently lose that text.
_LEVEL_TAGS = frozenset({
    "subsection", "paragraph", "subparagraph", "clause", "subclause",
    "item", "subitem", "subsubitem", "division", "subdivision", "level",
})

#: A level's own words, as opposed to its children's. ``chapeau`` opens a list, ``proviso``
#: and ``continuation`` close it.
_TEXT_TAGS = ("chapeau", "content", "continuation", "proviso")

#: Editorial apparatus: never statutory text, and it nests *inside* content. 5 U.S.C. 6307(d)
#: opens with ``<note type="footnote"><num>1</num> So in original. Probably should be
#: "(e)(1)".</note>``, so a plain ``itertext()`` of that chapeau puts the OLRC editor's aside
#: -- and a stray footnote marker digit -- into the middle of the quoted law.
_DROP_TAGS = frozenset({"note", "notes", "sourceCredit", "toc", "num", "heading"})
#: The marker that points at a footnote, in its two spellings. OLRC writes it as
#: ``<ref class="footnoteRef">1</ref>`` in some sections and a bare ``<sup>1</sup>`` in others
#: -- 5 U.S.C. 6304(f)(1)(H) has one of each, in the two subparagraphs both numbered (H).
#: Kept, a stray "1 " opens the quoted statute. Every one of the 42 ``<sup>`` elements in
#: title 5's operative text is a single digit, so dropping digit-only ones costs no prose.
_FOOTNOTE_REF = "footnoteRef"


def _parser() -> etree.XMLParser:
    """Parse settings stated rather than inherited, as ``corpus/parse.py`` argues.

    ``resolve_entities=False`` is the load-bearing one: it blocks entity expansion, which is
    both the XXE read primitive and the billion-laughs amplifier. A release point is a 19 MB
    file from a public web server, so ``huge_tree`` stays off and the document has to fit
    inside libxml2's ordinary limits.
    """
    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


def _local(el: etree._Element) -> str:
    """Local name of an element, namespace-agnostic.

    OLRC declares USLM as the default namespace and mixes XHTML tables into the same
    document, so tag names arrive as ``{uri}name`` under two different URIs. Matching on the
    local name is what keeps this parser working if OLRC moves to USLM 2.0, which changes the
    namespace URI and keeps the element names.
    """
    return etree.QName(el).localname


def _is_element(node: etree._Element) -> bool:
    return isinstance(node.tag, str)  # comments and processing instructions have callables


def _text_of(el: etree._Element) -> str:
    """Flatten one text-bearing element, dropping the apparatus nested inside it.

    Written as an explicit walk rather than ``itertext()`` because the apparatus is not a
    sibling that can be skipped once -- footnotes, and the marker that points at them, are
    embedded mid-sentence (see ``_DROP_TAGS``).
    """
    parts: list[str] = [el.text or ""]

    def rec(node: etree._Element) -> None:
        for child in node:
            if not _is_element(child):
                continue
            name = _local(child)
            drop = (name in _DROP_TAGS
                    or (name == "ref" and child.get("class") == _FOOTNOTE_REF)
                    or (name == "sup" and "".join(child.itertext()).strip().isdigit()))
            if drop:
                parts.append(child.tail or "")
                continue
            parts.append(child.text or "")
            rec(child)
            parts.append(child.tail or "")

    rec(el)
    return _WS.sub(" ", "".join(parts)).strip()


def _own_text(el: etree._Element) -> str:
    """A level's own words, joined into one string.

    Joined, not emitted separately, because a level has exactly one address. 5 U.S.C. 6304(c)
    is a chapeau ("Annual leave in excess of the amount allowable--"), two paragraphs, and a
    continuation that finishes the sentence the chapeau started; a citation to ``6304#c`` has
    to return both halves, and ``6304#c.2`` is an address no reader would ever write down.
    The intervening paragraphs keep their own anchors, so nothing is duplicated.
    """
    pieces = [_text_of(c) for c in el
              if _is_element(c) and _local(c) in _TEXT_TAGS]
    return " ".join(p for p in pieces if p)


def _table_text(el: etree._Element) -> str:
    """Flatten a table to one line per row, cells separated by ' | '.

    Serialised rather than split per cell, for the reason the eCFR parser gives: a statutory
    table is one semantic unit -- a pay schedule, a conversion table -- and splitting it
    destroys the row relationship that makes it answerable.
    """
    rows: list[str] = []
    for tr in el.iter():
        if not _is_element(tr) or _local(tr) != "tr":
            continue
        cells = [_text_of(td) for td in tr
                 if _is_element(td) and _local(td) in ("td", "th")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return _text_of(el)
    return "\n".join(rows)


_ROMAN = re.compile(r"^[ivxlcdm]+$")
_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(token: str) -> int | None:
    low = token.lower()
    if not _ROMAN.match(low):
        return None
    total = 0
    for i, ch in enumerate(low):
        v = _ROMAN_VALUE[ch]
        nxt = _ROMAN_VALUE.get(low[i + 1]) if i + 1 < len(low) else None
        total += -v if nxt and nxt > v else v
    return total


def _alpha_to_int(token: str) -> int | None:
    """Spreadsheet-style ordinal: a=1 ... z=26, aa=27."""
    low = token.lower()
    if not low.isalpha():
        return None
    n = 0
    for ch in low:
        n = n * 26 + (ord(ch) - 96)
    return n


def _is_successor(token: str, previous: str) -> bool:
    """Is ``token`` the next designator after ``previous`` in the same numbering system?

    Deliberately a copy of the same predicate in ``corpus/parse.py`` rather than an import of
    it: that one is private to the eCFR parser, and coupling two format parsers through a
    private helper means a fix for one silently rewrites the anchors of the other. The idea is
    what is being reused, and it is worth restating why it exists. Classifying a designator by
    character class alone gets ``(ii)`` wrong after ``(b)(1)(i)`` -- ``ii`` is equally the
    ninth letter and the second roman numeral, and only what came before decides. In the CFR
    corpus that ambiguity left 13% of citation addresses matching more than one paragraph.
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


def _place(stack: list[str], token: str, depth: int | None) -> None:
    """Put a designator at its level in the hierarchy, nesting first.

    USLM nests, and where it does the document's own structure outranks any inference from
    the designators. Checked, not assumed: across all 20,217 level elements of title 5 at
    release point 119-102, the anchor derived from element nesting agreed with OLRC's own
    ``identifier`` attribute 20,217 times. Sequence continuity would have been *wrong* in at
    least one real place -- 5 U.S.C. 7512 lists paragraphs (1)-(5) and then subparagraphs
    (A)-(F) as siblings at the same rung, and a designator stack would file (A) under (5) and
    mint ``7512#5-A``, which OLRC's ``/us/usc/t5/s7512/A`` flatly contradicts.

    ``depth is None`` is the case nesting cannot answer: a designator written into prose with
    no element of its own, which is how pre-USLM conversions and hand-repaired text arrive.
    There, and only there, sequence continuity decides, exactly as it does for the CFR.
    """
    if depth is not None:
        del stack[depth:]
        stack.append(token)
        return
    for i in range(len(stack) - 1, -1, -1):
        if _is_successor(token, stack[i]):
            del stack[i:]
            stack.append(token)
            return
    stack.append(token)


def _designator(el: etree._Element) -> str:
    """The level's designator, from ``<num value="a">(a)</num>``.

    ``@value`` is preferred over the element's text because the text is the *rendered* form --
    "(a)", "§ 6304." -- and putting that in an anchor makes ``6304#(a)`` instead of
    ``6304#a``. Every one of title 5's 20,217 level elements carries ``@value``; the text
    fallback is for titles where the converter did not emit it.
    """
    for child in el:
        if _is_element(child) and _local(child) == "num":
            value = (child.get("value") or "").strip()
            if value:
                return value
            return (child.text or "").strip().strip("()§. ")
    return ""


@dataclass(frozen=True)
class UscSection:
    """One section of the Code, parsed. The intermediate ``documents()`` builds a doc from."""

    identifier: str  # "/us/usc/t5/s6304"
    title: str  # "5"
    number: str  # "6304"
    heading: str
    units: list[Unit]
    references: list[str]
    source_credit: str = ""
    status: str = ""
    chapter: str = ""

    @property
    def citation(self) -> str:
        return f"{self.title} U.S.C. {self.number}"


def _emit(out: list[Unit], anchor: str, text: str, *, heading: str, locator: str,
          kind: str = KIND_PROSE) -> None:
    if text:
        out.append(Unit(anchor=anchor, text=text, heading=heading, kind=kind, locator=locator))


def _walk(container: etree._Element, depth: int, stack: list[str], out: list[Unit],
          heading: str, counter: list[int]) -> None:
    """Emit units for one container's children, depth-first in document order."""
    for child in container:
        if not _is_element(child):
            continue
        name = _local(child)
        if name in _DROP_TAGS:
            continue
        if name == "table":
            counter[1] += 1
            anchor = "-".join([*stack, f"t{counter[1]}"])
            _emit(out, anchor, _table_text(child), heading=heading,
                  locator=child.get("identifier") or "", kind=KIND_TABLE)
            continue
        if name in _TEXT_TAGS:
            # Text sitting directly under the container rather than inside a level element.
            # Under a level it has already been folded into that level's unit by _own_text;
            # under the section it is undesignated prose, unless the designator was written
            # into the sentence instead of marked up.
            if depth > 0:
                continue
            text = _text_of(child)
            if not text:
                continue
            match = _INLINE_LABEL.match(text)
            if match:
                _place(stack, match.group(1), None)
                for extra in _PAREN.findall(match.group(2) or ""):
                    _place(stack, extra, None)
                _emit(out, "-".join(stack), text[match.end():].strip(), heading=heading,
                      locator=child.get("identifier") or "")
            else:
                counter[0] += 1
                _emit(out, f"p{counter[0]}", text, heading=heading,
                      locator=child.get("identifier") or "")
            continue
        if name not in _LEVEL_TAGS:
            continue
        token = _designator(child)
        if token:
            _place(stack, token, depth)
            anchor = "-".join(stack)
        else:
            # A level with no designator of its own. Positional, so it stays citable and
            # unique; ``merge_anchors`` is the backstop if even that collides.
            counter[0] += 1
            anchor = "-".join([*stack[:depth], f"p{counter[0]}"])
            del stack[depth:]
            stack.append(f"p{counter[0]}")
        _emit(out, anchor, _own_text(child), heading=heading,
              locator=child.get("identifier") or "")
        _walk(child, depth + 1, stack, out, heading, counter)


def _section_element(root: etree._Element) -> etree._Element | None:
    """The first *operative* section in a document.

    "Operative" excludes sections quoted inside notes: an amendment note reproduces the text
    it inserted, wrapped in ``<quotedContent>``, and title 5 contains 193 such quoted sections
    against 1,163 real ones. Ingesting a quoted section as law would put repealed text in the
    store with a current validity date.
    """
    if _local(root) == "section":
        return root
    for el in root.iter():
        if not _is_element(el) or _local(el) != "section":
            continue
        if any(_is_element(a) and _local(a) in ("notes", "note", "quotedContent")
               for a in el.iterancestors()):
            continue
        return el
    return None


def _references(section: etree._Element, own: str) -> list[str]:
    """The section's own citation, then every USC section its operative text points at.

    Own citation first so the CFR->USC join has a key on every document even when the section
    cites nothing. Only ``/us/usc/`` hrefs are kept: the ``/us/pl/`` and ``/us/stat/`` refs in
    a section are its enactment chain, which says how the text got here rather than what it
    depends on, and mixing the two would make "what does 6304 rely on" return 30 public laws.
    """
    seen: dict[str, None] = {own: None}
    for el in section.iter():
        if not _is_element(el) or _local(el) != "ref":
            continue
        if any(_is_element(a) and _local(a) in ("notes", "note", "sourceCredit")
               for a in el.iterancestors()):
            continue
        match = _USC_HREF.match(el.get("href") or "")
        if match:
            seen.setdefault(f"{match.group(1)} U.S.C. {match.group(2)}", None)
    return list(seen)


def _child_text(el: etree._Element, name: str) -> str:
    for child in el:
        if _is_element(child) and _local(child) == name:
            return _text_of(child)
    return ""


def read_section(section: etree._Element, *, doc_id: str) -> UscSection:
    """Turn one USLM ``<section>`` element into units and metadata."""
    identifier = (section.get("identifier") or "").strip()
    match = _SECTION_ID.match(identifier)
    if not match:
        raise UscParseError(f"{doc_id}: section has no usable identifier "
                            f"(got {identifier!r})")
    title, number = match.group(1), match.group(2)
    # The heading is the section's, and it belongs on the unit's ``heading`` field, not in
    # its text: a retrieved paragraph needs to say which section it is from, and a heading
    # concatenated into the prose would be indexed and quoted as if Congress had written it
    # mid-sentence. ``_DROP_TAGS`` holds both ``num`` and ``heading`` for the same reason.
    heading = _child_text(section, "heading")
    units: list[Unit] = []
    _walk(section, 0, [], units, heading, [0, 0])
    return UscSection(
        identifier=identifier,
        title=title,
        number=number,
        heading=heading,
        # Not merely a backstop here: the Code itself collides. Congress enacted two
        # subparagraphs (H) in 5 U.S.C. 6304(f)(1) and two subsections (d) in 5 U.S.C.
        # 6307 -- OLRC footnotes both ("So in original. Two subpars. (H) have been
        # enacted.") and gives each pair the *same* identifier. So section-unique anchors
        # cannot be taken from the source, and ``6304#f-1-H.2`` -- an ugly citation that
        # resolves to exactly one paragraph -- is the only honest answer.
        units=merge_anchors(units),
        references=_references(section, f"{title} U.S.C. {number}"),
        source_credit=_child_text(section, "sourceCredit"),
        status=(section.get("status") or "").strip(),
        chapter=_ancestor_num(section, "chapter"),
    )


def _ancestor_num(el: etree._Element, name: str) -> str:
    for anc in el.iterancestors():
        if _is_element(anc) and _local(anc) == name:
            return _designator(anc)
    return ""


def parse_uslm(xml: bytes, *, doc_id: str) -> list[Unit]:
    """Units for the first operative section in a USLM document.

    ``doc_id`` names the section in anything raised. It is the only thing that makes a failure
    locatable when the input is one section out of a 19 MB title file.
    """
    if not xml.strip():
        raise UscParseError(f"{doc_id}: empty document, nothing to parse")
    try:
        root = etree.fromstring(xml, _parser())
    except etree.XMLSyntaxError as exc:
        raise UscParseError(f"{doc_id}: not well-formed USLM ({exc})") from exc
    if root is None:
        raise UscParseError(f"{doc_id}: no document element")
    section = _section_element(root)
    if section is None:
        raise UscParseError(f"{doc_id}: no <section> element in this document "
                            f"(root is <{_local(root)}>)")
    return read_section(section, doc_id=doc_id).units


# ---------------------------------------------------------------------------- fetching


@dataclass(frozen=True)
class ReleasePoint:
    """A published state of the Code, identified by the last public law folded into it."""

    congress: int
    law: int
    #: ISO date of that public law. This is what ``valid_from`` becomes.
    date: str

    @property
    def name(self) -> str:
        return f"{self.congress}-{self.law}"


#: "Public Law 119-102 (07/12/2026)" on the OLRC download page.
_RELEASE_POINT = re.compile(r"Public\s+Law\s+(\d+)-(\d+)\s*\((\d{2})/(\d{2})/(\d{4})\)")
_TITLE_CODE = re.compile(r"^(\d+)([a-z]?)$")


def parse_release_point(html: bytes) -> ReleasePoint:
    """Read the current release point out of the OLRC download page."""
    match = _RELEASE_POINT.search(html.decode("utf-8", "replace"))
    if not match:
        raise UscParseError("uscode.house.gov download page: no release point found; "
                            "the page layout changed and the release point must be "
                            "configured explicitly")
    congress, law, mm, dd, yyyy = match.groups()
    return ReleasePoint(congress=int(congress), law=int(law), date=f"{yyyy}-{mm}-{dd}")


def title_code(title: str) -> str:
    """``"5"`` -> ``"05"``, ``"5a"`` -> ``"05a"``, ``"42"`` -> ``"42"``."""
    match = _TITLE_CODE.match(str(title).strip().lower())
    if not match:
        raise ValueError(f"not a US Code title: {title!r}")
    return f"{int(match.group(1)):02d}{match.group(2)}"


def _looks_like_zip(blob: bytes) -> bool:
    """Is this actually a zip?

    Asked because govinfo answers a missing USLM granule with HTTP 200 and an HTML error
    page, and OLRC sits behind the same kind of front end. A source that trusts the status
    code caches the error page and reports zero sections rather than a failure.
    """
    return blob[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


@dataclass
class UscConfig:
    """Which slice of the Code to ingest.

    A title at a time, never the whole Code: the all-titles zip is 107 MB and 53 of its
    titles have nothing to do with federal employment. ``sections`` and ``chapters`` narrow
    it further and are unioned; leaving both empty takes the whole title, which is 1,163
    sections for title 5 and is a deliberate choice rather than an accident of an empty list.
    """

    title: str = "5"
    #: Section numbers as they are cited: "6304", "552a", "8331".
    sections: list[str] = field(default_factory=list)
    #: Chapter numbers, as ``<chapter><num value="63">``. Chapter 63 is leave.
    chapters: list[str] = field(default_factory=list)
    #: "119-102" pins a release point; empty discovers the current one.
    release_point: str = ""
    cache_dir: Path = Path("data/usc")
    request_delay_s: float = 1.0
    timeout_s: float = 180.0
    index_ttl_hours: float = INDEX_TTL_HOURS
    refresh: bool = False


@dataclass
class UscClient:
    """Polite, cached, retrying access to OLRC release points.

    Caching policy is split the way ``corpus/ecfr.py`` splits it, and for the same reason. A
    release point is immutable -- ``xml_usc05@119-102.zip`` will never change, because the
    next public law makes a new release point rather than editing this one -- so it is pinned
    forever. The download page that *announces* release points is the opposite: it is exactly
    where new law first appears, so pinning it would freeze the Code with no error to say so.
    """

    cache_dir: Path = Path("data/usc")
    delay_s: float = 1.0
    timeout_s: float = 180.0
    index_ttl_hours: float = INDEX_TTL_HOURS
    refresh: bool = False
    _last_request: float = 0.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)  # a public web server; do not hammer it
        r = httpx.get(url, headers={"User-Agent": USER_AGENT},
                      timeout=self.timeout_s, follow_redirects=True)
        self._last_request = time.monotonic()
        r.raise_for_status()
        return r.content

    def _cached(self, url: str, name: str, *, ttl_s: float | None) -> bytes:
        """Fetch through the on-disk cache. ``ttl_s is None`` means immutable.

        Writes are atomic: a truncated file left by a Ctrl-C during a 3 MB download would be
        trusted forever, because the cache only checks that the path exists, and the symptom
        would be a BadZipFile in a later build naming no URL.
        """
        path = self.cache_dir / name
        cached = path.read_bytes() if path.exists() else None
        fresh = cached is not None and (
            ttl_s is None or (time.time() - path.stat().st_mtime) <= ttl_s)
        if cached is not None and fresh and not self.refresh:
            log.debug("cache hit %s", name)
            return cached
        log.info("GET %s", url)
        try:
            content = self._fetch(url)
        except httpx.HTTPError as exc:
            if cached is not None:
                log.warning("refresh of %s failed (%s); using the cached copy", name, exc)
                return cached
            raise
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        log.info("cached %s (%d bytes)", name, len(content))
        return content

    def current_release_point(self) -> ReleasePoint:
        html = self._cached(RELEASE_POINT_INDEX, "download.shtml",
                            ttl_s=self.index_ttl_hours * 3600.0)
        rp = parse_release_point(html)
        log.info("current US Code release point: Public Law %s (%s)", rp.name, rp.date)
        return rp

    def title_xml(self, title: str, rp: ReleasePoint) -> bytes:
        """The whole title as USLM, unzipped.

        One request per title per release point. The zip holds a single member, ``uscNN.xml``;
        it is read by position rather than by name so an appendix title, whose member is named
        differently, does not need a special case.
        """
        code = title_code(title)
        url = TITLE_XML_URL.format(congress=rp.congress, law=rp.law, title=code)
        blob = self._cached(url, f"xml_usc{code}@{rp.name}.zip", ttl_s=None)
        if not _looks_like_zip(blob):
            raise UscParseError(f"{url}: served {len(blob)} bytes that are not a zip; "
                                f"this is an error page, not a release point")
        with zipfile.ZipFile(BytesIO(blob)) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".xml")]
            if not members:
                raise UscParseError(f"{url}: no XML member in the release point zip")
            return z.read(members[0])


# ---------------------------------------------------------------------------- source


@dataclass
class UscSource:
    """US Code sections as ``SourceDoc``s, at statute authority.

    One document per section, because the section is the unit a statute is cited by and the
    unit a regulation names when it states its authority.
    """

    config: UscConfig = field(default_factory=UscConfig)
    client: UscClient | None = None

    name: str = "usc"
    authority: int = AUTHORITY_STATUTE

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = UscClient(
                cache_dir=Path(self.config.cache_dir),
                delay_s=self.config.request_delay_s,
                timeout_s=self.config.timeout_s,
                index_ttl_hours=self.config.index_ttl_hours,
                refresh=self.config.refresh,
            )

    def release_point(self) -> ReleasePoint:
        pinned = (self.config.release_point or "").strip()
        if not pinned:
            return self.client.current_release_point()
        try:
            congress, law = (int(p) for p in pinned.split("-", 1))
        except ValueError as exc:
            raise ValueError(f"release_point must look like '119-102', got "
                             f"{pinned!r}") from exc
        # A pinned release point has no date attached, so the edition date is taken from the
        # document's own <dcterms:created>. Resolved in ``documents()`` where the XML is in
        # hand; an empty date here is a placeholder, not a claim.
        return ReleasePoint(congress=congress, law=law, date="")

    def _selected(self, root: etree._Element, rp: ReleasePoint) -> Iterator[etree._Element]:
        """Operative sections matching the configured sections and chapters."""
        wanted = {s.strip().lower() for s in self.config.sections if s.strip()}
        chapters = {c.strip().lower() for c in self.config.chapters if c.strip()}
        found: set[str] = set()
        for el in root.iter():
            if not _is_element(el) or _local(el) != "section":
                continue
            if any(_is_element(a) and _local(a) in ("notes", "note", "quotedContent")
                   for a in el.iterancestors()):
                continue
            match = _SECTION_ID.match((el.get("identifier") or "").strip())
            if not match:
                continue
            number = match.group(2).lower()
            if not wanted and not chapters:
                yield el
                continue
            if number in wanted or _ancestor_num(el, "chapter").lower() in chapters:
                found.add(number)
                yield el
        for missing in sorted(wanted - found):
            # Repealed, renumbered, or simply mistyped in the config. Logged and skipped:
            # the release point is a snapshot of the Code as it stands, and a section that is
            # not in it is a fact about the Code, not a failure of ingestion.
            log.warning("%s U.S.C. %s is not in release point %s; skipped",
                        self.config.title, missing, rp.name)

    def documents(self) -> Iterator[SourceDoc]:
        rp = self.release_point()
        xml = self.client.title_xml(self.config.title, rp)
        try:
            root = etree.fromstring(xml, _parser())
        except etree.XMLSyntaxError as exc:
            raise UscParseError(f"US Code title {self.config.title} at release point "
                                f"{rp.name}: not well-formed USLM ({exc})") from exc
        # The USC is republished as an edition, not amended in place: release point 119-102
        # *replaces* 119-101 wholesale, and there is no per-section amendment date anywhere
        # in the file. So ``valid_from`` is the edition date and ``valid_to`` is None -- open,
        # meaning "as far as this source knows, still the law". This is NOT the same claim the
        # CFR source makes, where valid_from is the date an amendment took effect and
        # valid_to closes at the next one. A reader of the store who treats a statute's
        # valid_from as an amendment date will conclude 5 U.S.C. 6304 changed in July 2026,
        # when in fact only the snapshot did; ``meta["release_point"]`` is what distinguishes
        # them, and it is recorded on every document for exactly that reason.
        valid_from = rp.date or _created_date(root) or ""
        package_url = TITLE_XML_URL.format(congress=rp.congress, law=rp.law,
                                           title=title_code(self.config.title))
        count = 0
        for el in self._selected(root, rp):
            identifier = (el.get("identifier") or "").strip()
            try:
                parsed = read_section(el, doc_id=identifier or "<unidentified section>")
            except UscParseError as exc:
                log.warning("skipping a section of title %s: %s", self.config.title, exc)
                continue
            if not parsed.units:
                # Repealed and omitted sections keep a heading and a note and nothing else.
                log.info("%s has no operative text (status %r); skipped",
                         parsed.citation, parsed.status or "none")
                continue
            count += 1
            yield SourceDoc(
                source=self.name,
                doc_id=f"usc-t{parsed.title}-s{parsed.number}",
                title=f"{parsed.citation} - {parsed.heading}".strip(" -"),
                authority=self.authority,
                units=parsed.units,
                valid_from=valid_from,
                valid_to=None,
                references=parsed.references,
                url=SECTION_URL.format(title=parsed.title, num=parsed.number),
                meta={
                    "release_point": rp.name,
                    "usc_title": parsed.title,
                    "section": parsed.number,
                    "chapter": parsed.chapter,
                    "status": parsed.status,
                    "identifier": parsed.identifier,
                    "source_credit": parsed.source_credit,
                    "package_url": package_url,
                },
            )
        log.info("usc: %d sections from title %s at release point %s",
                 count, self.config.title, rp.name)


_DCTERMS_CREATED = "{http://purl.org/dc/terms/}created"


def _created_date(root: etree._Element) -> str:
    """Date part of ``<dcterms:created>`` in the USLM ``<meta>`` block.

    The fallback edition date for a pinned release point, whose public-law date is not in the
    file. It is the date OLRC converted the release point, a few days after the public law --
    2026-07-16 for Public Law 119-102 of 2026-07-12 -- so it is an upper bound on when the
    text became the law, never a claim about when it changed.
    """
    for el in root.iter():
        if _is_element(el) and el.tag == _DCTERMS_CREATED:
            stamp = (el.text or "").strip()
            try:
                # Naive on purpose: OLRC stamps a local wall clock with no offset, and
                # attaching a timezone here would move the date by a day for no reason.
                return datetime.fromisoformat(stamp).date().isoformat()
            except ValueError:
                return stamp[:10]
    return ""
