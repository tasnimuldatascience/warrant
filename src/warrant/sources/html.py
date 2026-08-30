"""OPM fact sheets: the guidance layer, scraped out of a government web page.

Guidance is the one layer of the hierarchy in ``base.py`` that has no machine-readable
feed. There is no versioner API for "how OPM says its own regulation should be read"; there
is a CMS-rendered page that changes without notice, carries no version identifier, and
buries three screens of prose inside a template. Ingesting it is worth the trouble for
exactly one reason: a fact sheet names the sections it interprets, so ``references`` joins
guidance to the regulation, and a retrieval that finds the plain-language answer can be made
to also produce the rule it came from -- and, when the two disagree, to notice.

**Boilerplate is this module's apparatus problem.** ``corpus/apparatus.py`` exists because
43% of detected "changes" in the CFR corpus were editorial material moving around rather
than the law changing. HTML has the same disease and, measured, almost the same number.
The live OPM annual-leave fact sheet is 127,150 bytes of markup. Naive text extraction over
it yields 28,565 characters; what this module keeps is 15,693 characters in 21 units. The
other **12,872 characters -- 45.1% -- were chrome**: the mega menu, the breadcrumb trail,
the "share this page" toolbar, the agency banner, the footer link farm. A chunker pointed at
the raw page would embed the site menu once per fact sheet and retrieve it forever; a differ
pointed at it would report a nav-link edit as a change in guidance.

The removal is deliberately layered, because each layer alone is wrong:

  1. structural tags (``script``, ``nav``, ``footer`` ...) -- safe, and nowhere near enough:
     OPM's mega menu is ``<div class="usa-nav-container">``, not ``<nav>``.
  2. class/id patterns -- catches the rest, and is the dangerous one. A pattern like
     ``header`` matches ``page-header`` *and* ``content-header``, so anything that is or
     contains a main-content landmark is protected from this pass.
  3. landmark selection (``<main>``, ``[role=main]``, ``<article>``) -- correct when the page
     has one. Federal sites usually do; this one does.
  4. a density fallback for when it does not, and a logged fallback to ``<body>`` when even
     that fails, because silently emitting nothing is the failure mode this project exists
     to make visible.

Everything below ``extract_units`` is pure and offline-testable; the network lives in
``HtmlFetcher``, whose caching policy follows ``corpus/ecfr.py`` with one difference. eCFR
``/full/`` snapshots are immutable history and are pinned forever. A guidance page is the
opposite -- it is edited in place, with no notice and no changelog -- so it carries a TTL and
a recorded content hash, which is the only way a change is detectable at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from lxml import etree
from lxml import html as lxml_html
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    AUTHORITY_GUIDANCE,
    KIND_PROSE,
    KIND_TABLE,
    SourceDoc,
    Unit,
    merge_anchors,
)

log = logging.getLogger(__name__)

USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

#: Fact sheets to ingest. A list rather than a crawl: OPM's own index pages are themselves
#: mostly navigation, and a crawler would spend its politeness budget rediscovering pages
#: whose URLs have not moved in years.
OPM_FACT_SHEETS: tuple[str, ...] = (
    "https://www.opm.gov/policy-data-oversight/pay-leave/leave-administration/fact-sheets/annual-leave/",
)

#: How long a cached page stays authoritative. Guidance is edited in place, so caching it
#: forever means the corpus keeps quoting a fact sheet that no longer exists in that form,
#: with nothing anywhere reporting the divergence.
PAGE_TTL_HOURS = 24.0

#: Elements that are never content, whatever they contain. Note ``form``: OPM's search box
#: and its "was this page helpful" widget both live in one, and both are pure chrome.
BOILERPLATE_TAGS = frozenset(
    {"script", "style", "nav", "header", "footer", "aside", "form", "noscript"}
)

#: Class/id substrings that mark chrome. Substring rather than whole-token matching because
#: the real attribute values are ``usa-nav-container``, ``usa-banner__header``,
#: ``breadcrumb-list``, ``addthis_toolbox`` -- the marker is always a fragment of a longer
#: compound name, so anchoring to token boundaries would match almost none of them.
BOILERPLATE_ATTR = re.compile(
    r"nav|menu|breadcrumb|crumb|skip|social|share|addthis|cookie|banner|footer|masthead|"
    r"sidebar|side-bar|subscribe|newsletter|feedback|search|toolbar|utility|pagination|"
    r"related-|widget|dropdown|megamenu|mega-menu|usa-overlay|screen-reader|sr-only|"
    r"visually-hidden|print-only",
    re.I,
)

#: Landmarks that mean "the page's own content starts here", in preference order.
MAIN_XPATH = (
    "//main",
    "//*[@role='main']",
    "//article",
    "//*[@id='main-content' or @id='main' or @id='content']",
)

#: Containers the density heuristic will consider. Restricted to wrappers: allowing every
#: element would let a single dense ``<p>`` win on text-per-tag against the article holding
#: it, which is the standard way this heuristic is got wrong.
_CONTAINER_TAGS = frozenset({"body", "main", "article", "section", "div", "td"})

#: A density candidate must hold this share of the page's remaining text. Without it the
#: heuristic returns the tightest wrapper around *any* paragraph; with it, it returns the
#: tightest wrapper that still holds substantially all of the content.
DENSITY_MIN_SHARE = 0.4

#: Below this much text, the chosen region is not a fact sheet -- it is a redirect stub, an
#: error page rendered with a 200, or garbage that lxml salvaged into a stray ``<p>``.
#: Emitting units from it would put non-content in the store with a citable address.
MIN_REGION_CHARS = 120

#: Block elements that carry text. ``dt``/``dd``/``li`` are listed so list structure survives
#: as line breaks: OPM writes eligibility rules as bullets, and flattening them into one
#: run-on sentence destroys the enumeration a reader is meant to check themselves against.
_BLOCK_TAGS = frozenset(
    {"p", "li", "dt", "dd", "blockquote", "pre", "figcaption", "caption"}
)
_HEADING_TAGS = ("h1", "h2", "h3", "h4")
_TABLE_TAGS = frozenset({"table"})

_WS = re.compile(r"\s+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

#: Longest slug kept from a heading. OPM headings run to full sentences; an anchor is meant
#: to be pasted into a citation, and the ordinal below keeps it unique after truncation.
_SLUG_MAX = 60


class PageUnavailable(LookupError):
    """The page returned 404. A fact sheet can be retired; that is not a build failure."""


# -- citations -------------------------------------------------------------------

#: "5 CFR 630.306(a)", "5 C.F.R. § 630.306", "5 CFR part 630". The paragraph designators are
#: captured, not discarded: they are what says *which* rule the fact sheet is interpreting.
_CFR_CITE = re.compile(
    r"\b(\d{1,2})\s*C\.?\s*F\.?\s*R\.?\s*(?:§{1,2}\s*)?(?:(?:part|pt)\.?\s*)?"
    r"(\d{1,4}(?:\.\d{1,4})?)((?:\([A-Za-z0-9]{1,4}\))*)",
    re.I,
)
#: "5 U.S.C. 6304(d)", "5 USC 6329a".
_USC_CITE = re.compile(
    r"\b(\d{1,2})\s*U\.?\s*S\.?\s*C\.?\s*(?:§{1,2}\s*)?"
    r"(\d{1,5}[a-z]?)((?:\([A-Za-z0-9]{1,4}\))*)",
    re.I,
)
#: A bare "§ 630.306". Common in guidance, and useless unqualified.
_BARE_SECTION = re.compile(r"§{1,2}\s*(\d{1,4}\.\d{1,4})((?:\([A-Za-z0-9]{1,4}\))*)")

#: Title assumed for a bare "§". Every part this corpus covers is Title 5, and an OPM leave
#: fact sheet writing "§ 630.306" means 5 CFR 630.306 by convention -- but the assumption is
#: named here rather than hidden, because it is the one place a wrong join can be minted.
DEFAULT_CFR_TITLE = "5"


def extract_citations(text: str, *, default_title: str = DEFAULT_CFR_TITLE) -> list[str]:
    """CFR and USC citations from prose, normalised and in document order.

    Normalised to ``"5 CFR 630.306(a)"`` and ``"5 U.S.C. 6304(d)"`` -- one spelling per
    citation, so that guidance and regulation can be joined on a string. The paragraph
    designator is kept; a consumer wanting the section-level key takes the prefix before
    ``"("``, whereas a consumer wanting the paragraph cannot recover it once dropped.

    Bare ``§`` references inherit the title from the nearest CFR citation before them,
    falling back to ``default_title``. Spans already claimed by a qualified citation are
    blanked before the bare pass so ``5 CFR § 630.306`` is not counted twice.
    """
    found: list[tuple[int, str]] = []
    titles: list[tuple[int, str]] = []
    masked = list(text)

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            masked[i] = " "

    for m in _CFR_CITE.finditer(text):
        title, ident, paras = m.group(1), m.group(2), m.group(3) or ""
        found.append((m.start(), f"{title} CFR {ident}{paras.replace(' ', '')}"))
        titles.append((m.start(), title))
        blank(*m.span())
    for m in _USC_CITE.finditer(text):
        title, sec, paras = m.group(1), m.group(2), m.group(3) or ""
        found.append((m.start(), f"{title} U.S.C. {sec}{paras.replace(' ', '')}"))
        blank(*m.span())
    for m in _BARE_SECTION.finditer("".join(masked)):
        title = default_title
        for pos, seen in titles:
            if pos < m.start():
                title = seen
        found.append((m.start(), f"{title} CFR {m.group(1)}{m.group(2) or ''}"))

    out: list[str] = []
    for _, cite in sorted(found, key=lambda pair: pair[0]):
        if cite not in out:
            out.append(cite)
    return out


# -- boilerplate -----------------------------------------------------------------


def _text_of(el: etree._Element) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip()


def _drop(el: etree._Element) -> None:
    """Remove an element, keeping its tail text.

    Same reasoning as ``apparatus.strip_apparatus``: a stripped element sitting mid-sentence
    carries the rest of the sentence in its ``.tail``, and dropping that silently deletes
    content. It matters more here than in XML -- HTML wraps inline chrome (a "share" icon, a
    screen-reader span) inside running prose all the time.
    """
    parent = el.getparent()
    if parent is None:
        return
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _protected(doc: etree._Element) -> set[int]:
    """Elements that must survive the class/id pass: main landmarks and their ancestors.

    Without this guard the attribute patterns are a live grenade. OPM's content column is
    ``<main class="usa-layout-docs__main">`` on some templates and sits inside a wrapper
    whose class contains ``search`` on others; one accidental match and the page yields zero
    units, which reads exactly like a page that legitimately has no content.
    """
    keep: set[int] = set()
    for xp in MAIN_XPATH:
        for el in doc.xpath(xp):
            node: etree._Element | None = el
            while node is not None:
                keep.add(id(node))
                node = node.getparent()
    return keep


def strip_boilerplate(doc: etree._Element) -> etree._Element:
    """Remove chrome from a parsed document, in place.

    Idempotent: a second pass finds nothing left to remove.
    """
    keep = _protected(doc)
    for el in list(doc.iter()):
        if el is doc or not isinstance(el.tag, str):
            continue  # comments and PIs have a callable .tag
        if el.tag in BOILERPLATE_TAGS:
            _drop(el)
            continue
        if id(el) in keep:
            continue
        attrs = " ".join(filter(None, (el.get("class"), el.get("id"), el.get("role"))))
        if attrs and BOILERPLATE_ATTR.search(attrs):
            _drop(el)
    return doc


def _density_region(doc: etree._Element) -> etree._Element | None:
    """The container with the most text per tag, among those holding most of the text.

    Text-per-tag is the discriminator because chrome is tag-dense and word-sparse: a mega
    menu is fifty ``<a>`` elements holding two words each, while a fact sheet is six ``<p>``
    elements holding sixty. The share floor stops the ratio from selecting the innermost
    paragraph, which would win on density while holding a twentieth of the page.
    """
    body = doc.find("body")
    root = body if body is not None else doc
    total = len(_text_of(root))
    if total < MIN_REGION_CHARS:
        return None
    best: tuple[float, etree._Element] | None = None
    for el in root.iter():
        if not isinstance(el.tag, str) or el.tag not in _CONTAINER_TAGS:
            continue
        size = len(_text_of(el))
        if size < total * DENSITY_MIN_SHARE or size < MIN_REGION_CHARS:
            continue
        density = size / (1 + sum(1 for _ in el.iter()))
        if best is None or density > best[0]:
            best = (density, el)
    return best[1] if best else None


def main_region(doc: etree._Element, *, doc_id: str = "") -> etree._Element | None:
    """The page's content region: a landmark if there is one, else density, else ``<body>``.

    The ``<body>`` fallback is logged rather than silent. A page whose content region cannot
    be identified still produces units, because a noisy citation is recoverable and a missing
    document is not -- but the build has to be able to say which pages took that path.
    """
    for xp in MAIN_XPATH:
        hits = doc.xpath(xp)
        if hits and len(_text_of(hits[0])) >= MIN_REGION_CHARS:
            return hits[0]
    dense = _density_region(doc)
    if dense is not None:
        log.debug("%s: no content landmark; density picked <%s>", doc_id, dense.tag)
        return dense
    body = doc.find("body")
    if body is not None and len(_text_of(body)) >= MIN_REGION_CHARS:
        log.warning("%s: no content landmark and no dense region; using <body>", doc_id)
        return body
    return None


# -- structure -------------------------------------------------------------------


def slugify(text: str) -> str:
    slug = _SLUG_STRIP.sub("-", text.lower()).strip("-")
    return slug[:_SLUG_MAX].rstrip("-") or "section"


def _blocks(region: etree._Element):
    """Headings, prose blocks and tables in document order, without descending into them.

    Pruning at each block is what stops a ``<p>`` inside an ``<li>`` from being emitted twice
    -- once as itself and once inside its parent -- which is the HTML analogue of the
    double-counting ``corpus/parse.py`` walks around for ``EXTRACT``.
    """
    stack = [region] if region.tag in _BLOCK_TAGS or region.tag in _TABLE_TAGS else \
        list(reversed(list(region)))
    while stack:
        el = stack.pop()
        if not isinstance(el.tag, str):
            continue
        if el.tag in _HEADING_TAGS or el.tag in _BLOCK_TAGS or el.tag in _TABLE_TAGS:
            yield el
            continue
        stack.extend(reversed(list(el)))


def _table_text(node: etree._Element) -> str:
    """One line per row, cells joined by ' | ' -- the serialisation ``corpus/parse.py`` uses.

    Identical on purpose: a retrieval that has learned what a CFR wage table looks like
    should not have to learn a second shape for the fact sheet that explains it. Splitting a
    table per cell would destroy the row relationship that makes it answerable at all.
    """
    rows: list[str] = []
    for tr in node.iter("tr"):
        cells = [_WS.sub(" ", "".join(td.itertext())).strip() for td in tr.iter("td", "th")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        single = _text_of(node)
        return single
    return "\n".join(rows)


def _anchor_for(slug: str, used: dict[str, int]) -> str:
    """``slug`` for the first use, ``slug-2`` for the second.

    The ordinal is per-slug, not a running counter over the page, and that is the whole
    point. With a running counter, inserting one paragraph high on the page renumbers every
    anchor below it, so every stored citation silently starts pointing somewhere else on the
    next fetch. Per-slug ordinals only move when a *duplicate heading* is added above them.

    Duplicates are not hypothetical: the live annual-leave fact sheet carries two ``h3``
    elements both reading "Non-Federal Service or Uniformed Service".
    """
    used[slug] = used.get(slug, 0) + 1
    return slug if used[slug] == 1 else f"{slug}-{used[slug]}"


@dataclass
class _Section:
    heading: str
    anchor: str
    locator: str
    lines: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


def extract_units(html: bytes, *, doc_id: str, base_url: str = "") -> list[Unit]:
    """Parse a page into citable units. Pure, offline, and safe on malformed input.

    Structure follows the headings, because a fact sheet is a set of topics and a citation
    that says "somewhere on the annual leave page" is not a citation. Each ``h1``-``h4``
    opens a unit carrying the prose beneath it; tables become their own ``KIND_TABLE`` unit
    so a wage schedule is retrieved whole rather than as a sentence fragment.

    Never raises on bad input. lxml's HTML parser is lenient by design, so the failure mode
    is not an exception but a salvaged ``<p>`` full of junk -- which is why an under-sized
    content region yields zero units rather than a citable address for garbage.
    """
    try:
        doc = lxml_html.document_fromstring(html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError) as exc:
        log.warning("%s: unparseable HTML (%s); no units", doc_id, exc)
        return []
    if doc is None:
        return []

    strip_boilerplate(doc)
    region = main_region(doc, doc_id=doc_id)
    if region is None:
        log.warning("%s: under %d characters of content after boilerplate removal; no units",
                    doc_id, MIN_REGION_CHARS)
        return []

    used: dict[str, int] = {}
    lead = _Section(heading="", anchor=_anchor_for("intro", used), locator=base_url)
    sections: list[_Section] = [lead]
    trail: list[str] = []

    for el in _blocks(region):
        text = _text_of(el)
        if not text:
            continue
        if el.tag in _HEADING_TAGS:
            level = int(el.tag[1])
            del trail[level - 1:]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(text)
            frag = el.get("id") or ""
            locator = (urljoin(base_url, "#" + frag) if frag and base_url
                       else " > ".join(t for t in trail if t))
            sections.append(_Section(heading=text, anchor=_anchor_for(slugify(text), used),
                                     locator=locator))
        elif el.tag in _TABLE_TAGS:
            rows = _table_text(el)
            if rows:
                sections[-1].tables.append(rows)
        else:
            sections[-1].lines.append(text)

    units: list[Unit] = []
    for sec in sections:
        if sec.lines:
            units.append(Unit(anchor=sec.anchor, text="\n".join(sec.lines),
                              heading=sec.heading, kind=KIND_PROSE, locator=sec.locator))
        for i, rows in enumerate(sec.tables, start=1):
            units.append(Unit(anchor=f"{sec.anchor}-t{i}", text=rows, heading=sec.heading,
                              kind=KIND_TABLE, locator=sec.locator))
    # A backstop only. Per-slug ordinals already make these unique; this catches the case
    # where truncation to _SLUG_MAX collapses two long headings onto the same slug.
    return merge_anchors(units)


def page_title(html: bytes, *, doc_id: str = "") -> str:
    """The page's own title: its ``h1`` if it has one, else ``<title>``.

    ``h1`` first because ``<title>`` on a federal site is the topic plus the agency plus the
    site section, and the trailing two thirds of that is the same on every page in the
    corpus -- which is the definition of a field that carries no information.
    """
    try:
        doc = lxml_html.document_fromstring(html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return ""
    strip_boilerplate(doc)
    for xp in ("//h1", "//title"):
        hits = doc.xpath(xp)
        if hits:
            text = _text_of(hits[0])
            if text:
                return text
    return doc_id


def doc_id_for(url: str) -> str:
    """A short, stable document id from a URL path.

    The last two path segments, not the whole path: OPM's paths are five segments of site
    taxonomy, and a doc_id appears in citations, where
    ``fact-sheets-annual-leave`` is readable and
    ``policy-data-oversight-pay-leave-leave-administration-fact-sheets-annual-leave`` is not.
    """
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts:
        return slugify(urlparse(url).netloc)
    return slugify("-".join(parts[-2:]))


# -- fetching --------------------------------------------------------------------


@dataclass(frozen=True)
class FetchedPage:
    url: str
    content: bytes
    sha256: str
    fetched_at: str
    from_cache: bool


@dataclass
class HtmlFetcher:
    """Cached, rate-limited, retrying GET for pages that change without notice.

    The eCFR client pins ``/full/`` snapshots forever because they are immutable history.
    Nothing here is. A fact sheet is edited in place with no version identifier and no
    changelog, so the cache carries a TTL and every entry records a SHA-256 of the bytes --
    which is the only mechanism by which "this guidance changed" is detectable at all. A
    refresh that finds a different hash says so at INFO, and that log line is the closest
    thing to a publication notice this layer gets.
    """

    cache_dir: Path
    ttl_hours: float = PAGE_TTL_HOURS
    delay_s: float = 1.0
    timeout_s: float = 60.0
    refresh: bool = False
    _last_request: float = 0.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, url: str) -> tuple[Path, Path]:
        # The digest is in the name because two fact sheets can share their last two path
        # segments; the readable prefix is there so a human can find a file in the cache.
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        stem = f"{doc_id_for(url)}-{digest}"
        return self.cache_dir / f"{stem}.html", self.cache_dir / f"{stem}.json"

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str) -> bytes:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)  # a public site; do not hammer it
        r = httpx.get(url, headers={"User-Agent": USER_AGENT},
                      timeout=self.timeout_s, follow_redirects=True)
        self._last_request = time.monotonic()
        if r.status_code == 404:
            raise PageUnavailable(url)  # outside the retry set: a retired page stays retired
        r.raise_for_status()
        return r.content

    def fetch(self, url: str) -> FetchedPage:
        """Page bytes, from cache when fresh. Writes are atomic.

        A non-atomic write leaves a truncated file if the process dies mid-write, and the
        cache only checks existence, so the truncation is trusted forever -- the same trap
        ``ECFRClient._cached`` documents. A failed refresh falls back to the stale copy: an
        expired TTL is a reason to ask again, not a reason for an offline build to stop.
        """
        page, meta = self._paths(url)
        cached = page.read_bytes() if page.exists() else None
        stale = cached is None or self.refresh or (
            time.time() - page.stat().st_mtime > self.ttl_hours * 3600.0
        )
        if cached is not None and not stale:
            return FetchedPage(url, cached, _sha(cached), _stamp_of(meta), from_cache=True)

        log.info("GET %s", url)
        try:
            content = self._get(url)
        except PageUnavailable:
            raise
        except httpx.HTTPError as exc:
            if cached is not None:
                log.warning("refresh of %s failed (%s); using the cached copy", url, exc)
                return FetchedPage(url, cached, _sha(cached), _stamp_of(meta), from_cache=True)
            raise

        digest, now = _sha(content), _now_stamp()
        if cached is not None and _sha(cached) != digest:
            log.info("%s changed since %s (%s -> %s)", url, _stamp_of(meta),
                     _sha(cached)[:12], digest[:12])
        tmp = page.with_suffix(".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, page)
        meta.write_text(json.dumps({"url": url, "fetched": now, "sha256": digest,
                                    "bytes": len(content)}), encoding="utf-8")
        return FetchedPage(url, content, digest, now, from_cache=False)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _now_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stamp_of(meta: Path) -> str:
    try:
        return str(json.loads(meta.read_text(encoding="utf-8")).get("fetched", ""))
    except (OSError, ValueError):
        return ""


@dataclass
class HtmlGuidanceSource:
    """OPM fact sheets as ``SourceDoc``s at ``AUTHORITY_GUIDANCE``.

    ``valid_from`` is the fetch date, and that is the honest answer rather than a weak one.
    The page carries no publication or revision date anywhere in its markup, so the only
    thing this source actually knows is that the text was live when it was read; inventing an
    earlier date would assert applicability the source cannot support, and this system sorts
    on authority precisely so that a guidance claim never quietly outranks the rule.
    """

    cache_dir: Path
    urls: tuple[str, ...] = OPM_FACT_SHEETS
    name: str = "opm"
    authority: int = AUTHORITY_GUIDANCE
    ttl_hours: float = PAGE_TTL_HOURS
    refresh: bool = False

    def __post_init__(self) -> None:
        self.fetcher = HtmlFetcher(Path(self.cache_dir), ttl_hours=self.ttl_hours,
                                   refresh=self.refresh)

    def documents(self) -> Iterator[SourceDoc]:
        """Every configured page, parsed. One bad page never stops the rest.

        A 404 is a retired fact sheet, and a page that yields nothing is a template change;
        both are skipped with a log line rather than raised, because a guidance ingest that
        aborts on its third of twenty URLs leaves the store in a state no one can interpret.
        """
        seen: dict[str, int] = {}
        for url in self.urls:
            try:
                page = self.fetcher.fetch(url)
            except PageUnavailable:
                log.warning("skipping %s: 404, the fact sheet appears to be retired", url)
                continue
            except httpx.HTTPError as exc:
                log.warning("skipping %s: %s", url, exc)
                continue

            doc_id = doc_id_for(url)
            seen[doc_id] = seen.get(doc_id, 0) + 1
            if seen[doc_id] > 1:
                doc_id = f"{doc_id}-{seen[doc_id]}"
                log.warning("%s: duplicate doc_id from URL path; using %s", url, doc_id)

            units = extract_units(page.content, doc_id=doc_id, base_url=url)
            if not units:
                log.warning("skipping %s: no content survived boilerplate removal", url)
                continue
            yield SourceDoc(
                source=self.name,
                doc_id=doc_id,
                title=page_title(page.content, doc_id=doc_id),
                authority=self.authority,
                units=units,
                valid_from=page.fetched_at[:10],
                references=extract_citations("\n".join(u.text for u in units)),
                url=url,
                meta={"sha256": page.sha256, "fetched": page.fetched_at,
                      "bytes": str(len(page.content))},
            )
