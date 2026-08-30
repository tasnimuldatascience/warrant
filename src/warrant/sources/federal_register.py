"""Federal Register notices: the reasoning behind an amendment.

eCFR answers *what the rule is* and *when it changed*. It cannot answer *why*. The corpus
already knows that 5 CFR 630.306 has a version dated 2020-08-10; what it does not know is
that the amendment was OPM's interim rule "Scheduling of Annual Leave by Employees Determined
Necessary To Respond to Certain National Emergencies" (FR document 2020-16823, 85 FR 48096),
that OPM waived notice-and-comment because COVID-19 leave forfeitures were imminent, and that
the same document rescinded the September 11 analogue at 5 CFR 630.311. That reasoning is in
the Federal Register and nowhere else in the corpus.

``cfr_references`` is the join key. Every rule and proposed rule carries the list of CFR parts
it affects, in the same (title, part) vocabulary the eCFR ingester already uses, so a notice
can be attached to the sections it amends without text matching. Normalised here to
``"5 CFR 630"`` / ``"5 CFR 630.306"`` strings; resolving those to store ids is a later step.

Three things about this API were found by running it rather than by reading the docs:

1. ``raw_text_url`` is named ``.txt`` and serves **HTML** -- a ``<pre>`` block with anchor
   tags and escaped entities wrapped in ``<html><body>``. Feeding it to a splitter unchanged
   gives units full of ``<a href=...>`` and ``&#160;``. ``clean_raw_text`` undoes that.

2. The ``cfr_references`` schema has a ``section`` key, but OPM's notices do not populate it:
   over 258 references sampled from a term search, every one gave title and part only. The
   section-level form is still normalised, because a reference that arrives one day and is
   silently dropped is worse than one that never arrives.

3. The result window is capped near 10,000 documents regardless of ``per_page``: a term
   search reporting ``count`` 4,637 served ``total_pages`` 50 at ``per_page`` 3. Paging past
   the window returns nothing useful, so pagination stops there as well as at the caller's
   own document cap.

Caching policy is deliberately **not** uniform, and this is the one thing the eCFR client got
wrong badly enough to be worth repeating here: it pinned its index endpoints forever, so the
next build re-read the pinned files, found no new dates, and froze the corpus with no error
anywhere. A search is exactly where tomorrow's notice first appears, so searches carry a TTL.
A document fetched by ``document_number`` is the opposite: 2020-16823 is a printed historical
record and its JSON and text will never change, so those are cached forever.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlencode

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import AUTHORITY_NOTICE, KIND_HEADING, KIND_PROSE, SourceDoc, Unit, merge_anchors

log = logging.getLogger(__name__)

API = "https://www.federalregister.gov/api/v1"
USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

#: Requested on both endpoints. ``fields[]`` is not optional in practice -- omitting it
#: returns a much larger record per document, and the search is paged 100 at a time.
FIELDS = (
    "document_number", "title", "publication_date", "html_url", "raw_text_url",
    "abstract", "cfr_references", "agencies", "type", "action",
)

#: How long a cached *search* stays authoritative. Searches are not immutable: new notices
#: publish every business day, and a stale search reports no new notices, which is
#: indistinguishable from there being none.
SEARCH_TTL_HOURS = 24.0

#: The API's documented ceiling on ``per_page``.
PER_PAGE_MAX = 1000

#: The API stops serving results past roughly this offset whatever ``per_page`` is (see the
#: module docstring). Paging beyond it burns requests for empty pages.
RESULT_WINDOW = 10_000

#: Default ceiling on documents produced per run. A filter that loses its CFR condition
#: matches the whole Federal Register -- millions of documents -- and the failure mode is a
#: multi-day ingest, not an error. The cap makes a misconfigured filter finish and complain.
MAX_DOCUMENTS = 500


class NoticeUnavailable(LookupError):
    """The API has no document under this number, or no text for it."""


class RateLimited(RuntimeError):
    """The API asked us to slow down. Retried with backoff, never cached."""


class MalformedNotice(ValueError):
    """A response for a known document number did not parse. Names the document."""


# -- text ------------------------------------------------------------------------

_PRE = re.compile(r"(?is)<pre>(.*)</pre>")
_SCRIPT = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAG = re.compile(r"(?s)<[^>]+>")
#: GPO page break inside the running text, e.g. ``[[Page 48097]]``. Dropped from the prose
#: but kept as a unit locator: a disputed quote is checked against a printed page number.
_PAGE = re.compile(r"^\[\[Page (\d+[A-Za-z]?)\]\]$")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
#: Page furniture: the running head GPO repeats on every printed page, and the rule lines
#: around the masthead banner. Dropped whole -- the running head wraps onto a second line, so
#: removing only the matching line leaves "and Regulations" behind as a spurious heading.
_FURNITURE = re.compile(r"^(?:Federal Register\s*/\s*Vol\.|[-=_]{5,})")
_FR_VOLUME = re.compile(r"\[Federal Register Volume (\d+)", re.I)
_FR_PAGES = re.compile(r"\[Pages? (\d+)", re.I)


def clean_raw_text(payload: str) -> str:
    """Recover plain text from what ``raw_text_url`` actually serves.

    The URL ends in ``.txt`` and returns an HTML document whose ``<pre>`` holds the notice.
    Links are real anchors and the body is entity-escaped, so both have to be undone before
    anything looks at line structure -- the splitter keys on indentation and blank lines, and
    a stray ``</a>`` is enough to turn a heading into body text.
    """
    body = payload
    match = _PRE.search(payload)
    if match:
        body = match.group(1)
    body = _SCRIPT.sub("", body)
    body = _TAG.sub("", body)
    body = html.unescape(body)
    body = body.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # NUL bytes prefix every line of the GPO masthead banner in the newer rendering -- 16 of
    # them in 85 FR 48075 -- and they survive into unit text and into the store as literal
    # \x00 unless they are removed here.
    return _CONTROL.sub("", body)


#: An all-caps preamble label: ``SUMMARY:``, ``DATES:``, ``FOR FURTHER INFORMATION CONTACT:``.
#: Every notice carries these and they are the highest-value units in the document.
_LABEL = re.compile(r"^([A-Z][A-Z0-9 ,'/&().-]{2,58}[A-Z)]):\s+(.*)$")
#: An amendatory section heading. ``Sec.`` in the text stream, ``§`` in some conversions.
_SECTION = re.compile(r"^(?:Sec\.|§)\s*(?:Sec\.\s*)?([0-9]+\.[0-9]+[a-z]?)")
#: ``PART 630--ABSENCE AND LEAVE``, ``Subpart C--Annual Leave``, ``Appendix A to Part 630``.
_STRUCTURE = re.compile(r"^(PART|Part|Subpart|SUBPART|Appendix|APPENDIX)\s+\S")
#: A numbered or lettered heading: the amendment instructions (``1. Amend Sec. 630.306 by
#: revising...``) and the roman-numbered discussion headings some agencies use.
_ENUMERATED = re.compile(r"^(\d{1,3}|[IVXLC]{1,6}|[A-Z])\.\s+\S")
#: The GPO amendment bullet, which arrives as a bare ``0`` on its own line.
_BULLET = re.compile(r"^0$")

#: A line that opens something of its own: a lettered sub-instruction, a numbered clause, a
#: bare section number in a subpart's table of contents, or the ``Sec.`` that heads one.
_NEW_ITEM = re.compile(r"^(?:[A-Za-z]|\d{1,3}|[IVXLC]{1,6})\.\s+\S|^\d+\.\d+[a-z]?\s|^Sec\.\s*$")

#: Headings longer than this are prose, whatever they start with. An amendment instruction
#: runs to about 70 characters; a wrapped paragraph that starts at column 0 runs longer.
_MAX_HEADING_CHARS = 120
_MAX_ENUMERATED_CHARS = 250
#: A wrapped heading gets this much room before the wrap is assumed not to be one. Generous
#: because ``_starts_an_item`` does the real work; this only bounds the damage when it cannot.
_MAX_CONTINUATION_CHARS = 300


def _slug(text: str, *, max_words: int = 8, max_chars: int = 60) -> str:
    """A citable anchor fragment. Dots survive because ``630.306`` is the identifying part."""
    words = re.findall(r"[a-z0-9.]+", text.lower())[:max_words]
    return "-".join(words)[:max_chars].strip("-.") or "block"


@dataclass
class _Block:
    """One blank-line-separated run of lines, with where it started."""

    lines: list[str]
    line_no: int
    page: str = ""

    @property
    def flat(self) -> str:
        return " ".join(line.strip() for line in self.lines).strip()

    @property
    def indented(self) -> bool:
        return self.lines[0][:1].isspace()


def _blocks(text: str) -> list[_Block]:
    """Split into blank-line-separated blocks, dropping page markers but remembering them."""
    out: list[_Block] = []
    current: list[str] = []
    start = 0
    page = ""
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        marker = _PAGE.match(stripped)
        if marker:
            page = marker.group(1)
            continue
        if _BULLET.match(stripped):
            # GPO prints the amendatory bullet as a bare "0" line directly above the
            # instruction it belongs to. Left in, it glues the instruction to the preceding
            # block and every amendment instruction in the notice stops being its own unit.
            continue
        if stripped:
            if not current:
                start = i + 1
            current.append(line)
            continue
        if current:
            out.append(_Block(current, start, page))
            current = []
    if current:
        out.append(_Block(current, start, page))
    return [b for b in out if not _FURNITURE.match(b.flat)]


def _starts_an_item(line: str) -> bool:
    """Does this line begin something of its own rather than continue the line above?

    ``2. Amend Sec. 630.1201 as follows:`` is followed immediately, with no blank line, by
    ``a. Revise the section heading;`` and five more lettered clauses; ``7. Add subpart Q to
    read as follows:`` is followed immediately by the subpart's table of contents. Treating
    those as wrapped continuations produced a heading 400 characters long; treating the
    instruction as ending at its own line break truncated it mid-sentence. The line that
    follows has to be classified, not counted.
    """
    return bool(_NEW_ITEM.match(line) or _SECTION.match(line) or _STRUCTURE.match(line)
                or _LABEL.match(line))


def _lead(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a block into its opening line plus wrapped continuations, and the remainder."""
    head = [lines[0]]
    i = 1
    while i < len(lines):
        line = lines[i]
        if line[:1].isspace() or _starts_an_item(line.strip()):
            break
        if sum(len(h) for h in head) > _MAX_CONTINUATION_CHARS:
            break
        head.append(line)
        i += 1
    return head, lines[i:]


def _heading(block: _Block) -> tuple[str, list[str], list[str]] | None:
    """``(heading, body lines, lines to reconsider)`` if this block opens a unit, else None.

    Body text in a Federal Register notice is indented -- paragraphs open with four spaces and
    wrap to column 0 -- so an unindented line opening a block is the structural signal. Every
    rule below is anchored on that, which is why a long wrapped paragraph cannot be mistaken
    for a heading no matter what word it starts with.
    """
    if block.indented:
        return None
    first = block.lines[0].strip()
    if _BULLET.match(first):
        return None
    label = _LABEL.match(block.lines[0])
    if label:
        # Only the label is consumed; the rest of the block keeps its own line structure,
        # because ADDRESSES runs straight into an indented list of submission methods with no
        # blank line between them and flattening the block would fuse them into one sentence.
        return label.group(1), [label.group(2), *block.lines[1:]], []
    head, rest = _lead(block.lines)
    heading = " ".join(line.strip() for line in head).strip()
    plain = (len(heading) <= _MAX_HEADING_CHARS
             # Terminal punctuation rules out the signature block ("Alexys Stanley," /
             # "Regulatory Affairs Analyst.") and one-line body paragraphs; a leading bracket
             # rules out the masthead and the "[FR Doc. ... Filed ...]" trailer, which are
             # facts about the issue rather than headings in it.
             and not heading.endswith((".", ",", ";", ":"))
             and not heading.startswith("[") and any(c.isalpha() for c in heading))
    if (_SECTION.match(first)
            or (_STRUCTURE.match(first) and len(heading) <= _MAX_HEADING_CHARS)
            or (_ENUMERATED.match(first) and len(heading) <= _MAX_ENUMERATED_CHARS)
            or plain):
        return heading, [], rest
    return None


def _anchor_for(heading: str) -> str:
    """A stable, readable anchor.

    Sections and amendatory instructions get their identifying number rather than a slug of
    their prose, because those are the two things a reader cites by number: "§ 630.306" and
    "instruction 3 of the interim rule". Slugging them would produce
    ``3.-amend-sec.-630.306-by-revising``, which is neither.
    """
    section = _SECTION.match(heading)
    if section:
        return f"sec-{section.group(1)}"
    numbered = _ENUMERATED.match(heading)
    if numbered and numbered.group(1).isdigit():
        amended = re.search(r"\b(\d+\.\d+[a-z]?)\b", heading)
        return f"amdt-{numbered.group(1)}-{amended.group(1)}" if amended \
            else f"amdt-{numbered.group(1)}"
    return _slug(heading)


def _unwrap(lines: Sequence[str]) -> str:
    """Rejoin hard-wrapped lines into paragraphs.

    The text is wrapped at about 72 columns with a four-space paragraph indent, so an indented
    line opens a paragraph and everything after it at column 0 continues it. Joining the whole
    block with spaces would fuse the numbered paragraphs of an amended section into one wall
    of text and destroy the ``(a)(1)`` structure a citation depends on.
    """
    paragraphs: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not paragraphs or line[:1].isspace():
            paragraphs.append([stripped])
        else:
            paragraphs[-1].append(stripped)
    return "\n".join(" ".join(p) for p in paragraphs).strip()


def split_units(text: str, *, first_anchor: str = "front-matter") -> list[Unit]:
    """Split a notice on its real structure.

    A fixed token window would cut across ``SUMMARY``/``DATES`` and across the amendatory
    instructions, which are the two places a reader actually needs cited exactly: "the rule is
    effective August 10, 2020" is a sentence in DATES, and "revise paragraph (a)" is a
    sentence in an instruction. Federal Register notices publish with those boundaries already
    marked, so the splitter follows them.

    Anchors are made unique here rather than by ``merge_anchors``: a notice that amends
    § 630.210 twice (once ``[Amended]``, once with the replacement text) produces the same
    heading twice, and that is normal rather than a parser defect. ``merge_anchors`` still
    runs as the backstop it is meant to be.
    """
    units: list[Unit] = []
    seen: dict[str, int] = {}
    heading = ""
    anchor = first_anchor
    kind = KIND_PROSE
    body: list[str] = []
    locator = ""

    def flush() -> None:
        nonlocal anchor
        text_out = _unwrap(body)
        if not text_out and not heading:
            return
        key = anchor
        if key in seen:
            seen[key] += 1
            key = f"{key}-{seen[key]}"  # § 630.210 [Amended] and § 630.210 both appear
        else:
            seen[key] = 1
        units.append(Unit(anchor=key, text=text_out or heading, heading=heading,
                          kind=kind if text_out else KIND_HEADING, locator=locator))

    queue = deque(_blocks(text))
    while queue:
        block = queue.popleft()
        found = _heading(block)
        if found is None:
            if not body and not locator:
                locator = _locator(block)
            body.extend(block.lines)
            continue
        flush()
        heading, rest, reconsider = found
        anchor = _anchor_for(heading)
        kind = KIND_PROSE
        body = [line for line in rest if line.strip()]
        locator = _locator(block)
        if reconsider:
            # What followed the heading inside the same block starts something of its own --
            # the lettered clauses under an instruction, the table of contents under a new
            # subpart. Put it back so it is classified rather than swallowed.
            queue.appendleft(_Block(reconsider,
                                    block.line_no + len(block.lines) - len(reconsider),
                                    block.page))
    flush()
    return merge_anchors(units)


def _locator(block: _Block) -> str:
    """Where in the printed record this unit starts. Page when GPO gave one, else line."""
    return f"p.{block.page} line {block.line_no}" if block.page else f"line {block.line_no}"


def normalize_references(refs: Sequence[dict]) -> list[str]:
    """``cfr_references`` to ``"{title} CFR {part}.{section}"`` (or part-only) strings.

    The part-only form is the one that actually arrives (see the module docstring), and it is
    the right granularity anyway: a notice amends a part and names its sections in the
    amendatory instructions, so the coarse reference joins and the instructions refine it.

    A reference with no part is dropped. ``"5 CFR"`` joins to every section of title 5, which
    is not a join, it is a broadcast.
    """
    out: list[str] = []
    for ref in refs or ():
        title = ref.get("title")
        part = str(ref.get("part") or "").strip()
        if title is None or not part:
            log.debug("cfr_reference without a usable part: %r", ref)
            continue
        section = str(ref.get("section") or "").strip()
        cite = f"{title} CFR {part}.{section}" if section else f"{title} CFR {part}"
        if cite not in out:
            out.append(cite)
    return out


# -- source ----------------------------------------------------------------------


@dataclass
class FederalRegisterSource:
    """Notices affecting a set of CFR parts, as :class:`SourceDoc` objects.

    ``fetch`` replaces the transport wholesale. It exists so the tests can run offline against
    canned payloads without monkeypatching ``httpx``: the caching, pagination and error paths
    are the parts of this class most likely to be wrong, and they are only testable if the
    thing they wrap is substitutable.
    """

    cache_dir: Path
    #: The join to the eCFR corpus: title and parts, in the same vocabulary that ingester uses.
    cfr_title: int = 5
    cfr_parts: Sequence[str] = ("630",)
    #: Publication-date floor. The eCFR corpus starts in 2017, so notices before then explain
    #: amendments no snapshot covers; the default is looser because a notice can post-date the
    #: amendment history it explains by decades and still be the reasoning for it.
    published_since: str = "2000-01-01"
    #: Optional free-text condition, ANDed with the CFR filter by the API.
    term: str = ""
    delay_s: float = 1.0
    timeout_s: float = 60.0
    #: Backoff after a transport error or a 429: 2 s, 4 s, 8 s over four attempts.
    backoff_s: float = 2.0
    max_attempts: int = 4
    #: Applies to searches only. Documents are immutable and never expire -- see the module
    #: docstring for what happened when the eCFR client applied one policy to both.
    search_ttl_hours: float = SEARCH_TTL_HOURS
    per_page: int = 100
    max_documents: int = MAX_DOCUMENTS
    refresh: bool = False
    fetch: Callable[[str], bytes] | None = None
    #: Document numbers this run could not turn into a SourceDoc, and why. A skip that only
    #: appears in a log line is a skip nobody counts.
    skipped: dict[str, str] = field(default_factory=dict, repr=False)
    _last_request: float = 0.0

    name: ClassVar[str] = "federal_register"
    authority: ClassVar[int] = AUTHORITY_NOTICE

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.per_page > PER_PAGE_MAX:
            log.warning("per_page %d exceeds the API maximum; using %d",
                        self.per_page, PER_PAGE_MAX)
            self.per_page = PER_PAGE_MAX

    # -- urls --------------------------------------------------------------------

    def search_url(self, part: str, page: int) -> str:
        params: list[tuple[str, object]] = [
            ("per_page", self.per_page),
            ("page", page),
            ("conditions[cfr][title]", self.cfr_title),
            ("conditions[cfr][part]", part),
            ("conditions[publication_date][gte]", self.published_since),
        ]
        if self.term:
            params.append(("conditions[term]", self.term))
        params += [("fields[]", f) for f in FIELDS]
        return f"{API}/documents.json?{urlencode(params)}"

    def document_url(self, number: str) -> str:
        query = urlencode([("fields[]", f) for f in FIELDS])
        return f"{API}/documents/{number}.json?{query}"

    # -- transport ---------------------------------------------------------------

    def _fetch_once(self, url: str) -> bytes:
        """One request, no retries.

        The politeness gap is applied to the injected fetcher too. A stub does not need it,
        but a caller who injects a real client to add auth or proxying does, and making the
        delay conditional on which transport is in use is how it goes missing in production.
        """
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)  # a free, unauthenticated API; do not hammer it
        try:
            if self.fetch is not None:
                return self.fetch(url)
            r = httpx.get(url, headers={"User-Agent": USER_AGENT},
                          timeout=self.timeout_s, follow_redirects=True)
            if r.status_code == 404:
                raise NoticeUnavailable(url)
            if r.status_code == 429:
                raise RateLimited(url)
            r.raise_for_status()
            return r.content
        finally:
            self._last_request = time.monotonic()

    def _fetch(self, url: str) -> bytes:
        """``_fetch_once`` with backoff.

        ``NoticeUnavailable`` is deliberately *outside* the retry set, unlike the eCFR client
        where a 404 means "not published yet". Here a 404 means the document number does not
        exist, and retrying it four times with exponential backoff costs 14 s to learn that.
        ``RateLimited`` is inside it: 429 is precisely the case where waiting works, and the
        politeness gap between requests is what should have made it unnecessary.

        Built per call from instance fields rather than declared as a ``@retry`` decorator,
        because a decorator fixes the backoff at import time and the retry path is then only
        testable by waiting 14 real seconds for it.
        """
        retrying = Retrying(
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.HTTPStatusError, RateLimited)),
            wait=wait_exponential(multiplier=self.backoff_s, min=self.backoff_s, max=60),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        return retrying(self._fetch_once, url)

    # -- cache -------------------------------------------------------------------

    def _cached(self, url: str, name: str, *, ttl_s: float | None) -> bytes:
        """Fetch through the on-disk cache. ``ttl_s is None`` means immutable.

        Writes are atomic. A plain ``write_bytes`` interrupted mid-write -- Ctrl-C during a
        long ingest, a closed lid -- leaves a truncated file that the existence check trusts
        forever, and the symptom is a JSON decode error weeks later naming no document.

        A failed refresh of an expired entry falls back to the stale copy. An expired TTL is a
        reason to ask again, not a reason for an offline rebuild to stop working.
        """
        path = self.cache_dir / name
        cached = path.read_bytes() if path.exists() else None
        fresh = cached is not None and not self.refresh and (
            ttl_s is None or (time.time() - path.stat().st_mtime) <= ttl_s)
        if fresh:
            log.debug("cache hit %s", name)
            return cached
        log.info("GET %s", url)
        try:
            content = self._fetch(url)
        except (httpx.HTTPError, RateLimited) as exc:
            if cached is not None:
                log.warning("refresh of %s failed (%s); using the cached copy", name, exc)
                return cached
            raise
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        log.info("cached %s (%d bytes)", name, len(content))
        return content

    @staticmethod
    def _key(url: str) -> str:
        """Filenames have to survive ``conditions[term]=annual leave`` on Windows."""
        return hashlib.sha1(url.encode()).hexdigest()[:10]

    def _json(self, raw: bytes, what: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MalformedNotice(f"{what}: response is not JSON ({exc})") from exc

    # -- search ------------------------------------------------------------------

    def search_page(self, part: str, page: int) -> dict:
        """One page of results. Cached with a TTL: this is where a new notice first appears."""
        url = self.search_url(part, page)
        name = f"search-t{self.cfr_title}-p{part}-{self._key(url)}-page{page}.json"
        return self._json(self._cached(url, name, ttl_s=self.search_ttl_hours * 3600.0),
                          f"search t{self.cfr_title} part {part} page {page}")

    def _rows(self, part: str) -> Iterator[dict]:
        """Every result row for one part, oldest page first, stopping at every known ceiling.

        Three of them, and none is optional. ``total_pages`` ends a normal search; the result
        window ends a search too broad for the API to serve past ~10,000 documents; an empty
        page ends one where the API disagrees with its own ``total_pages``, which is cheaper
        to tolerate than to diagnose in the middle of an ingest.
        """
        page = 1
        while True:
            payload = self.search_page(part, page)
            rows = payload.get("results") or []
            if not rows:
                return
            yield from rows
            total_pages = payload.get("total_pages") or 0
            if page >= total_pages:
                return
            if page * self.per_page >= RESULT_WINDOW:
                log.warning("part %s: stopping at the API result window (%d of %s documents); "
                            "narrow the date floor or the term",
                            part, page * self.per_page, payload.get("count"))
                return
            page += 1

    # -- documents ---------------------------------------------------------------

    def record(self, number: str) -> dict:
        """The document's own JSON. Cached forever: a published notice is a printed fact."""
        url = self.document_url(number)
        return self._json(self._cached(url, f"doc-{number}.json", ttl_s=None), number)

    def raw_text(self, number: str, url: str) -> str:
        """The full notice body. Cached forever for the same reason as ``record``."""
        return clean_raw_text(
            self._cached(url, f"text-{number}.txt", ttl_s=None).decode("utf-8", "replace"))

    def to_doc(self, record: dict) -> SourceDoc:
        """Build a :class:`SourceDoc` from a document record, fetching its text."""
        number = str(record.get("document_number") or "").strip()
        if not number:
            raise MalformedNotice("document record has no document_number")
        date = record.get("publication_date")
        if not date:
            raise MalformedNotice(f"{number}: no publication_date")

        text_url = record.get("raw_text_url")
        units: list[Unit] = []
        if text_url:
            units = split_units(self.raw_text(number, text_url))
        if not units:
            # Some documents -- corrections, some notices -- have no full text served. The
            # abstract is a real, citable summary written by the agency, so the notice still
            # enters the corpus rather than vanishing between two log lines.
            abstract = (record.get("abstract") or "").strip()
            if not abstract:
                raise MalformedNotice(f"{number}: neither raw text nor an abstract")
            log.info("%s: no raw text served; indexing the abstract alone", number)
            units = [Unit(anchor="abstract", text=abstract, heading="Abstract")]

        agencies = ", ".join(
            a.get("raw_name") or a.get("name") or "" for a in (record.get("agencies") or []))
        meta = {k: v for k, v in {
            "document_number": number,
            "type": record.get("type") or "",
            "action": (record.get("action") or "").strip(),
            "agencies": agencies,
            "fr_citation": _fr_citation(units),
        }.items() if v}

        return SourceDoc(
            source=self.name,
            doc_id=f"FR-{number}",
            title=record.get("title") or number,
            authority=self.authority,
            units=units,
            # A notice is a historical fact: 85 FR 48096 was published on 2020-08-10 and that
            # never stops being true, even after a later rule supersedes what it says. Only
            # sources whose content is *replaced* -- a CFR section -- close their interval, so
            # valid_to stays None and "as of 2019" queries correctly exclude this notice while
            # "as of today" queries still find the reasoning behind a 2020 amendment.
            valid_from=str(date),
            valid_to=None,
            references=normalize_references(record.get("cfr_references") or []),
            url=record.get("html_url") or "",
            meta=meta,
        )

    def documents(self) -> Iterator[SourceDoc]:
        """Every notice matching the filters, deduplicated across parts.

        One notice commonly amends several parts at once -- 2026-03610 lists eight, including
        630 -- so the same document comes back from every part search it matches. Yielding it
        twice would double-count it in the corpus and give two ids to one printed page.

        A document that cannot be built is recorded on ``skipped`` and the run continues. A
        single unreachable notice is not a reason to abandon the other 499; a search page that
        will not parse is, because that is the index and a partial index is a silent one.
        """
        self.skipped.clear()
        seen: set[str] = set()
        for part in self.cfr_parts:
            for row in self._rows(part):
                number = str(row.get("document_number") or "").strip()
                if not number or number in seen:
                    continue
                seen.add(number)
                if len(seen) - len(self.skipped) > self.max_documents:
                    log.warning("stopping at the %d-document cap; %s part %s matched more",
                                self.max_documents, self.cfr_title, part)
                    return
                try:
                    yield self.to_doc(self.record(number))
                except NoticeUnavailable as exc:
                    self.skipped[number] = "not served"
                    log.warning("%s: no document at %s; skipping", number, exc)
                # ValueError covers MalformedNotice and the Unit/SourceDoc invariants: an
                # anchor collision the splitter could not resolve is a bad notice, not a bad
                # ingest, and it should cost one document rather than all of them.
                except (ValueError, httpx.HTTPError, RateLimited) as exc:
                    self.skipped[number] = str(exc)
                    log.warning("%s: could not be ingested (%s); skipping", number, exc)


def _fr_citation(units: Sequence[Unit]) -> str:
    """``"85 FR 48096"`` from the masthead, when the raw text carried one.

    Worth the eight lines: it is how a notice is cited in a brief, in an OPM memo and in the
    CFR's own source credits, and it is the string a user will paste in to look one up.
    """
    front = units[0].text if units else ""
    volume = _FR_VOLUME.search(front)
    pages = _FR_PAGES.search(front)
    return f"{volume.group(1)} FR {pages.group(1)}" if volume and pages else ""
