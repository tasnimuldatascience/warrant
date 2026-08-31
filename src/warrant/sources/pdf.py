"""PDF, images and OCR: the printed record, including the pages that are only pictures.

Every other source in Warrant is handed text. This one is handed paper. govinfo serves the
*printed* record -- the CFR annual editions, the Federal Register back to 1936 -- and a
Federal Register notice from 1994 is a scan of a page, not a text stream. There is no XML
behind it and no HTML rendering of it. OCR is not an enhancement here; it is the only way to
read the document at all, and a corpus that skips those pages simply does not contain the
1990s.

Four things come off a page, and they are kept apart because a verifier has to weigh them
differently:

    KIND_PROSE     text drawn as text        exact, quotable
    KIND_HEADING   ditto, larger or bold     the structure a citation hangs off
    KIND_TABLE     a ruled grid              one unit, one row per line
    KIND_OCR       recognised from a raster  evidence, with a confidence attached
    KIND_CAPTION   a figure the text pipeline cannot read, plus whatever names it

Two decisions in here are worth stating up front because they are the ones that go wrong
silently:

**Reading order is not y-then-x.** The Federal Register is set in three columns and the CFR
in one. Sorting blocks top-to-bottom interleaves the columns of an FR page into alternating
half-sentences, which reads as fluent nonsense -- it retrieves, it embeds, and it is wrong,
and nothing downstream can detect it. Columns are detected explicitly (see ``_gutters``).

**A table is one unit.** Serialised exactly as ``corpus/parse.py`` serialises CFR tables --
one row per line, cells joined by " | " -- so a downstream consumer sees one table format
whether the table arrived as GPO XML or as ruled lines on a scanned page.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    AUTHORITY_ARCHIVAL,
    KIND_CAPTION,
    KIND_HEADING,
    KIND_OCR,
    KIND_PROSE,
    KIND_TABLE,
    SourceDoc,
    Unit,
    merge_anchors,
)

log = logging.getLogger(__name__)

GOVINFO = "https://www.govinfo.gov/content/pkg"
USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

# -- guardrails ------------------------------------------------------------------------
# A public bulk archive will hand you a 900-page annual edition without warning, and the
# cost of a page is not bounded by its size: an image-only page costs an OCR pass, so 900
# scanned pages is roughly 900 x 0.4 s of ONNX inference on top of the download. Every cap
# below is a limit on *one document* so that one pathological input cannot wedge an ingest
# that is otherwise 300 documents wide.

#: Pages parsed per document. Past this the document is truncated, not rejected: the front
#: of a CFR volume is still worth having, and refusing it outright loses more than it saves.
MAX_PAGES = 400
#: Bytes accepted per document. Rejected outright rather than truncated, because a stream
#: this large is nearly always the wrong address rather than a document anybody meant to
#: ingest: govinfo serves a whole Federal Register issue at the package level, and
#: FR-1994-12-07.pdf -- one day of it, scanned -- is 313 MB. Notices are addressed as
#: granules for exactly this reason, and this cap is what says so out loud when they are not.
MAX_BYTES = 96 * 1024 * 1024
#: Wall-clock budget for parsing one document, checked between pages and before each OCR
#: pass. Generous, because OCR at 200 dpi is genuinely slow; it exists to bound a pathology,
#: not to tune throughput.
TIMEOUT_S = 300.0

#: Render resolution for OCR. 200 dpi is the floor at which RapidOCR reads 8-point Federal
#: Register body type reliably; 300 dpi roughly doubles the pixel count for a small accuracy
#: gain, and 150 dpi starts losing the footnote apparatus.
OCR_DPI = 200
#: Characters of extracted text per square inch below which a page is treated as a scan. A
#: normally typeset page runs 20-40; a scanned page carrying only a digital header stamp
#: ("Federal Register / Vol. 59, No. 176") runs well under 1.
SCAN_TEXT_DENSITY = 2.0
#: Fraction of the page an image must cover before it counts as the page itself rather than
#: as a figure on it. Scans are letterboxed by aspect-ratio fitting, so a full-page scan of a
#: 8.5x11 original often covers only ~75% of an A4 page box; 40% clears that with room, and
#: is still far above any seal, chart or signature block.
SCAN_IMAGE_COVERAGE = 0.4
#: Smallest image, as a fraction of page area, that earns a caption unit. Below this are
#: rules, bullets, logos in a running head and colour swatches -- furniture, not figures.
FIGURE_MIN_COVERAGE = 0.005
#: How far below (or above) an image to look for its caption. One or two lines of 9-point
#: type with leading; beyond that the "caption" is body text that happens to follow.
CAPTION_GAP_PT = 42.0


class PdfParseError(ValueError):
    """A PDF could not be read, naming the document.

    PyMuPDF raises ``FileDataError: Failed to open stream``, which says nothing about which
    of 300 documents failed and cannot be grepped for in a log. An ingest over govinfo hits
    encrypted filings, truncated downloads and HTML error pages served with a .pdf suffix;
    each has to name itself or the only way to find it is to re-run the ingest by hand.
    """


class PdfUnavailable(LookupError):
    """govinfo has no PDF at this package/granule address."""


# -- text geometry ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Block:
    """One text block off ``page.get_text("dict")``, reduced to what layout needs."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float  # largest span size in the block
    bold: bool
    lines: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


_WS = re.compile(r"\s+")
#: PyMuPDF span flag bit 4 (value 16) is "bold". Named because the raw ``flags & 16`` in a
#: sort key is unreadable and the bit assignment is not guessable from the value.
_FLAG_BOLD = 1 << 4


def _blocks(page: fitz.Page) -> list[_Block]:
    out: list[_Block] = []
    for blk in page.get_text("dict").get("blocks", ()):
        if blk.get("type") != 0:  # 1 is an image block; images are handled separately
            continue
        parts: list[str] = []
        size = 0.0
        bold = False
        lines = blk.get("lines", ())
        for line in lines:
            spans = line.get("spans", ())
            parts.append("".join(s.get("text", "") for s in spans))
            for span in spans:
                if not span.get("text", "").strip():
                    continue
                size = max(size, float(span.get("size", 0.0)))
                bold = bold or bool(int(span.get("flags", 0)) & _FLAG_BOLD)
        text = _WS.sub(" ", " ".join(parts)).strip()
        if not text:
            continue
        x0, y0, x1, y1 = blk["bbox"]
        out.append(_Block(x0, y0, x1, y1, text, size, bold, len(lines)))
    return out


#: A block this wide relative to the text area is spanning the page, not sitting in a
#: column. 0.6 rather than something nearer 1.0 because a two-column headline routinely
#: stops short of the right margin, and 0.6 still cannot be reached by a column that must
#: leave room for a gutter and a facing column.
_SPAN_FRACTION = 0.6
#: ...and this is the narrowest a block may be and still be treated as spanning when it
#: crosses a gutter. Wider than one column is impossible below 0.5 on a two-column page, so
#: 0.25 is generous; it exists only to exclude the folio and other centred furniture.
_SPAN_MIN_FRACTION = 0.25
#: Narrowest accepted gutter, in points. Absolute rather than a fraction of the page,
#: because PDF units are physical (1/72 inch) and a gutter is set in ems, not in percent --
#: and because a fraction is skewed by marginalia: the 2023 CFR carries a rotated ownership
#: stamp at x=18 which stretches the "text width" by 25% and shrank the relative gutter
#: below any workable threshold. Measured on that volume, the two columns run 132-303 and
#: 312-483: a 9 pt gutter. Inter-word space in 9 pt Times is about 2.2 pt, so 6 pt separates
#: the two cleanly with room on both sides.
_GUTTER_MIN_PT = 6.0
#: Projection resolution. Half a point: fine enough that a 9 pt gutter is 18 bins wide, and
#: coarse enough that a page is a few thousand bytes of bookkeeping.
_GUTTER_BIN_PT = 0.5
#: Share of a page's characters that may sit over a gutter and still leave it a gutter. The
#: projection is weighted by text length rather than being a plain occupancy mask because
#: page furniture straddles the channel: the folio "862" is centred on the CFR's gutter, is
#: three characters wide, and with a boolean mask it closed the gutter and interleaved the
#: whole page. A column contributes ~50% of the page's characters to every bin it covers, so
#: 2% separates the two by more than an order of magnitude.
_GUTTER_NOISE_FRACTION = 0.02


def _gutters(blocks: list[_Block]) -> list[float]:
    """The x positions of the vertical white channels separating columns, left to right.

    Detection is by *projection*: sweep the horizontal extent of the text and mark every
    x covered by a block, then look for a run of unmarked x that runs the full height of the
    text area. A real gutter is that run; a coincidental gap between two words is not,
    because some other block on the page covers it.

    Blocks wider than ``_SPAN_FRACTION`` of the text width are excluded from the projection
    before the search. A two-column page almost always carries a full-width masthead, title
    or footer, and leaving those in fills the gutter and hides it -- this was the entire
    failure: the Federal Register's own running head is what made its three columns look
    like one. Those spanning blocks are not lost, they are handled in ``_reading_order``.

    A gap is accepted only if it is at least ``_GUTTER_MIN_PT`` wide and both sides carry at
    least 15% of the page's characters. The character test is what does the real work: the
    same CFR page has a 109 pt hole between its rotated margin stamp and the left column,
    which is wider than any gutter and is not one, and only the fact that 1% of the page's
    characters sit to the left of it says so.

    Returns [] for single-column pages, which is the overwhelming majority of the CFR.
    """
    if len(blocks) < 4:
        return []
    left = min(b.x0 for b in blocks)
    right = max(b.x1 for b in blocks)
    width = right - left
    if width <= 0:
        return []
    narrow = [b for b in blocks if b.width < _SPAN_FRACTION * width]
    if len(narrow) < 4:
        return []

    bins = max(64, int(width / _GUTTER_BIN_PT))
    scale = bins / width
    total_chars = sum(len(b.text) for b in narrow)
    weight = [0] * bins
    for b in narrow:
        lo = max(0, int((b.x0 - left) * scale))
        hi = min(bins, int((b.x1 - left) * scale) + 1)
        for i in range(lo, hi):
            weight[i] += len(b.text)

    noise = _GUTTER_NOISE_FRACTION * total_chars
    min_run = max(2, int(_GUTTER_MIN_PT * scale))
    gutters: list[float] = []
    run_start = None
    for i in range(bins + 1):
        occupied = i == bins or weight[i] > noise
        if not occupied and run_start is None:
            run_start = i
        elif occupied and run_start is not None:
            if i - run_start >= min_run:
                x = left + ((run_start + i) / 2) / scale
                lhs = sum(len(b.text) for b in narrow if b.cx < x)
                if 0.15 <= lhs / max(total_chars, 1) <= 0.85:
                    gutters.append(x)
            run_start = None
    return gutters


def _reading_order(blocks: list[_Block]) -> list[_Block]:
    """Blocks in the order a person reads them.

    Bands first, then columns, then top-to-bottom. A *band* is the horizontal strip between
    two page-spanning blocks: a title spans, then two columns run under it, then a footer
    spans. Sorting on (band, column, y) reproduces that; sorting on y alone alternates
    between the columns and produces sentences that stop mid-clause.
    """
    gutters = _gutters(blocks)
    if not gutters:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    def column(b: _Block) -> int:
        return sum(1 for g in gutters if b.cx > g)

    text_width = max(b.x1 for b in blocks) - min(b.x0 for b in blocks)

    def spans(b: _Block) -> bool:
        # Crossing a gutter is not enough to be a spanning heading; a block has to be wide
        # enough to *be* one. The CFR's folio sits centred on the gutter and is three
        # characters wide, and treating it as a band boundary lifted the page number to the
        # top of the page it numbers.
        return (b.width >= _SPAN_MIN_FRACTION * text_width
                and any(b.x0 < g < b.x1 for g in gutters))

    spanners = sorted((b for b in blocks if spans(b)), key=lambda b: b.y0)

    def key(b: _Block) -> tuple[float, ...]:
        # A spanner opens the band beneath it -- hence ``s is b``, which puts a spanner in
        # its own band rather than the one above. Without it the GPO printing footer at the
        # foot of the page shares a band with the columns and, sorting ahead of them as
        # spanners do, is read before the text it sits under.
        band = sum(1 for s in spanners if s.y0 < b.y0 or s is b)
        return (band, 0 if spans(b) else 1, column(b), round(b.y0, 1), b.x0)

    return sorted(blocks, key=key)


def _body_size(pages: list[list[_Block]]) -> float:
    """Median font size of the document, weighted by characters.

    Weighted, because unweighted medians follow the *count* of blocks and a page of headings
    over a page of body text moves the estimate; what a heading has to be measured against
    is the size most of the ink is set in.
    """
    sizes: list[float] = []
    for blocks in pages:
        for b in blocks:
            if b.size > 0:
                sizes.extend([b.size] * max(1, len(b.text) // 10))
    return statistics.median(sizes) if sizes else 0.0


#: A block bigger than this multiple of body size is a heading whatever else it looks like.
_HEADING_SIZE_RATIO = 1.12
#: Longest a heading can be. Headings in the CFR run to about 90 characters ("Employees
#: whose services are required during a period of restricted leave"); a 200-character bold
#: run is an emphasised paragraph.
_HEADING_MAX_CHARS = 200


def _is_heading(b: _Block, body: float) -> bool:
    if b.lines > 3 or len(b.text) > _HEADING_MAX_CHARS or not b.text:
        return False
    # A folio is set in the same face as a heading and is not one. Left as a unit rather
    # than dropped -- "862" is how a reader of the printed volume finds the page -- but a
    # section's worth of prose must not end up filed under it.
    if b.text.replace("-", "").isdigit():
        return False
    if body > 0 and b.size >= body * _HEADING_SIZE_RATIO:
        return True
    # Same size but bold and short: how the CFR sets section headings, and how govinfo sets
    # "SUMMARY:" / "DATES:" in a Federal Register preamble.
    return b.bold and len(b.text) <= 90 and not b.text.endswith(".")


# -- tables ----------------------------------------------------------------------------


def _table_text(rows: list[list[str | None]]) -> str:
    """One line per row, cells joined by ' | ' -- the CFR serialisation, reused verbatim.

    Empty cells are dropped exactly as ``corpus/parse.py`` drops them. It costs column
    alignment in a sparse table and it is still right, because the alternative is two table
    formats in one index: a consumer that has learned to read a GPO table would have to
    learn a second grammar for the same table when it arrives as ruled lines instead.
    """
    out: list[str] = []
    for row in rows:
        cells = [_WS.sub(" ", c).strip() for c in row if c]
        cells = [c for c in cells if c]
        if cells:
            out.append(" | ".join(cells))
    return "\n".join(out)


def _inside(b: _Block, boxes: list[fitz.Rect]) -> bool:
    """Is this text block part of a table already emitted as a unit?

    Tested on the block's centre rather than on overlap, because ``find_tables`` returns a
    bbox tight to the ruled lines and cell text routinely pokes a point or two outside it.
    """
    cx, cy = b.cx, (b.y0 + b.y1) / 2
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in boxes)


# -- OCR -------------------------------------------------------------------------------

_OCR_LOCK = threading.Lock()
_OCR_ENGINE: object | None = None


def ocr_engine() -> object:
    """The process-wide RapidOCR instance, constructed on first use.

    Lazy and singleton, both deliberately. Lazy because importing ``rapidocr_onnxruntime``
    pulls in an ONNX runtime that most ingests never need -- the CFR is text -- and the
    import alone costs ~0.3 s. Singleton because construction loads three ONNX models
    (detection, classification, recognition) at ~0.3 s, which is comparable to the *whole*
    inference cost of a page: building one per page would roughly double OCR time and would
    do it invisibly, as a constant factor nobody looks for.
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        with _OCR_LOCK:
            if _OCR_ENGINE is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as exc:  # pragma: no cover - depends on the extra
                    raise PdfParseError(
                        "OCR was requested but rapidocr-onnxruntime is not installed. "
                        "Install warrant[sources] -- or, on Python 3.13, note that it "
                        "publishes no wheel there and is excluded by a marker, so use "
                        "3.12 for scans or pass ocr=False to read the text layer only."
                    ) from exc
                _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_lines(result: object) -> list[tuple[fitz.Rect, str, float]]:
    """RapidOCR's ``[[box, text, score], ...]`` reduced to (rect, text, confidence).

    The score arrives as a string in rapidocr-onnxruntime 1.3 and as a float in some
    builds; both are accepted rather than asserted, because a version bump that changed it
    would otherwise turn every scanned page into a parse failure.
    """
    lines: list[tuple[fitz.Rect, str, float]] = []
    for row in result or ():
        try:
            box, text, score = row[0], row[1], row[2]
        except (TypeError, IndexError, KeyError):
            continue
        if not str(text).strip():
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        try:
            conf = float(score)
        except (TypeError, ValueError):
            conf = 0.0
        lines.append((fitz.Rect(min(xs), min(ys), max(xs), max(ys)), str(text).strip(), conf))
    return lines


def _ocr_page(page: fitz.Page, *, dpi: int, engine: object | None) -> tuple[str, float, int]:
    """Recognise a rendered page. Returns (text, mean confidence, line count).

    Lines are re-sorted into bands before joining. RapidOCR emits detections in its own
    detector order, which is close to reading order and not identical to it; banding on line
    height puts a two-line running head back above the body it belongs to.
    """
    pix = page.get_pixmap(dpi=dpi)
    result = (engine or ocr_engine())(pix.tobytes("png"))
    # The engine returns (result, elapse) in rapidocr-onnxruntime; be tolerant of a bare list.
    if isinstance(result, tuple):
        result = result[0]
    lines = _ocr_lines(result)
    if not lines:
        return "", 0.0, 0
    heights = [r.y1 - r.y0 for r, _, _ in lines]
    band = max(statistics.median(heights) * 1.2, 1.0)
    lines.sort(key=lambda item: (int(item[0].y0 / band), item[0].x0))
    text = "\n".join(t for _, t, _ in lines)
    conf = sum(c for _, _, c in lines) / len(lines)
    return text, conf, len(lines)


def _needs_ocr(page: fitz.Page, text_chars: int, images: list[dict]) -> bool:
    """Is this page a picture of a page?

    Two conditions, both required. The text density test alone would send every blank page
    and every full-page chart to the OCR engine; the image-coverage test alone would send a
    perfectly readable page that happens to carry a large figure. Together they say what is
    actually meant: there is a page-sized raster here, and almost no text came off it.
    """
    area = page.rect.get_area()
    if area <= 0:
        return False
    covered = any(fitz.Rect(i["bbox"]).get_area() >= SCAN_IMAGE_COVERAGE * area for i in images)
    if not covered:
        return False
    density = text_chars / (area / 5184.0)  # 72 dpi => 5184 square points per square inch
    return density < SCAN_TEXT_DENSITY


# -- the parser ------------------------------------------------------------------------


def _bbox(rect: fitz.Rect) -> str:
    return f"{rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}"


def _caption_for(rect: fitz.Rect, blocks: list[_Block]) -> str:
    """Text most likely to be the caption of the image at ``rect``.

    Nearest horizontally-overlapping block within ``CAPTION_GAP_PT``, below first. Below
    first because that is where captions go in every federal style guide in use; above is
    accepted as a fallback for the tables-of-figures layout where the label sits on top.
    """
    def overlap(b: _Block) -> float:
        return min(rect.x1, b.x1) - max(rect.x0, b.x0)

    below = [b for b in blocks if overlap(b) > 0 and 0 <= b.y0 - rect.y1 <= CAPTION_GAP_PT]
    if below:
        return min(below, key=lambda b: b.y0 - rect.y1).text
    above = [b for b in blocks if overlap(b) > 0 and 0 <= rect.y0 - b.y1 <= CAPTION_GAP_PT]
    if above:
        return min(above, key=lambda b: rect.y0 - b.y1).text
    return ""


def _open(data: bytes, doc_id: str, max_bytes: int) -> fitz.Document:
    if len(data) > max_bytes:
        raise PdfParseError(
            f"{doc_id}: {len(data)} bytes exceeds the {max_bytes}-byte cap; "
            "refusing to parse (check the URL points at a document, not an archive)"
        )
    if not data.strip():
        raise PdfParseError(f"{doc_id}: empty response, nothing to parse")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises FileDataError, EmptyFileError, RuntimeError
        raise PdfParseError(
            f"{doc_id}: not a readable PDF ({type(exc).__name__}: {exc})"
        ) from exc
    if doc.needs_pass:
        doc.close()
        raise PdfParseError(f"{doc_id}: PDF is password-protected; no text can be extracted")
    if not doc.is_pdf:
        # ``filetype="pdf"`` is a hint, not an assertion: MuPDF sniffs the bytes, and it
        # reads an HTML error page as HTML and lays it out as one page of prose. govinfo
        # serves precisely that for a wrong granule id, so without this check a mis-typed
        # granule ingests as a one-unit document reading "404 Not Found" -- a corpus entry
        # that is worse than a failure, because nothing reports it.
        fmt = doc.metadata.get("format", "unknown") if doc.metadata else "unknown"
        doc.close()
        raise PdfParseError(f"{doc_id}: the response is not a PDF (MuPDF read it as {fmt})")
    if doc.page_count == 0:
        doc.close()
        raise PdfParseError(f"{doc_id}: PDF has no pages (a truncated download)")
    return doc


def parse_pdf(
    data: bytes,
    *,
    doc_id: str,
    max_pages: int = MAX_PAGES,
    max_bytes: int = MAX_BYTES,
    timeout_s: float = TIMEOUT_S,
    ocr: bool = True,
    ocr_dpi: int = OCR_DPI,
    ocr_engine: object | None = None,
) -> list[Unit]:
    """Every citable unit in a PDF, in reading order, whatever the PDF came from.

    Deliberately takes bytes and not a URL or a path: this is the part worth having, and it
    has to be testable on a PDF synthesised in three lines rather than on a network fetch.
    ``PdfSource`` is a thin fetch-and-label wrapper around it.

    ``doc_id`` is not decoration. It is the only identifier in any error raised from here,
    and it is what makes a failure in a 300-document ingest reproducible.

    ``ocr_engine`` overrides the process-wide RapidOCR singleton -- for tests, and for a
    caller that wants a differently-configured engine.

    Anchors are ``p{page}-b{n}`` for prose and headings, ``p{page}-t{n}`` for tables,
    ``p{page}-img{n}`` for figures and ``p{page}-ocr`` for a recognised page, which makes
    them unique by construction. ``merge_anchors`` is applied anyway as a backstop.
    """
    deadline = time.monotonic() + timeout_s
    doc = _open(data, doc_id, max_bytes)
    try:
        if doc.page_count > max_pages:
            log.warning("%s: %d pages, parsing the first %d (page cap)",
                        doc_id, doc.page_count, max_pages)
        pages = list(range(min(doc.page_count, max_pages)))

        # Two passes over the geometry: the heading threshold is a property of the document,
        # not of a page, and a page whose only text is its own title would otherwise have
        # that title measured against itself and come out as body text.
        try:
            per_page = [_blocks(doc[n]) for n in pages]
        except Exception as exc:
            raise PdfParseError(
                f"{doc_id}: text extraction failed ({type(exc).__name__}: {exc})"
            ) from exc
        body = _body_size(per_page)

        units: list[Unit] = []
        heading = ""
        for n in pages:
            # `>=`, not `>`. A zero budget means no time is allowed, and with `>` that only
            # trips if the clock has advanced -- which on Windows, whose monotonic clock has
            # ~15.6ms granularity by default, a four-page parse can finish inside. The test
            # for this failed on windows-3.12 and passed on ubuntu-3.12 and windows-3.13,
            # which is the signature of a threshold that depends on timer resolution rather
            # than on the thing being measured.
            if time.monotonic() >= deadline:
                raise PdfParseError(
                    f"{doc_id}: exceeded the {timeout_s:.0f}s parse budget at page {n + 1} "
                    f"of {len(pages)}"
                )
            # ``heading`` is carried across the page boundary: a section that breaks over a
            # page is still under its heading, and dropping it there leaves the second half
            # of every long section with no structural context at all.
            page_units, heading = _parse_page(
                doc[n], n + 1, per_page[n], body, doc_id=doc_id, ocr=ocr, ocr_dpi=ocr_dpi,
                engine=ocr_engine, heading=heading)
            units.extend(page_units)
        return merge_anchors(units)
    finally:
        doc.close()


def _parse_page(page: fitz.Page, number: int, blocks: list[_Block], body: float, *,
                doc_id: str, ocr: bool, ocr_dpi: int, engine: object | None,
                heading: str) -> tuple[list[Unit], str]:
    """Units for one page, and the heading still in force at the end of it.

    Tables are sorted into the text flow rather than emitted ahead of it. Emitting them
    first is the obvious implementation and it is wrong in a way that is easy to miss: the
    heading a table sits under is the text immediately above it, so a table emitted before
    its own page's headings gets whatever heading was in force on the *previous* page.
    """
    units: list[Unit] = []

    try:
        tables = list(page.find_tables().tables)
    except Exception as exc:
        # A table finder that trips on one page must not cost the page its prose. It is a
        # heuristic over vector graphics and it does trip -- on rotated pages, mostly.
        log.warning("%s p%d: table detection failed (%s: %s)",
                    doc_id, number, type(exc).__name__, exc)
        tables = []

    boxes: list[fitz.Rect] = []
    table_text: dict[int, str] = {}  # id(pseudo-block) -> serialised rows
    pseudo: list[_Block] = []
    for i, table in enumerate(tables, start=1):
        try:
            text = _table_text(table.extract())
        except Exception as exc:
            log.warning("%s p%d: table %d could not be extracted (%s)", doc_id, number, i, exc)
            continue
        if not text:
            continue
        rect = fitz.Rect(table.bbox)
        boxes.append(rect)
        # A table stands in the flow as a block of its own size, so a full-width table is
        # correctly seen as spanning the columns of a two-column page.
        block = _Block(rect.x0, rect.y0, rect.x1, rect.y1, text, 0.0, False,
                       text.count("\n") + 1)
        table_text[id(block)] = text
        pseudo.append(block)

    text_chars = 0
    counts = {"b": 0, "t": 0}
    flow = _reading_order([b for b in blocks if not _inside(b, boxes)] + pseudo)
    for b in flow:
        text_chars += len(b.text)
        rect = fitz.Rect(b.x0, b.y0, b.x1, b.y1)
        if id(b) in table_text:
            counts["t"] += 1
            units.append(Unit(
                anchor=f"p{number}-t{counts['t']}",
                text=b.text,
                heading=heading,
                kind=KIND_TABLE,
                locator=(f"page={number} table={counts['t']} rows={b.lines} "
                         f"bbox={_bbox(rect)}"),
            ))
            continue
        counts["b"] += 1
        kind = KIND_HEADING if _is_heading(b, body) else KIND_PROSE
        units.append(Unit(
            anchor=f"p{number}-b{counts['b']}",
            text=b.text,
            heading="" if kind == KIND_HEADING else heading,
            kind=kind,
            locator=f"page={number} block={counts['b']} bbox={_bbox(rect)}",
        ))
        if kind == KIND_HEADING:
            heading = b.text

    try:
        images = page.get_image_info()
    except Exception as exc:
        log.warning("%s p%d: image inventory failed (%s)", doc_id, number, exc)
        images = []

    area = page.rect.get_area() or 1.0
    scan = ocr and _needs_ocr(page, text_chars, images)
    for i, info in enumerate(images, start=1):
        rect = fitz.Rect(info["bbox"])
        coverage = rect.get_area() / area
        # A full-page scan is the page, not a figure on it; it is read by OCR below and a
        # caption unit for it would be a second, contentless citation to the same page.
        if coverage >= SCAN_IMAGE_COVERAGE or coverage < FIGURE_MIN_COVERAGE:
            continue
        caption = _caption_for(rect, blocks)
        units.append(Unit(
            anchor=f"p{number}-img{i}",
            # With no caption the unit still carries the figure's address, so a UI can put
            # the picture on screen. Retrieval will rarely reach it, which is correct: there
            # is nothing here to match against, and inventing a description would be worse.
            text=caption or f"[figure on page {number}, {rect.width:.0f}x{rect.height:.0f} pt]",
            heading=heading,
            kind=KIND_CAPTION,
            locator=(f"page={number} image={i} bbox={_bbox(rect)} "
                     f"px={info.get('width', 0)}x{info.get('height', 0)}"),
        ))

    if scan:
        text, conf, lines = _ocr_page(page, dpi=ocr_dpi, engine=engine)
        if text:
            units.append(Unit(
                anchor=f"p{number}-ocr",
                text=text,
                heading=heading,
                kind=KIND_OCR,
                # Confidence rides in the locator because it bounds what a citation to this
                # unit may claim. A verifier that cannot see 0.62 will quote it as if it
                # were parsed XML.
                locator=f"page={number} ocr dpi={ocr_dpi} conf={conf:.3f} lines={lines}",
            ))
        else:
            log.info("%s p%d: image-only page, OCR returned nothing", doc_id, number)
    return units, heading


# -- govinfo ---------------------------------------------------------------------------


@dataclass
class GovInfoClient:
    """Cached fetcher for govinfo package/granule PDFs.

    Cached without a TTL, unlike the eCFR index endpoints. govinfo serves the *printed*
    record: CFR-2023-title5-vol1 is the volume as it was printed in 2023 and it will never
    change, so a re-ingest costs no requests and an expiry would only re-download bytes that
    are known to be identical.
    """

    cache_dir: Path
    delay_s: float = 1.0
    timeout_s: float = 120.0
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
            time.sleep(self.delay_s - elapsed)  # a public archive; do not hammer it
        r = httpx.get(url, headers={"User-Agent": USER_AGENT},
                      timeout=self.timeout_s, follow_redirects=True)
        self._last_request = time.monotonic()
        if r.status_code == 404:
            raise PdfUnavailable(url)
        r.raise_for_status()
        return r.content

    def pdf(self, package: str, granule: str) -> bytes:
        """The granule PDF, from disk if it has ever been fetched.

        The write is atomic. A ``write_bytes`` interrupted mid-download leaves a truncated
        file that the cache -- which only checks existence -- then trusts forever, and the
        symptom is a ``PdfParseError`` on a document that downloads perfectly well.
        """
        name = f"{granule}.pdf"
        path = self.cache_dir / name
        if path.exists() and not self.refresh:
            log.debug("cache hit %s", name)
            return path.read_bytes()
        url = granule_url(package, granule)
        log.info("GET %s", url)
        content = self._fetch(url)
        tmp = path.with_suffix(".pdf.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        log.info("cached %s (%d bytes)", name, len(content))
        return content


def granule_url(package: str, granule: str) -> str:
    """govinfo's one addressing rule: /content/pkg/{package}/pdf/{granule}.pdf."""
    return f"{GOVINFO}/{package}/pdf/{granule}.pdf"


#: Package names lead with their date: CFR-2023-title5-vol1, FR-1994-09-13. Both forms are
#: parsed because both are ingested, and a document with no validity date is a document the
#: bitemporal store cannot place in time.
_PKG_DATE = re.compile(r"^[A-Za-z]+-(\d{4})(?:-(\d{2})-(\d{2}))?")


def valid_from_package(package: str) -> str:
    """The date this package's content became true, read off its name.

    A CFR annual edition carries no day, so it is dated to the start of its year -- the
    edition is "as of" the beginning of the year it names for the title it covers. Returns
    "" when the name says nothing, which the caller must leave as "" rather than guess:
    ``SourceDoc`` would rather hold an unknown validity than a fabricated one.
    """
    m = _PKG_DATE.match(package)
    if not m:
        return ""
    year, month, day = m.groups()
    return f"{year}-{month}-{day}" if month and day else f"{year}-01-01"


def _title_of(units: list[Unit]) -> str:
    """A title for a document whose caller did not supply one.

    First heading if there is one, otherwise the first line of text. A govinfo granule PDF
    opens with the section catchline -- "630.306 Time limit for use of restored annual
    leave." -- so the opening line is the title far more often than it is not, and it beats
    showing a reader the granule id.
    """
    for unit in units:
        if unit.kind == KIND_HEADING:
            return unit.text
    for unit in units:
        if unit.kind in (KIND_PROSE, KIND_OCR):
            return unit.text.splitlines()[0][:120].strip()
    return ""


@dataclass(frozen=True)
class PdfRef:
    """One document to ingest, addressed either on govinfo or on disk.

    ``path`` exists so the same source can ingest a PDF somebody dropped in a directory --
    an OPM fact sheet emailed as a PDF has no govinfo address and is still evidence.
    """

    package: str
    granule: str = ""
    title: str = ""
    valid_from: str = ""
    valid_to: str | None = None
    authority: int | None = None
    references: tuple[str, ...] = ()
    path: Path | None = None

    @property
    def doc_id(self) -> str:
        return self.granule or self.package

    @property
    def url(self) -> str:
        return "" if self.path is not None else granule_url(self.package, self.doc_id)


@dataclass
class PdfSource:
    """The ``Source`` for anything that arrives as paper.

    Defaults to ``AUTHORITY_ARCHIVAL``, which is the honest answer for a scan: the printed
    CFR volume *is* the regulation, but what this pipeline recovered from it went through
    OCR and layout heuristics, and ranking that alongside parsed eCFR XML would let a
    misread digit outrank the text it was printed from. A caller who knows better -- a
    born-digital Federal Register PDF, say -- overrides it per ref.
    """

    refs: list[PdfRef]
    cache_dir: Path | None = None
    name: str = "govinfo"
    authority: int = AUTHORITY_ARCHIVAL
    client: GovInfoClient | None = None
    ocr: bool = True
    ocr_dpi: int = OCR_DPI
    max_pages: int = MAX_PAGES
    max_bytes: int = MAX_BYTES
    timeout_s: float = TIMEOUT_S
    #: Documents that could not be read, doc_id -> reason. Recorded rather than raised: one
    #: encrypted filing must not abort a 300-document ingest, and one silently skipped is
    #: worse -- a gap in the corpus with nothing anywhere reporting it. Read it after
    #: ``documents()`` is exhausted; the failure budget counts it.
    failed: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.client is None and self.cache_dir is not None:
            self.client = GovInfoClient(Path(self.cache_dir))

    def _bytes(self, ref: PdfRef) -> bytes:
        if ref.path is not None:
            return Path(ref.path).read_bytes()
        if self.client is None:
            raise PdfParseError(f"{ref.doc_id}: no cache_dir or client to fetch it with")
        return self.client.pdf(ref.package, ref.doc_id)

    def documents(self) -> Iterator[SourceDoc]:
        self.failed.clear()
        for ref in self.refs:
            try:
                units = parse_pdf(
                    self._bytes(ref), doc_id=ref.doc_id, max_pages=self.max_pages,
                    max_bytes=self.max_bytes, timeout_s=self.timeout_s,
                    ocr=self.ocr, ocr_dpi=self.ocr_dpi,
                )
            except (PdfParseError, PdfUnavailable, OSError, httpx.HTTPError) as exc:
                self.failed[ref.doc_id] = f"{type(exc).__name__}: {exc}"
                log.warning("skipping %s: %s", ref.doc_id, exc)
                continue
            if not units:
                self.failed[ref.doc_id] = "parsed to zero units"
                log.warning("skipping %s: parsed to zero units", ref.doc_id)
                continue
            title = ref.title or _title_of(units)
            kinds: dict[str, int] = {}
            for u in units:
                kinds[u.kind] = kinds.get(u.kind, 0) + 1
            yield SourceDoc(
                source=self.name,
                doc_id=ref.doc_id,
                title=title or ref.doc_id,
                authority=ref.authority if ref.authority is not None else self.authority,
                units=units,
                valid_from=ref.valid_from or valid_from_package(ref.package),
                valid_to=ref.valid_to,
                references=list(ref.references),
                url=ref.url,
                # Kind counts travel with the document because "how much of this came from
                # OCR" is the first question anyone asks of a scanned corpus, and counting
                # it at query time means re-reading every unit.
                meta={f"units_{k}": str(v) for k, v in sorted(kinds.items())}
                | {"package": ref.package},
            )
