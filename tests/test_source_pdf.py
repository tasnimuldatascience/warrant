"""Tests for the PDF / image / OCR source.

Every PDF here is built by PyMuPDF at test time. Nothing is downloaded and nothing is
committed as a binary blob, which is deliberate: a checked-in fixture PDF is opaque -- when
a layout assertion fails there is no way to see what the page looked like except by opening
it in a viewer -- and the properties under test (a gutter, a ruled table, a page that is
only a raster) are all three lines of ``fitz`` to construct exactly.

The OCR tests run the real RapidOCR engine. They stay in the default suite because the
whole point of this module is the scanned-page path, and a mocked OCR test proves only that
the routing works, not that anything was read. It costs about a second: the engine is a
process-wide singleton, so the first test pays for the ONNX models and the rest do not.
"""

from __future__ import annotations

import fitz
import pytest

from warrant.sources.base import (
    KIND_CAPTION,
    KIND_HEADING,
    KIND_OCR,
    KIND_PROSE,
    KIND_TABLE,
    Source,
)
from warrant.sources.pdf import (
    AUTHORITY_ARCHIVAL,
    PdfParseError,
    PdfRef,
    PdfSource,
    granule_url,
    ocr_engine,
    parse_pdf,
    valid_from_package,
)

# -- page builders ---------------------------------------------------------------------


def _pdf(build) -> bytes:
    """Run ``build(doc)`` over a fresh document and hand back its bytes."""
    doc = fitz.open()
    try:
        build(doc)
        return doc.tobytes()
    finally:
        doc.close()


def _text_page(doc: fitz.Document, lines: list[tuple[float, str]], *, size: float = 10,
               bold: bool = False, width: float = 612, height: float = 792) -> fitz.Page:
    page = doc.new_page(width=width, height=height)
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
    return page


def _draw_table(page: fitz.Page, rows: list[list[str]], *, x: float = 60, y: float = 300,
                cell_w: float = 140, cell_h: float = 24) -> None:
    """A ruled grid, which is what ``page.find_tables`` looks for."""
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x + c * cell_w, y + r * cell_h,
                             x + (c + 1) * cell_w, y + (r + 1) * cell_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (3, 4, -3, -2), cell, fontsize=9)


def _raster(text: list[str], *, width: float = 320, height: float = 180) -> bytes:
    """A PNG of some rendered type -- a stand-in for a page off a 1994 microfiche scan."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        for i, line in enumerate(text):
            page.insert_text((20, 50 + i * 40), line, fontsize=20)
        return page.get_pixmap(dpi=150).tobytes("png")
    finally:
        doc.close()


def _scan_page(doc: fitz.Document, text: list[str], *, width: float = 320,
               height: float = 180) -> fitz.Page:
    """A page whose entire content is one image: no text layer at all."""
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=_raster(text, width=width, height=height))
    return page


class _StubOCR:
    """A RapidOCR stand-in. Returns ``(result, elapse)`` exactly as the real engine does."""

    def __init__(self, rows: list[tuple[str, float]] | None = None) -> None:
        self.rows = rows if rows is not None else [("SCANNED HEADING", 0.62), ("body", 0.94)]
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        out = []
        for i, (text, score) in enumerate(self.rows):
            y = 10 + i * 40
            out.append([[[10, y], [300, y], [300, y + 30], [10, y + 30]], text, str(score)])
        return out, [0.0, 0.0, 0.0]


def _texts(units, kind=None) -> list[str]:
    return [u.text for u in units if kind is None or u.kind == kind]


# -- text extraction and reading order -------------------------------------------------


def test_plain_text_comes_out_in_reading_order():
    data = _pdf(lambda d: _text_page(d, [(100, "First paragraph."),
                                         (140, "Second paragraph."),
                                         (180, "Third paragraph.")]))
    units = parse_pdf(data, doc_id="plain", ocr=False)
    assert _texts(units) == ["First paragraph.", "Second paragraph.", "Third paragraph."]
    assert all(u.kind == KIND_PROSE for u in units)
    assert units[0].locator.startswith("page=1 block=1 bbox=")


def test_a_larger_line_becomes_a_heading_and_labels_what_follows():
    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 100), "Restoration of Annual Leave", fontsize=18, fontname="hebo")
        for i, line in enumerate(["An employee is entitled to restored leave.",
                                  "The leave must be used within two years.",
                                  "Restored leave is credited to a separate account."]):
            page.insert_text((72, 140 + i * 24), line, fontsize=10)

    units = parse_pdf(_pdf(build), doc_id="heading", ocr=False)
    assert units[0].kind == KIND_HEADING
    assert units[0].text == "Restoration of Annual Leave"
    assert units[0].heading == ""  # a heading is not its own heading
    assert all(u.heading == "Restoration of Annual Leave" for u in units[1:])


def test_a_heading_carries_onto_the_next_page():
    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 100), "Subpart C--Restored Leave", fontsize=18, fontname="hebo")
        for i in range(3):
            page.insert_text((72, 140 + i * 24), f"Body sentence {i} on the first page.",
                             fontsize=10)
        second = doc.new_page()
        for i in range(3):
            second.insert_text((72, 100 + i * 24), f"Continued sentence {i}.", fontsize=10)

    units = parse_pdf(_pdf(build), doc_id="carry", ocr=False)
    page_two = [u for u in units if u.locator.startswith("page=2")]
    assert page_two
    assert all(u.heading == "Subpart C--Restored Leave" for u in page_two)


def _columns(page: fitz.Page, rows: int, *, top: float = 120) -> None:
    """Two columns of entries that share baselines, written a column at a time.

    Column-at-a-time, not row-at-a-time, because that is how a typesetter emits them and
    because MuPDF's block segmentation follows content-stream order: interleaving the two
    columns makes MuPDF merge each facing pair into one block, and no reordering downstream
    can undo that. Sharing the baselines is the part that matters -- it is what makes a
    naive top-to-bottom sort produce L0 R0 L1 R1 instead of the two columns in order.
    """
    for i in range(rows):
        page.insert_textbox(fitz.Rect(72, top + i * 40, 290, top + i * 40 + 30),
                            f"Left {i}.", fontsize=10)
    for i in range(rows):
        page.insert_textbox(fitz.Rect(322, top + i * 40, 540, top + i * 40 + 30),
                            f"Right {i}.", fontsize=10)


def test_two_columns_are_read_down_and_then_across():
    """The failure this exists for: y-sorting alternates the columns into fluent nonsense."""
    def build(doc):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "FEDERAL REGISTER NOTICE", fontsize=18, fontname="hebo")
        _columns(page, 5)

    units = parse_pdf(_pdf(build), doc_id="cols", ocr=False)
    body = _texts(units, KIND_PROSE)
    assert body == [f"Left {i}." for i in range(5)] + [f"Right {i}." for i in range(5)]
    # And the masthead is still read first, ahead of both columns.
    assert units[0].text == "FEDERAL REGISTER NOTICE"


def test_a_full_width_headline_does_not_hide_the_gutter():
    """A masthead spanning both columns must not fill the white channel between them."""
    def build(doc):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(60, 60, 552, 100),
                            "Office of Personnel Management, notice of proposed rulemaking "
                            "concerning the restoration of forfeited annual leave",
                            fontsize=14)
        _columns(page, 4, top=140)

    body = _texts(parse_pdf(_pdf(build), doc_id="span", ocr=False), KIND_PROSE)
    assert body == [f"Left {i}." for i in range(4)] + [f"Right {i}." for i in range(4)]


def test_the_printed_cfr_page_geometry():
    """The layout of CFR-2023-title5-vol1, rebuilt to the point.

    Three things on that page each broke the column detector on their own, and none of them
    is visible in a synthetic two-column test: the gutter is only 9 pt wide; a rotated
    ownership stamp in the left margin stretches the text extent by 25%; and the folio is
    centred *on* the gutter, three characters wide, closing it. Reading order must come out
    running head, left column, right column, furniture.
    """
    def build(doc):
        page = doc.new_page(width=612, height=792)
        page.insert_text((132, 165), "5 CFR Ch. I (1-1-23 Edition) 630.306", fontsize=9)
        for i in range(6):
            page.insert_textbox(fitz.Rect(132, 180 + i * 80, 303, 180 + i * 80 + 70),
                                f"Left column paragraph {i}. " * 4, fontsize=9)
        for i in range(6):
            page.insert_textbox(fitz.Rect(312, 180 + i * 80, 483, 180 + i * 80 + 70),
                                f"Right column paragraph {i}. " * 4, fontsize=9)
        page.insert_text((18, 690), "jspears on DSK121TN23PROD with CFR", fontsize=6,
                         rotate=90)
        page.insert_text((298.5, 725), "862", fontsize=9)  # folio, centred on the gutter
        page.insert_text((25, 772), "VerDate Sep<11>2014 11:11 Jun 05, 2023 Jkt 259008 "
                                    "PO 00000 Frm 00872 Fmt 8010 Sfmt 8010 259008", fontsize=6)

    units = parse_pdf(_pdf(build), doc_id="CFR-2023-title5-vol1-sec630-306", ocr=False)
    body = [u.text for u in units]
    left = [i for i, t in enumerate(body) if t.startswith("Left column paragraph")]
    right = [i for i, t in enumerate(body) if t.startswith("Right column paragraph")]
    assert len(left) == len(right) == 6
    assert left == sorted(left) and right == sorted(right)
    assert max(left) < min(right), "the columns interleaved"
    assert body[0].startswith("5 CFR Ch. I")
    assert body[-1].startswith("VerDate")


def test_a_single_column_page_is_not_split_into_two():
    """The false positive that matters: the CFR is set in one column and must stay that way."""
    def build(doc):
        page = doc.new_page()
        for i in range(8):
            page.insert_textbox(fitz.Rect(72, 100 + i * 30, 540, 100 + i * 30 + 24),
                                f"Paragraph {i} runs the full measure of the page.", fontsize=10)

    body = _texts(parse_pdf(_pdf(build), doc_id="single", ocr=False), KIND_PROSE)
    assert body == [f"Paragraph {i} runs the full measure of the page." for i in range(8)]


# -- tables ----------------------------------------------------------------------------


def test_a_table_is_one_unit_serialised_one_row_per_line():
    rows = [["Grade", "Step 1", "Step 2"], ["WG-1", "18.42", "19.10"], ["WG-2", "19.55", "20.30"]]

    def build(doc):
        page = doc.new_page()
        page.insert_text((60, 60), "Wage Schedule", fontsize=15, fontname="hebo")
        _draw_table(page, rows, y=100)
        page.insert_text((60, 260), "Nothing in this table changes eligibility.", fontsize=10)

    units = parse_pdf(_pdf(build), doc_id="wages", ocr=False)
    tables = [u for u in units if u.kind == KIND_TABLE]
    assert len(tables) == 1
    assert tables[0].text == ("Grade | Step 1 | Step 2\n"
                              "WG-1 | 18.42 | 19.10\n"
                              "WG-2 | 19.55 | 20.30")
    assert tables[0].anchor == "p1-t1"
    assert "rows=3" in tables[0].locator
    # It sits in the flow, so it is under the heading printed above it, not the one before.
    assert tables[0].heading == "Wage Schedule"


def test_table_cells_are_not_also_emitted_as_prose():
    def build(doc):
        page = doc.new_page()
        _draw_table(page, [["Grade", "Rate"], ["WG-1", "18.42"]], y=100)
        page.insert_text((60, 260), "Text outside the table.", fontsize=10)

    units = parse_pdf(_pdf(build), doc_id="dedupe", ocr=False)
    assert _texts(units, KIND_PROSE) == ["Text outside the table."]
    assert not any("18.42" in u.text for u in units if u.kind != KIND_TABLE)


# -- images and OCR --------------------------------------------------------------------


def test_an_image_only_page_is_routed_to_ocr_and_tagged_as_such():
    data = _pdf(lambda d: _scan_page(d, ["RESTORED LEAVE", "Public Law 103-353"]))
    units = parse_pdf(data, doc_id="scan-1994")
    assert [u.kind for u in units] == [KIND_OCR]
    unit = units[0]
    assert unit.anchor == "p1-ocr"
    assert "RESTORED" in unit.text.upper()
    assert "103-353" in unit.text
    assert "conf=" in unit.locator and "dpi=200" in unit.locator


def test_ocr_confidence_is_recorded_on_the_unit():
    data = _pdf(lambda d: _scan_page(d, ["anything"]))
    stub = _StubOCR([("SCANNED HEADING", 0.62), ("body text", 0.94)])
    units = parse_pdf(data, doc_id="conf", ocr_engine=stub)
    assert stub.calls == 1
    assert units[0].kind == KIND_OCR
    assert units[0].text == "SCANNED HEADING\nbody text"
    assert "conf=0.780" in units[0].locator  # mean of 0.62 and 0.94
    assert "lines=2" in units[0].locator


def test_a_page_with_text_is_never_sent_to_the_ocr_engine():
    class _Explode:
        def __call__(self, image):  # pragma: no cover - the assertion is that this is unused
            raise AssertionError("a typeset page was sent to OCR")

    data = _pdf(lambda d: _text_page(d, [(100 + i * 24, f"Line {i} of ordinary text.")
                                         for i in range(10)]))
    units = parse_pdf(data, doc_id="typeset", ocr_engine=_Explode())
    assert units and not any(u.kind == KIND_OCR for u in units)


def test_ocr_can_be_switched_off_entirely():
    data = _pdf(lambda d: _scan_page(d, ["RESTORED LEAVE"]))
    assert parse_pdf(data, doc_id="no-ocr", ocr=False) == []


def test_the_ocr_engine_is_built_once_per_process():
    """Constructing RapidOCR loads three ONNX models, which costs about as much as reading
    a page. Per-page construction would roughly double OCR time and look like nothing."""
    assert ocr_engine() is ocr_engine()


@pytest.mark.neural
def test_several_scanned_pages_each_get_their_own_ocr_unit():
    def build(doc):
        _scan_page(doc, ["PAGE ONE TEXT"])
        _scan_page(doc, ["PAGE TWO TEXT"])
        _scan_page(doc, ["PAGE THREE TEXT"])

    units = parse_pdf(_pdf(build), doc_id="multi-scan")
    assert [u.anchor for u in units] == ["p1-ocr", "p2-ocr", "p3-ocr"]
    assert all(u.kind == KIND_OCR for u in units)


def test_an_embedded_figure_becomes_a_caption_unit_with_a_bbox():
    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 80), "The seal appears below.", fontsize=11)
        page.insert_image(fitz.Rect(72, 100, 272, 250),
                          stream=_raster(["SEAL"], width=200, height=150))
        page.insert_text((72, 270), "Figure 1. Seal of the Office of Personnel Management.",
                         fontsize=9)

    units = parse_pdf(_pdf(build), doc_id="figure", ocr=False)
    captions = [u for u in units if u.kind == KIND_CAPTION]
    assert len(captions) == 1
    caption = captions[0]
    assert caption.anchor == "p1-img1"
    assert caption.text == "Figure 1. Seal of the Office of Personnel Management."
    assert "px=" in caption.locator
    x0, y0, x1, y1 = (float(v) for v in caption.locator.split("bbox=")[1].split()[0].split(","))
    # The bbox is where the image actually landed, which is inside the rect it was placed in
    # -- PyMuPDF letterboxes to preserve the aspect ratio -- and it must be a real rectangle
    # or a UI has nothing to crop to.
    assert x1 > x0 and y1 > y0
    assert 71 <= x0 and x1 <= 273 and 99 <= y0 and y1 <= 251


def test_an_uncaptioned_figure_still_carries_its_address():
    def build(doc):
        page = doc.new_page()
        for i in range(6):
            page.insert_text((72, 100 + i * 24), f"Body line {i} well away from the mark.",
                             fontsize=10)
        page.insert_image(fitz.Rect(380, 580, 540, 700),
                          stream=_raster(["X"], width=200, height=150))

    captions = [u for u in parse_pdf(_pdf(build), doc_id="bare", ocr=False)
                if u.kind == KIND_CAPTION]
    assert len(captions) == 1
    assert captions[0].text.startswith("[figure on page 1")


def test_a_full_page_scan_produces_no_caption_unit():
    """The scan *is* the page; a caption unit for it would be a second empty citation."""
    data = _pdf(lambda d: _scan_page(d, ["SCANNED"]))
    units = parse_pdf(data, doc_id="scan-only")
    assert not any(u.kind == KIND_CAPTION for u in units)


# -- anchors ---------------------------------------------------------------------------


def test_anchors_are_unique_across_a_mixed_document():
    def build(doc):
        for n in range(3):
            page = doc.new_page()
            page.insert_text((72, 60), f"Part {n}", fontsize=16, fontname="hebo")
            for i in range(4):
                page.insert_text((72, 100 + i * 24), f"Sentence {i} on page {n}.", fontsize=10)
            _draw_table(page, [["A", "B"], [str(n), str(i)]], y=250)
            page.insert_image(fitz.Rect(72, 400, 232, 520),
                              stream=_raster(["F"], width=200, height=150))
        _scan_page(doc, ["SCANNED PAGE"])

    units = parse_pdf(_pdf(build), doc_id="mixed")
    anchors = [u.anchor for u in units]
    assert len(anchors) == len(set(anchors))
    assert "p1-t1" in anchors and "p1-img1" in anchors and "p4-ocr" in anchors
    assert all(a.startswith("p") for a in anchors)


# -- guardrails ------------------------------------------------------------------------


def test_the_page_cap_truncates_rather_than_rejects():
    def build(doc):
        for n in range(6):
            page = doc.new_page()
            page.insert_text((72, 100), f"Page {n} body text.", fontsize=10)

    units = parse_pdf(_pdf(build), doc_id="long", max_pages=2, ocr=False)
    pages = {u.locator.split()[0] for u in units}
    assert pages == {"page=1", "page=2"}
    assert units, "the cap truncates the document, it does not empty it"


def test_the_byte_cap_names_the_document():
    data = _pdf(lambda d: _text_page(d, [(100, "Small enough.")]))
    with pytest.raises(PdfParseError) as exc:
        parse_pdf(data, doc_id="CFR-2023-title5-vol1", max_bytes=64)
    assert "CFR-2023-title5-vol1" in str(exc.value)
    assert "cap" in str(exc.value)


def test_the_parse_budget_names_the_document_and_the_page():
    def build(doc):
        for _ in range(4):
            page = doc.new_page()
            page.insert_text((72, 100), "Body.", fontsize=10)

    with pytest.raises(PdfParseError) as exc:
        parse_pdf(_pdf(build), doc_id="slow-doc", timeout_s=0.0, ocr=False)
    assert "slow-doc" in str(exc.value) and "budget" in str(exc.value)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.4 this is not a pdf", "not a readable PDF"),
        # MuPDF sniffs the bytes and lays this out as a page of HTML; govinfo serves exactly
        # this for a wrong granule id, so it has to fail loudly rather than ingest as a
        # one-unit document reading "404 Not Found".
        (b"<html>404 Not Found</html>", "not a PDF"),
        (b"", "empty response"),
    ],
)
def test_a_corrupt_pdf_raises_a_clear_error_naming_the_document(data, expected):
    with pytest.raises(PdfParseError) as exc:
        parse_pdf(data, doc_id="FR-1994-09-13-notice")
    message = str(exc.value)
    assert message.startswith("FR-1994-09-13-notice: ")
    assert expected in message
    assert "Traceback" not in message


def test_an_encrypted_pdf_says_so():
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "secret", fontsize=10)
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    with pytest.raises(PdfParseError) as exc:
        parse_pdf(data, doc_id="sealed")
    assert "sealed" in str(exc.value) and "password" in str(exc.value)


# -- the source ------------------------------------------------------------------------


def test_pdf_source_satisfies_the_protocol():
    assert isinstance(PdfSource(refs=[]), Source)


def test_pdf_source_reads_a_local_file_and_labels_it(tmp_path):
    path = tmp_path / "CFR-2023-title5-vol1-sec630-306.pdf"
    path.write_bytes(_pdf(lambda d: _text_page(
        d, [(100, "630.306 Time limit for use of restored annual leave.")], size=16, bold=True)))
    source = PdfSource(refs=[PdfRef(package="CFR-2023-title5-vol1",
                                    granule="CFR-2023-title5-vol1-sec630-306",
                                    path=path)], ocr=False)
    docs = list(source.documents())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "govinfo"
    assert doc.doc_id == "CFR-2023-title5-vol1-sec630-306"
    assert doc.authority == AUTHORITY_ARCHIVAL
    assert doc.valid_from == "2023-01-01"  # read off the package name
    assert doc.meta["package"] == "CFR-2023-title5-vol1"
    assert doc.meta["units_prose"] == "1"
    assert doc.title.startswith("630.306")  # the catchline, not the granule id
    assert not source.failed


def test_a_document_that_cannot_be_read_is_recorded_not_raised(tmp_path):
    good = tmp_path / "good.pdf"
    good.write_bytes(_pdf(lambda d: _text_page(d, [(100, "Readable.")])))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 truncated")
    source = PdfSource(
        refs=[PdfRef(package="FR-1994-09-13", granule="bad", path=bad),
              PdfRef(package="FR-1995-01-04", granule="good", path=good)],
        ocr=False,
    )
    docs = list(source.documents())
    assert [d.doc_id for d in docs] == ["good"]
    assert "bad" in source.failed
    assert "not a readable PDF" in source.failed["bad"]


def test_a_ref_can_override_authority_and_validity(tmp_path):
    path = tmp_path / "notice.pdf"
    path.write_bytes(_pdf(lambda d: _text_page(d, [(100, "Notice text.")])))
    source = PdfSource(refs=[PdfRef(package="FR-2020-08-07", granule="notice", path=path,
                                    title="Restored leave", authority=3,
                                    references=("5 CFR 630.306",))], ocr=False)
    doc = next(iter(source.documents()))
    assert doc.authority == 3
    assert doc.title == "Restored leave"
    assert doc.references == ["5 CFR 630.306"]
    assert doc.valid_from == "2020-08-07"


def test_the_govinfo_url_pattern():
    assert granule_url("CFR-2023-title5-vol1", "CFR-2023-title5-vol1-sec630-306") == (
        "https://www.govinfo.gov/content/pkg/CFR-2023-title5-vol1/pdf/"
        "CFR-2023-title5-vol1-sec630-306.pdf"
    )


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        ("CFR-2023-title5-vol1", "2023-01-01"),
        ("FR-1994-09-13", "1994-09-13"),
        ("BILLS-117hr3076enr", ""),  # no date in the name; guessing one would be a lie
        ("", ""),
    ],
)
def test_validity_is_read_off_the_package_name(package, expected):
    assert valid_from_package(package) == expected
