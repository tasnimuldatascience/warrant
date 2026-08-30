"""The Federal Register source: structure, joins, and the two ways an ingest goes quiet.

Everything here runs offline against a stub transport. That is not only for speed -- the
failures worth testing are invisible against the live API. A search cached forever reports no
new notices, which reads exactly like there being none; a notice dropped on a 404 leaves a
corpus that is merely smaller, with nothing saying so; an anchor collision produces a citation
that resolves to two paragraphs and no error at all. Each test asserts the absence of one of
those states rather than the presence of an exception.

The canned payloads reproduce the real shapes rather than convenient ones, because both of
the parser's assumptions come from the real shapes: ``raw_text_url`` serves HTML with escaped
entities and anchor tags despite the ``.txt`` name, and the body is hard-wrapped at about 72
columns with four-space paragraph indents, which is the only signal separating a heading from
a paragraph.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from warrant.sources.base import Source
from warrant.sources.federal_register import (
    API,
    FederalRegisterSource,
    MalformedNotice,
    NoticeUnavailable,
    RateLimited,
    clean_raw_text,
    normalize_references,
    split_units,
)

TEXT_URL = "https://www.federalregister.gov/documents/full_text/text/2020/08/10/2020-16823.txt"

#: The wrapper the API actually serves for a ``.txt`` URL: an HTML page whose <pre> holds the
#: notice, with links as anchors and the body entity-escaped.
NOTICE_TEXT = """<html>
<head><title>Federal Register, Volume 85 Issue 154</title></head>
<body><pre>
[Federal Register Volume 85, Number 154 (Monday, August 10, 2020)]
[Rules and Regulations]
[Pages 48096-48102]
From the Federal Register Online via the GPO [<a href="http://www.gpo.gov">www.gpo.gov</a>]
[FR Doc No: 2020-16823]

\x00========================================================================
\x00Rules and Regulations
\x00                                                Federal Register
\x00========================================================================

\x00\x00Federal Register / Vol. 85, No. 154 / Monday, August 10, 2020 / Rules
and Regulations\x00\x00

OFFICE OF PERSONNEL MANAGEMENT

5 CFR Part 630

AGENCY: Office of Personnel Management.

ACTION: Interim rule.

SUMMARY: OPM is issuing interim regulations to assist agencies and
employees responding to the National Emergency Concerning COVID-19.

DATES: The interim regulations are effective on August 10, 2020.
Comments must be received on or before October 9, 2020.

ADDRESSES: You may submit comments by the following method:
    Federal Rulemaking Portal: <a href="http://www.regulations.gov">http://www.regulations.gov</a>.

SUPPLEMENTARY INFORMATION: On March 13, 2020, the President declared a
``National Emergency'' (85 FR 15337).
    OPM is issuing interim regulations to assist such agencies.

Rescinding Regulations

    OPM is rescinding 5 CFR 630.311, which addressed the
&#8220;National Emergency by Reason of Certain Terrorist Attacks.&#8221;

[[Page 48097]]

    The statute requires that annual leave be scheduled in advance.

Waiver of Notice of Proposed Rule Making

    OPM finds good cause to waive the general notice of proposed
rulemaking.

List of Subjects in 5 CFR Part 630

    Government employees.

Office of Personnel Management.
Alexys Stanley,
Regulatory Affairs Analyst.

PART 630--ABSENCE AND LEAVE

0
1. The authority citation for part 630 continues to read as follows:

    Authority:  5 U.S.C. chapter 63.

Subpart C--Annual Leave

0
2. Amend Sec.  630.306 by revising paragraph (a) to read as follows:


Sec.  630.306   [Amended]

0
3. Amend Sec.  630.306 by revising paragraph (b) to read as follows:


Sec.  630.306   Time limit for use of restored annual leave.

    (a) Except as otherwise authorized under paragraphs (b) and (c) of
this section, annual leave restored under 5 U.S.C. 6304(d) must be
scheduled and used not later than the end of the leave year ending 2
years after:
    (1) The date of restoration of the annual leave forfeited because
of administrative error; or
    (2) The date fixed by the agency head.
* * * * *

0
4. Amend Sec.  630.308 as follows:
0
a. Revise the section heading;
0
b. Revise paragraph (a).


Sec.  630.308   Scheduling of annual leave by employees whose work is
essential.

    (a) Annual leave must be scheduled in writing.

[FR Doc. 2020-16823 Filed 8-7-20; 8:45 am]
</pre></body></html>
"""


def record(number: str, *, date: str = "2020-08-10", refs=None, text_url=TEXT_URL,
           abstract: str = "An interim rule about restored annual leave.") -> dict:
    return {
        "document_number": number,
        "title": f"Notice {number}",
        "publication_date": date,
        "html_url": f"https://www.federalregister.gov/documents/{number}",
        "raw_text_url": text_url,
        "abstract": abstract,
        "cfr_references": [{"title": 5, "part": "630", "chapter": None}] if refs is None
        else refs,
        "agencies": [{"raw_name": "OFFICE OF PERSONNEL MANAGEMENT"}],
        "type": "Rule",
        "action": "Interim rule.",
    }


class StubAPI:
    """A transport with no network behind it.

    Matches by substring so a test states the part of the URL it cares about -- the document
    number, ``page=2`` -- instead of reconstructing a query string whose parameter order is
    an implementation detail. Every call is recorded: several tests here are about how many
    requests happen, not about what comes back.
    """

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, list):  # a scripted sequence: fail, fail, then answer
                    value = value.pop(0) if len(value) > 1 else value[0]
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, bytes):
                    return value
                if isinstance(value, str):
                    return value.encode()
                return json.dumps(value).encode()
        raise NoticeUnavailable(url)

    def count(self, fragment: str) -> int:
        return sum(1 for c in self.calls if fragment in c)


def search(*numbers: str, total_pages: int = 1, count: int | None = None) -> dict:
    return {"count": count if count is not None else len(numbers),
            "total_pages": total_pages,
            "results": [{"document_number": n} for n in numbers]}


def source(tmp_path, routes: dict[str, object], **kwargs) -> tuple[FederalRegisterSource,
                                                                   StubAPI]:
    stub = StubAPI(routes)
    # backoff_s=0 keeps the retry path exercised without the 14 s of real sleeping it would
    # otherwise cost; delay_s=0 does the same for the politeness gap.
    src = FederalRegisterSource(cache_dir=tmp_path / "fr", delay_s=0.0, backoff_s=0.0,
                                fetch=stub, **kwargs)
    return src, stub


# -- structure ---------------------------------------------------------------------


def test_units_split_on_the_notice_own_headings():
    """SUMMARY, DATES and each amended section are separate units, not window slices.

    A fixed token window is wrong here in a way that shows up at citation time: the effective
    date lives in one sentence of DATES, and a chunk boundary two hundred tokens away would
    cite it together with the comment address and half the preamble.
    """
    units = split_units(clean_raw_text(NOTICE_TEXT))
    anchors = [u.anchor for u in units]
    for expected in ("summary", "dates", "addresses", "supplementary-information",
                     "sec-630.306", "amdt-1", "part-630-absence-and-leave"):
        assert expected in anchors

    dates = next(u for u in units if u.anchor == "dates")
    assert dates.text.startswith("The interim regulations are effective on August 10, 2020.")
    assert "comments by the following method" not in dates.text  # ADDRESSES did not leak in
    assert dates.heading == "DATES"

    summary = next(u for u in units if u.anchor == "summary")
    assert "COVID-19" in summary.text
    assert "effective on August 10" not in summary.text


def test_the_html_wrapper_and_entities_are_gone():
    """``raw_text_url`` is named .txt and serves HTML; units must not carry the markup."""
    text = clean_raw_text(NOTICE_TEXT)
    assert "<pre>" not in text and "<a href" not in text
    assert "&#8220;" not in text and "“" in text  # entities decoded, not stripped
    assert "http://www.regulations.gov" in text  # the anchor's own text survives


def test_wrapped_lines_rejoin_but_paragraphs_do_not():
    """Hard wrapping is undone; the ``(a)``/``(1)`` paragraph structure is not."""
    units = split_units(clean_raw_text(NOTICE_TEXT))
    section = next(u for u in units if u.anchor == "sec-630.306-2")
    assert "annual leave restored under 5 U.S.C. 6304(d) must be scheduled" in section.text
    lines = section.text.splitlines()
    assert lines[0].startswith("(a)")
    assert any(line.startswith("(1) The date of restoration") for line in lines)


def test_page_markers_become_locators_rather_than_prose():
    """``[[Page 48097]]`` is where a disputed quote is checked, so it is kept as a locator."""
    units = split_units(clean_raw_text(NOTICE_TEXT))
    assert not any("[[Page" in u.text for u in units)
    assert any(u.locator.startswith("p.48097") for u in units)
    assert all(u.locator for u in units)


def test_anchors_are_unique_when_one_section_is_amended_twice():
    """The canned notice amends 630.306 twice, once ``[Amended]``. That is normal, not a bug.

    A citation that matches two units is not a citation, and ``SourceDoc`` refuses to be
    built with duplicates -- so without this the whole notice would be dropped.
    """
    units = split_units(clean_raw_text(NOTICE_TEXT))
    anchors = [u.anchor for u in units]
    assert len(anchors) == len(set(anchors))
    assert anchors.count("sec-630.306") == 1
    assert "sec-630.306-2" in anchors
    # The two amendatory instructions keep the numbers a reader would cite them by.
    assert {"amdt-2-630.306", "amdt-3-630.306"} <= set(anchors)


def test_an_instruction_keeps_its_lettered_clauses_and_its_number():
    """``4. Amend Sec. 630.308 as follows:`` runs straight into ``a.``/``b.`` with no blank
    line, and the section's replacement text follows separately. Treated as one wrapped line
    the instruction became a 400-character heading; counted rather than classified, it was
    absorbed into the block above and four of the seven instructions in 85 FR 48075 vanished.
    """
    units = {u.anchor: u for u in split_units(clean_raw_text(NOTICE_TEXT))}
    instruction = units["amdt-4-630.308"]
    assert instruction.heading == "4. Amend Sec.  630.308 as follows:"
    assert "a. Revise the section heading;" in instruction.text
    assert "b. Revise paragraph (a)." in instruction.text
    # The replacement text is a separate unit, and its heading survived the line wrap whole.
    assert units["sec-630.308"].heading.endswith("whose work is essential.")
    assert units["sec-630.308"].text.startswith("(a) Annual leave")


def test_page_furniture_never_becomes_a_unit():
    """The GPO banner arrives NUL-prefixed and the running head wraps onto a second line.

    Left in, the NULs reach the store inside unit text and the running head becomes a heading
    unit in every notice served by the newer rendering.
    """
    units = split_units(clean_raw_text(NOTICE_TEXT))
    assert not any("\x00" in u.text or "\x00" in u.heading for u in units)
    assert not any("Federal Register / Vol." in u.text for u in units)
    assert not any("and Regulations" == u.heading for u in units)
    assert not any(u.heading and set(u.heading) <= {"=", "_", "-"} for u in units)


def test_a_notice_with_no_headings_still_produces_one_citable_unit():
    units = split_units("    A short correction notice with no structure at all.\n")
    assert len(units) == 1
    assert units[0].anchor == "front-matter"
    assert units[0].text.startswith("A short correction")


# -- references --------------------------------------------------------------------


def test_cfr_references_normalize_to_the_corpus_vocabulary():
    """The part-only form is what the API actually sends; the section form is the join goal."""
    assert normalize_references([
        {"title": 5, "part": "630", "chapter": None, "citation_url": None},
        {"title": 5, "part": "630", "section": "306"},
        {"title": 29, "part": "825", "section": ""},
    ]) == ["5 CFR 630", "5 CFR 630.306", "29 CFR 825"]


def test_a_reference_without_a_part_is_dropped():
    """``5 CFR`` would join a notice to every section of title 5. That is not a join."""
    assert normalize_references([{"title": 5, "part": None, "chapter": "I"}]) == []
    assert normalize_references([]) == []


def test_references_are_deduplicated():
    refs = [{"title": 5, "part": "630"}, {"title": 5, "part": "630", "chapter": "I"}]
    assert normalize_references(refs) == ["5 CFR 630"]


# -- documents ---------------------------------------------------------------------


def test_a_notice_is_valid_from_publication_and_never_expires(tmp_path):
    """A notice is a historical fact. 85 FR 48096 was published in 2020 and still was in 2026.

    Only sources whose content is replaced -- a CFR section -- close their interval. Setting
    valid_to on a notice would hide the reasoning behind an amendment from every query asked
    after the next amendment.
    """
    src, _ = source(tmp_path, {"documents/2020-16823.json": record("2020-16823"),
                               TEXT_URL: NOTICE_TEXT})
    doc = src.to_doc(src.record("2020-16823"))
    assert doc.valid_from == "2020-08-10"
    assert doc.valid_to is None
    assert doc.source == "federal_register"
    assert doc.authority == src.authority
    assert doc.doc_id == "FR-2020-16823"
    assert doc.references == ["5 CFR 630"]
    assert doc.meta["fr_citation"] == "85 FR 48096"
    assert doc.meta["type"] == "Rule"


def test_the_source_satisfies_the_protocol(tmp_path):
    src, _ = source(tmp_path, {})
    assert isinstance(src, Source)
    assert src.name == "federal_register"


def test_a_notice_with_no_served_text_falls_back_to_its_abstract(tmp_path):
    """Corrections and some notices serve no full text. The agency's abstract is still real."""
    src, _ = source(tmp_path, {"documents/2020-00001.json":
                               record("2020-00001", text_url=None)})
    doc = src.to_doc(src.record("2020-00001"))
    assert [u.anchor for u in doc.units] == ["abstract"]
    assert "restored annual leave" in doc.units[0].text


def test_malformed_json_names_the_document(tmp_path):
    """The error has to name the notice; "Expecting value: line 1" names nothing."""
    src, _ = source(tmp_path, {"documents/2020-16823.json": b"<html>404 Not Found</html>"})
    with pytest.raises(MalformedNotice, match="2020-16823"):
        src.record("2020-16823")


# -- the run -----------------------------------------------------------------------


def test_a_missing_document_does_not_abort_the_run(tmp_path):
    """One 404 costs one notice, and is counted. A bare ``continue`` here loses it silently."""
    src, stub = source(tmp_path, {
        "documents.json": search("2020-00001", "2020-00002", "2020-00003"),
        "documents/2020-00001.json": record("2020-00001"),
        "documents/2020-00002.json": NoticeUnavailable("gone"),
        "documents/2020-00003.json": record("2020-00003"),
        TEXT_URL: NOTICE_TEXT,
    })
    docs = list(src.documents())
    assert [d.doc_id for d in docs] == ["FR-2020-00001", "FR-2020-00003"]
    assert list(src.skipped) == ["2020-00002"]
    assert stub.count("documents/2020-00003.json") == 1  # the run went on, it did not restart


def test_a_malformed_notice_does_not_abort_the_run_either(tmp_path):
    src, _ = source(tmp_path, {
        "documents.json": search("2020-00001", "2020-00002"),
        "documents/2020-00001.json": b"{ not json",
        "documents/2020-00002.json": record("2020-00002"),
        TEXT_URL: NOTICE_TEXT,
    })
    assert [d.doc_id for d in src.documents()] == ["FR-2020-00002"]
    assert "2020-00001" in src.skipped


def test_pagination_walks_pages_and_stops_at_the_document_cap(tmp_path):
    """A filter that loses its CFR condition matches millions of documents and no error."""
    routes: dict[str, object] = {
        "page=1&": search(*[f"2020-0000{i}" for i in range(1, 4)], total_pages=9, count=27),
        "page=2&": search(*[f"2020-0001{i}" for i in range(1, 4)], total_pages=9, count=27),
        TEXT_URL: NOTICE_TEXT,
    }
    for page in (1, 2):
        for i in range(1, 4):
            number = f"2020-000{page - 1}{i}"
            routes[f"documents/{number}.json"] = record(number)
    src, stub = source(tmp_path, routes, per_page=3, max_documents=4)

    docs = list(src.documents())
    assert len(docs) == 4
    assert stub.count("documents.json") == 2  # stopped mid-page 2; page 3 was never asked for


def test_pagination_stops_when_the_api_runs_out_of_pages(tmp_path):
    src, stub = source(tmp_path, {
        "page=1&": search("2020-00001", total_pages=1),
        "documents/2020-00001.json": record("2020-00001"),
        TEXT_URL: NOTICE_TEXT,
    }, per_page=1)
    assert len(list(src.documents())) == 1
    assert stub.count("documents.json") == 1


def test_a_notice_amending_several_parts_is_yielded_once(tmp_path):
    """2026-03610 lists eight parts including 630. One printed page, one document id."""
    src, _ = source(tmp_path, {
        "part%5D=630": search("2020-16823"),
        "part%5D=353": search("2020-16823"),
        "documents/2020-16823.json": record("2020-16823"),
        TEXT_URL: NOTICE_TEXT,
    }, cfr_parts=("630", "353"))
    assert [d.doc_id for d in src.documents()] == ["FR-2020-16823"]


# -- caching -----------------------------------------------------------------------


def age(path, hours: float) -> None:
    """Backdate a cache entry. TTL behaviour is untestable in a test that takes 24 hours."""
    stamp = time.time() - hours * 3600.0
    os.utime(path, (stamp, stamp))


def test_a_search_expires_but_a_document_never_does(tmp_path):
    """The distinction the eCFR client got wrong, in one test.

    A search is where tomorrow's notice first appears, so it must be re-asked; a document
    fetched by number is a printed record that cannot change, so re-asking it is pure cost.
    Caching both forever freezes the corpus with no error anywhere -- the next build re-reads
    the pinned search, finds no new document numbers, and downloads nothing, for good.
    """
    src, stub = source(tmp_path, {
        "documents.json": search("2020-16823"),
        "documents/2020-16823.json": record("2020-16823"),
        TEXT_URL: NOTICE_TEXT,
    }, search_ttl_hours=24.0)

    assert len(list(src.documents())) == 1
    assert stub.count("documents.json") == 1
    assert stub.count("documents/2020-16823.json") == 1

    list(src.documents())  # everything is fresh: no request at all
    assert stub.count("documents.json") == 1
    assert stub.count("documents/2020-16823.json") == 1

    for entry in src.cache_dir.iterdir():
        age(entry, 48.0)  # two days on, past the search TTL and well past nothing else

    assert len(list(src.documents())) == 1
    assert stub.count("documents.json") == 2          # re-asked: a new notice may have posted
    assert stub.count("documents/2020-16823.json") == 1  # immutable: still the 2020 record
    assert stub.count(TEXT_URL) == 1


def test_a_failed_refresh_falls_back_to_the_stale_search(tmp_path):
    """An expired TTL is a reason to ask again, not a reason for an offline build to stop."""
    routes: dict[str, object] = {
        "documents.json": search("2020-16823"),
        "documents/2020-16823.json": record("2020-16823"),
        TEXT_URL: NOTICE_TEXT,
    }
    src, stub = source(tmp_path, routes)
    assert len(list(src.documents())) == 1

    for entry in src.cache_dir.iterdir():
        age(entry, 48.0)
    routes["documents.json"] = httpx.ConnectError("no network")
    assert len(list(src.documents())) == 1
    assert stub.count("documents.json") == 1 + src.max_attempts  # it retried, then fell back


def test_a_rate_limited_request_backs_off_and_succeeds(tmp_path):
    """429 is the one failure where waiting is the fix, so it is inside the retry set."""
    src, stub = source(tmp_path, {"documents/2020-16823.json": [
        RateLimited("slow down"), RateLimited("slow down"), record("2020-16823")]})
    assert src.record("2020-16823")["document_number"] == "2020-16823"
    assert stub.count("documents/2020-16823.json") == 3


def test_cache_writes_are_atomic(tmp_path):
    """A truncated cache file is trusted forever by an existence check."""
    src, _ = source(tmp_path, {"documents/2020-16823.json": record("2020-16823")})
    src.record("2020-16823")
    assert (src.cache_dir / "doc-2020-16823.json").exists()
    assert not list(src.cache_dir.glob("*.tmp"))


def test_search_urls_carry_the_filters(tmp_path):
    src, _ = source(tmp_path, {}, cfr_title=5, cfr_parts=("630",),
                    published_since="2017-01-01", term="annual leave", per_page=100)
    url = src.search_url("630", 2)
    assert url.startswith(f"{API}/documents.json?")
    for fragment in ("per_page=100", "page=2", "conditions%5Bcfr%5D%5Btitle%5D=5",
                     "conditions%5Bcfr%5D%5Bpart%5D=630",
                     "conditions%5Bpublication_date%5D%5Bgte%5D=2017-01-01",
                     "conditions%5Bterm%5D=annual+leave", "fields%5B%5D=cfr_references"):
        assert fragment in url


def test_per_page_is_clamped_to_the_api_maximum(tmp_path):
    src, _ = source(tmp_path, {}, per_page=5000)
    assert src.per_page == 1000
