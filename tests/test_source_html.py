"""HTML guidance extraction, entirely offline.

The fixtures below are shaped from the live OPM annual-leave fact sheet rather than
invented: the mega menu really is a ``<div class="usa-nav-container">`` and not a ``<nav>``,
the content column really is ``<main class="usa-layout-docs__main">``, and the page really
does carry two identical ``h3`` headings -- which is why the duplicate-anchor case is tested
against a duplicate heading rather than against a contrived one.

Almost every failure this file guards is silent. Boilerplate that survives does not raise,
it just embeds the site menu; a running-counter anchor scheme does not raise, it just
repoints every stored citation on the next fetch; a flattened bullet list does not raise, it
just turns four eligibility conditions into one unreadable sentence. So the assertions are
mostly about the *absence* of the wrong output.
"""

from __future__ import annotations

import json

import httpx
import pytest

from warrant.sources.base import AUTHORITY_GUIDANCE, KIND_PROSE, KIND_TABLE, Source
from warrant.sources.html import (
    HtmlFetcher,
    HtmlGuidanceSource,
    PageUnavailable,
    doc_id_for,
    extract_citations,
    extract_units,
    page_title,
)

#: A federal page in miniature: skip link, cookie banner, mega menu that is a div rather
#: than a nav, breadcrumbs, share widget, a real <main>, and a footer link farm.
PAGE = b"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Annual Leave - OPM.gov</title>
  <script>var ga = {track: "pageview", section: "pay-leave"};</script>
  <style>.usa-nav-container { display: flex; }</style>
</head>
<body>
  <a class="usa-skipnav" href="#main-content">Skip to main content</a>
  <div class="usa-banner__header">An official website of the United States government</div>
  <div id="cookie-consent-banner">We use cookies to understand how you use this site.
    Accept all cookies to continue.</div>
  <div class="usa-nav-container">
    <ul class="usa-nav__primary">
      <li><a href="/policy-data-oversight/">Policy</a></li>
      <li><a href="/policy-data-oversight/pay-leave/">Pay &amp; Leave</a></li>
      <li><a href="/retirement-center/">Retirement</a></li>
      <li><a href="/healthcare-insurance/">Healthcare</a></li>
    </ul>
  </div>
  <ol class="breadcrumb-list">
    <li><a href="/">Home</a></li><li><a href="/pay-leave/">Pay &amp; Leave</a></li>
  </ol>
  <main id="main-content" class="usa-layout-docs__main">
    <h1>Annual Leave</h1>
    <h2 id="entitlement">Annual Leave Entitlement</h2>
    <p>An employee may use annual leave for vacations, rest and relaxation, and personal
       business or emergencies. The amount accrued depends on years of service under
       5 U.S.C. 6303 and is capped by 5 CFR 630.201.</p>
    <p>Employees may carry over a ceiling of unused annual leave; see &#167; 630.306 for the
       treatment of leave forfeited at the end of the leave year.</p>
    <table class="usa-table">
      <tr><th>Employee Type</th><th>Less than 3 years</th><th>3 to 15 years</th></tr>
      <tr><td>Full-time</td><td>4 hours per pay period</td><td>6 hours per pay period</td></tr>
      <tr><td>Part-time</td><td>1 hour per 20 hours</td><td>1 hour per 13 hours</td></tr>
    </table>
    <h3 id="scheduling">Scheduling of Annual Leave</h3>
    <p>Employees and supervisors are mutually responsible for scheduling leave. An employee
       must schedule use-or-lose leave before the third biweekly pay period.</p>
    <ul>
      <li>Submit a written request before the deadline.</li>
      <li>Obtain supervisory approval in advance.</li>
      <li>Reschedule cancelled leave within the same leave year.</li>
    </ul>
    <dl>
      <dt>Leave year</dt>
      <dd>The period beginning with the first full biweekly pay period in a calendar year.</dd>
      <dt>Use or lose leave</dt>
      <dd>Annual leave in excess of the employee's applicable ceiling.</dd>
    </dl>
    <div class="addthis_toolbox share-buttons">
      <a href="#">Share on Facebook</a><a href="#">Share on X</a><a href="#">Print</a>
    </div>
    <h3 id="restoration">Restoration of Forfeited Leave</h3>
    <p>Forfeited leave may be restored under 5 U.S.C. 6304(d) when the forfeiture resulted
       from an exigency of the public business.</p>
  </main>
  <footer class="usa-footer">
    <ul class="usa-footer__nav">
      <li><a href="/about/">About OPM</a></li><li><a href="/foia/">FOIA</a></li>
      <li><a href="/privacy/">Privacy Policy</a></li>
      <li><a href="/accessibility/">Accessibility</a></li>
    </ul>
    <p>U.S. Office of Personnel Management, 1900 E Street NW, Washington, DC 20415</p>
  </footer>
</body>
</html>
"""

#: The same content with no landmark of any kind -- an older template. The density heuristic
#: has to find the content column on its own here.
NO_MAIN = b"""<html><body>
  <div class="usa-nav-container"><ul>
    <li><a href="/a/">Policy</a></li><li><a href="/b/">Pay</a></li>
    <li><a href="/c/">Retirement</a></li><li><a href="/d/">Insurance</a></li>
    <li><a href="/e/">Suitability</a></li><li><a href="/f/">Veterans</a></li>
  </ul></div>
  <div id="wrapper"><div class="content-column">
    <h2>Sick Leave Entitlement</h2>
    <p>A full-time employee earns four hours of sick leave for each biweekly pay period,
       without limitation on the amount that may be accumulated, under 5 CFR 630.401.</p>
    <p>Sick leave may be used for personal medical needs, family care, bereavement, and
       purposes relating to the adoption of a child.</p>
  </div></div>
</body></html>
"""

#: Two identical headings, exactly as the live fact sheet has them.
DUPLICATE_HEADINGS = b"""<html><body><main>
  <h3>Non-Federal Service or Uniformed Service</h3>
  <p>A newly-appointed employee may receive service credit for prior non-Federal service
     that directly relates to the duties of the position being filled.</p>
  <h3>Non-Federal Service or Uniformed Service</h3>
  <p>Credit may also be granted for active duty uniformed service that is directly related
     to the duties of the position being filled by the agency.</p>
</main></body></html>
"""


def units_by_anchor(raw: bytes) -> dict[str, object]:
    return {u.anchor: u for u in extract_units(raw, doc_id="fact-sheets-annual-leave")}


# -- boilerplate -----------------------------------------------------------------


def test_boilerplate_is_removed():
    """No unit may carry menu, banner, breadcrumb, share or footer text.

    Asserted as substrings over the whole extraction rather than per unit: chrome that
    survives does not land anywhere predictable, it lands wherever the template put it.
    """
    body = "\n".join(u.text for u in extract_units(PAGE, doc_id="d"))
    for chrome in ("Skip to main content", "cookies", "official website",
                   "Retirement", "Healthcare", "Home", "Share on Facebook",
                   "FOIA", "Privacy Policy", "1900 E Street", "var ga", "display: flex"):
        assert chrome not in body, f"boilerplate survived: {chrome!r}"


def test_real_content_survives():
    body = "\n".join(u.text for u in extract_units(PAGE, doc_id="d"))
    assert "vacations, rest and relaxation" in body
    assert "exigency of the public business" in body


def test_density_fallback_finds_the_content_column_without_a_landmark():
    """No <main>, no [role=main], no <article>: density has to do the work."""
    units = extract_units(NO_MAIN, doc_id="sick-leave")
    assert units, "density fallback produced nothing"
    body = "\n".join(u.text for u in units)
    assert "four hours of sick leave" in body
    assert "Retirement" not in body and "Veterans" not in body


# -- structure -------------------------------------------------------------------


def test_headings_start_separate_units():
    anchors = units_by_anchor(PAGE)
    assert "annual-leave-entitlement" in anchors
    assert "scheduling-of-annual-leave" in anchors
    assert "restoration-of-forfeited-leave" in anchors
    # A citation must point at a topic, not at the page: the restoration text must not have
    # been swept into the entitlement unit.
    assert "exigency" not in anchors["annual-leave-entitlement"].text
    assert "exigency" in anchors["restoration-of-forfeited-leave"].text
    assert anchors["scheduling-of-annual-leave"].heading == "Scheduling of Annual Leave"


def test_table_is_one_unit_serialised_one_row_per_line():
    tables = [u for u in extract_units(PAGE, doc_id="d") if u.kind == KIND_TABLE]
    assert len(tables) == 1
    rows = tables[0].text.split("\n")
    assert len(rows) == 3
    assert rows[0] == "Employee Type | Less than 3 years | 3 to 15 years"
    assert rows[1] == "Full-time | 4 hours per pay period | 6 hours per pay period"
    # The table belongs to the section it sits in, and is addressed under that section.
    assert tables[0].anchor == "annual-leave-entitlement-t1"


def test_list_items_stay_on_separate_lines():
    """Three eligibility steps, not one run-on sentence."""
    text = units_by_anchor(PAGE)["scheduling-of-annual-leave"].text
    lines = text.split("\n")
    assert "Submit a written request before the deadline." in lines
    assert "Obtain supervisory approval in advance." in lines
    assert "Reschedule cancelled leave within the same leave year." in lines


def test_definition_list_terms_and_definitions_stay_on_separate_lines():
    lines = units_by_anchor(PAGE)["scheduling-of-annual-leave"].text.split("\n")
    assert "Leave year" in lines
    assert "Use or lose leave" in lines
    assert any(line.startswith("The period beginning") for line in lines)


def test_prose_units_are_prose_kind():
    assert units_by_anchor(PAGE)["scheduling-of-annual-leave"].kind == KIND_PROSE


# -- anchors ---------------------------------------------------------------------


def test_anchors_are_stable_across_refetches_and_do_not_renumber():
    """Inserting a paragraph must not move any anchor below it.

    This is the property a running counter cannot have, and its failure is invisible: every
    stored citation keeps resolving, just to different text.
    """
    before = [u.anchor for u in extract_units(PAGE, doc_id="d")]
    edited = PAGE.replace(
        b"<h3 id=\"scheduling\">",
        b"<p>An agency may approve advanced annual leave in limited circumstances.</p>\n"
        b"    <h3 id=\"scheduling\">",
    )
    after = [u.anchor for u in extract_units(edited, doc_id="d")]
    assert before == after


def test_duplicate_headings_get_unique_stable_anchors():
    units = extract_units(DUPLICATE_HEADINGS, doc_id="d")
    anchors = [u.anchor for u in units]
    assert anchors == ["non-federal-service-or-uniformed-service",
                       "non-federal-service-or-uniformed-service-2"]
    assert len(anchors) == len(set(anchors))


def test_heading_id_becomes_a_deep_link_when_a_base_url_is_given():
    units = {u.anchor: u for u in extract_units(PAGE, doc_id="d", base_url="https://x.gov/fs/")}
    assert units["scheduling-of-annual-leave"].locator == "https://x.gov/fs/#scheduling"


# -- citations -------------------------------------------------------------------


def test_cfr_and_usc_citations_are_extracted_and_normalised():
    text = ("See 5 U.S.C. 6304(d) and 5 C.F.R. § 630.306, plus 5 CFR part 630 "
            "and 5 USC 6329a for details.")
    assert extract_citations(text) == [
        "5 U.S.C. 6304(d)", "5 CFR 630.306", "5 CFR 630", "5 U.S.C. 6329a",
    ]


def test_bare_section_symbol_inherits_the_title_of_the_nearest_cfr_citation():
    assert extract_citations("Under 45 CFR 46.101, and later § 46.116(a), consent ...") == [
        "45 CFR 46.101", "45 CFR 46.116(a)",
    ]
    # With nothing to inherit from, Title 5 is assumed -- the whole corpus is Title 5.
    assert extract_citations("as described in § 630.306") == ["5 CFR 630.306"]


def test_a_qualified_citation_is_not_counted_twice_by_the_bare_pass():
    assert extract_citations("see 5 CFR § 630.306 for the rule") == ["5 CFR 630.306"]


def test_paragraph_designators_are_kept():
    """Dropping "(d)" would lose the one detail saying which rule is being interpreted."""
    assert extract_citations("5 U.S.C. 6304(d)(1)(A)") == ["5 U.S.C. 6304(d)(1)(A)"]


def test_citations_are_deduplicated_in_document_order():
    text = "5 CFR 630.306 ... 5 U.S.C. 6304(d) ... 5 CFR 630.306 again"
    assert extract_citations(text) == ["5 CFR 630.306", "5 U.S.C. 6304(d)"]


# -- malformed input -------------------------------------------------------------


@pytest.mark.parametrize("raw", [
    b"",
    b"   \n\t ",
    b"\x00\xff\xfe not html at all",
    b"<<<< not html >>> &&&",
    b"<html><body></body></html>",
    b"<html><body><div class=\"usa-nav-container\"><a href=\"/\">Home</a></div></body></html>",
])
def test_garbage_yields_zero_units_without_raising(raw: bytes):
    """lxml.html is lenient, so the danger is not an exception -- it is a citable unit
    minted from salvaged junk or from a page that is nothing but chrome."""
    assert extract_units(raw, doc_id="junk") == []


def test_page_title_does_not_raise_on_garbage():
    assert page_title(b"") == ""


def test_page_title_prefers_the_h1_over_the_site_template_title():
    assert page_title(PAGE) == "Annual Leave"


# -- source --------------------------------------------------------------------


class _Transport(httpx.BaseTransport):
    """Serves fixtures; counts requests so cache behaviour is observable."""

    def __init__(self, pages: dict[str, bytes]):
        self.pages = pages
        self.calls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        body = self.pages.get(url)
        if body is None:
            return httpx.Response(404, request=request)
        return httpx.Response(200, content=body, request=request)


@pytest.fixture
def serve(monkeypatch):
    def install(pages: dict[str, bytes]) -> _Transport:
        transport = _Transport(pages)

        def fake_get(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.get(url, **{k: v for k, v in kwargs.items()
                                          if k in ("headers", "timeout")})

        monkeypatch.setattr("warrant.sources.html.httpx.get", fake_get)
        return transport
    return install


URL = "https://www.opm.gov/policy-data-oversight/pay-leave/fact-sheets/annual-leave/"


def test_doc_id_is_the_last_two_path_segments():
    assert doc_id_for(URL) == "fact-sheets-annual-leave"


def test_source_satisfies_the_protocol(tmp_path):
    src = HtmlGuidanceSource(cache_dir=tmp_path, urls=(URL,))
    assert isinstance(src, Source)
    assert src.name == "opm"
    assert src.authority == AUTHORITY_GUIDANCE


def test_documents_yields_a_parsed_doc_with_references(tmp_path, serve):
    serve({URL: PAGE})
    docs = list(HtmlGuidanceSource(cache_dir=tmp_path, urls=(URL,)).documents())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "opm" and doc.authority == AUTHORITY_GUIDANCE
    assert doc.title == "Annual Leave"
    assert doc.doc_id == "fact-sheets-annual-leave"
    # The join to the regulation is the entire reason guidance is ingested.
    assert "5 CFR 630.306" in doc.references
    assert "5 U.S.C. 6304(d)" in doc.references
    assert doc.meta["sha256"] and len(doc.meta["sha256"]) == 64
    assert doc.valid_to is None  # live guidance has no known end date


def test_a_404_is_skipped_rather_than_raised(tmp_path, serve, caplog):
    missing = "https://www.opm.gov/policy-data-oversight/fact-sheets/retired/"
    serve({URL: PAGE})
    with caplog.at_level("WARNING"):
        docs = list(HtmlGuidanceSource(cache_dir=tmp_path, urls=(missing, URL)).documents())
    assert [d.doc_id for d in docs] == ["fact-sheets-annual-leave"]
    assert any("404" in r.getMessage() for r in caplog.records)


def test_a_page_with_no_content_is_skipped(tmp_path, serve):
    empty = "https://www.opm.gov/policy-data-oversight/fact-sheets/stub/"
    serve({empty: b"<html><body><footer>OPM</footer></body></html>"})
    assert list(HtmlGuidanceSource(cache_dir=tmp_path, urls=(empty,)).documents()) == []


# -- cache ---------------------------------------------------------------------


def test_a_fresh_cache_entry_is_not_refetched(tmp_path, serve):
    transport = serve({URL: PAGE})
    fetcher = HtmlFetcher(tmp_path, delay_s=0.0)
    first = fetcher.fetch(URL)
    second = fetcher.fetch(URL)
    assert len(transport.calls) == 1
    assert second.from_cache and second.sha256 == first.sha256


def test_an_expired_ttl_refetches_and_reports_a_changed_hash(tmp_path, serve, caplog):
    """A guidance page is edited in place, so the hash is the only change signal there is."""
    transport = serve({URL: PAGE})
    fetcher = HtmlFetcher(tmp_path, delay_s=0.0, ttl_hours=0.0)
    first = fetcher.fetch(URL)
    transport.pages[URL] = PAGE.replace(b"vacations, rest", b"vacations, recreation, rest")
    with caplog.at_level("INFO"):
        second = fetcher.fetch(URL)
    assert len(transport.calls) == 2
    assert second.sha256 != first.sha256
    assert any("changed since" in r.getMessage() for r in caplog.records)
    meta = next(tmp_path.glob("*.json"))
    assert json.loads(meta.read_text())["sha256"] == second.sha256


def test_a_404_raises_page_unavailable_rather_than_being_cached(tmp_path, serve):
    serve({})
    with pytest.raises(PageUnavailable):
        HtmlFetcher(tmp_path, delay_s=0.0).fetch(URL)
    assert list(tmp_path.glob("*.html")) == []


def test_a_failed_refresh_falls_back_to_the_stale_copy(tmp_path, serve, monkeypatch):
    """An expired TTL is a reason to ask again, not a reason for an offline build to stop."""
    serve({URL: PAGE})
    fetcher = HtmlFetcher(tmp_path, delay_s=0.0, ttl_hours=0.0)
    first = fetcher.fetch(URL)

    def boom(url, **kwargs):
        raise httpx.ConnectError("network is down")

    monkeypatch.setattr("warrant.sources.html.httpx.get", boom)
    again = fetcher.fetch(URL)
    assert again.from_cache and again.sha256 == first.sha256
