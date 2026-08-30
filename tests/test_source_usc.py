"""USLM (US Code) XML into statute units.

The fragments below are hand-written from release point 119-102 of 5 U.S.C. 6304 and 6307
-- element names, namespace declaration, attribute spellings and the OLRC footnote markup
are transcribed from the real file, then cut down so the assertions stay readable. Offline
on purpose: a parser test that needs uscode.house.gov is a test that stops running.
"""

from __future__ import annotations

import pytest

from warrant.sources.base import AUTHORITY_STATUTE, KIND_TABLE, SourceDoc
from warrant.sources.usc import (
    ReleasePoint,
    UscClient,
    UscConfig,
    UscParseError,
    UscSource,
    parse_release_point,
    parse_uslm,
    read_section,
    title_code,
)

#: 5 U.S.C. 6304, abridged: subsection (a) plain, (b) with a chapeau and an (a)(1)(A)(i)
#: chain under it, (c) with a closing continuation, and (d) with no words of its own.
SEC6304 = b"""<?xml version="1.0" encoding="UTF-8"?>
<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0"
        xmlns:dcterms="http://purl.org/dc/terms/" identifier="/us/usc/t5">
  <meta><dcterms:created>2026-07-16T07:46:32</dcterms:created></meta>
  <main>
    <title identifier="/us/usc/t5"><num value="5">Title 5&#8212;</num>
      <chapter identifier="/us/usc/t5/ptIII/spE/ch63"><num value="63">CHAPTER 63&#8212;</num>
        <heading>LEAVE</heading>
        <section identifier="/us/usc/t5/s6304"><num value="6304">&#167; 6304.</num>
          <heading> Annual leave; accumulation</heading>
          <subsection identifier="/us/usc/t5/s6304/a"><num value="a">(a)</num>
            <content> Except as provided by subsections (b), (d), (e), (f), and (g) of this
            section, annual leave provided by <ref href="/us/usc/t5/s6303">section 6303 of
            this title</ref>, which is not used, accumulates.</content>
          </subsection>
          <subsection identifier="/us/usc/t5/s6304/b"><num value="b">(b)</num>
            <chapeau> Annual leave not used by an employee stationed outside the United
            States accumulates until it totals not more than 45 days:</chapeau>
            <paragraph identifier="/us/usc/t5/s6304/b/1"><num value="1">(1)</num>
              <content> Individuals directly recruited from the United States.</content>
            </paragraph>
            <paragraph identifier="/us/usc/t5/s6304/b/2"><num value="2">(2)</num>
              <chapeau> Individuals employed locally but&#8212;</chapeau>
              <subparagraph identifier="/us/usc/t5/s6304/b/2/A"><num value="A">(A)</num>
                <clause identifier="/us/usc/t5/s6304/b/2/A/i"><num value="i">(i)</num>
                  <content> who were originally recruited from the United States;</content>
                </clause>
                <clause identifier="/us/usc/t5/s6304/b/2/A/ii"><num value="ii">(ii)</num>
                  <content> who have been in substantially continuous employment; and</content>
                </clause>
              </subparagraph>
            </paragraph>
          </subsection>
          <subsection identifier="/us/usc/t5/s6304/c"><num value="c">(c)</num>
            <chapeau> Annual leave in excess of the amount allowable&#8212;</chapeau>
            <paragraph identifier="/us/usc/t5/s6304/c/1"><num value="1">(1)</num>
              <content> under subsection (a) or (b) which was accumulated under earlier
              statute; or</content>
            </paragraph>
            <continuation>remains to the credit of the employee until used.</continuation>
          </subsection>
          <subsection identifier="/us/usc/t5/s6304/d"><num value="d">(d)</num>
            <paragraph identifier="/us/usc/t5/s6304/d/1"><num value="1">(1)</num>
              <content> Annual leave lost by operation of <ref
              href="/us/usc/t5/s5562/a">section 5562(a) of this title</ref> shall be
              restored.</content>
            </paragraph>
          </subsection>
          <sourceCredit>(<ref href="/us/pl/89/554">Pub. L. 89&#8211;554</ref>,
            <date date="1966-09-06">Sept. 6, 1966</date>,
            <ref href="/us/stat/80/519">80 Stat. 519</ref>;
            <ref href="/us/pl/111/282/s2/b">Pub. L. 111&#8211;282, &#167;&#8239;2(b)</ref>,
            <date date="2010-10-15">Oct. 15, 2010</date>,
            <ref href="/us/stat/124/3038">124 Stat. 3038</ref>.)</sourceCredit>
          <notes type="uscNote">
            <note topic="amendments">
              <p>1973&#8212;Subsec. (b). <ref href="/us/pl/93/181">Pub. L. 93&#8211;181</ref>
              substituted text referring to <ref href="/us/usc/t5/s9999">section 9999</ref>.</p>
              <quotedContent>
                <section identifier="/us/usc/t5/s6304"><num value="6304">&#167; 6304.</num>
                  <heading> Repealed text that is not the law</heading>
                  <content>This is quoted, not enacted.</content>
                </section>
              </quotedContent>
            </note>
          </notes>
        </section>
      </chapter>
    </title>
  </main>
</uscDoc>
"""

#: 5 U.S.C. 6307(d) as OLRC actually publishes it: a footnote element and its marker sit
#: inside the chapeau, mid-sentence, and a bare <sup> carries the marker in the sibling.
SEC6307 = b"""<section xmlns="http://xml.house.gov/schemas/uslm/1.0"
         identifier="/us/usc/t5/s6307">
  <num value="6307">&#167; 6307.</num><heading> Sick leave; accrual and accumulation</heading>
  <subsection identifier="/us/usc/t5/s6307/d"><num value="d">(d)</num>
    <chapeau>(1) <ref class="footnoteRef" idref="fn1">1</ref><note type="footnote"
      id="fn1"><num>1</num> So in original. Probably should be &#8220;(e)(1)&#8221;.</note>
      For the purpose of this subsection, the term &#8220;family member&#8221; shall have
      such meaning as the Office shall prescribe.</chapeau>
  </subsection>
  <subsection identifier="/us/usc/t5/s6307/d"><num value="d">(d)</num>
    <content><sup>1</sup> Leave under this subsection is in addition to leave under
      subsection (a).</content>
  </subsection>
</section>
"""

#: The same hierarchy written flat: designators in prose, no level elements, nothing nested.
#: This is how pre-USLM conversions and hand-repaired sections arrive, and it is the only
#: case where the anchor has to be recovered by sequence continuity instead of by nesting.
FLAT = b"""<section xmlns="http://xml.house.gov/schemas/uslm/1.0"
         identifier="/us/usc/t5/s6304">
  <num value="6304">&#167; 6304.</num><heading> Annual leave; accumulation</heading>
  <content>Undesignated opening text.</content>
  <content>(a) Annual leave accumulates as follows.</content>
  <content>(1) Individuals directly recruited.</content>
  <content>(A) who were originally recruited;</content>
  <content>(i) and who remain so employed;</content>
  <content>(ii) or who do not.</content>
  <content>(B) who were temporarily absent.</content>
  <content>(2) Individuals employed locally.</content>
  <content>(b) This subsection applies to leave accrued abroad.</content>
</section>
"""

#: A section number that is not a bare integer, a level with no <num>, and a table.
ODDITIES = b"""<section xmlns="http://xml.house.gov/schemas/uslm/1.0"
         identifier="/us/usc/t5/s552a">
  <num value="552a">&#167; 552a.</num><heading> Records maintained on individuals</heading>
  <subsection identifier="/us/usc/t5/s552a/a"><num value="a">(a)</num>
    <content>Definitions apply as follows.</content>
    <table xmlns="http://www.w3.org/1999/xhtml">
      <tr><th>Grade</th><th>Rate</th></tr>
      <tr><td>GS-1</td><td>$21,986</td></tr>
    </table>
  </subsection>
  <subsection identifier="/us/usc/t5/s552a/b">
    <content>A level the converter could not number.</content>
  </subsection>
</section>
"""


#: A credit OLRC converted without <date> markup, which is how titles other than 5 sometimes
#: arrive. The dates are only in the prose, and one of them is impossible.
UNMARKED_CREDIT = b"""<section xmlns="http://xml.house.gov/schemas/uslm/1.0"
         identifier="/us/usc/t5/s6304">
  <num value="6304">&#167; 6304.</num><heading> Annual leave; accumulation</heading>
  <content>Annual leave which is not used accumulates.</content>
  <sourceCredit>(Pub. L. 89&#8211;554, Sept. 6, 1966, 80 Stat. 519; Pub. L. 114&#8211;328,
    div. A, title XI, &#167; 1104(a), Feb. 31, 2020, 130 Stat. 2447; Pub. L. 117&#8211;103,
    June 15, 2022, 136 Stat. 49.)</sourceCredit>
</section>
"""

#: A section with operative text and a credit that names no date at all. Constructed rather
#: than transcribed: title 5 at 119-102 has none, and the policy for one still has to hold.
UNDATABLE_CREDIT = b"""<section xmlns="http://xml.house.gov/schemas/uslm/1.0"
         identifier="/us/usc/t5/s6399">
  <num value="6399">&#167; 6399.</num><heading> A section nothing can date</heading>
  <content>This text is operative and its credit says when nothing.</content>
  <sourceCredit>(Added Pub. L. 89&#8211;554, 80 Stat. 519.)</sourceCredit>
</section>
"""


def anchors(units):
    return [u.anchor for u in units]


def by_anchor(units):
    return {u.anchor: u for u in units}


# -- hierarchy ---------------------------------------------------------------------


def test_nesting_becomes_dotted_anchors_including_an_a_1_A_chain():
    """(b) -> (b)(2) -> (b)(2)(A) -> (b)(2)(A)(i), the shape a citation is written in."""
    got = anchors(parse_uslm(SEC6304, doc_id="5 U.S.C. 6304"))
    assert got == ["a", "b", "b-1", "b-2", "b-2-A-i", "b-2-A-ii", "c", "c-1", "d-1"]


def test_anchors_are_unique_within_a_section():
    """A citation that matches two units is not a citation."""
    got = anchors(parse_uslm(SEC6304, doc_id="5 U.S.C. 6304"))
    assert len(got) == len(set(got))


def test_a_level_with_no_words_of_its_own_gets_no_unit():
    """6304(d) is a bare designator over its paragraphs. An empty unit at ``d`` would be a
    citable address that quotes nothing."""
    assert "d" not in anchors(parse_uslm(SEC6304, doc_id="5 U.S.C. 6304"))


def test_a_level_keeps_its_chapeau_and_its_closing_continuation_together():
    """6304(c) is one sentence wrapped around its paragraphs, and ``6304#c`` has to return
    the whole of it; the paragraphs keep their own anchors, so nothing is duplicated."""
    unit = by_anchor(parse_uslm(SEC6304, doc_id="5 U.S.C. 6304"))["c"]
    assert unit.text.startswith("Annual leave in excess of the amount allowable")
    assert unit.text.endswith("remains to the credit of the employee until used.")
    assert "accumulated under earlier statute" not in unit.text


def test_the_designator_stack_recovers_a_flat_section():
    """Nesting is the primary signal, but it is not guaranteed by the schema. With every
    designator written into prose, sequence continuity is all that is left -- and it has to
    tell the ninth letter (i) from the first roman numeral, which only order decides."""
    assert anchors(parse_uslm(FLAT, doc_id="5 U.S.C. 6304")) == [
        "p1", "a", "a-1", "a-1-A", "a-1-A-i", "a-1-A-ii", "a-1-B", "a-2", "b",
    ]


def test_a_designator_recovered_from_prose_is_not_left_in_the_prose():
    units = by_anchor(parse_uslm(FLAT, doc_id="5 U.S.C. 6304"))
    assert units["a-1-A"].text == "who were originally recruited;"


def test_a_level_without_a_num_still_gets_a_unique_anchor():
    got = anchors(parse_uslm(ODDITIES, doc_id="5 U.S.C. 552a"))
    assert "a" in got
    assert len(got) == len(set(got))
    assert all(a for a in got)


# -- heading and num ---------------------------------------------------------------


def test_the_heading_is_captured_and_not_duplicated_into_the_body():
    units = parse_uslm(SEC6304, doc_id="5 U.S.C. 6304")
    assert {u.heading for u in units} == {"Annual leave; accumulation"}
    assert not any("Annual leave; accumulation" in u.text for u in units)


def test_num_never_leaks_into_the_body_as_prose():
    """``<num>(a)</num>`` is a sibling of ``<content>``, so a naive itertext() over the
    subsection prefixes the law with its own designator -- and ``§ 6304.`` with it."""
    units = parse_uslm(SEC6304, doc_id="5 U.S.C. 6304")
    for unit in units:
        assert not unit.text.lstrip().startswith("(")
        assert "§" not in unit.text
        assert "6304." not in unit.text


def test_footnote_apparatus_is_not_quoted_as_statute():
    """OLRC's footnotes nest *inside* the chapeau, so they cannot be skipped as siblings.
    'So in original' is the editor speaking, not Congress."""
    units = by_anchor(parse_uslm(SEC6307, doc_id="5 U.S.C. 6307"))
    assert "So in original" not in units["d"].text
    assert units["d"].text.startswith("(1) For the purpose of this subsection")
    # The bare <sup>1</sup> marker in the duplicate (d) is dropped the same way.
    assert units["d.2"].text.startswith("Leave under this subsection")


def test_duplicate_designators_in_the_enacted_text_still_resolve_to_one_unit_each():
    """5 U.S.C. 6307 really does have two subsections (d), with the same OLRC identifier.
    An ugly anchor that matches one paragraph beats a clean one that matches two."""
    got = anchors(parse_uslm(SEC6307, doc_id="5 U.S.C. 6307"))
    assert got == ["d", "d.2"]


# -- namespaces --------------------------------------------------------------------


def test_the_uslm_namespace_is_resolved_rather_than_assumed():
    """USLM is the default namespace, so tags arrive as ``{uri}section`` with no prefix and
    a hard-coded one finds nothing. Moving the schema to a new URI must not break this."""
    moved = SEC6304.replace(b"http://xml.house.gov/schemas/uslm/1.0",
                            b"http://xml.house.gov/schemas/uslm/2.0")
    assert anchors(parse_uslm(moved, doc_id="5 U.S.C. 6304")) == \
        anchors(parse_uslm(SEC6304, doc_id="5 U.S.C. 6304"))


def test_a_prefixed_namespace_declaration_parses_identically():
    prefixed = SEC6307.replace(b'xmlns="http://xml.house.gov/schemas/uslm/1.0"',
                               b'xmlns:uslm="http://xml.house.gov/schemas/uslm/1.0"')
    prefixed = prefixed.replace(b"<section ", b"<uslm:section ")
    prefixed = prefixed.replace(b"</section>", b"</uslm:section>")
    for tag in (b"num", b"heading", b"subsection", b"chapeau", b"content", b"note",
                b"ref", b"sup"):
        prefixed = prefixed.replace(b"<" + tag, b"<uslm:" + tag)
        prefixed = prefixed.replace(b"</" + tag + b">", b"</uslm:" + tag + b">")
    assert anchors(parse_uslm(prefixed, doc_id="5 U.S.C. 6307")) == ["d", "d.2"]


def test_a_table_in_a_different_namespace_is_still_a_table():
    """OLRC embeds XHTML tables inside USLM, so two namespaces are in play at once."""
    tables = [u for u in parse_uslm(ODDITIES, doc_id="5 U.S.C. 552a") if u.kind == KIND_TABLE]
    assert len(tables) == 1
    assert tables[0].text.splitlines() == ["Grade | Rate", "GS-1 | $21,986"]


# -- apparatus ---------------------------------------------------------------------


def test_notes_and_source_credit_are_not_statutory_text():
    units = parse_uslm(SEC6304, doc_id="5 U.S.C. 6304")
    body = " ".join(u.text for u in units)
    assert "Pub. L." not in body
    assert "substituted text" not in body
    assert "This is quoted, not enacted." not in body


def test_a_section_quoted_inside_a_note_is_not_mistaken_for_the_section():
    """An amendment note reproduces the text it inserted, wrapped in <quotedContent>.
    Ingesting that as law would put superseded text in the store with a current date."""
    parsed = read_section_of(SEC6304)
    assert parsed.heading == "Annual leave; accumulation"
    assert parsed.number == "6304"


def read_section_of(xml: bytes):
    from lxml import etree

    from warrant.sources.usc import _parser, _section_element

    root = etree.fromstring(xml, _parser())
    return read_section(_section_element(root), doc_id="test")


# -- references --------------------------------------------------------------------


def test_references_lead_with_the_sections_own_citation():
    parsed = read_section_of(SEC6304)
    assert parsed.references[0] == "5 U.S.C. 6304"


def test_references_carry_the_cross_references_the_operative_text_makes():
    parsed = read_section_of(SEC6304)
    assert "5 U.S.C. 6303" in parsed.references
    assert "5 U.S.C. 5562" in parsed.references


def test_references_exclude_the_enactment_chain_and_the_notes():
    """Pub. L. and Stat. refs say how the text got here, not what it depends on; a ref that
    only a 1973 amendment note makes is not a cross-reference the current section makes."""
    parsed = read_section_of(SEC6304)
    assert not any(r.startswith("Pub") for r in parsed.references)
    assert "5 U.S.C. 9999" not in parsed.references


# -- validity ----------------------------------------------------------------------


def test_the_credit_chain_is_kept_whole_and_the_latest_date_bounds_the_text():
    """The bound is the *last* amendment, not the enactment: everything before it was
    superseded, and only the last one is a date the current text is known to postdate."""
    parsed = read_section_of(SEC6304)
    assert parsed.credit_dates == ("1966-09-06", "2010-10-15")
    assert parsed.last_amended == "2010-10-15"


def test_the_credit_date_comes_from_the_markup_not_from_the_rendered_prose():
    """OLRC renders "Oct. 15, 2010" and marks it up as date="2010-10-15". Reading the
    attribute is exact; reading the prose is the fallback, and it is only a fallback."""
    parsed = read_section_of(SEC6304)
    assert "2010-10-15" not in parsed.source_credit
    assert "2010-10-15" in parsed.credit_dates


def test_a_credit_with_no_date_markup_is_read_from_its_prose():
    """Titles other than 5 arrive converted without <date>. The same dates are still in the
    prose, and an impossible one -- Feb. 31 -- is dropped rather than repaired, because a
    silently corrected date becomes the date the store says a statute took effect."""
    parsed = read_section_of(UNMARKED_CREDIT)
    assert parsed.credit_dates == ("1966-09-06", "2022-06-15")
    assert parsed.last_amended == "2022-06-15"


def test_a_credit_with_no_date_at_all_yields_no_bound():
    parsed = read_section_of(UNDATABLE_CREDIT)
    assert parsed.credit_dates == ()
    assert parsed.last_amended == ""


# -- errors ------------------------------------------------------------------------


def test_malformed_xml_raises_naming_the_section():
    broken = SEC6304.replace(b"</subsection>", b"</subsecton>", 1)
    with pytest.raises(UscParseError) as exc:
        parse_uslm(broken, doc_id="5 U.S.C. 6304")
    assert "5 U.S.C. 6304" in str(exc.value)
    assert "not well-formed" in str(exc.value)


def test_an_empty_document_raises_naming_the_section():
    with pytest.raises(UscParseError, match="5 U.S.C. 6304"):
        parse_uslm(b"   ", doc_id="5 U.S.C. 6304")


def test_a_document_with_no_section_raises_naming_the_section():
    with pytest.raises(UscParseError, match="5 U.S.C. 6304"):
        parse_uslm(b'<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0"/>',
                   doc_id="5 U.S.C. 6304")


def test_a_section_without_a_usable_identifier_raises_naming_it():
    anon = SEC6307.replace(b'identifier="/us/usc/t5/s6307"', b'identifier="nonsense"', 1)
    with pytest.raises(UscParseError, match="6307"):
        parse_uslm(anon, doc_id="5 U.S.C. 6307")


# -- release points and titles -----------------------------------------------------


def test_release_point_is_read_from_the_olrc_download_page():
    page = b"<h3 class='releasepointinformation'>Public Law 119-102 (07/12/2026)</h3>"
    rp = parse_release_point(page)
    assert (rp.congress, rp.law, rp.date, rp.name) == (119, 102, "2026-07-12", "119-102")


def test_an_unrecognisable_download_page_says_so_rather_than_guessing():
    with pytest.raises(UscParseError, match="release point"):
        parse_release_point(b"<html><body>Service unavailable</body></html>")


def test_title_codes_are_zero_padded_and_keep_the_appendix_letter():
    assert (title_code("5"), title_code("5a"), title_code("42")) == ("05", "05a", "42")


def test_a_title_that_is_not_a_title_is_rejected():
    with pytest.raises(ValueError, match="US Code title"):
        title_code("../etc")


# -- the source ---------------------------------------------------------------------


class StubClient(UscClient):
    """The release point, without the 3 MB download."""

    xml: bytes = SEC6304

    def current_release_point(self) -> ReleasePoint:
        return ReleasePoint(congress=119, law=102, date="2026-07-12")

    def title_xml(self, title: str, rp: ReleasePoint) -> bytes:
        return self.xml


def source(tmp_path, *, xml: bytes = SEC6304, **kwargs) -> UscSource:
    config = UscConfig(title="5", cache_dir=tmp_path, **kwargs)
    client = StubClient(cache_dir=tmp_path)
    client.xml = xml
    return UscSource(config=config, client=client)


def test_documents_are_statutes_and_carry_an_open_interval(tmp_path):
    doc = next(iter(source(tmp_path, sections=["6304"]).documents()))
    assert isinstance(doc, SourceDoc)
    assert doc.authority == AUTHORITY_STATUTE
    assert doc.authority_name == "statute"
    assert doc.valid_to is None
    assert doc.meta["release_point"] == "119-102"


def test_a_document_is_addressable_as_a_citation(tmp_path):
    doc = next(iter(source(tmp_path, sections=["6304"]).documents()))
    assert doc.doc_id == "usc-t5-s6304"
    assert doc.title == "5 U.S.C. 6304 - Annual leave; accumulation"
    assert doc.references[0] == "5 U.S.C. 6304"
    assert "title5-section6304" in doc.url
    assert doc.meta["chapter"] == "63"


def test_sections_and_chapters_both_select(tmp_path):
    assert len(list(source(tmp_path, chapters=["63"]).documents())) == 1
    assert len(list(source(tmp_path, sections=["6304"]).documents())) == 1
    assert len(list(source(tmp_path, sections=["6304"], chapters=["63"]).documents())) == 1


def test_a_section_missing_from_the_edition_is_logged_and_skipped(tmp_path, caplog):
    """Sections are repealed, renumbered and transferred between release points. A config
    naming one is out of date; it is not a reason to abandon the other 1,162."""
    with caplog.at_level("WARNING"):
        docs = list(source(tmp_path, sections=["6304", "9999"]).documents())
    assert [d.doc_id for d in docs] == ["usc-t5-s6304"]
    assert "9999" in caplog.text


def test_a_pinned_release_point_is_dated_from_the_document(tmp_path):
    """No public-law date is attached to a pinned release point, so the *edition* date falls
    back to the converter's own <dcterms:created> rather than to today. valid_from is
    unaffected: it comes from the section, not from the edition."""
    doc = next(iter(source(tmp_path, sections=["6304"],
                           release_point="119-102").documents()))
    assert doc.meta["release_point"] == "119-102"
    assert doc.meta["release_point_date"] == "2026-07-16"
    assert doc.valid_from == "2010-10-15"


def test_valid_from_is_the_last_amendment_not_the_edition(tmp_path):
    """The bug this replaced: valid_from was the release-point date, so 5 U.S.C. 6304 --
    in force since 1966 -- claimed to begin on 2026-07-12, and the as-of predicate,
    ``valid_from <= :v`` in SQL, admitted none of it for any date in the 2017-onward corpus
    window. The section is not a statute that began in 2026; it is one that has held since
    at least 2010, and the whole window is inside that."""
    doc = next(iter(source(tmp_path, sections=["6304"]).documents()))
    assert doc.valid_from == "2010-10-15"
    assert doc.valid_from < "2017-01-01"  # corpus.history_floor in configs/default.yaml
    assert doc.meta["valid_from_basis"] == "source_credit"


def test_the_edition_is_still_recoverable_from_a_dated_document(tmp_path):
    """valid_from no longer says which OLRC file the text was quoted from, so something else
    has to. ``snapshot`` is what corpus/ingest.py writes into every chunk's source_snapshot."""
    doc = next(iter(source(tmp_path, sections=["6304"]).documents()))
    assert doc.meta["release_point"] == "119-102"
    assert doc.meta["release_point_date"] == "2026-07-12"
    assert doc.meta["snapshot"] == "2026-07-12"
    assert doc.meta["source_credit_dates"] == "1966-09-06 2010-10-15"


def test_a_section_that_cannot_be_dated_is_dropped_and_counted(tmp_path, caplog):
    """The interesting case, and the one the old code got wrong twice over. Falling back to
    the edition date hides the section from every dated query in the window, and inventing an
    earlier bound lets undatable text answer as current. Neither failure is visible from the
    outside, so the section is dropped somewhere the count can be read."""
    src = source(tmp_path, xml=UNDATABLE_CREDIT)
    with caplog.at_level("WARNING"):
        docs = list(src.documents())
    assert docs == []
    assert src.undated == ["5 U.S.C. 6399"]
    assert "no datable source credit" in caplog.text


def test_the_undated_list_reports_the_last_run_not_the_process(tmp_path):
    src = source(tmp_path, xml=UNDATABLE_CREDIT)
    list(src.documents())
    assert src.undated
    src.client.xml = SEC6304
    list(src.documents())
    assert src.undated == []


def test_a_malformed_release_point_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="119-102"):
        list(source(tmp_path, release_point="latest").documents())


def test_an_error_page_served_as_a_release_point_is_not_parsed_as_one(tmp_path):
    """govinfo answers a missing USLM granule with HTTP 200 and 44 KB of HTML. A source that
    trusts the status code caches the error page and reports zero sections, not a failure."""
    client = UscClient(cache_dir=tmp_path)
    client._fetch = lambda url: b"<!DOCTYPE html><html>Page Not Found</html>"
    with pytest.raises(UscParseError, match="not a zip"):
        client.title_xml("5", ReleasePoint(congress=119, law=102, date="2026-07-12"))
