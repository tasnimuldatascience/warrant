"""Parsing eCFR XML into retrieval and citation units."""

from __future__ import annotations

from warrant.corpus.parse import parse_sections, section_index

PART = b"""
<ECFR>
  <DIV5 N="630" TYPE="PART">
    <DIV6 N="A" TYPE="SUBPART">
      <HEAD>Subpart A&#8212;General Provisions</HEAD>
      <DIV8 N="630.101" TYPE="SECTION">
        <HEAD>&#167; 630.101 Purpose.</HEAD>
        <P>This subpart states the conditions governing the granting of leave.</P>
      </DIV8>
    </DIV6>
    <DIV6 N="L" TYPE="SUBPART">
      <HEAD>Subpart L&#8212;Family and Medical Leave</HEAD>
      <DIV8 N="630.1203" TYPE="SECTION">
        <HEAD>&#167; 630.1203 Leave entitlement.</HEAD>
        <XREF ID="20200810">Link to an amendment published at 85 FR 48090, Aug. 10, 2020.</XREF>
        <P>(a) An employee shall be entitled to a total of 12 administrative workweeks.</P>
        <P>(b)(2) The entitlement expires at the end of the 12-month period.</P>
        <P>An unlabelled concluding paragraph.</P>
      </DIV8>
    </DIV6>
  </DIV5>
</ECFR>
"""


def test_sections_are_found_with_identifiers():
    idx = section_index(PART)
    assert set(idx) == {"630.101", "630.1203"}


def test_heading_drops_the_section_number():
    """The number is already the identifier. Repeating it in the heading pollutes every
    lexical index entry with a token the query never contains."""
    idx = section_index(PART)
    assert idx["630.1203"].heading == "Leave entitlement"
    assert idx["630.101"].heading == "Purpose"


def test_subpart_is_carried_on_the_section():
    """Applicability is often scoped at subpart level, so the section must know its parent."""
    idx = section_index(PART)
    assert idx["630.101"].subpart == "A"
    assert idx["630.1203"].subpart == "L"


def test_paragraph_anchors_follow_the_designators():
    paras = section_index(PART)["630.1203"].paragraphs
    assert [p.anchor for p in paras] == ["a", "b-2", "p3"]


def test_apparatus_is_stripped_during_parse():
    idx = section_index(PART)
    assert "Link to an amendment" not in idx["630.1203"].text
    assert all("Link to an amendment" not in p.text for p in idx["630.1203"].paragraphs)


def test_section_text_includes_all_paragraphs():
    s = section_index(PART)["630.1203"]
    assert "12 administrative workweeks" in s.text
    assert "unlabelled concluding paragraph" in s.text


def test_parse_is_stable_across_calls():
    assert [s.identifier for s in parse_sections(PART)] == [
        s.identifier for s in parse_sections(PART)
    ]
