"""Parsing eCFR XML into retrieval and citation units."""

from __future__ import annotations

import time

import pytest

from warrant.corpus.parse import CorpusParseError, parse_sections, section_index

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


NESTED = b"""
<DIV8 N="550.112" TYPE="SECTION">
  <HEAD>&#167; 550.112 Computation of overtime work.</HEAD>
  <P>The computation of the amount of overtime work is subject to the following.</P>
  <P>(a) Time spent in principal activities.</P>
  <P>(1) An employee shall be compensated for every minute.</P>
  <P>(2) A quarter of an hour shall be the largest fraction.</P>
  <P>(b) Time spent in preshift activities.</P>
  <P>(1) (i) If the head of a department reasonably determines.</P>
  <P>(ii) If the time spent in a preshift activity is compensable.</P>
  <P>(2) A preshift activity that is not closely related.</P>
  <P>(c) Leave with pay.</P>
</DIV8>
"""


def test_nested_designators_produce_hierarchical_anchors():
    """(ii) after (b)(1)(i) is the second roman numeral, not the ninth letter. Type alone
    cannot tell those apart; only what came before can."""
    anchors = [p.anchor for p in section_index(NESTED)["550.112"].paragraphs]
    assert anchors == ["p1", "a", "a-1", "a-2", "b", "b-1-i", "b-1-ii", "b-2", "c"]


def test_anchors_are_unique_within_a_section():
    """A citation that matches four paragraphs is not a citation. Before the designator
    stack was tracked, 13% of addresses in the corpus were ambiguous."""
    anchors = [p.anchor for p in section_index(NESTED)["550.112"].paragraphs]
    assert len(anchors) == len(set(anchors))


def test_sibling_letters_close_deeper_levels():
    """(c) after (b)(1)(ii) must reset to the top level, not nest under it."""
    anchors = [p.anchor for p in section_index(NESTED)["550.112"].paragraphs]
    assert anchors[-1] == "c"


# -- hostile and malformed input ---------------------------------------------------
#
# `make build` walks 26 parts and roughly 200 cached snapshots. A parser that aborts on one
# of them with an lxml XMLSyntaxError naming a line and column in an anonymous string leaves
# no way to find the file except bisecting the cache by hand -- so every failure below has to
# name its source. The XML settings are stated explicitly in parse.py rather than inherited
# from libxml2's defaults, and these tests are what makes that statement a guarantee.

SOURCE = "title 5 part 630 as of 2020-08-10"


def test_empty_input_names_its_source():
    with pytest.raises(CorpusParseError) as exc:
        parse_sections(b"", source=SOURCE)
    assert SOURCE in str(exc.value)
    assert "empty" in str(exc.value)


def test_whitespace_only_input_is_treated_as_empty():
    with pytest.raises(CorpusParseError):
        parse_sections(b"   \n  ", source=SOURCE)


def test_truncated_input_names_its_source():
    """The shape a half-written cache file takes: a fetch interrupted by a Ctrl-C or a
    closed lid, trusted forever afterwards because the cache only checks for existence."""
    with pytest.raises(CorpusParseError) as exc:
        parse_sections(PART[:200], source=SOURCE)
    assert SOURCE in str(exc.value)
    assert "well-formed" in str(exc.value)


def test_an_undefined_entity_fails_with_a_useful_message():
    """`&nbsp;` is undefined in XML and entirely plausible in a document that has passed
    through an HTML-aware editor. It used to raise XMLSyntaxError straight to the caller and
    abort the whole build, naming no file."""
    xml = PART.replace(b"<P>This subpart", b"<P>&nbsp;This subpart")
    with pytest.raises(CorpusParseError) as exc:
        parse_sections(xml, source=SOURCE)
    assert SOURCE in str(exc.value)
    assert "nbsp" in str(exc.value)


BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
  <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
  <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
  <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
  <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<ECFR><DIV5 N="630" TYPE="PART">
  <DIV8 N="630.101" TYPE="SECTION"><HEAD>&#167; 630.101 Purpose.</HEAD>
  <P>&lol9;</P></DIV8>
</DIV5></ECFR>
"""


def test_a_billion_laughs_payload_is_refused_quickly():
    """`&lol9;` is a gigabyte of text once expanded. It has to be refused in bounded time and
    bounded memory -- and to say which snapshot it came from, since a build that dies of an
    OOM naming no file is the same investigation as a build that dies of a syntax error
    naming no file."""
    started = time.monotonic()
    with pytest.raises(CorpusParseError) as exc:
        parse_sections(BILLION_LAUGHS, source=SOURCE)
    assert time.monotonic() - started < 5
    assert SOURCE in str(exc.value)


def test_entity_definitions_are_never_expanded():
    """The reason the payload above costs nothing: with resolution off, a defined entity is
    left as written rather than substituted. Below libxml2's amplification limit there is no
    error to raise, so the guarantee has to be asserted on the output."""
    xml = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x "expanded">]>'
           b'<DIV8 N="630.101" TYPE="SECTION"><HEAD>&#167; 630.101 Purpose.</HEAD>'
           b"<P>&x;</P></DIV8>")
    section = section_index(xml)["630.101"]
    assert "expanded" not in section.text


def test_the_predefined_entities_and_character_references_still_resolve():
    """Turning entity resolution off must not cost the corpus its section symbols: 130,560
    `&quot;` and 21,012 `&amp;` in the cached snapshots, and every heading numbered with a
    `&#167;`."""
    xml = (b'<DIV8 N="630.101" TYPE="SECTION"><HEAD>&#167; 630.101 Purpose.</HEAD>'
           b"<P>Pay &amp; leave, &quot;as defined&quot; &#8212; see below.</P></DIV8>")
    section = section_index(xml)["630.101"]
    assert section.heading == "Purpose"
    assert 'Pay & leave, "as defined" — see below.' in section.text


def test_a_source_is_not_required_but_the_message_still_locates_the_input():
    """cli.py and the invariant tests call the parser without a source. The message has to
    stay usable there too, so it falls back to describing the input itself."""
    with pytest.raises(CorpusParseError) as exc:
        parse_sections(b"<DIV8>")
    assert "6 bytes" in str(exc.value)
