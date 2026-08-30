"""Parsing eCFR XML into retrieval and citation units."""

from __future__ import annotations

import glob
import time

import pytest

from warrant.corpus.parse import (
    _BELOW,
    CorpusParseError,
    _forms,
    parse_sections,
    section_index,
)

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


# -- the (i) problem ---------------------------------------------------------------
#
# `(i)` is the ninth letter and the first roman numeral, and 5 CFR 890.301 uses it as both,
# nine paragraphs apart. FEHB_301 is that section's numbering, verbatim in shape, with the
# prose cut to its first clause. It is the case the old greedy stack could not survive: it
# read the roman `(i)` under (h)(1) as the letter after (h), lost the top level there and
# never recovered it, so paragraphs (j) through (p) -- ordinary top-level subsections of a
# section 5 CFR 890.301 is cited for constantly -- were stored as `890.301#ii-7-j` and on.
# A citation to 890.301(n) resolved to nothing, and `890.301#ii-7-n` addressed a paragraph
# that does not exist in the law.

FEHB_301 = b"""
<DIV8 N="890.301" TYPE="SECTION">
  <HEAD>&#167; 890.301 Opportunities for employees to enroll or change enrollment.</HEAD>
  <P>(a) <I>Initial opportunity to enroll.</I> An employee who becomes eligible may enroll.</P>
  <P>(e) <I>Decreasing enrollment type.</I> (1) Subject to two exceptions, an employee may
     decrease enrollment type at any time.</P>
  <P>(i) An employee participating in health insurance premium conversion may decrease.</P>
  <P>(ii) An employee who is subject to a court order may not decrease enrollment type.</P>
  <P>(2) A decrease in enrollment type takes effect on the first day of the first pay
     period.</P>
  <P>(h) <I>Change in employment status.</I> An eligible employee may enroll following:</P>
  <P>(1) A return to pay status following loss of coverage under:</P>
  <P>(i) Section 890.304(a)(1)(v) due to the expiration of 365 days in leave without pay.</P>
  <P>(ii) Section 890.502(b)(5) due to the termination of coverage.</P>
  <P>(2) Reemployment after a break in service of more than 3 days.</P>
  <P>(7) A change, without a break in service, from an appointment.</P>
  <P>(i) Loss of coverage under this part or under another group insurance plan.</P>
  <P>(1) Loss of coverage under another FEHB enrollment.</P>
  <P>(4) Loss of coverage due to the discontinuance of an FEHB plan.</P>
  <P>(i) If the discontinuance is at the end of a contract year.</P>
  <P>(ii) If the whole plan is discontinued.</P>
  <P>(5) Loss of coverage under the Medicaid program.</P>
  <P>(j) <I>Move from comprehensive medical plan's area.</I> An employee may change.</P>
  <P>(n) <I>Determination of lowest-cost nationwide plan option.</I> OPM will determine.</P>
</DIV8>
"""

FEHB_301_ANCHORS = [
    "a",
    "e", "e-1-i", "e-1-ii", "e-2",
    "h", "h-1", "h-1-i", "h-1-ii", "h-2", "h-7",
    "i", "i-1", "i-4", "i-4-i", "i-4-ii", "i-5",
    "j", "n",
]


def test_a_roman_run_at_depth_returns_to_the_top_level():
    """The whole of §890.301, addressed as the regulation addresses it.

    Both readings of `(i)` are in here and neither is decidable where it stands: under (h)(1)
    it is a roman numeral, after (h)(7) it is the letter following (h). What separates them is
    what comes next -- `(ii)` continues a roman run, `(1)` cannot follow a roman numeral
    without a level being skipped -- so the parser has to defer the decision, and this is the
    assertion that it does.
    """
    anchors = [p.anchor for p in section_index(FEHB_301)["890.301"].paragraphs]
    assert anchors == FEHB_301_ANCHORS


def test_the_same_token_reads_as_both_a_numeral_and_a_letter_in_one_section():
    """The narrow claim inside the case above, asserted on its own so a failure says which
    half broke: the two `(i)` paragraphs of §890.301 get different addresses."""
    anchors = [p.anchor for p in section_index(FEHB_301)["890.301"].paragraphs]
    assert anchors[anchors.index("h-1") + 1] == "h-1-i"
    assert "i" in anchors and "i-1" in anchors


def test_no_anchor_is_rooted_at_a_roman_numeral():
    """The symptom that made the bug visible in the store: 125 anchors in 45 sections whose
    root was a roman numeral, `890.301#ii-7-n` among them. Nothing in 5 CFR is numbered from
    a roman numeral at the top level."""
    anchors = [p.anchor for p in section_index(FEHB_301)["890.301"].paragraphs]
    assert not [a for a in anchors if a.split("-")[0] in ("ii", "iii", "iv", "vi")]


# -- levels the markup does not carry ----------------------------------------------

def test_a_level_written_inline_with_its_parent_is_filled_in():
    """§890.301(e) runs "(e) Decreasing enrollment type. (1) Subject to two exceptions ..."
    and (e)(1) never gets a <P> of its own, so the roman numerals beneath it have a parent the
    markup does not carry. Addressing them `e-i` claimed a roman level directly under a
    lettered one, which the CFR's numbering does not have."""
    anchors = [p.anchor for p in section_index(FEHB_301)["890.301"].paragraphs]
    assert "e-1-i" in anchors and "e-i" not in anchors
    # ... and the level below it still closes: (2) is (e)(2), not (e)(1)(i)(2).
    assert "e-2" in anchors


TWO_BURIED = b"""
<DIV8 N="315.612" TYPE="SECTION">
  <HEAD>&#167; 315.612 Noncompetitive appointment of certain military spouses.</HEAD>
  <P>(e) <I>Proof of eligibility.</I> (1)(i) Prior to appointment, the spouse must submit:</P>
  <P>(A) Documentation verifying active duty status; and</P>
  <P>(B) Documentation verifying marriage to the member of the armed forces.</P>
  <P>(ii) For appointments made on or after January 1, 2029, the spouse must also submit.</P>
  <P>(2) Prior to appointment, the spouse of a member as defined in paragraph (b)(4)(ii).</P>
</DIV8>
"""


def test_two_levels_can_be_buried_in_one_chapeau():
    """§315.612(e) buries both (1) and (i) in its own sentence before the standalone (A).
    One implied level is not enough here, and stopping at one addressed the subparagraphs as
    `e-A`, which reads as a subparagraph hanging directly off a lettered paragraph."""
    anchors = [p.anchor for p in section_index(TWO_BURIED)["315.612"].paragraphs]
    assert anchors == ["e", "e-1-i-A", "e-1-i-B", "e-1-ii", "e-2"]


# -- definitions sections ----------------------------------------------------------

DEFINITIONS = b"""
<DIV8 N="551.104" TYPE="SECTION">
  <HEAD>&#167; 551.104 Definitions.</HEAD>
  <P>In this part&#8212;</P>
  <P>Agency means any instrumentality of the United States Government.</P>
  <P>Directly and closely related means work that is directly related to exempt work.</P>
  <P>(1) Work is closely related to exempt supervisory work when it contributes to it.</P>
  <P>(2) A management analyst may take extensive notes recording the flow of work.</P>
  <P>Employee means a person who is employed&#8212;</P>
  <P>(1) As a civilian in an Executive agency, as defined in section 105 of title 5.</P>
  <P>(2) As a civilian in a military department, as defined in section 102 of title 5.</P>
</DIV8>
"""


def test_a_definition_list_belongs_to_its_headword_not_to_the_last_designator():
    """A definitions section is a run of undesignated headwords, each with its own (1), (2).
    Reading those as children of the last designator seen made §551.104(6)(1) and
    §630.201(b)(7)(1) -- addresses that look like citations and are in no regulation. There is
    no CFR designator for them, so they are addressed under the paragraph they belong to."""
    anchors = [p.anchor for p in section_index(DEFINITIONS)["551.104"].paragraphs]
    assert anchors == ["p1", "p2", "p3", "p3-1", "p3-2", "p6", "p6-1", "p6-2"]


CONTINUED = b"""
<DIV8 N="630.306" TYPE="SECTION">
  <HEAD>&#167; 630.306 Time limit for restored leave.</HEAD>
  <P>(a) Restored leave must be scheduled and used as follows:</P>
  <P>(1) Not later than the end of the leave year in effect.</P>
  <P>(2) Not later than the end of the 2-year period.</P>
  <P>The time limits in this paragraph run from the date of restoration.</P>
  <P>(b) An agency may extend the time limit.</P>
</DIV8>
"""


def test_flush_text_inside_a_list_does_not_reroot_it():
    """The other thing an undesignated paragraph is: the sentence that closes a list. A list
    that carries on across one is ordinary and must keep its addresses, so only a list that
    *starts* across one is read as belonging to the undesignated paragraph."""
    anchors = [p.anchor for p in section_index(CONTINUED)["630.306"].paragraphs]
    assert anchors == ["a", "a-1", "a-2", "p4", "b"]


# -- numbering that does not start at (a) ------------------------------------------

NUMBERED_TOP = b"""
<DIV8 N="330.602" TYPE="SECTION">
  <HEAD>&#167; 330.602 Definitions.</HEAD>
  <P>Displaced describes an agency employee in one of the following two categories:</P>
  <P>(1) A current career or career-conditional competitive service employee who:</P>
  <P>(i) Received a reduction in force separation notice under part 351.</P>
  <P>(ii) Received a notice of proposed removal under part 752.</P>
  <P>(2) A current excepted service employee on an appointment without time limit who:</P>
  <P>(i) Is covered by a law providing noncompetitive appointment eligibility.</P>
</DIV8>
"""


def test_a_level_is_numbered_by_its_parent_not_by_its_depth():
    """§330.602's list opens at (1) with no lettered paragraph above it. Deciding what a level
    is numbered with from its absolute depth made the roman numerals two levels down instead
    of one and minted §330.602(1)(1)(i)."""
    anchors = [p.anchor for p in section_index(NUMBERED_TOP)["330.602"].paragraphs]
    assert anchors == ["p1", "p1-1", "p1-1-i", "p1-1-ii", "p1-2", "p1-2-i"]


DOUBLED = b"""
<DIV8 N="330.609" TYPE="SECTION">
  <HEAD>&#167; 330.609 Exceptions.</HEAD>
  <P>(y) Appointment of a former employee.</P>
  <P>(z) Appointment under a special authority.</P>
  <P>(aa) Appointment of a disabled veteran.</P>
  <P>(bb) Appointment under the Postal Reorganization Act.</P>
  <P>(cc) Appointment of a former overseas employee.</P>
</DIV8>
"""


def test_lettered_paragraphs_past_z_double_the_letter():
    """§330.609 runs (a) through (z) and on to (ee). Past (z) the CFR doubles the letter
    rather than counting in base 26, so (bb) is the 28th designator and not the 54th; reading
    them spreadsheet-style broke the sequence at (bb) and minted §330.609(bb)(1)(cc)."""
    anchors = [p.anchor for p in section_index(DOUBLED)["330.609"].paragraphs]
    assert anchors == ["y", "z", "aa", "bb", "cc"]


# -- properties that must hold over the whole cached corpus ------------------------

def _cached_snapshots() -> list[str]:
    """The newest snapshot of each cached part. Offline: `make fetch` put them there."""
    latest: dict[str, str] = {}
    for path in sorted(glob.glob("data/ecfr/full-t5-p*.xml")):
        part = path.replace("\\", "/").split("-p")[-1].split("-")[0]
        latest[part] = max(latest.get(part, ""), path)
    return sorted(latest.values())


def test_every_anchor_is_numbered_the_way_the_cfr_numbers():
    """The instrument that found the blast radius, kept as a gate.

    A designator's kind must be one its parent's kind can contain: (a) holds (1) holds (i)
    holds (A). An anchor that breaks the chain is wrong without anyone having to know what the
    right answer is, and `890.301#ii-7-j` -- a lettered paragraph under a number under a roman
    numeral -- breaks it twice. 604 of the 9,955 in-force anchors did, 6.07%, in 66 sections;
    over all 226 cached snapshots, 6,845 of 127,402. See
    results/eval-012-anchor-correctness.md.
    """
    files = _cached_snapshots()
    if not files:
        pytest.skip("no snapshots cached; run `make fetch`")
    wrong: list[str] = []
    for path in files:
        with open(path, "rb") as fh:
            for section in parse_sections(fh.read(), source=path):
                for para in section.paragraphs:
                    parts = para.anchor.split(".")[0].split("-")
                    kinds = [set(_forms(p)) for p in parts]
                    if any(not k for k in kinds):
                        continue  # a positional fallback: p3, t1, and their children
                    if not _chains(kinds):
                        wrong.append(f"{section.identifier}#{para.anchor}")
    assert wrong == []


def _chains(kinds: list[set[str]]) -> bool:
    """Can these designators be read as a parent chain of CFR levels?"""
    reachable = set(_BELOW) if len(kinds) == 1 else {"alpha", "digit", "roman", "upper"}
    for step in kinds:
        reachable &= step
        if not reachable:
            return False
        reachable = {_BELOW[k] for k in reachable}
    return True


def test_every_citation_address_in_the_corpus_is_unique():
    """A citation that matches two paragraphs is not a citation. The collision suffix in
    `_paragraphs` is a backstop and is expected to stay unused: it does, over all 9,955
    in-force anchors."""
    files = _cached_snapshots()
    if not files:
        pytest.skip("no snapshots cached; run `make fetch`")
    for path in files:
        with open(path, "rb") as fh:
            for section in parse_sections(fh.read(), source=path):
                anchors = [p.anchor for p in section.paragraphs]
                assert len(set(anchors)) == len(anchors), section.identifier
                assert not [a for a in anchors if "." in a], section.identifier


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
