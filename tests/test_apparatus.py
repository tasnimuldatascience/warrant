"""Apparatus stripping is load-bearing, so it gets real fixtures.

Every fixture below is shaped from actual eCFR Title 5 XML, not invented. The
pending-amendment pointer in particular is the exact form that made six of eight sampled
Part 630 "amendments" spurious in the initial spike.
"""

from __future__ import annotations

from lxml import etree

from warrant.corpus.apparatus import (
    AMENDMENT_LINK,
    is_apparatus,
    strip_apparatus,
    text_of,
)

PENDING_AMENDMENT = b"""
<DIV8 N="630.1203" TYPE="SECTION">
  <HEAD>&#167; 630.1203 Leave entitlement.</HEAD>
  <XREF ID="20200810" REFID="1">Link to an amendment published at 85 FR 48090, Aug. 10, 2020.</XREF>
  <P>(a) An employee shall be entitled to a total of 12 administrative workweeks of unpaid
  leave during any 12-month period.</P>
</DIV8>
"""

WITH_NOTES = b"""
<DIV8 N="630.101" TYPE="SECTION">
  <HEAD>&#167; 630.101 Purpose.</HEAD>
  <P>This subpart states the conditions governing the granting of leave.</P>
  <CITA TYPE="N">[33 FR 12475, Sept. 4, 1968]</CITA>
  <SOURCE><HED>Source:</HED><PSPACE>33 FR 12475, Sept. 4, 1968.</PSPACE></SOURCE>
  <EDNOTE><HED>Editorial Note:</HED><PSPACE>Nomenclature changes.</PSPACE></EDNOTE>
</DIV8>
"""

# A real cross-reference -- regulatory text, must survive.
REAL_XREF = b"""
<DIV8 N="630.306" TYPE="SECTION">
  <P>Except as authorized under <XREF TARGET="630.310">&#167; 630.310(d)</XREF>, annual
  leave must be scheduled and used before the end of the leave year.</P>
</DIV8>
"""

# Apparatus mid-paragraph: the sentence continues in the element's tail.
TAIL_TEXT = b"""
<DIV8 N="630.201" TYPE="SECTION">
  <P>Accrued leave means the leave earned by an employee<FTNT>See note.</FTNT> during the
  current leave year.</P>
</DIV8>
"""


def parse(raw: bytes) -> etree._Element:
    return etree.fromstring(raw)


def test_pending_amendment_pointer_is_removed():
    out = text_of(parse(PENDING_AMENDMENT))
    assert "Link to an amendment" not in out
    assert "12 administrative workweeks" in out


def test_editorial_notes_are_removed_but_regulation_survives():
    out = text_of(parse(WITH_NOTES))
    assert "conditions governing the granting of leave" in out
    for gone in ("Source:", "Editorial Note:", "33 FR 12475"):
        assert gone not in out


def test_real_cross_reference_is_kept():
    """Only the pending-amendment XREF is apparatus. Ordinary cross-references are the
    regulation citing itself, and dropping them would break applicability reasoning."""
    out = text_of(parse(REAL_XREF))
    assert "630.310(d)" in out
    assert "annual" in out and "leave year" in out


def test_tail_text_after_apparatus_is_preserved():
    """Regression: removing an element without reattaching its tail silently deletes
    regulatory prose, which surfaces much later as an unexplained retrieval miss."""
    out = text_of(parse(TAIL_TEXT))
    assert "during the current leave year" in out
    assert "See note" not in out


def test_strip_is_idempotent():
    once = strip_apparatus(parse(WITH_NOTES))
    twice = strip_apparatus(once)
    assert etree.tostring(once) == etree.tostring(twice)


def test_strip_does_not_mutate_input_by_default():
    node = parse(PENDING_AMENDMENT)
    before = etree.tostring(node)
    strip_apparatus(node)
    assert etree.tostring(node) == before


def test_is_apparatus_classifies_each_tag():
    node = parse(WITH_NOTES)
    tags = {el.tag: is_apparatus(el) for el in node.iter()}
    assert tags["CITA"] and tags["SOURCE"] and tags["EDNOTE"]
    assert not tags["P"] and not tags["HEAD"]


def test_amendment_link_pattern_tolerates_spacing_and_case():
    for s in ("Link to an amendment published at 85 FR 1",
              "link  to  a  amendment  published  at 90 FR 2",
              "LINK TO AN AMENDMENT PUBLISHED AT 91 FR 3"):
        assert AMENDMENT_LINK.search(s)
