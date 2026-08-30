"""The chunking policy: what a retrievable unit is, as distinct from what parses.

Every test here builds its own ``Section`` by hand. That is the point of keeping the policy
out of parse.py: the previous chunker could only be argued about by rebuilding a store and
looking at it, so the claim "the section heading is in the retrieved string" was checked by
reading code rather than by running anything -- and it was false in two of the three ranking
stages for as long as it was believed.

The invariant every case restates: ``Unit.text`` is verbatim. Context is what retrieval
ranks; text is what a citation points at, and a citation to text the chunker rewrote is not
a citation.
"""

from __future__ import annotations

import pytest

from warrant.corpus.build import section_chunks
from warrant.corpus.chunking import (
    ChunkPolicy,
    Distribution,
    ParentUnit,
    distribution,
    parent_texts,
    report,
    sentence_spans,
    split_spans,
    store_distribution,
    token_count,
    units,
)
from warrant.corpus.parse import Paragraph, Section
from warrant.index.store import Store

DEFINITIONS = Section(
    identifier="300.401",
    heading="Definitions",
    text="whole section",
    subpart="D",
    paragraphs=[
        Paragraph("p1", "For purposes of this subpart:"),
        Paragraph("a", "(a) Detail means the temporary assignment of an employee."),
        Paragraph("b", "(b) Transfer means a change of an employee."),
        Paragraph("p4", "An agency shall document each action taken under this subpart."),
    ],
)

DEEP = Section(
    identifier="330.605",
    heading="Agency responsibilities",
    text="whole section",
    subpart="F",
    paragraphs=[
        Paragraph("b", "(b) Each agency shall establish a program that:"),
        Paragraph("b-1", "(b)(1) Covers all competitive service positions, except:"),
        Paragraph("b-1-i", "(b)(1)(i) Positions in the excepted service; and"),
        Paragraph("b-1-ii", "(b)(1)(ii) Positions filled by reinstatement."),
    ],
)


def by_anchor(section: Section, policy: ChunkPolicy | None = None) -> dict:
    return {u.anchor: u for u in units(section, policy or ChunkPolicy())}


# -- contextual augmentation ------------------------------------------------------


def test_context_carries_subpart_section_heading_and_chapeau():
    """The three things a definition paragraph does not say about itself.

    "(a) Detail means the temporary assignment of an employee." never mentions §300.401,
    never mentions subpart D, and is severed from "For purposes of this subpart:" -- the
    sentence that says these are definitions rather than substantive rules.
    """
    u = by_anchor(DEFINITIONS)["a"]
    assert "Subpart D" in u.context
    assert "300.401" in u.context
    assert "Definitions" in u.context
    assert "For purposes of this subpart:" in u.context
    assert u.text == "(a) Detail means the temporary assignment of an employee."


def test_a_concluding_flush_paragraph_is_not_a_chapeau():
    """Flush text appears at both ends of a list and parse.py anchors both ``p{index}``.

    Only the colon distinguishes them. Without that test, "An agency shall document each
    action" would be attached as governing context to every paragraph that follows it,
    including paragraphs in a later list it has nothing to do with. Merging is off here so
    the assertion is about the chapeau rule and not about the enclosing section, which does
    legitimately carry every paragraph of itself.
    """
    concluding = "An agency shall document each action taken under this subpart."
    got = units(DEFINITIONS, ChunkPolicy(merge=False))
    assert all(concluding not in u.context for u in got)
    assert all("For purposes of this subpart:" in u.context
               for u in got if u.anchor in ("a", "b"))


def test_a_deep_anchor_carries_the_designators_that_govern_it():
    """Without (b) and (b)(1), "(b)(1)(ii) Positions filled by reinstatement." is a fragment
    of a sentence: it never says what those positions are excepted *from*."""
    u = by_anchor(DEEP)["b-1-ii"]
    assert "(b) Each agency shall establish a program that:" in u.context
    assert "(b)(1) Covers all competitive service positions, except:" in u.context
    assert u.context.index("(b) Each") < u.context.index("(b)(1) Covers")
    assert u.text == "(b)(1)(ii) Positions filled by reinstatement."


def test_context_is_bounded_by_the_budget():
    """Context is prepended to every ranking stage's input. An unbounded one pushes the
    paragraph out of the encoder window, which is the failure splitting exists to prevent."""
    long_parent = " ".join(f"word{i}" for i in range(400))
    section = Section(identifier="1.1", heading="H", text="t", paragraphs=[
        Paragraph("a", f"(a) {long_parent}"),
        Paragraph("a-1", "(a)(1) Short."),
    ])
    u = by_anchor(section)["a-1"]
    assert token_count(u.context) <= ChunkPolicy().max_context_tokens + 20


def test_legacy_policy_reproduces_the_previous_behaviour():
    for u in units(DEEP, ChunkPolicy.legacy()):
        assert u.context == ""
        assert u.parent_id == ""
        assert not u.merged
    assert [u.anchor for u in units(DEEP, ChunkPolicy.legacy())] == \
        [p.anchor for p in DEEP.paragraphs]


# -- small-to-big -----------------------------------------------------------------


def test_parent_id_points_at_the_enclosing_unit():
    """The parent paragraph where there is one, the section where there is not. Small-to-big
    has to bottom out somewhere, and the section is the unit parse.py already declares."""
    got = by_anchor(DEEP)
    assert got["b"].parent_id == "330.605"
    assert got["b-1"].parent_id == "330.605#b"
    assert got["b-1-i"].parent_id == "330.605#b-1"
    assert got["b-1-ii"].parent_id == "330.605#b-1"


def test_parent_helper_deduplicates_siblings():
    """Two siblings name one parent. Fetching it twice would spend the generator's context
    window proving that the two results came from the same subsection."""
    with Store(":memory:") as store:
        store.add(section_chunks(DEEP, title=5, part="330", valid_from="2017-01-01",
                                 snapshot="2017-01-01", config_hash="t"))
        parents = parent_texts(store, ["330.605#b-1-i", "330.605#b-1-ii"],
                               valid_date="2020-01-01")
    assert [p.parent_id for p in parents] == ["330.605#b-1"]
    assert parents[0].requested_by == ("330.605#b-1-i", "330.605#b-1-ii")
    assert parents[0].text == "(b)(1) Covers all competitive service positions, except:"


def test_parent_helper_returns_the_whole_section_for_a_top_level_paragraph():
    with Store(":memory:") as store:
        store.add(section_chunks(DEFINITIONS, title=5, part="300", valid_from="2017-01-01",
                                 snapshot="2017-01-01", config_hash="t"))
        parents = parent_texts(store, ["300.401#a"], valid_date="2020-01-01")
    assert [p.parent_id for p in parents] == ["300.401"]
    assert "(b) Transfer means a change of an employee." in parents[0].text
    assert "For purposes of this subpart:" in parents[0].text


def test_parent_helper_reads_the_version_in_force_on_the_date():
    """A parent fetched by chunk_id alone would hand a 2024 parent to a 2018 child, which is
    the one failure this store exists to prevent."""
    old = Section(identifier="1.1", heading="H", text="t", paragraphs=[
        Paragraph("a", "(a) the old rule applies to:"),
        Paragraph("a-1", "(a)(1) everyone."),
    ])
    new = Section(identifier="1.1", heading="H", text="t", paragraphs=[
        Paragraph("a", "(a) the new rule applies to:"),
        Paragraph("a-1", "(a)(1) everyone."),
    ])
    with Store(":memory:") as store:
        store.add(section_chunks(old, title=5, part="1", valid_from="2017-01-01",
                                 snapshot="2017-01-01", config_hash="t"))
        store.close_valid("1.1", "2020-01-01")
        store.add(section_chunks(new, title=5, part="1", valid_from="2020-01-01",
                                 snapshot="2020-01-01", config_hash="t"))
        before = parent_texts(store, ["1.1#a-1"], valid_date="2018-06-01")
        after = parent_texts(store, ["1.1#a-1"], valid_date="2021-06-01")
    assert before[0].text == "(a) the old rule applies to:"
    assert after[0].text == "(a) the new rule applies to:"


def test_parent_helper_on_an_empty_request():
    with Store(":memory:") as store:
        assert parent_texts(store, [], valid_date="2020-01-01") == []


# -- merging ----------------------------------------------------------------------


SHORT_LIST = Section(
    identifier="300.102",
    heading="Standards",
    text="whole section",
    subpart="A",
    paragraphs=[
        Paragraph("p1", "Employment practices shall:"),
        Paragraph("a", "(a) Be job related;"),
        Paragraph("b", "(b) Result in selection from among the best qualified candidates;"),
        Paragraph("c", "(c) Assure equal employment opportunity."),
    ],
)


def test_a_sub_threshold_unit_merges_but_keeps_its_citation_anchor():
    """Ten tokens naming neither their subject nor their section. The merge changes what
    retrieval ranks and leaves the address and the text a reader would look up untouched."""
    u = by_anchor(SHORT_LIST)["b"]
    assert u.merged
    assert u.anchor == "b"
    assert u.text == "(b) Result in selection from among the best qualified candidates;"
    assert "(a) Be job related;" in u.context
    assert "(c) Assure equal employment opportunity." in u.context
    assert u.text not in u.context          # its own text is not duplicated into it


def test_a_unit_over_the_floor_is_not_merged():
    long_enough = ("(a) An agency shall establish procedures to ensure that every applicant "
                   "receives fair and open consideration for the position applied for.")
    section = Section(identifier="1.1", heading="H", text="t", paragraphs=[
        Paragraph("a", long_enough), Paragraph("b", "(b) short.")])
    got = by_anchor(section)
    assert not got["a"].merged
    assert got["b"].merged


def test_merging_never_crosses_a_section_boundary():
    """Sections are the unit the regulation is amended in and cited by. A retrieval unit
    spanning two of them would put text from §300.103 behind a citation to §300.102."""
    other = Section(identifier="300.103", heading="Other", text="whole", subpart="A",
                    paragraphs=[Paragraph("a", "(a) Something else entirely.")])
    contexts = [u.context for u in units(SHORT_LIST)] + [u.context for u in units(other)]
    assert not any("Something else entirely" in c for c in contexts[:len(SHORT_LIST.paragraphs)])
    assert not any("best qualified" in c for c in contexts[len(SHORT_LIST.paragraphs):])


def test_merge_is_off_under_the_legacy_policy():
    assert not any(u.merged for u in units(SHORT_LIST, ChunkPolicy.legacy()))


# -- tables -----------------------------------------------------------------------


WAGE_TABLE = "\n".join(f"Step {i} | rate {i} | effective date" for i in range(1, 60))

TABLED = Section(
    identifier="532.313",
    heading="Continuation of local wage rates",
    text="whole section",
    subpart="D",
    paragraphs=[
        Paragraph("a", "(a) The schedule is:"),
        Paragraph("t1", WAGE_TABLE),
        Paragraph("t2", "Grade | Rate"),
    ],
)


def test_a_table_is_never_split():
    """A table is one semantic unit: splitting it destroys the row relationship that makes
    it answerable, which is the same argument parse.py makes for not splitting it per cell."""
    got = units(TABLED, ChunkPolicy(max_tokens=20))
    tables = [u for u in got if u.kind == "table"]
    assert [u.anchor for u in tables] == ["t1", "t2"]
    assert tables[0].text == WAGE_TABLE
    assert token_count(tables[0].text) > 20


def test_a_short_table_is_never_merged():
    """"Grade | Rate" is under the floor and still must not be glued to the prose around
    it: prose pulled into a table's retrieval unit reads as another row."""
    small = [u for u in units(TABLED) if u.anchor == "t2"][0]
    assert small.kind == "table"
    assert not small.merged
    assert small.text == "Grade | Rate"


def test_a_table_is_never_pulled_into_a_neighbours_context():
    u = [u for u in units(TABLED) if u.anchor == "a"][0]
    assert "Step 1 | rate 1" not in u.context


# -- splitting --------------------------------------------------------------------


LONG = ("(c) Nature of Reserve service. An employee who is a member of the Reserve shall "
        "notify the agency. The agency shall record the notification in writing. "
        "An employee may elect to use annual leave for the period of service. "
        "The agency may not require the employee to exhaust annual leave first. "
        "A returning employee shall be restored to the position held.")


def test_an_oversized_unit_splits_at_sentence_boundaries_with_overlap():
    section = Section(identifier="353.203", heading="Reserve service", text="t",
                      paragraphs=[Paragraph("c", LONG)])
    got = units(section, ChunkPolicy(max_tokens=20, overlap_sentences=1))
    assert len(got) > 2
    assert [u.anchor for u in got] == sorted(u.anchor for u in got)   # unique and ordered
    assert len({u.anchor for u in got}) == len(got)
    assert all(u.anchor.startswith("c.s") for u in got)
    for u in got:
        assert u.text in LONG                       # verbatim slices, never rewritten
        assert u.split_from == "c"
    # Overlap: the last sentence of a piece opens the next one.
    assert got[1].text.startswith(got[0].text.split(". ")[-1][:20])


def test_a_split_continuation_carries_the_designator_it_lost():
    """Piece two no longer says "(c) Nature of Reserve service" and cannot be ranked on a
    query about Reserve service without it."""
    section = Section(identifier="353.203", heading="Reserve service", text="t",
                      paragraphs=[Paragraph("c", LONG)])
    got = units(section, ChunkPolicy(max_tokens=20))
    assert "(c) Nature of Reserve service." in got[1].context
    assert "(c) Nature of Reserve service." not in got[0].context


def test_split_pieces_reconstruct_the_paragraph():
    section = Section(identifier="353.203", heading="Reserve service", text="t",
                      paragraphs=[Paragraph("c", LONG)])
    got = units(section, ChunkPolicy(max_tokens=25, overlap_sentences=0))
    assert "".join(u.text for u in got).replace(" ", "") == LONG.replace(" ", "")


def test_a_unit_under_the_ceiling_is_left_alone():
    got = units(DEFINITIONS)
    assert [u.anchor for u in got] == [p.anchor for p in DEFINITIONS.paragraphs]
    assert all(u.split_from is None for u in got)


def test_split_is_off_under_the_legacy_policy():
    section = Section(identifier="353.203", heading="Reserve service", text="t",
                      paragraphs=[Paragraph("c", LONG)])
    got = units(section, ChunkPolicy.legacy())
    assert [u.text for u in got] == [LONG]


@pytest.mark.parametrize("text, expected", [
    ("See 5 U.S.C. 3301. The agency shall act.", 2),
    ("Pay is set under § 531.203. An employee may appeal.", 2),
    ("The rate is 31.5 percent of basic pay.", 1),
    ("See Pub. L. 106-58. It applies.", 2),
])
def test_sentence_boundaries_survive_legal_citation(text: str, expected: int):
    """Splitting inside "5 U.S.C. 3301" produces a piece whose first words are a bare
    section number: unrankable, and worse, it looks like a citation to something else."""
    assert len(sentence_spans(text)) == expected


def test_a_single_oversized_sentence_falls_back_to_semicolons():
    """A 300-token CFR paragraph is usually one sentence with a dozen semicolons in it.
    Refusing to break it at all would leave the very chunks splitting exists for."""
    text = "; ".join(f"item number {i} of the enumerated list" for i in range(1, 20))
    pieces = split_spans(text, max_tokens=20, overlap_sentences=0)
    assert len(pieces) > 1
    assert all(text[a:b] in text for a, b in pieces)


# -- distribution reporting -------------------------------------------------------


def test_distribution_reports_the_numbers_a_reviewer_asks_for():
    """Hand-built so the arithmetic is checkable by eye: ten chunks of 1..10 tokens."""
    texts = [" ".join("w" * 1 for _ in range(n)) for n in range(1, 11)]
    d = distribution(texts, thresholds=(5, 10))
    assert d.n == 10
    assert d.mean == pytest.approx(5.5)
    assert d.median == 5           # nearest-rank: ceil(0.5 * 10) = 5th value
    assert d.p10 == 1
    assert d.p90 == 9
    assert d.maximum == 10
    assert d.under == {5: 4, 10: 9}
    assert d.share_under(10) == pytest.approx(0.9)


def test_distribution_of_nothing_is_not_an_exception():
    """A reporter that raises on an empty store cannot be called before the first build."""
    d = Distribution.of([])
    assert d.n == 0 and d.mean == 0.0 and d.share_under(10) == 0.0


def test_distribution_reads_the_store_in_both_units():
    """The citation unit barely moves and the retrieval unit is the whole point. Reporting
    only the first would read as "the policy did nothing"; only the second would hide that
    the verbatim guarantee still holds."""
    with Store(":memory:") as store:
        store.add(section_chunks(SHORT_LIST, title=5, part="300", valid_from="2017-01-01",
                                 snapshot="2017-01-01", config_hash="t"))
        cited = store_distribution(store)
        retrieved = store_distribution(store, unit="retrieval")
    assert cited.n == retrieved.n == len(SHORT_LIST.paragraphs)
    assert cited.maximum == max(token_count(p.text) for p in SHORT_LIST.paragraphs)
    assert retrieved.p10 > cited.p90
    with pytest.raises(ValueError, match="citation"):
        store_distribution(store, unit="both")


def test_report_prints_before_and_after():
    before = Distribution.of([1, 2, 3, 40], thresholds=(10,))
    after = Distribution.of([20, 21, 22, 40], thresholds=(10,))
    text = report(after, before)
    assert "before" in text and "after" in text
    assert "3 (75.0%)" in text.replace("  ", " ")
    assert "0 ( 0.0%)" in text.replace("  ", " ")


# -- the seam into ingestion ------------------------------------------------------


def test_section_chunks_carry_context_parent_and_kind():
    chunks = {c.anchor: c for c in
              section_chunks(TABLED, title=5, part="532", valid_from="2017-01-01",
                             snapshot="2017-01-01", config_hash="t")}
    assert chunks["a"].context.startswith("Subpart D")
    assert chunks["a"].parent_id == "532.313"
    assert chunks["t1"].kind == "table"
    assert chunks["a"].kind == "prose"
    assert chunks["a"].retrieval_text.endswith(chunks["a"].text)


def test_a_section_with_no_paragraphs_keeps_its_single_full_chunk():
    section = Section(identifier="630.101", heading="Purpose", text="whole section text")
    chunks = section_chunks(section, title=5, part="630", valid_from="2017-01-01",
                            snapshot="2017-01-01", config_hash="t")
    assert [c.chunk_id for c in chunks] == ["630.101#full"]
    assert chunks[0].anchor is None
    assert chunks[0].text == "whole section text"
    assert "630.101 Purpose" in chunks[0].context


def test_every_chunk_text_is_verbatim_paragraph_text():
    """The single invariant across all four policies: nothing here rewrites the law."""
    for section in (DEFINITIONS, DEEP, SHORT_LIST, TABLED):
        for u in units(section):
            source = {p.anchor: p.text for p in section.paragraphs}
            assert u.text in source[u.split_from or u.anchor]


def test_parent_unit_is_hashable_and_reads_as_a_record():
    p = ParentUnit("1.1#a", "text", ("1.1#a-1",))
    assert p.parent_id == "1.1#a" and p.requested_by == ("1.1#a-1",)
