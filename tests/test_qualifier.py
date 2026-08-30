"""Qualifier detection, cross-reference resolution, and the unstated-condition check.

Most of these are false-positive tests. A detector that fires on every "may not exceed" and
every "the documentation provided" would report a qualifier on a quarter of the corpus and be
useless, and the failures worth guarding are the ones where a phrase *looks* like a condition:
"excepted service", "no later than", "in other than the full fraction", a table caption.

Every regulatory string quoted here is real text from 5 CFR chapter I, and the chunk ids are
the ones the store holds. Made-up regulatory prose would test the regexes against a dialect
nobody writes in.
"""

from __future__ import annotations

import pytest

from warrant.verify.qualifier import (
    MIN_ACKNOWLEDGEMENT,
    acknowledgement,
    chapeau_ids,
    qualifiers,
    qualifiers_of,
    unstated_conditions,
)
from warrant.verify.xref import (
    DanglingReference,
    dangling_references,
    find_references,
    nameable_ids,
    resolve,
)

# §630.306(a), the paragraph the whole module exists for: a true quotation of it with the
# first eleven words removed is a wrong answer.
RESTORED_LEAVE = (
    "(a) Except as provided in paragraph (b) of this section, annual leave restored to an "
    "employee under 5 U.S.C. 6304(d) must be scheduled and used not later than the end of "
    "the leave year ending 2 years after the date of restoration."
)


# -- what is a qualifier ------------------------------------------------------------

def test_leading_exception_stops_at_the_comma():
    q, = [x for x in qualifiers(RESTORED_LEAVE, chunk_id="630.306#a") if x.kind == "except"]
    assert q.text == "Except as provided in paragraph (b) of this section"
    assert q.refers_to == ("630.306#b",)
    assert RESTORED_LEAVE[q.span[0]:q.span[1]] == q.text


def test_kinds_are_reported_separately():
    kinds = {q.kind for q in qualifiers(
        "Notwithstanding paragraph (c) of this section, an agency may not pay a retention "
        "incentive unless the employee signs a service agreement, subject to the "
        "requirements of subpart B.", chunk_id="575.305#b")}
    assert kinds == {"notwithstanding", "prohibition", "unless", "subject_to"}


@pytest.mark.parametrize("text", [
    # "excepted service" is a category of appointment, not an exception.
    "An employee serving in the excepted service is not covered by this subpart.",
    # "no later than" and "rather than" are comparatives.
    "The agency must act no later than 30 days after the request rather than at once.",
    # The past participle. 8 of every 10 bare "provided" in this corpus is one of these.
    "The decision is based solely on the information the individual provided with the "
    "request for review.",
    # "provided under" names an authority, not a proviso.
    "Benefits provided under this part continue during the period of service.",
])
def test_lookalikes_are_not_qualifiers(text):
    assert [q.kind for q in qualifiers(text)] == []


def test_numeric_ceiling_is_a_bound_not_an_unstated_condition():
    # 575.110(a). An answer that quotes "4 years" has said everything the clause says.
    q, = qualifiers("The service period may not exceed 4 years.", chunk_id="575.110#a")
    assert q.kind == "bound"
    assert not q.conditional


def test_ceiling_written_as_a_prohibition_is_still_a_bound():
    # 591.238(b), found by hand-labelling: the cap is real but the verb is "cause to exceed".
    q, = qualifiers("Agencies pay so much of the post differential as will not cause the "
                    "combined total to exceed 25 percent of the hourly rate of basic pay.")
    assert q.kind == "bound"


def test_prohibition_that_is_not_a_ceiling_stays_a_prohibition():
    q, = qualifiers("An employee may not directly or indirectly intimidate any other "
                    "employee.", chunk_id="532.504#c")
    assert q.kind == "prohibition"
    assert q.conditional


@pytest.mark.parametrize("text", [
    # 550.112(a)(2): a manner phrase, not an exclusion from the rule.
    "When overtime work is performed in other than the full fraction, odd minutes shall be "
    "rounded to the nearest full fraction of an hour.",
    # 890.301: "at a time other than X" identifies a time; nothing is excepted.
    "If the discontinuance is at a time other than the end of the contract year, OPM must "
    "establish an effective date.",
])
def test_other_than_as_a_comparative_is_not_an_exclusion(text):
    assert "other_than" not in {q.kind for q in qualifiers(text)}


def test_other_than_as_a_real_exclusion_is_kept():
    # 315.610(a)(1).
    q, = qualifiers("Was involuntarily separated (other than by removal for cause on charges "
                    "of misconduct or delinquency);", chunk_id="315.610#a-1")
    assert q.kind == "other_than"


def test_unless_otherwise_is_a_condition_in_this_corpus():
    # The anticipated false positive -- "unless otherwise noted" in a table caption -- does
    # not occur here. All 13 instances name an authority that can displace the rule, so the
    # detector keeps them and the results doc says why.
    q, = qualifiers("Meet the minimum standards for health benefits plans at § 890.201, "
                    "unless otherwise stated in this subpart;", chunk_id="890.1610#a-1")
    assert q.kind == "unless"


def test_chapeau_needs_children_not_just_a_colon():
    # 300.201(a) ends in a colon and enumerates inline, so nothing hangs off it in the store.
    text = "The Office does not release the following:"
    assert [q.kind for q in qualifiers(text, chunk_id="300.201#a", enumerates=False)] == []
    q, = qualifiers(text, chunk_id="430.309#a", enumerates=True)
    assert q.kind == "chapeau"
    assert q.span == (0, len(text))


def test_chapeau_recognises_the_em_dash_form():
    q, = qualifiers("(a) When rating senior executive performance, each agency must—",
                    chunk_id="430.309#a", enumerates=True)
    assert q.kind == "chapeau"


def test_two_triggers_in_one_sentence_are_two_qualifiers():
    # Suppressing on clause overlap instead of on the trigger lost the 480-hour cap in
    # 630.401(c) to a "subject to" that opened earlier in the same sentence.
    kinds = [q.kind for q in qualifiers(
        "The amount of sick leave granted may not exceed a total of 480 hours, subject to "
        "the limitation in paragraph (d) of this section.", chunk_id="630.401#c")]
    assert kinds == ["bound", "subject_to"]


def test_qualifiers_of_marks_chapeaux_from_the_corpus():
    evidence = {"430.309#a@2020-01-01": "When rating senior executive performance, each "
                                        "agency must—"}
    corpus = {"430.309#a", "430.309#a-1", "430.309#a-2"}
    found = qualifiers_of(evidence, in_corpus=corpus)
    assert [q.kind for q in found["430.309#a@2020-01-01"]] == ["chapeau"]
    assert qualifiers_of(evidence, in_corpus={"430.309#a"}) == {}


def test_chapeau_ids_names_only_parents():
    assert chapeau_ids({"630.306#a", "630.306#a-1", "630.306#a-1-i"}) == frozenset(
        {"630.306#a", "630.306#a-1"})


# -- did the answer say it ----------------------------------------------------------

def _exception_of(text, chunk_id):
    return next(q for q in qualifiers(text, chunk_id=chunk_id) if q.conditional)


def test_the_answer_that_drops_the_exception_is_flagged():
    q = _exception_of(RESTORED_LEAVE, "630.306#a")
    overlap, cued = acknowledgement(
        "Restored annual leave must be scheduled and used by the end of the leave year "
        "ending 2 years after restoration.", q)
    assert not (cued and overlap >= MIN_ACKNOWLEDGEMENT)


def test_the_answer_that_carries_the_exception_is_not_flagged():
    q = _exception_of(RESTORED_LEAVE, "630.306#a")
    overlap, cued = acknowledgement(
        "Restored annual leave must be used within 2 years, except as paragraph (b) of the "
        "section provides.", q)
    assert cued and overlap >= MIN_ACKNOWLEDGEMENT


def test_obligation_words_are_not_acknowledgement():
    # 890.204(a)(2). "must" is a mark of obligation, not of a bounded rule; while it counted
    # as a cue, an answer stating the notice period alone passed the check for "unless it is
    # waived in writing by the carrier".
    q, = qualifiers("The carrier shall be notified by certified mail at least 15 calendar "
                    "days in advance of the hearing, unless it is waived in writing by the "
                    "carrier.", chunk_id="890.204#a-2")
    _overlap, cued = acknowledgement(
        "The carrier must be notified by certified mail at least 15 calendar days before "
        "the hearing.", q)
    assert not cued


def test_an_answer_about_a_different_condition_does_not_acknowledge_this_one():
    # The cue alone cannot tell these apart, which is why overlap is the second signal:
    # both answers say "unless".
    q, = qualifiers("The minimum charge for leave is one hour, unless an agency establishes "
                    "a minimum charge of less than one hour.", chunk_id="630.206#a")
    overlap, cued = acknowledgement(
        "A carrier need not be notified 15 days in advance unless the Director so directs.",
        q)
    assert cued
    assert overlap < MIN_ACKNOWLEDGEMENT


def test_unstated_conditions_names_the_paragraph_that_would_have_said_it():
    evidence = {"630.306#a@2019-01-01": RESTORED_LEAVE}
    found = unstated_conditions(
        "Restored annual leave must be scheduled and used within 2 years.", evidence)
    assert [(u.source, u.qualifier.refers_to) for u in found] == [
        ("630.306#a@2019-01-01", ("630.306#b",))]


def test_a_stated_bound_is_never_an_unstated_condition():
    evidence = {"575.110#a@2020-01-01": "The service period may not exceed 4 years."}
    assert unstated_conditions("An agency may require a service period.", evidence) == []


def test_unstated_conditions_is_deterministic():
    evidence = {"630.306#a@2019-01-01": RESTORED_LEAVE,
                "575.110#a@2020-01-01": "The service period may not exceed 4 years."}
    answer = "Restored leave must be used within 2 years."
    first = unstated_conditions(answer, evidence)
    assert first == unstated_conditions(answer, evidence)


# -- cross-references ---------------------------------------------------------------

def test_paragraph_of_this_section_binds_to_the_paragraph():
    ref, = find_references("as provided in paragraph (b) of this section",
                           section_id="630.306")
    assert (ref.kind, ref.targets) == ("paragraph", ("630.306#b",))


def test_a_section_reference_beside_a_paragraph_reference_survives():
    # The section pattern matches "§ 630.309 and § 630.310(a)" as one phrase whose first half
    # the paragraph reference already owns. Discarding the whole match lost § 630.310.
    refs = find_references("under paragraph (d) of § 630.309 and § 630.310(a)",
                           section_id="630.306")
    assert [(r.kind, r.targets) for r in refs] == [
        ("paragraph", ("630.309#d",)), ("section", ("630.310#a",))]


def test_an_elided_prefix_continues_the_run_before_it():
    # 351.403(a)(5): "(4)" is (a)(4), not a top-level paragraph 4.
    ref, = find_references("paragraphs (a)(1) through (4) of this section",
                           section_id="351.403")
    assert ref.targets == ("351.403#a-1", "351.403#a-2", "351.403#a-3", "351.403#a-4")


def test_a_range_expands_between_its_endpoints():
    ref, = find_references("paragraphs (b)(1)(i) through (iv)", section_id="630.1503")
    assert ref.targets == ("630.1503#b-1-i", "630.1503#b-1-ii", "630.1503#b-1-iii",
                           "630.1503#b-1-iv")


def test_two_single_letters_are_letters_and_not_roman_numerals():
    # "(c) through (e)" also parses as roman 100 through 500. The cap and the ordering keep
    # 400 fabricated targets out of the dangling count.
    ref, = find_references("paragraphs (c) through (e) of this section", section_id="532.251")
    assert ref.targets == ("532.251#c", "532.251#d", "532.251#e")


def test_a_bare_digit_reference_is_relative_to_the_citing_paragraph():
    # 630.201(b)(6) says "paragraphs (2) through (5)" and means (b)(2) .. (b)(5). Top-level
    # CFR paragraphs are lettered, so a bare "(2)" cannot be one.
    ref, = find_references("paragraphs (2) through (5)", section_id="630.201", anchor="b-6")
    assert ref.targets == ("630.201#b-2", "630.201#b-3", "630.201#b-4", "630.201#b-5")


def test_a_reference_out_of_title_5_is_not_a_chunk():
    refs = find_references("under 5 U.S.C. 6304(d) and 29 CFR 1614.203")
    assert all(r.targets == () for r in refs)
    assert {r.kind for r in refs} == {"usc", "cfr"}


def test_uppercase_usc_is_still_usc():
    # 351.504: "5 U.S.C. Section 7116(a)(7)". Without the ignore-case flag these were not
    # references at all.
    ref, = find_references("Subject to the requirements of 5 U.S.C. Section 7116(a)(7)")
    assert ref.kind == "usc"


def test_this_section_resolves_to_the_citing_section():
    ref, = find_references("An employee under this section may be paid.",
                           section_id="550.162")
    assert (ref.kind, ref.targets) == ("scope", ("550.162",))


def test_this_subpart_names_nothing_resolvable():
    ref, = find_references("as provided in this subpart", section_id="550.162")
    assert ref.targets == ()


# -- dangling references ------------------------------------------------------------

CORPUS = nameable_ids({"630.306#a", "630.306#b", "630.310#d", "890.102#j-1", "890.102#j-2"})


def test_a_reference_the_evidence_set_does_not_cover_is_dangling():
    evidence = {"630.306#a@2019-01-01": RESTORED_LEAVE}
    out = dangling_references(evidence, in_corpus=CORPUS)
    assert out == [DanglingReference(source="630.306#a@2019-01-01", target="630.306#b",
                                     status="missing", reference=out[0].reference)]


def test_a_reference_the_evidence_set_covers_is_not_reported():
    evidence = {"630.306#a@2019-01-01": RESTORED_LEAVE,
                "630.306#b@2019-01-01": "(b) The head of an agency may extend the period."}
    assert dangling_references(evidence, in_corpus=CORPUS) == []


def test_a_chunk_referring_to_itself_is_not_dangling():
    evidence = {"630.306#a@2019-01-01": "An employee under this section may be paid."}
    assert dangling_references(evidence, in_corpus=CORPUS) == []


def test_the_corpus_boundary_is_kept_out_of_the_retrieval_number():
    evidence = {"630.306#a@2019-01-01": "restored under 5 U.S.C. 6304(d) and 29 CFR 1614.203"}
    assert dangling_references(evidence, in_corpus=CORPUS) == []
    outside = dangling_references(evidence, in_corpus=CORPUS, include_outside=True)
    assert {d.status for d in outside} == {"outside"}


def test_this_subpart_is_unscoped_rather_than_missing():
    evidence = {"630.306#a@2019-01-01": "Leave is restored as provided in this subpart."}
    out = dangling_references(evidence, in_corpus=CORPUS, include_outside=True)
    assert [d.status for d in out] == ["unscoped"]


def test_a_paragraph_with_no_chunk_of_its_own_is_still_nameable():
    # 890.102 is written with (j)'s chapeau inline with (j)(1), so the store holds j-1 .. j-5
    # and no bare "#j". The reference is to a real address and must not be charged to the
    # corpus boundary as ``outside``.
    assert resolve("890.102#j", CORPUS) == "890.102#j"
    evidence = {"890.501#a@2020-01-01": "as described in paragraph (j) of § 890.102"}
    out = dangling_references(evidence, in_corpus=CORPUS)
    assert [(d.target, d.status) for d in out] == [("890.102#j", "missing")]
    covered = dangling_references(
        {**evidence, "890.102#j-1@2020-01-01": "(j)(1) ..."}, in_corpus=CORPUS)
    assert covered == []


def test_a_reference_resolves_to_the_paragraph_the_chunker_actually_emitted():
    # 300.201 runs (a)(1) together with (a), so no "#a-1" is ever written and a reference to
    # (a)(1) is answered by (a). Reporting "300.201#a-1" missing would name an address the
    # store has never held.
    corpus = nameable_ids({"300.201#a", "300.201#b"})
    assert resolve("300.201#a-1", corpus) == "300.201#a"
    evidence = {"300.104#c@2020-01-01": "as provided in paragraph (a)(1) of § 300.201",
                "300.201#a@2020-01-01": "(a) The Office does not release the following: (1)"}
    assert dangling_references(evidence, in_corpus=corpus) == []


def test_a_descendant_covers_the_paragraph_it_hangs_off():
    evidence = {"630.306#a@2019-01-01": "as provided in paragraph (j) of § 890.102",
                "890.102#j-2@2020-01-01": "(2) ..."}
    assert dangling_references(evidence, in_corpus=CORPUS) == []


def test_an_ancestor_does_not_cover_its_descendant():
    # Both chunks point at 630.306(b) and neither of them is it, so both are reported: having
    # the chapeau of a paragraph says nothing about what its sub-paragraph requires.
    evidence = {"630.310#d@2019-01-01": "as provided in paragraph (b) of § 630.306",
                "630.306#a@2019-01-01": RESTORED_LEAVE}
    out = dangling_references(evidence, in_corpus=CORPUS)
    assert [(d.source, d.target) for d in out] == [
        ("630.306#a@2019-01-01", "630.306#b"), ("630.310#d@2019-01-01", "630.306#b")]


def test_dangling_references_are_ordered_and_deterministic():
    evidence = {"630.310#d@2019-01-01": "see § 630.306(b) and paragraph (j) of § 890.102",
                "630.306#a@2019-01-01": RESTORED_LEAVE}
    out = dangling_references(evidence, in_corpus=CORPUS)
    assert [d.source for d in out] == ["630.306#a@2019-01-01"] + ["630.310#d@2019-01-01"] * 2
    assert out == dangling_references(evidence, in_corpus=CORPUS)
