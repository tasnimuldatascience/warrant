"""Classifying snapshot-to-snapshot change.

Only ``substantive_localized`` grounds a benchmark question, so the cost of a
misclassification is a benchmark item whose ground truth is wrong -- the worst possible
failure for an evaluation harness, because it is invisible.
"""

from __future__ import annotations

from warrant.corpus.diff import Change, classify_pair, diff_snapshots

BEFORE = "Annual leave restored under 5 U.S.C. 6304(d) must be scheduled and used not later "
AFTER_LOCAL = BEFORE + "than the end of the leave year ending 2 years after restoration."


def test_identical_text_is_editorial():
    assert classify_pair("same text here", "same text here")[0] is Change.EDITORIAL


def test_punctuation_only_change_is_editorial():
    kind, _, _ = classify_pair(
        "the date fixed by the agency head; or, (3) the date the employee recovered",
        "the date fixed by the agency head; or (3) the date the employee recovered",
    )
    assert kind is Change.EDITORIAL


def test_tiny_change_is_editorial_not_substantive():
    """A one- or two-token difference is indistinguishable from typographic tidying, and a
    benchmark question generated from it would have no discernible answer."""
    kind, _, changed = classify_pair(BEFORE + "than the end of the leave year.",
                                     BEFORE + "than the close of the leave year.")
    assert kind is Change.EDITORIAL
    assert changed < 3


def test_localized_amendment_is_substantive():
    kind, ratio, changed = classify_pair(
        "(a) Except as authorized under paragraphs (b) and (c) of this section, annual "
        "leave restored under 5 U.S.C. 6304(d) must be scheduled and used not later than "
        "the end of the leave year ending 2 years after the date of restoration.",
        "(a) Except as authorized under paragraphs (b) and (c) of this section, section "
        "630.310(d), or other regulation, annual leave restored under 5 U.S.C. 6304(d) "
        "must be scheduled and used not later than the end of the leave year ending 2 "
        "years after the date of restoration.",
    )
    assert kind is Change.SUBSTANTIVE
    assert 0.5 < ratio < 1.0
    assert changed >= 3


def test_replacement_is_wholesale_not_substantive():
    kind, _, _ = classify_pair(
        "The agency shall grant leave without pay to an employee for military duty.",
        "Each agency must establish procedures for the administration of paid parental "
        "leave, including documentation, scheduling, and repayment obligations arising "
        "from a failure to complete the required work obligation.",
    )
    assert kind is Change.WHOLESALE


def test_apparatus_only_change_is_counted_when_raw_text_supplied():
    """Stripped text identical, raw text different -- publication churn, not an amendment.
    It is counted rather than dropped so the ingestion row of the budget stays honest."""
    stripped = {"630.1203": "An employee shall be entitled to 12 workweeks."}
    changes = diff_snapshots(
        stripped, dict(stripped), from_date="2018-05-10", to_date="2020-08-10",
        before_raw={"630.1203": "An employee shall be entitled to 12 workweeks."},
        after_raw={"630.1203": "Link to an amendment published at 85 FR 48090. "
                               "An employee shall be entitled to 12 workweeks."},
    )
    assert [c.kind for c in changes] == [Change.APPARATUS_ONLY]
    assert not changes[0].usable_for_benchmark


def test_added_and_removed_sections_are_reported():
    changes = diff_snapshots({"630.101": "old text about leave"},
                             {"630.102": "entirely different subject matter entirely"},
                             from_date="a", to_date="b")
    kinds = {c.identifier: c.kind for c in changes}
    assert kinds == {"630.101": Change.REMOVED, "630.102": Change.ADDED}


def test_renumbering_is_detected_rather_than_reported_as_add_plus_remove():
    """A section that moves is not an amendment. Counting it as one would put a question in
    the benchmark whose 'change' is an identifier, not a rule."""
    text = ("An employee shall be entitled to a total of 12 administrative workweeks of "
            "unpaid leave during any 12-month period for the birth of a son or daughter.")
    changes = diff_snapshots({"630.1203": text}, {"630.1205": text},
                             from_date="a", to_date="b")
    assert len(changes) == 1
    assert changes[0].kind is Change.RENUMBERED
    assert changes[0].renamed_to == "630.1205"


def test_only_substantive_changes_are_benchmark_usable():
    changes = diff_snapshots(
        {"a": "one two three four five six seven eight nine ten", "b": "kept identical"},
        {"a": "one two three four five six seven eight nine eleven twelve thirteen",
         "b": "kept identical"},
        from_date="a", to_date="b",
    )
    usable = [c for c in changes if c.usable_for_benchmark]
    assert [c.identifier for c in usable] == ["a"]
