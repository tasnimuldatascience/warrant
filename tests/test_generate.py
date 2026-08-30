"""Grounding: parsing the model's claims, and locating the span that supports each.

No model runs here. The two things worth pinning down are pure: what happens to a citation
the model invented, and when the aligner refuses to find a span. Both fail silently in the
worst way — an invented citation and a whole-chunk span both render as a grounded answer.
"""

from __future__ import annotations

from warrant.generate.answer import build_prompt, ground, parse_response
from warrant.verify.align import align

EXCERPTS = [
    ("630.306#a@2017-01-01", "Time limit for use of restored annual leave",
     "(a) Except as otherwise authorized under paragraphs (b) and (c) of this section, "
     "annual leave restored under 5 U.S.C. 6304(d) must be scheduled and used not later "
     "than the end of the leave year ending 2 years after the date of restoration."),
    ("315.904#a@2017-01-01", "Basic requirement",
     "(a) An employee is required to serve a probationary period of one year."),
]


# -- prompt ----------------------------------------------------------------------


def test_excerpts_are_numbered_not_addressed_by_version_id():
    """A 1.5B model copies 630.306#a@2020-08-10 wrongly often enough to matter, and a
    mis-copied citation is indistinguishable from a hallucinated one downstream."""
    user = build_prompt("q", EXCERPTS)[1]["content"]
    assert "[1]" in user and "[2]" in user
    assert "630.306#a@2017-01-01" not in user


def test_prompt_carries_the_heading_with_the_text():
    user = build_prompt("q", EXCERPTS)[1]["content"]
    assert "Time limit for use of restored annual leave" in user


# -- parsing ---------------------------------------------------------------------


def test_excerpt_numbers_map_back_to_version_ids():
    claims, found = parse_response(
        '{"claims": [{"text": "Two years.", "evidence": [1]}], "answer_found": true}',
        EXCERPTS)
    assert found
    assert claims[0].evidence == ["630.306#a@2017-01-01"]


def test_json_embedded_in_chatter_is_recovered():
    """Small instruct models preface JSON with prose no matter how the prompt is worded."""
    claims, _ = parse_response(
        'Sure! Here is the JSON:\n{"claims": [{"text": "T.", "evidence": [2]}], '
        '"answer_found": true}\nHope that helps.', EXCERPTS)
    assert claims[0].evidence == ["315.904#a@2017-01-01"]


def test_out_of_range_citation_is_dropped_not_clamped():
    """A citation to excerpt 9 when 2 were offered is hallucinated. Rewriting it to the
    nearest real excerpt would manufacture grounding out of a mistake."""
    claims, _ = parse_response(
        '{"claims": [{"text": "T.", "evidence": [9]}], "answer_found": true}', EXCERPTS)
    assert claims[0].evidence == []
    assert not claims[0].grounded


def test_duplicate_citations_are_collapsed():
    claims, _ = parse_response(
        '{"claims": [{"text": "T.", "evidence": [1, 1, 2]}], "answer_found": true}',
        EXCERPTS)
    assert claims[0].evidence == ["630.306#a@2017-01-01", "315.904#a@2017-01-01"]


def test_unparseable_response_returns_none_rather_than_an_empty_answer():
    """None means "retry"; an empty claim list would mean "the model abstained". Conflating
    them would report a broken parse as a considered refusal."""
    assert parse_response("I cannot answer that.", EXCERPTS) is None
    assert parse_response('{"oops": true}', EXCERPTS) is None
    assert parse_response('{"claims": [{"text": "x", "evidence": [1]}', EXCERPTS) is None


def test_explicit_abstention_is_parsed_as_an_answer_not_a_failure():
    claims, found = parse_response('{"claims": [], "answer_found": false}', EXCERPTS)
    assert claims == [] and found is False


def test_claim_without_text_is_dropped():
    claims, _ = parse_response(
        '{"claims": [{"text": "  ", "evidence": [1]}], "answer_found": true}', EXCERPTS)
    assert claims == []


# -- alignment -------------------------------------------------------------------


SOURCE = EXCERPTS[0][2]


def test_span_is_located_for_a_supported_claim():
    span = align("Restored annual leave must be scheduled and used within two years.", SOURCE)
    assert span is not None
    assert "scheduled and used" in span.text_of(SOURCE)


def test_no_span_for_an_unsupported_claim():
    """The aligner refusing is a finding about the answer, not a failure of the aligner:
    the model cited a chunk in which no supporting text exists. Returning the whole chunk
    would turn an ungrounded claim into a grounded-looking one."""
    assert align("Employees receive twelve weeks of paid parental leave.", SOURCE) is None


def test_shortest_window_wins_among_equally_supporting_ones():
    source = ("(a) The agency shall act. (b) The waiting period is one year. "
              "(c) Other provisions apply.")
    span = align("The waiting period is one year.", source)
    assert span is not None
    assert span.text_of(source).strip().startswith("(b)")


def test_citations_are_not_shattered_by_the_sentence_splitter():
    """Regulatory prose is full of '5 U.S.C. 6304(d)' and '§ 630.306'; splitting on every
    period would cut a citation in half and make its span unlocatable."""
    span = align("Annual leave restored under 5 U.S.C. 6304(d) must be scheduled.", SOURCE)
    assert span is not None
    assert "6304(d)" in span.text_of(SOURCE)


def test_empty_or_stopword_only_claim_yields_no_span():
    assert align("", SOURCE) is None
    assert align("It is that and the of.", SOURCE) is None


def test_ground_marks_a_claim_ungrounded_when_no_span_is_found():
    cited = {"630.306#a@2017-01-01": SOURCE}
    claims, _ = parse_response(
        '{"claims": [{"text": "Employees get twelve weeks of paid parental leave.", '
        '"evidence": [1]}], "answer_found": true}', EXCERPTS)
    grounded = ground(claims, cited)
    assert grounded[0].spans["630.306#a@2017-01-01"] is None
    assert not grounded[0].grounded
