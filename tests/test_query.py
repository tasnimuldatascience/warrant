"""The query-understanding stage.

Offline by construction: no store, no encoder, no generation. That is the property the stage
was built for, so it is also the property the tests hold it to -- every assertion here is a
regex and a date, and the whole file runs in milliseconds.

``TODAY`` is pinned. A stage whose output moves with the wall clock cannot be replayed, and
"currently" and "last year" are exactly where that would leak in.
"""

from __future__ import annotations

from datetime import date

from warrant.retrieve.query import (
    CLASSIFIER_RULES,
    LOOKUP_CONFIDENCE,
    SCOPE_RULES,
    QueryKind,
    classify,
    decontextualize,
    expand_abbreviations,
    extract_dates,
    extract_scope,
    find_citations,
    likely_multi_hop,
    normalise_citations,
    plan_query,
    rewrite,
)
from warrant.retrieve.scope import known_values

TODAY = date(2026, 8, 30)


def _classify(text: str):
    return classify(text, dates=extract_dates(text, today=TODAY))


#: A spread of shapes the benchmark and the human bucket actually produce, used by the
#: invariant tests below. Deliberately includes questions no rule matches.
QUERIES = (
    "How long is the probationary period for a new federal employee?",
    "What is the maximum annual leave carryover?",
    "What was the annual leave carryover limit as of June 2021?",
    "What changed in the annual leave rules between 2019 and 2023?",
    "Does the within-grade increase waiting period apply to a wage grade employee?",
    "Am I entitled to a within-grade increase?",
    "List all the parts that govern reduction in force.",
    "How many sections cover probationary periods?",
    "Is there any rule about remote work equipment reimbursement?",
    "Are there any provisions covering telework stipends?",
    "What are the RIF retention rules for a GS employee?",
    "How is SES performance appraised?",
    "Who may be appointed in the competitive service?",
    "Does probation apply in the excepted service?",
    "What does 5 CFR 630.306 say about carryover?",
    "Can I use sick leave to care for a family member?",
    "How does GS pay compare with wage grade pay?",
    "What did the rule say in 2019?",
    "What are the current rules on credit hours?",
    "Is annual leave as defined in 630.201 the same for part-time staff?",
    "FEHB enrollment during a RIF",
    "USERRA reemployment rights after military service",
)


# -- classification -----------------------------------------------------------------


def test_lookup_is_the_default_not_a_guess():
    """An unrecognised question is routed nowhere, at a confidence that says so. Guessing a
    route is worse than declining one: the unrouted query still gets the hybrid answer."""
    for query in ("How long is the probationary period for a new federal employee?",
                  "What is the maximum annual leave carryover?"):
        result = _classify(query)
        assert result.kind is QueryKind.LOOKUP, query
        assert result.confidence == LOOKUP_CONFIDENCE
    assert _classify("What was the carryover limit as of June 2021?").kind is not QueryKind.LOOKUP


def test_temporal_point_needs_a_date_or_an_as_of():
    for query in ("What was the annual leave carryover limit as of June 2021?",
                  "Back in 2018, how much leave could be carried over?"):
        assert _classify(query).kind is QueryKind.TEMPORAL_POINT, query
    # Two dates is a comparison, not a point.
    assert _classify("What changed between 2019 and 2021?").kind is not QueryKind.TEMPORAL_POINT


def test_past_tense_alone_does_not_route_to_a_date_nobody_named():
    """"What was the probationary period" is a phrasing habit. Routing it to temporal_point
    would claim a date the query never gave."""
    assert _classify("What was the probationary period?").kind is QueryKind.LOOKUP


def test_temporal_compare_is_recognised_however_it_is_phrased():
    for query in ("What changed in the annual leave rules between 2019 and 2023?",
                  "How has the probationary period changed since 2017?"):
        assert _classify(query).kind is QueryKind.TEMPORAL_COMPARE, query
    assert _classify("What is the probationary period?").kind is not QueryKind.TEMPORAL_COMPARE


def test_applicability_is_a_predicate_question():
    for query in ("Does the within-grade increase waiting period apply to a wage grade "
                  "employee?",
                  "Am I entitled to a within-grade increase?"):
        assert _classify(query).kind is QueryKind.APPLICABILITY, query
    assert _classify("List all the parts that govern RIF.").kind is not QueryKind.APPLICABILITY


def test_aggregate_is_a_set_question_over_the_corpus():
    for query in ("List all the parts that govern reduction in force.",
                  "How many sections cover probationary periods?"):
        assert _classify(query).kind is QueryKind.AGGREGATE, query
    # "How much" is a value question about one rule, not a count over the corpus.
    assert _classify("How much annual leave do I accrue?").kind is not QueryKind.AGGREGATE


def test_absence_is_recognised_so_nothing_can_be_a_correct_answer():
    for query in ("Is there any rule about remote work equipment reimbursement?",
                  "Are there any provisions covering telework stipends?"):
        assert _classify(query).kind is QueryKind.ABSENCE, query
    assert _classify("What is the rule on remote work?").kind is not QueryKind.ABSENCE


def test_every_classifier_rule_still_matches_and_wins_on_its_own_example():
    """The example is a field precisely so this can be asserted. A rule whose example has
    drifted out from under it is a pattern nobody can justify any more."""
    for rule in CLASSIFIER_RULES:
        assert rule.pattern.search(rule.example), rule.name
        if rule.standalone:
            assert _classify(rule.example).kind is rule.kind, rule.name


def test_confidence_rises_when_more_than_one_pattern_agrees():
    one = _classify("Am I entitled to a within-grade increase?")
    two = _classify("Does this apply to me, and am I eligible for it?")
    assert two.confidence > one.confidence
    assert len(two.signals) > 1


# -- dates --------------------------------------------------------------------------


def test_explicit_dates_in_every_written_form():
    assert [h.iso for h in extract_dates("as of 2019-06-01", today=TODAY)] == ["2019-06-01"]
    assert [h.iso for h in extract_dates("on June 1, 2021", today=TODAY)] == ["2021-06-01"]
    assert [h.iso for h in extract_dates("as of 1 June 2021", today=TODAY)] == ["2021-06-01"]


def test_a_partial_date_resolves_to_the_close_of_the_period_it_names():
    """"As of June 2021" means as of the close of June, which is the reading that makes the
    date recorded on the trace a date the corpus can be asked about."""
    (month,) = extract_dates("as of June 2021", today=TODAY)
    assert (month.iso, month.granularity) == ("2021-06-30", "month")
    (year,) = extract_dates("what did it say in 2019", today=TODAY)
    assert (year.iso, year.granularity) == ("2019-12-31", "year")


def test_before_an_amendment_is_the_day_the_period_had_not_started():
    (hint,) = extract_dates("what did it say before the 2020 amendment", today=TODAY)
    assert hint.iso == "2019-12-31"
    # The whole phrase is captured so the rewriter can remove exactly what was consumed.
    assert "before the 2020 amendment" in hint.text.lower()


def test_after_an_amendment_is_the_close_of_that_period():
    (hint,) = extract_dates("the rule after the 2020 amendment", today=TODAY)
    assert hint.iso == "2020-12-31"


def test_a_range_yields_two_dates_and_a_comparison():
    plan = plan_query("What changed between 2019 and 2021?", today=TODAY)
    assert plan.kind is QueryKind.TEMPORAL_COMPARE
    assert plan.compare_dates == ("2019-12-31", "2021-12-31")
    # The single-as_of API gets the later half rather than an arbitrary one.
    assert plan.as_of == "2021-12-31"


def test_a_range_is_read_once_and_not_also_as_two_bare_years():
    assert len(extract_dates("between 2019 and 2023", today=TODAY)) == 2


def test_relative_expressions_resolve_against_the_supplied_today():
    (now,) = extract_dates("what are the current rules", today=TODAY)
    assert now.iso == TODAY.isoformat() and now.is_now
    (past,) = extract_dates("what did it say last year", today=TODAY)
    assert (past.iso, past.is_now) == ("2025-12-31", False)


def test_a_date_that_has_not_happened_is_clamped_to_today():
    """2026 has not finished. An as-of in the future would answer from today's rows while
    claiming a date the corpus cannot know anything about."""
    (hint,) = extract_dates("in 2026", today=TODAY)
    assert hint.iso == TODAY.isoformat()


def test_asking_about_now_is_not_a_temporal_question():
    plan = plan_query("What are the current rules on credit hours?", today=TODAY)
    assert plan.kind is QueryKind.LOOKUP
    assert plan.as_of == TODAY.isoformat()


def test_an_unreal_calendar_date_is_not_a_date():
    assert extract_dates("as of February 30, 2021", today=TODAY) == []


# -- scope --------------------------------------------------------------------------


def test_facet_values_are_read_from_the_wording_the_parts_use():
    assert extract_scope("Does this apply to a GS employee?")[0].facets == {"pay_system": "GS"}
    assert extract_scope("pay for a wage grade employee")[0].facets == {"pay_system": "FWS"}
    assert extract_scope("prevailing rate systems")[0].facets == {"pay_system": "FWS"}
    assert extract_scope("appointment in the competitive service")[0].facets == {
        "service": "competitive"}
    assert extract_scope("an excepted service appointment")[0].facets == {"service": "excepted"}


def test_ses_settles_both_facets_because_it_is_both():
    """pay_system=SES keeps 511/531/532 out; service=SES keeps out 315/316/337, which govern
    the competitive service only. Setting one and not the other would half-filter."""
    scope, _ = extract_scope("How is SES performance appraised?")
    assert scope.facets == {"pay_system": "SES", "service": "SES"}


def test_no_scope_is_inferred_from_a_query_that_names_none():
    """The expensive failure. An unrequested pay_system filter removes five parts -- 41% of
    the corpus -- before the retriever has ranked anything."""
    for query in ("How much annual leave does a federal employee accrue?",
                  "What is the probationary period?",
                  "Can I use sick leave to care for a family member?"):
        scope, evidence = extract_scope(query)
        assert scope.facets == {}, query
        assert evidence == {}, query


def test_a_query_naming_two_pay_systems_infers_neither():
    """It is a comparison, and answering it under either system answers a different
    question."""
    scope, evidence = extract_scope("How does GS pay compare with wage grade pay?")
    assert scope.facets == {}
    assert evidence == {}


def test_a_conflicting_query_keeps_its_wording_for_retrieval():
    plan = plan_query("How does GS pay compare with wage grade pay?", today=TODAY)
    assert plan.scope.facets == {}
    lowered = plan.retrieval_query.lower()
    assert "gs" in lowered and "wage grade" in lowered


def test_every_scope_rule_still_matches_its_own_example():
    for rule in SCOPE_RULES:
        assert rule.pattern.search(rule.example), rule.name


def test_plan_never_proposes_a_facet_value_outside_the_declared_vocabulary():
    """`Scope.of` rejects an unknown value with a ValueError, so an invented one here would
    be a 500 on a request rather than a silent filter -- but only if it ever reached it."""
    for query in QUERIES:
        plan = plan_query(query, today=TODAY)
        for facet, value in plan.scope.facets.items():
            assert value in known_values(facet), (query, facet, value)


# -- rewriting ----------------------------------------------------------------------


def test_interrogative_scaffolding_is_stripped():
    assert rewrite("What is the maximum annual leave carryover?") == (
        "maximum annual leave carryover")
    assert rewrite("How long is the probationary period?") == "probationary period"
    assert rewrite("Can I carry over annual leave?") == "carry over annual leave"


def test_abbreviations_are_expanded_beside_the_acronym_not_instead_of_it():
    """eCFR spells out "reduction in force" in the operative text and uses "RIF" as a defined
    term; keeping only one form loses the paragraphs that use the other."""
    out = rewrite("What are the RIF retention rules?")
    assert "RIF" in out
    assert "reduction in force" in out.lower()
    assert "WGI within-grade increase" in expand_abbreviations("WGI")


def test_an_abbreviation_already_spelled_out_is_not_duplicated():
    out = expand_abbreviations("reduction in force (RIF) retention")
    assert out.lower().count("reduction in force") == 1


def test_dates_and_scope_leave_the_text_once_they_are_structured():
    """Leaving "2019" in the text makes BM25 match the token in every unrelated paragraph
    that cites a 2019 notice; the date is a predicate now."""
    plan = plan_query("What was the annual leave carryover for a GS employee in 2019?",
                      today=TODAY)
    assert "2019" not in plan.retrieval_query
    assert "GS" not in plan.retrieval_query
    assert plan.as_of == "2019-12-31"
    assert plan.scope.facets == {"pay_system": "GS"}
    assert "annual leave carryover" in plan.retrieval_query.lower()


def test_stripping_gives_back_the_scope_wording_when_it_was_the_subject():
    """"Prevailing rate" is both a facet signal and the whole topic. The predicate still
    applies; only the lexical text keeps the words."""
    plan = plan_query("What is prevailing rate pay?", today=TODAY)
    assert "prevailing rate" in plan.retrieval_query.lower()
    assert plan.scope.facets == {"pay_system": "FWS"}


def test_a_retrieval_query_is_never_empty():
    for query in (*QUERIES, "?", "it", "what about that"):
        plan = plan_query(query, today=TODAY)
        assert plan.retrieval_query.strip(), query


# -- citations ----------------------------------------------------------------------


def test_the_three_citation_forms_reach_the_index_the_same_way():
    forms = ("§630.306", "§ 630.306", "630.306", "5 CFR 630.306", "5 C.F.R. § 630.306",
             "section 630.306")
    rendered = {normalise_citations(f"what does {f} say about carryover") for f in forms}
    assert rendered == {"what does 630.306 say about carryover"}
    for form in forms:
        assert find_citations(f"what does {form} say") == ["630.306"], form


def test_a_paragraph_letter_is_dropped_with_the_rest_of_the_punctuation():
    """"(a)" tokenises to "a", which matches everything and ranks nothing."""
    assert find_citations("5 CFR 630.306(a)") == ["630.306"]
    assert normalise_citations("see 630.306(a)") == "see 630.306"


def test_a_decimal_that_is_not_a_citation_is_left_alone():
    assert find_citations("a similarity of 0.95 over 2019.06 rows") == []


# -- multi-hop ----------------------------------------------------------------------


def test_a_cross_reference_question_is_flagged_and_not_followed():
    assert plan_query("Is annual leave as defined in 630.201 the same for part-time staff?",
                      today=TODAY).needs_multi_hop
    assert likely_multi_hop("what are the exceptions to the carryover ceiling")
    assert not plan_query("How much annual leave does an employee accrue?",
                          today=TODAY).needs_multi_hop


def test_two_citations_in_one_question_is_a_relationship_question():
    assert likely_multi_hop("does 630.306 override 630.301", citations=["630.306", "630.301"])


# -- multi-turn ---------------------------------------------------------------------


def test_a_follow_up_inherits_the_subject_the_scope_and_the_as_of():
    """The failure this whole system exists to catch: a rewriter that drops the inherited
    profile answers for the wrong person, and one that drops the date answers for the wrong
    year, both with full confidence and a clean trace."""
    first = plan_query("How much annual leave does a GS employee accrue as of June 2021?",
                       today=TODAY)
    assert first.scope.facets == {"pay_system": "GS"}
    assert first.as_of == "2021-06-30"

    second = decontextualize("What about for part-time employees?", [first], today=TODAY)
    assert second.scope.facets == first.scope.facets
    assert second.as_of == first.as_of
    assert "annual leave" in second.retrieval_query.lower()
    assert "part-time" in second.retrieval_query.lower()
    assert set(second.inherited) == {"subject", "scope", "as_of"}


def test_a_follow_up_that_names_its_own_scope_does_not_inherit_the_old_one():
    first = plan_query("annual leave accrual for a GS employee", today=TODAY)
    second = decontextualize("what about wage grade employees?", [first], today=TODAY)
    assert second.scope.facets == {"pay_system": "FWS"}
    assert "scope" not in second.inherited


def test_a_follow_up_that_names_its_own_date_does_not_inherit_the_old_one():
    first = plan_query("annual leave carryover as of June 2021", today=TODAY)
    second = decontextualize("and what about in 2019?", [first], today=TODAY)
    assert second.as_of == "2019-12-31"
    assert "as_of" not in second.inherited


def test_a_pronoun_is_resolved_against_the_previous_subject():
    first = plan_query("What is the annual leave carryover ceiling?", today=TODAY)
    second = decontextualize("Does it apply to me?", [first], today=TODAY)
    assert "carryover" in second.retrieval_query.lower()
    assert second.kind is QueryKind.APPLICABILITY


def test_a_self_contained_follow_up_is_not_merged():
    first = plan_query("What is the annual leave carryover ceiling?", today=TODAY)
    second = decontextualize("How is a competitive area defined for a reduction in force?",
                             [first], today=TODAY)
    assert "carryover" not in second.retrieval_query.lower()
    assert "subject" not in second.inherited


def test_the_merged_query_does_not_repeat_a_term():
    """`hybrid.fts_query` deduplicates anyway; repeating here would still spend the 64-token
    budget twice on the same postings list."""
    first = plan_query("annual leave carryover ceiling", today=TODAY)
    second = decontextualize("what about the annual leave ceiling for part-time staff?",
                             [first], today=TODAY)
    tokens = [t.lower() for t in second.retrieval_query.split()]
    assert len(tokens) == len(set(tokens))


# -- invariants ---------------------------------------------------------------------


def test_planning_is_deterministic():
    for query in QUERIES:
        assert plan_query(query, today=TODAY) == plan_query(query, today=TODAY), query


def test_confidence_is_always_a_probability():
    for query in QUERIES:
        plan = plan_query(query, today=TODAY)
        assert 0.0 < plan.confidence <= 1.0, query


def test_compare_dates_are_only_ever_set_for_a_comparison():
    for query in QUERIES:
        plan = plan_query(query, today=TODAY)
        if plan.compare_dates is not None:
            assert plan.kind is QueryKind.TEMPORAL_COMPARE, query
            assert plan.as_of == plan.compare_dates[1]


def test_as_of_or_falls_back_to_the_callers_default():
    plan = plan_query("What is the probationary period?", today=TODAY)
    assert plan.as_of is None
    assert plan.as_of_or("2026-01-01") == "2026-01-01"
    dated = plan_query("What was the rule in 2019?", today=TODAY)
    assert dated.as_of_or("2026-01-01") == "2019-12-31"
