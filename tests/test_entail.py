"""Tests for `verify.entail`.

Split in two. Everything that decides what a score *means* -- calibration, the decision
bands, the aggregation, the combination with span alignment -- is pure Python and is tested
without weights, because that is the part a reviewer reads and the part most likely to be
wrong in a way no metric would catch. Everything that needs the 377 MB checkpoint carries
`@pytest.mark.neural`, which the default `pytest` invocation excludes, and additionally skips
when the weights are not already in the HuggingFace cache so that `-m neural` on a fresh
clone reports "skipped" rather than trying to reach the network mid-suite.

The neural tests are regression tests against `docs/results/eval-007-entailment.md`. They
assert the floors that document measured, not aspirations: 25 of the probe set's harder
pairs, and the eight adversarial contradictions that are the module's actual value.
"""

from __future__ import annotations

import math

import pytest

from warrant.verify.align import align
from warrant.verify.entail import (
    CONTRADICT_FLOOR,
    CONTRADICTED,
    DECISION_FLOOR,
    DEFAULT_MODEL,
    LABELS,
    SUPPORTED,
    UNCERTAIN,
    UNSUPPORTED,
    Bin,
    ClaimSupport,
    Entailer,
    Verdict,
    _chunks,
    _label_order,
    brier,
    combine,
    expected_calibration_error,
    fit_temperature,
    reliability,
    softmax,
)


def weights_cached(model: str = DEFAULT_MODEL) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return try_to_load_from_cache(model, "config.json") is not None


needs_weights = pytest.mark.skipif(not weights_cached(),
                                   reason="NLI weights not in the local cache")


# -- calibration, without weights -------------------------------------------------


def test_softmax_normalises_and_is_stable_on_large_logits():
    probs = softmax([900.0, 899.0, 1.0])
    assert math.isclose(sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_temperature_flattens_without_reordering():
    """The property the whole calibration story rests on: temperature scaling is monotone,
    so it can move a confidence but never a decision. If this ever fails, every accuracy in
    the results doc has to be re-run against the calibrated head instead of the raw one."""
    logits = [3.0, 1.0, -2.0]
    hot = softmax(logits, temperature=2.5)
    cold = softmax(logits, temperature=1.0)
    assert max(hot) < max(cold)
    assert hot.index(max(hot)) == cold.index(max(cold))


def test_fit_temperature_finds_the_overconfident_direction():
    # Confidently right most of the time and confidently wrong the rest: NLL is minimised by
    # softening, which is the shape of a model carried out of its training domain.
    logits = [[6.0, 0.0, 0.0]] * 8 + [[6.0, 0.0, 0.0]] * 2
    gold = [0] * 8 + [1, 2]
    assert fit_temperature(logits, gold) > 1.5


def test_fit_temperature_on_nothing_is_the_identity():
    assert fit_temperature([], []) == 1.0


def test_fit_temperature_cannot_change_accuracy():
    logits = [[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [0.0, 0.5, 4.0], [1.0, 0.9, 0.8]]
    gold = [0, 1, 2, 1]
    t = fit_temperature(logits, gold)
    before = [row.index(max(row)) for row in logits]
    after = [list(softmax(row, temperature=t)).index(max(softmax(row, temperature=t)))
             for row in logits]
    assert before == after


def test_reliability_drops_empty_bins_and_reports_the_gap():
    bins = reliability([0.95, 0.92, 0.55], [True, False, True], bins=10)
    assert [b.lo for b in bins] == [0.5, 0.9]
    top = bins[-1]
    assert top.n == 2 and top.accuracy == 0.5
    assert top.gap > 0            # confident and half wrong: overconfident


def test_a_perfectly_calibrated_set_has_zero_ece():
    conf = [0.75] * 4
    assert expected_calibration_error(conf, [True, True, True, False]) == pytest.approx(0.0)


def test_ece_is_zero_for_a_uniformly_useless_model_and_brier_is_not():
    """Why both numbers are published. A model that answers 0.5/0.5 on everything and is
    right half the time is perfectly calibrated and carries no information at all."""
    probs = [[0.5, 0.5, 0.0]] * 4
    conf = [0.5] * 4
    correct = [True, False, True, False]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)
    assert brier(probs, [0, 1, 0, 1]) > 0.4


def test_brier_rewards_the_confident_and_correct():
    gold = [0, 1]
    sharp = brier([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], gold)
    flat = brier([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]], gold)
    assert sharp < flat


def test_bin_gap_is_confidence_minus_accuracy():
    assert Bin(0.8, 0.9, 10, 0.85, 0.60).gap == pytest.approx(0.25)


# -- what a verdict licenses saying -----------------------------------------------


def test_the_argmax_is_reported_but_is_not_the_verdict():
    """The docstring's own example. An argmax exists for every distribution, including one
    that says nothing, and publishing that as support is how a signal becomes an authority."""
    v = Verdict(entail=0.41, neutral=0.39, contradict=0.20)
    assert v.label == "entail"
    assert v.report == UNCERTAIN


def test_a_confident_entailment_is_supported():
    assert Verdict(0.93, 0.05, 0.02).report == SUPPORTED


def test_a_confident_neutral_is_unsupported_not_contradicted():
    """`unsupported` and `contradicted` are different findings and the API must not blur
    them: the first says the citation does not carry the claim, the second says the cited
    text denies it."""
    assert Verdict(0.04, 0.93, 0.03).report == UNSUPPORTED


def test_contradiction_is_reported_at_a_lower_bar_than_support():
    below_the_decision_floor = Verdict(0.20, 0.25, 0.55)
    assert below_the_decision_floor.confidence < DECISION_FLOOR
    assert below_the_decision_floor.contradict >= CONTRADICT_FLOOR
    assert below_the_decision_floor.report == CONTRADICTED


def test_the_uncertain_band_is_where_the_floors_say_it_is():
    just_under = Verdict(DECISION_FLOOR - 0.01, 0.20, 0.11)
    just_over = Verdict(DECISION_FLOOR + 0.01, 0.19, 0.10)
    assert just_under.report == UNCERTAIN
    assert just_over.report == SUPPORTED


def test_confidence_is_the_top_probability():
    assert Verdict(0.1, 0.7, 0.2).confidence == pytest.approx(0.7)
    assert Verdict(0.1, 0.7, 0.2).label == LABELS[1]


# -- aggregating one claim over several citations ---------------------------------


def test_a_claim_needs_one_good_citation_not_a_good_average():
    support = ClaimSupport("c", {
        "a": Verdict(0.95, 0.03, 0.02),
        "b": Verdict(0.02, 0.95, 0.03),
        "c": Verdict(0.03, 0.94, 0.03),
    })
    assert support.report == SUPPORTED
    assert support.best[0] == "a"


def test_contradiction_is_surfaced_alongside_support_not_instead_of_it():
    """A claim entailed by the version in force and denied by the one it superseded is the
    correct answer to a temporal question, not a conflict to resolve, so both are reported."""
    support = ClaimSupport("c", {"now": Verdict(0.95, 0.03, 0.02),
                                 "then": Verdict(0.02, 0.08, 0.90)})
    assert support.report == SUPPORTED
    assert support.contradicted_by == ["then"]


def test_a_claim_citing_nothing_is_uncertain_not_unsupported():
    empty = ClaimSupport("c")
    assert empty.report == UNCERTAIN
    assert empty.best is None
    assert empty.contradicted_by == []


def test_contradiction_outranks_uncertainty_when_nothing_supports():
    support = ClaimSupport("c", {"a": Verdict(0.30, 0.35, 0.35),
                                 "b": Verdict(0.05, 0.10, 0.85)})
    assert support.report == CONTRADICTED


# -- the two signals together -----------------------------------------------------


def _span(text="an employee shall schedule restored leave by the end of the leave year"):
    return align("an employee shall schedule restored leave", text)


def test_a_span_and_an_entailment_agreeing_is_support():
    assert combine(_span(), ClaimSupport("c", {"a": Verdict(0.95, 0.03, 0.02)})) == SUPPORTED


def test_a_span_over_text_that_denies_the_claim_is_the_case_this_module_exists_for():
    """Lexical overlap cannot see polarity: "may" and "shall" share every content word the
    aligner counts. A span plus a contradiction is a citation pointing at text that denies
    the claim, and it must not come back as support."""
    support = ClaimSupport("c", {"a": Verdict(0.03, 0.12, 0.85)})
    assert _span() is not None
    assert combine(_span(), support) == CONTRADICTED


def test_an_entailment_with_no_span_says_the_span_is_what_is_missing():
    assert combine(None, ClaimSupport("c", {"a": Verdict(0.95, 0.03, 0.02)})) == SUPPORTED


def test_neither_signal_is_ungrounded():
    assert combine(None, ClaimSupport("c", {"a": Verdict(0.02, 0.95, 0.03)})) == UNSUPPORTED


def test_a_span_the_model_will_not_confirm_is_uncertain_not_support():
    assert combine(_span(), ClaimSupport("c", {"a": Verdict(0.45, 0.35, 0.20)})) == UNCERTAIN


def test_combine_never_claims_a_sentence_about_the_law_is_verified():
    outcomes = {combine(span, ClaimSupport("c", {"a": Verdict(*p)}))
                for span in (None, _span())
                for p in ((0.95, 0.03, 0.02), (0.02, 0.95, 0.03), (0.03, 0.12, 0.85),
                          (0.41, 0.39, 0.20))}
    assert outcomes <= {SUPPORTED, UNSUPPORTED, CONTRADICTED, UNCERTAIN}


# -- label order ------------------------------------------------------------------


def test_label_order_is_read_from_the_checkpoint():
    assert _label_order({0: "entailment", 1: "neutral", 2: "contradiction"}) == (0, 1, 2)
    assert _label_order({0: "contradiction", 1: "neutral", 2: "entailment"}) == (2, 1, 0)
    assert _label_order({0: "NEUTRAL", 1: "Entailment", 2: "contradict"}) == (1, 0, 2)


def test_an_unnamed_head_raises_rather_than_being_guessed():
    """A checkpoint labelled LABEL_0/1/2 states nothing about its own ordering. Guessing
    produces a verifier that reports every contradiction as support: in range, green, and
    inverted in every published number."""
    with pytest.raises(ValueError):
        _label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_a_two_way_head_raises():
    with pytest.raises(ValueError):
        _label_order({0: "entailment", 1: "not_entailment"})


# -- batching, without weights ----------------------------------------------------


def test_chunks_covers_every_item_once():
    items = list(range(23))
    assert [x for c in _chunks(items, 5) for x in c] == items
    assert [len(c) for c in _chunks(items, 5)] == [5, 5, 5, 5, 3]


def test_chunks_survives_a_zero_batch_size():
    assert [len(c) for c in _chunks([1, 2, 3], 0)] == [1, 1, 1]


def test_scoring_nothing_loads_nothing():
    """Reached by any claim whose every cited chunk is missing from the context. Loading
    377 MB of weights to score zero pairs would put a model download on a path that has no
    pairs to score, which on a clone with no weights is an exception rather than a no-op."""
    entailer = Entailer(model_name="a-model-that-does-not-exist")
    assert entailer.logits([]) == []
    assert entailer.score([]) == []
    support = entailer.score_claim("a claim", {})
    assert support.verdicts == {} and support.report == UNCERTAIN


def test_throughput_is_zero_before_anything_is_scored():
    assert Entailer().throughput == 0.0


# -- with weights -----------------------------------------------------------------

#: 25 pairs of the eval-007 probe set, chosen for difficulty rather than representativeness:
#: seven entailments including two the model gets wrong, seven neutrals of the kind the
#: generator actually produces, and eleven contradictions. Overall accuracy on the full
#: 182-pair set is 87.4%; on this subset it is 20/25, and the floor below leaves room for a
#: checkpoint revision to move a case without turning the suite red.
PAIRS = [
    ("gen-4", "315.907",
     "(a) Satisfactory completion of the prescribed probationary period is a "
     "prerequisite to continued service in the position. An employee who, for reasons of "
     "supervisory or managerial performance, does not satisfactorily complete the "
     "probationary period is entitled to be assigned, except as provided in paragraph "
     "(b) of this section, to a position in the agency of no lower grade and pay than "
     "the one the employee left to accept the supervisory or managerial position.",
     "Satisfactory completion of the prescribed probationary period is a prerequisite to "
     "continued service in the position. An employee who, for reasons of supervisory or "
     "managerial performance, does not satisfactorily complete the probationary period "
     "is entitled to be assigned, except as provided in paragraph (b) of this section, "
     "to a position in the agency of no lower grade and pay than the one the employee "
     "left to accept the supervisory or managerial position.",
     "E"),
    ("gen-37", "351.604",
     "(a) An agency may furlough a competing employee only when it intends within 1 year "
     "to recall the employee to duty in the position from which furloughed.",
     "An agency may furlough a competing employee only when it intends within 1 year to "
     "recall the employee to duty in the position from which furloughed.",
     "E"),
    ("gen-55", "432.102",
     "(a) Actions covered. This part covers reduction in grade and removal of employees "
     "based on unacceptable performance.",
     "Actions covered include reduction in grade and removal of employees based on "
     "unacceptable performance.",
     "E"),
    ("gen-87", "550.171",
     "(a) An employee is entitled to pay at his or her rate of basic pay plus premium "
     "pay at a rate equal to 25 percent of his or her rate of basic pay for each hour of "
     "Sunday work (as defined in § 550.103).",
     "An employee is entitled to pay at his or her rate of basic pay plus premium pay at "
     "a rate equal to 25 percent of his or her rate of basic pay for each hour of Sunday "
     "work.",
     "E"),
    ("gen-125", "630.306",
     "(1) A full-time employee shall schedule and use excess annual leave of 416 hours "
     "or less by the end of the leave year in progress 2 years after the date the "
     "employee is no longer subject to 5 U.S.C. 6304(d)(3). The agency shall extend this "
     "period by 1 leave year for each additional 208 hours of excess annual leave or any "
     "portion thereof.",
     "A full-time employee shall schedule and use excess annual leave of 416 hours or "
     "less by the end of the leave year in progress 2 years after the date the employee "
     "is no longer subject to 5 U.S.C. 6304(d)(3).",
     "E"),
    # Both of the next two are entailed and the model calls them neutral. They are in the
    # subset on purpose: a regression test that only contains cases the model passes is a
    # test of nothing.
    ("gen-0", "315.905",
     "The authority to determine the length of the probationary period is delegated to "
     "the head of each agency, provided that it be of reasonable fixed duration, "
     "appropriate to the position, and uniformly applied. An agency may establish "
     "different probationary periods for different occupations or a single one for all "
     "agency employees.",
     "The probationary period for a new federal employee is determined by the head of "
     "each agency.",
     "E"),
    ("gen-91", "532.207",
     "(f) The beginning month of appropriated and nonappropriated fund wage surveys and "
     "the fiscal year during which full-scale surveys will be conducted are set out as "
     "appendices A and B to this subpart and are incorporated in and made part of this "
     "section.",
     "The time schedule for wage surveys specifies the beginning month of appropriated "
     "and nonappropriated fund wage surveys and the fiscal year during which full-scale "
     "surveys will be conducted.",
     "E"),
    ("gen-9", "316.401",
     "(a) Appropriate use. An agency may make a temporary limited appointment—",
     "An agency may make a temporary appointment under part 332 of this chapter, by "
     "using competitive procedures, or under part 337 of this chapter, by using "
     "direct-hire procedures, as appropriate.",
     "N"),
    ("gen-27", "351.404",
     "(a) When a competing employee is to be released from a competitive level under "
     "this part, the agency shall establish a separate retention register for that "
     "competitive level. The retention register is prepared from the current retention "
     "records of employees. Upon displacing another employee under this part, an "
     "employee retains the same status and tenure in the new position. Except for an "
     "employee on military duty with a restoration right, the agency shall enter on the "
     "retention register, in the order of retention standing, the name of each competing "
     "employee who is:",
     "An agency must establish a retention register for a competitive level.",
     "N"),
    ("gen-34", "351.608",
     "(1) Notify in writing each higher standing employee in the same competitive level "
     "reached for release of the reasons for the exception and the date the lower "
     "standing employee's retention will end; and",
     "When a cutoff date is used, an employee will receive performance credit for the "
     "three most recent ratings of record received during the 4-year period prior to the "
     "cutoff date.",
     "N"),
    ("gen-70", "532.417",
     "(c) Creditable service. The following periods of time shall be considered "
     "creditable service for purposes of waiting periods for within-grade increases:",
     "The waiting period for within-grade increases varies depending on the position.",
     "N"),
    ("gen-105", "330.609",
     "(b) Reassign or demote an employee under part 432 or 752 of this chapter;",
     "In filling vacancies, an agency must give its RPL registrants placement priority "
     "for most competitive service vacancies before hiring someone from outside its own "
     "permanent competitive service workforce.",
     "N"),
    ("gen-117", "330.101",
     "Permanent competitive service workforce and permanent competitive service "
     "employees mean agency employees serving under career or career-conditional "
     "appointments, in tenure group I or II, respectively.",
     "An individual may attain career tenure only when employed (or reemployed) in a "
     "permanent appointment in the competitive service that provides or leads to "
     "competitive status.",
     "N"),
    ("gen-118", "315.201",
     "(iii) Nontemporary appointment to a nonappropriated fund (NAF) position in or "
     "under the Department of Defense or in or under the U.S. Coast Guard, Department of "
     "Homeland Security, provided the employee's NAF position was brought into the "
     "competitive service and, on that basis, the employee acquired competitive status "
     "or was converted to a career or career-conditional appointment;",
     "A person whose employment is converted to career or career-conditional employment "
     "under this section acquires a competitive status automatically on conversion.",
     "N"),
    ("gen-29", "351.601",
     "(a) Each agency must select competing employees for release from a competitive "
     "level (including release from a competitive level involving a pay band) under this "
     "part in the inverse order of retention standing, beginning with the employee with "
     "the lowest retention standing on the retention register. An agency may not release "
     "a competing employee from a competitive level while retaining in that level an "
     "employee with lower retention standing except:",
     "Employees are retained in a RIF in the competitive service in the inverse order of "
     "retention standing.",
     "C"),
    ("gen-96", "532.203",
     "(1) For grades NS-1 through NS-8, equal to the rate for step 2 of the "
     "corresponding grade of the nonsupervisory regular wage schedule for the area, plus "
     "20 percent of the rate for step 2 of NA-8;",
     "For grades NS-1 through NS-8, the rate for step 2 of the corresponding grade of "
     "the nonsupervisory regular wage schedule for the area is used.",
     "C"),
    ("gen-122", "550.409",
     "(c) An agency may terminate evacuation payments under the conditions listed in § "
     "550.407. An agency must make any necessary adjustments in pay consistent with § "
     "550.408 after the evacuation is terminated.",
     "An agency must terminate evacuation payments under the conditions listed in § "
     "550.407.",
     "C"),
    ("adv-1", "317.402",
     "(c) Each qualifications criterion in the standard must be job related. The "
     "standard may not emphasize agency-related experience, however, to the extent that "
     "it precludes otherwise well-qualified candidates from outside the agency from "
     "appointment consideration.",
     "The standard may emphasize agency-related experience even where doing so precludes "
     "well-qualified candidates from outside the agency.",
     "C"),
    ("adv-4", "317.703",
     "(c) Directing reinstatement. (1) To the extent practicable, OPM will direct "
     "reinstatement within 45 days of the date of receipt by OPM of the application for "
     "reinstatement or the date of separation from the Presidential appointment, "
     "whichever is later.",
     "OPM will direct reinstatement within 30 days of receiving the application.",
     "C"),
    ("adv-9", "330.213",
     "(a) Methods. An agency must adopt one of the selection methods in paragraphs (b), "
     "(c), or (d) of this section for a single RPL. The agency may adopt the same method "
     "for each RPL it establishes or may vary the method by location, but it must adopt "
     "a written policy for each RPL it establishes and maintains. While an agency may "
     "not vary the method used for an individual vacancy, it may at any time change the "
     "selection method for all positions covered by a single RPL.",
     "An agency may vary the selection method it uses for an individual vacancy.",
     "C"),
    ("adv-12", "337.204",
     "(3) Notification to the U.S. Office of Personnel Management (OPM). Once the head "
     "of a covered agency affirmatively determines the presence of a severe shortage and "
     "the direct hire authority is approved by the agency head, he or she must notify "
     "OPM within 10 business days. Such notification must include a description of the "
     "supporting evidence relied upon in making the determination.",
     "Notification to OPM need not describe the evidence the determination relied on.",
     "C"),
    ("adv-27", "550.905",
     "(b) Employees may not be paid a hazardous duty differential for hours for which "
     "they receive annual premium pay for regularly scheduled standby duty under "
     "550.141, annual premium pay for administratively uncontrollable overtime work "
     "under 550.151, or availability pay for criminal investigators under 550.181.",
     "An employee receiving availability pay under 550.181 may also be paid a hazardous "
     "duty differential for those hours.",
     "C"),
    ("adv-34", "630.1703",
     "(3) An employee with a seasonal work schedule may not use paid parental leave "
     "during the off-season period designated by the agency, the period during which the "
     "employee is scheduled to be released from work and placed in nonpay status.",
     "A seasonal employee may use paid parental leave during the off-season period "
     "designated by the agency.",
     "C"),
    ("adv-44", "530.304",
     "(c) In setting the level of special rates within a rate range for a category of "
     "employees, OPM will compute the special rate supplement by adding a fixed dollar "
     "amount or a fixed percentage to all GS rates within that range, except that an "
     "alternate method may be used",
     "OPM computes the special rate supplement by a fixed dollar amount or fixed "
     "percentage and no alternate method is available.",
     "C"),
    ("adv-46", "351.302",
     "(b) An employee whose position is transferred under this subpart solely for "
     "liquidation, and who is not identified with an operating function specifically "
     "authorized at the time of transfer to continue in operation more than 60 days, is "
     "not a competing employee for other positions in the competitive area gaining the "
     "function.",
     "An employee transferred solely for liquidation competes for other positions in the "
     "gaining competitive area.",
     "C"),
]

GOLD = {"E": "entail", "N": "neutral", "C": "contradict"}
#: The eight `adv-*` pairs are one minimal edit away from a verbatim quotation of 5 CFR --
#: a modality flipped, a number moved, a negation dropped. They are the module's reason to
#: exist, and the aligner finds a supporting span in every one of them.
FLIPS = [row for row in PAIRS if row[0].startswith("adv-")]


@pytest.fixture(scope="module")
def entailer():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # CPU: the numbers below are argmaxes, and fp32-CPU agreed with fp16-GPU on all 182
    # probe pairs (max logit difference 0.0055), so the assertions do not depend on a card.
    return Entailer(device="cpu", batch_size=8)


@pytest.mark.neural
@needs_weights
def test_the_head_order_is_read_off_the_checkpoint(entailer):
    """The two candidate checkpoints order their heads differently, so the order is read
    rather than assumed. Assuming is the difference between a working verifier and one that
    reports every contradiction as support with every number in range."""
    from warrant.verify.entail import _load

    _tok, model, order = _load(DEFAULT_MODEL, None, "cpu", True)
    assert sorted(order) == [0, 1, 2]
    names = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    assert names[order[0]].startswith("entail")
    assert names[order[2]].startswith("contradict")


@pytest.mark.neural
@needs_weights
def test_the_probe_subset_reproduces_the_measured_accuracy(entailer):
    verdicts = entailer.score([(p, c) for _n, _s, p, c, _g in PAIRS])
    correct = [v.label == GOLD[row[4]] for v, row in zip(verdicts, PAIRS, strict=True)]
    assert sum(correct) >= 18, [(row[0], row[4], v.label) for v, row, ok
                                in zip(verdicts, PAIRS, correct, strict=True) if not ok]


@pytest.mark.neural
@needs_weights
def test_the_residual_failure_of_both_signals_together_is_the_overgeneral_claim(entailer):
    """eval-007's open failure, asserted so it cannot quietly stop being reported. §351.404
    imposes the retention register *when a competing employee is to be released*; the claim
    states the duty unconditionally. Overlap finds a span, the model calls it entailment at
    0.95, and the two agreeing is exactly the case the combination cannot catch. Half of the
    generator's neutrals are of this shape, which is why nothing here is a gate.

    A checkpoint that fixes this should update eval-007 rather than delete this test."""
    _name, _sect, premise, claim, gold = next(r for r in PAIRS if r[0] == "gen-27")
    assert gold == "N"
    verdict = entailer.score([(premise, claim)])[0]
    assert align(claim, premise) is not None
    assert combine(align(claim, premise),
                   ClaimSupport(claim, {"v": verdict})) == SUPPORTED


@pytest.mark.neural
@needs_weights
def test_span_alignment_confirms_every_flipped_claim_it_should_reject(entailer):
    """The measurement that justifies the module. A claim that reverses its premise reuses
    its premise's vocabulary, so lexical overlap scores it as supported every time."""
    assert all(align(claim, premise) is not None
               for _n, _s, premise, claim, _g in FLIPS)


@pytest.mark.neural
@needs_weights
def test_entailment_flags_the_flipped_claims_span_alignment_confirms(entailer):
    verdicts = entailer.score([(p, c) for _n, _s, p, c, _g in FLIPS])
    flagged = [v.report == CONTRADICTED for v in verdicts]
    assert sum(flagged) >= 7, [(row[0], v.probs) for v, row, f
                               in zip(verdicts, FLIPS, flagged, strict=True) if not f]


@pytest.mark.neural
@needs_weights
def test_combining_the_two_signals_never_calls_a_reversed_claim_supported(entailer):
    for (_name, _sect, premise, claim, _gold), verdict in zip(
            FLIPS, entailer.score([(p, c) for _n, _s, p, c, _g in FLIPS]), strict=True):
        support = ClaimSupport(claim=claim, verdicts={"v": verdict})
        assert combine(align(claim, premise), support) != SUPPORTED


@pytest.mark.neural
@needs_weights
def test_direction_matters_premise_is_the_regulation(entailer):
    """NLI is directional and this module fixes the direction. A paragraph of regulation
    entails a short claim drawn from it; the claim does not entail the paragraph, which
    carries provisos the claim never mentions."""
    premise, claim = PAIRS[4][2], PAIRS[4][3]
    right, wrong = entailer.score([(premise, claim), (claim, premise)])
    assert right.entail > wrong.entail


@pytest.mark.neural
@needs_weights
def test_two_entailers_agree_to_the_last_bit(entailer):
    pairs = [(p, c) for _n, _s, p, c, _g in PAIRS[:8]]
    assert Entailer(device="cpu", batch_size=8).logits(pairs) == \
        Entailer(device="cpu", batch_size=8).logits(pairs)


@pytest.mark.neural
@needs_weights
def test_a_premise_too_long_for_the_encoder_is_windowed_not_truncated(entailer):
    """Truncation on a regulatory premise is not a rounding error: "Except as provided in
    paragraph (d)" and the paragraph it excepts can be hundreds of tokens apart, and
    dropping the tail turns a conditional rule into an absolute one."""
    filler = ("An agency shall record the determination in the employee's file. " * 60)
    premise = filler + "An employee may not carry over more than 240 hours."
    windows = entailer._windows(premise, "An employee may carry over 300 hours.")
    assert len(windows) > 1
    assert all(w is not None for w in windows)
    assert windows[0].start == 0 and windows[-1].end == len(premise)


@pytest.mark.neural
@needs_weights
def test_the_most_decisive_window_wins_not_the_most_entailing_one(entailer):
    """A chunk whose opening states a rule and whose tail revokes it must not report as
    clean support because the opening window scored well."""
    filler = ("An agency shall record the determination in the employee's file. " * 60)
    premise = ("An employee may carry over 300 hours of annual leave. " + filler
               + "An employee may not carry over more than 240 hours of annual leave.")
    support = entailer.score_claim("An employee may carry over 300 hours of annual leave.",
                                   {"chunk": premise})
    assert entailer.stats["windowed"] >= 1
    assert support.verdicts["chunk"].window is not None
    assert support.report != SUPPORTED


@pytest.mark.neural
@needs_weights
def test_score_answer_takes_the_generators_claims_without_importing_them(entailer):
    from warrant.generate.answer import Claim

    cited = {"v1": PAIRS[1][2], "v2": PAIRS[18][2]}
    claims = [Claim(text=PAIRS[1][3], evidence=["v1"]),
              Claim(text=PAIRS[18][3], evidence=["v2"]),
              Claim(text="a claim citing a chunk that is not in the context",
                    evidence=["missing"])]
    supports = entailer.score_answer(claims, cited)
    assert [s.report for s in supports] == [SUPPORTED, CONTRADICTED, UNCERTAIN]


@pytest.mark.neural
@needs_weights
def test_throughput_is_recorded_so_the_stage_can_be_costed(entailer):
    entailer.score([(p, c) for _n, _s, p, c, _g in PAIRS[:8]])
    assert entailer.stats["pairs"] >= 8
    assert entailer.stats["batches"] >= 1
    assert entailer.throughput > 0.0
