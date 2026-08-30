"""Abstention signals, the calibrated combiner, and the risk-coverage machinery.

Written against the failure modes eval-005 actually hit, not against the API surface.

Three of them are worth naming, because each one produces a plausible-looking number rather
than an exception, and a test suite that only checks shapes would pass through all three:

1. The PAVA fit merged blocks but kept the *lower* block's upper bound, so a test score
   landing inside a merged block was read off the next block up -- the one PAVA had just
   proved it did not belong to. Calibration silently got worse in exactly the region
   isotonic exists to repair.
2. ``risk_coverage`` walking one point per item rather than per distinct confidence would
   give the constant-confidence ``always answer`` policy a whole curve, and an AURC that
   reported item order as discrimination.
3. A threshold fitted on dev that is above every confidence on test answers nothing. That is
   coverage 0, a measurement; returning None there makes it indistinguishable from a policy
   nobody evaluated, and it is what the top-1 fusion baseline actually does on this corpus.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warrant.eval.bench import BenchItem
from warrant.index.store import Store
from warrant.retrieve.hybrid import Candidate, Trace
from warrant.verify.abstain import FEATURES, Signals, score_entropy, signals, signals_from_store
from warrant.verify.calibrate import (
    Combiner,
    Example,
    Isotonic,
    aurc,
    brier,
    collect_examples,
    design_matrix,
    ece,
    fit_policy,
    operating_point,
    reliability,
    risk_coverage,
    study,
)

# -- signals ------------------------------------------------------------------------


def trace(**kw) -> Trace:
    base = {"query": "restored annual leave scheduling deadline",
            "as_of": "2021-01-01", "scope": "government-wide"}
    return Trace(**base, **kw)


def test_feature_order_matches_the_vector():
    """A fitted combiner is a vector of coefficients and a feature order, and nothing else.

    If ``Signals.vector`` and ``FEATURES`` ever disagree, an already-fitted model reads the
    wrong columns and keeps returning probabilities. Nothing else in the system would notice.
    """
    s = Signals(*[float(i + 1) for i in range(len(FEATURES))])
    assert s.vector == tuple(s.as_dict()[name] for name in FEATURES)
    assert len(s.vector) == len(FEATURES)


def test_rank_agreement_counts_the_overlap_at_top_k_not_of_the_shorter_list():
    t = trace(lexical=["a@1", "b@1", "c@1", "d@1"], dense=["a@1", "z@1"],
              fused=[Candidate("a@1", 0.03)])
    # One id in common, measured against top_k -- not against the two-long dense list, which
    # would report 50% agreement for a dense ranker that found almost nothing.
    assert signals(t, top_k=4).rank_agreement == pytest.approx(0.25)


def test_rank_agreement_is_zero_without_a_dense_ranker():
    """Lexical-only is a supported configuration; it must not divide by an absent list."""
    t = trace(lexical=["a@1", "b@1"], fused=[Candidate("a@1", 0.03)])
    assert signals(t, top_k=4).rank_agreement == 0.0


def test_margins_come_off_the_fused_stage_not_the_reranked_one():
    """A margin between two RRF weights and a margin between two cross-encoder logits are
    different quantities. A combiner fitted across a mixture of the two would silently
    depend on whether the reranker was enabled for that request."""
    t = trace(fused=[Candidate("a@1", 0.030), Candidate("b@1", 0.020),
                     Candidate("c@1", 0.015), Candidate("d@1", 0.012),
                     Candidate("e@1", 0.010)],
              reranked=[Candidate("e@1", 9.0), Candidate("a@1", -4.0)],
              final=["e@1", "a@1"])
    s = signals(t, top_k=16)
    assert s.top_score == pytest.approx(0.030)
    assert s.margin_1_2 == pytest.approx(0.010)
    assert s.margin_1_5 == pytest.approx(0.020)


def test_margin_1_5_does_not_invent_a_rank_that_does_not_exist():
    """A head of three is itself thin evidence. Reporting a gap against a missing rank 5
    would manufacture a large margin out of a short list."""
    t = trace(fused=[Candidate("a@1", 0.03), Candidate("b@1", 0.02), Candidate("c@1", 0.01)])
    s = signals(t, top_k=16)
    assert s.margin_1_5 == pytest.approx(s.top_score - 0.01)


def test_term_coverage_reads_the_text_the_generator_will_see():
    t = trace(fused=[Candidate("a@1", 0.03)], final=["a@1"])
    covered = signals(t, texts={"a@1": "Restored annual leave must be scheduled and used."})
    missed = signals(t, texts={"a@1": "Firefighter overtime pay is computed hourly."})
    assert covered.term_coverage > missed.term_coverage
    assert 0.0 <= missed.term_coverage <= covered.term_coverage <= 1.0


def test_dead_query_yields_near_zero_signals_instead_of_raising():
    """A guard that raises on the queries it exists to catch is worse than no guard."""
    s = signals(trace())
    assert s.vector == (0.0,) * len(FEATURES)


def test_score_entropy_is_flat_at_one_and_peaked_below_it():
    assert score_entropy([0.25] * 4) == pytest.approx(1.0)
    assert score_entropy([0.97, 0.01, 0.01, 0.01]) < 0.3
    assert score_entropy([]) == 0.0
    assert score_entropy([0.5]) == 0.0


def test_guidance_top_fires_on_authority_of_the_top_hit_only():
    t = trace(fused=[Candidate("g@1", 0.03), Candidate("r@1", 0.02)], final=["g@1", "r@1"])
    assert signals(t, authority={"g@1": 4, "r@1": 2}).guidance_top == 1.0
    assert signals(t, authority={"g@1": 2, "r@1": 4}).guidance_top == 0.0


def test_signals_from_store_reads_authority_and_retrieval_text():
    with Store(":memory:") as s:
        s.db.execute(
            "INSERT INTO chunk (version_id, chunk_id, section_id, title, part, heading, "
            "text, authority, content_hash, valid_from, system_from, source_snapshot, "
            "config_hash) VALUES ('a@1','a','s',5,'630','Restored annual leave',"
            "'Restored leave must be scheduled.',4,'h','2017-01-01','2020-01-01','x','c')")
        s.db.commit()
        t = trace(fused=[Candidate("a@1", 0.03)], final=["a@1"], admitted=9000)
        out = signals_from_store(s, t)
    assert out.guidance_top == 1.0
    assert out.term_coverage > 0.0
    assert out.log_admitted == pytest.approx(math.log1p(9000))


# -- isotonic recalibration ---------------------------------------------------------


def test_isotonic_keeps_the_merged_blocks_upper_bound():
    """The bug this file exists for. y = [0, 1, 0, 1] over scores [1, 2, 3, 4] pools the
    middle pair into one block at 0.5 spanning scores 2 and 3. Discarding the popped block's
    bound left ``searchsorted`` reading score 3 off the *next* block, at 1.0 -- a confident
    prediction for the point PAVA had just pooled away from confidence."""
    iso = Isotonic.fit(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0, 1.0]))
    got = iso.apply(np.array([2.0, 3.0]))
    assert got[0] == pytest.approx(got[1])
    assert got[0] == pytest.approx(0.5)


def test_isotonic_is_monotone_non_decreasing():
    rng = np.random.default_rng(0)
    scores = np.sort(rng.random(200))
    y = (rng.random(200) < scores).astype(float)
    out = Isotonic.fit(scores, y).apply(np.linspace(0.0, 1.0, 501))
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_gives_equal_scores_equal_probabilities():
    """Two items the model cannot tell apart must not be assigned different probabilities
    on the strength of which one the sort happened to put first."""
    iso = Isotonic.fit(np.array([0.5, 0.5, 0.5, 0.9]), np.array([0.0, 1.0, 1.0, 1.0]))
    assert iso.apply(np.array([0.5]))[0] == pytest.approx(2 / 3)


def test_isotonic_never_returns_exact_certainty():
    """A pure block of four points is not proof of a probability of 1."""
    out = Isotonic.fit(np.array([0.1, 0.2, 0.3, 0.4]), np.ones(4)).apply(np.array([0.4]))
    assert 0.0 < out[0] < 1.0


# -- calibration error --------------------------------------------------------------


def test_ece_catches_a_confidently_wrong_bucket():
    """The failure this measure exists for: a 0.9 bucket that is right 60% of the time.
    Brier alone would call it mediocre; the point is that a reader was told to trust it."""
    probs = [0.95] * 100
    y = [True] * 60 + [False] * 40
    assert ece(probs, y) == pytest.approx(0.35, abs=0.01)
    assert brier(probs, y) > 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    probs = [0.3] * 100 + [0.8] * 100
    y = [True] * 30 + [False] * 70 + [True] * 80 + [False] * 20
    assert ece(probs, y) == pytest.approx(0.0, abs=1e-9)


def test_reliability_keeps_empty_bins_and_intervals_every_populated_one():
    table = reliability([0.05, 0.95, 0.96], [False, True, True], bins=10)
    assert len(table) == 10
    assert [b.n for b in table] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    populated = [b for b in table if b.n]
    assert all(b.ci.lo < b.ci.hi for b in populated)


# -- risk against coverage ----------------------------------------------------------


def test_risk_coverage_emits_one_point_per_distinct_confidence():
    """A constant-confidence policy is one operating point, not eight. Walking items would
    hand ``always answer`` a curve built entirely out of the order items arrived in."""
    points = risk_coverage([0.5] * 8, [False, True] * 4)
    assert len(points) == 1
    assert points[0].coverage == 1.0
    assert points[0].risk == pytest.approx(0.5)


def test_aurc_of_a_constant_policy_is_its_base_error_rate():
    assert aurc(risk_coverage([0.5] * 10, [True] * 2 + [False] * 8)) == pytest.approx(0.2)


def test_aurc_rewards_a_ranker_that_sorts_the_errors_to_the_bottom():
    error = [False] * 8 + [True] * 2
    perfect = list(range(10, 0, -1))            # confidence descending with correctness
    inverted = list(range(1, 11))
    assert aurc(risk_coverage(perfect, error)) < aurc(risk_coverage([0.5] * 10, error))
    assert aurc(risk_coverage(inverted, error)) > aurc(risk_coverage([0.5] * 10, error))


def test_operating_point_returns_none_rather_than_the_closest_miss():
    """"No threshold reaches 2%" is a finding. Returning a 3% point instead lets it be read
    as a 2% result, which is the specific way a selective-risk claim goes wrong."""
    points = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, False, False, False])
    assert operating_point(points, max_risk=0.02) is None
    assert operating_point(points, max_risk=0.30) is not None


def test_operating_point_takes_the_most_coverage_inside_the_budget():
    points = risk_coverage([0.9, 0.8, 0.7, 0.6, 0.5], [False] * 4 + [True])
    chosen = operating_point(points, max_risk=0.25)
    assert chosen is not None and chosen.coverage == 1.0


# -- the combiner -------------------------------------------------------------------


def example(*, features: tuple[float, ...], sufficient: bool, split: str,
            section: str, item: str) -> Example:
    return Example(item_id=item, section_id=section, split=split, bucket="temporal",
                   features=features, sufficient=sufficient)


def synthetic(n: int = 240, seed: int = 0) -> list[Example]:
    """A separable-ish population with one informative feature and seven constants.

    Deliberately shaped like the real one: the informative signal is weak, the positives
    dominate, and ``guidance_top`` is a constant zero -- which is what makes the ridge term
    load-bearing rather than decorative.
    """
    rng = np.random.default_rng(seed)
    out: list[Example] = []
    for i in range(n):
        good = rng.random() > 0.15
        signal = rng.normal(1.0 if good else -1.0, 1.0)
        feats = (signal, 0.001, 0.002, 0.99, rng.random(), 0.9, 9.1, 0.0)
        out.append(example(features=feats, sufficient=good,
                           split="dev" if i % 2 else "test",
                           section=f"s{i // 3}", item=f"i{i}"))
    return out


def test_fit_survives_a_feature_that_is_constant_on_the_fit_split():
    """``guidance_top`` is a constant on a single-source corpus. Dividing by its zero
    standard deviation gives NaN weights and a model that silently predicts nothing."""
    ex = synthetic()
    c = Combiner.fit(design_matrix(ex), np.array([1.0 if e.sufficient else 0.0 for e in ex]))
    assert np.all(np.isfinite(c.weights))
    assert c.scale[FEATURES.index("guidance_top")] == 1.0
    probs = c.probabilities(design_matrix(ex))
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_standardisation_travels_with_the_model():
    """The scaler is fitted on dev. A caller who standardised test against test's own
    moments would leak the reporting distribution into the prediction, and the resulting
    numbers would look slightly better and be untrue."""
    ex = synthetic()
    dev = [e for e in ex if e.split == "dev"]
    c = Combiner.fit(design_matrix(dev),
                     np.array([1.0 if e.sufficient else 0.0 for e in dev]))
    one = design_matrix([ex[0]])
    assert c.probabilities(one)[0] == pytest.approx(
        c.probabilities(np.vstack([one, design_matrix(ex[1:])]))[0])


def test_a_dev_threshold_above_every_test_confidence_is_coverage_zero_not_missing():
    """What the top-1 fusion baseline does on this corpus: no dev threshold reaches the 2%
    budget, so it ships a threshold of 1.0 and answers nothing. That is an operating point
    with coverage 0 and it has to be reported as one."""
    # Every dev item scores identically, so the curve has one point and it carries the base
    # error rate. No threshold reaches a 0% budget and ``fit_policy`` fails closed at 1.0.
    flat = (0.5, 0.001, 0.002, 0.99, 0.4, 0.9, 9.1, 0.0)
    dev = [example(features=flat, sufficient=i % 7 != 0, split="dev",
                   section=f"s{i}", item=f"d{i}") for i in range(21)]
    test = [e for e in synthetic() if e.split == "test"]
    policy = fit_policy(dev, target_risk=0.0)   # unreachable: forces threshold 1.0
    assert policy.threshold == 1.0
    s = study(dev, test, target_risk=0.0)
    assert s.learned.at_target is not None
    assert s.learned.at_target.coverage == 0.0
    assert len(s.learned.decisions) == len(test)


def test_study_is_deterministic():
    """Same input, same number twice. Two bootstraps and an IRLS loop sit under these."""
    ex = synthetic()
    dev = [e for e in ex if e.split == "dev"]
    test = [e for e in ex if e.split == "test"]
    a, b = study(dev, test, seed=0), study(dev, test, seed=0)
    assert (a.learned.aurc, a.ece_calibrated, a.policy.threshold) == \
           (b.learned.aurc, b.ece_calibrated, b.policy.threshold)
    assert a.learned.aurc_ci == b.learned.aurc_ci


def test_study_never_fits_on_test():
    """The dev-fitted policy must produce identical test confidences whatever else is in
    the test split. Any dependence would mean a moment, a threshold or an isotonic bound
    had been estimated from the reporting set."""
    ex = synthetic()
    dev = [e for e in ex if e.split == "dev"]
    test = [e for e in ex if e.split == "test"]
    full = study(dev, test, seed=0)
    half = study(dev, test[:40], seed=0)
    assert full.policy.threshold == half.policy.threshold
    assert full.policy.combiner.weights == half.policy.combiner.weights
    assert np.allclose(full.policy.confidences(design_matrix(test[:40])),
                       half.policy.confidences(design_matrix(test[:40])))


def test_beat_baseline_needs_both_the_point_estimate_and_the_paired_test():
    """A point estimate that wins while the paired test on the same items says nothing is
    the shape of a result that does not replicate. eval-005 reports a null on this."""
    ex = synthetic()
    s = study([e for e in ex if e.split == "dev"], [e for e in ex if e.split == "test"])
    assert s.beat_baseline == (s.learned.aurc < s.baseline_top_score.aurc
                               and s.against_baseline.significant)


# -- collection ---------------------------------------------------------------------


class _FakeRetriever:
    """Enough of ``Retriever`` to check what ``collect_examples`` puts in and leaves out."""

    def __init__(self, store: Store, returned: list[str]) -> None:
        self.store = store
        self.returned = returned
        self.calls = 0

    def retrieve(self, query, *, as_of, scope):     # noqa: ANN001, D102
        self.calls += 1
        return trace(fused=[Candidate(v, 0.03) for v in self.returned],
                     final=self.returned, admitted=9000)


def test_collect_examples_skips_the_tautological_bucket():
    """``scope-exclusion`` items carry an empty acceptable evidence set, so every possible
    ranked list satisfies them. Ninety-five items that cannot be got wrong would raise
    coverage and lower base risk without one of them being a measurement."""
    items = [
        BenchItem(id="a", bucket="temporal", query="q", as_of="2021-01-01", section_id="s",
                  part="630", heading="h", acceptable_evidence=[["a@1"]]),
        BenchItem(id="b", bucket="scope-exclusion", query="q", as_of="2021-01-01",
                  section_id="s", part="630", heading="h", acceptable_evidence=[[]]),
    ]
    with Store(":memory:") as s:
        r = _FakeRetriever(s, ["a@1"])
        got = collect_examples(r, items)
    assert [e.item_id for e in got] == ["a"]
    assert got[0].sufficient is True
    assert r.calls == 1                     # the skipped item costs no retrieval


def test_collect_examples_labels_insufficiency_from_the_final_cut():
    items = [BenchItem(id="a", bucket="human", query="q", as_of="2021-01-01",
                       section_id="s", part="630", heading="h",
                       acceptable_evidence=[["needed@1"]])]
    with Store(":memory:") as s:
        got = collect_examples(_FakeRetriever(s, ["other@1"]), items)
    assert got[0].sufficient is False
