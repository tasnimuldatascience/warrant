"""Turning the abstention signals into a probability, and reporting whether to believe it.

Three things happen here, and only the first is modelling:

**A combiner.** Logistic regression over `abstain.FEATURES`, fitted by IRLS with a ridge
penalty. Interpretable beats clever: eight coefficients can be read, argued with, and shown to
a reviewer, and on a dev split with a double-digit number of negatives anything with more
capacity would be fitting the noise. It is implemented here in numpy rather than pulled from
sklearn because numpy is already a dependency and sklearn is not, and a clone must install
from the same short list of wheels it installs today.

**A recalibration.** The combiner's own output is a maximum-likelihood fit to the dev split,
which makes it calibrated *there* almost by construction and says nothing about test. Isotonic
regression (pool-adjacent-violators, fitted on dev, applied to test) is the free non-parametric
correction. It is reported by **expected calibration error** against a reliability table,
because a model whose 0.9 bucket is right 60% of the time is worse than one with no confidence
score at all -- it converts a wrong answer into a wrong answer a reader has been told to trust.

**A risk-coverage curve.** The actual deliverable. For every threshold, what fraction of
questions still get answered (coverage) and what share of those answers had no sufficient
evidence behind them (selective risk). Abstention is a trade-off between the two and there is
no single right point on it; publishing one threshold and calling it "the abstention feature"
hides the curve that the operator is entitled to choose from.

Two baselines are computed on identical items and must be beaten honestly:

  ``always answer``   coverage 1.0 at the base error rate. This is today's system.
  ``top-1 fusion``    threshold the raw top RRF weight, one feature, no fitting.

The second is the one that matters. A learned combiner that does not beat a single free
feature has bought nothing but a fitting step and a model file to keep in sync, and this
module is written so that outcome is as easy to report as the other one.

**It is the outcome.** eval-005: AURC 0.0105 for the combiner against 0.0086 for the raw
top-1 RRF weight, a paired section-clustered difference of +0.0019 (-0.0017 to +0.0070), and
``beat_baseline`` False. The combiner's one measured advantage is a calibrated probability --
ECE 0.020 against an ordering that has none -- and it pays 0.002 AURC for it, because
isotonic pools items into blocks and pooling destroys the ordering inside a block. Read the
results doc before wiring the combiner in preference to a threshold on ``top_score``.

Nothing here calls a model, samples, or shuffles. Every number is a deterministic function of
the inputs, and the two bootstraps take an explicit seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from ..eval.bench import BenchItem
from ..eval.stats import Interval, cluster_bootstrap_ci, paired_delta, wilson_ci
from ..retrieve.hybrid import Retriever
from .abstain import FEATURES, TOP_K, Signals, signals_from_store

#: Ridge strength on the standardised features, intercept excluded. Not tuned, and it is not
#: optional: ``guidance_top`` is a constant on a single-source corpus, its standardised column
#: is exactly zero, and at ``l2 = 0`` the Hessian is singular and the fit raises. The penalty
#: is also what keeps a dev split with ten negatives from separating and sending every
#: prediction to 0 or 1. Sweeping it from 0.01 to 10 moves test AURC by 0.003 and never far
#: enough to overturn the verdict in eval-005, which is what "not tuned" has to mean here.
DEFAULT_L2 = 1.0
#: The selective-risk budget the operating point is chosen against.
TARGET_RISK = 0.02


@dataclass(frozen=True)
class Example:
    """One benchmark item, its signals, and whether the retrieved set could support an answer.

    ``sufficient`` is the label and it needs no human and no generator: the benchmark already
    carries a disjunction of minimal sufficient evidence sets, and whether one of them
    survived into the context is a set question. That is the same quantity
    ``eval.generation`` calls ``retrieved_evidence``, which is what made the six unsupported
    answers in eval-004 visible in the first place.
    """

    item_id: str
    section_id: str
    split: str
    bucket: str
    features: tuple[float, ...]
    sufficient: bool


#: Buckets whose sufficiency label carries no information and must not enter the study.
#: ``scope-exclusion`` items are written with an empty acceptable evidence set -- the whole
#: question is whether something is *absent* -- so ``is_satisfied_by`` returns True for every
#: possible ranked list. Ninety-five items that cannot be got wrong would raise coverage and
#: lower base risk without a single one of them being a measurement.
TAUTOLOGICAL_BUCKETS = frozenset({"scope-exclusion"})


def collect_examples(retriever: Retriever, items: Sequence[BenchItem], *,
                     top_k: int = TOP_K,
                     exclude: frozenset[str] = TAUTOLOGICAL_BUCKETS) -> list[Example]:
    """Run retrieval over benchmark items and label each by whether its context sufficed.

    The label needs no human and no generator, which is what makes this study reproducible
    from a clone: ``BenchItem`` already carries a disjunction of minimal sufficient evidence
    sets, and whether one of them survived into the final cut is a set question. It is the
    same quantity ``eval.run.score`` calls sufficiency and ``eval.generation`` calls
    ``retrieved_evidence`` -- the one that made the six unsupported answers in eval-004
    visible.

    One retrieval per item and no model call beyond the ones retrieval already makes.
    """
    out: list[Example] = []
    for item in items:
        if item.bucket in exclude:
            continue
        trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
        sig = signals_from_store(retriever.store, trace, top_k=top_k)
        out.append(Example(item_id=item.id, section_id=item.section_id, split=item.split,
                           bucket=item.bucket, features=sig.vector,
                           sufficient=item.is_satisfied_by(trace.final)))
    return out


def design_matrix(examples: Sequence[Example]) -> np.ndarray:
    return np.asarray([e.features for e in examples], dtype=np.float64).reshape(
        len(examples), len(FEATURES))


def labels(examples: Sequence[Example]) -> np.ndarray:
    return np.asarray([1.0 if e.sufficient else 0.0 for e in examples], dtype=np.float64)


# -- the combiner ------------------------------------------------------------------


@dataclass(frozen=True)
class Combiner:
    """A fitted logistic model: standardisation and weights, and nothing else.

    Standardisation is carried inside the model rather than applied by the caller. The scaler
    is fitted on dev, so a caller who standardised test against test's own moments would be
    quietly leaking the test distribution into the prediction -- and the resulting numbers
    would look slightly better and be untrue.
    """

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]          # len(FEATURES) + 1; last entry is the intercept
    names: tuple[str, ...] = FEATURES

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - np.asarray(self.mean)) / np.asarray(self.scale)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Log-odds for a design matrix."""
        w = np.asarray(self.weights)
        return self._z(np.atleast_2d(x)) @ w[:-1] + w[-1]

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        return expit(self.scores(x))

    def probability(self, signals: Signals) -> float:
        return float(self.probabilities(np.asarray([signals.vector]))[0])

    def coefficients(self) -> list[tuple[str, float]]:
        """Per-feature weights in standard-deviation units, largest magnitude first."""
        pairs = list(zip(self.names, self.weights[:-1], strict=True))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, l2: float = DEFAULT_L2,
            iterations: int = 100, tol: float = 1e-10) -> Combiner:
        """Ridge-penalised logistic regression by iteratively reweighted least squares.

        IRLS rather than gradient descent because it needs no learning rate, converges in
        under a dozen steps at this size, and -- the reason that matters here -- has no
        step-count knob that quietly doubles as regularisation. A run that stops early is a
        differently-regularised model wearing the same name.

        A feature that is constant on the fit split gets ``scale = 1``, so its standardised
        column is a constant that the ridge term shrinks toward zero. The alternative,
        dividing by zero standard deviation, produces NaN weights and a model that silently
        predicts nothing.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = np.hstack([(x - mean) / scale, np.ones((x.shape[0], 1))])

        penalty = np.eye(z.shape[1]) * l2
        penalty[-1, -1] = 0.0           # never shrink the intercept toward zero prevalence
        w = np.zeros(z.shape[1])
        for _ in range(iterations):
            p = expit(z @ w)
            # Floor the IRLS weights: a saturated probability makes p(1-p) underflow and the
            # Hessian singular, which is exactly what a near-separable split produces.
            s = np.clip(p * (1.0 - p), 1e-8, None)
            gradient = z.T @ (y - p) - penalty @ w
            hessian = z.T @ (z * s[:, None]) + penalty
            step = np.linalg.solve(hessian, gradient)
            w = w + step
            if np.max(np.abs(step)) < tol:
                break
        return cls(mean=tuple(mean), scale=tuple(scale), weights=tuple(w))


# -- recalibration -----------------------------------------------------------------


@dataclass(frozen=True)
class Isotonic:
    """Pool-adjacent-violators isotonic regression, fitted on dev and applied to test.

    Outputs are clipped away from 0 and 1. Isotonic blocks that happen to be pure produce
    exact certainty from a handful of points, and an exact 0 or 1 is a claim no finite sample
    licenses -- the same reason ``stats.wilson_ci`` exists instead of a bootstrap at the
    boundaries.
    """

    bounds: tuple[float, ...]           # inclusive upper score of each block
    values: tuple[float, ...]           # fitted probability of each block
    clip: float = 0.0

    def apply(self, scores: np.ndarray) -> np.ndarray:
        if not self.bounds:
            return np.asarray(scores, dtype=np.float64)
        idx = np.searchsorted(np.asarray(self.bounds), np.asarray(scores), side="left")
        idx = np.clip(idx, 0, len(self.values) - 1)
        out = np.asarray(self.values)[idx]
        return np.clip(out, self.clip, 1.0 - self.clip)

    @classmethod
    def fit(cls, scores: np.ndarray, y: np.ndarray) -> Isotonic:
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if scores.size == 0:
            return cls((), ())
        order = np.argsort(scores, kind="stable")
        scores, y = scores[order], y[order]

        # Equal scores must enter as one block: two items the model cannot tell apart cannot
        # be assigned different probabilities, and splitting them lets the fit order decide.
        blocks: list[list[float]] = []          # [total, count, upper score]
        for s, label in zip(scores, y, strict=True):
            if blocks and blocks[-1][2] == s:
                blocks[-1][0] += label
                blocks[-1][1] += 1
            else:
                blocks.append([label, 1.0, s])
            while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]
                                        > blocks[-1][0] / blocks[-1][1]):
                total, count, upper = blocks.pop()
                blocks[-1][0] += total
                blocks[-1][1] += count
                # The merged block now runs up to the *higher* of the two bounds. Dropping
                # the popped block's bound leaves a hole: every test score between the two
                # falls past the merged block in ``searchsorted`` and is read off the next
                # block up, which is the one block PAVA just proved it does not belong to.
                blocks[-1][2] = upper
        clip = 1.0 / (2.0 * len(y))
        return cls(bounds=tuple(b[2] for b in blocks),
                   values=tuple(b[0] / b[1] for b in blocks), clip=clip)


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    confidence: float                   # mean predicted probability in the bin
    empirical: float                    # share actually sufficient
    ci: Interval

    @property
    def gap(self) -> float:
        return self.empirical - self.confidence

    def row(self) -> list[str]:
        return [f"{self.lo:.1f}–{self.hi:.1f}", str(self.n),
                f"{self.confidence:.3f}" if self.n else "—",
                f"{self.empirical:.3f}" if self.n else "—",
                f"{self.gap:+.3f}" if self.n else "—",
                str(self.ci) if self.n else ""]


def reliability(probs: Sequence[float], y: Sequence[bool], *, bins: int = 10) -> list[Bin]:
    """Equal-width reliability table. Empty bins are kept: a gap in the table is information.

    Every bin's empirical rate gets a Wilson interval, because the whole failure mode this
    table exists to catch -- a 0.9 bucket that is right 60% of the time -- is usually
    estimated from a dozen items, and a bare point estimate there is indistinguishable from
    noise.
    """
    p = np.asarray(probs, dtype=np.float64)
    labels_ = np.asarray([bool(v) for v in y])
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[Bin] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        hits = int(labels_[mask].sum())
        out.append(Bin(lo=float(lo), hi=float(hi), n=n,
                       confidence=float(p[mask].mean()) if n else 0.0,
                       empirical=hits / n if n else 0.0,
                       ci=wilson_ci(hits, n) if n else Interval(0.0, 1.0)))
    return out


def ece(probs: Sequence[float], y: Sequence[bool], *, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted mean gap between confidence and truth."""
    table = reliability(probs, y, bins=bins)
    n = sum(b.n for b in table)
    if not n:
        return 0.0
    return sum(b.n * abs(b.gap) for b in table) / n


def brier(probs: Sequence[float], y: Sequence[bool]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray([1.0 if v else 0.0 for v in y], dtype=np.float64)
    return float(np.mean((p - t) ** 2)) if p.size else 0.0


# -- risk against coverage ---------------------------------------------------------


@dataclass(frozen=True)
class Point:
    threshold: float
    answered: int
    errors: int
    n: int

    @property
    def coverage(self) -> float:
        return self.answered / self.n if self.n else 0.0

    @property
    def risk(self) -> float:
        return self.errors / self.answered if self.answered else 0.0


def risk_coverage(confidence: Sequence[float], error: Sequence[bool]) -> list[Point]:
    """Operating points of ``answer iff confidence >= threshold``, ascending coverage.

    One point per **distinct** confidence value, not one per item. Items the policy scores
    identically are indistinguishable to it, so a curve that walked through them one at a time
    would be reporting the order they arrived in as if it were discrimination -- and the
    ``always answer`` baseline, whose confidence is constant, would get a whole spurious curve
    out of nothing but item order.
    """
    c = np.asarray(confidence, dtype=np.float64)
    e = np.asarray([1 if v else 0 for v in error], dtype=np.int64)
    n = c.size
    if n == 0:
        return []
    order = np.argsort(-c, kind="stable")
    c, e = c[order], e[order]
    points: list[Point] = []
    answered = errors = 0
    for i in range(n):
        answered += 1
        errors += int(e[i])
        if i + 1 < n and c[i + 1] == c[i]:
            continue                    # still inside a tie group; not a reachable threshold
        points.append(Point(threshold=float(c[i]), answered=answered, errors=errors, n=n))
    return points


def aurc(points: Sequence[Point]) -> float:
    """Area under the risk-coverage curve, integrated over coverage on [0, 1].

    Trapezoidal between reachable operating points, which is linear interpolation across a
    tie group -- the expectation over orderings within the group rather than a guess at one.
    Below the smallest reachable coverage the risk is held flat at that point's value; the
    alternative, extrapolating toward zero coverage, would invent a low-risk region the policy
    cannot actually reach.

    Lower is better, and the floor is not zero: a curve over a set with a positive error rate
    can only reach zero risk on the part of the coverage range it declines to answer.
    """
    if not points:
        return 0.0
    xs = [0.0] + [p.coverage for p in points]
    ys = [points[0].risk] + [p.risk for p in points]
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return area


def operating_point(points: Sequence[Point], *, max_risk: float = TARGET_RISK) -> Point | None:
    """The most coverage available at or under a selective-risk budget, or None.

    Ties on coverage are broken toward the lower risk, and the whole thing returns None rather
    than the closest miss: "no threshold reaches 2%" is a finding, and a function that
    answered with a 3% point instead would let it be read as a 2% result.
    """
    admissible = [p for p in points if p.answered and p.risk <= max_risk]
    if not admissible:
        return None
    return max(admissible, key=lambda p: (p.coverage, -p.risk))


# -- intervals ---------------------------------------------------------------------


def cluster_bootstrap_statistic(values: Sequence[float], flags: Sequence[bool],
                                keys: Sequence[str],
                                statistic: Callable[[np.ndarray, np.ndarray], float],
                                *, samples: int = 2000, seed: int = 0) -> Interval:
    """Percentile interval for a statistic that is not a pooled proportion.

    ``stats.cluster_bootstrap_ci`` is a ratio estimator over boolean flags and is the right
    tool for every rate reported here. AURC is not a ratio of pooled counts -- it is a
    functional of the whole ranking -- so it needs the same resampling scheme with a different
    estimator, which is what this is. Clusters are sections, for the reason stated in
    ``stats``: two paragraphs of one section are not independent trials.
    """
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    clusters = list(groups.values())
    if len(clusters) < 2:
        return Interval(0.0, 1.0)

    v = np.asarray(values, dtype=np.float64)
    f = np.asarray([bool(x) for x in flags])
    rng = random.Random(seed)
    n = len(clusters)
    draws: list[float] = []
    for _ in range(samples):
        picked: list[int] = []
        for _ in range(n):
            picked.extend(clusters[rng.randrange(n)])
        idx = np.asarray(picked, dtype=np.int64)
        draws.append(statistic(v[idx], f[idx]))
    draws.sort()
    return Interval(draws[int(0.025 * (samples - 1))],
                    draws[int(math.ceil(0.975 * (samples - 1)))])


def _aurc_of(confidence: np.ndarray, error: np.ndarray) -> float:
    return aurc(risk_coverage(confidence, error))


# -- the policy, which is the thing that gets wired in -----------------------------


@dataclass(frozen=True)
class Decision:
    answer: bool
    confidence: float
    signals: Signals


@dataclass(frozen=True)
class Policy:
    """A fitted combiner, its recalibration, and the threshold chosen from the curve.

    This is the whole serving surface: build one at start-up, call ``decide`` per request,
    and abstain when it says to. It holds a threshold because a caller has to be given one --
    but the threshold is a policy choice read off the risk-coverage curve, not a property of
    the model, and ``target_risk`` records which budget it was chosen against.
    """

    combiner: Combiner
    isotonic: Isotonic | None = None
    threshold: float = 1.0
    target_risk: float = TARGET_RISK

    def confidence(self, signals: Signals) -> float:
        """P(the retrieved set is sufficient), recalibrated."""
        return float(self.confidences(np.asarray([signals.vector]))[0])

    def confidences(self, x: np.ndarray) -> np.ndarray:
        p = self.combiner.probabilities(x)
        return self.isotonic.apply(p) if self.isotonic is not None else p

    def decide(self, signals: Signals) -> Decision:
        c = self.confidence(signals)
        return Decision(answer=c >= self.threshold, confidence=c, signals=signals)


def fit_policy(dev: Sequence[Example], *, l2: float = DEFAULT_L2,
               calibrate: bool = True, target_risk: float = TARGET_RISK) -> Policy:
    """Fit combiner, isotonic map and threshold on the dev split. Test is never touched.

    The threshold is chosen on dev too, and that is worth saying out loud: a threshold picked
    on test is a test-set hyperparameter and the coverage it reports is optimistic. The cost
    is that the risk it actually delivers on test can miss the budget, which is a real result
    and is reported as one.
    """
    x, y = design_matrix(dev), labels(dev)
    combiner = Combiner.fit(x, y, l2=l2)
    raw = combiner.probabilities(x)
    iso = Isotonic.fit(raw, y) if calibrate else None
    conf = iso.apply(raw) if iso is not None else raw
    chosen = operating_point(risk_coverage(conf, [not e.sufficient for e in dev]),
                             max_risk=target_risk)
    return Policy(combiner=combiner, isotonic=iso,
                  threshold=chosen.threshold if chosen else 1.0,
                  target_risk=target_risk)


# -- the study ---------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """One policy's measured behaviour on the test split."""

    name: str
    points: list[Point]
    aurc: float
    aurc_ci: Interval
    #: The single point this policy is *evaluated* at, which is not always a point that meets
    #: the budget. When a threshold was transferred from dev it is wherever that threshold
    #: lands on test, budget or no budget -- reporting only budget-meeting points would hide
    #: every policy whose dev threshold failed to transfer, which is the failure mode a
    #: threshold chosen on dev actually has. None only when no threshold was supplied and no
    #: point on the curve reaches the budget.
    at_target: Point | None
    risk_ci: Interval | None
    coverage_ci: Interval | None
    #: Per item: did the policy make the right call at its operating point -- answered with
    #: sufficient evidence, or abstained without it. The paired unit of comparison.
    decisions: list[bool] = field(default_factory=list)


def _curve(name: str, confidence: np.ndarray, examples: Sequence[Example], *,
           target_risk: float, threshold: float | None, seed: int) -> Curve:
    error = [not e.sufficient for e in examples]
    keys = [e.section_id or e.item_id for e in examples]
    points = risk_coverage(confidence, error)
    area = aurc(points)
    area_ci = cluster_bootstrap_statistic(confidence, error, keys, _aurc_of, seed=seed)

    chosen = (operating_point(points, max_risk=target_risk) if threshold is None
              else _point_at(points, threshold, n=len(examples)))
    risk_ci = coverage_ci = None
    decisions: list[bool] = []
    if chosen is not None:
        # Against ``threshold``, not ``chosen.threshold``: they select the same items -- no
        # confidence lies strictly between them -- but only when a threshold was supplied.
        # For a test-chosen point the two are identical anyway.
        cut = chosen.threshold if threshold is None else threshold
        answered = [c >= cut for c in confidence]
        decisions = [a == (not e) for a, e in zip(answered, error, strict=True)]
        errs = [e for a, e in zip(answered, error, strict=True) if a]
        answered_keys = [k for a, k in zip(answered, keys, strict=True) if a]
        risk_ci = (cluster_bootstrap_ci(errs, answered_keys, seed=seed) if any(errs)
                   else wilson_ci(0, len(errs)))
        coverage_ci = wilson_ci(chosen.answered, chosen.n)
    return Curve(name=name, points=points, aurc=area, aurc_ci=area_ci, at_target=chosen,
                 risk_ci=risk_ci, coverage_ci=coverage_ci, decisions=decisions)


def _point_at(points: Sequence[Point], threshold: float, *, n: int) -> Point:
    """The operating point a fixed threshold actually lands on.

    A threshold above every confidence in the set answers nothing, and that is a real
    operating point with coverage 0 rather than a missing one. Returning None there would
    make a policy transferred from dev and found to be too strict for test look
    indistinguishable from a policy that was never evaluated.
    """
    reachable = [p for p in points if p.threshold >= threshold]
    if not reachable:
        return Point(threshold=threshold, answered=0, errors=0, n=n)
    return max(reachable, key=lambda p: p.coverage)


@dataclass(frozen=True)
class Study:
    """Everything eval-005 reports, computed once so the doc cannot drift from the code."""

    n_dev: int
    n_test: int
    dev_insufficient: int
    test_insufficient: int
    base_risk_ci: Interval
    policy: Policy
    reliability_raw: list[Bin]
    reliability_calibrated: list[Bin]
    ece_raw: float
    ece_calibrated: float
    brier_raw: float
    brier_calibrated: float
    learned: Curve
    learned_uncalibrated: Curve
    baseline_top_score: Curve
    always_answer: Curve
    against_baseline: object            # stats.PairedDelta

    @property
    def beat_baseline(self) -> bool:
        """AURC lower than the single-feature baseline *and* the paired test agrees.

        Both, for the reason ``stats.PairedDelta.significant`` gives: a point estimate that
        wins while the paired test on the same items says nothing is the shape of a result
        that does not replicate.
        """
        return (self.learned.aurc < self.baseline_top_score.aurc
                and getattr(self.against_baseline, "significant", False))


def dev_threshold(confidence: np.ndarray, dev: Sequence[Example], *,
                  target_risk: float = TARGET_RISK) -> float:
    """The tightest-risk threshold a confidence signal reaches on dev, or 1.0.

    Every policy compared in ``study`` gets its threshold from this function on the dev
    split. An earlier version of this module chose the learned combiner's threshold on dev
    and the single-feature baseline's on test, which handed the baseline a hyperparameter
    fitted on the reporting set. The direction of that bias favoured the baseline, so it
    would have made a null result look safe -- but a comparison that is only unfair in the
    conservative direction is still not a comparison.
    """
    chosen = operating_point(risk_coverage(confidence, [not e.sufficient for e in dev]),
                             max_risk=target_risk)
    return chosen.threshold if chosen else 1.0


def study(dev: Sequence[Example], test: Sequence[Example], *, l2: float = DEFAULT_L2,
          target_risk: float = TARGET_RISK, seed: int = 0) -> Study:
    """Fit on dev, report on test, against both baselines."""
    policy = fit_policy(dev, l2=l2, target_risk=target_risk)
    x_dev, x_test = design_matrix(dev), design_matrix(test)
    y_test = [e.sufficient for e in test]

    raw = policy.combiner.probabilities(x_test)
    calibrated = policy.confidences(x_test)

    top = FEATURES.index("top_score")
    top_score = x_test[:, top]
    constant = np.zeros(len(test))

    learned = _curve("learned combiner", calibrated, test, target_risk=target_risk,
                     threshold=policy.threshold, seed=seed)
    baseline = _curve(
        "top-1 fusion score", top_score, test, target_risk=target_risk,
        threshold=dev_threshold(x_dev[:, top], dev, target_risk=target_risk), seed=seed)
    return Study(
        n_dev=len(dev), n_test=len(test),
        dev_insufficient=sum(1 for e in dev if not e.sufficient),
        test_insufficient=sum(1 for e in test if not e.sufficient),
        base_risk_ci=cluster_bootstrap_ci([not e.sufficient for e in test],
                                          [e.section_id or e.item_id for e in test], seed=seed),
        policy=policy,
        reliability_raw=reliability(raw, y_test),
        reliability_calibrated=reliability(calibrated, y_test),
        ece_raw=ece(raw, y_test), ece_calibrated=ece(calibrated, y_test),
        brier_raw=brier(raw, y_test), brier_calibrated=brier(calibrated, y_test),
        learned=learned,
        learned_uncalibrated=_curve(
            "learned, uncalibrated", raw, test, target_risk=target_risk,
            threshold=dev_threshold(policy.combiner.probabilities(x_dev), dev,
                                    target_risk=target_risk), seed=seed),
        baseline_top_score=baseline,
        # No threshold to transfer: this policy has one reachable operating point and it is
        # coverage 1.0. Passing a dev threshold would be theatre.
        always_answer=_curve("always answer", constant, test, target_risk=target_risk,
                             threshold=-np.inf, seed=seed),
        against_baseline=paired_delta(
            learned.decisions, baseline.decisions,
            [e.section_id or e.item_id for e in test], seed=seed),
    )
