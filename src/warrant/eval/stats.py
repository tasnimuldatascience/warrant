"""Intervals and comparisons that survive the way this benchmark is actually built.

Three corrections to the obvious approach, each forced by a measured property of the data.

**Clustering.** The temporal bucket's 721 items come from 89 sections, and one section --
§531.603, on locality pay areas -- supplies 36% of them. An i.i.d. bootstrap over items
treats those as 721 independent trials and reports an interval roughly **four times too
narrow**: 73.2-79.3 against a section-clustered 67.8-94.0. Leaving that section out moves
the headline from 76.4% to 85.0%, which is the clearest possible evidence that items are not
the unit of independence. Sections are, so sections are what gets resampled.

**Boundaries.** A percentile bootstrap over an all-true or all-false vector returns a
zero-width interval: every resample gives the same answer. Publishing ``100.0-100.0`` for
130/130 or ``0.0-0.0`` for 0/721 asserts certainty no finite sample can support. A closed
form is both correct and simpler here -- Wilson for a single proportion, whose one-sided
bound at 0/721 is 0.51% rather than 0.

**Pairing.** Every configuration is scored on the same items, so comparisons are paired and
the pairing is worth roughly a factor of two in resolution. Reading two marginal intervals
for overlap throws that away and is the anti-pattern this module exists to avoid: a paired
cluster bootstrap of the *difference* is what licenses a claim that one configuration beats
another.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_SAMPLES = 2000


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.lo * 100:.1f}-{self.hi * 100:.1f}"


def wilson_ci(successes: int, n: int, *, z: float = 1.959963985) -> Interval:
    """Wilson score interval for a proportion. Correct at 0/n and n/n.

    Used instead of a bootstrap for plain proportions: at the boundaries the bootstrap
    degenerates to a point, and everywhere else the closed form is the same answer without
    2,000 resamples.
    """
    if n == 0:
        return Interval(0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(flags: Sequence[bool], keys: Sequence[str], *,
                         samples: int = DEFAULT_SAMPLES, seed: int = 0) -> Interval:
    """Percentile bootstrap resampling **clusters**, not items.

    A ratio estimator: each resample draws whole clusters with replacement and divides the
    pooled successes by the pooled item count, so a cluster contributing 260 items carries
    the weight it actually has.
    """
    if not flags:
        return Interval(0.0, 0.0)
    groups: dict[str, list[bool]] = {}
    for flag, key in zip(flags, keys, strict=True):
        groups.setdefault(key, []).append(flag)
    clusters = list(groups.values())
    if len(clusters) < 2:
        return wilson_ci(sum(flags), len(flags))

    rng = random.Random(seed)
    n = len(clusters)
    means: list[float] = []
    for _ in range(samples):
        hits = total = 0
        for _ in range(n):
            picked = clusters[rng.randrange(n)]
            hits += sum(picked)
            total += len(picked)
        means.append(hits / total if total else 0.0)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(math.ceil(0.975 * (samples - 1)))]
    return Interval(lo, hi)


@dataclass(frozen=True)
class PairedDelta:
    delta: float
    ci: Interval
    wins: int          # items only A got right
    losses: int        # items only B got right
    p_value: float

    @property
    def significant(self) -> bool:
        """Zero outside the interval *and* the sign test agrees. Both, deliberately: a
        bootstrap interval and an exact discordant-pair test disagreeing is a signal to
        report neither rather than to pick the friendlier one."""
        return (self.ci.lo > 0 or self.ci.hi < 0) and self.p_value < 0.05


def _mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_delta(a: Sequence[bool], b: Sequence[bool], keys: Sequence[str], *,
                 samples: int = DEFAULT_SAMPLES, seed: int = 0) -> PairedDelta:
    """A minus B on the same items, with a clustered interval and an exact sign test.

    Both configurations saw identical items, so only the items they disagree on carry any
    information. Comparing two marginal intervals for overlap ignores that and is far less
    sensitive -- it is how a real effect gets called noise and a null gets called a win.
    """
    groups: dict[str, list[tuple[bool, bool]]] = {}
    for flag_a, flag_b, key in zip(a, b, keys, strict=True):
        groups.setdefault(key, []).append((flag_a, flag_b))
    clusters = list(groups.values())
    n_items = len(a) or 1
    delta = (sum(a) - sum(b)) / n_items

    rng = random.Random(seed)
    n = len(clusters)
    deltas: list[float] = []
    for _ in range(samples):
        diff = total = 0
        for _ in range(n):
            picked = clusters[rng.randrange(n)]
            diff += sum(1 for x, y in picked if x) - sum(1 for x, y in picked if y)
            total += len(picked)
        deltas.append(diff / total if total else 0.0)
    deltas.sort()
    ci = Interval(deltas[int(0.025 * (samples - 1))],
                  deltas[int(math.ceil(0.975 * (samples - 1)))])

    wins = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return PairedDelta(delta=delta, ci=ci, wins=wins, losses=losses,
                       p_value=_mcnemar_p(wins, losses))
