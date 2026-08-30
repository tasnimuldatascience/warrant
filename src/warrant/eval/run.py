"""Scoring a retrieval configuration against a benchmark bucket.

Two numbers per bucket, reported separately and never combined:

**Sufficiency** — did the retrieved set contain a complete minimal sufficient evidence set?
Not "was the gold chunk retrieved": a question can be answerable from more than one set of
paragraphs, and grading against a single gold chunk understates a system that found a
different but equally valid route (ARCHITECTURE.md section 6).

**Distractor rate** — did the retrieved set contain the superseded version, or a part that
does not govern the asker? This is worth reporting even when sufficiency is satisfied,
because an answer citing both the current and the repealed rule is wrong in the way that
matters most here.

Intervals are a seeded percentile bootstrap over items. A bucket of a few hundred items
cannot separate configurations differing by a few points, and reporting a winner anyway is
how ablation tables come to publish noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..retrieve.hybrid import Retriever
from .bench import BenchItem


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    satisfied: bool
    distractors_hit: list[str]
    retrieved: list[str]

    @property
    def leaked(self) -> bool:
        return bool(self.distractors_hit)


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    n: int
    sufficiency: float
    sufficiency_ci: tuple[float, float]
    distractor_rate: float
    distractor_rate_ci: tuple[float, float]
    results: list[ItemResult] = field(default_factory=list)

    @property
    def measures_absence(self) -> bool:
        """Exclusion buckets have nothing to retrieve; only the distractor rate is meaningful."""
        return self.bucket.endswith("exclusion")

    def row(self, label: str = "") -> list[str]:
        suff = "n/a" if self.measures_absence else f"{self.sufficiency * 100:.1f}%"
        suff_ci = "" if self.measures_absence else (
            f"{self.sufficiency_ci[0] * 100:.1f}-{self.sufficiency_ci[1] * 100:.1f}")
        return [label or self.bucket, str(self.n), suff, suff_ci,
                f"{self.distractor_rate * 100:.1f}%",
                f"{self.distractor_rate_ci[0] * 100:.1f}-"
                f"{self.distractor_rate_ci[1] * 100:.1f}"]


def bootstrap_ci(flags: list[bool], *, samples: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over items. Seeded, so a published interval is reproducible."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(samples))
    return (means[int(0.025 * samples)], means[min(int(0.975 * samples), samples - 1)])


def score(retriever: Retriever, items: list[BenchItem], *, bucket: str = "",
          samples: int = 1000) -> BucketResult:
    results: list[ItemResult] = []
    for item in items:
        trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
        results.append(ItemResult(
            item_id=item.id,
            satisfied=item.is_satisfied_by(trace.final),
            distractors_hit=item.leaked(trace.final),
            retrieved=trace.final,
        ))
    suff = [r.satisfied for r in results]
    leak = [r.leaked for r in results]
    n = len(results) or 1
    return BucketResult(
        bucket=bucket or (items[0].bucket if items else "?"),
        n=len(results),
        sufficiency=sum(suff) / n,
        sufficiency_ci=bootstrap_ci(suff, samples=samples),
        distractor_rate=sum(leak) / n,
        distractor_rate_ci=bootstrap_ci(leak, samples=samples),
        results=results,
    )
