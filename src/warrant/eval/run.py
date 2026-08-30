"""Scoring a retrieval configuration against a benchmark bucket.

Two numbers, reported separately and never combined into one:

**Sufficiency** — did the retrieved set contain a complete minimal sufficient evidence set?
Not "was the gold chunk retrieved": a question can be answerable from more than one set of
paragraphs, and asking about a single gold chunk understates a system that found a different
but equally valid route to the answer (ARCHITECTURE.md section 6).

**Distractor rate** — did the retrieved set contain the superseded or not-yet-in-force
version of the same paragraph? This is the failure the temporal bucket exists to detect, and
it is worth reporting even when sufficiency is satisfied, because an answer that cites both
the current and the repealed rule is wrong in the way that matters most here.

Confidence intervals are bootstrap over items. A bucket of a few hundred items cannot
distinguish configurations that differ by two or three points, and pretending otherwise is
how ablation tables come to report noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..index.store import Store
from .bench import TemporalItem


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
    results: list[ItemResult]

    def row(self) -> list[str]:
        return [
            self.bucket, str(self.n),
            f"{self.sufficiency * 100:.1f}%",
            f"{self.sufficiency_ci[0] * 100:.1f}-{self.sufficiency_ci[1] * 100:.1f}",
            f"{self.distractor_rate * 100:.1f}%",
            f"{self.distractor_rate_ci[0] * 100:.1f}-{self.distractor_rate_ci[1] * 100:.1f}",
        ]


def bootstrap_ci(flags: list[bool], *, samples: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over items. Seeded, so a published interval is reproducible."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(samples):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * samples)]
    hi = means[min(int(0.975 * samples), samples - 1)]
    return (lo, hi)


def retrieve(store: Store, item: TemporalItem, *, k: int,
             temporal: bool = True) -> list[str]:
    """Version ids retrieved for one item, best first."""
    rows = store.search(fts_query(item.query), valid_date=item.as_of,
                        limit=k, temporal=temporal)
    return [r["version_id"] for r in rows]


def fts_query(text: str) -> str:
    """Escape a natural-language query for FTS5.

    FTS5 treats bare punctuation as syntax, so a heading containing a colon or a hyphen is a
    parse error rather than a query. Quoting every token keeps the query literal, which is
    what a lexical baseline should be.
    """
    tokens = [t for t in ("".join(c if c.isalnum() else " " for c in text)).split() if t]
    return " OR ".join(f'"{t}"' for t in tokens) or '""'


def score(store: Store, items: list[TemporalItem], *, k: int, bucket: str = "temporal",
          temporal: bool = True, samples: int = 1000) -> BucketResult:
    results: list[ItemResult] = []
    for item in items:
        got = retrieve(store, item, k=k, temporal=temporal)
        results.append(ItemResult(
            item_id=item.id,
            satisfied=item.is_satisfied_by(got),
            distractors_hit=item.leaked(got),
            retrieved=got,
        ))
    suff = [r.satisfied for r in results]
    leak = [r.leaked for r in results]
    n = len(results) or 1
    return BucketResult(
        bucket=bucket,
        n=len(results),
        sufficiency=sum(suff) / n,
        sufficiency_ci=bootstrap_ci(suff, samples=samples),
        distractor_rate=sum(leak) / n,
        distractor_rate_ci=bootstrap_ci(leak, samples=samples),
        results=results,
    )
