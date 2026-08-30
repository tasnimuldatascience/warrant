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

from dataclasses import dataclass, field

from ..retrieve.hybrid import Retriever
from .bench import BenchItem
from .stats import Interval, cluster_bootstrap_ci, wilson_ci


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    satisfied: bool
    distractors_hit: list[str]
    retrieved: list[str]
    #: The clustering unit. Items from one section are not independent trials.
    section_id: str = ""
    #: Whether any distractor was even admitted by the predicates. When none was, a 0%
    #: distractor rate restates the WHERE clause rather than measuring anything.
    distractor_reachable: bool = False

    @property
    def leaked(self) -> bool:
        return bool(self.distractors_hit)


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    n: int
    sufficiency: float
    sufficiency_ci: Interval
    distractor_rate: float
    distractor_rate_ci: Interval
    #: Items whose distractor the predicates admitted at all. Zero means the distractor
    #: could not have been retrieved, so the distractor rate is enforced by construction.
    distractors_reachable: int = 0
    clusters: int = 0
    results: list[ItemResult] = field(default_factory=list)

    @property
    def measures_absence(self) -> bool:
        """Exclusion buckets have nothing to retrieve; only the distractor rate is meaningful."""
        return self.bucket.endswith("exclusion")

    @property
    def distractor_enforced(self) -> bool:
        """True when no distractor was ever admitted, so 0% is a tautology not a result."""
        return self.distractors_reachable == 0

    def row(self, label: str = "") -> list[str]:
        suff = "n/a" if self.measures_absence else f"{self.sufficiency * 100:.1f}%"
        suff_ci = "" if self.measures_absence else str(self.sufficiency_ci)
        rate = f"{self.distractor_rate * 100:.1f}%"
        if self.distractor_enforced:
            rate += "*"
        return [label or self.bucket, str(self.n), str(self.clusters), suff, suff_ci,
                rate, "by construction" if self.distractor_enforced
                else str(self.distractor_rate_ci)]


def score(retriever: Retriever, items: list[BenchItem], *, bucket: str = "",
          samples: int = 2000) -> BucketResult:
    store = retriever.store
    results: list[ItemResult] = []
    for item in items:
        trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
        # Was the distractor even a candidate? A distractor the predicates never admitted
        # cannot be retrieved, so counting it as a clean result measures the SQL, not the
        # system. Recorded per item so the bucket can say which of the two it is.
        reachable = False
        if item.distractors:
            admitted = store.candidate_ids(
                valid_date=item.as_of, temporal=retriever.temporal,
                exclude_parts=item.scope.excluded_parts(retriever.parts_universe)
                if retriever.parts_universe else [])
            rows = store.db.execute(
                "SELECT id, version_id FROM chunk WHERE version_id IN "
                f"({','.join('?' * len(item.distractors))})", item.distractors).fetchall()
            reachable = any(r["id"] in admitted for r in rows)
        results.append(ItemResult(
            item_id=item.id,
            satisfied=item.is_satisfied_by(trace.final),
            distractors_hit=item.leaked(trace.final),
            retrieved=trace.final,
            section_id=item.section_id,
            distractor_reachable=reachable,
        ))
    suff = [r.satisfied for r in results]
    leak = [r.leaked for r in results]
    keys = [r.section_id or r.item_id for r in results]
    n = len(results) or 1
    return BucketResult(
        bucket=bucket or (items[0].bucket if items else "?"),
        n=len(results),
        sufficiency=sum(suff) / n,
        sufficiency_ci=cluster_bootstrap_ci(suff, keys, samples=samples),
        distractor_rate=sum(leak) / n,
        distractor_rate_ci=(wilson_ci(sum(leak), len(results)) if any(leak)
                            else cluster_bootstrap_ci(leak, keys, samples=samples)),
        distractors_reachable=sum(1 for r in results if r.distractor_reachable),
        clusters=len(set(keys)),
        results=results,
    )
