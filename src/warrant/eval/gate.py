"""A quality floor, so a regression fails a build instead of being noticed a month later.

Every measurement in this repo is a snapshot of a day. Nothing stops the next refactor from
quietly costing four points of sufficiency, because nothing reads the old number back. This
does.

**The floor is a bootstrap lower bound, not a hand-picked threshold.** Sufficiency on the
test split is 97.8% with a 94.9-100 section-clustered interval; a gate set at 97.8% fails
about half the time on an unchanged system, and a gate set at "97.8 minus something that
felt safe" is a number nobody can defend. The recorded floor is the lower end of the
interval the reference run measured, so a passing build means "not distinguishable from the
reference", and a failing one means the drop is larger than the sampling noise that produced
the reference.

**A floor is only comparable within one configuration.** The config hash is recorded with
it, and a mismatch is reported as *incomparable* rather than as a pass or a fail. A gate
that silently compares a reranked run against a lexical-only floor is worse than no gate: it
passes exactly when the system got cheaper and worse.

**One-sided, on purpose.** An improvement never fails the gate. It does print, loudly,
because a recorded floor that nobody re-records goes stale, and the moment to re-record is
when the number moved up on purpose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Buckets whose only meaningful measure is the distractor rate -- there is nothing to
#: retrieve, so a sufficiency floor over them would compare against a number that is not
#: defined. `BucketResult.measures_absence` says the same thing at the type level; this is
#: the persisted form, which has to survive a round trip through JSON.
_ABSENCE_SUFFIX = "exclusion"


@dataclass(frozen=True)
class BucketFloor:
    bucket: str
    n: int
    #: Lower end of the reference run's clustered-bootstrap interval. None where the bucket
    #: measures absence and sufficiency is not defined for it.
    sufficiency_floor: float | None
    #: Upper end for the distractor rate: it is a rate to stay *below*, so its floor is a
    #: ceiling. Named for what it does rather than for the direction of the inequality,
    #: because "floor" applied to a badness rate is how a gate ends up backwards.
    distractor_ceiling: float | None
    #: 0 means no distractor was ever admitted, so the rate is a tautology and gating on it
    #: would gate on the SQL rather than on the system.
    distractors_reachable: int = 0


@dataclass(frozen=True)
class Floor:
    config_hash: str
    split: str
    recorded_at: str
    buckets: list[BucketFloor] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def load(cls, path: Path) -> Floor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(config_hash=data["config_hash"], split=data["split"],
                   recorded_at=data["recorded_at"],
                   buckets=[BucketFloor(**b) for b in data["buckets"]])


@dataclass(frozen=True)
class Violation:
    bucket: str
    metric: str
    floor: float
    observed: float

    def __str__(self) -> str:
        direction = "below" if self.metric == "sufficiency" else "above"
        return (f"{self.bucket}: {self.metric} {self.observed * 100:.1f}% is {direction} "
                f"the recorded floor {self.floor * 100:.1f}%")


@dataclass(frozen=True)
class GateResult:
    ok: bool
    comparable: bool
    violations: list[Violation] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    detail: str = ""


def record(results: dict[str, Any], *, config_hash: str, split: str,
           recorded_at: str) -> Floor:
    """Turn a scored run into the floor a later run is held to.

    ``recorded_at`` is passed rather than read from the clock so the file is reproducible
    from a script and so a test can assert on it. A gate artifact whose content changes when
    nothing changed is one people learn to regenerate without reading.
    """
    floors = []
    for name, bucket in sorted(results.items()):
        absence = name.endswith(_ABSENCE_SUFFIX)
        reachable = getattr(bucket, "distractors_reachable", 0)
        floors.append(BucketFloor(
            bucket=name,
            n=bucket.n,
            sufficiency_floor=None if absence else float(bucket.sufficiency_ci.lo),
            # A rate with no reachable distractor is 0% by construction. Recording a ceiling
            # for it would gate on a tautology, and the first real distractor to become
            # reachable would then read as a regression.
            distractor_ceiling=(float(bucket.distractor_rate_ci.hi)
                                if reachable else None),
            distractors_reachable=reachable,
        ))
    return Floor(config_hash=config_hash, split=split, recorded_at=recorded_at,
                 buckets=floors)


def check(floor: Floor, results: dict[str, Any], *, config_hash: str) -> GateResult:
    """Compare a fresh run against a recorded floor.

    Improvements are collected but never fail. They are surfaced because a floor nobody
    re-records goes stale, and the moment to re-record is exactly when a number moved up on
    purpose.
    """
    if floor.config_hash != config_hash:
        return GateResult(
            ok=True, comparable=False,
            detail=(f"floor was recorded at config {floor.config_hash}, this run is "
                    f"{config_hash}. Not comparable -- a gate that compares a reranked run "
                    "against a lexical-only floor passes exactly when the system got "
                    "cheaper and worse. Re-record with `warrant eval gate --record`."))

    violations: list[Violation] = []
    improvements: list[str] = []
    missing: list[str] = []

    for bucket_floor in floor.buckets:
        bucket = results.get(bucket_floor.bucket)
        if bucket is None:
            missing.append(bucket_floor.bucket)
            continue

        if bucket_floor.sufficiency_floor is not None:
            observed = float(bucket.sufficiency)
            if observed < bucket_floor.sufficiency_floor:
                violations.append(Violation(bucket_floor.bucket, "sufficiency",
                                            bucket_floor.sufficiency_floor, observed))
            elif observed > bucket_floor.sufficiency_floor + 0.02:
                improvements.append(
                    f"{bucket_floor.bucket}: sufficiency {observed * 100:.1f}% is above the "
                    f"recorded floor {bucket_floor.sufficiency_floor * 100:.1f}%")

        if bucket_floor.distractor_ceiling is not None:
            observed = float(bucket.distractor_rate)
            if observed > bucket_floor.distractor_ceiling:
                violations.append(Violation(bucket_floor.bucket, "distractor_rate",
                                            bucket_floor.distractor_ceiling, observed))

    # A bucket that vanished is a regression the metrics cannot see: the run scored nothing
    # where the reference scored something, and every remaining number would still pass.
    ok = not violations and not missing
    detail = ""
    if missing:
        detail = (f"buckets in the floor but not in this run: {', '.join(missing)}. "
                  "A bucket that disappears is a regression no metric can catch, because "
                  "every number that remains still passes.")
    return GateResult(ok=ok, comparable=True, violations=violations,
                      improvements=improvements, missing=missing, detail=detail)
