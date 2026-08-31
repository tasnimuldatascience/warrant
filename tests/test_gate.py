"""The quality gate, and the four ways a gate quietly stops gating.

A gate that never fails is indistinguishable from no gate, and every test here is about a
specific way that happens: a floor compared across configurations, a tautological metric
gated on, a bucket that disappears, and a badness rate gated in the wrong direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from warrant.eval.gate import Floor, check, record
from warrant.eval.stats import Interval

RECORDED = "2026-08-30T00:00:00+00:00"


@dataclass
class FakeBucket:
    bucket: str
    n: int
    sufficiency: float
    sufficiency_ci: Interval
    distractor_rate: float
    distractor_rate_ci: Interval
    distractors_reachable: int = 0


def bucket(name="temporal", sufficiency=0.978, lo=0.949, hi=1.0, rate=0.0,
           rate_hi=0.02, reachable=40) -> FakeBucket:
    return FakeBucket(name, 229, sufficiency, Interval(lo, hi), rate,
                      Interval(0.0, rate_hi), reachable)


def floor_of(*buckets) -> Floor:
    return record({b.bucket: b for b in buckets}, config_hash="cfg1", split="test",
                  recorded_at=RECORDED)


def test_the_floor_is_the_interval_lower_bound_not_the_point_estimate():
    """Sufficiency is 97.8% with a 94.9-100 clustered interval. A gate at 97.8% fails about
    half of all unchanged runs; the recorded floor has to be the lower bound so that passing
    means 'not distinguishable from the reference'."""
    floor = floor_of(bucket())
    assert floor.buckets[0].sufficiency_floor == pytest.approx(0.949)


def test_a_run_within_the_noise_band_passes():
    floor = floor_of(bucket())
    result = check(floor, {"temporal": bucket(sufficiency=0.955)}, config_hash="cfg1")
    assert result.ok and not result.violations


def test_a_real_regression_fails():
    floor = floor_of(bucket())
    result = check(floor, {"temporal": bucket(sufficiency=0.90)}, config_hash="cfg1")
    assert not result.ok
    assert "sufficiency 90.0% is below the recorded floor 94.9%" in str(result.violations[0])


def test_a_different_configuration_is_incomparable_not_a_pass():
    """A gate that compares a reranked run against a lexical-only floor passes exactly when
    the system got cheaper and worse. That has to be reported, not absorbed."""
    floor = floor_of(bucket())
    result = check(floor, {"temporal": bucket(sufficiency=0.10)}, config_hash="cfg2")
    assert not result.comparable
    assert result.violations == []
    assert "not comparable" in result.detail.lower()


def test_a_bucket_that_vanishes_fails_even_though_every_number_left_passes():
    """The regression no metric can see: the run scored nothing where the reference scored
    something, and each surviving number is still fine."""
    floor = floor_of(bucket("temporal"), bucket("human", sufficiency=0.79, lo=0.60))
    result = check(floor, {"temporal": bucket()}, config_hash="cfg1")
    assert not result.ok
    assert result.missing == ["human"]


def test_a_tautological_distractor_rate_is_not_gated_on():
    """0% with no reachable distractor restates the SQL rather than measuring the system.
    Recording a ceiling for it would make the first genuinely reachable distractor read as
    a regression."""
    floor = floor_of(bucket(reachable=0))
    assert floor.buckets[0].distractor_ceiling is None
    result = check(floor, {"temporal": bucket(rate=0.5, reachable=0)}, config_hash="cfg1")
    assert result.ok


def test_the_distractor_rate_is_gated_in_the_right_direction():
    """It is a rate to stay below, so its bound is a ceiling. Applying 'floor' to a badness
    rate is exactly how a gate ends up backwards and passes on every regression."""
    floor = floor_of(bucket(rate=0.01, rate_hi=0.05))
    assert check(floor, {"temporal": bucket(rate=0.02)}, config_hash="cfg1").ok
    worse = check(floor, {"temporal": bucket(rate=0.30)}, config_hash="cfg1")
    assert not worse.ok
    assert worse.violations[0].metric == "distractor_rate"


def test_an_absence_bucket_gets_no_sufficiency_floor():
    """An exclusion bucket has nothing to retrieve, so a sufficiency floor over it compares
    against a number that is not defined."""
    floor = floor_of(bucket("scope-exclusion"))
    assert floor.buckets[0].sufficiency_floor is None


def test_an_improvement_never_fails_but_is_reported():
    """A floor nobody re-records goes stale, and the moment to re-record is when a number
    moved up on purpose."""
    floor = floor_of(bucket())
    result = check(floor, {"temporal": bucket(sufficiency=0.999)}, config_hash="cfg1")
    assert result.ok
    assert result.improvements and "above the recorded floor" in result.improvements[0]


def test_the_floor_round_trips_through_json(tmp_path):
    """The artifact is committed and read back by a build. A field lost in serialisation is
    a gate that silently stops checking one thing."""
    floor = floor_of(bucket(), bucket("scope-exclusion"))
    path = tmp_path / "floor.json"
    path.write_text(floor.to_json(), encoding="utf-8")
    reloaded = Floor.load(path)
    assert reloaded == floor


def test_the_same_config_running_without_torch_is_incomparable():
    """The case the config hash cannot see. A CI runner with no torch loads the identical
    configs/default.yaml, hashes identically, and runs lexical-only. Comparing that against
    a reranked floor would report a pass on a system that got cheaper and worse -- which is
    the exact failure the config-hash check exists to prevent, arriving through the one door
    it does not cover."""
    floor = record({"temporal": bucket()}, config_hash="cfg1", split="test",
                   recorded_at=RECORDED,
                   models={"dense": "bge-small", "rerank": "ms-marco"})
    lexical = check(floor, {"temporal": bucket(sufficiency=0.10)},
                    config_hash="cfg1", models={})
    assert not lexical.comparable
    assert lexical.violations == []
    assert "dense" in lexical.detail and "rerank" in lexical.detail


def test_a_matching_model_set_compares_normally():
    models = {"dense": "bge-small"}
    floor = record({"temporal": bucket()}, config_hash="cfg1", split="test",
                   recorded_at=RECORDED, models=models)
    assert check(floor, {"temporal": bucket(sufficiency=0.955)},
                 config_hash="cfg1", models=models).ok


def test_a_floor_recorded_before_models_were_tracked_still_loads(tmp_path):
    """Old artifacts stay readable, and compare only against another modelless run -- which
    is the conservative direction: they never silently grade a reranked run."""
    path = tmp_path / "floor.json"
    path.write_text('{"config_hash": "cfg1", "split": "test", "recorded_at": "x",'
                    ' "buckets": []}', encoding="utf-8")
    assert Floor.load(path).models == {}


def test_a_grown_benchmark_is_incomparable():
    """The third thing the config hash cannot see. benchmarks/human.yaml is not in the
    config, so growing it from 29 items to 212 left the hash identical while the floor went
    on describing a smaller, easier set. A sufficiency floor means nothing apart from the
    items it was measured over."""
    floor = record({"human": bucket("human", sufficiency=0.79, lo=0.60)},
                   config_hash="cfg1", split="test", recorded_at=RECORDED)
    grown = bucket("human", sufficiency=0.79, lo=0.60)
    grown.n = 212
    result = check(floor, {"human": grown}, config_hash="cfg1")
    assert not result.comparable
    assert "benchmark changed" in result.detail


def test_an_unchanged_benchmark_still_compares():
    floor = record({"human": bucket("human")}, config_hash="cfg1", split="test",
                   recorded_at=RECORDED)
    assert check(floor, {"human": bucket("human")}, config_hash="cfg1").comparable
