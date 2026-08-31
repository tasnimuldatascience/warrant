"""External baselines: verifying each one actually runs the stages its name promises and no
others -- `bm25_only` really ignores both predicates, `NaiveDense` really searches
unrestricted, `DensePostFilter` really filters after ranking rather than before it. No neural
models here, matching `tests/test_hybrid.py`: `FakeDenseIndex` uses the query text itself as
the "vector" and a hand-written ranking table, so the suite stays offline and the ranking each
test sees is exactly the one it wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from warrant.eval.baseline import DensePostFilter, NaiveDense, Shortfall, bm25_only, shortfall_stats
from warrant.index.store import Chunk, Store
from warrant.retrieve.scope import Scope

T0 = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.add([
            Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531",
                  anchor="a", heading="Earning within-grade increase",
                  text="performance must be at an acceptable level of competence",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="532.203#a", section_id="532.203", title=5, part="532",
                  anchor="a", heading="Structure of regular wage schedules",
                  text="each nonsupervisory and leader regular wage schedule",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630",
                  anchor="a", heading="Time limit for use of restored annual leave",
                  text="annual leave restored must be scheduled within two years",
                  valid_from="2017-01-01", valid_to="2020-08-10"),
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630",
                  anchor="a", heading="Time limit for use of restored annual leave",
                  text="annual leave restored must be scheduled within three years",
                  valid_from="2020-08-10"),
        ], system_from=T0)
        yield s


def _ids(store: Store) -> dict[str, int]:
    """version_id -> row id, so a test can point a FakeDenseIndex ranking at real rows."""
    return {r["version_id"]: r["id"] for r in
            store.db.execute("SELECT id, version_id FROM chunk")}


@dataclass
class FakeDenseIndex:
    """Deterministic double for `DenseIndex`. Like `test_hybrid.py`'s `StubIndex`, `encode`
    returns the query text unchanged and `search` keys off it directly -- what these tests
    need to prove is that `baseline.py` wires `allowed`/`limit` correctly, not that a real
    encoder ranks well.
    """

    model: str = "fake"
    rankings: dict[str, list[int]] = field(default_factory=dict)

    def encode(self, text: str) -> str:
        return text

    def search(self, vector, *, allowed, limit):
        ids = self.rankings.get(vector, [])
        if allowed is not None:
            ids = [i for i in ids if i in allowed]
        ids = ids[:limit]
        return [(i, float(len(ids) - rank)) for rank, i in enumerate(ids)]


# -- bm25_only ---------------------------------------------------------------------


def test_bm25_only_ignores_the_as_of_predicate(store: Store):
    """The obvious way to break a temporal system: the superseded version is dated
    2017-2020, the query is asked in 2019, and a predicate-on retriever would never surface
    the 2020-08-10 text -- see test_hybrid.py's equivalent test with the predicate on."""
    r = bm25_only(store, final_k=10)
    trace = r.retrieve("annual leave restored scheduled", as_of="2019-01-01")
    versions = [v for v in trace.final if v.startswith("630.306")]
    assert len(versions) == 2, "both versions must be candidates with no as-of predicate"


def test_bm25_only_ignores_the_applicability_predicate(store: Store):
    r = bm25_only(store, final_k=10)
    trace = r.retrieve("performance acceptable level competence", as_of="2021-01-01",
                       scope=Scope.of(pay_system="FWS"))
    assert any(v.startswith("531.404") for v in trace.final), (
        "a part-531 (competitive-service) chunk must survive an FWS scope with no "
        "applicability predicate")


def test_bm25_only_runs_no_dense_or_rerank_stage(store: Store):
    trace = bm25_only(store).retrieve("annual leave", as_of="2021-01-01")
    assert trace.stages_run == ["lexical", "fused", "final"]


# -- NaiveDense ----------------------------------------------------------------------


def test_naive_dense_is_blind_to_the_as_of_predicate(store: Store):
    """Cosine top-k with no restriction: whatever the encoder ranks first comes back,
    including a version that was not in force on the day being asked about."""
    ids = _ids(store)
    after_id = ids["630.306#a@2020-08-10"]
    index = FakeDenseIndex(rankings={"annual leave": [after_id]})
    r = NaiveDense(store=store, dense_index=index, final_k=5)
    trace = r.retrieve("annual leave", as_of="2018-01-01")
    assert trace.final == ["630.306#a@2020-08-10"]


def test_naive_dense_is_blind_to_the_scope_predicate(store: Store):
    ids = _ids(store)
    fws_id = ids["532.203#a@2017-01-01"]
    index = FakeDenseIndex(rankings={"wage schedule": [fws_id]})
    r = NaiveDense(store=store, dense_index=index, final_k=5)
    trace = r.retrieve("wage schedule", as_of="2021-01-01",
                       scope=Scope.of(pay_system="GS"))
    assert trace.final == ["532.203#a@2017-01-01"]


def test_naive_dense_caps_at_final_k(store: Store):
    ids = _ids(store)
    index = FakeDenseIndex(rankings={"leave": list(ids.values())})
    trace = NaiveDense(store=store, dense_index=index, final_k=2).retrieve(
        "leave", as_of="2021-01-01")
    assert len(trace.final) == 2


def test_naive_dense_runs_only_the_dense_stage(store: Store):
    """`Trace.stages_run` always lists "lexical" first regardless of whether anything wrote
    to it (see hybrid.py) -- checked here via the stage contents instead, which is what
    actually distinguishes this baseline from the hybrid pipeline."""
    ids = _ids(store)
    index = FakeDenseIndex(rankings={"leave": [ids["531.404#a@2017-01-01"]]})
    trace = NaiveDense(store=store, dense_index=index).retrieve("leave", as_of="2021-01-01")
    assert trace.dense and trace.final
    assert not trace.lexical and not trace.reranked and not trace.fused


def test_naive_dense_records_a_total_timing(store: Store):
    """Latency has to be comparable across the four configurations in the report; the
    lexical/reranked Retriever already records `timings["total"]`, so this baseline must too."""
    trace = NaiveDense(store=store, dense_index=FakeDenseIndex()).retrieve(
        "leave", as_of="2021-01-01")
    assert "total" in trace.timings and trace.timings["total"] >= 0.0


def test_naive_dense_describes_an_unrestricted_candidate_space(store: Store):
    """`run.score` reads these two attributes to decide whether a distractor was ever a
    candidate at all; both must say "nothing is excluded" or the reachability accounting in
    the report would understate what this baseline actually searches over."""
    r = NaiveDense(store=store, dense_index=FakeDenseIndex())
    assert r.temporal is False
    assert r.parts_universe == []


# -- DensePostFilter -----------------------------------------------------------------


def test_post_filter_discards_what_is_not_in_force(store: Store):
    """The encoder ranks the superseded version first; the post-filter must still discard it
    when asked about a date only the newer version covers."""
    ids = _ids(store)
    before_id = ids["630.306#a@2017-01-01"]
    after_id = ids["630.306#a@2020-08-10"]
    index = FakeDenseIndex(rankings={"annual leave": [before_id, after_id]})
    r = DensePostFilter(store=store, dense_index=index, candidates_dense=10, final_k=5)
    trace = r.retrieve("annual leave", as_of="2021-01-01")
    assert trace.final == ["630.306#a@2020-08-10"]


def test_post_filter_does_not_apply_the_scope_predicate(store: Store):
    """Only `valid_from`/`valid_to` are filtered -- the module docstring is explicit that
    applicability is not. A part-531 chunk asked about under an FWS scope must still survive
    if it is in force, because this baseline never looks at scope at all."""
    ids = _ids(store)
    gs_id = ids["531.404#a@2017-01-01"]
    index = FakeDenseIndex(rankings={"performance": [gs_id]})
    r = DensePostFilter(store=store, dense_index=index, candidates_dense=10, final_k=5)
    trace = r.retrieve("performance", as_of="2021-01-01", scope=Scope.of(pay_system="FWS"))
    assert trace.final == ["531.404#a@2017-01-01"]


def test_post_filter_can_leave_fewer_than_final_k(store: Store):
    """The failure mode the report is built to surface: post-filtering does not top back up
    to k, it returns whatever survived -- here, nothing, because every ranked candidate is
    off the clock for the asked-about date."""
    ids = _ids(store)
    before_id = ids["630.306#a@2017-01-01"]  # closed 2020-08-10, asked about in 2021
    index = FakeDenseIndex(rankings={"annual leave": [before_id]})
    r = DensePostFilter(store=store, dense_index=index, candidates_dense=10, final_k=5)
    trace = r.retrieve("annual leave", as_of="2021-01-01")
    assert trace.final == []
    assert trace.admitted == 0


def test_post_filter_admitted_counts_survivors_before_the_final_k_cut(store: Store):
    """`trace.admitted` must reflect the post-filter survivor count, not the post-cut count,
    or a trace could never distinguish "exactly k survived" from "more than k survived"."""
    ids = _ids(store)
    live_ids = [ids["531.404#a@2017-01-01"], ids["532.203#a@2017-01-01"]]
    index = FakeDenseIndex(rankings={"q": live_ids})
    r = DensePostFilter(store=store, dense_index=index, candidates_dense=10, final_k=1)
    trace = r.retrieve("q", as_of="2021-01-01")
    assert trace.admitted == 2
    assert len(trace.final) == 1


def test_shortfall_stats_counts_items_that_ran_dry():
    """A bench item stub is enough here -- `shortfall_stats` only reads `.query`,
    `.as_of` and `.scope`, and building a real `BenchItem` would need evidence sets this
    test has no use for."""

    @dataclass
    class Item:
        query: str
        as_of: str
        scope: Scope = Scope()

    class AlwaysShort:
        final_k = 3

        def retrieve(self, query, *, as_of, scope):
            from warrant.retrieve.hybrid import Trace

            t = Trace(query=query, as_of=as_of, scope=scope.describe())
            t.record("final", ["only-one"])
            return t

    result = shortfall_stats(AlwaysShort(), [Item("a", "2021-01-01"), Item("b", "2021-01-01")])
    assert result == Shortfall(short=2, n=2)
    assert result.rate == 1.0


def test_shortfall_stats_rate_handles_zero_items():
    class Empty:
        final_k = 3

        def retrieve(self, query, *, as_of, scope):
            raise AssertionError("must not be called for an empty item list")

    assert shortfall_stats(Empty(), []).rate == 0.0
