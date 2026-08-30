"""Retrieval assembly: fusion, escaping, predicates, and the trace the autopsy reads.

No neural models here. Every test runs on the lexical path with a hand-built store, so the
suite stays offline and fast; the dense arm is exercised through a stub whose only job is to
prove that fusion and tracing treat it like any other ranking.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store
from warrant.retrieve.hybrid import (
    MAX_QUERY_TOKENS,
    Retriever,
    fts_query,
    reciprocal_rank_fusion,
)
from warrant.retrieve.scope import GOVERNMENT_WIDE, Scope

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


def make(store: Store, **kw) -> Retriever:
    return Retriever(store=store, candidates_lexical=50, candidates_dense=50,
                     rerank_top_k=20, final_k=5, parts_universe=["531", "532", "630"], **kw)


# -- fusion ----------------------------------------------------------------------


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[0] in {"a", "b"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_ranks_a_consensus_item_above_a_single_list_leader():
    """The property RRF is chosen for: agreement beats one strong opinion, without ever
    comparing a BM25 score to a cosine similarity."""
    fused = reciprocal_rank_fusion([["x", "consensus"], ["y", "consensus"]])
    assert fused[0] == "consensus"


def test_rrf_is_deterministic_for_tied_scores():
    a = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    b = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    assert a == b


def test_rrf_handles_a_single_ranking():
    assert reciprocal_rank_fusion([["only", "one"]]) == ["only", "one"]


# -- query escaping --------------------------------------------------------------


def test_fts_query_strips_punctuation_that_fts5_reads_as_syntax():
    """A heading with a colon is an FTS5 parse error, not a query."""
    q = fts_query("Earning within-grade increase: acceptable level")
    assert ":" not in q and "-" not in q
    assert '"within"' in q and '"grade"' in q


def test_fts_query_never_returns_empty_syntax():
    assert fts_query("!!!") == '""'


# -- predicates ------------------------------------------------------------------


def test_as_of_predicate_selects_the_version_in_force(store: Store):
    """Asserted on the versions of 630.306 specifically: a bag-of-words OR query also
    matches unrelated sections, and pinning the whole result list would make this test fail
    for reasons that have nothing to do with dating."""
    r = make(store)

    def versions_of_630(as_of: str) -> list[str]:
        return [v for v in r.retrieve("annual leave restored scheduled", as_of=as_of).final
                if v.startswith("630.306")]

    assert versions_of_630("2019-01-01") == ["630.306#a@2017-01-01"]
    assert versions_of_630("2021-01-01") == ["630.306#a@2020-08-10"]


def test_applicability_predicate_excludes_parts_that_do_not_govern(store: Store):
    r = make(store)
    fws = r.retrieve("wage schedule regular", as_of="2021-01-01",
                     scope=Scope.of(pay_system="FWS"))
    assert "532" not in fws.excluded_parts
    assert "531" in fws.excluded_parts
    assert all(not v.startswith("531.") for v in fws.final)


def test_government_wide_scope_excludes_nothing(store: Store):
    trace = make(store).retrieve("wage schedule", as_of="2021-01-01",
                                 scope=GOVERNMENT_WIDE)
    assert trace.excluded_parts == []


def test_predicates_shrink_the_admitted_set_before_ranking(store: Store):
    """The predicate is applied to the candidate space, not to a ranked list. If admitted
    counted everything, superseded text would be consuming candidate slots."""
    r = make(store)
    wide = r.retrieve("leave", as_of="2021-01-01")
    narrow = r.retrieve("leave", as_of="2021-01-01", scope=Scope.of(pay_system="FWS"))
    assert narrow.admitted < wide.admitted


# -- trace -----------------------------------------------------------------------


def test_trace_records_every_stage_that_ran(store: Store):
    trace = make(store).retrieve("annual leave restored", as_of="2021-01-01")
    assert trace.stages_run == ["lexical", "fused", "final"]
    assert trace.lexical and trace.fused
    assert not trace.dense and not trace.reranked


def test_trace_reports_the_dense_stage_when_an_index_is_present(store: Store):
    class StubIndex:
        """The index owns its encoder, so a stub needs no model and the suite stays
        offline. That is the point of `DenseIndex.encode` existing at all."""

        model = "stub"

        def encode(self, text):
            return text

        def search(self, vector, *, allowed, limit):
            return [(i, 1.0) for i in sorted(allowed)[:limit]]

    r = make(store, dense_index=StubIndex())
    trace = r.retrieve("annual leave", as_of="2021-01-01")
    assert trace.dense
    assert "dense" in trace.stages_run


def test_final_is_capped_at_final_k(store: Store):
    trace = make(store).retrieve("leave wage performance schedule", as_of="2021-01-01")
    assert len(trace.final) <= 5


# -- query cost ------------------------------------------------------------------


def test_repeated_tokens_are_deduplicated():
    """A bag-of-words OR gains nothing from a repeat and loses a great deal: FTS5 merges
    the same postings list against itself once per occurrence. Measured on the real corpus,
    one token repeated 2,600 times cost 29 seconds against 16 ms for a normal query --
    ~1,800x amplification from a 15.6 KB URL that fits inside the HTTP header limit."""
    q = fts_query("leave " * 2600)
    assert q.count('"') // 2 == 1


def test_token_count_is_capped():
    q = fts_query(" ".join(f"tok{i}" for i in range(5000)))
    assert q.count('"') // 2 == MAX_QUERY_TOKENS


def test_deduplication_preserves_order_and_content():
    """The cap must not silently reorder a query -- earlier terms are the ones a user
    actually typed first, and truncation should drop the tail, not a random subset."""
    assert fts_query("annual leave annual restored leave") == (
        '"annual" OR "leave" OR "restored"')
