"""The scored trace, and the store it survives in.

Two things are worth pinning down here. The first is that every score a stage computes is now
kept -- BM25 out of the SQL, the RRF weight, the cross-encoder logit -- because each of them
used to be discarded on the line after it was produced, and a trace without them can say that
an order changed but never that a preference was strong. The second is that keeping them cost
no consumer anything: `trace.lexical` and its siblings still hand back bare version ids, which
is what localisation and evaluation read, so this is an extension rather than a migration.

Offline throughout: a hand-built store and a stub index, no torch, no corpus.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store
from warrant.observe import TraceStore
from warrant.retrieve.hybrid import Candidate, Retriever, Trace, reciprocal_rank_fusion
from warrant.retrieve.scope import Scope

T0 = "2020-01-01T00:00:00+00:00"


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
                  valid_from="2017-01-01"),
        ], system_from=T0)
        yield s


class StubIndex:
    """Scores by row id, descending. The index owns its encoder, so no model is needed."""

    model = "stub-encoder"

    def encode(self, text: str) -> str:
        return text

    def search(self, vector, *, allowed, limit):
        return [(i, 1.0 / i) for i in sorted(allowed)[:limit]]


class ReverseReranker:
    """Prefers whatever the fused order put last. A deterministic disagreement."""

    def predict(self, pairs):
        return [float(i) for i in range(len(pairs))]


def make(store: Store, **kw) -> Retriever:
    return Retriever(store=store, candidates_lexical=50, candidates_dense=50,
                     rerank_top_k=20, final_k=3, parts_universe=["531", "532", "630"],
                     config_hash="cfg0001", **kw)


# -- scores ----------------------------------------------------------------------


def test_lexical_candidates_carry_the_bm25_score_the_query_already_selected(store: Store):
    """`Store.search` selects bm25() and the retriever threw it away one line later."""
    got = make(store).retrieve("annual leave restored", as_of="2021-01-01")
    assert got.candidates("lexical")
    assert all(c.score is not None for c in got.candidates("lexical"))


def test_ranks_number_each_stage_from_one_in_its_own_order(store: Store):
    got = make(store).retrieve("leave wage performance", as_of="2021-01-01")
    for stage in ("lexical", "fused", "final"):
        assert [c.rank for c in got.candidates(stage)] == \
            list(range(1, len(got.candidates(stage)) + 1))


def test_the_final_cut_renumbers_rather_than_keeping_its_old_ranks(store: Store):
    """The final list is a slice of an earlier stage. A slice that kept the ranks of the
    stage it came from would report the third answer as rank 14."""
    got = make(store, reranker=ReverseReranker()).retrieve(
        "leave wage performance", as_of="2021-01-01")
    assert [c.rank for c in got.candidates("final")] == [1, 2, 3]
    # The score is the reranker's, not the position's: the ordering is inherited, the
    # number behind it is the one the last stage to score it produced.
    assert [c.score for c in got.candidates("final")] == \
        [c.score for c in got.candidates("reranked")[:3]]


def test_fused_candidates_carry_the_rrf_weight_in_descending_order(store: Store):
    got = make(store, dense_index=StubIndex()).retrieve("annual leave", as_of="2021-01-01")
    weights = [c.score for c in got.candidates("fused")]
    assert all(w is not None for w in weights)
    assert weights == sorted(weights, reverse=True)


def test_dense_candidates_carry_the_similarity_the_index_returned(store: Store):
    got = make(store, dense_index=StubIndex()).retrieve("annual leave", as_of="2021-01-01")
    assert [c.score for c in got.candidates("dense")] == \
        sorted((c.score for c in got.candidates("dense")), reverse=True)


def test_a_stage_that_ordered_without_scoring_records_no_score(store: Store):
    """None, not 0.0. Several scorers legitimately produce zero, and a trace that cannot
    tell "unscored" from "scored zero" invents a preference nobody expressed."""
    assert Trace(query="q", as_of="2021-01-01", scope="government-wide",
                 lexical=["a@1"]).candidates("lexical")[0].score is None


# -- backward compatibility ------------------------------------------------------


def test_stage_properties_still_return_bare_strings(store: Store):
    """Every downstream consumer -- autopsy, eval, the API, the generator's excerpt lookup,
    which binds these straight into SQL -- reads bare ids. Not a subclass of str: a subclass
    would pass every equality assertion here and still be a different type at a driver
    boundary."""
    got = make(store, dense_index=StubIndex(), reranker=ReverseReranker()).retrieve(
        "leave wage performance", as_of="2021-01-01")
    for stage in ("lexical", "dense", "fused", "reranked", "final"):
        ids = getattr(got, stage)
        assert ids, f"{stage} produced nothing to check"
        assert all(type(v) is str for v in ids)
        assert ids == got.stage(stage) == got.ids(stage)


def test_a_trace_written_from_bare_ids_reads_back_as_bare_ids():
    """How the autopsy tests build a trace, unchanged."""
    got = Trace(query="q", as_of="2021-01-01", scope="government-wide",
                lexical=["e@1"], fused=["x@1", "e@1"], reranked=["e@1", "x@1"],
                final=["e@1"])
    assert got.lexical == ["e@1"]
    assert got.fused == ["x@1", "e@1"]
    assert got.stages_run == ["lexical", "fused", "reranked", "final"]


def test_reciprocal_rank_fusion_still_returns_bare_ids():
    """The scored form is `fuse`; this name is imported elsewhere and keeps its contract."""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert all(type(v) is str for v in fused)
    assert set(fused) == {"a", "b"}


def test_an_unknown_stage_name_is_an_error_not_an_empty_list():
    got = Trace(query="q", as_of="2021-01-01", scope="government-wide")
    with pytest.raises(KeyError):
        got.candidates("reranking")


# -- timings, config, models -----------------------------------------------------


def test_timings_are_recorded_for_the_stages_that_ran(store: Store):
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    assert {"predicates", "lexical", "fusion", "total"} <= set(got.timings)
    assert all(ms >= 0.0 for ms in got.timings.values())


def test_a_stage_that_did_not_run_has_no_timing_rather_than_a_zero(store: Store):
    """Zero is a measurement. Absence is not, and reporting a stage that never ran as
    instantaneous is how a latency budget quietly stops adding up."""
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    assert "dense" not in got.timings and "rerank" not in got.timings
    with_dense = make(store, dense_index=StubIndex()).retrieve(
        "annual leave", as_of="2021-01-01")
    assert "dense" in with_dense.timings


def test_the_trace_records_what_produced_it(store: Store):
    got = make(store, dense_index=StubIndex(), reranker=ReverseReranker(),
               reranker_model="stub-cross-encoder").retrieve("leave", as_of="2021-01-01")
    assert got.config_hash == "cfg0001"
    assert got.models == {"dense": "stub-encoder", "rerank": "stub-cross-encoder"}


def test_the_scope_is_recorded_in_a_form_replay_can_rebuild(store: Store):
    """`describe()` is written for a person and is lossy to parse back. Counterfactual
    replay has to re-ask the same question, not a similar one."""
    got = make(store).retrieve("wage schedule", as_of="2021-01-01",
                               scope=Scope.of(pay_system="FWS"))
    assert got.scope == "pay_system=FWS"
    assert got.scope_facets == {"pay_system": "FWS"}


def test_the_admitted_count_is_the_rows_the_predicates_let_through(store: Store):
    narrow = make(store).retrieve("leave", as_of="2021-01-01",
                                  scope=Scope.of(pay_system="FWS"))
    wide = make(store).retrieve("leave", as_of="2021-01-01")
    assert narrow.admitted < wide.admitted


# -- the store -------------------------------------------------------------------


def test_a_trace_round_trips_with_its_scores_and_timings_intact(store: Store):
    got = make(store, dense_index=StubIndex(), reranker=ReverseReranker()).retrieve(
        "leave wage performance", as_of="2021-01-01", scope=Scope.of(pay_system="GS"))
    with TraceStore(":memory:") as traces:
        stored = traces.load(traces.record(got))

    for stage in ("lexical", "dense", "fused", "reranked", "final"):
        assert stored.candidates(stage) == got.candidates(stage)
    assert stored.timings == got.timings
    assert stored.config_hash == got.config_hash
    assert stored.models == got.models
    assert stored.scope_facets == got.scope_facets
    assert stored.excluded_parts == got.excluded_parts
    assert stored.admitted == got.admitted


def test_a_stored_trace_rebuilds_the_object_every_consumer_already_takes(store: Store):
    """The point of storing a trace rather than logging one: a month-old request can be
    handed to the failure autopsy without re-running a single query."""
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    with TraceStore(":memory:") as traces:
        rebuilt = traces.load(traces.record(got)).to_trace()
    assert rebuilt.final == got.final
    assert rebuilt.candidates("lexical") == got.candidates("lexical")
    assert rebuilt.stages_run == got.stages_run


def test_the_generated_columns_are_optional(store: Store):
    """A retrieval-only run has no prompt and no answer, and must still record a complete
    retrieval trace -- that install has no torch to produce one with."""
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    with TraceStore(":memory:") as traces:
        bare = traces.load(traces.record(got))
        full = traces.load(traces.record(
            got, prompt="answer using only...",
            answer={"claims": [{"text": "two years", "evidence": ["630.306#a@2017-01-01"]}]}))
    assert bare.prompt is None and bare.answer is None
    assert full.answer["claims"][0]["evidence"] == ["630.306#a@2017-01-01"]


def test_traces_come_back_newest_first(store: Store):
    r = make(store)
    with TraceStore(":memory:") as traces:
        first = traces.record(r.retrieve("leave", as_of="2019-01-01"),
                              created_at="2026-01-01T00:00:00+00:00")
        second = traces.record(r.retrieve("wage", as_of="2021-01-01"),
                               created_at="2026-02-01T00:00:00+00:00")
        assert [t.trace_id for t in traces.recent()] == [second, first]
        assert traces.count() == 2


def test_an_unrecorded_trace_id_is_a_key_error():
    with TraceStore(":memory:") as traces, pytest.raises(KeyError):
        traces.load("nosuchtrace")


def test_a_deleted_trace_takes_its_candidates_with_it(store: Store):
    """Traces are telemetry and have to be deletable; a delete that left the candidate rows
    behind would grow the file forever and resurrect the request on the next id collision."""
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    with TraceStore(":memory:") as traces:
        tid = traces.record(got)
        assert traces.delete(tid) == 1
        assert traces.db.execute(
            "SELECT COUNT(*) FROM trace_candidate WHERE trace_id = ?", (tid,)).fetchone()[0] == 0


def test_the_stored_shape_is_json_safe(store: Store):
    import json

    got = make(store, dense_index=StubIndex()).retrieve("annual leave", as_of="2021-01-01")
    with TraceStore(":memory:") as traces:
        payload = json.loads(json.dumps(traces.load(traces.record(got)).to_dict()))
    assert payload["stages"]["final"][0]["rank"] == 1
    assert payload["config_hash"] == "cfg0001"


def test_candidate_scores_survive_as_floats_not_strings(store: Store):
    got = make(store).retrieve("annual leave", as_of="2021-01-01")
    with TraceStore(":memory:") as traces:
        stored = traces.load(traces.record(got))
    assert all(isinstance(c.score, float) for c in stored.candidates("lexical"))
    assert isinstance(stored.candidates("final")[0], Candidate)
