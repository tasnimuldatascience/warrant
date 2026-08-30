"""The two replay modes, and the difference in what they promise.

Artifact replay is tested against a corpus that is gone: the store is closed before the replay
runs, so any query at all would raise. That is the guarantee -- reconstruction from the record
alone, still exact after the index has been rebuilt underneath it.

Counterfactual replay is tested by changing the retriever and asserting the diff names the
stage that moved. The interesting cases are the ones a single `changed` boolean would flatten:
a narrower cut is not the same regression as a reranker that changed its mind, and the
earliest stage to move is the one worth reporting even when a later one moved louder.

Offline: a hand-built store, a stub index, a stub reranker, no torch.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store
from warrant.observe import TraceStore, artifact_replay, counterfactual_replay
from warrant.observe.replay import counterfactual_sweep
from warrant.retrieve.hybrid import Retriever
from warrant.retrieve.scope import Scope

#: Belief times. Explicit throughout so the suite never depends on the wall clock.
T0 = "2020-01-01T00:00:00+00:00"
T1 = "2021-06-01T00:00:00+00:00"
NOW = "2026-01-01T00:00:00+00:00"

CHUNKS = [
    Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531", anchor="a",
          heading="Earning within-grade increase",
          text="performance must be at an acceptable level of competence",
          valid_from="2017-01-01"),
    Chunk(chunk_id="532.203#a", section_id="532.203", title=5, part="532", anchor="a",
          heading="Structure of regular wage schedules",
          text="each nonsupervisory and leader regular wage schedule",
          valid_from="2017-01-01"),
    Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
          heading="Time limit for use of restored annual leave",
          text="annual leave restored must be scheduled within two years",
          valid_from="2017-01-01"),
    Chunk(chunk_id="630.402#a", section_id="630.402", title=5, part="630", anchor="a",
          heading="Application for advanced leave",
          text="an employee must apply in writing for advanced annual leave",
          valid_from="2017-01-01"),
]


class ReverseReranker:
    """Prefers whatever the fused order put last. A deterministic disagreement."""

    def predict(self, pairs):
        return [float(i) for i in range(len(pairs))]


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.add(CHUNKS, system_from=T0)
        yield s


@pytest.fixture
def traces() -> TraceStore:
    with TraceStore(":memory:") as t:
        yield t


def make(store: Store, **kw) -> Retriever:
    settings = dict(candidates_lexical=50, candidates_dense=50, rerank_top_k=20, final_k=3,
                    parts_universe=["531", "532", "630"], config_hash="cfg0001")
    return Retriever(store=store, **(settings | kw))


def record(store: Store, traces: TraceStore, *, query: str = "annual leave restored",
           as_of: str = "2021-01-01", scope: Scope | None = None, **kw) -> str:
    trace = make(store, **kw).retrieve(query, as_of=as_of, system_time=NOW,
                                       scope=scope or Scope())
    return traces.record(trace)


# -- artifact replay -------------------------------------------------------------


def test_artifact_replay_reproduces_the_request_from_the_record_alone(store: Store,
                                                                     traces: TraceStore):
    """The corpus is closed before the replay runs, so a single query would raise. This is
    the mode that survives an index rebuild, and it survives it by never asking."""
    original = make(store).retrieve("annual leave restored", as_of="2021-01-01")
    trace_id = traces.record(original)
    store.close()

    got = artifact_replay(traces, trace_id)
    assert got.final == original.final
    assert got.candidates("fused") == original.candidates("fused")
    assert got.timings == original.timings


def test_artifact_replay_still_holds_after_the_corpus_is_rebuilt(store: Store,
                                                                 traces: TraceStore):
    """A re-chunk changes every version id in the store. The stored trace is a record of
    what happened, not a query against what is there now, so it does not move."""
    trace_id = record(store, traces)
    before = traces.load(trace_id).final

    store.db.execute("DELETE FROM chunk")
    store.add([Chunk(chunk_id="630.306#a1", section_id="630.306", title=5, part="630",
                     anchor="a-1", heading="Time limit",
                     text="annual leave restored must be scheduled within two years",
                     valid_from="2017-01-01")], system_from=T1)

    assert artifact_replay(traces, trace_id).final == before
    assert before, "the fixture must retrieve something for this to mean anything"


def test_replaying_an_unrecorded_id_is_a_key_error(traces: TraceStore):
    with pytest.raises(KeyError):
        artifact_replay(traces, "nosuchtrace")


# -- counterfactual replay -------------------------------------------------------


def test_an_unchanged_pipeline_diverges_nowhere(store: Store, traces: TraceStore):
    trace_id = record(store, traces)
    got = counterfactual_replay(traces, trace_id, make(store), system_time=NOW)
    assert not got.changed
    assert got.first_divergence is None
    assert not got.answer_set_changed
    assert not got.config_changed


def test_a_narrower_final_k_is_reported_as_a_final_stage_change(store: Store,
                                                                traces: TraceStore):
    trace_id = record(store, traces)
    got = counterfactual_replay(traces, trace_id, make(store, final_k=1), system_time=NOW)

    assert got.changed
    assert got.first_divergence == "final"
    assert got.left_final and not got.entered_final
    assert got.answer_set_changed
    assert len(got.replayed.final) == 1


def test_first_divergence_names_the_earliest_stage_not_the_loudest(store: Store,
                                                                   traces: TraceStore):
    """Narrowing the candidate list moves every stage after it. Reporting the final cut
    would send a reader to tune the last knob in the chain rather than the first."""
    trace_id = record(store, traces)
    got = counterfactual_replay(traces, trace_id, make(store, candidates_lexical=1),
                                system_time=NOW)
    assert got.first_divergence == "lexical"
    assert got.stage("fused").changed and got.stage("final").changed


def test_a_reordering_is_distinguished_from_a_substitution(store: Store,
                                                           traces: TraceStore):
    """The reranker changed its mind about an unchanged set. Same evidence reaches the
    generator, in a different order -- a ranking regression, not an evidence one."""
    trace_id = record(store, traces, reranker=ReverseReranker(), rerank_top_k=3)
    got = counterfactual_replay(
        traces, trace_id, make(store, rerank_top_k=3), system_time=NOW)

    final = got.stage("final")
    assert final.reordered
    assert not got.answer_set_changed
    assert sorted(final.before) == sorted(final.after)


def test_a_config_change_is_reported_beside_the_diff_not_as_its_cause(store: Store,
                                                                      traces: TraceStore):
    """The hashes bracket the change; they do not explain it. Embeddings and chunk
    boundaries are recorded by hash and never rebuilt, so a diff across a hash change
    attributes nothing on its own."""
    trace_id = record(store, traces)
    retriever = make(store, final_k=1)
    retriever.config_hash = "cfg0002"
    got = counterfactual_replay(traces, trace_id, retriever, system_time=NOW)

    assert got.config_changed
    assert got.to_dict()["config_hash"] == {"then": "cfg0001", "now": "cfg0002",
                                            "changed": True}


def test_the_stored_scope_is_re_asked_as_asked(store: Store, traces: TraceStore):
    """Replaying a scoped request government-wide would report a scope regression as a
    retrieval one, so the profile is rebuilt from the recorded facets."""
    trace_id = record(store, traces, query="wage schedule regular",
                      scope=Scope.of(pay_system="FWS"))
    got = counterfactual_replay(traces, trace_id, make(store), system_time=NOW)

    assert not got.changed
    assert "531" in got.replayed.excluded_parts
    assert all(not v.startswith("531.") for v in got.replayed.final)


def test_a_belief_time_change_is_replayable_because_the_corpus_is_bitemporal(
        store: Store, traces: TraceStore):
    """The half of the historical state that *is* reconstructible. A corrected parse closes
    system time on the old row; pinning the trace's own system time replays the request
    against what was believed then, while the default asks what a user would get now."""
    trace_id = record(store, traces)
    stored = traces.load(trace_id)
    assert "630.306#a@2017-01-01" in stored.final

    store.retract("630.306#a@2017-01-01", system_to=T1)
    store.add([Chunk(chunk_id="630.306#a-1", section_id="630.306", title=5, part="630",
                     anchor="a-1", heading="Time limit for use of restored annual leave",
                     text="annual leave restored must be scheduled within two years",
                     valid_from="2017-01-01")], system_from=T1)

    then = counterfactual_replay(traces, trace_id, make(store), system_time=T0)
    now = counterfactual_replay(traces, trace_id, make(store), system_time=NOW)
    assert not then.changed
    assert "630.306#a-1@2017-01-01" in now.entered_final
    assert "630.306#a@2017-01-01" in now.left_final


def test_the_sweep_replays_every_stored_request_including_the_quiet_ones(store: Store,
                                                                        traces: TraceStore):
    """"3 of 200 moved" and "3 moved" are different findings, so unchanged requests are
    replayed and returned rather than filtered out."""
    record(store, traces, query="annual leave restored")
    record(store, traces, query="wage schedule regular")

    diffs = counterfactual_sweep(traces, make(store, final_k=1), system_time=NOW)
    assert len(diffs) == 2
    assert all(d.first_divergence == "final" for d in diffs)


def test_the_diff_serialises_to_something_an_api_can_return(store: Store,
                                                            traces: TraceStore):
    import json

    trace_id = record(store, traces)
    got = counterfactual_replay(traces, trace_id, make(store, final_k=1), system_time=NOW)
    payload = json.loads(json.dumps(got.to_dict()))

    assert payload["first_divergence"] == "final"
    assert payload["final"]["answer_set_changed"] is True
    assert [s["stage"] for s in payload["stages"]] == \
        ["lexical", "dense", "fused", "reranked", "final"]
