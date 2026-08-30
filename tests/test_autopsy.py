"""Stage attribution.

The tests that earn their keep are the ones that pin down *which* stage gets blamed when
more than one could be, because that is where an attribution instrument quietly becomes
decorative. Blaming the reranker whenever a reranker happened to run put 124 failures on it
in the real corpus; removing the reranker entirely moved the bucket by 0.1 points.
"""

from __future__ import annotations

import pytest

from warrant.autopsy.localize import Budget, observational
from warrant.eval.bench import BenchItem
from warrant.index.store import Store
from warrant.retrieve.hybrid import Trace

ALL = {"e@1", "d@1", "x@1", "y@1", "z@1"}


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.db.execute(
            "INSERT INTO chunk (version_id, chunk_id, section_id, title, part, text, "
            "content_hash, valid_from, system_from, source_snapshot, config_hash) "
            "VALUES ('e@1','e','s',5,'630','t','h','2017-01-01','2020-01-01','x','c')")
        s.db.commit()
        yield s


def item(evidence: str = "e@1") -> BenchItem:
    return BenchItem(id="i", bucket="temporal", query="q", as_of="2021-01-01",
                     section_id="s", part="630", heading="h",
                     acceptable_evidence=[[evidence]])


def trace(**kw) -> Trace:
    base = {"query": "q", "as_of": "2021-01-01", "scope": "government-wide"}
    return Trace(**base, **kw)


def localize(it: BenchItem, tr: Trace, store: Store, *, temporal=ALL, scope=ALL, top_k=3):
    return observational(it, tr, store, admitted_temporal=temporal,
                         admitted_scope=scope, rerank_top_k=top_k)[0]


def test_evidence_absent_from_the_corpus_is_ingestion(store: Store):
    assert localize(item("never-ingested@1"), trace(), store) == "ingestion"


def test_evidence_excluded_by_scope_is_applicability(store: Store):
    assert localize(item(), trace(excluded_parts=["630"]), store, scope=set()) == \
        "applicability"


def test_applicability_is_checked_before_temporal(store: Store):
    """A scope exclusion removes the part outright. Testing temporal first would report
    every scope error as a dating error and send the reader to the wrong subsystem."""
    assert localize(item(), trace(excluded_parts=["630"]), store,
                    scope=set(), temporal=set()) == "applicability"


def test_evidence_not_in_force_is_temporal(store: Store):
    assert localize(item(), trace(), store, temporal=set()) == "temporal"


def test_evidence_never_retrieved_is_retrieval(store: Store):
    tr = trace(lexical=["x@1"], dense=["y@1"], fused=["x@1", "y@1"], final=["x@1"])
    assert localize(item(), tr, store) == "retrieval"


def test_evidence_retrieved_but_cut_before_the_head_is_fusion(store: Store):
    tr = trace(lexical=["e@1"], fused=["x@1", "y@1", "z@1", "e@1"],
               final=["x@1", "y@1", "z@1"])
    assert localize(item(), tr, store, top_k=3) == "fusion"


def test_evidence_in_the_head_but_cut_by_final_k_is_truncation(store: Store):
    """No reranker ran, so the loss is the final cut being narrower than the head."""
    tr = trace(lexical=["e@1"], fused=["x@1", "y@1", "e@1"], final=["x@1"])
    assert localize(item(), tr, store, top_k=3) == "truncation"


def test_reranker_demoting_evidence_is_rerank(store: Store):
    """Evidence was inside the fused top-k and the reranker moved it out. Only this counts
    against the reranker."""
    tr = trace(lexical=["e@1"], fused=["e@1", "x@1"], reranked=["x@1", "e@1"],
               final=["x@1"])
    assert localize(item(), tr, store, top_k=3) == "rerank"


def test_reranker_is_not_blamed_for_evidence_that_was_never_in_the_top_k(store: Store):
    """The bias this module exists to avoid. Evidence sat below final_k in the fused order,
    so plain truncation would have lost it too; the reranker merely also ran."""
    tr = trace(lexical=["e@1"], fused=["x@1", "y@1", "e@1"],
               reranked=["x@1", "y@1", "e@1"], final=["x@1"])
    assert localize(item(), tr, store, top_k=3) == "truncation"


def test_successful_retrieval_blames_nothing(store: Store):
    tr = trace(lexical=["e@1"], fused=["e@1"], final=["e@1"])
    assert localize(item(), tr, store) == "none"


def test_budget_shares_are_of_failures_not_of_items():
    b = Budget(n=100, failures=20)
    b.observational.update({"retrieval": 15, "fusion": 5})
    rows = dict((s, share) for s, _, share in b.rows())
    assert rows["retrieval"] == "75.0%"
    assert b.success_rate == 0.8


def test_budget_reports_stages_in_pipeline_order():
    b = Budget(n=10, failures=4)
    b.observational.update({"truncation": 1, "retrieval": 2, "ingestion": 1})
    assert [s for s, _, _ in b.rows()] == ["ingestion", "retrieval", "truncation"]


# -- generation and grounding ----------------------------------------------------


def answer_with(claims, *, abstained=False, parse_failed=False):
    from warrant.generate.answer import Answer, Claim
    from warrant.verify.align import Span

    built = [
        Claim(text=t, evidence=list(ev),
              spans={v: (Span(0, 5, 1.0) if grounded else None) for v in ev})
        for t, ev, grounded in claims
    ]
    return Answer(question="q", as_of="2021-01-01", scope="government-wide",
                  claims=built, answer_found=not abstained, cited={},
                  parse_failed=parse_failed)


def reached_context(**kw) -> Trace:
    """A trace in which the evidence made it all the way to the context."""
    return trace(lexical=["e@1"], fused=["e@1"], reranked=["e@1"], final=["e@1"], **kw)


def test_retrieval_only_run_stops_at_truncation(store: Store):
    """Scoring retrieval without a model is the common case, and it must not invent a
    generation verdict it has no evidence for."""
    assert localize(item(), reached_context(), store) == "none"


def test_abstention_despite_good_evidence_is_generation(store: Store):
    """The right paragraph was in the prompt and the model declined to use it."""
    got = observational(item(), reached_context(), store, admitted_temporal=ALL,
                        admitted_scope=ALL, rerank_top_k=3,
                        answer=answer_with([], abstained=True))
    assert got[0] == "generation"
    assert got[1]["reason"] == "abstained"


def test_unparseable_response_is_generation_not_grounding(store: Store):
    got = observational(item(), reached_context(), store, admitted_temporal=ALL,
                        admitted_scope=ALL, rerank_top_k=3,
                        answer=answer_with([], abstained=True, parse_failed=True))
    assert got[0] == "generation"
    assert got[1]["reason"] == "parse_failed"


def test_citing_the_wrong_chunk_is_generation(store: Store):
    """Fluent prose written from something other than the sufficient evidence is still a
    generation failure."""
    got = observational(item(), reached_context(), store, admitted_temporal=ALL,
                        admitted_scope=ALL, rerank_top_k=3,
                        answer=answer_with([("something else", ["x@1"], True)]))
    assert got[0] == "generation"


def test_citing_the_right_chunk_with_no_locatable_span_is_grounding(store: Store):
    """It cited correctly and the aligner could not find support inside the text. That is
    the distinction grounding exists to draw -- a right citation is not the same as a
    supported claim."""
    got = observational(item(), reached_context(), store, admitted_temporal=ALL,
                        admitted_scope=ALL, rerank_top_k=3,
                        answer=answer_with([("unsupported", ["e@1"], False)]))
    assert got[0] == "grounding"


def test_correct_and_grounded_answer_blames_nothing(store: Store):
    got = observational(item(), reached_context(), store, admitted_temporal=ALL,
                        admitted_scope=ALL, rerank_top_k=3,
                        answer=answer_with([("supported", ["e@1"], True)]))
    assert got[0] == "none"


def test_retrieval_failure_outranks_a_generation_failure(store: Store):
    """First-loss ordering: if the evidence never reached the context, the model was never
    given a chance and blaming it would be the bias this module exists to avoid."""
    tr = trace(lexical=["x@1"], fused=["x@1"], final=["x@1"])
    got = observational(item(), tr, store, admitted_temporal=ALL, admitted_scope=ALL,
                        rerank_top_k=3, answer=answer_with([], abstained=True))
    assert got[0] == "retrieval"


def test_ladder_covers_every_stage_the_pipeline_has():
    """A stage that exists in the pipeline and not in the ladder makes the instrument blind
    to it. Generation and grounding shipped before this list caught up once already."""
    from warrant.autopsy.localize import LADDER

    assert LADDER[-2:] == ["generation", "grounding"]
    assert "truncation" in LADDER
