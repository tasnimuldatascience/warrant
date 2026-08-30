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
