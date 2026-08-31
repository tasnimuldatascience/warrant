"""External baselines: what a competent engineer would build in an afternoon, without this
codebase, scored on the same held-out **test** split by the same `warrant.eval.run.score`.

Every ablation in this repository so far has been internal -- the system against itself with
a stage removed. That answers "does the as-of predicate help *this* pipeline", not "is this
pipeline better than the obvious thing a skeptic would build instead". These four exist to
answer the second question:

  1. ``NaiveDense``      -- embed everything, cosine top-k. No predicate, no fusion, no
                            rerank. The "vector DB + embeddings" baseline most RAG projects
                            ship. The most important row in the report.
  2. ``bm25_only``        -- FTS5 top-k over everything believed, ignoring both predicates.
                            The "just use search" baseline. Built from the real `Retriever`
                            with `temporal=False` and an empty `parts_universe` -- the same
                            two flags `cli._paired` already uses to turn a predicate off --
                            rather than a second BM25 code path.
  3. ``DensePostFilter``  -- cosine top-k over everything, *then* discard what is not in
                            force at ``as_of``. The obvious way to bolt temporality onto a
                            vector-DB baseline, and the one the architecture argues against:
                            pushing the predicate into the query means the candidate slots a
                            post-filter would reclaim are never spent on dead law to begin
                            with. This baseline spends them and then throws the receipt away.
  4. Full Warrant         -- not defined here. It is the `Retriever` this repository already
                            ships, unmodified, used as the reference column. Building a
                            fourth code path for "the code that already exists" would be the
                            reimplementation this module exists to avoid.

All three baselines here read the same `Store` and the same `DenseIndex` the real pipeline
reads -- built by `retrieve.dense.build`, over every believed chunk including superseded
versions, so a temporal distractor is exactly as reachable to a naive dense search as it is
to the real one before the real one's predicate excludes it. No baseline gets its own corpus
or its own encoder; the only variable is which stages run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..index.store import Store
from ..retrieve.dense import DenseIndex
from ..retrieve.hybrid import Candidate, Retriever, Trace
from ..retrieve.scope import GOVERNMENT_WIDE, Scope
from .bench import BenchItem


def bm25_only(store: Store, *, final_k: int = 8, candidates_lexical: int = 100) -> Retriever:
    """FTS5 top-k, no predicates, no dense arm, no reranking.

    A plain `Retriever` rather than a hand-rolled query: with `dense_index=None` fusion
    receives one ranking, and RRF over a single list is rank-preserving (strictly decreasing
    scores, no ties to break), so the fused order is exactly the BM25 order and nothing about
    this baseline's actual behaviour depends on code this repository does not already run.
    """
    return Retriever(store=store, candidates_lexical=candidates_lexical,
                     final_k=final_k, temporal=False, parts_universe=[])


def _in_force(row, as_of: str) -> bool:
    """Same clause `Store.search` puts in SQL, applied here to an already-ranked row."""
    return row["valid_from"] <= as_of and (row["valid_to"] is None or row["valid_to"] > as_of)


@dataclass
class NaiveDense:
    """Cosine top-k over the whole dense index. No predicate of any kind, no fusion, no
    rerank -- the honest "vector DB + embeddings" baseline.

    `temporal` and `parts_universe` below are not behaviour; `warrant.eval.run.score` reads
    them off whatever it is scoring to decide whether a distractor was ever a *candidate*.
    Both describe this baseline honestly: the dense index holds every believed chunk
    including superseded versions (`retrieve.dense.build`), and search here restricts nothing
    before scoring, so every distractor is reachable exactly the way an unfiltered candidate
    set says it is.
    """

    store: Store
    dense_index: DenseIndex
    final_k: int = 8
    temporal: bool = False
    parts_universe: list[str] = field(default_factory=list)

    def retrieve(self, query: str, *, as_of: str, scope: Scope = GOVERNMENT_WIDE,
                 system_time: str | None = None) -> Trace:
        started = time.perf_counter()
        trace = Trace(query=query, as_of=as_of, scope=scope.describe(),
                      scope_facets=dict(scope.facets), models={"dense": self.dense_index.model})
        vec = self.dense_index.encode(query)
        hits = self.dense_index.search(vec, allowed=None, limit=self.final_k)
        rows = self.store.rows_by_id([i for i, _ in hits])
        candidates = [Candidate(rows[i]["version_id"], float(s))
                      for i, s in hits if i in rows]
        trace.record("dense", candidates)
        trace.record("final", candidates)
        trace.timings["total"] = (time.perf_counter() - started) * 1000.0
        return trace


@dataclass
class DensePostFilter:
    """Cosine top-``candidates_dense`` over the whole index, then drop anything not in force
    at ``as_of``, then cut to ``final_k``. See the module docstring for why this is worth
    measuring rather than dismissing: it is the *obvious* way to add temporality to a vector
    baseline, and this repository's architecture is an argument against doing it this way --
    an argument that should be measured, not asserted.

    ``temporal=False`` and ``parts_universe=[]`` describe candidate *generation*, which is
    unrestricted, matching `NaiveDense`: the filter below runs after ranking, not before it,
    so a distractor is exactly as reachable to the ranker as it is to naive dense. Whether the
    post-filter then catches it is what `distractor_rate` in the scored result measures.
    """

    store: Store
    dense_index: DenseIndex
    candidates_dense: int = 100
    final_k: int = 8
    temporal: bool = False
    parts_universe: list[str] = field(default_factory=list)

    def retrieve(self, query: str, *, as_of: str, scope: Scope = GOVERNMENT_WIDE,
                 system_time: str | None = None) -> Trace:
        started = time.perf_counter()
        trace = Trace(query=query, as_of=as_of, scope=scope.describe(),
                      scope_facets=dict(scope.facets), models={"dense": self.dense_index.model})
        vec = self.dense_index.encode(query)
        hits = self.dense_index.search(vec, allowed=None, limit=self.candidates_dense)
        rows = self.store.rows_by_id([i for i, _ in hits])
        dense = [(i, Candidate(rows[i]["version_id"], float(s)))
                for i, s in hits if i in rows]
        trace.record("dense", [c for _, c in dense])
        survivors = [c for i, c in dense if _in_force(rows[i], as_of)]
        # Recorded before the final_k cut, so a trace inspected on its own (replay, a
        # one-off query) can already answer "did the post-filter run dry" -- `shortfall_stats`
        # below reruns retrieval to tally it across a whole bucket instead.
        trace.admitted = len(survivors)
        trace.record("final", survivors[: self.final_k])
        trace.timings["total"] = (time.perf_counter() - started) * 1000.0
        return trace


@dataclass(frozen=True)
class Shortfall:
    """How often the post-filter baseline runs dry: fewer than ``final_k`` survivors after
    discarding what the unrestricted top-k retrieved but was not in force.

    This is the failure mode the report exists to surface. A predicate pushed into the query
    never does this -- it can only narrow the candidate pool the ranker draws from, never
    shrink an already-ranked list below what was asked for.
    """

    short: int
    n: int

    @property
    def rate(self) -> float:
        return self.short / self.n if self.n else 0.0


def shortfall_stats(retriever: DensePostFilter, items: list[BenchItem]) -> Shortfall:
    """Run every item through a post-filter retriever and count the ones it ran dry on."""
    short = sum(1 for item in items
               if len(retriever.retrieve(item.query, as_of=item.as_of,
                                         scope=item.scope).final) < retriever.final_k)
    return Shortfall(short=short, n=len(items))
