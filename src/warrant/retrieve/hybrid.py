"""The retrieval pipeline, and the per-stage record the failure budget is computed from.

Every stage writes what it saw and what it passed on into a `Trace`. That is not
instrumentation bolted on afterwards: an autopsy that has to re-run retrieval to find out
where evidence was lost can only ever be an approximation of what actually happened. The
trace is the primary artifact and the ranked answer is a view of it.

Stage order, and why:

  1. predicates      as-of and applicability, pushed into the query, not applied after
  2. lexical         BM25 over FTS5
  3. dense           cosine over the embedding matrix, restricted before scoring
  4. fusion          reciprocal rank fusion; rank-based, so the two score scales never meet
  5. rerank          cross-encoder over the fused head, optional

RRF rather than a weighted score blend: BM25 scores and cosine similarities are not
commensurable, and normalising them introduces a tuning parameter that has to be justified
per corpus. RRF needs only the ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..index.store import Store
from .scope import GOVERNMENT_WIDE, Scope

if TYPE_CHECKING:  # pragma: no cover
    from .dense import DenseIndex

RRF_K = 60


def fts_query(text: str) -> str:
    """Escape a natural-language query for FTS5.

    FTS5 reads bare punctuation as syntax, so a heading containing a colon or a hyphen is a
    parse error rather than a query. Quoting each token keeps the query literal, which is
    what a lexical baseline should be.
    """
    tokens = [t for t in ("".join(c if c.isalnum() else " " for c in text)).split() if t]
    return " OR ".join(f'"{t}"' for t in tokens) or '""'


@dataclass
class Trace:
    """What each stage saw and passed on. Keyed by version id throughout."""

    query: str
    as_of: str
    scope: str
    excluded_parts: list[str] = field(default_factory=list)
    admitted: int = 0                       # rows surviving the predicates
    lexical: list[str] = field(default_factory=list)
    dense: list[str] = field(default_factory=list)
    fused: list[str] = field(default_factory=list)
    reranked: list[str] = field(default_factory=list)
    final: list[str] = field(default_factory=list)

    def stage(self, name: str) -> list[str]:
        return {"lexical": self.lexical, "dense": self.dense, "fused": self.fused,
                "reranked": self.reranked, "final": self.final}[name]

    @property
    def stages_run(self) -> list[str]:
        run = ["lexical"]
        if self.dense:
            run.append("dense")
        run.append("fused")
        if self.reranked:
            run.append("reranked")
        run.append("final")
        return run


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass
class Retriever:
    store: Store
    dense_index: DenseIndex | None = None
    reranker: object | None = None          # a sentence_transformers CrossEncoder
    candidates_lexical: int = 100
    candidates_dense: int = 100
    rerank_top_k: int = 30
    final_k: int = 8
    temporal: bool = True
    parts_universe: list[str] = field(default_factory=list)
    _query_cache: dict[str, object] = field(default_factory=dict, repr=False)

    def retrieve(self, query: str, *, as_of: str, scope: Scope = GOVERNMENT_WIDE,
                 system_time: str | None = None) -> Trace:
        excluded = scope.excluded_parts(self.parts_universe) if self.parts_universe else []
        trace = Trace(query=query, as_of=as_of, scope=scope.describe(),
                      excluded_parts=excluded)

        allowed = self.store.candidate_ids(valid_date=as_of, system_time=system_time,
                                           temporal=self.temporal, exclude_parts=excluded)
        trace.admitted = len(allowed)

        rows = self.store.search(fts_query(query), valid_date=as_of, system_time=system_time,
                                 limit=self.candidates_lexical, temporal=self.temporal,
                                 exclude_parts=excluded)
        trace.lexical = [r["version_id"] for r in rows]

        rankings = [trace.lexical]
        if self.dense_index is not None:
            trace.dense = self._dense(query, allowed)
            rankings.append(trace.dense)

        trace.fused = reciprocal_rank_fusion(rankings)
        head = trace.fused[: self.rerank_top_k]

        if self.reranker is not None and head:
            trace.reranked = self._rerank(query, head)
            trace.final = trace.reranked[: self.final_k]
        else:
            trace.final = head[: self.final_k]
        return trace

    # -- stages ------------------------------------------------------------------

    def _dense(self, query: str, allowed: set[int]) -> list[str]:
        from .dense import encode_query

        assert self.dense_index is not None
        vec = self._query_cache.get(query)
        if vec is None:
            vec = encode_query(query, model_name=self.dense_index.model)
            self._query_cache[query] = vec
        hits = self.dense_index.search(vec, allowed=allowed, limit=self.candidates_dense)
        rows = self.store.rows_by_id([i for i, _ in hits])
        return [rows[i]["version_id"] for i, _ in hits if i in rows]

    def _rerank(self, query: str, keys: list[str]) -> list[str]:
        rows = {r["version_id"]: r for r in self.store.db.execute(
            f"SELECT version_id, heading, text FROM chunk WHERE version_id IN "
            f"({','.join('?' * len(keys))})", keys)}
        pairs, present = [], []
        for k in keys:
            r = rows.get(k)
            if r is None:
                continue
            present.append(k)
            pairs.append((query, f"{r['heading']}. {r['text']}" if r["heading"]
                          else r["text"]))
        if not pairs:
            return []
        scores = self.reranker.predict(pairs)  # type: ignore[union-attr]
        return [k for k, _ in sorted(zip(present, scores, strict=True),
                                     key=lambda kv: -kv[1])]
