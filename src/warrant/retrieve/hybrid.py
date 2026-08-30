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

Each stage records the score it ranked by, not only the order it produced. Every one of those
numbers used to be computed and dropped on the next line -- BM25 selected in SQL, RRF weights
summed, cross-encoder logits predicted -- which left "the reranker demoted it" and "the
reranker barely preferred anything" indistinguishable after the fact, and left replay
(ARCHITECTURE.md section 8) with nothing to compare but two orderings.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..index.store import Store
from .scope import GOVERNMENT_WIDE, Scope

if TYPE_CHECKING:  # pragma: no cover
    from .dense import DenseIndex

RRF_K = 60


#: A bag-of-words OR gains nothing from a token appearing twice, and loses a great deal:
#: FTS5 merges the same postings list against itself once per repeat. Measured against the
#: 12,858-chunk corpus, a query of one token repeated 2,600 times -- 15.6 KB, comfortably
#: inside the HTTP parser's header limit -- took 29 seconds of pinned CPU against 16 ms for
#: a normal query. That is a ~1,800x amplification available to one unauthenticated GET,
#: and forty of them wedge the whole threadpool.
MAX_QUERY_TOKENS = 64


def fts_query(text: str, *, max_tokens: int = MAX_QUERY_TOKENS) -> str:
    """Escape a natural-language query for FTS5, deduplicated and capped.

    FTS5 reads bare punctuation as syntax, so a heading containing a colon or a hyphen is a
    parse error rather than a query. Quoting each token keeps the query literal, which is
    what a lexical baseline should be.

    Deduplication is a denial-of-service fix rather than a tidy-up -- see
    ``MAX_QUERY_TOKENS`` -- and it costs nothing in retrieval quality, because a repeated
    term contributes no postings that its first occurrence did not.
    """
    tokens = [t for t in ("".join(c if c.isalnum() else " " for c in text)).split() if t]
    unique = list(dict.fromkeys(tokens))[:max_tokens]
    return " OR ".join(f'"{t}"' for t in unique) or '""'


#: Stage names, in pipeline order. The order is load-bearing: replay reports the *first*
#: stage whose output moved, and a first divergence is only meaningful along a fixed order.
STAGES = ("lexical", "dense", "fused", "reranked", "final")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One row as a single stage saw it: its address, that stage's score, its place.

    Scores are recorded in each stage's own units and never rescaled to meet each other. A
    BM25 score (negative, lower is better, as SQLite's ``bm25()`` returns it), a cosine
    similarity and an RRF weight are three different quantities, and making them look
    comparable is exactly the mistake rank fusion is here to avoid.

    ``score`` is None where a stage ordered without scoring: a trace assembled by hand, or a
    stage whose ranking is inherited rather than computed.
    """

    version_id: str
    score: float | None = None
    rank: int = 0                           # 1-based, within this stage only


#: What a stage may hand to `Trace.record`. Bare ids are accepted so that a trace can still be
#: written by hand -- the autopsy tests build one that way -- without inventing fake scores.
CandidateLike = Candidate | str | tuple[str, float | None]


def _as_candidate(item: CandidateLike, rank: int) -> Candidate:
    if isinstance(item, Candidate):
        return Candidate(item.version_id, item.score, rank)
    if isinstance(item, str):
        return Candidate(item, None, rank)
    version_id, score = item
    return Candidate(version_id, None if score is None else float(score), rank)


class Trace:
    """What each stage saw and passed on, at the scores it saw them at. Keyed by version id.

    Every stage is stored as ``Candidate`` triples and read two ways. ``trace.lexical`` and
    friends return bare version ids -- what localisation and evaluation want, because they ask
    set questions ("did any sufficient evidence set survive this stage") that a score would
    only get in the way of. ``trace.candidates("lexical")`` returns the scores and ranks
    behind that same list, which is what an autopsy of *why* an order came out this way needs,
    and what replay diffs against.

    Deliberately not a dataclass. The constructor has to keep accepting bare ids per stage
    (``Trace(lexical=["a@1"])``), while the attribute of the same name has to be a derived
    view over the scored form; one generated ``__init__`` cannot be both.
    """

    def __init__(self, query: str, as_of: str, scope: str,
                 excluded_parts: list[str] | None = None, admitted: int = 0,
                 lexical: Iterable[CandidateLike] = (),
                 dense: Iterable[CandidateLike] = (),
                 fused: Iterable[CandidateLike] = (),
                 reranked: Iterable[CandidateLike] = (),
                 final: Iterable[CandidateLike] = (),
                 scope_facets: dict[str, str] | None = None,
                 system_time: str | None = None, config_hash: str = "",
                 models: dict[str, str] | None = None,
                 timings: dict[str, float] | None = None) -> None:
        self.query = query
        self.as_of = as_of
        self.scope = scope
        #: The machine form of ``scope``. ``describe()`` is written for a person and is lossy
        #: to parse back; counterfactual replay has to rebuild the exact profile or it is
        #: quietly answering a different question than the one that was asked.
        self.scope_facets = dict(scope_facets or {})
        #: Belief time this request was answered at. None means "as believed now", which is
        #: the ordinary serving path -- recorded because a replay that cannot pin system time
        #: cannot separate a corpus correction from a retrieval change.
        self.system_time = system_time
        self.excluded_parts = list(excluded_parts or [])
        self.admitted = admitted            # rows surviving the predicates
        self.config_hash = config_hash
        self.models = dict(models or {})
        #: Wall-clock milliseconds per stage. Keys are stage names plus ``predicates`` and
        #: ``total``; a stage that did not run has no key rather than a zero, because zero is
        #: a measurement and absence is not.
        self.timings = dict(timings or {})
        self._stages: dict[str, list[Candidate]] = {}
        for name, items in zip(STAGES, (lexical, dense, fused, reranked, final), strict=True):
            self.record(name, items)

    def __repr__(self) -> str:
        counts = ", ".join(f"{s}={len(self._stages[s])}" for s in STAGES if self._stages[s])
        return f"Trace(query={self.query!r}, as_of={self.as_of!r}, {counts})"

    # -- writing -----------------------------------------------------------------

    def record(self, stage: str, items: Iterable[CandidateLike]) -> None:
        """Set one stage's output, renumbering ranks to the order given.

        Rank is assigned here rather than carried in from the caller so that it can never
        disagree with the position it is stored at. The final cut is a slice of an earlier
        stage, and a slice that kept the ranks of the stage it came from would report the
        third answer as rank 14.
        """
        if stage not in STAGES:
            raise KeyError(stage)
        self._stages[stage] = [_as_candidate(item, rank)
                               for rank, item in enumerate(items, start=1)]

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        """Record how long one stage took, in milliseconds.

        ``perf_counter`` rather than process time: what a request costs is how long it stood
        still, and a retrieval stage spends most of that inside SQLite or a model rather than
        on this thread's CPU.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = (time.perf_counter() - start) * 1000.0

    # -- reading -----------------------------------------------------------------

    def candidates(self, stage: str) -> list[Candidate]:
        """One stage's output with scores and ranks."""
        if stage not in STAGES:
            raise KeyError(stage)
        return list(self._stages.get(stage, ()))

    def ids(self, stage: str) -> list[str]:
        """One stage's output as bare version ids."""
        return [c.version_id for c in self.candidates(stage)]

    def stage(self, name: str) -> list[str]:
        return self.ids(name)

    @property
    def lexical(self) -> list[str]:
        return self.ids("lexical")

    @property
    def dense(self) -> list[str]:
        return self.ids("dense")

    @property
    def fused(self) -> list[str]:
        return self.ids("fused")

    @property
    def reranked(self) -> list[str]:
        return self.ids("reranked")

    @property
    def final(self) -> list[str]:
        return self.ids("final")

    @property
    def stages_run(self) -> list[str]:
        run = ["lexical"]
        if self._stages["dense"]:
            run.append("dense")
        run.append("fused")
        if self._stages["reranked"]:
            run.append("reranked")
        run.append("final")
        return run


def fuse(rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[Candidate]:
    """Reciprocal rank fusion, keeping the weight each candidate was fused at."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Candidate(key, score, rank) for rank, (key, score) in enumerate(ordered, start=1)]


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = RRF_K) -> list[str]:
    """The fused order alone. Kept as the plain-id form the rest of the codebase reads."""
    return [c.version_id for c in fuse(rankings, k=k)]


def model_name(component: object | None) -> str:
    """Best-effort name of a loaded model, or "" when it will not say.

    A trace that cannot name what produced it cannot be replayed against anything, and a
    guessed name is worse than none: it would let a counterfactual replay report "same model,
    different answer" about two different checkpoints. So this reads names off the object and
    returns empty rather than inferring one from config. ``DenseIndex`` records its own model
    name; a sentence_transformers ``CrossEncoder`` keeps the checkpoint only on the wrapped
    HuggingFace config, which is why the last hop reaches through ``.model.config``.
    """
    if component is None:
        return ""
    for attr in ("model", "model_name_or_path", "model_name"):
        name = getattr(component, attr, None)
        if isinstance(name, str):
            return name
    inner = getattr(getattr(component, "model", None), "config", None)
    name = getattr(inner, "_name_or_path", "")
    return name if isinstance(name, str) else ""


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
    #: Stamped onto every trace this retriever produces. Supplied by whoever loaded the
    #: config rather than read here, because a Retriever built with hand-picked arguments --
    #: which is how the interventional autopsy builds its oracle -- is not running any
    #: config, and claiming a hash it does not match would misdirect every later replay.
    config_hash: str = ""
    #: Only needed for a reranker that will not name itself; see `model_name`.
    reranker_model: str = ""

    def model_names(self) -> dict[str, str]:
        """The models behind this ranking. Absent components get no key at all."""
        names = {}
        if self.dense_index is not None:
            names["dense"] = model_name(self.dense_index)
        if self.reranker is not None:
            names["rerank"] = self.reranker_model or model_name(self.reranker)
        return names

    def retrieve(self, query: str, *, as_of: str, scope: Scope = GOVERNMENT_WIDE,
                 system_time: str | None = None) -> Trace:
        excluded = scope.excluded_parts(self.parts_universe) if self.parts_universe else []
        trace = Trace(query=query, as_of=as_of, scope=scope.describe(),
                      scope_facets=dict(scope.facets), system_time=system_time,
                      excluded_parts=excluded, config_hash=self.config_hash,
                      models=self.model_names())
        started = time.perf_counter()

        with trace.timed("predicates"):
            allowed = self.store.candidate_ids(valid_date=as_of, system_time=system_time,
                                               temporal=self.temporal,
                                               exclude_parts=excluded)
        trace.admitted = len(allowed)

        with trace.timed("lexical"):
            rows = self.store.search(fts_query(query), valid_date=as_of,
                                     system_time=system_time,
                                     limit=self.candidates_lexical, temporal=self.temporal,
                                     exclude_parts=excluded)
        # bm25() is selected by the query and was being thrown away one line later. It is
        # negative and ascending-better, kept exactly as SQLite reports it.
        trace.record("lexical", [(r["version_id"], r["score"]) for r in rows])

        rankings = [trace.lexical]
        if self.dense_index is not None:
            with trace.timed("dense"):
                trace.record("dense", self._dense(query, allowed))
            rankings.append(trace.dense)

        with trace.timed("fusion"):
            trace.record("fused", fuse(rankings))
        head = trace.candidates("fused")[: self.rerank_top_k]

        if self.reranker is not None and head:
            with trace.timed("rerank"):
                trace.record("reranked", self._rerank(query, [c.version_id for c in head]))
            trace.record("final", trace.candidates("reranked")[: self.final_k])
        else:
            trace.record("final", head[: self.final_k])
        trace.timings["total"] = (time.perf_counter() - started) * 1000.0
        return trace

    # -- stages ------------------------------------------------------------------

    def _dense(self, query: str, allowed: set[int]) -> list[Candidate]:
        assert self.dense_index is not None
        # The index embeds the query with its own encoder. Caching is bounded inside
        # encode_query: an unbounded dict keyed on raw query text is a memory-exhaustion
        # vector -- ~1.8 KB per entry, 1.7 GiB per million distinct queries, for the
        # lifetime of the process.
        vec = self.dense_index.encode(query)
        hits = self.dense_index.search(vec, allowed=allowed, limit=self.candidates_dense)
        rows = self.store.rows_by_id([i for i, _ in hits])
        return [Candidate(rows[i]["version_id"], float(score))
                for i, score in hits if i in rows]

    def _rerank(self, query: str, keys: list[str]) -> list[Candidate]:
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
        return [Candidate(k, float(s)) for k, s in sorted(zip(present, scores, strict=True),
                                                          key=lambda kv: -kv[1])]
