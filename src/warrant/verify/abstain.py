"""Signals that say whether the retrieved set can support an answer at all.

eval-004 measured the gap this module exists to close: on 29 held-out human questions the
generator answered 29 times, and 6 of those answers were written from a context that did not
contain sufficient evidence. A 1.5% hallucination rate is only worth quoting if the system
also declines, and on the one axis where declining matters it declined zero times.

Everything here is computed from a `Trace` and the rows it already retrieved. **No second
model call.** That is a hard constraint rather than an optimisation: generation on this
machine runs at 21.3 tokens/s and a full answer takes ~20 seconds, so a verifier that costs
another forward pass would double the latency of the stage it is meant to protect, and would
be the first thing shed under load -- exactly when the guard is most needed.

The eight signals, and what each is supposed to notice:

``top_score``       how much rank-fusion mass the winner actually accumulated. Low means no
                    ranker was confident, not merely that they disagreed.
``margin_1_2``      whether there is a winner at all, or a tie at the top.
``margin_1_5``      the same question over the whole head. A query whose top five are
                    separated by nothing is a query the corpus answers five ways or not at all.
``entropy``         the shape of the head as a distribution, normalised to [0, 1]. Flat is bad.
``rank_agreement``  how many of the top-k BM25 and dense both found. This is the strongest
                    free signal available: two rankers built on unrelated evidence -- term
                    statistics and embedding geometry -- converging on the same paragraphs is
                    hard to arrange by accident, and disagreement is the ordinary signature of
                    a query that matches nothing in particular.
``term_coverage``   the share of the query's content words that appear anywhere in the top-k
                    text. Cheap, and it is the one signal that looks at what was *asked*
                    rather than at how the ranking came out.
``log_admitted``    rows surviving the as-of and applicability predicates. A near-empty
                    admitted set means the question is outside the corpus as scoped and dated,
                    which no amount of ranking can repair. Logged because the count spans
                    zero to ~13k and only its order of magnitude carries information.
``guidance_top``    whether the best hit is notice/guidance/archival rather than statute or
                    regulation. On this corpus every row is eCFR regulation, so the feature is
                    a constant zero and is kept for the multi-source path rather than because
                    it measures anything today; the results doc says so rather than letting a
                    reader assume all eight features earned their place.

Scores are read off the **fused** stage, not the final one. When a cross-encoder runs, the
final stage carries its logits, and a margin between two logits is a different quantity from a
margin between two RRF weights -- fitting a combiner across a mixture of the two would make
the calibration depend on whether the reranker happened to be enabled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..index.store import Store
from ..retrieve.dense import retrieval_text
from ..retrieve.hybrid import Candidate, Trace
from .align import _content_words

#: Depth every set-valued signal is measured at. It is ``retrieve.final_k`` from the served
#: config, and it has to be the same number at fit time and at serve time or the calibration
#: describes a different system than the one it is guarding.
TOP_K = 16

#: ``chunk.authority`` is ordered: 1 statute, 2 regulation, 3 notice, 4 guidance, 5 archival.
#: At and above this value the top hit is explanatory material rather than the law itself.
FIRST_GUIDANCE_AUTHORITY = 3

#: Feature order. Fixed and public, because a fitted combiner is a vector of coefficients and
#: nothing else -- if this order drifts, an old model silently reads the wrong columns.
FEATURES: tuple[str, ...] = (
    "top_score", "margin_1_2", "margin_1_5", "entropy",
    "rank_agreement", "term_coverage", "log_admitted", "guidance_top",
)


@dataclass(frozen=True, slots=True)
class Signals:
    """One query's confidence features, in ``FEATURES`` order."""

    top_score: float = 0.0
    margin_1_2: float = 0.0
    margin_1_5: float = 0.0
    entropy: float = 0.0
    rank_agreement: float = 0.0
    term_coverage: float = 0.0
    log_admitted: float = 0.0
    guidance_top: float = 0.0

    @property
    def vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in FEATURES)

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURES}


def score_entropy(scores: Sequence[float]) -> float:
    """Normalised Shannon entropy of a score head, in [0, 1]. 1.0 is a flat head.

    Normalised by ``log(n)`` so that a head of 8 and a head of 3 are comparable; without it
    the feature would mostly encode how many candidates came back, which ``rank_agreement``
    and ``log_admitted`` already say more directly.

    A head of zero or one candidate returns 0.0. That reads as "maximally peaked", which is
    the wrong intuition for an empty head and the right arithmetic -- there is no distribution
    to be uncertain about. Nothing rests on it: an empty retrieval is caught by
    ``term_coverage`` and ``log_admitted``, both of which go to zero with it.
    """
    positive = [s for s in scores if s > 0]
    total = sum(positive)
    if len(positive) < 2 or total <= 0:
        return 0.0
    h = -sum((s / total) * math.log(s / total) for s in positive)
    return h / math.log(len(positive))


def _head_scores(cands: Sequence[Candidate], top_k: int) -> list[float]:
    return [c.score for c in cands[:top_k] if c.score is not None]


def signals(trace: Trace, *, texts: Mapping[str, str] | None = None,
            authority: Mapping[str, int] | None = None, top_k: int = TOP_K) -> Signals:
    """Confidence features for one trace. Never raises; a dead query yields near-zero signals.

    ``texts`` and ``authority`` are the retrieval text and authority level of the returned
    rows, keyed by version id. They are passed in rather than fetched so that this function
    stays a pure transform over a trace -- the serving path has the rows in hand already, and
    a replayed trace has no store to fetch them from.
    """
    texts = texts or {}
    authority = authority or {}

    fused = trace.candidates("fused") or trace.candidates("final")
    scores = _head_scores(fused, top_k)
    top = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else top
    # As deep as the list actually goes. A head shorter than five is itself thin evidence, and
    # reporting a margin against a rank that does not exist would invent a gap.
    fifth = scores[min(4, len(scores) - 1)] if scores else top

    lexical = trace.lexical[:top_k]
    dense = trace.dense[:top_k]
    # Denominator is top_k, not the shorter list: a query that returned three candidates
    # cannot reach full agreement, which is the intended reading rather than a rounding
    # artifact. In a lexical-only configuration there is no second list and this is a
    # constant zero -- see ``Standardizer`` in calibrate.py, which refuses to divide by it.
    agreement = len(set(lexical) & set(dense)) / top_k if dense else 0.0

    final = trace.final[:top_k] or [c.version_id for c in fused[:top_k]]
    wanted = _content_words(trace.query)
    seen: set[str] = set()
    for version_id in final:
        seen |= _content_words(texts.get(version_id, ""))
    coverage = len(wanted & seen) / len(wanted) if wanted else 0.0

    guidance = 0.0
    if final and authority.get(final[0], 0) >= FIRST_GUIDANCE_AUTHORITY:
        guidance = 1.0

    return Signals(
        top_score=top,
        margin_1_2=top - second,
        margin_1_5=top - fifth,
        entropy=score_entropy(scores),
        rank_agreement=agreement,
        term_coverage=coverage,
        log_admitted=math.log1p(max(0, trace.admitted)),
        guidance_top=guidance,
    )


def signals_from_store(store: Store, trace: Trace, *, top_k: int = TOP_K) -> Signals:
    """``signals`` with the row lookup done for you: one query over the returned ids."""
    fused = trace.candidates("fused") or trace.candidates("final")
    keys = trace.final[:top_k] or [c.version_id for c in fused[:top_k]]
    if not keys:
        return signals(trace, top_k=top_k)
    rows = store.db.execute(
        "SELECT version_id, authority, heading, context, text FROM chunk "
        f"WHERE version_id IN ({','.join('?' * len(keys))})", keys).fetchall()
    return signals(
        trace,
        texts={r["version_id"]: retrieval_text(r) for r in rows},
        authority={r["version_id"]: r["authority"] for r in rows},
        top_k=top_k,
    )
