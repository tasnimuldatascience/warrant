"""Follow-up turns: an exchange pinned to one trace, answered from its own evidence.

Warrant is single-shot by design -- question, as-of, scope in; evidence and cited claims out
-- and that is not an omission to be completed by adding a chat window. A conversation that
carries state across an as-of change is a wrong-version bug waiting to happen: ask "by when
must restored leave be scheduled?" as of 2019, then "and what's the exception?", and a chat
layer either carries 2019 forward silently (the user never learns which law a later turn
answered from) or re-resolves to today (two turns now answer from different regulations
without saying so). The as-of predicate is worth +96.1 points of wrong-version rate
(README, results/eval-004) precisely because it is never allowed to drift underneath a
question; a conversational layer that reinterprets the date per turn throws that away by
default.

So a follow-up here answers **from the evidence a prior turn already retrieved**, never by
running retrieval again on its own initiative. The unit of state is an `Exchange`: the
question, the as-of it resolved to, the scope, and the pinned set of evidence version ids --
reconstructed from a recorded `Trace` (`observe.trace_store`), not held in a second,
in-memory session store. A trace is already the durable, replayable record of one request;
inventing a parallel representation of "what this exchange saw" would just be two sources of
truth that can drift. The trace_id `/api/ask` already hands back as `AskResponse.trace_id`
*is* the exchange id -- nothing new to remember, nothing that expires when the process
restarts.

Three kinds of follow-up, and this module produces exactly two of them; the third is what it
deliberately has no parameter for:

    answered       the pinned evidence covers it -- generate and cite from `exchange.evidence`
    insufficient   the generator abstains -- report it, and name what might close the gap
    (new exchange) changing the as-of or scope is not a turn at all

The third is enforced by omission, not by a check: nothing in this module, and nothing a
follow-up endpoint should accept, takes an `as_of` or a scope. `Exchange.as_of` comes from
the trace a follow-up names and from nowhere else, so there is no parameter through which a
later turn could smuggle in a different date. Asking about a different date is a new
question, and a new question gets a new `/api/ask`.

For the "insufficient" case, `verify.xref.dangling_references` already does the relevant
measurement: 43.5% of in-force chunks make at least one reference, and 77.3% of *sets*
retrieved for the human benchmark have at least one reference their own set does not satisfy
(results/eval-013). That is exactly the affordance a follow-up wants -- "§630.309 is
referenced but was not retrieved -- fetch it?" -- so `widen` resolves one named chunk id to
its in-force version at `exchange.as_of` and folds it into the pinned set. It is a single
indexed lookup by address, never a ranked search: that is the whole difference between
"widen" and silently re-retrieving, and it is why widening can never smuggle in a chunk from
a different as-of than the one already pinned.

**What this module does not do.** No memory across exchanges: an `Exchange` lives exactly as
long as its trace, and nothing here writes to a session, a cookie, or a user profile. No
personalisation: `widen` offers the same targets to anyone looking at the same evidence. Both
absences are load-bearing, not missing features -- see `results/eval-022-followups.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..generate.answer import Answer, excerpts_for
from ..index.store import Store
from ..index.store import now as utc_now
from ..observe.trace_store import TraceStore
from ..retrieve.hybrid import Trace
from ..verify.xref import DanglingReference, corpus_chunk_ids, dangling_references

#: A widen suggestion list this long is not a suggestion any more. Bounds a pathological
#: chunk -- a definitions section cross-referencing half the part -- from turning one
#: abstention into a wall of buttons nobody will read.
MAX_WIDEN_SUGGESTIONS = 8


class NoSuchExchange(KeyError):
    """The trace a follow-up names was never recorded, or has since been deleted.

    Traces are telemetry (`observe.trace_store`'s module docstring) and deletable by design,
    so this is a normal outcome, not a corruption -- a follow-up to a turn old enough to have
    been swept should read the same as a follow-up to a trace_id that was never real: the
    exchange it would answer within no longer exists.
    """


class PinnedEvidenceViolation(RuntimeError):
    """A follow-up's answer cited a version id outside its exchange's pinned evidence.

    Structurally unreachable through the path this module offers: `excerpts_for_exchange`
    hands the generator nothing but `exchange.evidence`, and `generate.answer.parse_response`
    can only resolve an excerpt *number* back to a version id that was actually offered at
    that number -- there is no id for it to hallucinate that would not already be one of
    those. This is the assertion that makes that guarantee checkable rather than merely
    argued, the same way `serve.guard.validate_answer` checks the live path's version of the
    same invariant (retrieved-set membership) after generation, not instead of before it.
    """


@dataclass(frozen=True)
class Exchange:
    """One pinned turn: everything a follow-up is allowed to know.

    ``evidence`` is version ids, in the order the parent trace ranked them -- the same list
    `AskResponse.evidence` and the `evidence` SSE frame already showed the user, so a
    follow-up's context is never a fact only the server can see.
    """

    trace_id: str
    question: str
    as_of: str
    scope: str                          # human description -- Trace.scope / StoredTrace.scope
    scope_facets: dict[str, str]
    evidence: tuple[str, ...]           # version ids, pinned
    config_hash: str = ""

    def widened(self, trace_id: str, evidence: tuple[str, ...]) -> Exchange:
        """The exchange after a `widen` call lands and is recorded under its own trace id.

        A new trace id, deliberately: widening changes what the exchange can answer from, and
        that change is itself a recorded, replayable request (`trace_context`) rather than a
        silent mutation of the one that came before it. A follow-up after this point pins to
        the widened set by naming *this* trace id, not the one it started from.
        """
        return Exchange(trace_id=trace_id, question=self.question, as_of=self.as_of,
                        scope=self.scope, scope_facets=self.scope_facets, evidence=evidence,
                        config_hash=self.config_hash)


def load_exchange(traces: TraceStore, trace_id: str) -> Exchange:
    """The pinned state a follow-up answers within, read back from a recorded trace.

    Raises `NoSuchExchange` rather than `TraceStore.load`'s own `KeyError`: a mistyped or
    already-swept trace_id is a fact about the *follow-up request*, and a caller should not
    have to import `observe.trace_store` to recognise its exception.
    """
    try:
        stored = traces.load(trace_id)
    except KeyError as exc:
        raise NoSuchExchange(trace_id) from exc
    return Exchange(trace_id=trace_id, question=stored.query, as_of=stored.as_of,
                    scope=stored.scope, scope_facets=dict(stored.scope_facets),
                    evidence=tuple(stored.final), config_hash=stored.config_hash)


def trace_context(exchange: Exchange, kind: str, **extra: str) -> dict[str, str]:
    """The `context` payload one follow-up trace should be recorded with.

    Kept to a few short strings rather than the whole `Exchange`: `TraceStore`'s `context`
    column is schemaless JSON, and that is exactly the place a private implementation detail
    quietly becomes part of the permanent record if nobody keeps the shape narrow.
    """
    return {"parent_trace_id": exchange.trace_id, "kind": kind, **extra}


def exchange_trace(exchange: Exchange, question: str) -> Trace:
    """A `Trace` whose only populated stage is `final`, pinned to `exchange.evidence`.

    Not a placeholder shape -- this is the real `retrieve.hybrid.Trace`, the type
    `TraceStore.record` and every replay tool already know how to persist and read back. A
    follow-up recorded this way is inspectable the same way a retrieval is: `stages_run`
    correctly reports empty lexical/dense/reranked stages, because none of them ran, and
    that absence is the honest description of what happened. Public so a caller recording a
    `widen` -- which changes the pinned set without generating anything -- can build the same
    shape of trace for it, rather than either module inventing its own.
    """
    return Trace(query=question, as_of=exchange.as_of, scope=exchange.scope,
                scope_facets=dict(exchange.scope_facets), final=exchange.evidence,
                admitted=len(exchange.evidence), config_hash=exchange.config_hash)


def excerpts_for_exchange(store: Store, exchange: Exchange, question: str, *,
                          limit: int | None = None) -> list[tuple[str, str, str]]:
    """The excerpts a follow-up may see: exactly `exchange.evidence`, looked up once.

    Delegates to `generate.answer.excerpts_for`, which already reads nothing but
    `trace.final` -- building the real `Trace` type here instead of a shortcut that fetches
    the same rows means a future change to that function's contract shows up here too,
    instead of two implementations silently drifting apart.

    ``limit`` defaults to the *whole* pinned set, not `MAX_CONTEXT_CHUNKS`. That default was
    live-tested and wrong: a `widen` call appends to the tail of `exchange.evidence`, and an
    exchange already at the 16-chunk cap silently lost the just-widened chunk the moment this
    function re-truncated to the same 16 before the caller ever saw it -- the one silent
    failure this whole module exists to rule out, reintroduced by an off-the-shelf default.
    The generation-context cap still applies; it belongs to `serve.guard.bound_excerpts`,
    which the caller runs on this function's output next and which reports what it drops
    (`Prompt.dropped`) instead of dropping it before anyone could know.
    """
    return excerpts_for(store, exchange_trace(exchange, question),
                        limit=limit if limit is not None else len(exchange.evidence))


@dataclass(frozen=True)
class FollowupAnswer:
    """One follow-up turn, answered strictly from `exchange`'s pinned evidence."""

    exchange: Exchange
    question: str
    kind: str                            # "answered" | "insufficient"
    answer: Answer
    #: Populated only when `kind == "insufficient"`. Every reference the pinned chunks make
    #: that the pinned set itself does not satisfy -- not filtered to the question, because
    #: this module never re-retrieves to find out which of them the question actually needed.
    dangling: list[DanglingReference]
    trace: Trace


def finish_followup(exchange: Exchange, question: str, answer: Answer, *,
                    store: Store) -> FollowupAnswer:
    """Package a generated `Answer` as one follow-up turn.

    Takes an already-generated `Answer` rather than a generator: the model call belongs to
    whichever caller already owns the generation slot and its deadline
    (`serve.api._generate_answer`), and duplicating that machinery here would be exactly the
    second pipeline the design brief rules out. This function's job starts once the answer
    exists -- verifying it never stepped outside the pinned set, and naming what is missing
    when it could not answer at all.
    """
    _check_pinned(answer, exchange.evidence)
    dangling: list[DanglingReference] = []
    if answer.abstained:
        nameable = corpus_chunk_ids(store, as_of=exchange.as_of)
        dangling = dangling_references(dict(answer.cited), in_corpus=nameable)
    kind = "insufficient" if answer.abstained else "answered"
    return FollowupAnswer(exchange=exchange, question=question, kind=kind, answer=answer,
                          dangling=dangling, trace=exchange_trace(exchange, question))


def _check_pinned(answer: Answer, evidence: Sequence[str]) -> None:
    pinned = set(evidence)
    for i, claim in enumerate(answer.claims, start=1):
        for vid in claim.evidence:
            if vid not in pinned:
                raise PinnedEvidenceViolation(
                    f"claim {i} cites {vid!r}, outside the {len(pinned)}-chunk pinned "
                    "evidence set for this exchange")


def widenable(dangling: Sequence[DanglingReference], *,
             limit: int = MAX_WIDEN_SUGGESTIONS) -> list[dict[str, str]]:
    """The subset of `dangling` this module can actually act on.

    Only ``status == "missing"``: the corpus holds that chunk and `widen` can fetch it by
    address. ``outside`` (a U.S.C. or non-chapter-I title this corpus does not carry) and
    ``unscoped`` ("this subpart", naming no single paragraph) have no fetchable target and
    are worth reporting, never worth a button that cannot do anything when pressed.

    Deduplicated by target: several cited chunks referencing the same missing section should
    read as one offer, not one per citing paragraph.
    """
    seen: dict[str, str] = {}
    for d in dangling:
        if d.status != "missing" or d.target in seen:
            continue
        seen[d.target] = d.reference.text if d.reference is not None else d.target
    return [{"chunk_id": chunk_id, "text": text}
            for chunk_id, text in list(seen.items())[:limit]]


def widen(store: Store, exchange: Exchange, chunk_id: str, *,
         system_time: str | None = None) -> tuple[str, tuple[str, ...]]:
    """Resolve `chunk_id` to its version in force at `exchange.as_of`, and fold it in.

    Returns ``(version_id, evidence)`` -- the new pinned set, unchanged if `chunk_id` was
    already in it. This is a single lookup by exact address, never a ranked query: a `widen`
    that searched would be a second retrieval path with its own ranking to reason about, and
    the one property this whole module trades on is that nothing here ranks anything.

    Bound to `exchange.as_of`, never to today -- widening the evidence must not also widen
    which version of the law the exchange is about. Raises `KeyError` when nothing by that
    address was in force on that date: the target of an ``outside`` dangling reference, or a
    section renumbered since.
    """
    row = _lookup_chunk(store, chunk_id, as_of=exchange.as_of, system_time=system_time)
    if row is None:
        raise KeyError(chunk_id)
    version_id = row["version_id"]
    if version_id in exchange.evidence:
        return version_id, exchange.evidence
    return version_id, (*exchange.evidence, version_id)


def _lookup_chunk(store: Store, chunk_id: str, *, as_of: str,
                  system_time: str | None) -> Any:
    sys_t = system_time or utc_now()
    return store.db.execute(
        "SELECT * FROM chunk WHERE chunk_id = ? AND valid_from <= ? "
        "AND (valid_to IS NULL OR valid_to > ?) "
        "AND system_from <= ? AND (system_to IS NULL OR system_to > ?)",
        (chunk_id, as_of, as_of, sys_t, sys_t)).fetchone()
