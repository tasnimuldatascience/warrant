"""Two ways to replay a stored request, with two different guarantees.

**Artifact replay** -- *what exactly happened on request X?* Reconstructed from the stored
trace alone: no retrieval is re-run, no query touches the corpus, and nothing here needs a
`Retriever` at all. It is therefore exact, and it stays exact after the dense index has been
rebuilt, the config has moved on, and the encoder has been swapped. What it cannot do is tell
you anything about the system as it is now.

**Counterfactual replay** -- *what would today's pipeline do with that request?* Re-runs the
stored query, scope and as-of date through the current `Retriever` and diffs the two, stage by
stage: which stage's output first moved, what entered and left the final k, whether the answer
set changed at all. This is the regression harness over real traffic, and it is what gates a
config change. It is not exact and does not pretend to be.

**What counterfactual replay does not reconstruct: a historical index.** The corpus is
bitemporal, so the *text* of a past request is genuinely replayable -- pass the trace's
``system_time`` and the store returns what it believed then. Embeddings and chunking are not:
they are recorded by config hash, not rebuilt, so a re-run uses today's vectors and today's
chunk boundaries whatever the trace says. A diff across a config-hash change therefore
attributes nothing on its own; it says the output moved and names the hashes that bracket the
move. Pretending otherwise would mean claiming to rebuild an index nobody kept, and that
honest limit is the entire reason these are two modes rather than one function with a flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..retrieve.hybrid import STAGES, Retriever, Trace
from ..retrieve.scope import Scope
from .trace_store import StoredTrace, TraceStore


@dataclass(frozen=True)
class StageDiff:
    """One stage's output then and now."""

    stage: str
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)

    @property
    def entered(self) -> list[str]:
        """Present now, absent then, in today's order."""
        was = set(self.before)
        return [v for v in self.after if v not in was]

    @property
    def left(self) -> list[str]:
        """Present then, absent now, in the stored order."""
        is_now = set(self.after)
        return [v for v in self.before if v not in is_now]

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @property
    def reordered(self) -> bool:
        """Same rows, different order. Kept separate from membership on purpose.

        A reordering and a substitution are different regressions with different causes -- a
        reranker that changed its mind versus a candidate list that no longer contains the
        row -- and a single `changed` boolean reports them as one event.
        """
        return self.changed and not self.entered and not self.left

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "changed": self.changed, "reordered": self.reordered,
                "entered": self.entered, "left": self.left,
                "before": self.before, "after": self.after}


@dataclass(frozen=True)
class ReplayDiff:
    """What today's pipeline does differently on one stored request."""

    trace_id: str
    query: str
    as_of: str
    scope: str
    config_hash_then: str
    config_hash_now: str
    stages: list[StageDiff]
    #: The trace the re-run produced, for anything that wants to go further than the diff --
    #: localising the new failure, or recording the counterfactual as a trace of its own.
    replayed: Trace | None = None

    @property
    def changed(self) -> bool:
        return any(s.changed for s in self.stages)

    @property
    def config_changed(self) -> bool:
        return self.config_hash_then != self.config_hash_now

    @property
    def first_divergence(self) -> str | None:
        """The earliest stage whose output moved, or None if nothing moved.

        Earliest, not the loudest: a lexical candidate list that shifted by one row explains
        every stage after it, and reporting the final cut as the change would send a reader
        to tune the wrong knob. This is the same first-loss discipline the failure autopsy
        uses, applied across time instead of down the ladder.
        """
        return next((s.stage for s in self.stages if s.changed), None)

    def stage(self, name: str) -> StageDiff:
        return next(s for s in self.stages if s.stage == name)

    @property
    def entered_final(self) -> list[str]:
        return self.stage("final").entered

    @property
    def left_final(self) -> list[str]:
        return self.stage("final").left

    @property
    def answer_set_changed(self) -> bool:
        """Did the evidence reaching the generator change as a *set*?

        Order inside the final k is a ranking change; membership is a change in what the
        model can possibly say. They are worth different amounts of alarm, so they are
        reported separately rather than summed into one regression count.
        """
        final = self.stage("final")
        return bool(final.entered or final.left)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "as_of": self.as_of,
            "scope": self.scope,
            "config_hash": {"then": self.config_hash_then, "now": self.config_hash_now,
                            "changed": self.config_changed},
            "changed": self.changed,
            "first_divergence": self.first_divergence,
            "final": {
                "entered": self.entered_final,
                "left": self.left_final,
                "reordered": self.stage("final").reordered,
                "answer_set_changed": self.answer_set_changed,
            },
            "stages": [s.to_dict() for s in self.stages],
        }


def artifact_replay(traces: TraceStore, trace_id: str) -> StoredTrace:
    """Reconstruct one past request exactly, from the stored trace alone.

    No retrieval runs and the corpus is never read, so this is the only mode that stays
    correct once the index has been rebuilt underneath it: what comes back is what happened,
    not an approximation re-derived from a system that has since changed. Raises ``KeyError``
    for a trace id that was never recorded.

    Use `StoredTrace.to_trace` to hand the result to anything that consumes a live trace --
    the failure autopsy in particular, which can then localise a month-old failure without
    issuing a single query.
    """
    return traces.load(trace_id)


def counterfactual_replay(traces: TraceStore, trace_id: str, retriever: Retriever, *,
                          system_time: str | None = None) -> ReplayDiff:
    """Re-run one stored request through the current pipeline and diff the two.

    ``system_time`` pins the corpus to a belief time: pass the stored trace's own to hold the
    text fixed and isolate a retrieval change, or leave it None -- the default -- to ask the
    question the regression harness actually asks, which is what a user would get today. What
    neither setting does is restore the embeddings or the chunk boundaries of the original
    run; see the module docstring.

    Raises ``ValueError`` from `Scope.of` if the stored profile names a facet or value this
    build no longer knows. That is the correct loud failure: a vocabulary change means the
    stored request cannot be re-asked as asked, and silently widening it to government-wide
    would report a scope regression as a retrieval one.
    """
    stored = traces.load(trace_id)
    scope = Scope.of(**stored.scope_facets)
    current = retriever.retrieve(stored.query, as_of=stored.as_of, scope=scope,
                                 system_time=system_time)
    return diff(stored, current)


def diff(stored: StoredTrace, current: Trace) -> ReplayDiff:
    """Compare a stored trace against a fresh one, stage by stage, in pipeline order."""
    return ReplayDiff(
        trace_id=stored.trace_id,
        query=stored.query,
        as_of=stored.as_of,
        scope=stored.scope,
        config_hash_then=stored.config_hash,
        config_hash_now=current.config_hash,
        stages=[StageDiff(stage, stored.ids(stage), current.ids(stage)) for stage in STAGES],
        replayed=current,
    )


def counterfactual_sweep(traces: TraceStore, retriever: Retriever, *, limit: int = 100,
                         system_time: str | None = None) -> list[ReplayDiff]:
    """Replay the most recent stored requests, newest first.

    The regression harness proper: a config change is gated on what it does to traffic that
    actually happened, rather than on a benchmark chosen before the change was contemplated.
    Every request is replayed, including the ones that did not change, because "3 of 200
    moved" and "3 moved" are different findings.
    """
    return [diff(stored, retriever.retrieve(stored.query, as_of=stored.as_of,
                                            scope=Scope.of(**stored.scope_facets),
                                            system_time=system_time))
            for stored in traces.recent(limit)]
