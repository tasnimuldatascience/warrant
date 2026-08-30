"""Localizing a failure to the stage responsible.

Two passes, reported separately, because they answer different questions and only one of
them is cheap.

**Observational** — walk the trace and record the first stage at which no sufficient evidence
set survives. Runs on every failure. It answers *where the evidence visibly disappeared*, and
it is biased: attributing to the first stage that lost the evidence systematically
under-counts upstream causes. A paragraph that the chunker split badly is retrieved
"successfully" and the blame lands on whatever ran last.

**Interventional** — replace one stage with an oracle and re-run. Runs on a sample, because
each intervention costs a retrieval. It answers *which repair fixes the answer*, which is
strictly more useful and still not causal proof: oracle substitution shows a repair works,
not that the stage was the unique cause, and stages interact. This module calls it repair
attribution and claims nothing stronger.

Interventional attribution is **multi-label**, so its totals do not sum to the failure count.
A failure can implicate more than one repair. Forcing the columns to add up to N
would be tidier and less true.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..eval.bench import BenchItem
from ..index.store import Store
from ..retrieve.hybrid import Retriever, Trace

#: Observational ladder, in pipeline order. The first stage that loses every sufficient
#: evidence set is the one blamed.
LADDER = ["ingestion", "applicability", "temporal", "retrieval", "fusion", "rerank"]

#: Depth used by the "would more candidates have found it?" intervention. Large on purpose:
#: the question is whether the evidence is reachable at all, not whether it is reachable
#: cheaply.
ORACLE_DEPTH = 2000


@dataclass(frozen=True)
class Autopsy:
    item_id: str
    observational: str                      # the stage blamed, or "none" if it succeeded
    repairs: list[str] = field(default_factory=list)   # multi-label, may be empty
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.observational != "none"


def _survives(item: BenchItem, available: set[str]) -> bool:
    return any(set(s) <= available for s in item.acceptable_evidence)


def observational(item: BenchItem, trace: Trace, store: Store, *,
                  admitted_temporal: set[str], admitted_scope: set[str],
                  rerank_top_k: int) -> tuple[str, dict[str, str]]:
    """The first stage at which no sufficient evidence set survives."""
    detail: dict[str, str] = {}

    in_corpus = {
        r["version_id"] for r in store.db.execute(
            "SELECT version_id FROM chunk WHERE system_to IS NULL AND version_id IN "
            f"({','.join('?' * len(item.all_evidence))})", list(item.all_evidence))
    } if item.all_evidence else set()
    if not _survives(item, in_corpus):
        detail["missing"] = ",".join(sorted(set(item.all_evidence) - in_corpus))
        return "ingestion", detail

    # Applicability before temporal: a scope exclusion removes the part outright, so testing
    # it second would misreport every scope error as a dating error.
    if not _survives(item, admitted_scope):
        detail["excluded_parts"] = ",".join(trace.excluded_parts)
        return "applicability", detail
    if not _survives(item, admitted_temporal):
        detail["as_of"] = trace.as_of
        return "temporal", detail

    candidates = set(trace.lexical) | set(trace.dense)
    if not _survives(item, candidates):
        detail["lexical_n"] = str(len(trace.lexical))
        detail["dense_n"] = str(len(trace.dense))
        return "retrieval", detail

    if not _survives(item, set(trace.fused[:rerank_top_k])):
        detail["fused_head"] = str(rerank_top_k)
        return "fusion", detail

    if not _survives(item, set(trace.final)):
        detail["final_k"] = str(len(trace.final))
        # Distinguish the reranker demoting evidence from the final cut simply being
        # narrower than the fused head. Blaming the reranker whenever one happened to run
        # is the textbook form of the bias this module warns about: on this corpus it put
        # 124 failures on the reranker, and rerunning with the reranker removed changed the
        # bucket by 0.1 points. The evidence was never in the top k to begin with.
        k = len(trace.final)
        if trace.reranked and _survives(item, set(trace.fused[:k])):
            return "rerank", detail
        return "truncation", detail

    return "none", detail


def interventional(item: BenchItem, retriever: Retriever, *,
                   depth: int = ORACLE_DEPTH) -> list[str]:
    """Which repair makes the evidence reachable. Multi-label; may be empty.

    Two interventions are meaningful for a retrieval-only pipeline:

    ``ranking``     the evidence is reachable at oracle depth, so the scorers found it and
                    placed it too low. Note what this does *not* say: it does not say that
                    raising the operating depth would fix the answer, because the evidence
                    still has to reach the final k. Measured on this corpus, raising
                    candidates from 100 to 1000 moved the temporal bucket by well under a
                    point -- reachability and rank are different problems, and reporting this
                    label as "depth" invited exactly that confusion.
    ``unreachable`` neither scorer surfaces it even at ``ORACLE_DEPTH``. No candidate budget
                    helps; the query and the text do not meet.
    """
    wide = Retriever(
        store=retriever.store,
        dense_index=retriever.dense_index,
        reranker=None,                       # the question is reachability, not ordering
        candidates_lexical=depth,
        candidates_dense=depth,
        rerank_top_k=depth,
        final_k=depth,
        temporal=retriever.temporal,
        parts_universe=retriever.parts_universe,
    )
    trace = wide.retrieve(item.query, as_of=item.as_of, scope=item.scope)
    reachable = set(trace.lexical) | set(trace.dense)
    if _survives(item, reachable):
        return ["ranking"]
    return ["unreachable"]


@dataclass
class Budget:
    """The artifact. Counts of failures by blamed stage."""

    n: int
    failures: int
    observational: Counter[str] = field(default_factory=Counter)
    repairs: Counter[str] = field(default_factory=Counter)
    autopsies: list[Autopsy] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.n - self.failures) / self.n if self.n else 0.0

    def rows(self) -> list[tuple[str, int, str]]:
        out = []
        for stage in [*LADDER, "truncation"]:
            count = self.observational.get(stage, 0)
            if count:
                share = count / self.failures * 100 if self.failures else 0.0
                out.append((stage, count, f"{share:.1f}%"))
        return out


def run(items: list[BenchItem], retriever: Retriever, *,
        interventional_sample: int = 0) -> Budget:
    """Retrieve every item, then localize each failure.

    ``interventional_sample`` bounds the expensive pass. It is a deterministic stride over
    the failures rather than a random sample, so a published budget is reproducible without
    carrying a seed.
    """
    store = retriever.store
    budget = Budget(n=len(items), failures=0)

    for item in items:
        trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)

        # Re-derive what each predicate alone would have admitted, so the ladder can tell an
        # applicability exclusion apart from a dating exclusion instead of lumping both into
        # a single "filtered" bucket.
        excluded = trace.excluded_parts
        admitted_scope = _version_ids(store, exclude_parts=excluded, valid_date=item.as_of,
                                      temporal=False)
        admitted_temporal = _version_ids(store, exclude_parts=excluded,
                                         valid_date=item.as_of,
                                         temporal=retriever.temporal)

        stage, detail = observational(item, trace, store,
                                      admitted_temporal=admitted_temporal,
                                      admitted_scope=admitted_scope,
                                      rerank_top_k=retriever.rerank_top_k)
        budget.autopsies.append(Autopsy(item.id, stage, [], detail))
        if stage != "none":
            budget.failures += 1
            budget.observational[stage] += 1

    if interventional_sample:
        failed = [a for a in budget.autopsies if a.failed]
        stride = max(1, len(failed) // interventional_sample) if failed else 1
        by_id = {i.id: i for i in items}
        for autopsy in failed[::stride][:interventional_sample]:
            repairs = interventional(by_id[autopsy.item_id], retriever)
            budget.autopsies[budget.autopsies.index(autopsy)] = Autopsy(
                autopsy.item_id, autopsy.observational, repairs, autopsy.detail)
            for r in repairs:
                budget.repairs[r] += 1
    return budget


def _version_ids(store: Store, *, exclude_parts: list[str], valid_date: str,
                 temporal: bool) -> set[str]:
    ids = store.candidate_ids(valid_date=valid_date, temporal=temporal,
                              exclude_parts=exclude_parts)
    rows = store.db.execute(
        "SELECT id, version_id FROM chunk WHERE system_to IS NULL").fetchall()
    return {r["version_id"] for r in rows if r["id"] in ids}
