"""A second hop along the references the retrieved text actually makes.

At the shipped ``final_k: 16``, **77.3% of evidence sets are missing at least one reference
whose target the corpus holds**, 4.14 of them on average (results/eval-006 §1). Widening
retrieval makes that worse rather than better: every admitted chunk brings in more references
than the set it joined satisfies, so 4 -> 8 -> 16 moves the per-answer count 1.32 -> 2.39 ->
4.14. There is no ``final_k`` at which the conditional chain closes, because the missing
paragraph is usually not *similar* to the query -- §630.306(a) opens "Except as provided in
paragraph (b) of this section", and (b) is reached by citation, not by resemblance. A single
similarity ranking cannot find it however wide it is opened.

So this module follows the edge instead of widening the beam.

**Why a graph walk and not an agent.** The edges are written into the text by the drafter and
``verify.xref`` already parses and resolves them -- 94.7% of in-corpus targets to that exact
chunk id, 4.6% to an ancestor, 0.8% to the section. Nothing about "which reference should I
follow" needs inference: the answer is *all of them the set does not already hold*, in the
order the drafter wrote them. Handing that to a model would add a generation's worth of
latency to a 1 ms lookup, make the same query expand two different ways on two runs, and
introduce a new failure mode -- a wrong hop, attributable to nothing -- in a step that has a
correct answer already. Determinism is not a stylistic preference here; replay
(ARCHITECTURE.md §8) compares two orderings, and a stage that reorders on its own cannot be
diffed.

**The predicates hold on hop 2.** A reference names a chunk id, never a version id -- the 2017
text of §630.306 cites §630.310, not any particular version of it (``xref`` module docstring)
-- so choosing the version is still the as-of predicate's job. Every candidate this module
considers is drawn by a query carrying the same valid-time, system-time, applicability,
source and authority clauses hop 1 ran under, and the resolution set is built from what that
query returned. A reference into a superseded version therefore resolves to nothing rather
than being filtered out afterwards, and so does a reference into a part the asker's scope
excludes. That is not tidiness: the as-of predicate scores +96.1 points on wrong-version rate,
and a second hop that filtered after ranking would be a fresh way to lose all of it.

**Termination.** ``depth`` hops (2 = one round of expansion from the first-hop set), a slot
budget, and a visited set that carries every chunk id already present or already admitted.
Regulations cite in loops -- §630.306(a) points at (b), and (b) points back -- and a cycle
terminates on the visited set rather than on the depth cap: the target is already in the
evidence set, so it is not dangling, so it is not followed. Depth 3 was measured before being
claimed; see results/eval-013.

**Budget.** Hop-2 admissions do not extend ``final_k``, they compete for it. ``budget`` slots
are taken from the tail of the first-hop ranking, and any the walk does not fill are handed
straight back, so a query whose evidence set is already closed retrieves exactly what it
retrieves today. Displacement is the cost, and eval-013 measures it against sufficiency on
the held-out split rather than assuming it is small.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..index.store import Store, _authority_clause, _exclusion_clause, now
from ..verify import xref
from ..verify.xref import Reference
from .hybrid import Candidate, Retriever, Trace
from .scope import GOVERNMENT_WIDE, Scope

#: The trace stage this hop's output is recorded under, between ``reranked`` and ``final``.
#: Not yet a member of ``hybrid.STAGES`` -- that tuple, and the ``Trace`` constructor that
#: zips against it, belong to ``hybrid.py``. Until the name is added there ``Expansion.record``
#: offers the stage to ``Trace.record`` and always writes ``trace.admissions``, which is the
#: record the failure budget actually needs: a ``Candidate`` carries an id, a score and a
#: rank, and cannot carry the reference phrase that caused the admission.
HOP_STAGE = "expanded"

#: Where the provenance lands on the trace. Deliberately not ``trace.expanded``: that is the
#: name the derived stage property will take when ``HOP_STAGE`` joins ``hybrid.STAGES``, and
#: assigning over a property is an AttributeError the day the wiring lands.
TRACE_ATTR = "admissions"

#: The first hop is depth 1, so the first round of expansion is depth 2.
FIRST_HOP_DEPTH = 1


@dataclass(frozen=True, slots=True)
class Admission:
    """One chunk admitted because retrieved text pointed at it, and the pointer that did it.

    ``reference`` is the phrase as the drafter wrote it, not a reconstruction. Without it a
    hop-2 chunk in the evidence set is indistinguishable from a hop-1 chunk that ranked badly,
    and the failure budget cannot charge anything to this stage.
    """

    version_id: str
    chunk_id: str
    #: 2 for a reference made by the first-hop set, 3 for one made by a hop-2 admission.
    depth: int
    source: str                     # version id of the citing chunk
    source_rank: int                # its rank in the first-hop list; the ordering key
    reference: str                  # the phrase in the citing text
    kind: str                       # paragraph | section | cfr
    #: The address the phrase named, after ``xref.resolve``. Differs from ``chunk_id`` on the
    #: 5.4% of targets that resolve to an ancestor or to the bare section.
    target: str


@dataclass(frozen=True)
class Expansion:
    """What the second hop saw, what it admitted, and what it cost."""

    #: This stage's output: the whole re-ordered pool, uncut, the way every other stage
    #: records its ranking rather than the slice taken from it.
    ordered: tuple[Candidate, ...] = ()
    #: ``ordered`` cut to ``final_k``. What the generator is handed.
    final: tuple[Candidate, ...] = ()
    admitted: tuple[Admission, ...] = ()
    #: Resolvable targets the evidence set did not already hold -- the dangling references
    #: this walk was offered. Larger than ``len(admitted)`` whenever the budget binds.
    dangling: int = 0
    #: Citing chunks whose references were read, across every depth.
    expanded_from: int = 0
    depth_reached: int = FIRST_HOP_DEPTH
    elapsed_ms: float = 0.0

    def record(self, trace: Trace) -> None:
        """Write this stage into the trace, then re-cut ``final`` to what it produced."""
        setattr(trace, TRACE_ATTR, list(self.admitted))
        trace.timings[HOP_STAGE] = self.elapsed_ms
        try:
            trace.record(HOP_STAGE, self.ordered)
        except KeyError:
            # hybrid.STAGES does not carry the name yet; ``trace.admissions`` above is the
            # record until it does. Swallowed rather than raised because a trace that cannot
            # name a stage is a wiring gap, not a reason to fail the request.
            pass
        trace.record("final", self.final)


@dataclass(frozen=True, slots=True)
class _Pointer:
    """A parsed reference waiting to be resolved, with where it came from."""

    target: str
    reference: Reference
    source: str
    source_rank: int


class ReferenceExpander:
    """Admits the chunks the first-hop set points at and does not contain.

    Holds no state between queries. The predicate arguments are the retriever's own, passed
    in rather than re-derived, because a second hop running under a different applicability
    filter than the first would be a scope failure that no test of either half could see.
    """

    def __init__(self, store: Store, *, budget: int = 4, depth: int = 2,
                 temporal: bool = True, sources: Sequence[str] | None = None,
                 max_authority: int | None = None) -> None:
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        if depth < FIRST_HOP_DEPTH:
            raise ValueError(f"depth must be >= {FIRST_HOP_DEPTH}, got {depth}")
        self.store = store
        self.budget = budget
        self.depth = depth
        self.temporal = temporal
        self.sources = tuple(sources) if sources else None
        self.max_authority = max_authority

    # -- the walk ----------------------------------------------------------------

    def expand(self, pool: Sequence[Candidate], *, final_k: int, as_of: str,
               exclude_parts: Sequence[str] = (),
               system_time: str | None = None) -> Expansion:
        """Re-cut ``pool`` to ``final_k`` with up to ``budget`` slots given to references.

        ``pool`` is the full ranked list the first hop produced -- ``reranked`` if the
        cross-encoder ran, ``fused`` otherwise -- not the ``final_k`` slice of it. The slice
        is made here so that unfilled budget is returned to the first hop instead of
        shortening the answer: an evidence set with no dangling reference comes out
        byte-identical to what ships today.

        References are read from the whole ``final_k`` slice, and displacement is then taken
        off its tail. Reading only the surviving head instead -- which is the obvious
        implementation -- makes the budget **non-monotone**: at 8 slots of 16 only 8 chunks
        are read, so a larger budget offers the walk fewer pointers than a smaller one, and
        the measured dangling rate turns back upward at budget 6 (47.1% -> 52.6% -> 57.6%).
        The references being closed are the ones the shipped evidence set makes, so the
        shipped evidence set is what gets read.
        """
        started = time.perf_counter()
        budget = min(self.budget, final_k)
        if budget == 0 or self.depth <= FIRST_HOP_DEPTH:
            ordered = _dedup(pool)
            return Expansion(ordered=tuple(ordered), final=tuple(ordered[:final_k]),
                             elapsed_ms=(time.perf_counter() - started) * 1000.0)

        citing = list(pool[:final_k])
        present = {xref.chunk_id_of(c.version_id) for c in citing}
        texts = self._texts([c.version_id for c in citing])
        frontier = [(c.version_id, rank, texts.get(c.version_id, ""))
                    for rank, c in enumerate(citing, start=1) if texts.get(c.version_id)]
        admitted: list[Admission] = []
        dangling = expanded_from = 0
        reached = FIRST_HOP_DEPTH

        for depth in range(FIRST_HOP_DEPTH + 1, self.depth + 1):
            if not frontier or len(admitted) >= budget:
                break
            reached = depth
            expanded_from += len(frontier)
            pointers = _pointers(frontier)
            rows = self._lookup(pointers, as_of=as_of, exclude_parts=exclude_parts,
                                system_time=system_time)
            fresh, seen = self._admit(pointers, rows, present, depth,
                                      budget - len(admitted))
            dangling += seen
            admitted.extend(fresh)
            # The next frontier is what this round admitted, and only that. Re-reading a
            # chunk that has already been expanded would re-emit every target it names for
            # ``present`` to discard one round later, at the cost of a second regex pass over
            # its text -- and it is the shape that turns a citation loop into an infinite one.
            texts = self._texts([a.version_id for a in fresh])
            frontier = [(a.version_id, a.source_rank, texts.get(a.version_id, ""))
                        for a in fresh if texts.get(a.version_id)]

        # Displacement is exactly what the walk used: the tail of the first-hop cut gives up
        # one slot per admission and no more.
        keep = final_k - len(admitted)
        ordered = _dedup([*pool[:keep],
                          *(Candidate(a.version_id) for a in admitted),
                          *pool[keep:]])
        return Expansion(ordered=tuple(ordered), final=tuple(ordered[:final_k]),
                         admitted=tuple(admitted), dangling=dangling,
                         expanded_from=expanded_from, depth_reached=reached,
                         elapsed_ms=(time.perf_counter() - started) * 1000.0)

    def _admit(self, pointers: Sequence[_Pointer], rows: _Corpus, present: set[str],
               depth: int, budget: int) -> tuple[list[Admission], int]:
        """Resolve each pointer against what the predicates admitted, and take the misses.

        ``present`` is mutated as the round proceeds, so two chunks citing the same paragraph
        spend one slot rather than two, and a target satisfied by an admission made three
        pointers ago is no longer dangling. That incremental check is why this walks the
        pointers itself instead of calling ``xref.dangling_references``, which computes
        presence once against a fixed evidence set.
        """
        out: list[Admission] = []
        dangling = 0
        for p in pointers:
            if _covered(p.target, present):
                continue
            located = xref.resolve(p.target, rows.nameable)
            # "" means the corpus holds no address under this target *that the predicates
            # admitted*: another CFR title, the U.S.C., a superseded version, or a part this
            # scope excludes. The four are indistinguishable here on purpose -- all four are
            # "do not follow", and telling them apart is what eval-006's status column is
            # for.
            if not located or _covered(located, present):
                continue
            row = rows.representative(located)
            if row is None or row["chunk_id"] in present:
                continue
            dangling += 1
            if len(out) >= budget:
                continue
            present.add(row["chunk_id"])
            out.append(Admission(
                version_id=row["version_id"], chunk_id=row["chunk_id"], depth=depth,
                source=p.source, source_rank=p.source_rank, reference=p.reference.text,
                kind=p.reference.kind, target=located))
        return out, dangling

    # -- the store -----------------------------------------------------------------

    def _predicate(self, params: dict[str, object], *, as_of: str,
                   exclude_parts: Sequence[str], system_time: str | None,
                   alias: str = "") -> str:
        """The clauses hop 1 ran under, as SQL.

        Assembled from ``store``'s own helpers rather than written out again. Two copies of
        an applicability predicate is how a hop-2 candidate comes to be admitted under a
        filter the first hop would have refused it under, and neither module's tests would
        show it.
        """
        params["v"] = as_of
        params["s"] = system_time or now()
        valid = (f"AND {alias}valid_from <= :v "
                 f"AND ({alias}valid_to IS NULL OR {alias}valid_to > :v) "
                 if self.temporal else "")
        system = (f"AND {alias}system_from <= :s "
                  f"AND ({alias}system_to IS NULL OR {alias}system_to > :s) ")
        return (valid + system + _exclusion_clause(exclude_parts, params, alias) + " "
                + _authority_clause(self.sources, self.max_authority, params, alias))

    def _lookup(self, pointers: Sequence[_Pointer], *, as_of: str,
                exclude_parts: Sequence[str], system_time: str | None) -> _Corpus:
        """Every admitted row that could answer one of these pointers, in document order.

        Scoped to the sections the pointers actually name -- ~40 of the corpus's sections on
        a 16-chunk evidence set -- rather than to the whole in-force corpus. The corpus-wide
        set would have to be rebuilt or cached per (as_of, scope) pair, and serving sees a
        different as_of per request, so the cache would miss on exactly the path it was for.
        """
        sections = sorted({xref.section_of(p.target) for p in pointers})
        if not sections:
            return _Corpus.empty()
        params: dict[str, object] = {}
        marks = []
        for i, section in enumerate(sections):
            params[f"sec{i}"] = section
            marks.append(f":sec{i}")
        predicate = self._predicate(params, as_of=as_of, exclude_parts=exclude_parts,
                                    system_time=system_time, alias="")
        rows = self.store.db.execute(
            f"SELECT id, version_id, chunk_id, section_id FROM chunk "
            f"WHERE section_id IN ({', '.join(marks)}) {predicate} ORDER BY id",
            params).fetchall()
        return _Corpus.of(rows)

    def _texts(self, version_ids: Sequence[str]) -> dict[str, str]:
        """Chunk text by version id. The trace carries addresses, not prose.

        ``system_to IS NULL`` matters: ``version_id`` is deliberately not UNIQUE, because one
        valid-time version can be believed more than once over system time, and a retracted
        row would otherwise supply the text whose parse was corrected.
        """
        if not version_ids:
            return {}
        marks = ",".join("?" * len(version_ids))
        return {r["version_id"]: r["text"] for r in self.store.db.execute(
            f"SELECT version_id, text FROM chunk WHERE version_id IN ({marks}) "
            f"AND system_to IS NULL", list(version_ids))}


@dataclass(frozen=True)
class _Corpus:
    """The admitted rows for the sections one round of pointers names."""

    nameable: frozenset[str]
    by_chunk: Mapping[str, sqlite3.Row]
    by_section: Mapping[str, Sequence[sqlite3.Row]]

    @classmethod
    def empty(cls) -> _Corpus:
        return cls(frozenset(), {}, {})

    @classmethod
    def of(cls, rows: Iterable[sqlite3.Row]) -> _Corpus:
        by_chunk: dict[str, sqlite3.Row] = {}
        by_section: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_chunk.setdefault(row["chunk_id"], row)
            by_section.setdefault(row["section_id"], []).append(row)
        return cls(xref.nameable_ids(by_chunk), by_chunk, by_section)

    def representative(self, located: str) -> sqlite3.Row | None:
        """The one chunk to admit for an address, or None if the predicates admitted none.

        ``located`` comes back from ``xref.resolve`` against ``nameable``, which materialises
        the ancestors real chunk ids imply -- so it can name an address no row carries.
        §890.102 is written with paragraph (j)'s chapeau running into (j)(1), so the store
        holds ``j-1``..``j-5`` and no bare ``#j``, and a reference to "paragraph (j)" resolves
        to an address that exists only as an ancestor. One chunk per reference, not the whole
        subtree: the budget is the scarce thing, and admitting five paragraphs for one pointer
        would spend an evidence set on a single citation.
        """
        row = self.by_chunk.get(located)
        if row is not None:
            return row
        section = xref.section_of(located)
        prefix = f"{located}-" if "#" in located else ""
        for candidate in self.by_section.get(section, ()):
            if not prefix or candidate["chunk_id"].startswith(prefix):
                return candidate
        return None


def _pointers(frontier: Sequence[tuple[str, int, str]]) -> list[_Pointer]:
    """Every target the frontier names, in (citing rank, document order) order.

    That order is the whole ranking policy of this stage and it is deliberately not a score:
    the references of the chunk the first hop liked best are followed first, and within a
    chunk the drafter's own order is kept. Nothing here compares a reference to the query,
    because a reference's relevance is not a similarity -- it is that the text the reader was
    shown says to go and read it.
    """
    out: list[_Pointer] = []
    seen: set[str] = set()
    for version_id, rank, text in sorted(frontier, key=lambda f: (f[1], f[0])):
        chunk_id = xref.chunk_id_of(version_id)
        for ref in xref.find_references(text, section_id=xref.section_of(chunk_id),
                                        anchor=xref.anchor_of(chunk_id)):
            for target in ref.targets:
                if target == chunk_id or target in seen:
                    continue
                seen.add(target)
                out.append(_Pointer(target, ref, version_id, rank))
    return out


def _covered(target: str, present: set[str]) -> bool:
    """Is ``target`` already answered by the evidence set?

    Same reading as ``xref._satisfies`` and kept in step with it deliberately: a section-level
    target is covered by any paragraph of that section, a paragraph-level one by itself or by
    a descendant, and never by an ancestor -- holding the chapeau of (b) says nothing about
    what (b)(2) requires. Restated here rather than imported because this is the check that
    decides whether a slot is spent, and it runs against a set that grows inside the round.
    """
    if "#" not in target:
        return any(xref.section_of(c) == target for c in present)
    prefix = f"{target}-"
    return any(c == target or c.startswith(prefix) for c in present)


def _dedup(candidates: Iterable[Candidate]) -> list[Candidate]:
    """First occurrence wins. A hop-2 admission and the first-hop tail can name the same row."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        if c.version_id in seen:
            continue
        seen.add(c.version_id)
        out.append(c)
    return out


@dataclass
class MultiHopRetriever(Retriever):
    """``Retriever`` plus the second hop. Substitutable everywhere the first one is.

    A subclass rather than a wrapper so that ``eval.run.score``, the autopsy, the API and the
    replay path all keep working against the type they already hold. The trace it returns is
    a first-hop trace with one more stage on it, which is the only shape the failure budget
    can attribute anything to.
    """

    #: Slots of ``final_k`` that reference-directed candidates may take. 0 disables the hop
    #: and makes this class byte-identical to ``Retriever``.
    hop_budget: int = 0
    #: 1 = no expansion. 2 = one round from the first-hop set. See eval-013 for whether 3 buys
    #: anything: on the held-out split it did not.
    hop_depth: int = 2
    _expander: ReferenceExpander | None = field(default=None, init=False, repr=False)

    @property
    def expander(self) -> ReferenceExpander:
        if self._expander is None:
            self._expander = ReferenceExpander(
                self.store, budget=self.hop_budget, depth=self.hop_depth,
                temporal=self.temporal, sources=self.sources,
                max_authority=self.max_authority)
        return self._expander

    def retrieve(self, query: str, *, as_of: str, scope: Scope = GOVERNMENT_WIDE,
                 system_time: str | None = None) -> Trace:
        trace = super().retrieve(query, as_of=as_of, scope=scope, system_time=system_time)
        if self.hop_budget <= 0:
            return trace
        started = trace.timings.get("total", 0.0)
        # The reranked list where the cross-encoder ran, the fused one otherwise: the second
        # hop re-cuts the ranking rather than the cut, so budget it does not spend goes back
        # to the first hop instead of shortening the answer.
        pool = trace.candidates("reranked") or trace.candidates("fused")
        expansion = self.expander.expand(
            pool, final_k=self.final_k, as_of=as_of,
            exclude_parts=trace.excluded_parts, system_time=system_time)
        expansion.record(trace)
        trace.timings["total"] = started + expansion.elapsed_ms
        return trace
