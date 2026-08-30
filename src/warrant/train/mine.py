"""Contrastive training triples mined from the amendment benchmark.

The corpus supplies a labelled contrastive set that almost no corpus supplies: the *same
paragraph before and after a real amendment*, near-identical legal prose differing by one
clause, with ground truth attached. 616 (query, gold, other-version) pairs exist across the
whole temporal bucket; 198 of them fall on the training side of the split.

Three kinds of negative are mined, and they are not equally useful. The counts are returned
per kind so the caller can say which it trained on:

``amendment``   the item's own ``distractors`` -- other versions of the same paragraph.
``sibling``     other paragraphs of the same section, in force on the item's date.
``lexical``     top BM25 hits for the query that are not the gold, via ``Store.search`` --
                the confusables the serving pipeline actually has to outrank.

**A measured caveat on the amendment negatives, recorded because it decides what this
training set can teach.** The temporal miner emits two items per amendment that *share a
query* and differ only in ``as_of``: the before-side's gold is the after-side's distractor
and vice versa. Measured on the dev split of this corpus, **55 of 63 distinct queries carry
two mutually exclusive positives**. A bi-encoder never sees ``as_of`` -- valid time is
enforced by the SQL predicate in ``Store.candidate_ids`` before any vector is scored -- so
asking it to pull a query toward the before text and away from the after text, and then the
exact opposite for the same query string, is a contradiction by construction, not a hard
example. It is mined anyway because it is what the benchmark labels, but a comparison that
does not also run without it cannot tell a null result apart from cancellation.

**Split discipline.** ``assign_split`` hashes the *section*, so training on dev and
reporting on test guarantees no section is ever on both sides. Every text that reaches the
trainer -- positive *and* negative -- is checked against that hash here, not only the
positives: a held-out section's paragraph used as a negative is still that section's text in
the training set, and the whole value of the reported number is that nobody has to take that
on trust.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..eval.bench import BenchItem, assign_split, mine_temporal
from ..index.store import Store
from ..retrieve.hybrid import fts_query

#: The item's own distractors: another version of the same paragraph.
AMENDMENT = "amendment"
#: Another paragraph of the same section, in force on the item's date.
SIBLING = "sibling"
#: A BM25 hit for the query that is not the gold. Mined through ``Store.search``, the same
#: lexical path the pipeline serves from, so the negatives are the ones the retriever
#: actually loses to rather than ones a second implementation happened to surface.
LEXICAL = "lexical"
KINDS = (AMENDMENT, SIBLING, LEXICAL)


@dataclass(frozen=True)
class Triple:
    """One (query, positive, negative) example.

    ``query`` carries no instruction prefix. BGE is asymmetric and the serving path adds
    ``QUERY_INSTRUCTION`` in ``dense.encode_query``; adding it here as well would train on a
    doubled prefix, and adding it in neither place would train and serve two different
    things. The trainer owns that decision in one place -- see ``train.finetune``.
    """

    query: str
    positive: str
    negative: str
    #: Clustering and split key. Items from one section are not independent examples.
    section_id: str
    #: The gold paragraph's stable id, e.g. ``630.1203#a``. The batch sampler needs it: two
    #: examples for the same paragraph in one batch make each other's in-batch negative,
    #: which labels the correct answer as wrong.
    chunk_id: str
    kind: str


@dataclass(frozen=True)
class Mined:
    triples: list[Triple]
    #: Per-kind triple counts, in ``KINDS`` order. Reported rather than summed because the
    #: kinds carry very different signal (see the module docstring).
    counts: dict[str, int]
    items_used: int
    #: Items skipped for want of a distractor -- a pure addition or deletion has no other
    #: version to contrast with.
    items_without_distractor: int
    sections: list[str]
    #: Distinct queries that appear with more than one gold. These are the before/after
    #: pairs whose amendment negatives cancel; the caller should print it.
    contradictory_queries: int = 0

    def __len__(self) -> int:
        return len(self.triples)

    def of_kind(self, *kinds: str) -> list[Triple]:
        keep = set(kinds)
        return [t for t in self.triples if t.kind in keep]

    def summary(self) -> str:
        per_kind = ", ".join(f"{k}={self.counts.get(k, 0)}" for k in KINDS)
        return (f"{len(self.triples)} triples ({per_kind}) from {self.items_used} items "
                f"over {len(self.sections)} sections; "
                f"{self.contradictory_queries} contradictory queries; "
                f"{self.items_without_distractor} items had no distractor")


def document_text(heading: str | None, text: str) -> str:
    """The exact string ``dense.build`` embeds for a chunk.

    Duplicated deliberately rather than imported: ``build`` inlines it inside a list
    comprehension. If the two ever diverge the fine-tune optimises a document form that is
    never indexed, and nothing would fail -- the scores would stay finite and slightly worse.
    """
    return f"{heading}. {text}" if heading else text


def _rows_by_version(store: Store, version_ids: Sequence[str]) -> dict[str, dict]:
    if not version_ids:
        return {}
    out: dict[str, dict] = {}
    ids = list(dict.fromkeys(version_ids))
    for start in range(0, len(ids), 500):     # SQLite's default variable limit is 999
        window = ids[start:start + 500]
        marks = ",".join("?" * len(window))
        for r in store.db.execute(
                f"SELECT version_id, chunk_id, section_id, heading, text FROM chunk "
                f"WHERE version_id IN ({marks}) AND system_to IS NULL", window):
            out[r["version_id"]] = dict(r)
    return out


def _siblings(store: Store, item: BenchItem, gold_chunk_id: str) -> list[dict]:
    """Other paragraphs of the same section, as they stood on the item's own date.

    In force on ``as_of``, not every version ever: those are precisely the rows the
    as-of predicate admits when this item is served, so they are the section-internal
    competitors the retriever has to rank the gold above.
    """
    rows = store.db.execute(
        "SELECT version_id, chunk_id, section_id, heading, text FROM chunk "
        "WHERE section_id = ? AND system_to IS NULL AND chunk_id != ? "
        "AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?) ORDER BY chunk_id",
        (item.section_id, gold_chunk_id, item.as_of, item.as_of)).fetchall()
    return [dict(r) for r in rows]


def _lexical(store: Store, item: BenchItem, gold_chunk_id: str, *, limit: int) -> list[dict]:
    rows = store.search(fts_query(item.query), valid_date=item.as_of, limit=limit)
    return [dict(r) for r in rows
            if r["chunk_id"] != gold_chunk_id and r["version_id"] not in item.all_evidence]


def triples_from_items(store: Store, items: Sequence[BenchItem], *,
                       split: str = "dev", siblings_per_item: int = 2,
                       lexical_per_item: int = 2, lexical_pool: int = 30,
                       seed: int = 0) -> Mined:
    """Triples for the given items, admitting only sections on ``split``'s side.

    The split is re-derived from ``assign_split`` rather than trusted from ``BenchItem.split``
    for both positives and negatives. A leak here would be invisible: every downstream number
    would still compute, and would simply be wrong.
    """
    on_split = [i for i in items if assign_split(i.section_id) == split]
    gold_ids = [i.acceptable_evidence[0][0] for i in on_split if i.acceptable_evidence
                and i.acceptable_evidence[0]]
    distractor_ids = [d for i in on_split for d in i.distractors]
    rows = _rows_by_version(store, gold_ids + distractor_ids)

    rng = random.Random(seed)
    seen: set[tuple[str, str, str]] = set()
    triples: list[Triple] = []
    counts = dict.fromkeys(KINDS, 0)
    used = 0
    without_distractor = 0
    by_query: dict[str, set[str]] = {}

    for item in on_split:
        if not item.acceptable_evidence or not item.acceptable_evidence[0]:
            continue
        gold = rows.get(item.acceptable_evidence[0][0])
        if gold is None:
            continue
        by_query.setdefault(item.query, set()).add(gold["version_id"])
        positive = document_text(gold["heading"], gold["text"])

        candidates: list[tuple[str, dict]] = []
        hard = [rows[d] for d in item.distractors if d in rows]
        if not hard:
            # No counterpart version, so nothing this item can teach about near-duplicates.
            # Skipped whole rather than kept on its weaker negatives: an item that cannot
            # supply the hard case would otherwise dilute the batch with an easy one.
            without_distractor += 1
            continue
        candidates += [(AMENDMENT, r) for r in hard]

        sibling_rows = _siblings(store, item, gold["chunk_id"])
        rng.shuffle(sibling_rows)
        candidates += [(SIBLING, r) for r in sibling_rows[:siblings_per_item]]

        lexical_rows = _lexical(store, item, gold["chunk_id"], limit=lexical_pool)
        candidates += [(LEXICAL, r) for r in lexical_rows[:lexical_per_item]]

        emitted = 0
        for kind, row in candidates:
            if assign_split(row["section_id"]) != split:
                # A held-out section's text is held out even as a negative.
                continue
            negative = document_text(row["heading"], row["text"])
            if negative == positive:
                # Identical text is not a negative. It happens: a paragraph can be
                # re-published unchanged under a new valid_from when a sibling was amended.
                continue
            key = (item.query, positive, negative)
            if key in seen:
                continue
            seen.add(key)
            triples.append(Triple(query=item.query, positive=positive, negative=negative,
                                  section_id=item.section_id, chunk_id=gold["chunk_id"],
                                  kind=kind))
            counts[kind] += 1
            emitted += 1
        used += emitted > 0

    return Mined(
        triples=triples, counts=counts, items_used=used,
        items_without_distractor=without_distractor,
        sections=sorted({t.section_id for t in triples}),
        contradictory_queries=sum(1 for golds in by_query.values() if len(golds) > 1),
    )


def mine(store: Store, *, horizon: str, split: str = "dev", siblings_per_item: int = 2,
         lexical_per_item: int = 2, seed: int = 0) -> Mined:
    """Mine the temporal bucket into triples for one side of the split.

    ``horizon`` is the same date the benchmark is mined at; passing a different one would
    train against items the reported evaluation never sees.
    """
    items = mine_temporal(store, horizon=horizon)
    return triples_from_items(store, items, split=split, seed=seed,
                              siblings_per_item=siblings_per_item,
                              lexical_per_item=lexical_per_item)


#: Kinds included by default in a training run. All three: the amendment negatives are what
#: the benchmark labels, and dropping them before measuring anything would be choosing the
#: result. ``finetune`` accepts an override so the contradiction described in the module
#: docstring can be ablated rather than argued about.
DEFAULT_KINDS: tuple[str, ...] = KINDS


@dataclass(frozen=True)
class Batch:
    """One training batch: parallel query/positive/negative columns."""

    queries: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)


def batches(triples: Sequence[Triple], *, batch_size: int, seed: int = 0) -> list[Batch]:
    """Shuffle into batches holding **at most one example per paragraph**.

    In-batch negatives are the whole point of ``MultipleNegativesRankingLoss``: every other
    row's positive is a negative for this row. Two examples for the same paragraph in one
    batch therefore label the correct passage as wrong -- and on this data that is not a rare
    accident, since the before and after sides of one amendment share a query and 55 of 63
    dev queries appear twice. Grouping by ``chunk_id`` costs nothing and removes it.
    """
    rng = random.Random(seed)
    pool = list(triples)
    rng.shuffle(pool)

    out: list[Batch] = []
    while pool:
        current: list[Triple] = []
        taken: set[str] = set()
        leftover: list[Triple] = []
        for t in pool:
            if len(current) < batch_size and t.chunk_id not in taken:
                current.append(t)
                taken.add(t.chunk_id)
            else:
                leftover.append(t)
        pool = leftover
        if len(current) < 2:
            # A batch of one has no in-batch negatives at all; its loss is the explicit
            # negative only, which is a different objective. Dropped rather than trained on.
            break
        out.append(Batch([t.query for t in current], [t.positive for t in current],
                         [t.negative for t in current]))
    return out
