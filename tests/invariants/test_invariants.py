"""The deterministic correctness gates from ARCHITECTURE.md section 9.

These are assertions about the system, not tests of a function, and CI runs them as a
separate job so a failure reads as "the system is wrong" rather than "a test broke". They
move correctness out of probabilistic evaluation: a dated query returning two versions of one
section is a bug that can be caught before any model runs, with no ground truth and no
threshold to argue about.

Each invariant is checked twice: once on a small synthetic store, so it runs on a fresh
clone with no corpus, and once on the real corpus when one has been built.
"""

from __future__ import annotations

from collections import Counter

import pytest

from warrant.config import Config
from warrant.index.store import Chunk, Store
from warrant.retrieve.hybrid import Retriever
from warrant.retrieve.scope import Scope

T0 = "2026-01-01T00:00:00+00:00"

DATES = ["2018-06-01", "2021-06-01", "2024-06-01"]


@pytest.fixture
def synthetic() -> Store:
    with Store(":memory:") as s:
        s.add([
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630",
                  anchor="a", heading="Restored annual leave",
                  text="annual leave restored must be scheduled within two years",
                  valid_from="2017-01-01", valid_to="2020-08-10"),
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630",
                  anchor="a", heading="Restored annual leave",
                  text="annual leave restored must be scheduled within three years",
                  valid_from="2020-08-10"),
            Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531",
                  anchor="a", heading="Within-grade increase",
                  text="performance must be at an acceptable level of competence",
                  valid_from="2017-01-01"),
        ], system_from=T0)
        yield s


@pytest.fixture(scope="module")
def corpus() -> Store:
    cfg = Config.load()
    if not cfg.store_path.exists():
        pytest.skip("no corpus built; run `make fetch && make build`")
    with Store(cfg.store_path) as s:
        yield s


def _one_version_per_section(rows) -> None:
    """At most one *version* of a section in force, not at most one row.

    A section has many paragraphs and each is a row, so counting rows per section would
    fail on every healthy corpus. The invariant is that all rows for a section share one
    ``valid_from``, and that no paragraph address appears twice.
    """
    versions: dict[str, set[str]] = {}
    for r in rows:
        versions.setdefault(r["section_id"], set()).add(r["valid_from"])
    offenders = {s: sorted(v) for s, v in versions.items() if len(v) > 1}
    assert not offenders, f"two versions in force at once: {sorted(offenders)[:5]}"

    counts = Counter(r["chunk_id"] for r in rows)
    dupes = {c: n for c, n in counts.items() if n > 1}
    assert not dupes, f"paragraph address in force twice: {sorted(dupes)[:5]}"


# -- invariant 1: at most one version of a section on any date --------------------


def test_synthetic_dated_query_sees_one_version_per_section(synthetic: Store):
    for date in DATES:
        _one_version_per_section(synthetic.as_of(date))


def test_corpus_dated_query_sees_one_version_per_section(corpus: Store):
    """Two versions of one section reaching a prompt is a filter bug, and it is detectable
    without a model, a label, or a threshold."""
    for date in DATES:
        _one_version_per_section(corpus.as_of(date))


# -- invariant 2: every retrieved chunk is valid at the as-of date ----------------


def test_retrieved_chunks_are_in_force_at_the_as_of_date(corpus: Store):
    r = Retriever(store=corpus, candidates_lexical=50, rerank_top_k=20, final_k=10,
                  parts_universe=Config.load().corpus.parts)
    for date in DATES:
        trace = r.retrieve("annual leave restored scheduled", as_of=date)
        rows = {row["version_id"]: row for row in corpus.db.execute(
            "SELECT version_id, valid_from, valid_to FROM chunk WHERE system_to IS NULL")}
        for vid in trace.final:
            row = rows[vid]
            assert row["valid_from"] <= date, f"{vid} not yet in force on {date}"
            assert row["valid_to"] is None or row["valid_to"] > date, \
                f"{vid} superseded before {date}"


# -- invariant 3: every retrieved chunk is applicable to the resolved scope --------


def test_retrieved_chunks_are_applicable_to_the_scope(corpus: Store):
    cfg = Config.load()
    r = Retriever(store=corpus, candidates_lexical=50, rerank_top_k=20, final_k=10,
                  parts_universe=cfg.corpus.parts)
    for scope in (Scope.of(pay_system="GS"), Scope.of(pay_system="FWS"),
                  Scope.of(service="competitive")):
        trace = r.retrieve("pay schedule rate increase", as_of="2024-06-01", scope=scope)
        parts = {v: row["part"] for v, row in
                 ((row["version_id"], row) for row in corpus.db.execute(
                     "SELECT version_id, part FROM chunk WHERE system_to IS NULL"))}
        for vid in trace.final:
            assert scope.governs(parts[vid]), \
                f"{vid} is in part {parts[vid]}, which does not govern {scope.describe()}"


# -- invariant 4: every citation address is unambiguous ---------------------------


def test_version_ids_are_unique(corpus: Store):
    """A citation that matches more than one paragraph is not a citation. 13% of addresses
    were ambiguous before paragraph designators were tracked hierarchically."""
    total = corpus.db.execute(
        "SELECT COUNT(*) n FROM chunk WHERE system_to IS NULL").fetchone()["n"]
    distinct = corpus.db.execute(
        "SELECT COUNT(DISTINCT version_id) n FROM chunk WHERE system_to IS NULL"
    ).fetchone()["n"]
    assert total == distinct, f"{total - distinct} ambiguous citation addresses"


# -- invariant 5: validity intervals never overlap for a section ------------------


def test_validity_intervals_do_not_overlap(corpus: Store):
    """Overlapping intervals would make invariant 1 unenforceable no matter how the query is
    written, because two versions would genuinely both be in force."""
    rows = corpus.db.execute(
        "SELECT chunk_id, valid_from, valid_to FROM chunk WHERE system_to IS NULL "
        "ORDER BY chunk_id, valid_from").fetchall()
    previous: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        prior = previous.get(row["chunk_id"])
        if prior is not None:
            prior_to = prior[1]
            assert prior_to is not None, \
                f"{row['chunk_id']} has an open interval followed by another version"
            assert prior_to <= row["valid_from"], \
                f"{row['chunk_id']} intervals overlap at {row['valid_from']}"
        previous[row["chunk_id"]] = (row["valid_from"], row["valid_to"])
