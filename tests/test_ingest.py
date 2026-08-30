"""Ingesting any source into the bitemporal store.

The tests that matter are the ones about *re-ingest*. A guidance page is re-fetched on a
schedule and is usually byte-identical; a source that inserted a new version every time would
inflate the store, break the one-version-in-force invariant, and make every citation
ambiguous — which is exactly what the CFR path did before it was guarded.
"""

from __future__ import annotations

import pytest

from warrant.corpus.ingest import doc_chunks, ingest
from warrant.index.store import Store
from warrant.sources.base import (
    AUTHORITY_GUIDANCE,
    AUTHORITY_NOTICE,
    KIND_OCR,
    KIND_TABLE,
    SourceDoc,
    Unit,
)

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-06-01T00:00:00+00:00"


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def doc(doc_id: str = "85-FR-48089", *, texts=("first unit", "second unit"),
        valid_from: str = "2020-08-10", **kw) -> SourceDoc:
    return SourceDoc(
        source=kw.pop("source", "federal_register"),
        doc_id=doc_id,
        title=kw.pop("title", "Absence and Leave; Restored Annual Leave"),
        authority=kw.pop("authority", AUTHORITY_NOTICE),
        units=[Unit(anchor=f"u{i}", text=t) for i, t in enumerate(texts, start=1)],
        valid_from=valid_from,
        **kw,
    )


def test_units_become_chunks_carrying_provenance():
    chunks = doc_chunks(doc(), config_hash="cfg")
    assert [c.chunk_id for c in chunks] == ["85-FR-48089#u1", "85-FR-48089#u2"]
    assert all(c.source == "federal_register" for c in chunks)
    assert all(c.authority == AUTHORITY_NOTICE for c in chunks)
    assert all(c.doc_id == "85-FR-48089" for c in chunks)


def test_section_id_falls_back_to_the_document_id():
    """Every grouping in this codebase keys on section_id — the clustered bootstrap, the
    one-version-in-force gate, the dev/test split. A non-CFR document has to have one or all
    of that quietly stops applying to it."""
    chunks = doc_chunks(doc(), config_hash="cfg")
    assert all(c.section_id == "85-FR-48089" for c in chunks)


def test_unit_kind_and_locator_survive_to_the_store(store: Store):
    d = SourceDoc(source="govinfo", doc_id="CFR-2023-sec630", title="Restored leave",
                  authority=AUTHORITY_GUIDANCE, valid_from="2023-01-01",
                  units=[Unit(anchor="p1-t1", text="A | B", kind=KIND_TABLE,
                              locator="page 3 bbox 40,80,520,300"),
                         Unit(anchor="p4-ocr", text="scanned text", kind=KIND_OCR,
                              locator="page 4 conf 0.88")])
    ingest(store, [d], source="govinfo", config_hash="cfg", system_from=T0)
    rows = {r["anchor"]: r for r in store.db.execute("SELECT * FROM chunk")}
    assert rows["p1-t1"]["kind"] == KIND_TABLE
    assert "bbox" in rows["p1-t1"]["locator"]
    assert rows["p4-ocr"]["kind"] == KIND_OCR


def test_ingest_inserts_once(store: Store):
    stats = ingest(store, [doc()], source="federal_register", config_hash="cfg",
                   system_from=T0)
    assert stats.documents == 1
    assert stats.units_inserted == 2
    assert store.count() == 2


def test_reingesting_identical_content_is_a_no_op(store: Store):
    """A page re-fetched on a schedule is usually byte-identical. Inserting a version every
    time would inflate the store and put two versions of one document in force at once."""
    ingest(store, [doc()], source="federal_register", config_hash="cfg", system_from=T0)
    stats = ingest(store, [doc()], source="federal_register", config_hash="cfg",
                   system_from=T1)
    assert stats.documents_unchanged == 1
    assert stats.units_inserted == 0
    assert store.count() == 2


def test_changed_content_closes_the_old_version(store: Store):
    ingest(store, [doc(texts=("original text here", "second unit"))],
           source="opm", config_hash="cfg", system_from=T0)
    ingest(store, [doc(texts=("revised text here", "second unit"),
                       valid_from="2021-01-01")],
           source="opm", config_hash="cfg", system_from=T1)

    versions = store.versions_of("85-FR-48089")
    assert len(versions) == 4, "two units, two versions each"
    in_force = [v for v in versions if v["valid_to"] is None]
    assert len(in_force) == 2
    assert all(v["valid_from"] == "2021-01-01" for v in in_force)


def test_only_one_version_of_a_document_is_in_force_after_a_change(store: Store):
    ingest(store, [doc(texts=("a", "b"))], source="opm", config_hash="cfg", system_from=T0)
    ingest(store, [doc(texts=("c", "d"), valid_from="2021-01-01")],
           source="opm", config_hash="cfg", system_from=T1)
    rows = store.as_of("2022-01-01")
    assert len({r["valid_from"] for r in rows}) == 1


def test_a_document_that_parses_to_nothing_is_counted_not_ignored(store: Store):
    """A page that parses cleanly to zero units is a different bug from one that raised, and
    the silent version is the one that reaches production."""
    empty = SourceDoc(source="opm", doc_id="empty-page", title="Nothing",
                      authority=AUTHORITY_GUIDANCE, units=[], valid_from="2024-01-01")
    stats = ingest(store, [empty], source="opm", config_hash="cfg", system_from=T0)
    assert stats.documents_empty == 1
    assert stats.units_inserted == 0


def test_one_failing_document_does_not_abort_the_run(store: Store):
    """A source that fails on its 400th page must leave the first 399 readable."""
    class Exploding(SourceDoc):
        pass

    good_a, good_b = doc("doc-a"), doc("doc-b")

    def stream():
        yield good_a
        broken = doc("doc-bad")
        object.__setattr__(broken, "units", None)  # forces a TypeError inside doc_chunks
        yield broken
        yield good_b

    stats = ingest(store, stream(), source="federal_register", config_hash="cfg",
                   system_from=T0)
    assert stats.documents == 2
    assert stats.documents_failed == 1
    assert "doc-bad" in stats.failures[0]
    assert store.count() == 4


def test_documents_from_different_sources_coexist(store: Store):
    ingest(store, [doc("85-FR-48089", source="federal_register")],
           source="federal_register", config_hash="cfg", system_from=T0)
    ingest(store, [doc("5-USC-6304", source="usc", authority=1)],
           source="usc", config_hash="cfg", system_from=T0)
    by_source = dict(store.db.execute(
        "SELECT source, COUNT(*) FROM chunk GROUP BY source").fetchall())
    assert by_source == {"federal_register": 2, "usc": 2}


def test_authority_is_stored_so_retrieval_can_rank_by_it(store: Store):
    """When sources disagree, a statute outranks guidance. Retrieval can only apply that if
    ingestion recorded it."""
    ingest(store, [doc("5-USC-6304", source="usc", authority=1)],
           source="usc", config_hash="cfg", system_from=T0)
    ingest(store, [doc("opm-fact-sheet", source="opm", authority=AUTHORITY_GUIDANCE)],
           source="opm", config_hash="cfg", system_from=T0)
    ranked = store.db.execute(
        "SELECT DISTINCT source, authority FROM chunk ORDER BY authority").fetchall()
    assert [r["source"] for r in ranked] == ["usc", "opm"]
