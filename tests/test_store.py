"""The bitemporal store, including the property the second time axis exists for.

The tests that matter here are the ones that fail if the store is quietly reduced to a
single ``ingested_at`` column: retraction must preserve the retracted text, and a query at
a past system time must still see it.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-06-01T00:00:00+00:00"
T2 = "2026-09-01T00:00:00+00:00"


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def chunk(cid: str, text: str, valid_from: str, valid_to: str | None = None) -> Chunk:
    section = cid.split("#")[0]
    return Chunk(chunk_id=cid, section_id=section, title=5, part="630", text=text,
                 anchor=cid.split("#")[-1], heading="Leave entitlement",
                 valid_from=valid_from, valid_to=valid_to, source_snapshot=valid_from,
                 config_hash="test")


def test_add_and_count(store: Store):
    store.add([chunk("630.1203#a", "twelve administrative workweeks", "2018-05-10")])
    assert store.count() == 1


def test_as_of_selects_the_version_in_force(store: Store):
    store.add([
        chunk("630.1203#a@2018", "twelve administrative workweeks", "2018-05-10", "2020-08-10"),
        chunk("630.1203#a@2020", "twelve administrative workweeks of paid leave", "2020-08-10"),
    ], system_from=T0)
    assert [r["chunk_id"] for r in store.as_of("2019-06-01")] == ["630.1203#a@2018"]
    assert [r["chunk_id"] for r in store.as_of("2021-06-01")] == ["630.1203#a@2020"]


def test_valid_to_is_exclusive_at_the_boundary(store: Store):
    """A rule that ends on 2020-08-10 is not in force on 2020-08-10; the successor is.
    Off-by-one here would silently return the wrong law on every amendment date."""
    store.add([
        chunk("old", "before", "2018-05-10", "2020-08-10"),
        chunk("new", "after", "2020-08-10"),
    ], system_from=T0)
    assert [r["chunk_id"] for r in store.as_of("2020-08-09")] == ["old"]
    assert [r["chunk_id"] for r in store.as_of("2020-08-10")] == ["new"]


def test_as_of_returns_at_most_one_version_per_section(store: Store):
    """ARCHITECTURE.md section 9: a dated query seeing two versions of one section is a
    filter bug, detectable before any model runs."""
    store.add([
        chunk("630.1203#a@2018", "before", "2018-05-10", "2020-08-10"),
        chunk("630.1203#a@2020", "after", "2020-08-10"),
    ], system_from=T0)
    for date in ("2019-01-01", "2020-08-10", "2026-01-01"):
        sections = [r["section_id"] for r in store.as_of(date)]
        assert len(sections) == len(set(sections))


def test_retract_preserves_the_text_and_past_system_time_still_sees_it(store: Store):
    """The whole reason for a second time axis. A corrected parse must not destroy the
    ability to reconstruct what the system believed before the correction."""
    store.add([chunk("630.1203#a", "parse with a truncated sentence", "2018-05-10")],
              system_from=T0)
    store.retract("630.1203#a", system_to=T1)
    store.add([chunk("630.1203#a", "parse with the full sentence", "2018-05-10")],
              system_from=T1)

    now_rows = store.as_of("2019-01-01", system_time=T2)
    assert [r["text"] for r in now_rows] == ["parse with the full sentence"]

    then_rows = store.as_of("2019-01-01", system_time=T0)
    assert [r["text"] for r in then_rows] == ["parse with a truncated sentence"]

    assert store.count() == 2, "retraction must not delete the superseded row"


def test_system_time_boundary_is_exclusive(store: Store):
    store.add([chunk("c", "first belief", "2018-01-01")], system_from=T0)
    store.retract("c", system_to=T1)
    store.add([chunk("c", "second belief", "2018-01-01")], system_from=T1)
    assert [r["text"] for r in store.as_of("2019-01-01", system_time=T1)] == ["second belief"]


def test_close_valid_supersedes_only_the_believed_row(store: Store):
    store.add([chunk("c", "text", "2018-01-01")], system_from=T0)
    store.retract("c", system_to=T1)
    store.add([chunk("c", "corrected", "2018-01-01")], system_from=T1)
    assert store.close_valid("c", "2024-01-01") == 1
    live = [r for r in store.versions_of("c") if r["system_to"] is None]
    assert len(live) == 1 and live[0]["valid_to"] == "2024-01-01"


def test_search_applies_the_as_of_predicate(store: Store):
    """The predicate is inside the SQL. Filtering afterwards would let superseded text
    consume candidate slots before being thrown away."""
    store.add([
        chunk("old", "restored annual leave must be scheduled within two years",
              "2018-05-10", "2020-08-10"),
        chunk("new", "restored annual leave must be scheduled within three years",
              "2020-08-10"),
    ], system_from=T0)
    hits_2019 = store.search("restored annual leave", valid_date="2019-01-01")
    assert [r["chunk_id"] for r in hits_2019] == ["old"]
    hits_2021 = store.search("restored annual leave", valid_date="2021-01-01")
    assert [r["chunk_id"] for r in hits_2021] == ["new"]


def test_search_never_returns_two_versions_of_a_section(store: Store):
    store.add([
        chunk("630.306#a@2018", "annual leave restored under 6304(d)", "2018-05-10",
              "2020-08-10"),
        chunk("630.306#a@2020", "annual leave restored under 6304(d) or 630.310",
              "2020-08-10"),
    ], system_from=T0)
    hits = store.search("annual leave restored", valid_date="2021-01-01")
    assert len({r["section_id"] for r in hits}) == len(hits)


def test_open_interval_means_still_in_force(store: Store):
    store.add([chunk("c", "current rule", "2024-01-01")], system_from=T0)
    assert len(store.as_of("2099-01-01")) == 1


def test_content_hash_is_recorded_for_change_detection(store: Store):
    store.add([chunk("c", "some regulatory text", "2018-01-01")], system_from=T0)
    row = store.versions_of("c")[0]
    assert row["content_hash"] and len(row["content_hash"]) == 16
