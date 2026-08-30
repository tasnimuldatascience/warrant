"""The deterministic correctness gates from ARCHITECTURE.md section 9.

These are assertions about the system, not tests of a function, and CI runs them as a
separate job so a failure reads as "the system is wrong" rather than "a test broke". They
move correctness out of probabilistic evaluation: a dated query returning two versions of one
section is a bug catchable before any model runs, with no ground truth and no threshold to
argue about.

Every invariant is written once and run against **two** stores: a synthetic one built in
memory, so the gate is real on a fresh clone with no corpus and no network, and the actual
corpus when one has been built. A CI job that skips its whole suite in the absence of data is
worse than no job at all, because it still reports green.
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

SYNTHETIC = [
    # Two versions of one paragraph, so the one-version-in-force gate has something to catch.
    Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
          heading="Restored annual leave",
          text="annual leave restored must be scheduled within two years",
          valid_from="2017-01-01", valid_to="2020-08-10"),
    Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
          heading="Restored annual leave",
          text="annual leave restored must be scheduled within three years",
          valid_from="2020-08-10"),
    # A second paragraph of the same section, so "one version" is not confused with "one row".
    Chunk(chunk_id="630.306#b", section_id="630.306", title=5, part="630", anchor="b",
          heading="Restored annual leave",
          text="restored leave is forfeited if not scheduled in time",
          valid_from="2017-01-01", valid_to="2020-08-10"),
    Chunk(chunk_id="630.306#b", section_id="630.306", title=5, part="630", anchor="b",
          heading="Restored annual leave",
          text="restored leave is forfeited when the time limit expires",
          valid_from="2020-08-10"),
    # Restricted parts, so the applicability gate has something to exclude.
    Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531", anchor="a",
          heading="Within-grade increase",
          text="performance must be at an acceptable level of competence for a pay increase",
          valid_from="2017-01-01"),
    Chunk(chunk_id="532.203#a", section_id="532.203", title=5, part="532", anchor="a",
          heading="Structure of regular wage schedules",
          text="each nonsupervisory wage schedule has five steps and a pay rate range",
          valid_from="2017-01-01"),
]

PARTS = ["531", "532", "630"]


@pytest.fixture(scope="module")
def synthetic() -> Store:
    with Store(":memory:") as s:
        s.add(SYNTHETIC, system_from=T0)
        yield s


@pytest.fixture(scope="module")
def corpus() -> Store:
    cfg = Config.load()
    if not cfg.store_path.exists():
        pytest.skip("no corpus built; run `make fetch && make build`")
    with Store(cfg.store_path) as s:
        yield s


@pytest.fixture(params=["synthetic", "corpus"])
def store(request) -> Store:
    """Each invariant runs against both. The synthetic case keeps CI honest offline."""
    return request.getfixturevalue(request.param)


def parts_of(store: Store) -> list[str]:
    return sorted({r["part"] for r in store.db.execute(
        "SELECT DISTINCT part FROM chunk WHERE system_to IS NULL")})


# -- 1. at most one version of a section in force on any date ---------------------


def test_one_version_of_a_section_is_in_force_at_a_time(store: Store):
    """A section has many paragraphs, so this counts *versions*, not rows. Two versions of
    630.306 reaching one prompt is a filter bug, detectable before the model runs."""
    for date in DATES:
        rows = store.as_of(date)
        versions: dict[str, set[str]] = {}
        for r in rows:
            versions.setdefault(r["section_id"], set()).add(r["valid_from"])
        offenders = {s: sorted(v) for s, v in versions.items() if len(v) > 1}
        assert not offenders, f"{date}: two versions in force at once {sorted(offenders)[:5]}"

        dupes = {c: n for c, n in Counter(r["chunk_id"] for r in rows).items() if n > 1}
        assert not dupes, f"{date}: paragraph address in force twice {sorted(dupes)[:5]}"


# -- 2. every retrieved chunk is in force on the as-of date -----------------------


def test_retrieved_chunks_are_in_force_at_the_as_of_date(store: Store):
    r = Retriever(store=store, candidates_lexical=50, rerank_top_k=20, final_k=10,
                  parts_universe=parts_of(store))
    rows = {row["version_id"]: row for row in store.db.execute(
        "SELECT version_id, valid_from, valid_to FROM chunk WHERE system_to IS NULL")}
    for date in DATES:
        for vid in r.retrieve("annual leave restored scheduled", as_of=date).final:
            row = rows[vid]
            assert row["valid_from"] <= date, f"{vid} not yet in force on {date}"
            assert row["valid_to"] is None or row["valid_to"] > date, \
                f"{vid} was superseded before {date}"


# -- 3. every retrieved chunk is applicable to the resolved scope -----------------


def test_retrieved_chunks_are_applicable_to_the_scope(store: Store):
    universe = parts_of(store)
    r = Retriever(store=store, candidates_lexical=50, rerank_top_k=20, final_k=10,
                  parts_universe=universe)
    parts = {row["version_id"]: row["part"] for row in store.db.execute(
        "SELECT version_id, part FROM chunk WHERE system_to IS NULL")}
    for scope in (Scope.of(pay_system="GS"), Scope.of(pay_system="FWS"),
                  Scope.of(service="competitive")):
        trace = r.retrieve("pay rate schedule increase leave", as_of="2024-06-01",
                           scope=scope)
        for vid in trace.final:
            assert scope.governs(parts[vid]), \
                f"{vid} is in part {parts[vid]}, which does not govern {scope.describe()}"


# -- 4. every citation address is unambiguous -------------------------------------


def test_citation_addresses_are_unambiguous(store: Store):
    """A citation matching more than one paragraph is not a citation. 13% of addresses were
    ambiguous before paragraph designators were tracked hierarchically."""
    total = store.db.execute(
        "SELECT COUNT(*) n FROM chunk WHERE system_to IS NULL").fetchone()["n"]
    distinct = store.db.execute(
        "SELECT COUNT(DISTINCT version_id) n FROM chunk WHERE system_to IS NULL"
    ).fetchone()["n"]
    assert total == distinct, f"{total - distinct} ambiguous citation addresses"


# -- 5. validity intervals never overlap ------------------------------------------


def test_validity_intervals_do_not_overlap(store: Store):
    """Overlapping intervals would make invariant 1 unenforceable however the query is
    written, because two versions would genuinely both be in force."""
    rows = store.db.execute(
        "SELECT chunk_id, valid_from, valid_to FROM chunk WHERE system_to IS NULL "
        "ORDER BY chunk_id, valid_from").fetchall()
    previous: dict[str, str | None] = {}
    for row in rows:
        if row["chunk_id"] in previous:
            prior_to = previous[row["chunk_id"]]
            assert prior_to is not None, \
                f"{row['chunk_id']} has an open interval followed by another version"
            assert prior_to <= row["valid_from"], \
                f"{row['chunk_id']} intervals overlap at {row['valid_from']}"
        previous[row["chunk_id"]] = row["valid_to"]


# -- 6. the chunker captures the section it claims to have ingested ---------------


CORPUS_COVERAGE_FLOOR = 0.95


def test_chunking_captures_almost_all_of_each_section():
    """The chunker must not silently drop body text.

    This is the invariant the failure budget cannot provide. Its ``ingestion`` row asks
    whether a gold chunk is in the store, and gold chunks are minted by the same parser --
    so text the parser never emitted can never be missed, and the row reads zero no matter
    how much was lost. Reading only ``<P>`` dropped 18,705 words (4.5% of the corpus),
    including 88% of §532.313, and every instrument in the repository reported clean.
    """
    import glob
    import re

    from lxml import etree

    from warrant.corpus.parse import parse_sections

    files = sorted(glob.glob("data/ecfr/full-t5-p*.xml"), reverse=True)
    if not files:
        pytest.skip("no snapshots cached; run `make fetch`")

    ws = re.compile(r"\s+")
    seen: set[str] = set()
    captured = missing = 0
    for path in files:
        part = path.split("-p")[1].split("-")[0]
        if part in seen:
            continue
        seen.add(part)
        raw = open(path, "rb").read()
        sections = {s.identifier: s for s in parse_sections(raw)}
        for div in etree.fromstring(raw).iter("DIV8"):
            if div.get("TYPE") != "SECTION" or div.get("N") not in sections:
                continue
            got = sum(len(p.text.split()) for p in sections[div.get("N")].paragraphs)
            total = len(ws.sub(" ", "".join(div.itertext())).split())
            head = div.find("HEAD")
            heading = len(ws.sub(" ", "".join(head.itertext())).split()) if head is not None else 0
            captured += got
            missing += max(total - got - heading, 0)

    coverage = captured / (captured + missing)
    assert coverage >= CORPUS_COVERAGE_FLOOR, (
        f"chunker captured {coverage:.1%} of section body text "
        f"({missing:,} words dropped); floor is {CORPUS_COVERAGE_FLOOR:.0%}")
