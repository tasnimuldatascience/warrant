"""Ingesting snapshots into bitemporal versions.

The two rules worth protecting are the dating rules. Both were wrong in the first working
version, and both fail silently rather than loudly: the corpus simply has no evidence for
some dates, and a question about those dates gets an honest "I do not know" that is in fact
a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from warrant.corpus.build import build_part, section_chunks
from warrant.corpus.parse import Paragraph, Section
from warrant.index.store import Store

FLOOR = "2017-01-01"


def part_xml(sections: dict[str, str]) -> bytes:
    body = "".join(
        f'<DIV8 N="{ident}" TYPE="SECTION"><HEAD>&#167; {ident} Heading.</HEAD>'
        f"<P>{text}</P></DIV8>"
        for ident, text in sections.items()
    )
    return f'<ECFR><DIV5 N="630" TYPE="PART">{body}</DIV5></ECFR>'.encode()


@dataclass
class FakeClient:
    """Serves canned snapshots. Ingestion logic is worth testing without the network."""

    canned: dict[str, dict[str, str]]

    def snapshots(self, title: int, part: str, *, floor: str = FLOOR):
        for date in sorted(self.canned):
            yield date, part_xml(self.canned[date])


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def build(store: Store, canned: dict[str, dict[str, str]]):
    return build_part(store, FakeClient(canned), title=5, part="630",
                      floor=FLOOR, config_hash="test")


def test_unchanged_section_keeps_one_open_version(store: Store):
    """Re-observing identical text must not create a second version. Getting this wrong
    inflates the corpus and puts two identical candidates in every result list."""
    build(store, {
        "2018-05-10": {"630.101": "the text"},
        "2020-08-10": {"630.101": "the text"},
    })
    assert len(store.versions_of("630.101")) == 1
    assert store.versions_of("630.101")[0]["valid_to"] is None


def test_changed_section_closes_the_old_version_and_opens_a_new_one(store: Store):
    build(store, {
        "2018-05-10": {"630.101": "twelve workweeks"},
        "2020-08-10": {"630.101": "twelve workweeks of paid leave"},
    })
    versions = store.versions_of("630.101")
    assert len(versions) == 2
    assert versions[0]["valid_to"] == "2020-08-10"
    assert versions[1]["valid_from"] == "2020-08-10"
    assert versions[1]["valid_to"] is None


def test_first_snapshot_is_dated_from_the_history_floor(store: Store):
    """eCFR records a version date whenever a part is amended, so no version date between
    the floor and the first snapshot is positive evidence the text did not change. Dating
    from the snapshot instead would make every earlier date silently unanswerable."""
    build(store, {"2020-10-16": {"315.803": "probationary period rules"}})
    assert store.versions_of("315.803")[0]["valid_from"] == FLOOR
    assert len(store.as_of("2018-06-01")) == 1


def test_sections_appearing_later_are_not_backfilled(store: Store):
    """A section that first appears in a later snapshot did not exist before. Backfilling it
    would invent law that was never in force."""
    build(store, {
        "2018-05-10": {"630.101": "original"},
        "2020-08-10": {"630.101": "original", "630.102": "brand new section"},
    })
    assert store.versions_of("630.101")[0]["valid_from"] == FLOOR
    assert store.versions_of("630.102")[0]["valid_from"] == "2020-08-10"
    assert not [r for r in store.as_of("2019-01-01") if r["section_id"] == "630.102"]


def test_removed_section_is_closed_not_deleted(store: Store):
    build(store, {
        "2018-05-10": {"630.101": "a", "630.102": "b"},
        "2020-08-10": {"630.101": "a"},
    })
    assert [r["valid_to"] for r in store.versions_of("630.102")] == ["2020-08-10"]
    assert not [r for r in store.as_of("2021-01-01") if r["section_id"] == "630.102"]
    assert [r for r in store.as_of("2019-01-01") if r["section_id"] == "630.102"]


def test_reappearing_rule_is_answerable_at_every_date(store: Store):
    """The shape of the strongest benchmark item found in the corpus: a requirement added in
    one amendment and removed in a later one has three different correct answers."""
    build(store, {
        "2020-10-16": {"315.803": "the agency shall use the probationary period"},
        "2020-11-16": {"315.803": "the agency shall use the probationary period and must "
                                  "notify supervisors three months prior"},
        "2022-12-12": {"315.803": "the agency shall use the probationary period"},
    })
    def has_rule(date: str) -> bool:
        return any("three months prior" in r["text"] for r in store.as_of(date))

    assert not has_rule("2018-01-01")
    assert has_rule("2021-06-01")
    assert not has_rule("2023-06-01")


def test_section_without_paragraphs_still_yields_a_chunk():
    """A section that parses to no paragraphs must not vanish. A silent corpus hole is the
    hardest kind of retrieval failure to explain after the fact."""
    section = Section(identifier="630.101", heading="Purpose", text="whole section text",
                      paragraphs=[])
    chunks = section_chunks(section, title=5, part="630", valid_from=FLOOR,
                            snapshot=FLOOR, config_hash="t")
    assert [c.chunk_id for c in chunks] == ["630.101#full"]
    assert chunks[0].text == "whole section text"


def test_paragraph_chunks_carry_anchors_and_share_the_section_interval():
    section = Section(identifier="630.1203", heading="Leave entitlement", text="full",
                      paragraphs=[Paragraph("a", "(a) first"), Paragraph("b-2", "(b)(2) second")])
    chunks = section_chunks(section, title=5, part="630", valid_from="2020-08-10",
                            snapshot="2020-08-10", config_hash="t")
    assert [c.chunk_id for c in chunks] == ["630.1203#a", "630.1203#b-2"]
    assert {c.valid_from for c in chunks} == {"2020-08-10"}


def test_build_stats_report_what_happened(store: Store):
    stats = build(store, {
        "2018-05-10": {"630.101": "a", "630.102": "b"},
        "2020-08-10": {"630.101": "a changed", "630.102": "b"},
    })
    assert stats.snapshots == 2
    assert stats.versions_inserted == 3   # two at first snapshot, one amendment
    assert stats.unchanged == 1
    assert stats.sections_closed == 1
