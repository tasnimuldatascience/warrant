"""Mining the temporal benchmark, and the property that makes it worth reporting.

The load-bearing test is `test_query_uses_only_shared_wording`. If a query leaks wording
unique to one version, lexical matching alone can pick the right version and the bucket
silently stops measuring temporal reasoning while still producing a plausible number.
"""

from __future__ import annotations

import pytest

from warrant.eval.bench import TemporalItem, mine, sample_date, shared_query
from warrant.index.store import Chunk, Store

T0 = "2026-01-01T00:00:00+00:00"
HORIZON = "2026-08-26"


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def add(store: Store, anchor: str, text: str, valid_from: str, valid_to: str | None):
    store.add([Chunk(chunk_id=f"315.803#{anchor}", section_id="315.803", title=5, part="315",
                     anchor=anchor, heading="Agency action during probationary period",
                     text=text, valid_from=valid_from, valid_to=valid_to,
                     source_snapshot=valid_from, config_hash="t")], system_from=T0)


OLD = ("(a) The agency shall utilize the probationary period as fully as possible to "
       "determine the fitness of the employee for continued employment.")
NEW = ("(a) The agency shall utilize the probationary period as fully as possible to "
       "determine the fitness of the employee for continued employment. The agency must "
       "notify supervisors three months prior to expiration of the probationary period.")


def test_amendment_yields_one_item_on_each_side(store: Store):
    add(store, "a", OLD, "2017-01-01", "2020-11-16")
    add(store, "a", NEW, "2020-11-16", None)
    items = mine(store, horizon=HORIZON)
    assert {i.provenance["side"] for i in items} == {"before", "after"}
    assert len(items) == 2


def test_the_two_sides_share_a_query_but_differ_in_date_and_evidence(store: Store):
    """A system that ignores the as-of date must get one of the pair wrong. That is the
    entire reason the bucket is reported on its own."""
    add(store, "a", OLD, "2017-01-01", "2020-11-16")
    add(store, "a", NEW, "2020-11-16", None)
    before, after = sorted(mine(store, horizon=HORIZON), key=lambda i: i.as_of)
    assert before.query == after.query
    assert before.as_of < "2020-11-16" < after.as_of
    assert before.acceptable_evidence != after.acceptable_evidence
    assert before.acceptable_evidence[0] == after.distractors
    assert after.acceptable_evidence[0] == before.distractors


def test_evidence_is_version_qualified(store: Store):
    """chunk_id alone repeats across versions, so evidence and distractor would be the same
    string and the benchmark would grade every answer correct."""
    add(store, "a", OLD, "2017-01-01", "2020-11-16")
    add(store, "a", NEW, "2020-11-16", None)
    for item in mine(store, horizon=HORIZON):
        assert "@" in item.acceptable_evidence[0][0]
        assert set(item.acceptable_evidence[0]).isdisjoint(item.distractors)


def test_query_uses_only_shared_wording():
    q = shared_query(OLD, NEW, "Agency action")
    assert "notify" not in q.lower(), "wording unique to the new version leaks the answer"
    assert "supervisors" not in q.lower()
    assert "probationary" in q.lower()


def test_query_drops_function_words():
    q = shared_query("the agency shall determine fitness", "the agency shall determine "
                     "fitness and notify", "Heading")
    assert " the " not in f" {q.lower()} "
    assert "agency" in q.lower()


def test_unchanged_section_produces_no_items(store: Store):
    add(store, "a", OLD, "2017-01-01", None)
    assert mine(store, horizon=HORIZON) == []


def test_tiny_change_produces_no_items(store: Store):
    """Below the substantive threshold there is nothing a question could ask about."""
    add(store, "a", OLD, "2017-01-01", "2020-11-16")
    add(store, "a", OLD.replace("fitness", "suitability"), "2020-11-16", None)
    assert mine(store, horizon=HORIZON) == []


def test_pure_addition_produces_no_items(store: Store):
    """A paragraph with no counterpart in the other version cannot test discrimination:
    there is no wrong version for the retriever to prefer."""
    add(store, "a", OLD, "2017-01-01", "2020-11-16")
    add(store, "a", OLD, "2020-11-16", None)
    add(store, "b", "(b) A brand new paragraph with entirely different content here.",
        "2020-11-16", None)
    assert mine(store, horizon=HORIZON) == []


def test_sample_date_stays_clear_of_boundaries():
    d = sample_date("2020-01-01", "2021-01-01", horizon=HORIZON)
    assert "2020-01-15" < d < "2020-12-18"


def test_sample_date_declines_intervals_that_are_too_short():
    """A question dated one day after an amendment asks about snapshot bookkeeping."""
    assert sample_date("2020-01-01", "2020-01-10", horizon=HORIZON) is None


def test_open_interval_is_sampled_up_to_the_horizon():
    d = sample_date("2024-01-01", None, horizon=HORIZON)
    assert d is not None and "2024-01-01" < d < HORIZON


def test_is_satisfied_by_accepts_any_complete_set():
    item = TemporalItem(id="x", query="q", as_of="2021-01-01", section_id="s", part="p",
                        heading="h", acceptable_evidence=[["a@1", "b@1"], ["c@1"]],
                        distractors=["a@2"])
    assert item.is_satisfied_by(["c@1", "z@1"])
    assert item.is_satisfied_by(["a@1", "b@1"])
    assert not item.is_satisfied_by(["a@1"]), "a partial set is not sufficient"


def test_leaked_reports_the_superseded_version():
    item = TemporalItem(id="x", query="q", as_of="2021-01-01", section_id="s", part="p",
                        heading="h", acceptable_evidence=[["a@1"]], distractors=["a@2"])
    assert item.leaked(["a@1", "a@2"]) == ["a@2"]
    assert item.leaked(["a@1"]) == []
