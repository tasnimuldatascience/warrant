"""The eCFR client's caching policy, and the transactionality of an ingest.

Every failure covered here is silent by construction. A frozen index reports no new
amendments, which reads exactly like there being none; a pinned 404 removes the current text
of a part, which reads exactly like the part having no current text; a half-applied snapshot
closes sections with no replacement, which reads exactly like the law having been repealed.
None of them raises anything, so each test asserts the *absence* of the wrong state rather
than the presence of an exception.

The ingest tests live here rather than beside the other ingestion tests because they are
about the same property as the caching ones: a write that half-happened, like a cache entry
that is half-trusted, is worse than one that did not happen at all.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
import pytest

from warrant.corpus.build import build_part
from warrant.corpus.ecfr import ECFRClient, SnapshotUnavailable
from warrant.index.store import Store

TITLE = 5
FLOOR = "2017-01-01"
LATEST = "2026-08-26"
V1 = "https://www.ecfr.gov/api/versioner/v1"


def titles_json(latest: str = LATEST) -> bytes:
    return json.dumps({"titles": [{"number": TITLE, "latest_issue_date": latest}]}).encode()


#: One section, one paragraph. Enough to ingest; the parser is exercised in test_parse.
PART_XML = ('<ECFR><DIV5 N="630" TYPE="PART">'
            '<DIV8 N="630.101" TYPE="SECTION"><HEAD>&#167; 630.101 Purpose.</HEAD>'
            "<P>{text}</P></DIV8></DIV5></ECFR>")


def versions_json(*dates: str) -> bytes:
    """/versions is per section, not per snapshot: repeated dates are the normal shape."""
    rows = [{"date": d, "identifier": f"630.{i}"} for d in dates for i in (1, 2, 3)]
    return json.dumps({"content_versions": rows}).encode()


def base_responses() -> dict[str, bytes]:
    return {
        f"{V1}/titles.json": titles_json(),
        f"{V1}/versions/title-5.json?part=630": versions_json("2018-05-10", LATEST),
        f"{V1}/full/2018-05-10/title-5.xml?part=630": b"<a/>",
        f"{V1}/full/{LATEST}/title-5.xml?part=630": b"<b/>",
    }


@dataclass
class FakeTransport:
    """Serves a dict and records every URL, so a cache hit is provable rather than assumed."""

    responses: dict[str, bytes]
    calls: list[str]

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        body = self.responses.get(url)
        if body is None:
            raise SnapshotUnavailable(url)
        return body


def client(tmp_path, responses: dict[str, bytes], **kwargs) -> ECFRClient:
    """A client whose transport is a dict. ``delay_s=0``: no test should sleep a second."""
    c = ECFRClient(cache_dir=tmp_path, delay_s=0.0, **kwargs)
    c._fetch = c._fetch_once = FakeTransport(responses, [])  # type: ignore[method-assign]
    return c


def urls(c: ECFRClient) -> list[str]:
    return c._fetch.calls  # type: ignore[attr-defined]


def age_out(path, days: float = 10.0) -> None:
    """Backdate a cache entry. Deterministic where waiting for a TTL to elapse is not."""
    when = time.time() - days * 86400
    os.utime(path, (when, when))


# -- index freshness ---------------------------------------------------------------


def test_snapshots_are_cached_forever(tmp_path):
    """/full/ is immutable history: Part 630 as of 2018-05-10 will never change again."""
    c = client(tmp_path, base_responses())
    c.snapshot(TITLE, "630", "2018-05-10")
    age_out(tmp_path / "full-t5-p630-2018-05-10.xml", days=3650)
    c.snapshot(TITLE, "630", "2018-05-10")
    assert len(urls(c)) == 1


def test_the_index_is_refetched_once_its_ttl_expires(tmp_path):
    """The bug this exists for: /titles.json and /versions were cached with the same
    never-expire policy as the snapshots, so the two endpoints that report a new amendment
    could never report one and `make fetch` downloaded zero bytes, permanently."""
    c = client(tmp_path, base_responses(), index_ttl_hours=1.0)
    assert c.latest_issue_date(TITLE) == LATEST
    age_out(tmp_path / "titles.json")
    c._issue_dates.clear()  # the per-process memo; the disk cache is what is under test
    assert c.latest_issue_date(TITLE) == LATEST
    assert len(urls(c)) == 2


def test_a_fresh_index_is_not_refetched(tmp_path):
    c = client(tmp_path, base_responses(), index_ttl_hours=1.0)
    c.latest_issue_date(TITLE)
    c._issue_dates.clear()
    c.latest_issue_date(TITLE)
    assert len(urls(c)) == 1


def test_a_new_amendment_becomes_visible_once_the_index_expires(tmp_path):
    """End to end: a date published after the first fetch has to reach version_dates."""
    responses = base_responses()
    c = client(tmp_path, responses, index_ttl_hours=1.0)
    assert c.version_dates(TITLE, "630", floor=FLOOR) == ["2018-05-10", LATEST]

    responses[f"{V1}/versions/title-5.json?part=630"] = versions_json("2018-05-10", "2026-08-28")
    responses[f"{V1}/titles.json"] = titles_json("2026-08-28")
    age_out(tmp_path / "titles.json")
    age_out(tmp_path / "versions-t5-p630.json")
    c._issue_dates.clear()
    assert c.version_dates(TITLE, "630", floor=FLOOR) == ["2018-05-10", "2026-08-28"]


def test_refresh_forces_a_refetch_of_a_fresh_entry(tmp_path):
    """The --refresh switch: settle a suspected-stale cache by asking, not by deleting."""
    c = client(tmp_path, base_responses(), index_ttl_hours=1.0)
    c.latest_issue_date(TITLE)
    c.latest_issue_date(TITLE, refresh=True)
    assert len(urls(c)) == 2


def test_a_failed_refresh_falls_back_to_the_stale_copy(tmp_path):
    """An expired TTL is a reason to ask again, not a reason for an offline `make build` to
    stop working. The cached index is stale, not wrong."""
    c = client(tmp_path, base_responses(), index_ttl_hours=1.0)
    c.latest_issue_date(TITLE)
    age_out(tmp_path / "titles.json")

    def offline(url: str) -> bytes:
        raise httpx.ConnectError("no route to host")

    c._fetch = c._fetch_once = offline  # type: ignore[method-assign]
    c._issue_dates.clear()
    assert c.latest_issue_date(TITLE) == LATEST


# -- negative caching --------------------------------------------------------------


def test_a_404_near_the_issue_date_is_not_recorded(tmp_path):
    """The 26 markers in data/ecfr were exactly this: one per part, all for a date that 404s
    only because eCFR's issue date lags the calendar."""
    c = client(tmp_path, base_responses())
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2026-08-30")
    assert not list(tmp_path.glob("*.404"))


def test_the_text_appears_as_soon_as_the_date_becomes_real(tmp_path):
    """The consequence of the test above, stated as the behaviour that actually matters:
    with the 404 pinned, the current text of the part is absent from the corpus for good."""
    responses = base_responses()
    c = client(tmp_path, responses)
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2026-08-30")
    responses[f"{V1}/full/2026-08-30/title-5.xml?part=630"] = b"<published/>"
    assert c.snapshot(TITLE, "630", "2026-08-30") == b"<published/>"


def test_an_old_404_is_recorded_and_reused(tmp_path):
    """Well behind the issue date, a 404 is absence. Re-probing a permanent 404 on every
    rebuild is a slow way to be rude to someone else's server."""
    c = client(tmp_path, base_responses())
    for _ in range(2):
        with pytest.raises(SnapshotUnavailable):
            c.snapshot(TITLE, "630", "2019-01-01")
    # one titles.json, one probe; the second call never left the marker
    assert len(urls(c)) == 2
    assert (tmp_path / "full-t5-p630-2019-01-01.xml.404").exists()


def test_a_recorded_404_carries_a_timestamp(tmp_path):
    """An undated marker cannot expire, which is how a transient 404 becomes permanent."""
    c = client(tmp_path, base_responses())
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2019-01-01")
    stamp = (tmp_path / "full-t5-p630-2019-01-01.xml.404").read_text(encoding="utf-8")
    assert abs(datetime.fromisoformat(stamp).timestamp() - time.time()) < 60


def test_an_expired_404_marker_is_probed_again(tmp_path):
    c = client(tmp_path, base_responses())
    marker = tmp_path / "full-t5-p630-2019-01-01.xml.404"
    marker.write_text("2020-01-01T00:00:00+00:00", encoding="utf-8")
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2019-01-01")
    assert len(urls(c)) == 2  # titles.json, then the re-probe


def test_an_unstamped_legacy_marker_is_stamped_from_its_mtime(tmp_path):
    """The markers already on disk are empty files. Stamping them from *now* would hand
    every one of them a fresh 30-day lease each time it was read, which is the never-expire
    behaviour again wearing a timestamp."""
    marker = tmp_path / "full-t5-p630-2019-01-01.xml.404"
    marker.write_bytes(b"")
    when = time.time() - 10 * 86400
    os.utime(marker, (when, when))

    c = client(tmp_path, base_responses())
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2019-01-01")
    stamped = datetime.fromisoformat(marker.read_text(encoding="utf-8")).timestamp()
    assert abs(stamped - when) < 2
    assert len(urls(c)) == 1  # 10 days old, inside the window: trusted, not re-probed


def test_a_stamped_legacy_marker_still_expires(tmp_path):
    marker = tmp_path / "full-t5-p630-2019-01-01.xml.404"
    marker.write_bytes(b"")
    when = time.time() - 400 * 86400
    os.utime(marker, (when, when))

    c = client(tmp_path, base_responses())
    with pytest.raises(SnapshotUnavailable):
        c.snapshot(TITLE, "630", "2019-01-01")
    assert len(urls(c)) == 2


# -- skipped dates -----------------------------------------------------------------


def test_snapshots_counts_the_dates_it_could_not_retrieve(tmp_path):
    """A bare `continue` here is how the current text of a part goes missing with no error
    anywhere -- the failure ARCHITECTURE.md section 2 exists to prevent."""
    responses = base_responses()
    del responses[f"{V1}/full/{LATEST}/title-5.xml?part=630"]
    c = client(tmp_path, responses)
    assert [d for d, _ in c.snapshots(TITLE, "630", floor=FLOOR)] == ["2018-05-10"]
    assert c.skipped_dates(TITLE, "630") == [LATEST]


def test_nothing_is_reported_skipped_when_everything_was_retrieved(tmp_path):
    c = client(tmp_path, base_responses())
    assert len(list(c.snapshots(TITLE, "630", floor=FLOOR))) == 2
    assert c.skipped_dates(TITLE, "630") == []


def test_build_stats_carry_the_skipped_dates(tmp_path):
    responses = base_responses()
    del responses[f"{V1}/full/{LATEST}/title-5.xml?part=630"]
    responses[f"{V1}/full/2018-05-10/title-5.xml?part=630"] = PART_XML.format(text="one").encode()
    c = client(tmp_path, responses)
    with Store(":memory:") as store:
        stats = build_part(store, c, title=TITLE, part="630", floor=FLOOR, config_hash="t")
    assert stats.snapshots == 1
    assert stats.snapshots_skipped == 1
    assert stats.skipped_dates == [LATEST]


# -- transactional ingest ----------------------------------------------------------


@dataclass
class CannedClient:
    """Serves canned snapshots, like the ingestion fixtures: no network, no cache."""

    canned: dict[str, str]

    def snapshots(self, title: int, part: str, *, floor: str = FLOOR):
        for date in sorted(self.canned):
            yield date, PART_XML.format(text=self.canned[date]).encode()


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def failing_add_after(store: Store, n: int) -> Store:
    """Make the ``n``th insert raise, standing in for a crash mid-snapshot."""
    real_add = store.add
    calls = {"n": 0}

    def flaky_add(chunks, *, system_from=None):
        calls["n"] += 1
        if calls["n"] == n:
            raise RuntimeError("crash between the closures and the insert")
        return real_add(chunks, system_from=system_from)

    store.add = flaky_add  # type: ignore[method-assign]
    return store


def test_a_failure_mid_snapshot_leaves_the_store_unchanged(store: Store):
    """The window this closes: sections are closed one statement at a time and the
    replacement text is inserted afterwards. Crash in between and the store believes the law
    stopped existing on that date -- `as_of` returns nothing, and the one-version-in-force
    invariant still passes, because zero is not two."""
    failing_add_after(store, 2)
    canned = CannedClient({"2018-05-10": "twelve workweeks",
                           "2020-08-10": "twelve workweeks of paid leave"})
    with pytest.raises(RuntimeError):
        build_part(store, canned, title=5, part="630", floor=FLOOR, config_hash="t")

    versions = store.versions_of("630.101")
    assert len(versions) == 1, "the failed snapshot must not have inserted anything"
    assert versions[0]["valid_to"] is None, "and its closure must have rolled back with it"
    assert [r["text"] for r in store.as_of("2021-01-01")] == ["twelve workweeks"]


def test_a_completed_snapshot_survives_a_later_failure(store: Store):
    """Atomic per snapshot, not per part: a build that dies on the third snapshot keeps the
    two that landed, so re-running resumes rather than starting over."""
    failing_add_after(store, 3)
    canned = CannedClient({"2018-05-10": "one", "2019-06-01": "two", "2020-08-10": "three"})
    with pytest.raises(RuntimeError):
        build_part(store, canned, title=5, part="630", floor=FLOOR, config_hash="t")

    assert [r["valid_from"] for r in store.versions_of("630.101")] == [FLOOR, "2019-06-01"]
    assert [r["text"] for r in store.as_of("2021-01-01")] == ["two"]


def test_the_store_is_usable_again_after_a_failed_snapshot(store: Store):
    """The snapshot transaction replaces the store's transaction context for the duration of
    one date. Not restoring it on the way out would leave every later write uncommitted --
    a larger version of the bug it exists to fix."""
    failing_add_after(store, 1)
    with pytest.raises(RuntimeError):
        build_part(store, CannedClient({"2018-05-10": "one"}), title=5, part="630",
                   floor=FLOOR, config_hash="t")
    assert store.count() == 0
    del store.add  # drop the failure injection; the store itself must be untouched
    build_part(store, CannedClient({"2018-05-10": "one"}), title=5, part="630",
               floor=FLOOR, config_hash="t")
    assert store.count() == 1
