"""Client for the eCFR versioner API.

Three endpoints carry the entire corpus:

    /titles.json                        the latest issue date of each title
    /versions/title-{t}.json?part={p}   section-level version records
    /full/{date}/title-{t}.xml?part={p} the part as it stood on {date}

Two things about this API are easy to get wrong, and both were found by running it rather
than by reading the docs:

1. ``/versions`` returns one row per *section* version, not per snapshot. Part 630 returns
   226 rows that collapse to 8 distinct dates. Counting rows overstates the amount of
   diffable history by more than an order of magnitude.

2. Every part reports a first version date of 2016-12-27, and ``/full/`` returns 404 for it.
   The usable point-in-time window starts in 2017.

Responses are cached on disk keyed by request, and the caching policy is **not** uniform,
because the endpoints are not the same kind of thing:

  ``/full/``  is immutable history. Part 630 as of 2019-06-01 will never change, so a
              rebuild costs no requests and the cache never expires.
  the indexes are the opposite: ``/titles.json`` and ``/versions`` are exactly where a new
              amendment first appears. Caching them forever freezes the corpus -- the next
              fetch re-reads the pinned files, finds no new dates, and downloads nothing,
              for good. They therefore carry a TTL (default 24 h) and honour ``refresh``.

Negative caching is asymmetric for the same reason. A 404 from ``/full/`` for a date close
to the title's latest issue date is eCFR's publication lag, not an absent snapshot, and
recording it permanently drops the current text of that part from the corpus with no error
anywhere -- the failure ARCHITECTURE.md section 2 exists to prevent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

API = "https://www.ecfr.gov/api/versioner/v1"
USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

#: /full/ 404s at the 2016-12-27 floor every part advertises.
HISTORY_FLOOR = "2017-01-01"

#: How long a cached *index* response stays authoritative. One day: eCFR publishes at most
#: daily, and a stale index is invisible -- it reports no new amendments, which is
#: indistinguishable from there being none.
INDEX_TTL_HOURS = 24.0

#: How long a recorded 404 is believed. Absences are re-probed eventually because "the API
#: has no text for this date" has been temporary before.
NEGATIVE_CACHE_TTL_DAYS = 30.0

#: A 404 for a date within this many days of the title's latest issue date is publication
#: lag rather than absence, and is never written to the negative cache.
ISSUE_DATE_LAG_DAYS = 30


class SnapshotUnavailable(LookupError):
    """The API has a version record for this date but no retrievable text."""


def _now_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _days_between(later: str, earlier: str) -> int | None:
    try:
        return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).days
    except ValueError:
        return None


@dataclass
class ECFRClient:
    cache_dir: Path
    delay_s: float = 1.0
    timeout_s: float = 180.0
    #: TTL for the index endpoints only; ``/full/`` snapshots are never expired. Named and
    #: scaled exactly as the ``corpus`` config fields are, so wiring the two together cannot
    #: quietly pass hours where seconds were meant.
    index_ttl_hours: float = INDEX_TTL_HOURS
    negative_cache_ttl_days: float = NEGATIVE_CACHE_TTL_DAYS
    issue_date_lag_days: int = ISSUE_DATE_LAG_DAYS
    #: Force every request past the cache. This is the ``--refresh`` switch: it exists so a
    #: suspected-stale cache can be settled by re-fetching rather than by deleting files.
    refresh: bool = False
    _last_request: float = 0.0
    #: Snapshot dates that had no retrievable text, per ``t{title}-p{part}``. Read back with
    #: ``skipped_dates``; a skip that only appears in a log line is a skip nobody counts.
    skipped: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _issue_dates: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- transport ---------------------------------------------------------------

    def _fetch_once(self, url: str) -> bytes:
        """One request, no retries. Used when a stale cache entry is standing by.

        Spending the full backoff before falling back would cost 14 s per index file, and an
        offline rebuild reads 27 of them.
        """
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)  # a public API; do not hammer it
        r = httpx.get(url, headers={"User-Agent": USER_AGENT},
                      timeout=self.timeout_s, follow_redirects=True)
        self._last_request = time.monotonic()
        if r.status_code == 404:
            raise SnapshotUnavailable(url)
        r.raise_for_status()
        return r.content

    @retry(
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError, SnapshotUnavailable)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
        """``_fetch_once`` with backoff.

        ``SnapshotUnavailable`` is in the retry set deliberately. It used to be raised before
        ``raise_for_status`` and was the *only* failure the client did not retry, so a single
        transient 404 -- one served while a snapshot is being published -- became final, and
        the caller then wrote a marker that made it permanent.
        """
        return self._fetch_once(url)

    # -- cache -------------------------------------------------------------------

    @property
    def _index_ttl_s(self) -> float:
        return self.index_ttl_hours * 3600.0

    @property
    def _negative_ttl_s(self) -> float:
        return self.negative_cache_ttl_days * 86400.0

    @staticmethod
    def _age_s(path: Path) -> float:
        return max(time.time() - path.stat().st_mtime, 0.0)

    def _expired(self, path: Path, ttl_s: float | None) -> bool:
        """``ttl_s is None`` means immutable: that is the ``/full/`` snapshot policy."""
        return ttl_s is not None and self._age_s(path) > ttl_s

    def _absence_expired(self, marker: Path) -> bool:
        """Age of a negative-cache entry, stamping it if it predates stamping.

        The first markers written by this client were empty files, so the only evidence of
        when they were recorded is the mtime. They are rewritten with an explicit timestamp
        on first sight; the mtime is what that timestamp is taken from, so nothing is
        silently given a fresh lease by being read.
        """
        stamp = ""
        try:
            stamp = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return True
        recorded = None
        if stamp:
            try:
                recorded = datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                recorded = None
        if recorded is None:
            recorded = marker.stat().st_mtime
            marker.write_text(datetime.fromtimestamp(recorded, UTC).isoformat(timespec="seconds"),
                              encoding="utf-8")
        return (time.time() - recorded) > self._negative_ttl_s

    def _cached(self, url: str, name: str, *, negative_cache: bool | Callable[[], bool] = True,
                ttl_s: float | None = None, refresh: bool = False) -> bytes:
        """Fetch through the on-disk cache.

        Writes are atomic. A non-atomic ``write_bytes`` leaves a truncated file if the
        process dies mid-write -- a Ctrl-C during the ten-minute fetch, a closed lid, an OOM
        -- and the cache only checks for existence, so that truncation is trusted forever.
        The symptom is an XMLSyntaxError partway through a later build, naming no file.

        ``ttl_s`` re-fetches an entry older than the TTL; ``None`` never expires it.
        ``negative_cache=False`` refuses to persist a 404, for the index endpoints and for
        snapshot dates inside eCFR's publication lag. It may be a callable, because deciding
        which of those a date is costs a request of its own, and a cache hit must not pay it:
        an offline rebuild reads 200 cached snapshots and should make no requests at all.

        A failed refresh falls back to the stale copy rather than raising. An expired TTL is
        a reason to ask again, not a reason for an offline ``make build`` to stop working.
        """
        path = self.cache_dir / name
        force = refresh or self.refresh
        cached = path.read_bytes() if path.exists() else None
        if cached is not None and not force and not self._expired(path, ttl_s):
            log.debug("cache hit %s (%.0fs old)", name, self._age_s(path))
            return cached
        if cached is not None:
            log.info("refreshing %s (%s)", name, "forced" if force else "stale")

        def may_record() -> bool:
            return negative_cache() if callable(negative_cache) else negative_cache

        missing = self.cache_dir / (name + ".404")
        if cached is None and not force and missing.exists() and may_record():
            if not self._absence_expired(missing):
                log.debug("negative cache hit %s", name)
                raise SnapshotUnavailable(url)
            log.info("negative cache entry for %s has expired; re-probing", name)
            missing.unlink(missing_ok=True)

        log.info("GET %s", url)
        try:
            content = self._fetch_once(url) if cached is not None else self._fetch(url)
        except SnapshotUnavailable:
            if cached is not None:
                log.warning("%s is no longer served; keeping the cached copy", name)
                return cached
            if may_record():
                missing.write_text(_now_stamp(), encoding="utf-8")
                log.info("no text at %s; absence recorded for %.0f days",
                         url, self.negative_cache_ttl_days)
            else:
                log.info("no text at %s; not recorded (issue-date lag, not absence)", url)
            raise
        except httpx.HTTPError as exc:
            if cached is not None:
                log.warning("refresh of %s failed (%s); using the cached copy", name, exc)
                return cached
            raise
        missing.unlink(missing_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        log.info("cached %s (%d bytes)", name, len(content))
        return content

    # -- corpus ------------------------------------------------------------------

    def latest_issue_date(self, title: int, *, refresh: bool = False) -> str:
        """The most recent date for which the API will serve this title.

        Not today. ``/full/`` returns 404 for any date after the title issue date, and the
        gap is days wide -- 2026-08-30 was rejected while 2026-08-26 was served. Reading the
        API statement of its own currency keeps ingestion deterministic instead of dependent
        on the wall clock and on how far behind eCFR happens to be.

        This is one of the two endpoints that report new amendments, so it is cached with a
        TTL rather than pinned. Memoised per process as well: it is consulted once per
        snapshot probe, and the answer cannot change mid-run without making a build
        internally inconsistent.
        """
        force = refresh or self.refresh
        if not force and title in self._issue_dates:
            return self._issue_dates[title]
        raw = self._cached(f"{API}/titles.json", "titles.json", negative_cache=False,
                           ttl_s=self._index_ttl_s, refresh=refresh)
        for row in json.loads(raw).get("titles", []):
            if row.get("number") == title:
                self._issue_dates[title] = row["latest_issue_date"]
                return self._issue_dates[title]
        raise LookupError(f"title {title} not listed by the eCFR API")

    def version_dates(self, title: int, part: str, *, floor: str = HISTORY_FLOOR,
                      include_current: bool = True, refresh: bool = False) -> list[str]:
        """Distinct snapshot dates for a part, oldest first.

        Distinct *dates* -- not the row count, which is per-section and much larger.

        ``include_current`` appends the title latest issue date, and it is not a convenience.
        A part that has not been amended since 2017 advertises exactly one version date, the
        2016-12-27 floor, which ``/full/`` refuses to serve. Without the current snapshot
        those parts ingest to nothing and their in-force text is silently absent from the
        corpus: parts 511, 530, 536 and 610 all behaved this way, so questions about hours of
        duty had no evidence to retrieve and no error to explain why.

        Cached with a TTL for the same reason as ``latest_issue_date``: this endpoint is
        where an amendment published tomorrow first becomes visible.
        """
        raw = self._cached(f"{API}/versions/title-{title}.json?part={part}",
                           f"versions-t{title}-p{part}.json", negative_cache=False,
                           ttl_s=self._index_ttl_s, refresh=refresh)
        rows = json.loads(raw).get("content_versions", [])
        dates = {r["date"] for r in rows if r.get("date") and r["date"] >= floor}
        if include_current:
            dates.add(self.latest_issue_date(title, refresh=refresh))
        return sorted(dates)

    def _absence_is_final(self, title: int, date: str) -> bool:
        """May a 404 for this snapshot date be written to the negative cache?

        Only well behind the title's latest issue date. Near it -- and 26 markers in
        data/ecfr were exactly this, one per part, all for the same date -- a 404 means eCFR
        has not published that issue yet, and pinning it removes the *current* text of the
        part from the corpus the moment the date becomes real.
        """
        try:
            latest = self.latest_issue_date(title)
        except (LookupError, httpx.HTTPError):
            return False  # cannot tell how far behind the date is; re-probe next time
        gap = _days_between(latest, date)
        return gap is not None and gap > self.issue_date_lag_days

    def snapshot(self, title: int, part: str, date: str, *, refresh: bool = False) -> bytes:
        """Full XML of a part as it stood on ``date``."""
        return self._cached(f"{API}/full/{date}/title-{title}.xml?part={part}",
                            f"full-t{title}-p{part}-{date}.xml",
                            negative_cache=lambda: self._absence_is_final(title, date),
                            refresh=refresh)

    def skipped_dates(self, title: int, part: str) -> list[str]:
        """Dates the last ``snapshots`` pass over this part could not retrieve."""
        return list(self.skipped.get(f"t{title}-p{part}", ()))

    def snapshots(self, title: int, part: str, *, floor: str = HISTORY_FLOOR):
        """Yield ``(date, xml)`` for every retrievable snapshot of a part.

        Dates with no retrievable text are counted onto ``skipped`` and logged rather than
        dropped on the floor. A bare ``continue`` here is how the current text of a part goes
        missing from the corpus with nothing anywhere reporting it.
        """
        key = f"t{title}-p{part}"
        skipped: list[str] = []
        self.skipped[key] = skipped
        dates = self.version_dates(title, part, floor=floor)
        for date in dates:
            try:
                yield date, self.snapshot(title, part, date)
            except SnapshotUnavailable:
                skipped.append(date)
        if skipped:
            log.warning("part %s: %d of %d advertised dates had no retrievable text (%s)",
                        part, len(skipped), len(dates), ", ".join(skipped))
