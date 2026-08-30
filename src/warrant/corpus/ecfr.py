"""Client for the eCFR versioner API.

Two endpoints carry the entire corpus:

    /versions/title-{t}.json?part={p}   section-level version records
    /full/{date}/title-{t}.xml?part={p} the part as it stood on {date}

Two things about this API are easy to get wrong, and both were found by running it rather
than by reading the docs:

1. ``/versions`` returns one row per *section* version, not per snapshot. Part 630 returns
   226 rows that collapse to 8 distinct dates. Counting rows overstates the amount of
   diffable history by more than an order of magnitude.

2. Every part reports a first version date of 2016-12-27, and ``/full/`` returns 404 for it.
   The usable point-in-time window starts in 2017.

Responses are cached on disk keyed by request, because the corpus is immutable history: a
snapshot of Part 630 as of 2019-06-01 will never change. A rebuild should cost no requests.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

API = "https://www.ecfr.gov/api/versioner/v1"
USER_AGENT = "warrant/0.1 (+https://github.com/tasnimuldatascience/warrant)"

#: /full/ 404s at the 2016-12-27 floor every part advertises.
HISTORY_FLOOR = "2017-01-01"


class SnapshotUnavailable(LookupError):
    """The API has a version record for this date but no retrievable text."""


@dataclass
class ECFRClient:
    cache_dir: Path
    delay_s: float = 1.0
    timeout_s: float = 180.0
    _last_request: float = 0.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- transport ---------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
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

    def _cached(self, url: str, name: str, *, negative_cache: bool = True) -> bytes:
        """Fetch through the on-disk cache.

        Writes are atomic. A non-atomic ``write_bytes`` leaves a truncated file if the
        process dies mid-write -- a Ctrl-C during the ten-minute fetch, a closed lid, an OOM
        -- and the cache only checks for existence, so that truncation is trusted forever.
        The symptom is an XMLSyntaxError partway through a later build, naming no file.

        ``negative_cache=False`` is for the index endpoints. A 404 there is usually eCFR's
        issue date lagging the calendar, not a permanent absence, and persisting it would
        silently drop the current text of every part once the date became real.
        """
        path = self.cache_dir / name
        if path.exists():
            return path.read_bytes()
        missing = self.cache_dir / (name + ".404")
        if negative_cache and missing.exists():
            raise SnapshotUnavailable(url)
        try:
            content = self._fetch(url)
        except SnapshotUnavailable:
            # Record the absence too. Re-probing a permanent 404 on every rebuild is a
            # slow way to be rude to someone else's server.
            if negative_cache:
                missing.write_bytes(b"")
            raise
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        return content

    # -- corpus ------------------------------------------------------------------

    def latest_issue_date(self, title: int) -> str:
        """The most recent date for which the API will serve this title.

        Not today. ``/full/`` returns 404 for any date after the title issue date, and the
        gap is days wide -- 2026-08-30 was rejected while 2026-08-26 was served. Reading the
        API statement of its own currency keeps ingestion deterministic instead of dependent
        on the wall clock and on how far behind eCFR happens to be.
        """
        raw = self._cached(f"{API}/titles.json", "titles.json")
        for row in json.loads(raw).get("titles", []):
            if row.get("number") == title:
                return row["latest_issue_date"]
        raise LookupError(f"title {title} not listed by the eCFR API")

    def version_dates(self, title: int, part: str, *, floor: str = HISTORY_FLOOR,
                      include_current: bool = True) -> list[str]:
        """Distinct snapshot dates for a part, oldest first.

        Distinct *dates* -- not the row count, which is per-section and much larger.

        ``include_current`` appends the title latest issue date, and it is not a convenience.
        A part that has not been amended since 2017 advertises exactly one version date, the
        2016-12-27 floor, which ``/full/`` refuses to serve. Without the current snapshot
        those parts ingest to nothing and their in-force text is silently absent from the
        corpus: parts 511, 530, 536 and 610 all behaved this way, so questions about hours of
        duty had no evidence to retrieve and no error to explain why.
        """
        raw = self._cached(f"{API}/versions/title-{title}.json?part={part}",
                           f"versions-t{title}-p{part}.json")
        rows = json.loads(raw).get("content_versions", [])
        dates = {r["date"] for r in rows if r.get("date") and r["date"] >= floor}
        if include_current:
            dates.add(self.latest_issue_date(title))
        return sorted(dates)

    def snapshot(self, title: int, part: str, date: str) -> bytes:
        """Full XML of a part as it stood on ``date``."""
        return self._cached(f"{API}/full/{date}/title-{title}.xml?part={part}",
                            f"full-t{title}-p{part}-{date}.xml")

    def snapshots(self, title: int, part: str, *, floor: str = HISTORY_FLOOR):
        """Yield ``(date, xml)`` for every retrievable snapshot of a part."""
        for date in self.version_dates(title, part, floor=floor):
            try:
                yield date, self.snapshot(title, part, date)
            except SnapshotUnavailable:
                continue
