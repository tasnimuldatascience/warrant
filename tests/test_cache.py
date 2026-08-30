"""The answer cache, and the claim that its expiry is computed rather than guessed.

Offline and model-free, like `tests/test_hybrid.py`: a hand-built in-memory `Store` whose
validity intervals are chosen per test, a dict for a payload, and an explicit ``now`` on every
call so that nothing here depends on the wall clock. Every assertion below is about a date the
store already knew, which is the whole argument the module makes.
"""

from __future__ import annotations

import pytest

from warrant.index.store import Chunk, Store
from warrant.retrieve.hybrid import fts_query
from warrant.retrieve.scope import Scope
from warrant.serve.cache import AnswerCache, cache_key, normalise_query

T0 = "2020-01-01T00:00:00+00:00"          # when the system started believing the fixture
NOW = "2026-01-01T00:00:00+00:00"
DAY = 24 * 60 * 60.0

#: Long enough that the TTL never accidentally decides a test that is about a computed bound.
LONG_TTL = 365 * DAY

PAYLOAD = {"claims": [{"text": "Restored leave must be scheduled within two years."}]}


def chunk(chunk_id: str, *, valid_from: str = "2017-01-01", valid_to: str | None = None,
          doc_id: str = "cfr-630", text: str = "annual leave restored") -> Chunk:
    section = chunk_id.split("#")[0]
    return Chunk(chunk_id=chunk_id, section_id=section, title=5, part=section.split(".")[0],
                 anchor="a", heading="Restored annual leave", text=text, doc_id=doc_id,
                 valid_from=valid_from, valid_to=valid_to)


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        s.add([
            # Still in force: no valid-time bound of its own.
            chunk("630.306#a"),
            # Closes in the near future: an amendment the store already knows is coming.
            chunk("630.305#a", valid_to="2026-03-01"),
            # Closes later than 630.305, so min() has something to choose between.
            chunk("630.304#a", valid_to="2026-06-01"),
            # Long since superseded: what a historical question cites.
            chunk("531.404#a", valid_to="2020-08-10", doc_id="cfr-531"),
            # A different document, for invalidate_document.
            chunk("532.203#a", doc_id="cfr-532"),
        ], system_from=T0)
        yield s


@pytest.fixture
def cache(store: Store) -> AnswerCache:
    with AnswerCache(":memory:", store, max_entries=64, max_ttl_seconds=LONG_TTL) as c:
        yield c


def key(query: str = "restored annual leave", *, as_of: str = "2026-01-01",
        scope: Scope | None = None, config_hash: str = "cfg1") -> str:
    return cache_key(query, as_of=as_of, scope=scope or Scope(), config_hash=config_hash)


# -- keys --------------------------------------------------------------------------


def test_a_stored_answer_comes_back_whole(cache: AnswerCache):
    k = key()
    assert cache.put(k, PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW) is not None
    hit = cache.get(k, now=NOW)
    assert hit is not None
    assert hit.payload == PAYLOAD
    assert hit.cited == ["630.306#a@2017-01-01"]
    assert cache.stats().hits == 1


def test_a_different_scope_is_a_different_question(cache: AnswerCache):
    cache.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.get(key(scope=Scope.of(pay_system="FWS")), now=NOW) is None
    assert cache.stats().misses == 1


def test_a_different_config_hash_is_a_different_answer(cache: AnswerCache):
    """An answer produced under one retrieval config is not the answer the current config
    would give. Serving it because the query text matched reverts a tuning result silently,
    with a 200 and plausible claims."""
    cache.put(key(config_hash="cfg1"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.get(key(config_hash="cfg2"), now=NOW) is None
    assert cache.stats().misses == 1


def test_a_different_as_of_is_a_different_question(cache: AnswerCache):
    cache.put(key(as_of="2026-01-01"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.get(key(as_of="2019-01-01"), now=NOW) is None


def test_the_key_uses_retrieval_s_own_normalisation(cache: AnswerCache):
    """Not a second normalisation that can drift from the first: two queries FTS5 cannot tell
    apart are one entry, and the normal form is `fts_query` itself."""
    assert normalise_query("annual leave!!") == fts_query("annual leave!!")
    cache.put(key("annual leave"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.get(key("annual, leave"), now=NOW) is not None
    assert cache.get(key("leave annual"), now=NOW) is None


# -- computed expiry ----------------------------------------------------------------


def test_expiry_is_the_earliest_cited_valid_to(cache: AnswerCache):
    """Two citations, two closing dates: the answer becomes wrong at the first of them, not
    the last, and not at whatever TTL someone would have guessed."""
    entry = cache.put(key(), PAYLOAD, now=NOW,
                      cited=["630.304#a@2017-01-01", "630.305#a@2017-01-01"])
    assert entry is not None
    assert entry.expires_at == "2026-03-01T00:00:00+00:00"


def test_an_open_ended_citation_contributes_no_bound(cache: AnswerCache):
    """NULL valid_to is infinity, not a sentinel date -- so only the TTL holds this one."""
    entry = cache.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert entry is not None
    assert entry.expires_at == "2027-01-01T00:00:00+00:00"       # NOW + LONG_TTL


def test_the_computed_bound_wins_when_it_is_sooner_than_the_ttl(store: Store):
    """The required case: a cited chunk closes *before* the configured TTL would have run
    out, so the configured number is not what expires the entry."""
    with AnswerCache(":memory:", store, max_ttl_seconds=180 * DAY) as c:
        entry = c.put(key(), PAYLOAD, cited=["630.305#a@2017-01-01"], now=NOW)
        assert entry is not None
        assert entry.expires_at == "2026-03-01T00:00:00+00:00"   # < NOW + 180 days


def test_the_ttl_caps_a_far_future_expiry(store: Store):
    """A guidance page can be rewritten at its source without any version in this corpus
    closing, so an unbounded computed expiry still needs a ceiling."""
    store.add([chunk("630.999#a", valid_to="2099-01-01")], system_from=T0)
    with AnswerCache(":memory:", store, max_ttl_seconds=DAY) as c:
        entry = c.put(key(), PAYLOAD, cited=["630.999#a@2017-01-01"], now=NOW)
        assert entry is not None
        assert entry.expires_at == "2026-01-02T00:00:00+00:00"


def test_the_entry_expires_at_that_instant_and_not_before(store: Store):
    with AnswerCache(":memory:", store, max_ttl_seconds=LONG_TTL) as c:
        c.put(key(), PAYLOAD, cited=["630.305#a@2017-01-01"], now=NOW)
        assert c.get(key(), now="2026-02-28T23:59:59+00:00") is not None
        assert c.get(key(), now="2026-03-01T00:00:00+00:00") is None
        assert c.stats().expired == 1


def test_a_bare_date_bound_is_read_as_midnight_not_end_of_day(cache: AnswerCache):
    """valid_to is exclusive, so the cited text stops being law at the first instant of that
    day. Comparing the bare date against a timestamp without canonicalising would have
    expired this entry a day early."""
    cache.put(key(), PAYLOAD, cited=["630.305#a@2017-01-01"], now=NOW)
    assert cache.get(key(), now="2026-02-28T12:00:00+00:00") is not None
    assert cache.get(key(), now="2026-03-01T00:00:01+00:00") is None


def test_a_historical_answer_is_not_expired_by_a_bound_already_in_the_past(cache: AnswerCache):
    """Valid time does not move. An answer about 2019 cites text superseded in 2020, and that
    interval is closed forever -- treating its valid_to as an expiry would make the most
    cacheable queries in the corpus the only uncacheable ones."""
    k = key(as_of="2019-01-01")
    entry = cache.put(k, PAYLOAD, cited=["531.404#a@2017-01-01"], now=NOW)
    assert entry is not None
    assert entry.expires_at == "2027-01-01T00:00:00+00:00"       # the TTL, not 2020-08-10
    assert cache.get(k, now=NOW) is not None


def test_an_abstention_cites_nothing_and_is_held_only_by_the_ttl(cache: AnswerCache):
    """Absence of evidence has no valid_to: nothing in the store predicts when a question it
    cannot answer becomes answerable, which is precisely what the cap is for."""
    entry = cache.put(key(), {"claims": [], "abstained": True}, cited=[], now=NOW)
    assert entry is not None
    assert entry.expires_at == "2027-01-01T00:00:00+00:00"
    assert cache.get(key(), now=NOW) is not None


# -- belief invalidation -------------------------------------------------------------


def test_a_retracted_citation_invalidates_immediately_despite_a_future_expiry(
        cache: AnswerCache, store: Store):
    """The difference between a cache and a liability. expires_at is a year away and the
    entry is dead anyway, because the system no longer believes the text it was built from."""
    k = key()
    entry = cache.put(k, PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert entry is not None and entry.expires_at > NOW

    store.retract("630.306#a@2017-01-01", system_to="2026-01-02T00:00:00+00:00")

    assert cache.get(k, now=NOW) is None
    stats = cache.stats()
    assert (stats.invalidated, stats.expired) == (1, 0)


def test_a_corrected_parse_invalidates_even_though_the_version_id_returns(
        cache: AnswerCache, store: Store):
    """A correction retracts the row and re-inserts the *same* version_id with better text.
    Checking only that the id is believed again would happily serve an answer assembled from
    the sentence that was wrong, so the citation carries the content hash it was written at."""
    k = key()
    cache.put(k, PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)

    store.retract("630.306#a@2017-01-01", system_to="2026-01-02T00:00:00+00:00")
    store.add([chunk("630.306#a", text="annual leave restored -- corrected")],
              system_from="2026-01-02T00:00:00+00:00")

    assert cache.get(k, now=NOW) is None
    assert cache.stats().invalidated == 1


def test_stats_keeps_expiry_and_belief_invalidation_apart(cache: AnswerCache, store: Store):
    """Two different failures with two different fixes -- an amendment landed, versus the
    corpus was corrected under us. A single hit rate cannot say which is happening."""
    aged, doubted = key("aged"), key("doubted")
    cache.put(aged, PAYLOAD, cited=["630.305#a@2017-01-01"], now=NOW)
    cache.put(doubted, PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    store.retract("630.306#a@2017-01-01", system_to="2026-01-02T00:00:00+00:00")

    after = "2026-03-02T00:00:00+00:00"
    assert cache.get(aged, now=after) is None
    assert cache.get(doubted, now=after) is None
    assert cache.get(key("never asked"), now=after) is None

    stats = cache.stats()
    assert (stats.expired, stats.invalidated, stats.misses, stats.hits) == (1, 1, 1, 0)
    assert stats.lookups == 3


def test_a_dead_entry_is_dropped_rather_than_rechecked_forever(cache: AnswerCache,
                                                               store: Store):
    cache.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    store.retract("630.306#a@2017-01-01", system_to="2026-01-02T00:00:00+00:00")
    cache.get(key(), now=NOW)
    assert cache.count() == 0


def test_an_answer_citing_evidence_the_store_disowns_is_never_written(cache: AnswerCache):
    """Refused at write time rather than stored and thrown away on the next read: the entry
    could never be served, and a slot spent on it is a slot taken from one that could."""
    assert cache.put(key(), PAYLOAD, cited=["630.306#a@1900-01-01"], now=NOW) is None
    assert cache.count() == 0
    assert cache.stats().refused == 1


# -- invalidation and sweeping --------------------------------------------------------


def test_invalidate_document_clears_everything_citing_that_document(cache: AnswerCache):
    """A re-ingest knows which document it rewrote before any reader would notice."""
    for i, cited in enumerate([["630.306#a@2017-01-01"],
                               ["630.305#a@2017-01-01", "532.203#a@2017-01-01"]]):
        cache.put(key(f"q{i}"), PAYLOAD, cited=cited, now=NOW)
    cache.put(key("q2"), PAYLOAD, cited=["532.203#a@2017-01-01"], now=NOW)
    assert cache.count() == 3

    assert cache.invalidate_document("cfr-630") == 2
    assert cache.count() == 1
    assert cache.get(key("q2"), now=NOW) is not None


def test_invalidate_document_leaves_no_orphan_citations(cache: AnswerCache):
    """An orphaned citation row would keep matching this document forever, so a later flush
    would report work it did not do."""
    cache.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.invalidate_document("cfr-630") == 1
    assert cache.invalidate_document("cfr-630") == 0


def test_sweep_removes_the_dead_without_waiting_for_a_lookup(cache: AnswerCache,
                                                             store: Store):
    cache.put(key("aged"), PAYLOAD, cited=["630.305#a@2017-01-01"], now=NOW)
    cache.put(key("doubted"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    cache.put(key("live"), PAYLOAD, cited=["532.203#a@2017-01-01"], now=NOW)
    store.retract("630.306#a@2017-01-01", system_to="2026-01-02T00:00:00+00:00")

    assert cache.sweep("2026-03-02T00:00:00+00:00") == 2
    assert cache.count() == 1
    stats = cache.stats()
    assert (stats.expired, stats.invalidated) == (1, 1)


# -- bounds ----------------------------------------------------------------------------


def test_the_cache_evicts_least_recently_used_at_the_cap(store: Store):
    """An endpoint anyone can call with arbitrary text must not be able to grow storage
    without limit; this project already had one unbounded cache keyed on raw query text."""
    with AnswerCache(":memory:", store, max_entries=3, max_ttl_seconds=LONG_TTL) as c:
        for i in range(3):
            c.put(key(f"q{i}"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
        assert c.get(key("q0"), now=NOW) is not None          # q1 is now the coldest

        c.put(key("q3"), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)

        assert c.count() == 3
        assert c.get(key("q1"), now=NOW) is None
        assert c.get(key("q0"), now=NOW) is not None
        assert c.get(key("q3"), now=NOW) is not None
        assert c.stats().evicted == 1


def test_writing_the_same_key_twice_replaces_rather_than_grows(cache: AnswerCache):
    for _ in range(3):
        cache.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
    assert cache.count() == 1
    assert cache.get(key(), now=NOW) is not None


# -- on disk -----------------------------------------------------------------------------


def test_a_stale_schema_is_discarded_rather_than_refused(store: Store, tmp_path):
    """Where the corpus and trace stores raise and tell an operator to delete the file, a
    cache drops its own tables and starts empty. Its contents are worth nothing, and the one
    thing a cache must never do is make a deployment fail."""
    path = tmp_path / "answers.sqlite3"
    with AnswerCache(path, store) as c:
        c.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW)
        assert c.count() == 1

    import sqlite3

    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version = 99")

    with AnswerCache(path, store) as c:
        assert c.count() == 0
        assert c.put(key(), PAYLOAD, cited=["630.306#a@2017-01-01"], now=NOW) is not None


def test_entries_written_on_one_thread_are_readable_on_another(tmp_path):
    """Connections are thread-local because sqlite3 forbids sharing them, and FastAPI runs
    synchronous endpoints in a threadpool -- so a write and the read that follows it routinely
    land on different threads.

    On disk on both sides. The `store` fixture is ``:memory:``, whose connection `Store`
    shares deliberately because each ``:memory:`` connection is its own empty database; the
    belief check reads the corpus, so this test would fail inside `Store`, not the cache."""
    from concurrent.futures import ThreadPoolExecutor

    with Store(tmp_path / "corpus.sqlite3") as s:
        s.add([chunk("630.306#a")], system_from=T0)
        with AnswerCache(tmp_path / "answers.sqlite3", s) as c, \
                ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(c.put, key(), PAYLOAD,
                        cited=["630.306#a@2017-01-01"], now=NOW).result()
            hit = pool.submit(c.get, key(), now=NOW).result()
            assert hit is not None and hit.payload == PAYLOAD
