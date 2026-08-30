"""The scale generator and its measurement harness.

Two things are under test and neither of them is the system's speed. First: does the
synthetic corpus have the shape it claims to have -- the token-length distribution, the
nesting, the valid-time history, the vocabulary -- because every conclusion in
`results/eval-011-scale.md` rests on that and on nothing else. Second: does the harness
report what it says it reports, which is the difference between a measurement and a number.

Nothing here builds a large store. The 500k run is something to execute and write up; a
test suite that takes four minutes is a test suite people stop running. Tolerances are
stated at each assertion, and they are the sampling noise of a 6,000-row draw, not slack.
"""

from __future__ import annotations

import collections
import math
import time

import numpy as np
import pytest

from warrant.bench import scale
from warrant.index.store import Store

SEED = 7
#: Big enough that a 20-bucket quantile table is sampled everywhere, small enough that the
#: whole module builds one store in about a second.
N = 6000
SHAPE = scale.REAL_5CFR
AS_OF = SHAPE.window[1]


@pytest.fixture(scope="module")
def chunks() -> list:
    return list(scale.generate_chunks(SHAPE, N, seed=SEED))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple:
    path = tmp_path_factory.mktemp("scale") / "small.sqlite3"
    report = scale.build_store(path, N, seed=SEED)
    return path, report


@pytest.fixture
def store(built) -> Store:
    with Store(built[0]) as s:
        yield s


def tokens(chunks) -> list[int]:
    return sorted(len(c.text.split()) for c in chunks)


def pct(values: list[int], p: float) -> int:
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


# -- the distribution --------------------------------------------------------------


def test_mean_token_length_matches_the_shape(chunks):
    """Within 5%. The mean is what sizes the vocabulary and predicts the index, so a
    generator that drifts here is measuring a corpus nobody has."""
    measured = sum(tokens(chunks)) / len(chunks)
    assert measured == pytest.approx(SHAPE.mean_tokens, rel=0.05)


def test_token_quantiles_match_the_real_corpus(chunks):
    """The published shape: median 31, p10 10, p90 79 on in-force chunks; 29 / 9 / 79 over
    all versions, which is what is generated here. Tolerances are the 6,000-draw noise."""
    t = tokens(chunks)
    assert pct(t, 50) == pytest.approx(29, abs=2)
    assert pct(t, 10) == pytest.approx(9, abs=2)
    assert pct(t, 90) == pytest.approx(79, abs=8)


def test_the_short_chunk_mass_is_reproduced(chunks):
    """48.2% of in-force chunks are under 30 tokens (50.3% over all versions), and that is
    not incidental -- it is why `retrieval_text` prepends context at all. Within 3 points."""
    t = tokens(chunks)
    under = sum(1 for x in t if x < 30) / len(t)
    assert under == pytest.approx(0.503, abs=0.03)


def test_the_long_tail_is_reachable(chunks):
    """The real maximum is 1013 tokens. A generator that truncates the tail would understate
    the worst-case FTS5 posting, which is the only thing the tail is here for."""
    assert max(tokens(chunks)) > 200


def test_versions_per_paragraph_matches(chunks):
    """8,214 of 10,185 paragraphs have exactly one version; the tail runs to five."""
    counts = collections.Counter(collections.Counter(c.chunk_id for c in chunks).values())
    total = sum(counts.values())
    assert counts[1] / total == pytest.approx(8214 / 10185, abs=0.03)
    assert max(counts) == 5


def test_in_force_fraction_matches(chunks):
    """9,961 of 13,145 rows are in force. The admitted-set measurements are more sensitive
    to this number than to any other, because it *is* the admitted set."""
    open_ended = sum(1 for c in chunks if c.valid_to is None)
    assert open_ended / len(chunks) == pytest.approx(9961 / 13145, abs=0.03)


def test_paragraph_and_section_nesting_matches(chunks):
    """7.72 paragraphs per section on average, and no section wider than the observed 92."""
    per_section = collections.defaultdict(set)
    for c in chunks:
        per_section[c.section_id].add(c.chunk_id)
    sizes = [len(v) for v in per_section.values()]
    assert sum(sizes) / len(sizes) == pytest.approx(7.72, rel=0.25)
    assert max(sizes) <= SHAPE.paragraphs_per_section[-1]


def test_vocabulary_growth_follows_heaps(chunks):
    """The realised type count, not the latent one. A Zipf sampler over V latent types
    yields far fewer than V distinct terms, and the FTS5 term dictionary is built from what
    was realised. Within 15% of K * N**beta."""
    freq: collections.Counter = collections.Counter()
    for c in chunks:
        freq.update(c.text.lower().split())
    n_tokens = sum(freq.values())
    expected = SHAPE.heaps_k * n_tokens ** SHAPE.heaps_beta
    assert len(freq) == pytest.approx(expected, rel=0.15)


def test_the_head_of_the_distribution_is_the_measured_one(chunks):
    """"the" is 6.9% of the real corpus. A pure Zipf sampler gives it 23.8%, which would
    make the stopword postings list -- the thing that decides what lexical retrieval costs
    -- more than three times too long. Within 15% relative."""
    freq: collections.Counter = collections.Counter()
    for c in chunks:
        freq.update(c.text.lower().split())
    n_tokens = sum(freq.values())
    assert freq["the"] / n_tokens == pytest.approx(SHAPE.head_weights[0], rel=0.15)
    assert freq["of"] / n_tokens == pytest.approx(SHAPE.head_weights[1], rel=0.15)


def test_snapshot_dates_grow_sublinearly_and_stay_inside_the_window():
    """66 distinct dates at 26 parts, and the calendar is the ceiling: the admitted-set
    cache is keyed per as-of date, so an unbounded key space would be a finding."""
    rng = np.random.default_rng(0)
    small = scale.date_pool(SHAPE, 26, rng)
    large = scale.date_pool(SHAPE, 2600, rng)
    assert len(small) == pytest.approx(66, abs=6)
    assert len(large) < len(small) * 100          # sublinear in parts
    assert all(SHAPE.window[0] <= d <= SHAPE.window[1] for d in small + large)


# -- the structure -----------------------------------------------------------------


def test_valid_intervals_chain_without_gaps_or_overlaps(chunks):
    """Non-overlapping validity intervals is a CI invariant of the real store
    (ARCHITECTURE.md section 9). A generator that violates it would make every predicate
    measurement here a measurement of a store the system refuses to serve."""
    by_paragraph = collections.defaultdict(list)
    for c in chunks:
        by_paragraph[c.chunk_id].append(c)
    for versions in by_paragraph.values():
        ordered = sorted(versions, key=lambda c: c.valid_from)
        for a, b in zip(ordered, ordered[1:], strict=False):
            assert a.valid_to == b.valid_from
        assert len({c.valid_from for c in ordered}) == len(ordered)
        for c in ordered:
            assert c.valid_to is None or c.valid_from < c.valid_to


def test_at_most_one_version_of_a_paragraph_is_in_force(chunks):
    by_paragraph = collections.Counter(c.chunk_id for c in chunks if c.valid_to is None)
    assert by_paragraph and max(by_paragraph.values()) == 1


def test_version_ids_are_unique(chunks):
    ids = [c.version_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_the_generator_overshoots_by_less_than_one_paragraph():
    """Whole paragraphs are emitted, so the count lands at or just past the target -- never
    short, and never by more than the widest version history."""
    for target in (500, 6000):
        got = sum(1 for _ in scale.generate_chunks(SHAPE, target, seed=SEED))
        assert target <= got <= target + max(v for v, _ in SHAPE.versions_per_paragraph)


def test_generation_is_deterministic():
    a = [(c.version_id, c.text) for c in scale.generate_chunks(SHAPE, 800, seed=3)]
    b = [(c.version_id, c.text) for c in scale.generate_chunks(SHAPE, 800, seed=3)]
    c = [(x.version_id, x.text) for x in scale.generate_chunks(SHAPE, 800, seed=4)]
    assert a == b
    assert a != c


def test_the_store_admits_exactly_the_in_force_rows_at_the_ceiling(store, chunks):
    """The predicate the whole cost model turns on, checked against the generator's own
    intent rather than against itself."""
    admitted = store.candidate_ids(valid_date=AS_OF)
    assert len(admitted) == sum(1 for c in chunks if c.valid_to is None)


# -- the shape can be re-measured --------------------------------------------------


def test_measuring_a_generated_store_recovers_the_shape(store):
    """`CorpusShape.measure` is what points this at a corpus other than 5 CFR. If it cannot
    read back the shape it was handed, nothing it reads off a real store can be trusted."""
    got = scale.CorpusShape.measure(store.db, source="test")
    assert got.measured_rows == store.count()
    assert got.mean_tokens == pytest.approx(SHAPE.mean_tokens, rel=0.08)
    assert got.removed_fraction == pytest.approx(SHAPE.removed_fraction, abs=0.02)
    assert got.zipf_s == pytest.approx(SHAPE.zipf_s, rel=0.25)
    assert got.heaps_beta == pytest.approx(SHAPE.heaps_beta, rel=0.15)
    assert got.head_words[0] == "the"
    assert got.measured_from == "test"


def test_measure_refuses_an_empty_store():
    with Store(":memory:") as empty, pytest.raises(ValueError):
        scale.CorpusShape.measure(empty.db)


# -- the harness reports what it claims --------------------------------------------


def test_percentile_is_nearest_rank():
    values = list(range(1, 11))
    assert scale.percentile(values, 50) == 5
    assert scale.percentile(values, 95) == 10
    assert scale.percentile(values, 10) == 1
    assert scale.percentile(values, 100) == 10
    assert math.isnan(scale.percentile([], 50))


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    samples = [float(x) for x in range(1, 61)]
    lo, hi = scale.bootstrap_ci(samples, 50, seed=1)
    assert (lo, hi) == scale.bootstrap_ci(samples, 50, seed=1)
    assert lo <= scale.percentile(samples, 50) <= hi
    assert math.isnan(scale.bootstrap_ci([1.0], 50)[0])


def test_stage_summary_reports_one_sample_per_query(store):
    queries = scale.sample_queries(
        scale.Vocabulary.for_corpus(SHAPE, 1000, seed=SEED), n=5, seed=SEED)
    stats = scale.measure_stages(store, None, queries, as_of=AS_OF)
    for name in ("predicates_cold", "predicates_warm", "lexical", "fusion"):
        assert len(stats[name].samples) == 5
        assert stats[name].summary()["n"] == 5
        assert stats[name].summary()["p50"] <= stats[name].summary()["p95"]
    # No index, no dense stage. An absent stage must have no samples rather than zeros:
    # zero is a measurement and absence is not.
    assert stats["dense"].samples == []
    assert stats["isin"].samples == []
    assert stats["rerank"].samples == []


def test_the_cold_and_warm_predicate_paths_are_measured_separately(store):
    """The cache in `Store.candidate_ids` is the only difference between the two, and if the
    harness cannot see it the cache's whole justification is unmeasured."""
    queries = scale.sample_queries(
        scale.Vocabulary.for_corpus(SHAPE, 1000, seed=SEED), n=6, seed=SEED)
    stats = scale.measure_stages(store, None, queries, as_of=AS_OF)
    cold = scale.percentile(stats["predicates_cold"].samples, 50)
    warm = scale.percentile(stats["predicates_warm"].samples, 50)
    assert warm < cold


class _StubReranker:
    """A cross-encoder that costs a known amount of time and records what it was given."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.batches: list[int] = []

    def predict(self, pairs):
        self.batches.append(len(pairs))
        time.sleep(self.delay_s)
        return [0.0] * len(pairs)


def test_the_rerank_stage_times_the_reranker_and_bounds_its_batch(store):
    reranker = _StubReranker(0.01)
    queries = scale.sample_queries(
        scale.Vocabulary.for_corpus(SHAPE, 1000, seed=SEED), n=3, seed=SEED)
    stats = scale.measure_stages(store, None, queries, as_of=AS_OF, reranker=reranker,
                                 rerank_top_k=7)
    assert len(stats["rerank"].samples) == len(reranker.batches)
    assert all(s >= 10.0 for s in stats["rerank"].samples)
    assert reranker.batches and max(reranker.batches) <= 7


def test_content_only_queries_exclude_the_function_words():
    vocab = scale.Vocabulary.for_corpus(SHAPE, 200_000, seed=SEED)
    head = set(SHAPE.head_words)
    natural = scale.sample_queries(vocab, n=20, seed=SEED)
    content = scale.sample_queries(vocab, n=20, seed=SEED, content_only=True)
    assert any(t in head for q in natural for t in q.split())
    assert not any(t in head for q in content for t in q.split())
    assert content == scale.sample_queries(vocab, n=20, seed=SEED, content_only=True)


def test_diagnostics_price_the_cause_and_the_remedies(store):
    """Every one of these is a claim `results/eval-011-scale.md` makes; the test is that the
    harness can make it at all, not that any particular value holds at 6,000 rows."""
    index = scale.synthetic_index(store, seed=SEED)
    vocab = scale.Vocabulary.for_corpus(SHAPE, 200_000, seed=SEED)
    dates = [r[0] for r in store.db.execute(
        "SELECT DISTINCT valid_from FROM chunk ORDER BY 1")]
    got = scale.diagnose(store, index, scale.sample_queries(vocab, n=5, seed=SEED),
                         as_of=AS_OF, dates=dates, seed=SEED)

    # A natural query ORs its stopwords, so FTS5 matches most of the corpus and has to
    # score all of it to honour ORDER BY bm25 LIMIT k.
    assert 0.5 < got["match_fraction_p50"] <= 1.0
    assert got["match_rows_p50"] == pytest.approx(
        got["match_fraction_p50"] * store.count(), rel=0.01)
    assert got["admitted_fraction"] == pytest.approx(9961 / 13145, abs=0.05)

    # The three representations of the admitted set, smallest last.
    assert (got["admitted_mask_bytes"] < got["admitted_array_bytes"]
            < got["admitted_set_bytes"])
    assert got["apply_mask_ms"] < got["apply_isin_ms"]

    assert got["cold_empty_cache_ms"] > 0
    assert got["cold_on_flush_ms"] > 0
    assert got["warm_gc_on_ms"] > 0 and got["warm_gc_off_ms"] > 0
    assert got["dense_mask_ms"] > 0 and got["dense_gather_ms"] > 0


def test_the_warm_predicate_path_is_not_measuring_a_deallocation(store):
    """The harness used to hold the previous query's admitted set into the warm call, so
    freeing it landed inside the timer. A dict hit must read as a dict hit."""
    queries = scale.sample_queries(
        scale.Vocabulary.for_corpus(SHAPE, 1000, seed=SEED), n=8, seed=SEED)
    stats = scale.measure_stages(store, None, queries, as_of=AS_OF)
    warm = scale.percentile(stats["predicates_warm"].samples, 50)
    cold = scale.percentile(stats["predicates_cold"].samples, 50)
    assert warm < cold / 20


def test_probe_dates_are_distinct_and_inside_the_store(store):
    probe = scale._probe_dates(store, 65)
    lo, hi = store.db.execute("SELECT MIN(valid_from), MAX(valid_from) FROM chunk").fetchone()
    assert len(set(probe)) == 65
    assert probe[0] == lo and probe[-1] == hi


def test_set_bytes_counts_the_elements_not_just_the_table():
    small = scale.set_bytes(set(range(1000, 1100)))
    large = scale.set_bytes(set(range(1000, 2000)))
    assert large > small
    # 28 bytes per int on CPython, plus the set table. Anything near `getsizeof(s)` alone
    # would mean the elements were not counted, which is where the memory actually is.
    assert large / 900 > 28


def test_cache_pressure_agrees_with_tracemalloc_and_reports_its_entries(store):
    dates = [r[0] for r in store.db.execute(
        "SELECT DISTINCT valid_from FROM chunk ORDER BY 1")]
    got = scale.cache_pressure(store, dates, limit=8, trace=True)
    assert got.entries == min(8, len(dates))
    assert got.analytic_bytes == pytest.approx(got.traced_bytes, rel=0.30)
    assert got.bytes_per_entry > 0
    # The RSS delta is an upper bound, not a second opinion: freed arenas are not returned.
    assert got.rss_delta_bytes >= 0


def test_the_cache_cap_is_what_bounds_it(store):
    dates = [r[0] for r in store.db.execute(
        "SELECT DISTINCT valid_from FROM chunk ORDER BY 1")]
    assert len(dates) > 4
    assert scale.cache_pressure(store, dates, limit=4).entries == 4


def test_rss_is_readable_on_this_platform():
    """Zero means the platform would not say, and every memory row would then be blank."""
    assert scale.rss_bytes() > 0


def test_build_report_describes_the_store_it_built(built):
    path, report = built
    assert report.rows >= N
    assert report.db_bytes == path.stat().st_size
    assert report.fts_bytes > 0
    assert report.text_bytes > 0
    assert report.rows_per_second > 0
    with Store(path) as store:
        assert store.count() == report.rows
        stats = scale.store_stats(store.db)
    assert stats["paragraphs"] < report.rows          # versions exceed paragraphs
    assert stats["in_force"] < report.rows


def test_synthetic_index_covers_every_believed_row_and_respects_the_predicate(store):
    index = scale.synthetic_index(store, seed=SEED)
    believed = {r[0] for r in store.db.execute(
        "SELECT id FROM chunk WHERE system_to IS NULL")}
    assert set(index.ids.tolist()) == believed
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)

    allowed = store.candidate_ids(valid_date=AS_OF)
    vec = index.vectors[0]
    hits = index.search(vec, allowed=allowed, limit=20)
    assert hits and all(i in allowed for i, _ in hits)


def test_tail_size_realises_the_type_count_it_was_asked_for():
    """The generator's one non-obvious sizing step, checked directly: latent types are not
    realised types, and the index is built from the realised ones."""
    mass, s, offset, n_tokens = 0.465, SHAPE.zipf_s, 100, 500_000
    for want in (2_000, 20_000):
        size = scale._tail_size(want, n_tokens, mass, s, offset)
        weights = scale._tail_weights(size, mass, s, offset)
        realised = float(np.sum(-np.expm1(-n_tokens * weights)))
        assert realised == pytest.approx(want, rel=0.02)
        assert size > want                              # latent exceeds realised


def test_vocabulary_from_texts_uses_the_observed_frequencies():
    vocab = scale.Vocabulary.from_texts(["a a a b b c"])
    assert vocab.words[:3] == ["a", "b", "c"]
    assert vocab.weights[0] == pytest.approx(0.5)
    assert vocab.weights[1] == pytest.approx(1 / 3)


def test_vocabulary_refuses_to_be_empty():
    with pytest.raises(ValueError):
        scale.Vocabulary([], [])


def test_render_markdown_carries_the_numbers(built):
    path, report = built
    point = scale.run_point("small", path, build=False, queries=3, seed=SEED)
    table = scale.render_markdown([point])
    assert "small" in table
    assert f"{point.rows:,}" in table
    assert "### build" in table and "### memory" in table
    # A point measured without a build has no build row to report, and must say so rather
    # than printing a zero.
    assert "| small | " in table
    assert point.build is None
    assert point.cache.entries > 0
    assert point.dense_matrix_bytes == point.rows * scale.DENSE_DIM * 4
