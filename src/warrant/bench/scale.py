"""Does the mechanism hold at 10x?

Every number in `results/` was taken against 13,145 chunk versions. Several design choices
are size-dependent in ways nothing had measured:

  * `Store.candidate_ids` materialises the admitted set as a Python ``set[int]`` -- 9,961
    today -- and caches up to 64 of them, keyed on (as-of, sources, authority).
  * Dense search is exact: a full ``(n, 384)`` matmul per query, with the predicate applied
    by ``np.isin`` against that set. There is no ANN index at all.
  * FTS5 ranks **every** posting a query matches, not the top k, and `fts_query` ORs the
    query's stopwords in along with everything else.

Fetching more of the CFR is a ten-minute network job per part and would not answer the
question any faster. So this module generates a corpus with the *statistical shape* of the
real one at an arbitrary size and measures the mechanism against it.

**What synthetic text licenses.** It bounds *cost*: index size, build time, per-stage
latency, resident memory. Those depend on the row count, the token-length distribution, the
number of vocabulary types and how postings are spread across them -- all reproduced here,
none of which care whether the sentences mean anything. It says **nothing** about retrieval
*quality* at scale. Recall against synthetic questions over synthetic prose measures the
generator, not the system, and no such number is reported. Quality at scale needs real text
and real questions, and that is a different job.

The shape below was measured from `data/warrant.sqlite3` at 13,145 chunk versions on
2026-08-30; `CorpusShape.measure` recomputes it from any store, and the sweep re-runs the
anchor point against the real store so the synthetic and the real can be compared directly
rather than assumed equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
import tracemalloc
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ..index.store import Chunk, Store
from ..retrieve.dense import DenseIndex
from ..retrieve.hybrid import fts_query, fuse

#: Dimension of `BAAI/bge-small-en-v1.5`, the encoder `index/dense.py` defaults to. The
#: vectors here are random, but the matrix has to be the shape the real one would be: the
#: cost of an exact scan is entirely (rows x dim x 4 bytes).
DENSE_DIM = 384

#: Type lengths are drawn from a quantile table whose last entry is the observed maximum --
#: 142 characters, one URL-shaped token. Interpolating into that bucket would make ~5% of
#: the synthetic vocabulary absurdly long and inflate the FTS5 term dictionary; real p95 is
#: 14. Clamped, and the clamp costs well under 1% of index bytes.
MAX_TYPE_LEN = 20


# ---------------------------------------------------------------------------- shape


@dataclass(frozen=True)
class CorpusShape:
    """The statistical shape of a real corpus, in exactly the terms the cost model needs.

    Quantile tables rather than fitted distributions. The token-length distribution is
    bimodal-ish and heavy-tailed -- 48.2% of in-force chunks under 30 tokens, a maximum of
    1013 -- and every two-parameter family that fits its middle misses one of the two ends
    that actually drive cost: the short chunks set the row count for a given token budget,
    the long ones set the worst-case FTS5 posting.
    """

    #: Tokens per chunk, empirical inverse CDF at 0..100%.
    token_quantiles: tuple[int, ...]
    #: The same, at 99.0..100% in tenths. The top percentile spans 172..1013 and
    #: interpolating it out of the 101-point table alone over-produces long chunks by ~9%.
    token_tail_quantiles: tuple[int, ...]
    #: Paragraphs (distinct chunk_ids) per section, inverse CDF at 0..100% in 5% steps.
    paragraphs_per_section: tuple[int, ...]
    #: Words in a section heading, same form.
    heading_quantiles: tuple[int, ...]
    #: Characters per vocabulary *type*, same form. Types, not tokens: this sizes the FTS5
    #: term dictionary, which stores each distinct term once.
    type_length_quantiles: tuple[int, ...]
    #: Sections per part, as observed. Sampled from with replacement as parts are added.
    sections_per_part: tuple[int, ...]
    #: (versions, weight) for one paragraph's valid-time history.
    versions_per_paragraph: tuple[tuple[int, float], ...]
    #: Zipf exponent, fitted on log freq vs log rank over ranks 10..5000.
    zipf_s: float
    #: Heaps' law V = K * N**beta, fitted on a shuffled token stream. Without it a synthetic
    #: 500k-chunk corpus would reuse the real 17,173-type vocabulary and understate both the
    #: FTS5 term dictionary and the number of distinct postings lists a query can touch.
    heaps_k: float
    heaps_beta: float
    #: Distinct snapshot dates grow as parts are added: 4.3 for one part, 66.0 for 26,
    #: fitted as a * parts**b. It matters because the admitted-set cache is keyed per
    #: as-of date, so this is the size of the key space the 64-entry cap is defending.
    dates_a: float
    dates_b: float
    #: Fraction of paragraphs whose last version is closed -- a repealed paragraph, in force
    #: on no date. 1 - 9961/10185 in the real store.
    removed_fraction: float
    #: The point-in-time window, `corpus.history_floor` to the latest issue date.
    window: tuple[str, str]
    #: The real vocabulary head. Function words carry most of the tokens and all of the
    #: expensive postings lists, and `fts_query` ORs them into every natural query.
    head_words: tuple[str, ...]
    measured_rows: int = 0
    measured_from: str = ""

    @property
    def mean_versions_per_paragraph(self) -> float:
        total = sum(w for _, w in self.versions_per_paragraph)
        return sum(v * w for v, w in self.versions_per_paragraph) / total

    @property
    def mean_paragraphs_per_section(self) -> float:
        return _quantile_mean(self.paragraphs_per_section)

    @property
    def mean_tokens(self) -> float:
        """Mean tokens per chunk implied by the two quantile tables, as sampled."""
        body = self.token_quantiles[:-1]
        head = sum((body[i] + body[i + 1]) / 2 for i in range(len(body) - 1))
        head = head / (len(body) - 1) * 0.99
        tail = self.token_tail_quantiles
        tail_mean = sum((tail[i] + tail[i + 1]) / 2 for i in range(len(tail) - 1))
        return head + tail_mean / (len(tail) - 1) * 0.01

    @classmethod
    def measure(cls, conn: sqlite3.Connection, *, seed: int = 0,
                source: str = "") -> CorpusShape:
        """Recompute the shape from a real store. One pass, everything in memory.

        Believed rows only (``system_to IS NULL``), because that is the set the dense index
        embeds and the set every predicate query scans.
        """
        rows = conn.execute(
            "SELECT part, section_id, chunk_id, valid_from, valid_to, heading, text "
            "FROM chunk WHERE system_to IS NULL"
        ).fetchall()
        if not rows:
            raise ValueError("cannot measure the shape of an empty store")
        texts = [r[6] for r in rows]
        tokens = sorted(len(t.split()) for t in texts)

        paras: dict[str, set[str]] = {}
        sections: dict[str, set[str]] = {}
        dates_by_part: dict[str, set[str]] = {}
        versions: dict[str, int] = {}
        closed: dict[str, bool] = {}
        for part, section_id, chunk_id, valid_from, valid_to, _h, _t in rows:
            paras.setdefault(section_id, set()).add(chunk_id)
            sections.setdefault(part, set()).add(section_id)
            dates_by_part.setdefault(part, set()).add(valid_from)
            versions[chunk_id] = versions.get(chunk_id, 0) + 1
            # A paragraph is "removed" when no version of it is open-ended.
            closed[chunk_id] = closed.get(chunk_id, True) and valid_to is not None

        counts: dict[int, float] = {}
        for v in versions.values():
            counts[v] = counts.get(v, 0.0) + 1.0

        freq: dict[str, int] = {}
        for t in texts:
            for w in t.lower().split():
                freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        n_tokens = sum(freq.values())

        all_dates = sorted({r[3] for r in rows})
        return cls(
            token_quantiles=_quantiles(tokens, 100),
            token_tail_quantiles=tuple(
                tokens[min(len(tokens) - 1, int((99 + i * 0.1) / 100 * len(tokens)))]
                for i in range(11)),
            paragraphs_per_section=_quantiles(sorted(len(v) for v in paras.values()), 20),
            heading_quantiles=_quantiles(sorted(len((r[5] or "").split()) for r in rows), 20),
            type_length_quantiles=_quantiles(sorted(len(t) for t in freq), 20),
            sections_per_part=tuple(sorted(len(v) for v in sections.values())),
            versions_per_paragraph=tuple(sorted(counts.items())),
            zipf_s=_zipf_exponent(ranked),
            heaps_k=_heaps_k(len(freq), n_tokens, _heaps_beta(texts, seed)),
            heaps_beta=_heaps_beta(texts, seed),
            dates_a=_dates_a(dates_by_part, seed),
            dates_b=_dates_b(dates_by_part, seed),
            removed_fraction=sum(closed.values()) / len(closed),
            window=(all_dates[0], all_dates[-1]),
            head_words=tuple(w for w, _ in ranked[:100]),
            measured_rows=len(rows),
            measured_from=source,
        )


def _quantiles(values: Sequence[int], steps: int) -> tuple[int, ...]:
    n = len(values)
    return tuple(values[min(n - 1, int(i / steps * n))] for i in range(steps + 1))


def _quantile_mean(table: Sequence[int]) -> float:
    """Mean of the distribution the table describes, under linear interpolation."""
    return sum((table[i] + table[i + 1]) / 2 for i in range(len(table) - 1)) / (len(table) - 1)


def _zipf_exponent(ranked: Sequence[tuple[str, int]]) -> float:
    """Least-squares slope of log frequency against log rank, over ranks 10..5000.

    Not from rank 1: the very head of an English corpus is above the Zipf line and fitting
    through it biases the exponent toward 1, which would flatten the tail the term
    dictionary is made of.
    """
    hi = min(5000, len(ranked))
    if hi <= 10:
        return 1.0
    xs = [math.log(r) for r in range(10, hi + 1)]
    ys = [math.log(ranked[r - 1][1]) for r in range(10, hi + 1)]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    return -num / den if den else 1.0


def _heaps_beta(texts: Sequence[str], seed: int) -> float:
    """Vocabulary growth exponent, on a shuffled token stream.

    Shuffled because the corpus is ordered by part and each part has its own jargon; in
    document order the curve is a staircase and the fit reads the ordering rather than the
    language.
    """
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    seen: set[str] = set()
    n = 0
    first: tuple[int, int] | None = None
    for i in order:
        for w in texts[i].lower().split():
            n += 1
            seen.add(w)
            if first is None and n == 1000:
                first = (n, len(seen))
    if first is None or n <= first[0]:
        return 0.6
    return (math.log(len(seen)) - math.log(first[1])) / (math.log(n) - math.log(first[0]))


def _heaps_k(n_types: int, n_tokens: int, beta: float) -> float:
    return n_types / (n_tokens ** beta) if n_tokens else 1.0


def _dates_fit(dates_by_part: dict[str, set[str]], seed: int) -> tuple[float, float]:
    """Fit distinct-dates = a * parts**b by two points: one part, and all of them."""
    parts = sorted(dates_by_part)
    rng = random.Random(seed)
    one = sum(len(dates_by_part[p]) for p in parts) / len(parts)
    for _ in range(40):
        rng.sample(parts, 1)  # keep the draw sequence stable across refactors
    allp = len({d for p in parts for d in dates_by_part[p]})
    if len(parts) < 2 or one <= 0 or allp <= one:
        return (max(one, 1.0), 1.0)
    return (one, math.log(allp / one) / math.log(len(parts)))


def _dates_a(dates_by_part: dict[str, set[str]], seed: int) -> float:
    return _dates_fit(dates_by_part, seed)[0]


def _dates_b(dates_by_part: dict[str, set[str]], seed: int) -> float:
    return _dates_fit(dates_by_part, seed)[1]


#: Measured from `data/warrant.sqlite3` on 2026-08-30: 13,145 chunk versions, 10,185
#: paragraphs, 1,320 sections, 26 parts of 5 CFR, 66 distinct snapshot dates, 17,173
#: vocabulary types over 514,629 tokens. Baked in so the generator runs without a store --
#: tests need it, and a reviewer with no `data/` still gets the shape the numbers were
#: taken at. `CorpusShape.measure` regenerates it.
REAL_5CFR = CorpusShape(
    token_quantiles=(
        1, 3, 3, 4, 5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
        16, 16, 17, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25,
        26, 26, 27, 27, 28, 29, 29, 30, 30, 31, 32, 33, 33, 34, 35, 35, 36, 37, 38, 39, 40,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 59, 60, 62, 64, 66,
        68, 71, 73, 76, 79, 82, 87, 91, 97, 105, 114, 126, 143, 172, 1013),
    token_tail_quantiles=(172, 175, 180, 188, 199, 205, 212, 225, 236, 259, 1013),
    paragraphs_per_section=(1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 10, 11, 14, 18,
                            25, 92),
    heading_quantiles=(1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 11, 13, 24),
    type_length_quantiles=(1, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12,
                           14, 142),
    sections_per_part=(7, 8, 12, 15, 17, 22, 23, 23, 25, 26, 28, 30, 31, 33, 34, 41, 54,
                       55, 57, 58, 62, 65, 81, 148, 175, 190),
    versions_per_paragraph=((1, 8214.0), (2, 1309.0), (3, 406.0), (4, 185.0), (5, 71.0)),
    zipf_s=1.254,
    heaps_k=7.637,
    heaps_beta=0.5869,
    dates_a=4.3,
    dates_b=0.838,
    removed_fraction=0.022,
    window=("2017-01-01", "2026-08-25"),
    head_words=(
        "the", "of", "a", "to", "or", "in", "and", "an", "for", "is", "under", "this",
        "employee", "agency", "as", "by", "be", "that", "pay", "not", "may", "on", "rate",
        "service", "must", "leave", "from", "any", "if", "position", "which", "at",
        "shall", "who", "u.s.c.", "period", "(2)", "employee's", "paragraph", "(a)",
        "part", "(b)", "are", "section", "when", "other", "(1)", "time", "opm",
        "employees", "has", "date", "than", "individual", "such", "covered", "enrollment",
        "subpart", "(c)", "will", "appointment", "work", "means", "basic", "after",
        "provided", "within", "(3)", "duty", "during", "hours", "annual", "his", "each",
        "was", "one", "performance", "special", "office", "eligible", "her", "family",
        "executive", "health", "required", "plan", "federal", "coverage", "united",
        "employing", "except", "change", "more", "schedule", "wage", "before", "days",
        "system", "grade", "made", "general"),
    measured_rows=13145,
    measured_from="data/warrant.sqlite3 @ 2026-08-30",
)


# ------------------------------------------------------------------------ generation


def _draw(table: Sequence[int], u: float) -> int:
    """Interpolate a quantile table at ``u`` in [0, 1)."""
    pos = u * (len(table) - 1)
    lo = min(len(table) - 2, int(pos))
    frac = pos - lo
    return int(round(table[lo] + frac * (table[lo + 1] - table[lo])))


def _draw_tokens(shape: CorpusShape, u: float) -> int:
    if u < 0.99:
        return max(1, _draw(shape.token_quantiles[:-1], u / 0.99))
    return max(1, _draw(shape.token_tail_quantiles, (u - 0.99) / 0.01))


_CONSONANTS = "bcdfghklmnprstvwz"
_VOWELS = "aeiou"


class Vocabulary:
    """A Zipfian vocabulary sized for a token budget by Heaps' law.

    The head is the real corpus's own top words, because they are what makes lexical
    retrieval expensive: `fts_query` ORs every query token, so "the" contributes a postings
    list the length of the corpus to a query that meant nothing by it. Beyond the head the
    types are synthetic, generated at the observed type-length distribution so the term
    dictionary costs what a real one of that size would.
    """

    def __init__(self, shape: CorpusShape, n_tokens: int, *, seed: int) -> None:
        target = int(shape.heaps_k * max(n_tokens, 1) ** shape.heaps_beta)
        self.size = max(len(shape.head_words) + 1, target)
        rng = random.Random(seed ^ 0x5CA1E)
        words = list(shape.head_words)
        used = set(words)
        for i in range(len(words), self.size):
            words.append(_pseudo_word(shape, rng, used, i))
        self.words: list[str] = words
        ranks = np.arange(1, self.size + 1, dtype=np.float64)
        weights = ranks ** -shape.zipf_s
        self._cdf = np.cumsum(weights)
        self._cdf /= self._cdf[-1]

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """``size`` word indices, Zipf-distributed."""
        return np.searchsorted(self._cdf, rng.random(size))

    def word(self, index: int) -> str:
        return self.words[index]


def _pseudo_word(shape: CorpusShape, rng: random.Random, used: set[str], index: int) -> str:
    length = min(MAX_TYPE_LEN, max(1, _draw(shape.type_length_quantiles, rng.random())))
    for _ in range(8):
        chars = [(_CONSONANTS if i % 2 == 0 else _VOWELS)[
            rng.randrange(len(_CONSONANTS if i % 2 == 0 else _VOWELS))]
            for i in range(length)]
        word = "".join(chars)
        if word not in used:
            used.add(word)
            return word
    word = f"{''.join(chars)}{index:x}"
    used.add(word)
    return word


class _Tape:
    """A refilling block of pre-drawn word indices.

    Drawing 20 million tokens one at a time through `random` costs about 40 seconds; one
    numpy `searchsorted` over a million-element block costs about 30 ms. The generator is
    not the measurement, and it should not dominate the run.
    """

    def __init__(self, vocab: Vocabulary, rng: np.random.Generator,
                 block: int = 1 << 20) -> None:
        self._vocab = vocab
        self._rng = rng
        self._block = block
        self._buf = vocab.draw(rng, block)
        self._at = 0

    def take(self, k: int) -> str:
        if self._at + k > self._buf.size:
            self._buf = self._vocab.draw(self._rng, max(self._block, k))
            self._at = 0
        chunk = self._buf[self._at:self._at + k]
        self._at += k
        words = self._vocab.words
        return " ".join([words[i] for i in chunk])


def _date_pool(shape: CorpusShape, n_parts: int, rng: np.random.Generator) -> list[str]:
    """The distinct snapshot dates a corpus of this many parts would carry.

    Bounded by the days in the window: more parts cannot invent more amendment dates than
    the calendar has, and the bound is what keeps the admitted-set cache key space finite.
    """
    start = date.fromisoformat(shape.window[0])
    end = date.fromisoformat(shape.window[1])
    span = max(1, (end - start).days)
    want = int(round(shape.dates_a * max(n_parts, 1) ** shape.dates_b))
    n = max(1, min(span, want))
    offsets = sorted(set(rng.choice(span, size=min(n * 2, span), replace=False).tolist()))
    return [(start + timedelta(days=int(o))).isoformat() for o in offsets[:n]]


def generate_chunks(shape: CorpusShape, n_versions: int, *,
                    seed: int = 0, title: int = 5) -> Iterator[Chunk]:
    """Yield about ``n_versions`` synthetic chunk versions with the shape of the real corpus.

    Whole paragraphs are emitted, so the count overshoots the target by fewer than
    ``max(versions_per_paragraph)`` rows -- stopping mid-paragraph would leave a closed
    interval with no successor and quietly move the in-force fraction, which is the one
    number the predicate measurements are most sensitive to.

    Structure follows the regulation: parts hold sections, sections hold paragraphs,
    paragraphs hold valid-time versions. Growth is in *parts*, because that is what "more of
    the CFR" means; sections-per-part and paragraphs-per-section are properties of how
    regulation is written, not of how much of it you ingested.
    """
    rng = np.random.default_rng(seed)
    per_part = (shape.mean_versions_per_paragraph * shape.mean_paragraphs_per_section
                * (sum(shape.sections_per_part) / len(shape.sections_per_part)))
    n_parts = max(1, int(math.ceil(n_versions / max(per_part, 1.0))))
    dates = _date_pool(shape, n_parts, rng)
    floor = shape.window[0]

    tokens_est = int(n_versions * shape.mean_tokens)
    vocab = Vocabulary(shape, tokens_est, seed=seed)
    tape = _Tape(vocab, rng)

    version_values = [v for v, _ in shape.versions_per_paragraph]
    version_weights = np.array([w for _, w in shape.versions_per_paragraph], dtype=np.float64)
    version_weights /= version_weights.sum()

    emitted = 0
    part_index = 0
    while emitted < n_versions:
        part = f"{900 + part_index // 900}{part_index % 900:03d}"
        n_sections = max(1, _draw(shape.sections_per_part, float(rng.random())))
        for s in range(n_sections):
            if emitted >= n_versions:
                break
            section_id = f"{part}.{101 + s}"
            subpart = chr(ord("A") + s // 12 % 26)
            heading = tape.take(max(1, _draw(shape.heading_quantiles, float(rng.random()))))
            n_paras = max(1, _draw(shape.paragraphs_per_section, float(rng.random())))
            for p in range(n_paras):
                if emitted >= n_versions:
                    break
                anchor = _designator(p)
                chunk_id = f"{section_id}#{anchor}"
                n_v = int(rng.choice(version_values, p=version_weights))
                # The first version always opens at the history floor. That is the real
                # corpus's own shape: eCFR records a version date whenever a part is
                # amended, so the absence of one before the first snapshot is positive
                # evidence the text did not change, and ingest backfills to the floor
                # (ARCHITECTURE.md section 2).
                starts = [floor]
                if n_v > 1 and len(dates) > 1:
                    picked = rng.choice(len(dates), size=min(n_v - 1, len(dates)),
                                        replace=False)
                    starts += sorted(dates[int(i)] for i in picked)
                    starts = sorted(set(starts))
                removed = bool(rng.random() < shape.removed_fraction)
                for vi, valid_from in enumerate(starts):
                    last = vi == len(starts) - 1
                    valid_to = None if (last and not removed) else (
                        starts[vi + 1] if not last else shape.window[1])
                    yield Chunk(
                        chunk_id=chunk_id,
                        section_id=section_id,
                        title=title,
                        part=part,
                        subpart=subpart,
                        anchor=anchor,
                        heading=heading,
                        text=tape.take(_draw_tokens(shape, float(rng.random()))),
                        valid_from=valid_from,
                        valid_to=valid_to,
                        source_snapshot=valid_from,
                        config_hash="synthetic",
                    )
                    emitted += 1
        part_index += 1


def _designator(i: int) -> str:
    """(a), (b) ... then a1, a2 for sections wider than the alphabet."""
    if i < 26:
        return chr(ord("a") + i)
    return f"{chr(ord('a') + i % 26)}{i // 26}"


# ---------------------------------------------------------------------------- build


@dataclass
class BuildReport:
    rows: int
    seconds: float
    db_bytes: int
    fts_bytes: int
    text_bytes: int
    parts: int
    sections: int
    paragraphs: int
    dates: int
    in_force: int

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else 0.0


def build_store(path: Path, n_versions: int, *, shape: CorpusShape = REAL_5CFR,
                seed: int = 0, batch: int = 5000) -> BuildReport:
    """Generate and insert ``n_versions`` chunks, timing the insert.

    Batched at 5,000 because the FTS5 `AFTER INSERT` trigger fires per row and a single
    transaction over half a million of them holds the whole index delta in memory. The
    timing therefore includes trigger work and index maintenance, which is what a build
    actually costs -- an insert measured with the triggers dropped would be a number about a
    schema nobody runs.
    """
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            side.unlink()

    started = time.perf_counter()
    rows = 0
    with Store(path) as store:
        pending: list[Chunk] = []
        for chunk in generate_chunks(shape, n_versions, seed=seed):
            pending.append(chunk)
            if len(pending) >= batch:
                rows += store.add(pending)
                pending.clear()
        if pending:
            rows += store.add(pending)
        db = store.db
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        elapsed = time.perf_counter() - started
        stats = _store_stats(db)
    return BuildReport(rows=rows, seconds=elapsed, db_bytes=path.stat().st_size, **stats)


def _store_stats(db: sqlite3.Connection) -> dict[str, int]:
    """Size and structure of a built store, read from SQLite's own shadow tables.

    `dbstat` is not compiled into CPython's SQLite, so the FTS5 share cannot be read off a
    page census. The shadow tables give it exactly: `chunk_fts_data` holds the segment
    blocks, which is the index.
    """
    one = lambda sql: db.execute(sql).fetchone()[0] or 0  # noqa: E731
    return {
        "fts_bytes": one("SELECT SUM(LENGTH(block)) FROM chunk_fts_data"),
        "text_bytes": one("SELECT SUM(LENGTH(text) + LENGTH(COALESCE(heading, ''))) "
                          "FROM chunk"),
        "parts": one("SELECT COUNT(DISTINCT part) FROM chunk"),
        "sections": one("SELECT COUNT(DISTINCT section_id) FROM chunk"),
        "paragraphs": one("SELECT COUNT(DISTINCT chunk_id) FROM chunk"),
        "dates": one("SELECT COUNT(DISTINCT valid_from) FROM chunk"),
        "in_force": one("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL "
                        "AND system_to IS NULL"),
    }


def synthetic_index(store: Store, *, dim: int = DENSE_DIM, seed: int = 0) -> DenseIndex:
    """A dense index of random unit vectors over every believed row.

    The vectors are noise and the *scores* they produce are meaningless. The *cost* is not:
    `DenseIndex.search` is a full matmul, an `np.isin` against the admitted set, a `where`
    and an `argpartition`, and every one of those is oblivious to the values it operates on.
    What it measures is the shape of the matrix, which is the thing that scales.
    """
    ids = np.asarray([r[0] for r in store.db.execute(
        "SELECT id FROM chunk WHERE system_to IS NULL ORDER BY id")], dtype=np.int64)
    rng = np.random.default_rng(seed ^ 0xD3115E)
    vectors = rng.standard_normal((ids.size, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return DenseIndex(ids=ids, vectors=vectors, model="synthetic", config_hash="scale")


# ----------------------------------------------------------------------- measurement


def rss_bytes() -> int:
    """Resident memory of this process, or 0 where the platform will not say.

    Stdlib only. psutil is not a dependency and must not become one for a benchmark: a
    reviewer runs `pip install -e .` to reproduce the failure budget, not to acquire a
    process-inspection library.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        psapi = ctypes.WinDLL("psapi")
        handle = ctypes.WinDLL("kernel32").GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            import os
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes here and macOS reports bytes. Only macOS reaches this.
        return int(peak)
    except Exception:  # pragma: no cover - platform of last resort
        return 0


def set_bytes(values: set[int]) -> int:
    """Bytes a ``set[int]`` of store row ids actually costs.

    The set object alone is a third of it. Row ids run past 256, so none of them are
    CPython's cached small integers and each one is a separate 28-byte object.
    """
    return sys.getsizeof(values) + sum(sys.getsizeof(v) for v in values)


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation: a p95 of 30 samples is one observation,
    and interpolating between two of them dresses that up as more resolution than it has."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(p / 100 * len(ordered))) - 1))
    return ordered[idx]


def bootstrap_ci(samples: Sequence[float], p: float, *, resamples: int = 400,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval on a percentile.

    Reported because these numbers were taken on a contended machine and a bare p95 from 60
    queries invites a reader to believe a digit that is not there.
    """
    if len(samples) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(samples)
    draws = sorted(percentile([samples[rng.randrange(n)] for _ in range(n)], p)
                   for _ in range(resamples))
    return (draws[int(0.025 * resamples)], draws[min(resamples - 1, int(0.975 * resamples))])


@dataclass
class StageStats:
    name: str
    samples: list[float] = field(default_factory=list)

    def summary(self, *, seed: int = 0) -> dict[str, float]:
        return {
            "p50": percentile(self.samples, 50),
            "p95": percentile(self.samples, 95),
            "p50_lo": bootstrap_ci(self.samples, 50, seed=seed)[0],
            "p50_hi": bootstrap_ci(self.samples, 50, seed=seed)[1],
            "p95_lo": bootstrap_ci(self.samples, 95, seed=seed)[0],
            "p95_hi": bootstrap_ci(self.samples, 95, seed=seed)[1],
            "n": float(len(self.samples)),
        }


def sample_queries(shape: CorpusShape, *, n: int, seed: int, content_only: bool = False,
                   vocab: Vocabulary | None = None, tokens: int = 10) -> list[str]:
    """Natural-shaped queries drawn from the corpus's own token distribution.

    ``content_only`` drops the top 100 ranks -- the function words. It is not a realistic
    workload; it is the control that separates "the corpus got bigger" from "`fts_query`
    ORed 'the' into the query", which are two different findings with two different fixes.
    """
    if vocab is None:
        vocab = Vocabulary(shape, 512, seed=seed)
    rng = np.random.default_rng(seed ^ 0x0FFEE)
    out = []
    for _ in range(n):
        idx = vocab.draw(rng, tokens * 4)
        if content_only:
            idx = idx[idx >= len(shape.head_words)]
        picked = idx[:tokens]
        if picked.size == 0:
            picked = vocab.draw(rng, tokens)
        out.append(" ".join(vocab.word(int(i)) for i in picked))
    return out


def measure_stages(store: Store, index: DenseIndex | None, queries: Sequence[str], *,
                   as_of: str, candidates_lexical: int = 100, candidates_dense: int = 100,
                   reranker: object | None = None, rerank_top_k: int = 50,
                   dim: int = DENSE_DIM, seed: int = 0) -> dict[str, StageStats]:
    """Per-stage wall-clock, one sample per query, in milliseconds.

    Stages are driven directly rather than through `Retriever.retrieve` for one reason: the
    dense stage there begins with `DenseIndex.encode`, which needs a 130 MB torch encoder
    and costs the same on every corpus size. Including it would add a constant to every row
    and hide the slope, which is the entire question. Query encoding is measured once,
    separately, and stated as the constant it is.

    `predicates` is measured twice. Cold is what a first request at a new as-of date pays;
    warm is what every request after it pays, and the gap is the whole argument for the
    cache in `Store.candidate_ids`.
    """
    stats = {name: StageStats(name) for name in
             ("predicates_cold", "predicates_warm", "lexical", "dense", "fusion",
              "rerank", "isin")}
    rng = np.random.default_rng(seed ^ 0xA11)
    for query in queries:
        store._admits.clear()  # noqa: SLF001 - measuring the cold path is the point
        t = time.perf_counter()
        allowed = store.candidate_ids(valid_date=as_of)
        stats["predicates_cold"].samples.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        allowed = store.candidate_ids(valid_date=as_of)
        stats["predicates_warm"].samples.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        rows = store.search(fts_query(query), valid_date=as_of, limit=candidates_lexical)
        stats["lexical"].samples.append((time.perf_counter() - t) * 1000)
        lex = [r["version_id"] for r in rows]

        dense_ids: list[str] = []
        if index is not None:
            vec = rng.standard_normal(dim, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            t = time.perf_counter()
            hits = index.search(vec, allowed=allowed, limit=candidates_dense)
            stats["dense"].samples.append((time.perf_counter() - t) * 1000)
            # The predicate mask on its own. `np.isin` sorts the admitted set on every
            # call, so this is the term that grows with the corpus while the matmul grows
            # with it too -- separating them is what says which remedy would help.
            t = time.perf_counter()
            np.isin(index.ids, np.fromiter(allowed, dtype=index.ids.dtype,
                                           count=len(allowed)))
            stats["isin"].samples.append((time.perf_counter() - t) * 1000)
            fetched = store.rows_by_id([i for i, _ in hits])
            dense_ids = [fetched[i]["version_id"] for i, _ in hits if i in fetched]

        rankings = [lex, dense_ids] if dense_ids else [lex]
        t = time.perf_counter()
        fuse(rankings)
        stats["fusion"].samples.append((time.perf_counter() - t) * 1000)

        if reranker is not None:
            head = [c.version_id for c in fuse(rankings)][:rerank_top_k]
            pairs = [(query, r["text"]) for r in store.db.execute(
                f"SELECT text FROM chunk WHERE version_id IN "
                f"({','.join('?' * len(head))})", head)] if head else []
            if pairs:
                t = time.perf_counter()
                reranker.predict(pairs)
                stats["rerank"].samples.append((time.perf_counter() - t) * 1000)
    return stats


@dataclass
class ScalePoint:
    label: str
    rows: int
    build: BuildReport | None
    stages: dict[str, dict[str, float]]
    admitted: int
    admitted_set_bytes: int
    admitted_cache_bytes: int
    dense_matrix_bytes: int
    rss_after_load_bytes: int
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "label": self.label, "rows": self.rows, "admitted": self.admitted,
            "admitted_set_bytes": self.admitted_set_bytes,
            "admitted_cache_bytes": self.admitted_cache_bytes,
            "dense_matrix_bytes": self.dense_matrix_bytes,
            "rss_after_load_bytes": self.rss_after_load_bytes,
            "build": None if self.build is None else vars(self.build),
            "stages": self.stages, "notes": self.notes,
        }


def cache_pressure(store: Store, dates: Sequence[str], *, limit: int = 64) -> int:
    """Bytes the admitted-set cache holds once ``limit`` distinct as-of dates have been seen.

    Measured with `tracemalloc` rather than an RSS delta: CPython's allocator does not
    return freed arenas to the OS, so an RSS delta over a cache that is being filled and
    cleared reports the high-water mark of the whole process and attributes none of it.
    """
    store._admits.clear()  # noqa: SLF001
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for d in list(dates)[:limit]:
        store.candidate_ids(valid_date=d)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return after - before


def run_point(label: str, path: Path, *, rows: int | None = None,
              shape: CorpusShape = REAL_5CFR, seed: int = 0, queries: int = 60,
              build: bool = True, index_path: Path | None = None,
              reranker: object | None = None, as_of: str | None = None,
              notes: str = "") -> ScalePoint:
    """One point on every curve: build (or open) a store, then measure it."""
    report = build_store(path, rows, shape=shape, seed=seed) if (build and rows) else None
    with Store(path) as store:
        stats = _store_stats(store.db)
        n = store.count()
        as_of_date = as_of or shape.window[1]
        if index_path is not None:
            index = DenseIndex.load(index_path)
        else:
            index = synthetic_index(store, seed=seed)
        rss = rss_bytes()

        store._admits.clear()  # noqa: SLF001
        admitted = store.candidate_ids(valid_date=as_of_date)
        admitted_bytes = set_bytes(admitted)
        dates = [r[0] for r in store.db.execute(
            "SELECT DISTINCT valid_from FROM chunk ORDER BY 1")]
        cache_bytes = cache_pressure(store, dates)

        qs = sample_queries(shape, n=queries, seed=seed)
        measured = measure_stages(store, index, qs, as_of=as_of_date, reranker=reranker,
                                  seed=seed)
        summary = {k: v.summary(seed=seed) for k, v in measured.items() if v.samples}
        return ScalePoint(
            label=label, rows=n, build=report, stages=summary,
            admitted=len(admitted), admitted_set_bytes=admitted_bytes,
            admitted_cache_bytes=cache_bytes,
            dense_matrix_bytes=int(index.vectors.nbytes),
            rss_after_load_bytes=rss,
            notes=notes or f"{stats['parts']} parts, {stats['sections']} sections, "
                           f"{stats['dates']} dates, {stats['in_force']} in force",
        )


# --------------------------------------------------------------------------- report


def _mb(b: float) -> str:
    return f"{b / 1e6:,.1f}"


def render_markdown(points: Sequence[ScalePoint]) -> str:
    """The tables that go into results/eval-011. Numbers only; the prose is written by hand."""
    out: list[str] = []
    out.append("| corpus | rows | parts | in force | build s | rows/s | db MB | FTS MB |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in points:
        b = p.build
        if b is None:
            out.append(f"| {p.label} | {p.rows:,} | — | — | — | — | — | — |")
            continue
        out.append(f"| {p.label} | {b.rows:,} | {b.parts:,} | {b.in_force:,} | "
                   f"{b.seconds:,.1f} | {b.rows_per_second:,.0f} | {_mb(b.db_bytes)} | "
                   f"{_mb(b.fts_bytes)} |")

    stages = ("predicates_cold", "predicates_warm", "lexical", "dense", "isin", "fusion",
              "rerank")
    out.append("")
    out.append("| corpus | " + " | ".join(f"{s} p50 / p95" for s in stages) + " |")
    out.append("|---|" + "---:|" * len(stages))
    for p in points:
        cells = []
        for s in stages:
            v = p.stages.get(s)
            cells.append("—" if v is None else f"{v['p50']:.2f} / {v['p95']:.2f}")
        out.append(f"| {p.label} | " + " | ".join(cells) + " |")

    out.append("")
    out.append("| corpus | admitted | admitted set MB | 64-entry cache MB | "
               "dense matrix MB | process RSS MB |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for p in points:
        out.append(f"| {p.label} | {p.admitted:,} | {_mb(p.admitted_set_bytes)} | "
                   f"{_mb(p.admitted_cache_bytes)} | {_mb(p.dense_matrix_bytes)} | "
                   f"{_mb(p.rss_after_load_bytes)} |")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m warrant.bench.scale",
        description="Build synthetic stores at several sizes and measure the mechanism.")
    parser.add_argument("--sizes", default="13145,50000,150000,500000",
                        help="comma-separated chunk-version counts")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the scratch stores and the json record")
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--real", type=Path, default=None,
                        help="a copy of the real store, measured as the anchor point")
    parser.add_argument("--real-index", type=Path, default=None,
                        help="the real dense index stem, e.g. data/dense")
    parser.add_argument("--keep", action="store_true", help="do not delete the stores")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    points: list[ScalePoint] = []
    if args.real is not None:
        points.append(run_point("real 5 CFR", args.real, build=False,
                                index_path=args.real_index, queries=args.queries,
                                seed=args.seed, notes="the measured corpus"))
    for size in (int(s) for s in args.sizes.split(",")):
        path = args.out / f"scale-{size}.sqlite3"
        points.append(run_point(f"synthetic {size:,}", path, rows=size,
                                queries=args.queries, seed=args.seed))
        print(f"done {size}", file=sys.stderr, flush=True)
        if not args.keep:
            for suffix in ("", "-wal", "-shm"):
                side = path.with_name(path.name + suffix)
                if side.exists():
                    side.unlink()

    record = args.out / "scale.json"
    record.write_text(json.dumps([p.to_json() for p in points], indent=2), encoding="utf-8")
    print(render_markdown(points))
    print(f"\nrecord: {record}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
