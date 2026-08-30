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
generator, not the system, and no such number is reported anywhere. Quality at scale needs
real text and real questions, and that is a different job.

The shape below was measured from `data/warrant.sqlite3` at 13,145 chunk versions on
2026-08-30; `CorpusShape.measure` recomputes it from any store, and the sweep re-runs the
anchor point against the real store so synthetic and real can be compared directly rather
than assumed equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

#: Dimension of `BAAI/bge-small-en-v1.5`, the encoder `retrieve/dense.py` defaults to. The
#: vectors here are random, but the matrix has to be the shape the real one would be: the
#: cost of an exact scan is entirely (rows x dim x 4 bytes).
DENSE_DIM = 384

#: Type lengths are drawn from a quantile table whose last entry is the observed maximum --
#: 142 characters, one URL-shaped token. Interpolating into that bucket would make ~5% of
#: the synthetic vocabulary absurdly long and inflate the FTS5 term dictionary; real p95 is
#: 14. Clamped, and the clamp costs well under 1% of index bytes.
MAX_TYPE_LEN = 20

#: How many of the vocabulary's top ranks count as function words. `sample_queries` uses it
#: to build the content-only control workload.
HEAD_RANKS = 100


# ---------------------------------------------------------------------------- shape


@dataclass(frozen=True)
class CorpusShape:
    """The statistical shape of a real corpus, in exactly the terms the cost model needs.

    Quantile tables rather than fitted distributions. The token-length distribution is
    heavy-tailed and front-loaded -- 48.2% of in-force chunks under 30 tokens, a maximum of
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
    #: Zipf exponent, fitted on log frequency against log rank over ranks 10..5000.
    zipf_s: float
    #: Heaps' law V = K * N**beta, fitted on a shuffled token stream. Without it a synthetic
    #: 500k-chunk corpus would reuse the real 17,173-type vocabulary and understate both the
    #: FTS5 term dictionary and the number of distinct postings lists a query can touch.
    heaps_k: float
    heaps_beta: float
    #: Distinct snapshot dates grow as parts are added: 4.3 for one part, 66.0 for 26,
    #: fitted as a * parts**b. It matters because the admitted-set cache is keyed per as-of
    #: date, so this is the size of the key space the 64-entry cap is defending against.
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
    #: Their measured share of all tokens, parallel to `head_words` and summing to 0.535.
    #: The head is given empirically rather than left to the Zipf law because English
    #: departs from Zipf exactly there, and by a lot: a pure r**-1.254 sampler gives "the"
    #: 23.8% of the corpus against a measured 6.9%, which would make the stopword postings
    #: list -- the thing that decides what lexical retrieval costs -- 3.5x too long.
    head_weights: tuple[float, ...]
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
    def mean_sections_per_part(self) -> float:
        return sum(self.sections_per_part) / len(self.sections_per_part)

    @property
    def mean_tokens(self) -> float:
        """Mean tokens per chunk under the sampler in `_draw_tokens`, not under the corpus.

        The two agree to within a token, and it is the sampler's mean that has to be right
        here: it is what sizes the vocabulary and predicts the index.
        """
        return (0.99 * _quantile_mean(self.token_quantiles[:-1])
                + 0.01 * _quantile_mean(self.token_tail_quantiles))

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
            # A paragraph counts as removed when no version of it is open-ended.
            closed[chunk_id] = closed.get(chunk_id, True) and valid_to is not None

        counts: dict[int, float] = {}
        for v in versions.values():
            counts[v] = counts.get(v, 0.0) + 1.0

        ranked = ranked_vocabulary(texts)
        n_tokens = sum(c for _, c in ranked)
        # FTS5's unicode61 tokenizer drops a term with no alphanumeric character, so a
        # head rank spent on the corpus's one U+FFFD replacement char indexes nothing.
        indexable = [(w, c) for w, c in ranked if any(ch.isalnum() for ch in w)]
        head = indexable[:HEAD_RANKS]
        head_total = sum(c for _, c in indexable) or 1
        beta = _heaps_beta(texts, seed)
        dates_a, dates_b = _dates_fit(dates_by_part)
        all_dates = sorted({r[3] for r in rows})
        return cls(
            token_quantiles=_quantiles(tokens, 100),
            token_tail_quantiles=tuple(
                tokens[min(len(tokens) - 1, int((99 + i * 0.1) / 100 * len(tokens)))]
                for i in range(11)),
            paragraphs_per_section=_quantiles(sorted(len(v) for v in paras.values()), 20),
            heading_quantiles=_quantiles(sorted(len((r[5] or "").split()) for r in rows), 20),
            type_length_quantiles=_quantiles(sorted(len(t) for t, _ in ranked), 20),
            sections_per_part=tuple(sorted(len(v) for v in sections.values())),
            versions_per_paragraph=tuple(sorted(counts.items())),
            zipf_s=_zipf_exponent(ranked),
            heaps_k=len(ranked) / (n_tokens ** beta) if n_tokens else 1.0,
            heaps_beta=beta,
            dates_a=dates_a,
            dates_b=dates_b,
            removed_fraction=sum(closed.values()) / len(closed),
            window=(all_dates[0], all_dates[-1]),
            head_words=tuple(w for w, _ in head),
            head_weights=tuple(c / head_total for _, c in head),
            measured_rows=len(rows),
            measured_from=source,
        )


def ranked_vocabulary(texts: Sequence[str]) -> list[tuple[str, int]]:
    """Types by descending frequency, ties broken lexically so the order is deterministic."""
    freq: dict[str, int] = {}
    for t in texts:
        for w in t.lower().split():
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))


def _quantiles(values: Sequence[int], steps: int) -> tuple[int, ...]:
    n = len(values)
    return tuple(values[min(n - 1, int(i / steps * n))] for i in range(steps + 1))


def _quantile_mean(table: Sequence[int]) -> float:
    """Mean of the distribution the table describes, under linear interpolation."""
    return sum((table[i] + table[i + 1]) / 2 for i in range(len(table) - 1)) / (len(table) - 1)


def _zipf_exponent(ranked: Sequence[tuple[str, int]]) -> float:
    """Least-squares slope of log frequency against log rank, over ranks 10..5000.

    Not from rank 1: the very head of an English corpus sits above the Zipf line, and
    fitting through it drags the exponent toward 1 and flattens the tail that the term
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

    Shuffled because the store is ordered by part and each part has its own jargon; in
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
            if first is None and n >= 1000:
                first = (n, len(seen))
    if first is None or n <= first[0]:
        return 0.6
    return (math.log(len(seen)) - math.log(first[1])) / (math.log(n) - math.log(first[0]))


def _dates_fit(dates_by_part: dict[str, set[str]]) -> tuple[float, float]:
    """Fit distinct-dates = a * parts**b through two exact points: one part, and all of them.

    Two points because there are only 26 parts to fit on and the curve between them is
    monotone and smooth. A least-squares fit over random subsets moved the exponent by 0.01
    and needed a seed to be reproducible, which is a worse trade than it sounds.
    """
    parts = sorted(dates_by_part)
    one = sum(len(dates_by_part[p]) for p in parts) / len(parts)
    allp = len({d for p in parts for d in dates_by_part[p]})
    if len(parts) < 2 or one <= 0 or allp <= one:
        return (max(one, 1.0), 1.0)
    return (one, math.log(allp / one) / math.log(len(parts)))


#: Measured from `data/warrant.sqlite3` on 2026-08-30: 13,145 chunk versions, 10,185
#: paragraphs, 1,320 sections, 26 parts of 5 CFR, 66 distinct snapshot dates, 17,173
#: vocabulary types over 514,629 tokens. Baked in so the generator runs without a store --
#: the unit tests need it, and a reviewer with no `data/` still gets the shape the published
#: numbers were taken at. `CorpusShape.measure` regenerates it from any store.
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
        "service", "must", "with", "5", "leave", "from", "any", "if", "position", "which",
        "at", "shall", "who", "u.s.c.", "period", "(2)", "employee's", "paragraph", "(a)",
        "part", "(b)", "are", "section", "when", "other", "(1)", "time", "opm",
        "employees", "has", "date", "than", "covered", "individual", "such", "enrollment",
        "subpart", "(c)", "will", "appointment", "work", "means", "basic", "after",
        "provided", "within", "(3)", "duty", "during", "hours", "annual", "his", "each",
        "was", "one", "performance", "special", "eligible", "office", "her", "family",
        "executive", "health", "required", "plan", "federal", "coverage", "united",
        "employing", "except", "change", "more", "schedule", "wage", "before",
        "competitive", "official"),
    head_weights=(
        0.069185, 0.044226, 0.025397, 0.023383, 0.022549, 0.020814, 0.018287, 0.015902,
        0.013949, 0.011116, 0.011006, 0.010064, 0.009785, 0.007876, 0.007835, 0.007527,
        0.007281, 0.007067, 0.006907, 0.005881, 0.005863, 0.005499, 0.004959, 0.004698,
        0.004413, 0.004326, 0.003941, 0.003929, 0.003759, 0.003645, 0.003639, 0.003585,
        0.003561, 0.003520, 0.003277, 0.003277, 0.003263, 0.003177, 0.003021, 0.002993,
        0.002958, 0.002860, 0.002809, 0.002790, 0.002692, 0.002692, 0.002584, 0.002563,
        0.002541, 0.002488, 0.002300, 0.002292, 0.002230, 0.002144, 0.002036, 0.002020,
        0.002020, 0.002020, 0.001997, 0.001981, 0.001966, 0.001954, 0.001948, 0.001913,
        0.001911, 0.001856, 0.001852, 0.001842, 0.001836, 0.001793, 0.001793, 0.001750,
        0.001735, 0.001713, 0.001703, 0.001693, 0.001652, 0.001623, 0.001615, 0.001566,
        0.001537, 0.001537, 0.001435, 0.001421, 0.001419, 0.001415, 0.001370, 0.001363,
        0.001355, 0.001327, 0.001316, 0.001306, 0.001294, 0.001284, 0.001272, 0.001267,
        0.001255, 0.001245, 0.001237, 0.001227),
    measured_rows=13145,
    measured_from="data/warrant.sqlite3 @ 2026-08-30",
)


# ------------------------------------------------------------------------ generation


def _draw(table: Sequence[int], u: float) -> int:
    """Interpolate a quantile table at ``u`` in [0, 1]."""
    pos = min(max(u, 0.0), 1.0) * (len(table) - 1)
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
    """A ranked word list plus the unigram distribution over it.

    Two ways in. `for_corpus` builds a synthetic vocabulary: the real corpus's measured head
    (`head_words` at `head_weights`) followed by a Zipf tail of pseudo-words, sized so the
    number of types the corpus *realises* matches Heaps' law. `from_texts` reads a real
    store's own ranked vocabulary and its own frequencies, which is how the real anchor
    point gets queries that match anything at all.
    """

    def __init__(self, words: Sequence[str], weights: Sequence[float]) -> None:
        if not words:
            raise ValueError("a vocabulary needs at least one word")
        self.words: list[str] = list(words)
        self.size = len(self.words)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self._cdf = np.cumsum(self.weights)
        self._cdf[-1] = 1.0

    @classmethod
    def for_corpus(cls, shape: CorpusShape, n_tokens: int, *, seed: int) -> Vocabulary:
        head = np.asarray(shape.head_weights, dtype=np.float64)
        tail_mass = max(1e-9, 1.0 - float(head.sum()))
        target = shape.heaps_k * max(n_tokens, 1) ** shape.heaps_beta
        size = _tail_size(target - head.size, max(n_tokens, 1), tail_mass,
                          shape.zipf_s, head.size)
        rng = random.Random(seed ^ 0x5CA1E)
        words = list(shape.head_words)
        used = set(words)
        for i in range(size):
            words.append(_pseudo_word(shape, rng, used, i))
        return cls(words, np.concatenate([head, _tail_weights(size, tail_mass, shape.zipf_s,
                                                              head.size)]))

    @classmethod
    def from_texts(cls, texts: Sequence[str], *, limit: int = 100_000) -> Vocabulary:
        ranked = ranked_vocabulary(texts)[:limit]
        return cls([w for w, _ in ranked], [float(c) for _, c in ranked])

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """``size`` word indices drawn from the unigram distribution."""
        return np.minimum(np.searchsorted(self._cdf, rng.random(size)), self.size - 1)

    def word(self, index: int) -> str:
        return self.words[index]


def _tail_weights(size: int, mass: float, zipf_s: float, offset: int) -> np.ndarray:
    """Zipf weights for ranks ``offset+1 .. offset+size``, summing to ``mass``."""
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    w = np.arange(offset + 1, offset + size + 1, dtype=np.float64) ** -zipf_s
    return w * (mass / w.sum())


def _tail_size(want_types: float, n_tokens: int, mass: float, zipf_s: float,
               offset: int) -> int:
    """How many latent tail types it takes to *realise* ``want_types`` in ``n_tokens`` draws.

    Not the same number. A type with expected count well under 1 usually never appears, so
    a Zipf sampler over V latent types yields far fewer than V distinct terms -- at the 13k
    anchor, 17,228 latent gave 13,064 realised, and an index built from that is 24% short of
    the vocabulary the real corpus has. Expected distinct terms is
    ``sum(1 - exp(-N * p_r))``, which is monotone in V, so bisection is exact enough.

    The latent count comes out several times the realised one. That is the price of a
    memoryless unigram sampler: real vocabulary growth is driven by morphology and proper
    nouns clustering into documents, and this reproduces the *count* rather than the cause.
    """
    if want_types <= 0:
        return 0
    def realised(v: int) -> float:
        return float(np.sum(-np.expm1(-n_tokens * _tail_weights(v, mass, zipf_s, offset))))

    lo, hi = 1, max(2, int(want_types))
    while realised(hi) < want_types and hi < 1 << 26:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if realised(mid) < want_types:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _pseudo_word(shape: CorpusShape, rng: random.Random, used: set[str], index: int) -> str:
    length = min(MAX_TYPE_LEN, max(1, _draw(shape.type_length_quantiles, rng.random())))
    word = ""
    for _ in range(8):
        word = "".join(
            (_CONSONANTS if i % 2 == 0 else _VOWELS)[
                rng.randrange(len(_CONSONANTS if i % 2 == 0 else _VOWELS))]
            for i in range(length))
        if word not in used:
            used.add(word)
            return word
    word = f"{word}{index:x}"
    used.add(word)
    return word


class _Tape:
    """A refilling block of pre-drawn word indices.

    Drawing 20 million tokens one at a time through `random` costs about 40 seconds; one
    numpy `searchsorted` over a million-element block costs about 30 ms. The generator is
    not the measurement and must not dominate the run.
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
        block = self._buf[self._at:self._at + k]
        self._at += k
        words = self._vocab.words
        return " ".join([words[i] for i in block])


def parts_for(shape: CorpusShape, n_versions: int) -> int:
    """How many parts a corpus of this many chunk versions implies."""
    per_part = (shape.mean_versions_per_paragraph * shape.mean_paragraphs_per_section
                * shape.mean_sections_per_part)
    return max(1, int(math.ceil(n_versions / max(per_part, 1.0))))


def date_pool(shape: CorpusShape, n_parts: int, rng: np.random.Generator) -> list[str]:
    """The distinct snapshot dates a corpus of this many parts would carry.

    Bounded by the days in the window: more parts cannot invent more amendment dates than
    the calendar has, and that bound is what keeps the admitted-set cache's key space finite
    rather than growing with the corpus.
    """
    start = date.fromisoformat(shape.window[0])
    end = date.fromisoformat(shape.window[1])
    span = max(1, (end - start).days)
    want = int(round(shape.dates_a * max(n_parts, 1) ** shape.dates_b))
    n = max(1, min(span, want))
    offsets = sorted(rng.choice(span, size=n, replace=False).tolist())
    return [(start + timedelta(days=int(o))).isoformat() for o in offsets]


def generate_chunks(shape: CorpusShape, n_versions: int, *,
                    seed: int = 0, title: int = 5) -> Iterator[Chunk]:
    """Yield about ``n_versions`` synthetic chunk versions with the shape of the real corpus.

    Whole paragraphs are emitted, so the count overshoots the target by fewer than
    ``max(versions_per_paragraph)`` rows. Stopping mid-paragraph would leave a closed
    interval with no successor and quietly move the in-force fraction, which is the one
    number every predicate measurement here is most sensitive to.

    Structure follows the regulation: parts hold sections, sections hold paragraphs,
    paragraphs hold valid-time versions. Growth is in *parts*, because that is what "more of
    the CFR" means; sections-per-part and paragraphs-per-section are properties of how
    regulation is written, not of how much of it was ingested.
    """
    rng = np.random.default_rng(seed)
    dates = date_pool(shape, parts_for(shape, n_versions), rng)
    floor, ceiling = shape.window

    vocab = Vocabulary.for_corpus(shape, int(n_versions * shape.mean_tokens), seed=seed)
    tape = _Tape(vocab, rng)

    version_values = np.array([v for v, _ in shape.versions_per_paragraph])
    version_weights = np.array([w for _, w in shape.versions_per_paragraph], dtype=np.float64)
    version_weights /= version_weights.sum()

    emitted = 0
    part_index = 0
    while emitted < n_versions:
        part = f"{part_index + 1:04d}"
        n_sections = max(1, _draw(shape.sections_per_part, float(rng.random())))
        for s in range(n_sections):
            if emitted >= n_versions:
                break
            section_id = f"{part}.{101 + s}"
            heading = tape.take(max(1, _draw(shape.heading_quantiles, float(rng.random()))))
            n_paras = max(1, _draw(shape.paragraphs_per_section, float(rng.random())))
            for p in range(n_paras):
                if emitted >= n_versions:
                    break
                anchor = _designator(p)
                # The first version always opens at the history floor. That is the real
                # corpus's own shape: eCFR records a version date whenever a part is
                # amended, so the absence of one before the first snapshot is positive
                # evidence the text did not change, and ingest backfills to the floor
                # (ARCHITECTURE.md section 2).
                starts = [floor]
                n_v = int(rng.choice(version_values, p=version_weights))
                if n_v > 1 and len(dates) > 1:
                    picked = rng.choice(len(dates), size=min(n_v - 1, len(dates)),
                                        replace=False)
                    starts = sorted({floor, *(dates[int(i)] for i in picked)})
                # A repealed paragraph is in force on no date. Never closed at its own
                # start: a zero-width interval satisfies no as-of query at all, which is the
                # trap `Store.close_valid` documents.
                removed = bool(rng.random() < shape.removed_fraction) and starts[-1] < ceiling
                for vi, valid_from in enumerate(starts):
                    last = vi == len(starts) - 1
                    valid_to = (ceiling if removed else None) if last else starts[vi + 1]
                    yield Chunk(
                        chunk_id=f"{section_id}#{anchor}",
                        section_id=section_id,
                        title=title,
                        part=part,
                        subpart=chr(ord("A") + s // 12 % 26),
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
    return chr(ord("a") + i) if i < 26 else f"{chr(ord('a') + i % 26)}{i // 26}"


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


def store_stats(db: sqlite3.Connection) -> dict[str, int]:
    """Size and structure of a built store, read from SQLite's own shadow tables.

    `dbstat` is not compiled into CPython's SQLite, so the FTS5 share cannot be read off a
    page census. The shadow tables give it exactly: `chunk_fts_data` holds the segment
    blocks, which is the index.
    """
    def one(sql: str) -> int:
        return db.execute(sql).fetchone()[0] or 0

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


def build_store(path: Path, n_versions: int, *, shape: CorpusShape = REAL_5CFR,
                seed: int = 0, batch: int = 5000) -> BuildReport:
    """Generate and insert ``n_versions`` chunks, timing the insert.

    Batched at 5,000 because the FTS5 `AFTER INSERT` trigger fires per row and one
    transaction over half a million of them holds the whole index delta in memory. The
    timing therefore includes trigger work and index maintenance, which is what a build
    actually costs -- an insert measured with the triggers dropped would be a number about a
    schema nobody runs.
    """
    unlink_store(path)
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
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        elapsed = time.perf_counter() - started
        stats = store_stats(store.db)
    return BuildReport(rows=rows, seconds=elapsed, db_bytes=path.stat().st_size, **stats)


def unlink_store(path: Path) -> None:
    """Remove a store and both WAL side files. Leaving `-wal` behind reattaches the old
    contents to a fresh file of the same name, which reads as a build that generated
    nothing."""
    for suffix in ("", "-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            side.unlink()


def synthetic_index(store: Store, *, dim: int = DENSE_DIM, seed: int = 0) -> DenseIndex:
    """A dense index of random unit vectors over every believed row.

    The vectors are noise and the *scores* they produce are meaningless. The *cost* is not:
    `DenseIndex.search` is a matmul, an `np.isin` against the admitted set, a `where` and an
    `argpartition`, and every one of those is oblivious to the values it operates on. What
    this measures is the shape of the matrix, which is the thing that scales.
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
        handle = ctypes.WinDLL("kernel32").GetCurrentProcess()
        if ctypes.WinDLL("psapi").GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pass
    try:
        import resource
        # macOS reports ru_maxrss in bytes. It is a peak rather than a current figure, and
        # only macOS reaches this branch.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError):  # pragma: no cover - platform of last resort
        return 0


def set_bytes(values: set[int]) -> int:
    """Bytes a ``set[int]`` of store row ids actually costs.

    The set object alone is about a third of it. Row ids run past 256, so none of them are
    CPython's cached small integers and each is a separate 28-byte object.
    """
    return sys.getsizeof(values) + sum(sys.getsizeof(v) for v in values)


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation: a p95 of 60 samples is three observations,
    and interpolating between two of them dresses that up as resolution it does not have."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


def bootstrap_ci(samples: Sequence[float], p: float, *, resamples: int = 400,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval on a percentile.

    Reported because these runs were taken on a contended machine, and a bare p95 from 60
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
        p50_lo, p50_hi = bootstrap_ci(self.samples, 50, seed=seed)
        p95_lo, p95_hi = bootstrap_ci(self.samples, 95, seed=seed)
        return {
            "p50": percentile(self.samples, 50), "p50_lo": p50_lo, "p50_hi": p50_hi,
            "p95": percentile(self.samples, 95), "p95_lo": p95_lo, "p95_hi": p95_hi,
            "mean": sum(self.samples) / len(self.samples) if self.samples else float("nan"),
            "n": float(len(self.samples)),
        }


def sample_queries(vocab: Vocabulary, *, n: int, seed: int, tokens: int = 10,
                   content_only: bool = False, head_ranks: int = HEAD_RANKS) -> list[str]:
    """Queries drawn from the corpus's own token distribution.

    ``content_only`` drops the top ranks -- the function words. It is not a realistic
    workload; it is the control that separates "the corpus got bigger" from "`fts_query`
    ORed 'the' into the query", which are two different findings with two different fixes.
    """
    rng = np.random.default_rng(seed ^ 0x0FFEE)
    out = []
    for _ in range(n):
        idx = vocab.draw(rng, tokens * 8)
        if content_only:
            idx = idx[idx >= head_ranks]
        picked = idx[:tokens]
        if picked.size == 0:
            picked = vocab.draw(rng, tokens)
        out.append(" ".join(vocab.word(int(i)) for i in picked))
    return out


#: Every stage this harness times, in pipeline order. `isin` is not a pipeline stage: it is
#: the predicate mask inside `DenseIndex.search`, timed separately because it and the matmul
#: both grow with the corpus and only separating them says which remedy would help.
TIMED = ("predicates_cold", "predicates_warm", "lexical", "dense", "isin", "fusion",
         "rerank")


def measure_stages(store: Store, index: DenseIndex | None, queries: Sequence[str], *,
                   as_of: str, candidates_lexical: int = 100, candidates_dense: int = 100,
                   reranker: object | None = None, rerank_top_k: int = 50,
                   seed: int = 0) -> dict[str, StageStats]:
    """Per-stage wall-clock, one sample per query, in milliseconds.

    Stages are driven directly rather than through `Retriever.retrieve`, for one reason: the
    dense stage there begins with `DenseIndex.encode`, which needs a 130 MB torch encoder and
    costs the same on every corpus size. Folding it in would add a constant to every row and
    flatten the slope, which is the entire question. It is a constant and is reported as one.

    `predicates` is measured twice. Cold is what a first request at a new as-of date pays;
    warm is what every request after it pays, and the gap is the whole argument for the cache
    in `Store.candidate_ids`.
    """
    stats = {name: StageStats(name) for name in TIMED}
    rng = np.random.default_rng(seed ^ 0xA11)
    dim = index.vectors.shape[1] if index is not None else DENSE_DIM
    for query in queries:
        store._admits.clear()
        t = time.perf_counter()
        store.candidate_ids(valid_date=as_of)
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

            t = time.perf_counter()
            np.isin(index.ids, np.fromiter(allowed, dtype=index.ids.dtype,
                                           count=len(allowed)))
            stats["isin"].samples.append((time.perf_counter() - t) * 1000)

            fetched = store.rows_by_id([i for i, _ in hits])
            dense_ids = [fetched[i]["version_id"] for i, _ in hits if i in fetched]

        rankings = [lex, dense_ids] if dense_ids else [lex]
        t = time.perf_counter()
        fused = fuse(rankings)
        stats["fusion"].samples.append((time.perf_counter() - t) * 1000)

        if reranker is not None and fused:
            head = [c.version_id for c in fused[:rerank_top_k]]
            pairs = [(query, r[0]) for r in store.db.execute(
                f"SELECT text FROM chunk WHERE version_id IN "
                f"({','.join('?' * len(head))})", head)]
            if pairs:
                t = time.perf_counter()
                reranker.predict(pairs)
                stats["rerank"].samples.append((time.perf_counter() - t) * 1000)
    return stats


def cache_pressure(store: Store, dates: Sequence[str],
                   *, limit: int = 64) -> tuple[int, int]:
    """Bytes the admitted-set cache holds once ``limit`` distinct as-of dates have been seen.

    Measured with `tracemalloc` rather than an RSS delta: CPython's allocator does not return
    freed arenas to the OS, so an RSS delta over a cache being filled reports the high-water
    mark of the whole process and attributes none of it to the cache.

    Returns (bytes, entries). Entries matters because a store with fewer than ``limit``
    distinct dates cannot fill the cache, and the byte figure would otherwise read as if it
    had.
    """
    store._admits.clear()
    keys = list(dates)[:limit]
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for d in keys:
        store.candidate_ids(valid_date=d)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return after - before, len(store._admits)


@dataclass
class ScalePoint:
    label: str
    rows: int
    build: BuildReport | None
    stages: dict[str, dict[str, float]]
    stages_content_only: dict[str, dict[str, float]]
    admitted: int
    admitted_set_bytes: int
    admitted_cache_bytes: int
    admitted_cache_entries: int
    dense_matrix_bytes: int
    rss_bytes: int
    notes: str = ""

    def to_json(self) -> dict:
        out = {k: v for k, v in vars(self).items() if k != "build"}
        out["build"] = None if self.build is None else vars(self.build)
        return out


def run_point(label: str, path: Path, *, rows: int | None = None,
              shape: CorpusShape = REAL_5CFR, seed: int = 0, queries: int = 60,
              build: bool = True, index_path: Path | None = None,
              reranker: object | None = None, as_of: str | None = None,
              notes: str = "") -> ScalePoint:
    """One point on every curve: build (or open) a store, then measure it.

    ``build=False`` against an existing ``path`` is how the real corpus is measured by the
    same harness as the synthetic ones. Its queries are drawn from its *own* vocabulary --
    synthetic pseudo-words match nothing in it, and would time an FTS5 scan over an empty
    postings list rather than a query.
    """
    report = build_store(path, rows, shape=shape, seed=seed) if (build and rows) else None
    with Store(path) as store:
        stats = store_stats(store.db)
        n = store.count()
        as_of_date = as_of or shape.window[1]
        index = (DenseIndex.load(index_path) if index_path is not None
                 else synthetic_index(store, seed=seed))

        if build:
            vocab = Vocabulary.for_corpus(shape, int(n * shape.mean_tokens), seed=seed)
        else:
            vocab = Vocabulary.from_texts([r[0] for r in store.db.execute(
                "SELECT text FROM chunk WHERE system_to IS NULL")])

        store._admits.clear()
        admitted = store.candidate_ids(valid_date=as_of_date)
        admitted_bytes = set_bytes(admitted)
        dates = [r[0] for r in store.db.execute(
            "SELECT DISTINCT valid_from FROM chunk ORDER BY 1")]
        cache_bytes, cache_entries = cache_pressure(store, dates)

        natural = measure_stages(store, index, sample_queries(vocab, n=queries, seed=seed),
                                 as_of=as_of_date, reranker=reranker, seed=seed)
        content = measure_stages(
            store, index,
            sample_queries(vocab, n=queries, seed=seed, content_only=True),
            as_of=as_of_date, seed=seed)
        return ScalePoint(
            label=label, rows=n, build=report,
            stages={k: v.summary(seed=seed) for k, v in natural.items() if v.samples},
            stages_content_only={k: v.summary(seed=seed)
                                 for k, v in content.items() if v.samples},
            admitted=len(admitted), admitted_set_bytes=admitted_bytes,
            admitted_cache_bytes=cache_bytes, admitted_cache_entries=cache_entries,
            dense_matrix_bytes=int(index.vectors.nbytes), rss_bytes=rss_bytes(),
            notes=notes or f"{stats['parts']} parts, {stats['sections']} sections, "
                           f"{stats['dates']} dates, {stats['in_force']} in force")


# --------------------------------------------------------------------------- report


def _mb(b: float) -> str:
    return f"{b / 1e6:,.1f}"


def render_markdown(points: Sequence[ScalePoint]) -> str:
    """The tables that go into results/eval-011. Numbers only; the prose is written by hand."""
    out: list[str] = ["### build", "",
                      "| corpus | rows | parts | in force | build s | rows/s | db MB "
                      "| FTS MB | text MB |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in points:
        b = p.build
        if b is None:
            out.append(f"| {p.label} | {p.rows:,} | — | — | — | — | — | — | — |")
            continue
        out.append(f"| {p.label} | {b.rows:,} | {b.parts:,} | {b.in_force:,} | "
                   f"{b.seconds:,.1f} | {b.rows_per_second:,.0f} | {_mb(b.db_bytes)} | "
                   f"{_mb(b.fts_bytes)} | {_mb(b.text_bytes)} |")

    for heading, key in (("stage latency, natural queries (ms)", "stages"),
                         ("stage latency, content-only queries (ms)",
                          "stages_content_only")):
        out += ["", f"### {heading}", "",
                "| corpus | " + " | ".join(f"{s} p50 / p95" for s in TIMED) + " |",
                "|---|" + "---:|" * len(TIMED)]
        for p in points:
            table = getattr(p, key)
            cells = ["—" if table.get(s) is None
                     else f"{table[s]['p50']:.2f} / {table[s]['p95']:.2f}" for s in TIMED]
            out.append(f"| {p.label} | " + " | ".join(cells) + " |")

    out += ["", "### memory", "",
            "| corpus | admitted | admitted set MB | cache MB (entries) | dense matrix MB "
            "| process RSS MB |", "|---|---:|---:|---:|---:|---:|"]
    for p in points:
        out.append(f"| {p.label} | {p.admitted:,} | {_mb(p.admitted_set_bytes)} | "
                   f"{_mb(p.admitted_cache_bytes)} ({p.admitted_cache_entries}) | "
                   f"{_mb(p.dense_matrix_bytes)} | {_mb(p.rss_bytes)} |")
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
        print("done real", file=sys.stderr, flush=True)
    for size in (int(s) for s in args.sizes.split(",")):
        path = args.out / f"scale-{size}.sqlite3"
        points.append(run_point(f"synthetic {size:,}", path, rows=size,
                                queries=args.queries, seed=args.seed))
        print(f"done {size}", file=sys.stderr, flush=True)
        if not args.keep:
            unlink_store(path)

    record = args.out / "scale.json"
    record.write_text(json.dumps([p.to_json() for p in points], indent=2), encoding="utf-8")
    print(render_markdown(points))
    print(f"\nrecord: {record}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
