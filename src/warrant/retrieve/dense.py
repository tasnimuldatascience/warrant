"""Dense retrieval over the bitemporal store.

Vectors live in a single float32 matrix on disk beside an array of store row ids. At ~13k
chunks that is about 20 MB and an exhaustive cosine scan takes single-digit milliseconds, so
an ANN index would add a dependency, a build step and an approximation for no measurable
gain. When the corpus outgrows that, the honest signal is a latency measurement, not a
preference for the fancier structure.

The predicate is applied by **restricting the search space before scoring**, not by filtering
a ranked list afterwards. `Store.candidate_ids` returns the rows admitted by the as-of and
applicability predicates; everything else is masked out before the top-k. This is the dense
equivalent of putting the predicate in the SQL, and it matters for the same measured reason:
in the temporal ablation, superseded near-duplicates crowd correct evidence out of the
candidate list.

The index records the model name and the config hash it was built under. A query issued
against vectors built by a different model is a silent quality regression, so it is refused
rather than served.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..index.store import Store

#: Small on purpose: 384 dimensions, ~130 MB of weights, runs on CPU in a pinch. A reviewer
#: without a GPU has to be able to build this index, and the corpus is small enough that a
#: larger encoder buys less than the failure budget will show the reranker buying.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
#: BGE models are trained with an asymmetric instruction on the query side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class ModelMismatch(RuntimeError):
    """The index on disk was built by a different encoder than the one being queried."""


def uncovered(index: DenseIndex, store) -> int:
    """How many believed chunks the dense index has no vector for.

    Zero for a freshly built index, and non-zero the moment anything is ingested afterwards
    -- which degrades silently and asymmetrically. A chunk with no vector is still found by
    BM25, so it appears in one of the two rank lists RRF fuses instead of two, and its fused
    score is roughly halved against neighbours that are in both. It does not vanish, which
    is what makes it hard to notice: it just quietly loses.

    Counted rather than raised on. An index one document behind the store is a normal state
    between `corpus ingest` and `index build`, and refusing to serve would be worse than
    saying so.
    """
    believed = {r["id"] for r in store.db.execute(
        "SELECT id FROM chunk WHERE system_to IS NULL")}
    return len(believed - set(index.ids.tolist()))


def retrieval_text(row) -> str:
    """The one string every ranking stage sees for a chunk.

    Context first, then the paragraph. This function exists so there is exactly one answer to
    "what text represents this chunk": the section heading used to be prepended here, inside
    the encoder, and nowhere else -- so BM25 matched the bare paragraph, the vectors encoded
    heading-plus-paragraph, and the cross-encoder scored a third variant. Three stages ranking
    three different strings for one row, with no measurement able to see it.

    It matters most for the short paragraphs, which are most of the corpus: 48.2% of in-force
    chunks are under 30 tokens, and "(c) The agency continues its own recruiting efforts."
    carries almost no retrievable signal without the section it sits in.
    """
    context = (row["context"] or "").strip() if "context" in row.keys() else ""
    if not context:
        heading = (row["heading"] or "").strip() if "heading" in row.keys() else ""
        context = heading
    text = row["text"]
    return f"{context}\n{text}" if context else text


@dataclass
class DenseIndex:
    ids: np.ndarray            # store row ids, parallel to `vectors`
    vectors: np.ndarray        # (n, d) float32, L2-normalised
    model: str
    config_hash: str

    # -- persistence -------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        np.save(path.with_suffix(".ids.npy"), self.ids)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"model": self.model, "config_hash": self.config_hash,
                        "revision": self.revision,
                        "n": int(self.ids.size), "dim": int(self.vectors.shape[1])}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, expect_model: str | None = None) -> DenseIndex:
        """Load an index, refusing one built by a different encoder.

        ``ModelMismatch`` existed as a class for a while without anything ever raising it,
        so the docstring promised a guarantee the code did not provide: renaming the encoder
        in config served a stale index silently. ``expect_model`` is what makes the promise
        real, and the serving path passes it.
        """
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if expect_model is not None and meta["model"] != expect_model:
            raise ModelMismatch(
                f"index at {path} was built by {meta['model']!r} but the configured encoder "
                f"is {expect_model!r}; rebuild with `make index`")
        index = cls(
            ids=np.load(path.with_suffix(".ids.npy"), allow_pickle=False),
            vectors=np.load(path.with_suffix(".vectors.npy"), allow_pickle=False),
            model=meta["model"],
            config_hash=meta["config_hash"],
            revision=meta.get("revision"),
        )
        if index.ids.size != index.vectors.shape[0]:
            raise ModelMismatch(
                f"index at {path} is corrupt: {index.ids.size} ids against "
                f"{index.vectors.shape[0]} vectors")
        return index

    @classmethod
    def exists(cls, path: Path) -> bool:
        return path.with_suffix(".meta.json").exists()

    # -- querying ----------------------------------------------------------------

    revision: str | None = None

    def encode(self, text: str) -> np.ndarray:
        """Embed a query with the encoder this index was built by.

        The index owns its encoder rather than the caller choosing one, because a query
        embedded by a different model than the vectors is a silent quality regression --
        the scores stay finite and plausible and the ranking is noise.
        """
        return encode_query(text, model_name=self.model, revision=self.revision)

    def search(self, query_vector: np.ndarray, *, allowed: set[int] | None,
               limit: int) -> list[tuple[int, float]]:
        """Top ``limit`` (row id, score) pairs, restricted to ``allowed`` before ranking.

        Excluded rows are scored and then driven to ``-inf`` rather than sliced out. Slicing
        looks like it should be cheaper and is not: measured on this corpus, gathering the
        9,262 admitted rows into a new matrix costs 2.15 ms and allocates 14.2 MB per query,
        to save 0.14 ms of matmul over the other 3,596. Masking the scores allocates one
        float array, and the ranking semantics are identical -- nothing excluded can reach
        the top-k, which is the property the predicate is for.
        """
        if self.ids.size == 0:
            return []
        scores = self.vectors @ query_vector
        if allowed is not None:
            if not allowed:
                return []
            keep = np.isin(self.ids, np.fromiter(allowed, dtype=self.ids.dtype,
                                                 count=len(allowed)))
            if not keep.any():
                return []
            scores = np.where(keep, scores, -np.inf)
            limit = min(limit, int(keep.sum()))
        take = min(limit, scores.size)
        if take <= 0:
            return []
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top
                if np.isfinite(scores[i])]


#: Encoders are cached per process. Constructing a SentenceTransformer costs seconds and
#: several hundred MB; doing it per query turned a millisecond retrieval into the dominant
#: cost of the whole evaluation, which is the kind of thing a latency budget exists to catch.
_ENCODERS: dict[tuple[str, str | None], object] = {}
_ENCODER_LOCK = threading.Lock()


def _encoder(model_name: str, revision: str | None = None):
    """The encoder, built once per (model, revision) and reused.

    ``revision`` pins a HuggingFace commit. Unpinned, a bare repo name resolves to whatever
    ``main`` is on the day of the run, so a published retrieval number silently depends on a
    repository nobody here controls. Loading is also guarded: check-then-set from a
    threadpool let two cold requests each build a ~130 MB encoder.
    """
    key = (model_name, revision)
    with _ENCODER_LOCK:
        if key not in _ENCODERS:
            from sentence_transformers import SentenceTransformer  # lazily: ~2 GB of torch

            _ENCODERS[key] = SentenceTransformer(model_name, revision=revision)
        return _ENCODERS[key]


@lru_cache(maxsize=4096)
def encode_query(text: str, *, model_name: str = DEFAULT_MODEL,
                 revision: str | None = None) -> np.ndarray:
    """Encode one query. Bounded cache: repeated queries are common in a benchmark sweep,
    and an unbounded one keyed on user input is a memory-exhaustion vector."""
    vec = _encoder(model_name, revision).encode([QUERY_INSTRUCTION + text],
                                      normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def build(store: Store, *, model_name: str = DEFAULT_MODEL, config_hash: str = "",
          batch_size: int = 32, progress: bool = False,
          revision: str | None = None) -> DenseIndex:
    """Embed every believed chunk in the store.

    Every believed chunk, including superseded versions: the store is bitemporal, and an
    index that held only currently-in-force text could not answer a dated question at all.
    """
    rows = store.db.execute(
        "SELECT id, heading, context, text FROM chunk WHERE system_to IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return DenseIndex(np.empty(0, dtype=np.int64), np.empty((0, 384), dtype=np.float32),
                          model_name, config_hash)
    texts = [retrieval_text(r) for r in rows]
    vectors = _encoder(model_name, revision).encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=progress, convert_to_numpy=True,
    )
    return DenseIndex(
        ids=np.asarray([r["id"] for r in rows], dtype=np.int64),
        vectors=np.asarray(vectors, dtype=np.float32),
        model=model_name,
        config_hash=config_hash,
        revision=revision,
    )
