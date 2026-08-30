"""Configuration, and the config hash that ties a stored answer to the settings behind it.

Every field that can change retrieval behaviour contributes to ``Config.hash``. That hash is
recorded on every trace, so counterfactual replay can say *what changed* rather than merely
*the answer is different now* -- see ARCHITECTURE.md section 8.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class CorpusConfig(BaseModel):
    title: int = 5
    parts: list[str] = Field(default_factory=list)
    history_floor: str = "2017-01-01"
    request_delay_s: float = 1.0
    cache_dir: str = "data/ecfr"
    #: How long a cached /titles.json or /versions response stays authoritative. Snapshots
    #: are immutable and cached forever; these two are where a new amendment first appears,
    #: so pinning them freezes the corpus with no error to say so.
    index_ttl_hours: float = 24.0
    #: How long a recorded 404 is believed before the date is probed again.
    negative_cache_ttl_days: float = 30.0
    #: A 404 within this many days of the title's latest issue date is eCFR's publication
    #: lag, not an absent snapshot, and is never written to the negative cache.
    issue_date_lag_days: int = 30


class StoreConfig(BaseModel):
    path: str = "data/warrant.sqlite3"
    dense_path: str = "data/dense"
    human_benchmark: str = "benchmarks/human.yaml"
    #: Where `warrant autopsy run --json` writes the budget the API and UI read. Recorded
    #: rather than recomputed per request: recomputing takes minutes, and worse, it would let
    #: the dashboard drift from the numbers in results/ that the README quotes.
    budget: str = "results/failure-budget.json"
    #: Recorded request traces. A separate database from the corpus: the corpus is
    #: rebuilt wholesale and traces must survive that, since a trace whose corpus was
    #: replaced is exactly the one worth replaying.
    traces: str = "data/traces.sqlite3"


class DiffConfig(BaseModel):
    wholesale_threshold: float = 0.50
    min_changed_tokens: int = 3


#: Why every model field has a ``revision`` beside it: a bare HuggingFace repo name resolves
#: to whatever ``main`` points at today. Three repositories nobody here controls can therefore
#: change the published numbers without a commit in this one, and the config hash on every
#: stored trace would not move -- so counterfactual replay would report "nothing changed"
#: about the one thing that did. ``null`` keeps today's behaviour; a commit SHA or tag pins it.
_REVISION_DOC = "HuggingFace revision (commit SHA, tag or branch). null = whatever main is."


class DenseConfig(BaseModel):
    enabled: bool = True
    # Small on purpose: 384 dimensions, ~130 MB. A reviewer without a GPU must be able to
    # build this index, and at 13k chunks a larger encoder buys less than the reranker does.
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str | None = Field(default=None, description=_REVISION_DOC)
    batch_size: int = 64


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision: str | None = Field(default=None, description=_REVISION_DOC)


class GenerateConfig(BaseModel):
    #: null means the module default in ``warrant.generate.answer`` (Qwen2.5-1.5B-Instruct).
    #: Naming the same repo in two places is how the two come to disagree.
    model: str | None = None
    revision: str | None = Field(default=None, description=_REVISION_DOC)


class FusionConfig(BaseModel):
    method: str = "rrf"
    k: int = 60


class IndexConfig(BaseModel):
    dense: DenseConfig = Field(default_factory=DenseConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class RetrieveConfig(BaseModel):
    candidates_lexical: int = 100
    candidates_dense: int = 100
    rerank_top_k: int = 30
    final_k: int = 8
    #: Excerpts handed to the generator. Defaults to final_k rather than a second constant:
    #: they were independently 16 and 8 for a while, so retrieval was widened on the
    #: evidence of the failure budget and the generator kept seeing half of what the fix
    #: delivered -- a tuning result silently thrown away one module downstream.
    context_chunks: int | None = None

    @property
    def context_k(self) -> int:
        return self.context_chunks if self.context_chunks is not None else self.final_k


class EvalConfig(BaseModel):
    buckets: list[str] = Field(default_factory=lambda: ["temporal", "generated", "human"])
    bootstrap_samples: int = 1000


class Config(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    diff: DiffConfig = Field(default_factory=DiffConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            path = REPO_ROOT / "configs" / "default.yaml"
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @property
    def cache_path(self) -> Path:
        p = Path(self.corpus.cache_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def store_path(self) -> Path:
        p = Path(self.store.path)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def dense_path(self) -> Path:
        p = Path(self.store.dense_path)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def human_path(self) -> Path:
        p = Path(self.store.human_benchmark)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def traces_path(self) -> Path:
        p = Path(self.store.traces)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def budget_path(self) -> Path:
        p = Path(self.store.budget)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def hash(self) -> str:
        """Stable short hash of every behaviour-affecting setting.

        Only settings the code actually reads belong here. A field that is declared,
        hashed, and never read asserts a difference that does not exist: editing it changes
        the hash, changes no behaviour, and quietly tells a reader it is tunable when it is
        not. Ten such fields were removed rather than documented.
        """
        payload = self.model_dump_json(exclude={"eval"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]
