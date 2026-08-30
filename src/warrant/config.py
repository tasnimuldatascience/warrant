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


class StoreConfig(BaseModel):
    path: str = "data/warrant.sqlite3"


class ChunkConfig(BaseModel):
    unit: str = "section"
    citation_unit: str = "paragraph"
    parent_expansion: bool = True
    split_tables: bool = False


class DiffConfig(BaseModel):
    wholesale_threshold: float = 0.50
    min_changed_tokens: int = 3


class LexicalConfig(BaseModel):
    k1: float = 1.2
    b: float = 0.75


class DenseConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-m3"
    batch_size: int = 16


class FusionConfig(BaseModel):
    method: str = "rrf"
    k: int = 60


class IndexConfig(BaseModel):
    lexical: LexicalConfig = Field(default_factory=LexicalConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class RetrieveConfig(BaseModel):
    candidates_lexical: int = 100
    candidates_dense: int = 100
    rerank_top_k: int = 30
    final_k: int = 8


class EvalConfig(BaseModel):
    buckets: list[str] = Field(default_factory=lambda: ["temporal", "generated", "human"])
    bootstrap_samples: int = 1000


class Config(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    diff: DiffConfig = Field(default_factory=DiffConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
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
    def hash(self) -> str:
        """Stable short hash of every behaviour-affecting setting."""
        payload = self.model_dump_json(exclude={"eval"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]
