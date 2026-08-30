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
    #: The recorded quality floor. Under results/ rather than data/ because it is an
    #: artifact of a run someone was willing to defend, and it belongs in the repository
    #: next to the report it came from -- a floor that is gitignored gates nothing.
    floor: str = "results/eval-floor.json"
    #: The hand-labelled entailment probe set. Tracked in git like human_benchmark: the
    #: labels are the measurement, and a benchmark that only exists in a results table is
    #: a number nobody can re-run -- which is how this repo shipped a docstring citing 148
    #: pairs that existed nowhere.
    entailment_benchmark: str = "benchmarks/entailment.yaml"


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
    #: Which sources retrieval may draw from, and how weak an authority it may cite.
    #: Empty and null mean "everything in the store", which is correct while the store holds
    #: only the regulation. `max_authority: 2` is the setting to reach for once guidance is
    #: ingested and a fact sheet starts outranking the law it summarises: statute is 1 and
    #: archival OCR is 5, so it reads as "no weaker than regulation".
    sources: list[str] = Field(default_factory=list)
    max_authority: int | None = None
    #: Drop query terms appearing in more than this fraction of indexed documents. null
    #: keeps every term, which is what every published number was measured under. Terms are
    #: ORed, so one common word admits everything it appears in: a query MATCHes 88-90% of
    #: the corpus at every size, and FTS5 has no top-k pruning, so it scores all of them.
    max_document_frequency: float | None = None
    #: Slots of final_k that reference-directed second-hop candidates may take. 0 disables
    #: the hop. See results/eval-013: at 8/depth 3 the share of evidence sets carrying an
    #: unsatisfied reference falls 70.8% -> 32.2% for -0.88 points of sufficiency (CI -2.09
    #: to 0.00, 0 won / 3 lost) -- a benefit on an intermediate metric against a cost on the
    #: outcome metric, so it is off until generated answers can settle it.
    hop_budget: int = 0
    hop_depth: int = 2

    @property
    def context_k(self) -> int:
        return self.context_chunks if self.context_chunks is not None else self.final_k


class FederalRegisterConfig(BaseModel):
    enabled: bool = False
    cache_dir: str = "data/federal_register"
    #: Notices are found by the CFR parts they amend, which is the join to the eCFR corpus.
    #: Narrower than `corpus.parts` on purpose: the search is one request per part and the
    #: point is the parts a benchmark question can actually reach.
    parts: list[str] = Field(default_factory=lambda: ["630"])
    published_since: str = "2000-01-01"
    term: str = ""
    max_documents: int = 200


class UscConfigModel(BaseModel):
    enabled: bool = False
    cache_dir: str = "data/usc"
    title: str = "5"
    #: Chapter 63 is leave -- the statute behind 5 CFR 630, which is where the temporal
    #: benchmark lives. Empty sections and chapters take the whole title (1,163 sections).
    chapters: list[str] = Field(default_factory=lambda: ["63"])
    sections: list[str] = Field(default_factory=list)
    #: Empty discovers the current release point; a value like "119-102" pins it. Pinning is
    #: what makes an ingest reproducible, and it is left open by default so a first run works.
    release_point: str = ""


class OpmGuidanceConfig(BaseModel):
    enabled: bool = False
    cache_dir: str = "data/opm"
    #: Empty uses sources.html.OPM_FACT_SHEETS.
    urls: list[str] = Field(default_factory=list)
    ttl_hours: float = 24.0


class GovInfoConfig(BaseModel):
    enabled: bool = False
    cache_dir: str = "data/govinfo"
    ocr: bool = True
    #: package/granule pairs, e.g. "CFR-2023-title5-vol1/CFR-2023-title5-vol1-sec630-306".
    #: Listed rather than discovered: govinfo's printed volumes are large and the archival
    #: tier is a corroborating source, not a corpus to sweep.
    granules: list[str] = Field(default_factory=list)


class SourcesConfig(BaseModel):
    """The non-eCFR sources, each off by default.

    Off by default because every one of them reaches a different public API, and a clone
    that fails on first run because a network it never asked to use was unavailable is a
    clone nobody evaluates. `warrant corpus build` gives the full P0 corpus with no source
    enabled; `warrant corpus ingest --source usc` is the opt-in.
    """

    federal_register: FederalRegisterConfig = Field(default_factory=FederalRegisterConfig)
    usc: UscConfigModel = Field(default_factory=UscConfigModel)
    opm: OpmGuidanceConfig = Field(default_factory=OpmGuidanceConfig)
    govinfo: GovInfoConfig = Field(default_factory=GovInfoConfig)


class EntailConfig(BaseModel):
    """The NLI verifier, off by default, with the number that decides it written here.

    Off for the same reason ``index.rerank`` carries its own number: on real generator
    output it buys **+2.3 points over span alignment, p=0.55 -- not measurable**. The repo's
    standard for a stage that costs something and cannot be shown to help is that it ships
    behind a flag with the measurement beside it, not that it ships on and nobody re-checks.

    What it *does* buy is a second channel rather than a better score. On claims flipped to
    contradict their evidence it is +49.1 points (p=8.7e-07), because the span aligner finds
    a supporting span in all 22 of them -- a contradiction shares *more* vocabulary with its
    premise than a genuine paraphrase does, so no amount of tuning the aligner reaches it.

    The class breakdown is the caveat that matters: macro accuracy on generator output is
    60.1%, and **half of the generator's neutral pairs come back as entailment**. Reporting
    only the 86.8% micro number would ship a verifier that is 50% accurate on exactly the
    cases it exists to catch.
    """

    enabled: bool = False
    model: str = "cross-encoder/nli-deberta-v3-base"
    revision: str | None = None
    #: Measured peak at 458 pairs/s on GPU; an answer is 2.39 pairs, so one batched call.
    batch_size: int = 16
    #: Fitted out-of-fold; leave-one-section-out refits land in 1.66-1.74. It moves the
    #: verdict, so it is hashed into the config like every other behaviour-affecting field.
    temperature: float = 1.72


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
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    entail: EntailConfig = Field(default_factory=EntailConfig)
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
    def entailment_path(self) -> Path:
        p = Path(self.store.entailment_benchmark)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def floor_path(self) -> Path:
        p = Path(self.store.floor)
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
