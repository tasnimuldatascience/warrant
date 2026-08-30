"""HTTP API and static host for the Warrant UI.

Endpoints exist to make the system's distinctive behaviour *visible*, not to be a generic
search API. Each one answers a question the README makes a claim about:

    /api/meta                    what is in the corpus, and the date range the slider spans
    /api/ask                     the answer for a scope and a date, with claims, spans, trace
    /api/section/{id}            a section's whole version history
    /api/section/{id}/versions   only the dates, for the timeline
    /api/diff                    what changed between two versions of a section
    /api/budget                  the failure budget, read from a recorded run
    /health, /ready              is the process alive, and can it actually answer

Models are built once, in the lifespan, before the first request arrives. They used to be
lazy check-then-set with no lock, which is a thundering herd rather than a cache: measured
cold, a SentenceTransformer is 127 MB, a CrossEncoder 87 MB and Qwen2.5-1.5B 2,944 MB, so
forty concurrent first requests could each construct their own -- ~6.3 GB transient on an
8 GB card, an OOM before the first answer was ever returned. Construction still takes a lock
as a guard for any path that reaches a model without passing through the lifespan.

Both throughput limits are enforced here rather than hoped for, because both were measured
and neither is fixable by asking politely:

    retrieval    peaks at 4 threads (66 QPS) and *falls* to 25.6 QPS at 16 -- THREAD_LIMIT
    generation   21.3 tok/s unbatched, 0.051 req/s -- _GENERATION_SLOT

Every response is a declared model. Annotating handlers ``-> dict`` documented every endpoint
in OpenAPI as ``{"type": "object", "additionalProperties": true}``: no client could be
generated from it and no field rename could ever be caught by it.
"""

from __future__ import annotations

import datetime as dt
import difflib
import logging
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config
from ..generate.answer import excerpts_for
from ..index.store import Store
from ..retrieve.dense import DenseIndex
from ..retrieve.hybrid import Retriever
from ..retrieve.scope import PART_RESTRICTIONS, Scope

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parents[3] / "ui" / "dist"

#: Sync endpoints run in anyio's threadpool, whose default is 40 tokens. Measured on this
#: corpus, retrieval throughput peaks at 4 threads (66 QPS) and *falls* to 25.6 QPS at 16,
#: while per-query latency goes 25 ms -> 624 ms: the work is SQLite, numpy indexing and
#: Python list handling, none of which releases the GIL for long enough to overlap, so the
#: extra threads only add scheduler churn and 25x the tail. Serving 4 at a time and making
#: the rest wait is both faster and honest about the queue.
THREAD_LIMIT = 4

#: One generation at a time. Measured ceiling is 21.3 tok/s unbatched, and a 420-token answer
#: is therefore ~19.7 s, i.e. 0.051 req/s. 100 concurrent requests took ~33 minutes to drain
#: and past ~35 in flight the GPU OOMs, so the queue is bounded and over-capacity load is
#: refused with 503 + Retry-After rather than accepted and abandoned 20 minutes later. This
#: does not raise the ceiling -- nothing here can -- it makes the ceiling visible to callers.
_GENERATION_SLOT = threading.Semaphore(1)
#: How long a request will wait for the slot before being told to come back.
GENERATE_QUEUE_WAIT_S = 20.0
#: Total budget for one request's generation, queue time included.
GENERATE_DEADLINE_S = 90.0
#: 420 max_new_tokens at 21.3 tok/s = 19.7 s for one attempt, and `Generator.answer` retries
#: an unparseable response once. Starting a generation with less budget than this burns the
#: GPU on an answer whose deadline has already expired, so it is refused at the door instead.
GENERATE_FLOOR_S = 40.0
RETRY_AFTER_S = 30

#: Local dev only, and spelled out rather than "*". The API is unauthenticated and read-only,
#: so a wildcard leaks nothing today, but it also means any page on the internet can drive a
#: reviewer's local instance -- including /api/ask, the one endpoint that costs 20 s of GPU.
#: A regex rather than a list because Vite picks a different port whenever 5173 is taken.
LOCAL_ORIGIN_RE = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _date(value: str, param: str) -> str:
    """Validate one ISO date, naming the parameter that was wrong.

    Both halves are load-bearing. The shape check alone accepted ``as_of=2021-13-45``, which
    matched the regex, returned HTTP 200, and matched no row -- an empty answer rendered as a
    confident "nothing is in force", for a date that does not exist. The parse alone is not
    enough either: ``date.fromisoformat`` accepts the compact ``20211345`` form on 3.11+, so
    a caller sending an unpunctuated date would be silently answered about another day.

    422 rather than 400, for every one of the three date parameters, because that is what
    FastAPI already returns for a query string that fails validation: two statuses for one
    class of mistake only teaches a client to treat both as "something went wrong".
    """
    if not _ISO_DATE.match(value):
        raise HTTPException(422, f"{param}: expected an ISO date YYYY-MM-DD, got {value!r}")
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, f"{param}: {value!r} is not a real calendar date") from exc
    return value


# -- response models ------------------------------------------------------------------


class PartSummary(BaseModel):
    part: str
    sections: int
    chunks: int


class MetaResponse(BaseModel):
    chunks: int
    sections: int
    parts: list[PartSummary]
    part_count: int
    #: The answerable range: min and max valid_from over believed chunks. ``latest`` used to
    #: be assigned history_floor, so it equalled ``earliest`` and the UI slider spanned one
    #: day of a corpus that covers eight years.
    earliest: str | None
    latest: str | None
    #: Where ingestion starts, which is a corpus-construction fact, not a data range: eCFR
    #: returns 404 below it. Separate from ``earliest`` because they answer different
    #: questions and conflating them is what produced the one-day slider.
    history_floor: str
    facets: dict[str, list[str]]
    config_hash: str
    final_k: int


class Stage(BaseModel):
    name: str
    out: int


class TraceView(BaseModel):
    admitted: int
    corpus: int
    stages: list[Stage]


class Evidence(BaseModel):
    version_id: str
    chunk_id: str
    section_id: str
    anchor: str | None = None
    heading: str | None = None
    part: str
    subpart: str | None = None
    text: str
    valid_from: str
    valid_to: str | None = None


class SpanView(BaseModel):
    start: int
    end: int
    score: float


class Citation(BaseModel):
    version_id: str
    span: SpanView | None = None


class ClaimView(BaseModel):
    text: str
    grounded: bool
    citations: list[Citation]


class AskResponse(BaseModel):
    #: Quote this to have a maintainer replay the exact request. None when recording is off
    #: or the write failed -- the answer is still valid, only the audit row is missing.
    trace_id: str | None = None
    question: str
    as_of: str
    scope: str
    excluded_parts: list[str]
    trace: TraceView
    evidence: list[Evidence]
    claims: list[ClaimView] = Field(default_factory=list)
    abstained: bool = True
    #: Null when generation did not run, which is a different fact from "it ran and parsed".
    parse_failed: bool | None = None


class Paragraph(BaseModel):
    anchor: str | None = None
    version_id: str
    text: str


class SectionVersion(BaseModel):
    valid_from: str
    valid_to: str | None = None
    heading: str | None = None
    part: str
    paragraphs: list[Paragraph]


class SectionResponse(BaseModel):
    section_id: str
    heading: str | None = None
    part: str
    versions: list[SectionVersion]


class VersionSummary(BaseModel):
    valid_from: str
    valid_to: str | None = None
    heading: str | None = None
    paragraph_count: int


class VersionsResponse(BaseModel):
    section_id: str
    versions: list[VersionSummary]


class DiffOp(BaseModel):
    op: str
    before: str
    after: str


class DiffResponse(BaseModel):
    section_id: str
    a: str
    b: str
    similarity: float
    ops: list[DiffOp]


class BudgetStage(BaseModel):
    stage: str
    failures: int
    share: str


class BudgetRepair(BaseModel):
    repair: str
    implicated: int


class BudgetResponse(BaseModel):
    bucket: str
    config_hash: str
    items: int
    failures: int
    success_rate: float
    observational: list[BudgetStage]
    interventional: list[BudgetRepair]
    stages: list[str]


class HealthResponse(BaseModel):
    """Liveness: this process is running and can route a request. Nothing more."""

    status: str
    uptime_s: float


class ReadyResponse(BaseModel):
    """Readiness: this process can actually answer. Deliberately not the same question.

    Merging the two makes both useless. A liveness probe that fails while models load
    restarts the process into the same load, forever; a readiness probe that passes without a
    corpus sends traffic to an instance that can only return 503.
    """

    ready: bool
    corpus: bool
    models: bool
    generator: bool
    chunks: int | None = None
    detail: str | None = None


# -- runtime --------------------------------------------------------------------------


@dataclass
class Runtime:
    """Process-wide singletons, built in the lifespan and guarded by a lock after that."""

    cfg: Config
    generate: bool = True
    record_traces: bool = True
    _traces: Any = None
    _store: Store | None = None
    _retriever: Retriever | None = None
    _generator: Any = None
    #: Reentrant because `warm` holds it across the properties, which take it themselves.
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _pragma: threading.local = field(default_factory=threading.local)
    stats: Counter[str] = field(default_factory=Counter)
    warm_error: str | None = None
    started: float = field(default_factory=time.monotonic)

    @property
    def store(self) -> Store:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    if not self.cfg.store_path.exists():
                        raise HTTPException(
                            503, "no corpus built; run `make fetch && make build`")
                    # Store hands out a thread-local connection, which is what makes this
                    # safe under FastAPI's threadpool for sync endpoints.
                    self._store = Store(self.cfg.store_path)
        return self._store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    index = None
                    if self.cfg.index.dense.enabled and DenseIndex.exists(self.cfg.dense_path):
                        # expect_model is the whole point of ModelMismatch: without it,
                        # renaming the encoder in config serves the old vectors silently,
                        # and the scores stay finite and plausible while the ranking is noise.
                        index = DenseIndex.load(self.cfg.dense_path,
                                                expect_model=self.cfg.index.dense.model)
                    reranker = None
                    if self.cfg.index.rerank.enabled:
                        from sentence_transformers import CrossEncoder

                        reranker = CrossEncoder(self.cfg.index.rerank.model)
                    self._retriever = Retriever(
                        store=self.store, dense_index=index, reranker=reranker,
                        candidates_lexical=self.cfg.retrieve.candidates_lexical,
                        candidates_dense=self.cfg.retrieve.candidates_dense,
                        rerank_top_k=self.cfg.retrieve.rerank_top_k,
                        final_k=self.cfg.retrieve.final_k,
                        parts_universe=self.cfg.corpus.parts,
                config_hash=self.cfg.hash,
                reranker_model=self.cfg.index.rerank.model,
                    )
        return self._retriever

    @property
    def traces(self) -> Any:
        """The trace store, opened lazily and shared.

        A separate database from the corpus on purpose: the corpus is rebuilt wholesale and
        traces have to survive that, because a trace whose corpus was replaced underneath it
        is precisely the one worth replaying.
        """
        if self._traces is None and self.record_traces:
            with self._lock:
                if self._traces is None:
                    from ..observe.trace_store import TraceStore

                    self._traces = TraceStore(self.cfg.traces_path)
        return self._traces

    @property
    def generator(self) -> Any:
        if self._generator is None and self.generate:
            with self._lock:
                if self._generator is None:
                    from ..generate.model import Generator

                    self._generator = Generator()
        return self._generator

    @property
    def models_loaded(self) -> bool:
        return self._retriever is not None and (self._generator is not None
                                                or not self.generate)

    def read_only(self) -> None:
        """Assert ``PRAGMA query_only`` on this thread's connection, once per thread.

        Three of the four read endpoints never called this. The pragma is per-connection and
        connections are thread-local, so a worker that had only ever served /api/section held
        a fully writable connection to the corpus for the life of the process -- a guarantee
        asserted in a docstring and enforced on one read path out of four.

        Not a FastAPI dependency: sync dependencies and sync endpoints are each dispatched to
        the threadpool separately, so the dependency can set the pragma on one thread while
        the handler queries on another, which is exactly the bug this is meant to close.
        """
        if getattr(self._pragma, "done", False):
            return
        self.store.read_only()
        self._pragma.done = True

    def warm(self) -> None:
        """Build every model once, before the first request can ask for forty of them.

        The dense encoder is not constructed by loading the index -- it is constructed by the
        first ``encode`` -- so one throwaway query is embedded here. Otherwise the 127 MB
        SentenceTransformer is still built lazily, under concurrency, by whichever requests
        happen to arrive first, which is the case this whole method exists to remove.
        """
        with self._lock:
            _ = self.store
            retriever = self.retriever
            if retriever.dense_index is not None:
                retriever.dense_index.encode("warm the query encoder")
            if self.generate:
                _ = self.generator


def _generate_answer(rt: Runtime, question: str, excerpts: list[tuple[str, str, str]], *,
                     as_of: str, scope: str):
    """Run one generation, or refuse honestly if the queue is already full.

    Generation is serialised (`_GENERATION_SLOT`) because the measured ceiling is 21.3 tok/s
    unbatched -- 0.051 req/s -- and nothing in this module can raise it. What it can do is
    stop pretending: without the semaphore, 100 concurrent requests drained in ~33 minutes
    and the GPU OOM'd past ~35 in flight, so callers got a timeout, a truncated stream or a
    500, all of which read as "broken" rather than "at capacity".

    It also happens to serialise the unguarded ``_LOADED`` dict in ``generate.model``, which
    is populated on the first ``complete`` call rather than at construction, so the 2,944 MB
    model can only ever be built by one request at a time.
    """
    deadline = time.monotonic() + GENERATE_DEADLINE_S
    if not _GENERATION_SLOT.acquire(timeout=GENERATE_QUEUE_WAIT_S):
        rt.stats["generate_rejected"] += 1
        raise HTTPException(
            503, f"generator at capacity (1 concurrent, ~{GENERATE_QUEUE_WAIT_S:.0f}s queue)",
            headers={"Retry-After": str(RETRY_AFTER_S)})
    try:
        if deadline - time.monotonic() < GENERATE_FLOOR_S:
            rt.stats["generate_rejected"] += 1
            raise HTTPException(
                503, "queued too long to finish within the request deadline",
                headers={"Retry-After": str(RETRY_AFTER_S)})
        rt.stats["generate_calls"] += 1
        return rt.generator.answer(question, excerpts, as_of=as_of, scope=scope)
    finally:
        _GENERATION_SLOT.release()


def _record(rt: Runtime, trace: Any, answer: Any = None) -> str | None:
    """Persist one request, and never fail the request because of it.

    An audit trail that can take the service down is a liability rather than an asset, so a
    write failure is logged and swallowed. The id is handed back to the caller so a user who
    saw a bad answer can quote something a maintainer can replay.
    """
    traces = rt.traces
    if traces is None:
        return None
    try:
        payload = None
        if answer is not None:
            payload = {
                "abstained": answer.abstained,
                "parse_failed": answer.parse_failed,
                "claims": [{"text": c.text, "evidence": list(c.evidence),
                            "grounded": c.grounded} for c in answer.claims],
            }
        return traces.record(trace, answer=payload)
    except Exception:  # noqa: BLE001 - recording must never break serving
        log.exception("could not record trace")
        return None


def create_app(cfg: Config | None = None, *, generate: bool = True, store: Store | None = None,
               warm: bool = True, thread_limit: int = THREAD_LIMIT) -> FastAPI:
    """Build the app.

    ``store`` injects an already-open store, and ``warm=False`` skips the lifespan build.
    Together they are what lets the API be tested offline, with no torch and no corpus --
    this module was at 0% coverage, and every bug the audit found had shipped behind that.
    """
    rt = Runtime(cfg or Config.load(), generate=generate, _store=store)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        anyio.to_thread.current_default_thread_limiter().total_tokens = thread_limit
        if warm:
            try:
                await anyio.to_thread.run_sync(rt.warm)
            except Exception as exc:
                # A missing corpus or a stale dense index must not stop the process from
                # starting: /ready is how that is reported, and a crash-looping container
                # cannot report anything at all.
                rt.warm_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="warrant", docs_url="/api/docs", openapi_url="/api/openapi.json",
                  lifespan=lifespan)

    # Regulation text is highly repetitive and gzips ~4:1; a §315.201 section response is
    # 65,551 bytes of JSON, so this is the difference between a 64 KB and a 16 KB timeline
    # load. minimum_size skips the small metadata responses, where the header costs more
    # than the compression saves.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN_RE, allow_credentials=False,
        allow_methods=["GET", "OPTIONS"], allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next) -> Response:
        """Echo the caller's request id, or mint one, so one line of a log names one request.

        Added last, so it is the outermost middleware and stamps every response including
        errors and CORS preflights -- an id that is missing on exactly the failures is worse
        than no id at all.
        """
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # -- infrastructure ---------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness. Touches nothing, so a missing corpus cannot make it fail."""
        return HealthResponse(status="ok", uptime_s=round(time.monotonic() - rt.started, 3))

    @app.get("/ready", response_model=ReadyResponse)
    def ready(response: Response) -> ReadyResponse:
        """Readiness: corpus present and non-empty, models built. 503 until both hold."""
        corpus, chunks, detail = False, None, rt.warm_error
        try:
            chunks = rt.store.count()
            corpus = chunks > 0
            if not corpus:
                detail = detail or "corpus is empty; run `make build`"
        except HTTPException as exc:
            detail = detail or str(exc.detail)
        models = rt.models_loaded
        if corpus and not models:
            detail = detail or "models not built yet"
        ok = corpus and models
        if not ok:
            response.status_code = 503
        return ReadyResponse(ready=ok, corpus=corpus, chunks=chunks, models=models,
                             generator=rt._generator is not None, detail=detail)

    # -- metadata ---------------------------------------------------------------

    @lru_cache(maxsize=1)
    def _meta() -> MetaResponse:
        db = rt.store.db
        row = db.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT section_id) sections, "
            "MIN(valid_from) lo, MAX(valid_from) hi FROM chunk WHERE system_to IS NULL"
        ).fetchone()
        parts = [PartSummary(**dict(r)) for r in db.execute(
            "SELECT part, COUNT(DISTINCT section_id) sections, COUNT(*) chunks "
            "FROM chunk WHERE system_to IS NULL GROUP BY part ORDER BY part")]
        facets: dict[str, list[str]] = {}
        for restriction in PART_RESTRICTIONS.values():
            for facet, values in restriction.items():
                facets.setdefault(facet, [])
                for v in sorted(values):
                    if v not in facets[facet]:
                        facets[facet].append(v)
        return MetaResponse(
            chunks=row["n"], sections=row["sections"],
            parts=parts, part_count=len(parts),
            earliest=row["lo"], latest=row["hi"],
            history_floor=rt.cfg.corpus.history_floor,
            facets={k: sorted(v) for k, v in facets.items()},
            config_hash=rt.cfg.hash,
            final_k=rt.cfg.retrieve.final_k,
        )

    @app.get("/api/meta", response_model=MetaResponse)
    def meta() -> MetaResponse:
        rt.read_only()
        return _meta()

    def _scope(pay_system: str | None, service: str | None) -> Scope:
        facets = {k: v for k, v in
                  {"pay_system": pay_system, "service": service}.items() if v}
        try:
            return Scope.of(**facets)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- ask --------------------------------------------------------------------

    @app.get("/api/ask", response_model=AskResponse)
    def ask(q: str = Query(min_length=2, max_length=512),
            as_of: str = Query(max_length=32),
            pay_system: str | None = None,
            service: str | None = None,
            generate: bool = True) -> AskResponse:
        as_of = _date(as_of, "as_of")
        scope = _scope(pay_system, service)
        rt.read_only()
        trace = rt.retriever.retrieve(q, as_of=as_of, scope=scope)
        rows = _rows(rt.store, trace.final)

        payload = AskResponse(
            question=q, as_of=as_of, scope=scope.describe(),
            excluded_parts=trace.excluded_parts,
            trace=TraceView(
                admitted=trace.admitted,
                corpus=_meta().chunks,
                stages=[
                    Stage(name="predicates", out=trace.admitted),
                    Stage(name="lexical", out=len(trace.lexical)),
                    Stage(name="dense", out=len(trace.dense)),
                    Stage(name="fusion", out=len(trace.fused)),
                    Stage(name="rerank", out=len(trace.reranked)),
                    Stage(name="final", out=len(trace.final)),
                ],
            ),
            evidence=[_evidence(r) for r in rows],
        )
        if not (generate and rt.generate):
            payload.trace_id = _record(rt, trace)
            return payload

        answer = _generate_answer(rt, q, excerpts_for(rt.store, trace),
                                  as_of=as_of, scope=scope.describe())
        payload.abstained = answer.abstained
        payload.parse_failed = answer.parse_failed
        payload.claims = [
            ClaimView(
                text=c.text,
                grounded=c.grounded,
                citations=[
                    Citation(version_id=vid,
                             span=None if sp is None else
                             SpanView(start=sp.start, end=sp.end, score=round(sp.score, 3)))
                    for vid, sp in c.spans.items()
                ],
            )
            for c in answer.claims
        ]
        payload.trace_id = _record(rt, trace, answer=answer)
        return payload

    # -- version history --------------------------------------------------------

    @app.get("/api/section/{section_id}", response_model=SectionResponse)
    def section(section_id: str) -> SectionResponse:
        rt.read_only()
        rows = rt.store.db.execute(
            "SELECT version_id, chunk_id, anchor, heading, part, text, valid_from, valid_to "
            "FROM chunk WHERE section_id = ? AND system_to IS NULL "
            "ORDER BY valid_from, id", (section_id,)).fetchall()
        if not rows:
            raise HTTPException(404, f"no section {section_id}")
        versions: dict[str, SectionVersion] = {}
        for r in rows:
            v = versions.setdefault(r["valid_from"], SectionVersion(
                valid_from=r["valid_from"], valid_to=r["valid_to"],
                heading=r["heading"], part=r["part"], paragraphs=[]))
            v.paragraphs.append(
                Paragraph(anchor=r["anchor"], version_id=r["version_id"], text=r["text"]))
        return SectionResponse(
            section_id=section_id, heading=rows[0]["heading"], part=rows[0]["part"],
            versions=sorted(versions.values(), key=lambda v: v.valid_from))

    @app.get("/api/section/{section_id}/versions", response_model=VersionsResponse)
    def section_versions(section_id: str) -> VersionsResponse:
        """The timeline's four dates, without the text under them.

        Measured on §315.201, the full history is 188 rows and 65,551 bytes of JSON, of which
        the timeline uses four valid_from values. Aggregating in SQL rather than counting
        rows in Python keeps that ratio honest: nothing large is read to be thrown away.
        """
        rt.read_only()
        rows = rt.store.db.execute(
            # MIN(heading) rather than a bare column: SQLite picks an arbitrary row for a
            # bare column in an aggregate query, and one version's paragraphs all carry the
            # same heading anyway, so this is the same value chosen deterministically.
            "SELECT valid_from, valid_to, MIN(heading) heading, COUNT(*) paragraph_count "
            "FROM chunk WHERE section_id = ? AND system_to IS NULL "
            "GROUP BY valid_from, valid_to ORDER BY valid_from", (section_id,)).fetchall()
        if not rows:
            raise HTTPException(404, f"no section {section_id}")
        return VersionsResponse(
            section_id=section_id,
            versions=[VersionSummary(**dict(r)) for r in rows])

    @app.get("/api/diff", response_model=DiffResponse)
    def diff(section_id: str, a: str, b: str) -> DiffResponse:
        """Word-level before/after between two versions of a section."""
        # Validated before the lookup so a malformed date reads as "a is not a date" rather
        # than as "no such version", which is the same 404 a real but absent version gets.
        a, b = _date(a, "a"), _date(b, "b")
        rt.read_only()
        left, right = _version_text(rt.store, section_id, a), _version_text(
            rt.store, section_id, b)
        if left is None or right is None:
            raise HTTPException(404, "no such version")
        sm = difflib.SequenceMatcher(None, left.split(), right.split(), autojunk=False)
        ops = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            ops.append(DiffOp(op=tag,
                              before=" ".join(left.split()[i1:i2]),
                              after=" ".join(right.split()[j1:j2])))
        return DiffResponse(section_id=section_id, a=a, b=b,
                            similarity=round(sm.ratio(), 4), ops=ops)

    # -- failure budget ---------------------------------------------------------

    @app.get("/api/budget", response_model=BudgetResponse)
    def budget() -> BudgetResponse:
        """The recorded failure budget. Read from disk, never recomputed per request.

        Recomputing would take minutes and, worse, would let the dashboard drift from the
        numbers in `results/` that the README quotes.
        """
        path = rt.cfg.budget_path
        if not path.exists():
            raise HTTPException(503, "no recorded budget; run `make autopsy`")
        import json

        return BudgetResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))

    if UI_DIR.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app


def _rows(store: Store, version_ids: list[str]) -> list:
    if not version_ids:
        return []
    rows = {r["version_id"]: r for r in store.db.execute(
        "SELECT * FROM chunk WHERE version_id IN "
        f"({','.join('?' * len(version_ids))})", version_ids)}
    return [rows[v] for v in version_ids if v in rows]


def _evidence(row) -> Evidence:
    return Evidence(
        version_id=row["version_id"], chunk_id=row["chunk_id"],
        section_id=row["section_id"], anchor=row["anchor"],
        heading=row["heading"], part=row["part"], subpart=row["subpart"],
        text=row["text"], valid_from=row["valid_from"], valid_to=row["valid_to"],
    )


def _version_text(store: Store, section_id: str, valid_from: str) -> str | None:
    rows = store.db.execute(
        "SELECT text FROM chunk WHERE section_id = ? AND valid_from = ? "
        "AND system_to IS NULL ORDER BY id", (section_id, valid_from)).fetchall()
    return " ".join(r["text"] for r in rows) if rows else None
