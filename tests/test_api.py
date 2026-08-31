"""Tests for the HTTP API.

This module was at 0% coverage, which is why a date range that spanned one day, an accepted
``as_of=2021-13-45``, three read paths that never asserted ``PRAGMA query_only`` and an
OpenAPI document in which every response was ``{"type": "object"}`` all shipped together.
None of those needed a corpus, a GPU or a network to catch -- they needed one client.

Everything here runs against a synthetic in-memory store, with the dense encoder, the
cross-encoder and the generator all switched off, so the suite stays offline and torch-free.
The one thing the synthetic store must be faithful about is *size*: the gzip assertion needs
a section of the same class as the real §315.201 (188 rows, 65,551 bytes of JSON), because a
compression threshold that is only ever exercised on 300-byte fixtures tests nothing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from warrant.config import Config
from warrant.generate.answer import Answer, Claim
from warrant.index.store import Chunk, Store
from warrant.observe.trace_store import TraceStore
from warrant.retrieve.hybrid import Trace
from warrant.serve import api, guard
from warrant.serve.api import create_app
from warrant.verify.align import Span

T0 = "2026-01-01T00:00:00+00:00"

#: ~640 characters, so a 104-row section lands in the same size class as the real one.
BODY = (
    "An agency may grant a career or career-conditional appointment to a person who "
    "completes the service requirement, subject to the conditions in this subpart and to "
    "any additional requirement the Office of Personnel Management prescribes. "
) * 3

SPANS = [("2017-01-01", "2019-05-01"), ("2019-05-01", "2021-03-15"),
         ("2021-03-15", "2023-07-01"), ("2023-07-01", None)]


def _big_section() -> list[Chunk]:
    """Four versions of a 26-paragraph section: the timeline case, at realistic weight."""
    return [
        Chunk(chunk_id=f"315.201#{chr(97 + p)}", section_id="315.201", title=5, part="315",
              anchor=chr(97 + p), heading="Career and career-conditional employment",
              text=f"({chr(97 + p)}) As in force from {lo}. {BODY}",
              valid_from=lo, valid_to=hi)
        for lo, hi in SPANS for p in range(26)
    ]


SYNTHETIC = [
    # Two versions of one paragraph, so a diff has something to diff.
    Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
          heading="Restored annual leave",
          text="annual leave restored must be scheduled within two years",
          valid_from="2017-01-01", valid_to="2020-08-10"),
    Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
          heading="Restored annual leave",
          text="annual leave restored must be scheduled within three years",
          valid_from="2020-08-10"),
    # Restricted parts, so scope filtering has something to exclude.
    Chunk(chunk_id="531.404#a", section_id="531.404", title=5, part="531", anchor="a",
          heading="Within-grade increase",
          text="performance must be at an acceptable level of competence for a pay increase",
          valid_from="2017-01-01"),
    Chunk(chunk_id="532.203#a", section_id="532.203", title=5, part="532", anchor="a",
          heading="Structure of regular wage schedules",
          text="each nonsupervisory wage schedule has five steps and a pay rate range",
          valid_from="2017-01-01"),
    *_big_section(),
]

PARTS = ["315", "531", "532", "630"]

#: Shaped like `autopsy.localize.Budget.to_dict`, so /api/budget has something real to
#: validate against without an autopsy run.
BUDGET = {
    "bucket": "temporal", "config_hash": "63fcbc7607bc", "items": 721, "failures": 170,
    "success_rate": 0.7642,
    "observational": [{"stage": "rerank", "failures": 64, "share": "37.6%"}],
    "interventional": [{"repair": "ranking", "implicated": 40}],
    "stages": ["ingestion", "retrieval", "rerank"],
}

ROUTES = [
    "/health",
    "/ready",
    "/api/meta",
    "/api/ask?q=restored%20annual%20leave&as_of=2024-06-01",
    "/api/ask?q=restored%20annual%20leave&as_of=2018-06-01&pay_system=GS",
    "/api/section/630.306",
    "/api/section/630.306/versions",
    "/api/diff?section_id=630.306&a=2017-01-01&b=2020-08-10",
    "/api/budget",
    "/api/openapi.json",
    "/api/docs",
]


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Store:
    """A synthetic store, on disk rather than in ``:memory:``.

    Not a preference: each connection to ``:memory:`` is its own empty database, so `Store`
    shares one connection for that case, and sqlite3 refuses a connection used off its
    creating thread. TestClient dispatches sync endpoints to the threadpool, so an in-memory
    store fails every request with ProgrammingError before reaching any code under test. A
    temp file gets the real thread-local connection path, which is what production uses.
    """
    with Store(tmp_path_factory.mktemp("store") / "warrant.sqlite3") as s:
        s.add(SYNTHETIC, system_from=T0)
        yield s


@pytest.fixture(scope="module")
def cfg(tmp_path_factory) -> Config:
    """The real config with every model switched off. Offline is the point."""
    c = Config.load()
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    c.corpus.parts = PARTS
    # Every test that uses this fixture injects its own `store` (see the `store` fixture
    # below), so `c.store.path` is never opened as a corpus -- but `Runtime.cache` derives
    # the answer cache's file from it (`store_path.with_name("cache.sqlite3")`), and leaving
    # it at the default would write a real `data/cache.sqlite3` into the checkout every time
    # a generate=True test runs.
    c.store.path = str(tmp_path_factory.mktemp("cache_home") / "warrant.sqlite3")
    budget = tmp_path_factory.mktemp("budget") / "failure-budget.json"
    budget.write_text(json.dumps(BUDGET), encoding="utf-8")
    c.store.budget = str(budget)
    return c


@pytest.fixture(scope="module")
def client(cfg: Config, store: Store) -> TestClient:
    app = create_app(cfg, generate=False, store=store, warm=True, thread_limit=2,
        guards=guard.Guards(enabled=False))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def bare_client(tmp_path_factory) -> TestClient:
    """An instance with no corpus at all: the state /ready exists to report."""
    c = Config.load()
    c.store.path = str(tmp_path_factory.mktemp("empty") / "warrant.sqlite3")
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    app = create_app(c, generate=False, warm=False, guards=guard.Guards(enabled=False))
    with TestClient(app) as client:
        yield client


# -- every route answers ----------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_returns_a_non_server_error(client: TestClient, route: str):
    assert client.get(route).status_code < 500


def test_openapi_documents_a_schema_for_every_response(client: TestClient):
    """The regression test for `-> dict`.

    Annotating handlers ``-> dict`` documented every response as an open object: the schema
    was syntactically valid, no client could be generated from it, and renaming a field could
    never break anything. A ``$ref`` per response is what makes the document load-bearing.
    """
    schema = client.get("/api/openapi.json").json()
    documented = 0
    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            body = op["responses"].get("200", {}).get("content", {})
            if "application/json" not in body:
                continue
            assert "$ref" in body["application/json"]["schema"], f"{method} {path} is untyped"
            documented += 1
    assert documented >= 8, f"only {documented} endpoints documented"
    assert "latest" in schema["components"]["schemas"]["MetaResponse"]["properties"]


# -- meta -------------------------------------------------------------------------


def test_meta_reports_a_real_date_range(client: TestClient):
    """`latest` was assigned history_floor, so it equalled `earliest` and the slider spanned
    one day of an eight-year corpus."""
    meta = client.get("/api/meta").json()
    assert meta["earliest"] == "2017-01-01"
    assert meta["latest"] == "2023-07-01"
    assert meta["latest"] > meta["earliest"]
    assert meta["history_floor"] == "2017-01-01"
    assert meta["chunks"] == len(SYNTHETIC)
    assert {p["part"] for p in meta["parts"]} == set(PARTS)


# -- validation -------------------------------------------------------------------


def test_an_unknown_pay_system_is_rejected_not_filtered(client: TestClient):
    """A typo used to match no part's allowed set, silently remove 41% of the corpus, and
    return a confident degraded answer with HTTP 200."""
    r = client.get("/api/ask", params={"q": "pay", "as_of": "2024-06-01",
                                       "pay_system": "bogus"})
    assert r.status_code == 400
    assert "pay_system" in r.json()["detail"]


@pytest.mark.parametrize("as_of", ["2021-13-45", "2021-02-30", "not-a-date", "20210101"])
def test_a_bad_as_of_is_rejected(client: TestClient, as_of: str):
    r = client.get("/api/ask", params={"q": "leave", "as_of": as_of})
    assert r.status_code == 422
    assert "as_of" in json.dumps(r.json())


@pytest.mark.parametrize("param", ["a", "b"])
def test_diff_rejects_a_non_date(client: TestClient, param: str):
    """A bad date used to surface as a generic 404, indistinguishable from a real version
    that simply is not there."""
    params = {"section_id": "630.306", "a": "2017-01-01", "b": "2020-08-10",
              param: "yesterday"}
    r = client.get("/api/diff", params=params)
    assert r.status_code == 422
    assert f"{param}:" in r.json()["detail"]


def test_diff_reports_the_amendment(client: TestClient):
    body = client.get("/api/diff", params={"section_id": "630.306", "a": "2017-01-01",
                                           "b": "2020-08-10"}).json()
    assert 0.0 < body["similarity"] < 1.0
    assert any(op["op"] == "replace" for op in body["ops"])


def test_a_missing_section_is_404(client: TestClient):
    assert client.get("/api/section/999.999").status_code == 404
    assert client.get("/api/section/999.999/versions").status_code == 404


def test_a_well_formed_but_absent_version_is_404(client: TestClient):
    """The 404 the date validation had to be lifted above: this one means what it says."""
    r = client.get("/api/diff", params={"section_id": "630.306", "a": "2017-01-01",
                                        "b": "2019-01-01"})
    assert r.status_code == 404
    assert r.json()["detail"] == "no such version"


def test_a_missing_budget_is_503_not_a_crash(cfg: Config, store: Store, tmp_path):
    c = cfg.model_copy(deep=True)
    c.store.budget = str(tmp_path / "never-recorded.json")
    app = create_app(c, generate=False, store=store, warm=False, guards=guard.Guards(enabled=False))
    with TestClient(app) as client:
        r = client.get("/api/budget")
    assert r.status_code == 503
    assert "make autopsy" in r.json()["detail"]


def test_budget_is_validated_against_its_recorded_shape(client: TestClient):
    body = client.get("/api/budget").json()
    assert body["observational"][0]["stage"] == "rerank"
    assert body["success_rate"] == BUDGET["success_rate"]


# -- the timeline endpoint --------------------------------------------------------


def test_versions_is_a_fraction_of_the_full_section(client: TestClient):
    """The reason the endpoint exists: four dates should not cost 64 KB of text."""
    full = client.get("/api/section/315.201")
    summary = client.get("/api/section/315.201/versions")
    body = summary.json()
    assert [v["valid_from"] for v in body["versions"]] == [lo for lo, _ in SPANS]
    assert all(v["paragraph_count"] == 26 for v in body["versions"])
    assert len(summary.content) * 20 < len(full.content), (
        f"{len(summary.content)} bytes against {len(full.content)}")


# -- transport ---------------------------------------------------------------------


def test_a_large_section_is_gzipped_when_asked(client: TestClient):
    r = client.get("/api/section/315.201", headers={"Accept-Encoding": "gzip"})
    assert len(r.content) > 64_000, "fixture is not in the size class being tested"
    assert r.headers["content-encoding"] == "gzip"
    # Regulation text is repetitive; measured ~4:1. Assert 3:1 so the test is about the
    # middleware being wired, not about zlib's exact ratio.
    assert int(r.headers["content-length"]) * 3 < len(r.content)


def test_identity_encoding_is_respected(client: TestClient):
    r = client.get("/api/section/315.201", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in r.headers


def test_small_responses_are_not_compressed(client: TestClient):
    assert "content-encoding" not in client.get("/health").headers


def test_request_id_is_echoed_or_minted(client: TestClient):
    assert client.get("/health", headers={"X-Request-ID": "abc123"}
                      ).headers["x-request-id"] == "abc123"
    minted = client.get("/health").headers["x-request-id"]
    assert minted and minted != client.get("/health").headers["x-request-id"]


def test_cors_allows_a_local_dev_origin_and_nothing_else(client: TestClient):
    local = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
    remote = client.get("/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in remote.headers


# -- liveness against readiness ----------------------------------------------------


def test_health_and_ready_answer_different_questions(bare_client: TestClient):
    """With no corpus the process is alive and cannot serve. One probe cannot say both."""
    assert bare_client.get("/health").status_code == 200
    ready = bare_client.get("/ready")
    assert ready.status_code == 503
    body = ready.json()
    assert body["ready"] is False and body["corpus"] is False
    assert "no corpus" in body["detail"]
    # A read endpoint is genuinely unavailable here; that is a 5xx on purpose.
    assert bare_client.get("/api/meta").status_code == 503


def test_ready_is_true_once_the_corpus_and_models_are_there(client: TestClient):
    body = client.get("/ready").json()
    assert body == {"ready": True, "corpus": True, "chunks": len(SYNTHETIC),
                    "models": True, "generator": False, "detail": None,
                    # null, not 0: this fixture runs lexical-only, so there is no index
                    # for anything to be missing from. Zero would claim a covered index.
                    "uncovered_chunks": None}


def test_a_failed_warm_up_is_reported_not_fatal(tmp_path):
    """A model that cannot be built must not stop the process from starting: a container
    that crash-loops on a stale index reports nothing at all, and /ready is the thing whose
    entire job is to say so."""
    c = Config.load()
    c.store.path = str(tmp_path / "absent.sqlite3")
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    with TestClient(create_app(c, generate=False, warm=True,
        guards=guard.Guards(enabled=False))) as client:
        assert client.get("/health").status_code == 200
        body = client.get("/ready").json()
    assert body["ready"] is False
    assert "no corpus" in body["detail"]


# -- read-only enforcement ---------------------------------------------------------


READ_ROUTES = [r for r in ROUTES if r.startswith("/api/") and "openapi" not in r
               and "docs" not in r and "budget" not in r]


@pytest.mark.parametrize("route", READ_ROUTES)
def test_every_read_endpoint_asserts_query_only(cfg: Config, store: Store, monkeypatch,
                                                route: str):
    """It was asserted on one read path of four. A fresh app per route because the assertion
    is memoised per connection, so a shared client would only ever prove it for whichever
    route happened to be requested first."""
    calls: list[str] = []
    original = store.read_only
    monkeypatch.setattr(store, "read_only", lambda: (calls.append(route), original())[1])
    app = create_app(cfg, generate=False, store=store, warm=False, thread_limit=2,
        guards=guard.Guards(enabled=False))
    with TestClient(app) as c:
        assert c.get(route).status_code == 200
    assert calls, f"{route} never asserted query_only"


def test_query_only_is_asserted_once_per_connection(cfg: Config, store: Store):
    """Per connection, not per request and not per process: connections are thread-local, so
    a worker that never called it held a writable handle to the corpus for the life of the
    process, and calling it on every request would issue one pointless PRAGMA per query."""
    rt = api.Runtime(cfg, generate=False, _store=store)
    seen: list[object] = []

    def worker() -> None:
        rt.read_only()
        seen.append(store.db.execute("PRAGMA query_only").fetchone()[0])
        rt.read_only()
        seen.append(getattr(rt._pragma, "done", False))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.db.execute("DELETE FROM chunk WHERE section_id = '630.306'")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [1, True]


# -- generation admission control --------------------------------------------------


class _FakeGenerator:
    """Stands in for 2,944 MB of Qwen. The admission control is what is under test."""

    def answer(self, question, excerpts, *, as_of, scope):
        return f"{question}@{as_of}/{scope}"


class _AnsweringGenerator:
    """Returns a real `Answer`, so the claim/citation mapping is exercised without torch."""

    def answer(self, question, excerpts, *, as_of, scope):
        vid = excerpts[0][0]
        return Answer(question, as_of, scope,
                      [Claim("Restored leave must be scheduled within two years.", [vid],
                             {vid: Span(0, 12, 0.87654), "630.306#b@2017-01-01": None})],
                      True, {vid: excerpts[0][2]})


class _CountingGenerator:
    """`_AnsweringGenerator`, plus a class-level call count.

    Class-level rather than instance-level: `Runtime.generator` builds and caches exactly one
    instance per app, and a cache test needs to tell "answered once, served again from SQLite"
    apart from "answered twice", without reaching into the app for the instance itself.
    """

    calls = 0

    def answer(self, question, excerpts, *, as_of, scope):
        type(self).calls += 1
        return _AnsweringGenerator().answer(question, excerpts, as_of=as_of, scope=scope)


def test_ask_returns_claims_and_spans_when_generation_runs(cfg: Config, store: Store,
                                                           monkeypatch):
    monkeypatch.setattr("warrant.generate.model.Generator", _AnsweringGenerator)
    app = create_app(cfg, generate=True, store=store, warm=False, thread_limit=2,
        guards=guard.Guards(enabled=False))
    with TestClient(app) as client:
        body = client.get("/api/ask", params={"q": "restored annual leave",
                                              "as_of": "2018-06-01"}).json()
    assert body["abstained"] is False and body["parse_failed"] is False
    claim = body["claims"][0]
    assert claim["grounded"] is True
    spans = {c["version_id"]: c["span"] for c in claim["citations"]}
    assert spans["630.306#b@2017-01-01"] is None
    assert next(s for s in spans.values() if s)["score"] == 0.877


def test_ask_without_generation_returns_evidence_and_abstains(client: TestClient):
    body = client.get("/api/ask", params={"q": "restored annual leave",
                                          "as_of": "2018-06-01"}).json()
    assert body["abstained"] is True and body["claims"] == []
    assert body["parse_failed"] is None, "generation did not run; that is not a parse result"
    assert body["evidence"], "retrieval returned nothing on the synthetic corpus"
    assert [s["name"] for s in body["trace"]["stages"]][0] == "predicates"
    assert body["trace"]["corpus"] == len(SYNTHETIC)


def test_scope_excludes_the_parts_it_does_not_govern(client: TestClient):
    body = client.get("/api/ask", params={"q": "pay rate schedule", "as_of": "2024-06-01",
                                          "pay_system": "GS", "service": "competitive"}).json()
    assert body["scope"] == "pay_system=GS, service=competitive"
    assert body["excluded_parts"] == ["532"]
    assert all(e["part"] != "532" for e in body["evidence"])


def test_generation_runs_one_at_a_time(cfg: Config):
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())
    got = anyio.run(lambda: api._generate_answer(
        rt, "q", [], as_of="2024-06-01", scope="government-wide",
        deadline=time.monotonic() + api.GENERATE_DEADLINE_S))
    assert got == "q@2024-06-01/government-wide"
    assert rt.stats["generate_calls"] == 1


def test_a_full_queue_is_refused_with_retry_after(cfg: Config, monkeypatch):
    """503 + Retry-After, not a 33-minute wait. The measured ceiling is 0.051 req/s and
    nothing here can raise it; the queue can only be honest about it."""
    monkeypatch.setattr(api, "GENERATE_QUEUE_WAIT_S", 0.05)
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())

    async def run() -> HTTPException:
        async with api._GENERATION_SLOT:
            with pytest.raises(HTTPException) as exc:
                await api._generate_answer(
                    rt, "q", [], as_of="2024-06-01", scope="government-wide",
                    deadline=time.monotonic() + api.GENERATE_DEADLINE_S)
            return exc.value

    exc = anyio.run(run)
    assert exc.status_code == 503
    assert exc.headers["Retry-After"] == str(api.RETRY_AFTER_S)
    assert rt.stats["generate_rejected"] == 1


def test_a_request_that_cannot_meet_its_deadline_is_refused(cfg: Config):
    """A 420-token answer takes ~19.7 s at 21.3 tok/s and may be retried once, so starting
    one with less budget than that only burns the GPU on an answer nobody will receive."""
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())

    async def run() -> HTTPException:
        with pytest.raises(HTTPException) as exc:
            # A deadline already in the past: the same effect the old
            # `monkeypatch.setattr(api, "GENERATE_DEADLINE_S", 0.0)` had, expressed the way a
            # real caller now produces it -- the budget is fixed at arrival, not recomputed.
            await api._generate_answer(rt, "q", [], as_of="2024-06-01",
                                       scope="government-wide", deadline=time.monotonic())
        return exc.value

    exc = anyio.run(run)
    assert exc.status_code == 503
    assert "deadline" in exc.detail
    assert api._GENERATION_SLOT.value == 1, "the slot must be released on refusal"


def test_a_refused_request_never_occupies_a_thread(cfg: Config, monkeypatch):
    """The whole point of acquiring the semaphore before the thread hop: a caller that
    cannot get the slot must be told so from the event loop, never having taken one of
    `THREAD_LIMIT`'s four tokens. results/eval-010-capacity.md measured the old shape doing
    this backwards -- a 503's floor was 20.1s, paid entirely inside the thread pool that
    `/api/section` and `/health` also share."""
    monkeypatch.setattr(api, "GENERATE_QUEUE_WAIT_S", 0.05)
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())

    def _boom(*a, **k):
        raise AssertionError("a refused request must never reach the thread pool")

    monkeypatch.setattr(anyio.to_thread, "run_sync", _boom)

    async def run() -> HTTPException:
        async with api._GENERATION_SLOT:
            with pytest.raises(HTTPException) as exc:
                await api._generate_answer(
                    rt, "q", [], as_of="2024-06-01", scope="government-wide",
                    deadline=time.monotonic() + api.GENERATE_DEADLINE_S)
            return exc.value

    exc = anyio.run(run)
    assert exc.status_code == 503


def test_deadline_is_stamped_at_request_arrival(cfg: Config, store: Store, monkeypatch,
                                                 tmp_path):
    """`_generate_answer` used to compute its own deadline, after however long retrieval and
    the (unbounded, pre-fix) thread-pool wait took -- so the budget was always full by the
    time it was checked, and `GENERATE_FLOOR_S` could never fire across 14,628 requests
    (results/eval-010-capacity.md). The `deadline` middleware fixes that by stamping
    `request.state.deadline` before any of that happens; this asserts the value `/api/ask`
    hands to `_generate_answer` really is close to when the request arrived, not to when
    generation was finally reached."""
    captured = {}

    async def fake_generate_answer(rt, q, excerpts, *, as_of, scope, deadline):
        captured["deadline"] = deadline
        return _AnsweringGenerator().answer(q, excerpts, as_of=as_of, scope=scope)

    monkeypatch.setattr(api, "_generate_answer", fake_generate_answer)
    # A private cache file: `cfg` is module-scoped and shared by every other generate=True
    # test, and this question would otherwise be served from whatever those left behind.
    c = cfg.model_copy(deep=True)
    c.store.path = str(tmp_path / "warrant.sqlite3")
    app = create_app(c, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guard.Guards(enabled=False))
    before = time.monotonic()
    with TestClient(app) as client:
        r = client.get("/api/ask", params={"q": "restored annual leave",
                                           "as_of": "2018-06-01"})
    after = time.monotonic()

    assert r.status_code == 200
    lo, hi = before + api.GENERATE_DEADLINE_S, after + api.GENERATE_DEADLINE_S
    assert lo <= captured["deadline"] <= hi


# -- threadpool cap ----------------------------------------------------------------


def test_lifespan_caps_the_threadpool(cfg: Config, store: Store):
    """Retrieval peaks at 4 threads (66 QPS) and falls to 25.6 QPS at 16; anyio's default is
    40, which is 40 workers contending for a GIL none of them releases."""
    async def limit() -> float:
        app = create_app(cfg, generate=False, store=store, warm=False, thread_limit=3,
            guards=guard.Guards(enabled=False))
        async with app.router.lifespan_context(app):
            return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(limit) == 3


# -- against the real corpus, when there is one ------------------------------------


def test_meta_over_the_built_corpus_spans_years():
    """The synthetic store cannot prove the range is right on real data, only that the query
    is. Skipped rather than marked neural: it needs a corpus, not a GPU."""
    c = Config.load()
    if not c.store_path.exists():
        pytest.skip("no corpus built; run `make fetch && make build`")
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    with TestClient(create_app(c, generate=False, warm=False,
        guards=guard.Guards(enabled=False))) as client:
        meta = client.get("/api/meta").json()
    assert meta["latest"] > meta["earliest"] >= meta["history_floor"]


def test_metrics_exposes_prometheus_text(client: TestClient):
    client.get("/api/meta")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE warrant_requests_total counter" in body
    assert "# TYPE warrant_request_duration_ms histogram" in body
    assert 'warrant_requests_total{endpoint="/api/meta",status="2xx"}' in body
    assert "warrant_corpus_chunks" in body


def test_metrics_labels_by_route_template_not_by_path(client: TestClient):
    """`/api/section/630.306` and `/api/section/630.307` must be one series, not two.
    Labelling by raw path makes a metric whose cardinality is the size of the corpus, which
    is the usual way a self-instrumented service takes down the collector scraping it."""
    for _ in range(2):
        client.get("/api/ask", params={"q": "restored annual leave", "as_of": "2024-01-01"})
    body = client.get("/metrics").text
    import re

    endpoints = set(re.findall(r'warrant_requests_total\{endpoint="([^"]+)"', body))
    # Several status classes per endpoint is fine and intended -- status is a bounded label.
    # What must never happen is a distinct endpoint label per request.
    assert "/api/ask" in endpoints
    assert not any(e.startswith("/api/ask") and e != "/api/ask" for e in endpoints), endpoints
    assert "restored" not in body, "a query string reached a label"


def test_a_scrape_never_takes_the_service_down(client: TestClient, monkeypatch):
    """Gauges are sampled at scrape time, so a broken store would otherwise turn a
    monitoring request into a 500 -- and the moment you most want metrics is the moment the
    store is unhappy."""
    import warnings

    from warrant.serve import api as api_module

    monkeypatch.setattr(api_module, "uncovered",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert client.get("/metrics").status_code == 200


def test_a_rate_limited_response_is_still_readable_by_the_browser(tmp_path):
    """Every other test in this module disables the limiter, because TestClient presents
    one client identity and thirty requests then read as one caller at thirty times the
    ceiling. Disabling it is right -- raising the limit would make the measured ceiling a
    fiction -- but it would also mean nothing here ever exercises a 429 through the real
    middleware stack. This does.

    The assertions are about position in that stack, not about the limiter's arithmetic
    (tests/test_guard.py owns that). A 429 whose CORS headers were stripped is invisible to
    the page that provoked it, and one without an X-Request-ID is a rejection the user
    cannot quote back. Both are decided by where the middleware sits, and both would still
    pass every unit test of the limiter itself.
    """
    store = Store(":memory:")
    store.add(SYNTHETIC)
    cfg = Config.load()
    app = api.create_app(cfg, generate=False, store=store, warm=False, thread_limit=2,
                         guards=guard.Guards(answer=guard.RateLimiter(1e-6, burst=1),
                                             read=guard.RateLimiter(1e-6, burst=1)))
    with TestClient(app) as client:
        origin = {"Origin": "http://localhost:5173"}
        first = client.get("/api/meta", headers=origin)
        assert first.status_code == 200

        limited = client.get("/api/meta", headers=origin)
        assert limited.status_code == 429
        assert limited.headers.get("Retry-After"), "a 429 with no Retry-After is a guess"
        assert limited.headers.get("X-Request-ID"), "outside request_id: nothing to quote"
        assert limited.headers.get("access-control-allow-origin") == origin["Origin"], (
            "inside CORS: the page that provoked this cannot read it")


def _sse_events(body: str) -> list[tuple[str, dict]]:
    out = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name = text = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                text = line[6:]
        if name is not None:
            out.append((name, json.loads(text)))
    return out


def test_the_stream_delivers_evidence_before_generation(client: TestClient):
    """The asymmetry is the whole point: retrieval finishes in 18ms and generation takes
    about nineteen seconds. A spinner over both would hide a result that was ready in a
    fiftieth of a second, so the evidence must reach the client before anything waits on
    the model -- ordering, not merely presence."""
    with client.stream("GET", "/api/ask/stream",
                       params={"q": "restored annual leave",
                               "as_of": "2024-01-01"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names.index("evidence") < names.index("done")
    assert names[0] == "retrieval"
    evidence = dict(events)["evidence"]
    assert evidence and all("version_id" in row and "text" in row for row in evidence)


def test_no_partial_claim_is_ever_emitted(client: TestClient):
    """Tokens are deliberately not streamed. The model emits a JSON envelope whose partial
    states are half-written citations, and putting an unresolved reference in front of a
    reader for several seconds is the precise failure this project exists to detect. Every
    claim frame must therefore be complete and already validated."""
    with client.stream("GET", "/api/ask/stream",
                       params={"q": "restored annual leave",
                               "as_of": "2024-01-01"}) as response:
        events = _sse_events("".join(response.iter_text()))
    for name, data in events:
        if name == "claim":
            assert data["text"] and "citations" in data and "grounded" in data


def test_a_refusal_after_the_stream_opens_arrives_as_an_event(cfg: Config, store: Store,
                                                              monkeypatch):
    """Admission control returns 503, but by then the response is already a 200 with an
    open body, so the status code is spent. If the refusal were left to propagate the
    client would see a truncated stream and could not tell a refusal from a dropped
    connection -- and the two call for opposite responses, one a retry after the advertised
    delay and the other an immediate reconnect."""
    from warrant.serve import api as api_module

    def refuse(*a, **k):
        raise api_module.HTTPException(503, "generator at capacity")

    monkeypatch.setattr(api_module, "_generate_answer", refuse)
    app = create_app(cfg, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guard.Guards(enabled=False))
    with TestClient(app) as client, client.stream(
            "GET", "/api/ask/stream",
            params={"q": "restored annual leave", "as_of": "2024-01-01"}) as response:
        assert response.status_code == 200
        events = dict(_sse_events("".join(response.iter_text())))
    assert events.get("error", {}).get("status") == 503


# -- the answer cache ---------------------------------------------------------------


def test_a_cache_hit_returns_the_same_payload_without_generating_again(
        cfg: Config, store: Store, monkeypatch, tmp_path):
    """`serve/cache.py` had 25 tests and was imported by nothing
    (results/eval-010-capacity.md section 8) -- every request paid full generation. Wired in,
    an identical second question must come back byte-for-byte without touching the model."""
    _CountingGenerator.calls = 0
    monkeypatch.setattr("warrant.generate.model.Generator", _CountingGenerator)
    c = cfg.model_copy(deep=True)
    c.store.path = str(tmp_path / "warrant.sqlite3")  # a cache file this test alone writes
    app = create_app(c, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guard.Guards(enabled=False))
    params = {"q": "restored annual leave", "as_of": "2018-06-01"}
    with TestClient(app) as client:
        first = client.get("/api/ask", params=params)
        second = client.get("/api/ask", params=params)
        metrics = client.get("/metrics").text

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert _CountingGenerator.calls == 1, "the second identical question must not regenerate"
    assert 'warrant_cache_total{outcome="miss"} 1' in metrics
    assert 'warrant_cache_total{outcome="hit"} 1' in metrics


def test_a_pinned_system_time_bypasses_the_cache(cfg: Config, store: Store, monkeypatch,
                                                 tmp_path):
    """`serve/cache.py`'s module docstring: a replay pinned to a past `system_time` must not
    read this cache, because belief invalidation is defined against *current* belief -- an
    entry correctly dead for a live request may be exactly what a pinned request wants. Since
    `/api/ask` has no other way to reach a non-live `system_time`, this is also the only place
    that rule is reachable at all."""
    _CountingGenerator.calls = 0
    monkeypatch.setattr("warrant.generate.model.Generator", _CountingGenerator)
    c = cfg.model_copy(deep=True)
    c.store.path = str(tmp_path / "warrant.sqlite3")
    app = create_app(c, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guard.Guards(enabled=False))
    # After T0 (when the synthetic store started believing everything, see SYNTHETIC's
    # `store.add(..., system_from=T0)`), so the pinned request still finds its evidence
    # in force and does not fail output validation for an unrelated reason.
    params = {"q": "restored annual leave", "as_of": "2018-06-01",
              "system_time": "2026-03-01T00:00:00+00:00"}
    with TestClient(app) as client:
        first = client.get("/api/ask", params=params)
        second = client.get("/api/ask", params=params)
        metrics = client.get("/metrics").text

    assert first.status_code == second.status_code == 200
    assert _CountingGenerator.calls == 2, "a pinned system_time must generate every time"
    assert 'warrant_cache_total{outcome="hit"}' not in metrics
    assert 'warrant_cache_total{outcome="miss"}' not in metrics


def test_an_omitted_as_of_resolves_to_the_corpus_latest_snapshot(client: TestClient):
    """Defaulting to today's date instead would rotate every cache key at midnight UTC for a
    reason the corpus itself cannot see -- the snapshot date only moves when something is
    ingested. `_meta().latest` is that snapshot date."""
    meta = client.get("/api/meta").json()
    r = client.get("/api/ask", params={"q": "restored annual leave"})
    assert r.status_code == 200
    assert r.json()["as_of"] == meta["latest"]


# -- follow-up, streamed ------------------------------------------------------------------


class _AbstainingGenerator:
    """Abstains, echoing every offered excerpt back as `cited` so `dangling_references` has
    real text to scan -- the same shape `finish_followup` builds its `widen` offers from."""

    def answer(self, question, excerpts, *, as_of, scope):
        cited = {vid: text for vid, _heading, text in excerpts}
        return Answer(question, as_of, scope, [], False, cited)


def _followup_stream_app(cfg: Config, store: Store, tmp_path, *, guards=None) -> TestClient:
    c = cfg.model_copy(deep=True)
    # A private cache file, same reason every other generate=True test in this module takes
    # one: `cfg` is shared, and an identical question served from another test's cache would
    # never reach the generator this test is asserting about.
    c.store.path = str(tmp_path / "warrant.sqlite3")
    app = create_app(c, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guards or guard.Guards(enabled=False))
    return TestClient(app)


def test_followup_stream_pins_before_any_claim_and_the_plain_endpoint_still_works(
        cfg: Config, store: Store, monkeypatch, tmp_path):
    """The whole point of `pinned`: which exchange a turn answers from -- date, scope, how
    much evidence -- is known the instant the parent trace loads, seconds before generation
    finishes. It has to precede every `claim` frame, not merely accompany the `done` one."""
    monkeypatch.setattr("warrant.generate.model.Generator", _AnsweringGenerator)
    with _followup_stream_app(cfg, store, tmp_path) as client:
        parent = client.get("/api/ask", params={"q": "restored annual leave",
                                                 "as_of": "2018-06-01"})
        trace_id = parent.json()["trace_id"]
        assert trace_id

        with client.stream("GET", "/api/ask/followup/stream",
                           params={"trace_id": trace_id, "q": "how long exactly?"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = _sse_events("".join(response.iter_text()))

        names = [name for name, _ in events]
        assert names[0] == "pinned"
        if "claim" in names:
            assert names.index("pinned") < names.index("claim")
        by_name = dict(events)
        pinned = by_name["pinned"]
        assert pinned["parent_trace_id"] == trace_id
        assert pinned["as_of"] == "2018-06-01"
        assert pinned["scope"] == "government-wide"
        assert pinned["evidence_count"] > 0
        done = by_name["done"]
        assert done["kind"] == "answered"
        assert done["abstained"] is False
        assert done["trace_id"]

        # The non-streaming form is kept, unchanged, for the tests and scripts that already
        # depend on it -- this is a transport addition, not a replacement.
        plain = client.get("/api/ask/followup", params={"trace_id": trace_id,
                                                         "q": "how long exactly?"})
        assert plain.status_code == 200
        assert plain.json()["kind"] == "answered"


def test_followup_stream_offers_widen_only_on_abstention(cfg: Config, monkeypatch, tmp_path):
    """`widen` is sent when the generator abstains, naming what the pinned text refers to
    that the pinned set itself does not hold -- the case a follow-up exists to close.

    The parent exchange is pinned directly, the same way `tests/test_followup.py` builds one,
    rather than through a live `/api/ask` retrieval: with only two chunks in the corpus,
    retrieval admits both regardless of relevance, and the widened-away chunk would then
    already be in the pinned set -- which is precisely the case this test needs to rule out,
    not accidentally reintroduce."""
    monkeypatch.setattr("warrant.generate.model.Generator", _AbstainingGenerator)
    c = cfg.model_copy(deep=True)
    c.store.path = str(tmp_path / "warrant.sqlite3")
    traces_path = tmp_path / "traces.sqlite3"
    c.store.traces = str(traces_path)
    with Store(tmp_path / "corpus.sqlite3") as corpus, TraceStore(traces_path) as traces:
        corpus.add([
            Chunk(chunk_id="630.306#a", section_id="630.306", title=5, part="630", anchor="a",
                  heading="Restored annual leave",
                  text="annual leave restored must be scheduled within two years, except "
                       "as provided in paragraph (c) of this section",
                  valid_from="2017-01-01"),
            Chunk(chunk_id="630.306#c", section_id="630.306", title=5, part="630", anchor="c",
                  heading="Restored annual leave",
                  text="the head of the agency may extend the deadline in an emergency",
                  valid_from="2017-01-01"),
        ], system_from=T0)
        trace = Trace(query="how long do I have?", as_of="2018-06-01", scope="government-wide",
                      final=("630.306#a@2017-01-01",), admitted=1, config_hash=c.hash)
        trace_id = traces.record(trace)

        app = create_app(c, generate=True, store=corpus, warm=False, thread_limit=2,
                         guards=guard.Guards(enabled=False))
        with TestClient(app) as client, client.stream(
                "GET", "/api/ask/followup/stream",
                params={"trace_id": trace_id, "q": "what's the exception?"}) as response:
            events = _sse_events("".join(response.iter_text()))

    by_name = dict(events)
    assert by_name["pinned"]["evidence_count"] == 1
    assert by_name["done"]["kind"] == "insufficient"
    assert by_name["done"]["abstained"] is True
    assert any(o["chunk_id"] == "630.306#c" for o in by_name["widen"])


def test_a_followup_stream_refusal_arrives_as_an_event_after_pinned(
        cfg: Config, store: Store, monkeypatch, tmp_path):
    """Same reasoning as `/api/ask/stream`: by the time admission control can refuse, this
    response has already been a 200 with an open body for one `pinned` frame's worth of time,
    so the refusal has nowhere to go but an event."""
    monkeypatch.setattr("warrant.generate.model.Generator", _AnsweringGenerator)
    with _followup_stream_app(cfg, store, tmp_path) as client:
        parent = client.get("/api/ask", params={"q": "restored annual leave",
                                                 "as_of": "2018-06-01"})
        trace_id = parent.json()["trace_id"]
        assert trace_id

        from warrant.serve import api as api_module

        def refuse(*a, **k):
            raise api_module.HTTPException(503, "generator at capacity")

        monkeypatch.setattr(api_module, "_generate_answer", refuse)

        with client.stream(
                "GET", "/api/ask/followup/stream",
                params={"trace_id": trace_id, "q": "how long exactly?"}) as response:
            assert response.status_code == 200
            events = _sse_events("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "pinned"
    assert dict(events)["error"]["status"] == 503


def test_followup_stream_on_an_unknown_trace_id_is_an_error_event_not_a_404(
        cfg: Config, store: Store, tmp_path):
    """`ask_followup` (non-streaming) 404s on a bad trace_id before anything opens. Streamed,
    the 200 is already committed by the time the trace lookup runs, so the same fact has to
    travel as an `error` frame instead."""
    with _followup_stream_app(cfg, store, tmp_path) as client, client.stream(
            "GET", "/api/ask/followup/stream",
            params={"trace_id": "never-recorded", "q": "anything at all"}) as response:
        assert response.status_code == 200
        events = _sse_events("".join(response.iter_text()))
    by_name = dict(events)
    assert "pinned" not in by_name, "nothing to pin to -- the exchange does not exist"
    assert by_name["error"]["status"] == 404


def test_followup_stream_is_metered_as_a_generation_route(cfg: Config, store: Store, tmp_path):
    """`/api/ask/followup/stream` costs a generation slot exactly like `/api/ask` and
    `/api/ask/followup` -- it must sit in the rate limiter's `answer_paths`, not the cheaper
    read bucket a widen or a section lookup uses."""
    c = cfg.model_copy(deep=True)
    c.store.path = str(tmp_path / "warrant.sqlite3")
    app = create_app(c, generate=True, store=store, warm=False, thread_limit=2,
                     guards=guard.Guards(answer=guard.RateLimiter(1e-6, burst=1),
                                         read=guard.RateLimiter(1e-6, burst=1)))
    with TestClient(app) as client:
        origin = {"Origin": "http://localhost:5173"}
        # Burn the bucket's single token on the route under test rather than on /api/ask.
        # The trace id is deliberately unknown, so `load_exchange` raises before anything
        # reaches the generator -- which is what keeps this a test of the *limiter* and lets
        # it run on an install with no torch, the way CI installs it.
        with client.stream("GET", "/api/ask/followup/stream",
                           params={"trace_id": "no-such-exchange", "q": "anything at all"},
                           headers=origin) as first:
            assert first.status_code == 200, "the route exists and the stream opens"
            assert "error" in "".join(first.iter_text()), "an unknown exchange is an event"

        with client.stream("GET", "/api/ask/followup/stream",
                           params={"trace_id": "no-such-exchange", "q": "anything at all"},
                           headers=origin) as second:
            assert second.status_code == 429, "the second call must hit the answer bucket"


def test_an_exchange_can_be_reopened_from_its_trace(client: TestClient):
    """The durable half already existed -- an exchange is its trace id, and traces persist.
    The client was the only thing throwing the handle away, so a refresh lost the date, the
    evidence and every turn while the server could have handed all of it back."""
    body = client.get("/api/ask", params={"q": "restored annual leave",
                                          "as_of": "2024-01-01"}).json()
    tid = body["trace_id"]
    assert tid, "an ask must leave a trace to reopen"

    again = client.get(f"/api/exchange/{tid}").json()
    assert again["trace_id"] == tid
    assert again["as_of"] == body["as_of"]
    assert [e["version_id"] for e in again["evidence"]] == \
           [e["version_id"] for e in body["evidence"]], (
        "reopening must return the evidence the turn was answered from, not today's ranking")


def test_reopening_an_unknown_exchange_is_a_404(client: TestClient):
    assert client.get("/api/exchange/nosuchtrace").status_code == 404
