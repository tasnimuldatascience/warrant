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

import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from warrant.config import Config
from warrant.generate.answer import Answer, Claim
from warrant.index.store import Chunk, Store
from warrant.serve import api
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
    budget = tmp_path_factory.mktemp("budget") / "failure-budget.json"
    budget.write_text(json.dumps(BUDGET), encoding="utf-8")
    c.store.budget = str(budget)
    return c


@pytest.fixture(scope="module")
def client(cfg: Config, store: Store) -> TestClient:
    app = create_app(cfg, generate=False, store=store, warm=True, thread_limit=2)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def bare_client(tmp_path_factory) -> TestClient:
    """An instance with no corpus at all: the state /ready exists to report."""
    c = Config.load()
    c.store.path = str(tmp_path_factory.mktemp("empty") / "warrant.sqlite3")
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    app = create_app(c, generate=False, warm=False)
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
    app = create_app(c, generate=False, store=store, warm=False)
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
                    "models": True, "generator": False, "detail": None}


def test_a_failed_warm_up_is_reported_not_fatal(tmp_path):
    """A model that cannot be built must not stop the process from starting: a container
    that crash-loops on a stale index reports nothing at all, and /ready is the thing whose
    entire job is to say so."""
    c = Config.load()
    c.store.path = str(tmp_path / "absent.sqlite3")
    c.index.dense.enabled = False
    c.index.rerank.enabled = False
    with TestClient(create_app(c, generate=False, warm=True)) as client:
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
    app = create_app(cfg, generate=False, store=store, warm=False, thread_limit=2)
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


def test_ask_returns_claims_and_spans_when_generation_runs(cfg: Config, store: Store,
                                                           monkeypatch):
    monkeypatch.setattr("warrant.generate.model.Generator", _AnsweringGenerator)
    app = create_app(cfg, generate=True, store=store, warm=False, thread_limit=2)
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
    got = api._generate_answer(rt, "q", [], as_of="2024-06-01", scope="government-wide")
    assert got == "q@2024-06-01/government-wide"
    assert rt.stats["generate_calls"] == 1


def test_a_full_queue_is_refused_with_retry_after(cfg: Config, monkeypatch):
    """503 + Retry-After, not a 33-minute wait. The measured ceiling is 0.051 req/s and
    nothing here can raise it; the queue can only be honest about it."""
    monkeypatch.setattr(api, "GENERATE_QUEUE_WAIT_S", 0.05)
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())
    with api._GENERATION_SLOT, pytest.raises(HTTPException) as exc:
        api._generate_answer(rt, "q", [], as_of="2024-06-01", scope="government-wide")
    assert exc.value.status_code == 503
    assert exc.value.headers["Retry-After"] == str(api.RETRY_AFTER_S)
    assert rt.stats["generate_rejected"] == 1


def test_a_request_that_cannot_meet_its_deadline_is_refused(cfg: Config, monkeypatch):
    """A 420-token answer takes ~19.7 s at 21.3 tok/s and may be retried once, so starting
    one with less budget than that only burns the GPU on an answer nobody will receive."""
    monkeypatch.setattr(api, "GENERATE_DEADLINE_S", 0.0)
    rt = api.Runtime(cfg, generate=True, _generator=_FakeGenerator())
    with pytest.raises(HTTPException) as exc:
        api._generate_answer(rt, "q", [], as_of="2024-06-01", scope="government-wide")
    assert exc.value.status_code == 503
    assert "deadline" in exc.value.detail
    assert not api._GENERATION_SLOT._value < 1, "the slot must be released on refusal"


# -- threadpool cap ----------------------------------------------------------------


def test_lifespan_caps_the_threadpool(cfg: Config, store: Store):
    """Retrieval peaks at 4 threads (66 QPS) and falls to 25.6 QPS at 16; anyio's default is
    40, which is 40 workers contending for a GIL none of them releases."""
    async def limit() -> float:
        app = create_app(cfg, generate=False, store=store, warm=False, thread_limit=3)
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
    with TestClient(create_app(c, generate=False, warm=False)) as client:
        meta = client.get("/api/meta").json()
    assert meta["latest"] > meta["earliest"] >= meta["history_floor"]
