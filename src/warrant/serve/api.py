"""HTTP API and static host for the Warrant UI.

Endpoints exist to make the system's distinctive behaviour *visible*, not to be a generic
search API. Each one answers a question the README makes a claim about:

    /api/meta                 what is in the corpus, and the date range the slider spans
    /api/ask                  the answer for a scope and a date, with claims, spans, trace
    /api/section/{id}         a section's whole version history
    /api/diff                 what changed between two versions of a section
    /api/budget               the failure budget, read from a recorded run

Models load lazily and are shared across requests. The dense encoder, the cross-encoder and
the generator are each several hundred MB to a few GB; constructing one per request turned a
millisecond retrieval into the dominant cost of an evaluation once already, and an API is the
same trap with more traffic.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..generate.answer import excerpts_for
from ..index.store import Store
from ..retrieve.dense import DenseIndex
from ..retrieve.hybrid import Retriever
from ..retrieve.scope import PART_RESTRICTIONS, Scope

UI_DIR = Path(__file__).resolve().parents[3] / "ui" / "dist"


@dataclass
class Runtime:
    """Lazily-constructed, process-wide singletons."""

    cfg: Config
    generate: bool = True
    _store: Store | None = None
    _retriever: Retriever | None = None
    _generator: Any = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def store(self) -> Store:
        if self._store is None:
            if not self.cfg.store_path.exists():
                raise HTTPException(503, "no corpus built; run `make fetch && make build`")
            # Store hands out a thread-local connection, which is what makes this safe
            # under FastAPI's threadpool for sync endpoints. query_only is set per
            # connection, so the serving path asserts it on first use in each thread.
            self._store = Store(self.cfg.store_path)
        return self._store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            index = (DenseIndex.load(self.cfg.dense_path)
                     if self.cfg.index.dense.enabled
                     and DenseIndex.exists(self.cfg.dense_path) else None)
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
            )
        return self._retriever

    @property
    def generator(self):
        if self._generator is None and self.generate:
            from ..generate.model import Generator

            self._generator = Generator()
        return self._generator


def create_app(cfg: Config | None = None, *, generate: bool = True) -> FastAPI:
    rt = Runtime(cfg or Config.load(), generate=generate)
    app = FastAPI(title="warrant", docs_url="/api/docs", openapi_url="/api/openapi.json")

    # -- metadata ---------------------------------------------------------------

    @lru_cache(maxsize=1)
    def _meta() -> dict:
        db = rt.store.db
        row = db.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT section_id) sections, "
            "MIN(valid_from) lo, MAX(valid_from) hi FROM chunk WHERE system_to IS NULL"
        ).fetchone()
        parts = [dict(r) for r in db.execute(
            "SELECT part, COUNT(DISTINCT section_id) sections, COUNT(*) chunks "
            "FROM chunk WHERE system_to IS NULL GROUP BY part ORDER BY part")]
        facets: dict[str, list[str]] = {}
        for restriction in PART_RESTRICTIONS.values():
            for facet, values in restriction.items():
                facets.setdefault(facet, [])
                for v in sorted(values):
                    if v not in facets[facet]:
                        facets[facet].append(v)
        return {
            "chunks": row["n"], "sections": row["sections"],
            "parts": parts, "part_count": len(parts),
            "earliest": row["lo"], "latest": rt.cfg.corpus.history_floor,
            "history_floor": rt.cfg.corpus.history_floor,
            "horizon": max(r["hi"] for r in [row]) if row["hi"] else None,
            "facets": {k: sorted(v) for k, v in facets.items()},
            "config_hash": rt.cfg.hash,
            "final_k": rt.cfg.retrieve.final_k,
        }

    @app.get("/api/meta")
    def meta() -> dict:
        return _meta()

    def _scope(pay_system: str | None, service: str | None) -> Scope:
        facets = {k: v for k, v in
                  {"pay_system": pay_system, "service": service}.items() if v}
        try:
            return Scope.of(**facets)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- ask --------------------------------------------------------------------

    @app.get("/api/ask")
    def ask(q: str = Query(min_length=2, max_length=512),
            as_of: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$"),
            pay_system: str | None = None,
            service: str | None = None,
            generate: bool = True) -> dict:
        scope = _scope(pay_system, service)
        rt.store.read_only()
        trace = rt.retriever.retrieve(q, as_of=as_of, scope=scope)
        rows = _rows(rt.store, trace.final)

        payload: dict[str, Any] = {
            "question": q, "as_of": as_of, "scope": scope.describe(),
            "excluded_parts": trace.excluded_parts,
            "trace": {
                "admitted": trace.admitted,
                "corpus": _meta()["chunks"],
                "stages": [
                    {"name": "predicates", "out": trace.admitted},
                    {"name": "lexical", "out": len(trace.lexical)},
                    {"name": "dense", "out": len(trace.dense)},
                    {"name": "fusion", "out": len(trace.fused)},
                    {"name": "rerank", "out": len(trace.reranked)},
                    {"name": "final", "out": len(trace.final)},
                ],
            },
            "evidence": [_evidence(r) for r in rows],
            "claims": [],
            "abstained": True,
        }
        if not (generate and rt.generate):
            return payload

        answer = rt.generator.answer(q, excerpts_for(rt.store, trace),
                                     as_of=as_of, scope=scope.describe())
        payload["abstained"] = answer.abstained
        payload["parse_failed"] = answer.parse_failed
        payload["claims"] = [
            {
                "text": c.text,
                "grounded": c.grounded,
                "citations": [
                    {"version_id": vid,
                     "span": None if sp is None else
                     {"start": sp.start, "end": sp.end, "score": round(sp.score, 3)}}
                    for vid, sp in c.spans.items()
                ],
            }
            for c in answer.claims
        ]
        return payload

    # -- version history --------------------------------------------------------

    @app.get("/api/section/{section_id}")
    def section(section_id: str) -> dict:
        rows = rt.store.db.execute(
            "SELECT version_id, chunk_id, anchor, heading, part, text, valid_from, valid_to "
            "FROM chunk WHERE section_id = ? AND system_to IS NULL "
            "ORDER BY valid_from, id", (section_id,)).fetchall()
        if not rows:
            raise HTTPException(404, f"no section {section_id}")
        versions: dict[str, dict] = {}
        for r in rows:
            v = versions.setdefault(r["valid_from"], {
                "valid_from": r["valid_from"], "valid_to": r["valid_to"],
                "heading": r["heading"], "part": r["part"], "paragraphs": []})
            v["paragraphs"].append(
                {"anchor": r["anchor"], "version_id": r["version_id"], "text": r["text"]})
        return {"section_id": section_id,
                "heading": rows[0]["heading"],
                "part": rows[0]["part"],
                "versions": sorted(versions.values(), key=lambda v: v["valid_from"])}

    @app.get("/api/diff")
    def diff(section_id: str, a: str, b: str) -> dict:
        """Word-level before/after between two versions of a section."""
        left, right = _version_text(rt.store, section_id, a), _version_text(
            rt.store, section_id, b)
        if left is None or right is None:
            raise HTTPException(404, "no such version")
        sm = difflib.SequenceMatcher(None, left.split(), right.split(), autojunk=False)
        ops = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            ops.append({"op": tag,
                        "before": " ".join(left.split()[i1:i2]),
                        "after": " ".join(right.split()[j1:j2])})
        return {"section_id": section_id, "a": a, "b": b,
                "similarity": round(sm.ratio(), 4), "ops": ops}

    # -- failure budget ---------------------------------------------------------

    @app.get("/api/budget")
    def budget() -> dict:
        """The recorded failure budget. Read from disk, never recomputed per request.

        Recomputing would take minutes and, worse, would let the dashboard drift from the
        numbers in `results/` that the README quotes.
        """
        path = rt.cfg.budget_path
        if not path.exists():
            raise HTTPException(503, "no recorded budget; run `make autopsy`")
        import json

        return json.loads(path.read_text(encoding="utf-8"))

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


def _evidence(row) -> dict:
    return {
        "version_id": row["version_id"], "chunk_id": row["chunk_id"],
        "section_id": row["section_id"], "anchor": row["anchor"],
        "heading": row["heading"], "part": row["part"], "subpart": row["subpart"],
        "text": row["text"], "valid_from": row["valid_from"], "valid_to": row["valid_to"],
    }


def _version_text(store: Store, section_id: str, valid_from: str) -> str | None:
    rows = store.db.execute(
        "SELECT text FROM chunk WHERE section_id = ? AND valid_from = ? "
        "AND system_to IS NULL ORDER BY id", (section_id, valid_from)).fetchall()
    return " ".join(r["text"] for r in rows) if rows else None
