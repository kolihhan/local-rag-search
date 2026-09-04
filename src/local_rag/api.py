from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .index import build_search_service
from .query import QueryPlan


class IndexRequest(BaseModel):
    path: str
    embedding: str = "simple"


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    rerank: bool = False
    explain: bool = False


class TypedQueryRequest(BaseModel):
    original: str
    intent: str = ""
    lex: list[str] = []
    vec: list[str] = []
    hyde: list[str] = []
    limit: int = Field(default=10, ge=1, le=100)
    rerank: bool = False
    explain: bool = False


class BatchRequest(BaseModel):
    doc_ids: list[str]


def create_app(*, default_corpus: str | Path | None = None, cache_dir: str | Path = ".rag-cache") -> FastAPI:
    app = FastAPI(title="Local RAG Search", version="0.1.0")
    state = {"service": None, "cache_dir": Path(cache_dir)}
    if default_corpus is not None:
        state["service"] = build_search_service(default_corpus, cache_dir=cache_dir)

    def service():
        if state["service"] is None:
            raise HTTPException(status_code=409, detail="index a corpus first")
        return state["service"]

    @app.get("/health")
    def health():
        current = state["service"]
        return {"ok": True, "indexed_documents": len(current.documents) if current else 0}

    @app.post("/index")
    def index_corpus(request: IndexRequest):
        state["service"] = build_search_service(request.path, cache_dir=state["cache_dir"], embedding=request.embedding)
        return {"indexed_documents": len(state["service"].documents)}

    @app.post("/search")
    def search(request: SearchRequest):
        results = service().search(request.query, mode=request.mode, limit=request.limit, rerank=request.rerank, explain=request.explain)
        return {"results": [asdict(row) for row in results]}

    @app.post("/query")
    def typed_query(request: TypedQueryRequest):
        plan = QueryPlan(request.original, request.intent, tuple(request.lex), tuple(request.vec), tuple(request.hyde))
        results = service().query(plan, limit=request.limit, rerank=request.rerank, explain=request.explain)
        return {"results": [asdict(row) for row in results]}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str):
        try:
            return asdict(service().get(doc_id))
        except KeyError:
            raise HTTPException(status_code=404, detail="document not found")

    @app.post("/documents/batch")
    def get_documents(request: BatchRequest):
        return {"documents": [asdict(doc) for doc in service().get_many(request.doc_ids)]}

    return app


app = create_app()
_REPO_ROOT = Path(__file__).resolve().parents[2]
demo_app = create_app(default_corpus=_REPO_ROOT / "demo_docs", cache_dir=_REPO_ROOT / ".rag-cache")
