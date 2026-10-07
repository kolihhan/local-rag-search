from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .index import build_search_service
from .query import QueryPlan


class IndexRequest(BaseModel):
    path: str
    embedding: Literal["simple", "ollama"] = "simple"


class SearchRequest(BaseModel):
    query: str
    mode: Literal["lex", "vec", "hybrid"] = "hybrid"
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


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local RAG Search</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f6f7fb; color: #18202a; }
    main { width: min(960px, calc(100% - 32px)); margin: 56px auto; }
    .eyebrow { font-size: 13px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #667085; }
    h1 { margin: 8px 0 12px; font-size: clamp(34px, 6vw, 58px); letter-spacing: -.045em; line-height: 1; }
    .lede { max-width: 720px; margin: 0 0 28px; color: #596273; font-size: 18px; line-height: 1.6; }
    .pipeline { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px; }
    .chip { padding: 7px 11px; border: 1px solid #e0e4ea; border-radius: 999px; background: white; color: #3b4554; font-size: 13px; font-weight: 650; }
    .search { display: flex; gap: 10px; padding: 10px; border: 1px solid #dfe3ea; border-radius: 16px; background: white; box-shadow: 0 12px 35px rgba(25, 32, 44, .07); }
    input { flex: 1; min-width: 0; border: 0; outline: 0; padding: 10px 12px; font: inherit; font-size: 16px; background: transparent; }
    button { border: 0; border-radius: 11px; padding: 11px 18px; font: inherit; font-weight: 700; cursor: pointer; background: #18202a; color: white; }
    button:disabled { opacity: .55; cursor: wait; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 2px 30px; }
    .examples button { padding: 7px 10px; border: 1px solid #dfe3ea; border-radius: 999px; background: transparent; color: #667085; font-size: 12px; font-weight: 600; }
    .meta { display: flex; justify-content: space-between; gap: 12px; margin: 0 2px 12px; color: #7b8493; font-size: 13px; }
    #results { display: grid; gap: 12px; }
    .result { padding: 18px; border: 1px solid #e1e5eb; border-radius: 15px; background: white; }
    .result-head { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
    .title { font-weight: 750; font-size: 16px; }
    .docid { margin-top: 4px; color: #8a93a2; font-size: 12px; }
    .snippet { margin: 12px 0 14px; color: #4d5766; line-height: 1.55; }
    .ranks { display: flex; flex-wrap: wrap; gap: 7px; }
    .rank { padding: 5px 8px; border-radius: 8px; background: #f2f4f7; color: #53606f; font-size: 12px; font-weight: 650; }
    .status { padding: 22px 4px; color: #7b8493; }
    @media (max-width: 620px) { main { margin-top: 34px; } .search { flex-direction: column; } .search button { width: 100%; } .result-head { flex-direction: column; } }
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Local-first retrieval engine</div>
  <h1>Local RAG Search</h1>
  <p class="lede">Hybrid search that keeps lexical precision and semantic recall visible: BM25 + Dense retrieval, fused with RRF, with rank provenance on every result.</p>
  <div class="pipeline">
    <span class="chip">BM25</span><span class="chip">+</span><span class="chip">Dense</span><span class="chip">→</span><span class="chip">RRF</span><span class="chip">→</span><span class="chip">Explainable results</span>
  </div>
  <form class="search" id="search-form">
    <input id="query" autocomplete="off" value="why were buyers unable to finish a purchase?" aria-label="Search query">
    <button id="submit" type="submit">Search</button>
  </form>
  <div class="examples">
    <button type="button" data-query="ORA-12516">Exact identifier</button>
    <button type="button" data-query="why were buyers unable to finish a purchase?">Semantic question</button>
    <button type="button" data-query="payment authorization outage">Mixed signal</button>
  </div>
  <div class="meta"><span id="health">Loading corpus…</span><span>POST /search · hybrid · explain=true</span></div>
  <div id="results"><div class="status">Run a query to inspect BM25, Dense, and RRF provenance.</div></div>
</main>
<script>
  const form = document.querySelector('#search-form');
  const input = document.querySelector('#query');
  const submit = document.querySelector('#submit');
  const results = document.querySelector('#results');

  const rank = (label, value) => value == null ? '' : `<span class="rank">${label} #${value}</span>`;
  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

  async function search() {
    const query = input.value.trim();
    if (!query) return;
    submit.disabled = true;
    submit.textContent = 'Searching…';
    results.innerHTML = '<div class="status">Searching local index…</div>';
    try {
      const response = await fetch('/search', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({query, mode: 'hybrid', limit: 5, explain: true})
      });
      if (!response.ok) throw new Error((await response.json()).detail || 'Search failed');
      const data = await response.json();
      results.innerHTML = data.results.map((row, index) => `
        <article class="result">
          <div class="result-head">
            <div><div class="title">${index + 1}. ${escapeHtml(row.title)}</div><div class="docid">${escapeHtml(row.doc_id)}</div></div>
            <span class="rank">RRF #${row.rrf_rank}</span>
          </div>
          <div class="snippet">${escapeHtml(row.snippet)}</div>
          <div class="ranks">${rank('BM25', row.bm25_rank)}${rank('Dense', row.dense_rank)}<span class="rank">${escapeHtml(row.collection)}</span></div>
        </article>`).join('') || '<div class="status">No results.</div>';
    } catch (error) {
      results.innerHTML = `<div class="status">${escapeHtml(error.message)}</div>`;
    } finally {
      submit.disabled = false;
      submit.textContent = 'Search';
    }
  }

  form.addEventListener('submit', event => { event.preventDefault(); search(); });
  document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => { input.value = button.dataset.query; search(); }));
  fetch('/health').then(response => response.json()).then(data => { document.querySelector('#health').textContent = `${data.indexed_documents} local documents indexed`; });
  search();
</script>
</body>
</html>"""


def create_app(*, default_corpus: str | Path | None = None, cache_dir: str | Path = ".rag-cache") -> FastAPI:
    app = FastAPI(title="Local RAG Search", version="0.1.0")
    state = {"service": None, "cache_dir": Path(cache_dir)}
    if default_corpus is not None:
        state["service"] = build_search_service(default_corpus, cache_dir=cache_dir)

    def service():
        if state["service"] is None:
            raise HTTPException(status_code=409, detail="index a corpus first")
        return state["service"]

    @app.get("/", response_class=HTMLResponse)
    def portfolio_demo():
        return DEMO_HTML

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
