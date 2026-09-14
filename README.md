# Local RAG Search

A small local search service for RAG projects. It combines BM25 with Qwen embeddings and merges the two rankings with reciprocal rank fusion (RRF).

The same `SearchService` is available through a CLI and FastAPI.

## Results

I compared BM25, dense retrieval, and RRF on 12 Confluence questions from the EnterpriseRAG-Bench v1.0.0 metadata-extra set.

| Method | Recall@10 | MRR@20 | Mean latency |
|---|---:|---:|---:|
| BM25 | 0.7500 | 0.5511 | 60.57 ms |
| Qwen Dense | 0.8333 | 0.7917 | 315.18 ms |
| BM25 + Dense RRF | **0.9167** | **0.7986** | 372.06 ms |

RRF recovered two BM25 top-10 misses in this set, but the dense path was also noticeably slower. I kept hybrid search because it helped on these cases without losing BM25 hits.

This is a small 12-query subset, not an EnterpriseRAG-Bench leaderboard result.

## How it works

```text
query
  |
  +--> BM25
  |
  +--> Qwen embeddings
          |
      merge with RRF
          |
        results
```

BM25 is useful for exact names, IDs, and error codes. Dense retrieval helps more with paraphrases and conceptual matches. RRF combines the rankings without trying to compare BM25 and cosine scores directly.

A few other things in the repo:

- stable document IDs and collection context
- persistent embedding cache
- optional typed query fields (`lex`, `vec`, `hyde`, `intent`)
- explain mode showing where each result came from
- CLI and FastAPI on top of the same service

Search works without an LLM. Query expansion is optional.

## Quickstart

```bash
uv sync
uv run rag-search --corpus demo_docs query "why were buyers unable to finish a purchase?" --explain
```

FastAPI:

```bash
uv run uvicorn local_rag.api:demo_app --reload
```

With Ollama embeddings:

```bash
ollama pull qwen3-embedding:0.6b
uv run rag-search --embedding ollama --corpus demo_docs query "payment failure"
```

Windows users can use `run-demo.cmd` or `run-api.cmd`.

## Examples

```bash
rag-search search "ORA-12516"
rag-search vsearch "why were buyers unable to finish a purchase?"
rag-search query "payment incident" --lex "payment authorization" --vec "users could not finish checkout" --explain
```

The repo also includes a small deterministic 6-query demo test for checking the plumbing. I keep that separate from the 12-query Qwen comparison above.

More detail on the benchmark setup is in `docs/p2-enterprise-rag-benchmark.md`. The project was partly inspired by QMD's local retrieval and typed-query ideas; `docs/qmd-inspiration.md` explains what I reused and what I changed.

## Limits

- Twelve questions are too few for broad retrieval-quality claims.
- The deterministic embedding path is only for demos/tests; the measured dense result uses local `qwen3-embedding:0.6b`.
- The optional token-overlap reranker is demo-only.

## Development

```bash
python -m pytest -q
python -m compileall -q src evaluation tests
```
