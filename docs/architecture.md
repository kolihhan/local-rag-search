# Architecture

## Serious retrieval decision path

```text
query
 ├─ existing BM25 ────────────────┐
 └─ Qwen3-Embedding-0.6B Dense ───┤
                                  ↓
                           RRF (k = 60)
                                  ↓
                             top 20 IDs
```

This is the complete serious P2 architecture. Dense uses the frozen query/document encoding contract, local L2 normalization, exact dot-product scoring, and one strict persistent cache. RRF is the existing small rank-fusion formula.

The current portfolio-facing validation is the frozen **64-query Confluence core slice** in `runs/enterprise-rag-qwen-core-v1/report.json`; it retains the hybrid path on that slice. The earlier 12-query metadata-extra run is historical development evidence, not the current source of truth.

The offline demo remains lightweight and does not make Ollama mandatory.

## Demo/product surface

```text
CLI / FastAPI
      ↓
SearchService
 ┌──────────────┬──────────────┐
 │ BM25         │ Dense index  │
 │ lex          │ vec / hyde   │
 └──────────────┴──────────────┘
          ↓
         RRF
          ↓
 optional demo-only bounded reranker
          ↓
SearchResult + provenance ranks + context
```

`SearchService` also owns stable `get()` and `get_many()` document primitives. Typed HyDE input and token-overlap reranking remain demonstration features; they are not serious benchmark treatments or evidence. The API and CLI do not implement retrieval logic.

Rank provenance is part of `SearchResult` itself. The current `explain` argument is retained for interface compatibility/presentation, but it does not change retrieval math or suppress provenance fields in the service result.

The bundled browser demo uses the deterministic lightweight `SimpleEmbeddingProvider`; the measured serious path uses local Qwen3-Embedding-0.6B through Ollama. These are intentionally different product-demo and benchmark configurations.
