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

This is the complete serious P2 architecture. Dense uses explicit instructed
query encoding, plain document encoding, local L2 normalization, exact dot
scoring, and one strict persistent cache. RRF is the existing small formula.
The frozen DEV verdict is **KEEP HYBRID**. The offline demo remains lightweight
and does not make Ollama mandatory.

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
SearchResult + explain ranks + context
```

`SearchService` also owns stable `get()` and `get_many()` document primitives.
Typed HyDE input and token-overlap reranking remain demonstration features;
they are not serious benchmark treatments or evidence. The API and CLI do not
implement retrieval logic.
