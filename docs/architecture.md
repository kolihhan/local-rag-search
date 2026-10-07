# Architecture

## Canonical retrieval decision path

```text
query
 ├─ BM25 ─────────────────────────┐
 └─ Qwen3-Embedding-0.6B Dense ───┤
                                  ↓
                           RRF (k = 60)
                                  ↓
                             ranked IDs
```

This is the measured retrieval architecture used by the current portfolio evidence. Dense uses instructed query encoding, plain document encoding, local L2 normalization, exact dot scoring, and a persistent identity-checked cache. RRF combines rank positions rather than incompatible raw BM25/vector score scales.

The current canonical report is `runs/enterprise-rag-qwen-core-v1/report.json`: on the frozen 64-query Confluence slice, RRF reached 0.7630 Recall@10 and 0.7418 MRR@20. The older 12-query metadata-extra DEV experiment remains historical development evidence, not the current headline.

The offline demo remains deliberately lightweight and does not make Ollama mandatory.

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
SearchResult + rank provenance + context
```

`SearchService` owns retrieval and stable `get()` / `get_many()` document primitives. Explain mode is a presentation concern: the service always computes provenance, while CLI/FastAPI decide whether to display/serialize it.

Typed HyDE input and token-overlap reranking remain demonstration features; they are not canonical benchmark treatments or evidence. The API and CLI do not implement retrieval math.
