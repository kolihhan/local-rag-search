# Design decisions

## RRF instead of score mixing
BM25 and dense scores have different scales. Reciprocal-rank fusion uses rank positions and avoids pretending those raw scores are calibrated.

On the current frozen 64-query Confluence core slice, RRF reached Recall@10 0.7630 and MRR@20 0.7418 versus BM25 0.7341 / 0.6792 and Dense 0.6797 / 0.6919. It recovered five BM25 top-10 misses and lost two BM25 top-10 hits. This supports retaining hybrid retrieval on this slice; it is not a universal retrieval claim.

The earlier 12-query metadata-extra development run (BM25 0.7500, Dense 0.8333, RRF 0.9167 Recall@10) is preserved as historical evidence rather than used as the current source of truth.

## Persistent dense cache
The Dense index uses one atomic JSON cache. It records schema, full provider and
model identity, corpus identity, ordered document IDs, count, dimension, and
vectors. Model/corpus/order changes rebuild it. A compatible cache with corrupt
counts, dimensions, or vectors raises instead of silently scoring bad data.

## Explicit Qwen query/document contract

The real provider exposes `embed_query` and `embed_documents`. The serious benchmark protocol freezes this query form:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

Documents have no instruction. Ollama owns tokenization/pooling, while this project validates, L2-normalizes, and scores the returned 1,024-dimensional vectors by dot product. The frozen local model tag and digest are validated before cache reuse or inference.

## Context weighting
Collection context is repeated in the demo dense representation so a document's role (incident, runbook, architecture) can influence semantic ranking. This is explicit and test-covered rather than a hidden rank boost.

The EnterpriseRAG corpus has empty collection context, so this demo weighting does not distinguish the serious treatments.

## `vsearch` is only CLI sugar
It calls `SearchService.search(..., mode="vec")`. FastAPI uses the existing `/search` route with `mode: "vec"`; there is no duplicate vector-search endpoint.

## Why provenance is always returned

`SearchResult` always carries BM25/Dense/RRF provenance. The current `explain` flag is therefore presentation/interface compatibility rather than a switch that changes the underlying result schema. Keeping the provenance stable makes the browser, CLI, API, and evaluation code share one result contract.
