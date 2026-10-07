# Design decisions

## RRF instead of score mixing

BM25 and dense cosine scores have different scales. Reciprocal-rank fusion uses rank positions and avoids pretending those raw scores are calibrated.

The current frozen 64-query Confluence core-compatible run retained RRF: BM25 Recall@10 was 0.7341, Qwen Dense 0.6797, and RRF 0.7630. RRF recovered five BM25 top-10 misses while losing two BM25 top-10 hits, and reached 0.7418 MRR@20. This supports hybrid retrieval on this slice; it is not a universal retrieval claim.

The older 12-query metadata-extra DEV result (0.7500 BM25 / 0.8333 Dense / 0.9167 RRF Recall@10) is preserved as historical development evidence, not the current portfolio headline.

## Persistent dense cache

The Dense index uses one atomic JSON cache. It records schema, full provider and model identity, corpus identity, ordered document IDs, count, dimension, and vectors. Model/corpus/order changes rebuild it. A compatible cache with corrupt counts, dimensions, or vectors raises instead of silently scoring bad data.

## Explicit Qwen query/document contract

The real provider exposes `embed_query` and `embed_documents`. Queries use the official Qwen instruction template; documents have no instruction. Ollama owns tokenization/pooling, while this project validates, L2-normalizes, and scores the returned 1,024-dimensional vectors by dot product. Frozen benchmark runs validate the expected local model identity before cache reuse or inference.

## Context weighting

Collection context is repeated in the demo dense representation so a document's role (incident, runbook, architecture) can influence semantic ranking. This is explicit and test-covered rather than a hidden rank boost.

The EnterpriseRAG corpus has empty collection context, so this demo weighting does not distinguish the canonical benchmark treatments.

## Explain is presentation, not retrieval behavior

`SearchService` always computes rank provenance because BM25/Dense/RRF ranks are part of the retrieval result. CLI `--explain` and FastAPI `explain: true` control whether that provenance is displayed or serialized; they do not alter ranking.

## `vsearch` is only CLI sugar

It calls `SearchService.search(..., mode="vec")`. FastAPI uses the existing `/search` route with `mode: "vec"`; there is no duplicate vector-search endpoint.
