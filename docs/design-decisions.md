# Design decisions

## RRF instead of score mixing
BM25 and dense cosine scores have different scales. Reciprocal-rank fusion uses rank positions and avoids pretending those raw scores are calibrated.

The frozen EnterpriseRAG metadata-extra Confluence slice retained RRF: it improved Recall@10 from
0.7500 (BM25) and 0.8333 (Dense) to 0.9167 with no BM25 top-10 loss. This is a
narrow development decision, not a general retrieval claim.

## Persistent dense cache
The Dense index uses one atomic JSON cache. It records schema, full provider and
model identity, corpus identity, ordered document IDs, count, dimension, and
vectors. Model/corpus/order changes rebuild it. A compatible cache with corrupt
counts, dimensions, or vectors raises instead of silently scoring bad data.

## Explicit Qwen query/document contract

The real provider exposes `embed_query` and `embed_documents`. Queries use the
official Qwen instruction template; documents have no instruction. Ollama owns
tokenization/pooling, while this project validates, L2-normalizes, and scores
the returned 1,024-dimensional vectors by dot product. The frozen local model
tag and digest are validated before cache reuse or inference.

## Context weighting
Collection context is repeated in the demo dense representation so a document's role (incident, runbook, architecture) can influence semantic ranking. This is explicit and test-covered rather than a hidden rank boost.

The EnterpriseRAG corpus has empty collection context, so this demo weighting
does not distinguish the serious treatments.

## `vsearch` is only CLI sugar
It calls `SearchService.search(..., mode="vec")`. FastAPI uses the existing `/search` route with `mode: "vec"`; there is no duplicate vector-search endpoint.
