# Local RAG Search

A local retrieval service for RAG applications. It combines lexical BM25 retrieval with Qwen embeddings and reciprocal rank fusion (RRF), with the same `SearchService` exposed through a CLI and FastAPI.

## What it does

```text
query
  |
  +--> BM25
  |
  +--> Qwen dense retrieval
          |
      merge with RRF
          |
        results
```

BM25 helps with exact names, IDs, and error codes. Dense retrieval helps with paraphrases and conceptual matches. RRF combines the rankings without trying to compare BM25 and vector-similarity scores directly.

## Frozen evaluation

The current resume-facing benchmark is the frozen **EnterpriseRAG-Bench v1.0.0 core** run, not the older 12-query exploratory result that used to headline this README.

Benchmark scope:

- **64 official Confluence-only, qrel-compatible queries**
- **5,189 documents**
- BM25, Qwen dense retrieval, and RRF evaluated under one frozen corpus/query contract
- Recall@K, MRR@20, Hit@10, latency, and per-query rankings recorded in the run artifact
- local `qwen3-embedding:0.6b` embeddings with persistent cache

Source of truth: [`runs/enterprise-rag-qwen-core-v1/report.json`](runs/enterprise-rag-qwen-core-v1/report.json).

The report includes the benchmark release/revision, corpus identity, selected question IDs, model identity, configuration, per-arm aggregate metrics, and per-query rankings. This makes the retrieval comparison reproducible and lets regressions be inspected query by query instead of relying on one headline score.

### Why the README changed

An earlier version of this README highlighted a **12-query exploratory subset**. That run was useful while building the pipeline, but the repository now contains the larger frozen 64-query core evaluation above. The resume and README therefore point to the same evidence.

## Service features

- stable document IDs and collection context
- BM25 and local dense retrieval
- RRF fusion
- persistent embedding cache
- optional typed query fields (`lex`, `vec`, `hyde`, `intent`)
- explain mode showing where each result came from
- shared CLI and FastAPI service layer

Search itself works without a generation LLM. Query expansion is optional.

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

## Limits

- The frozen benchmark is a 64-query Confluence-only slice of EnterpriseRAG-Bench, not a leaderboard claim over the entire benchmark.
- Results are specific to the frozen corpus, query contract, and local embedding configuration recorded in the artifact.
- Retrieval metrics measure ranking quality; they do not by themselves prove end-to-end answer quality.

## Development

```bash
python -m pytest -q
python -m compileall -q src evaluation tests
```
