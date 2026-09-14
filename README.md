# Local RAG Search

A local-first **hybrid retrieval engine for RAG applications**. It combines BM25 and Qwen dense retrieval with reciprocal rank fusion, exposes rank provenance, and serves the same `SearchService` through CLI and FastAPI interfaces.

## Key result

Frozen comparison on all **12 Confluence rows** from the EnterpriseRAG-Bench v1.0.0 metadata-extra question set:

| Retrieval arm | Recall@10 | MRR@20 | Mean ms |
|---|---:|---:|---:|
| BM25 | 0.7500 | 0.5511 | 60.57 |
| Qwen Dense | 0.8333 | 0.7917 | 315.18 |
| BM25 + Dense RRF | **0.9167** | **0.7986** | 372.06 |

**Verdict: `KEEP HYBRID`.** RRF recovered two BM25 top-10 misses with no BM25 top-10 loss in this frozen subset, but the latency and embedding-cache cost remain material. This is **not** the EnterpriseRAG-Bench core leaderboard benchmark.

## Architecture

```text
Query
  ↓
lexical / dense signals
  ↓
BM25 + Qwen Dense
  ↓
Reciprocal Rank Fusion
  ↓
Optional bounded demo rerank
  ↓
Context-aware results + stable doc IDs
```

Why hybrid retrieval: exact identifiers, error codes, and names often need lexical search, while paraphrases and conceptual questions benefit from semantic retrieval. RRF merges ranked lists without pretending BM25 and cosine scores live on the same scale.

## What the system exposes

- BM25 lexical retrieval and Qwen dense retrieval.
- Reciprocal rank fusion with per-result rank provenance.
- Stable document IDs and collection context.
- Persistent dense-vector cache keyed by model identity, ordered document IDs, corpus identity, and embedding dimension.
- Shared `SearchService` behind CLI and FastAPI interfaces.
- Optional bounded reranking for product demonstration; it is excluded from the serious frozen comparison.

Typed query fields (`lex`, `vec`, `hyde`, `intent`) are supported for explicit experimentation. Search itself works without an LLM; model-based query expansion is optional.

This repository is **inspired by QMD's local retrieval and typed-query ideas**, but it is not a QMD clone; the architecture and evaluation here are intentionally narrower.

## Quickstart

CLI demo:

```bash
uv sync
uv run rag-search --corpus demo_docs query "why were buyers unable to finish a purchase?" --explain
```

FastAPI demo:

```bash
uv run uvicorn local_rag.api:demo_app --reload
```

Then open `http://127.0.0.1:8000/docs`.

Use Ollama embeddings when desired:

```bash
ollama pull qwen3-embedding:0.6b
uv run rag-search --embedding ollama --corpus demo_docs query "payment failure"
```

Windows users can use `run-demo.cmd` or `run-api.cmd`.

## Examples

Exact identifier:

```bash
rag-search search "ORA-12516"
```

Vector-only query:

```bash
rag-search vsearch "why were buyers unable to finish a purchase?"
```

Typed hybrid query:

```bash
rag-search query "payment incident" \
  --lex "payment authorization" \
  --vec "users could not finish checkout" \
  --hyde "Payment checkout failed because authorization was rejected" \
  --explain
```

Explain mode shows BM25 rank, dense rank, RRF rank, optional reranker score, matched typed signals, and collection context.

## Evaluation

The serious P2 comparison is frozen to exactly three arms: BM25, real `qwen3-embedding:0.6b` Dense, and BM25 + Dense RRF. Token-overlap reranking, HyDE, generation, agents, and a vector database are not treatments.

See [`docs/p2-enterprise-rag-benchmark.md`](docs/p2-enterprise-rag-benchmark.md) for the frozen protocol, canonical artifact, and final decision.

The repository also ships a **6-query deterministic demo evaluation** mixing exact-identifier and paraphrase cases. It is a product sanity check, not research evidence:

| Demo mode | Recall@10 | MRR@20 |
|---|---:|---:|
| BM25 lexical | 83.3% | 70.0% |
| Demo dense | 100.0% | 70.8% |
| Hybrid RRF | **100.0%** | **77.4%** |

Run it with:

```bash
python evaluation/evaluate.py --mode lex
python evaluation/evaluate.py --mode vec
python evaluation/evaluate.py --mode hybrid
```

Evaluation queries and gold document IDs are stored in physically separate files; runtime code never receives gold IDs.

## Design decisions

- **Search works without an LLM.** Explicit typed queries are a reliable baseline; model-based expansion is optional.
- **Persistent dense cache.** Compatible cache entries are reused; incompatible inputs rebuild and compatible corruption fails closed.
- **Context-aware documents.** Collection context such as incident/runbook/architecture participates in dense representation and is returned with results.
- **RRF for fusion.** Raw BM25 and cosine scores are never treated as directly comparable.
- **Thin interfaces.** CLI and FastAPI wrap the same `SearchService`.

See `docs/architecture.md`, `docs/qmd-inspiration.md`, and `docs/design-decisions.md`.

## Limitations

- The frozen 12-query subset is too small for broad retrieval-quality claims.
- The shipped deterministic embedding is a demo/test fixture; the serious Dense path is the frozen local Qwen adapter.
- Auto LLM query expansion is not required for the MVP.
- Token-overlap reranking is demo-only and is not a recommended serious arm.

## Development

```bash
python -m pytest -q
python -m compileall -q src evaluation tests
```
