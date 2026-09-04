# Local RAG Search

**A local-first retrieval engine for RAG applications.** It provides BM25,
Qwen dense retrieval, transparent rank fusion, typed demo queries, contextual
documents, and stable document IDs.

```text
Query
  ↓
lex / vec / hyde + intent
  ↓
BM25 + Dense
  ↓
Reciprocal Rank Fusion
  ↓
Optional demo-only bounded rerank
  ↓
Context-aware results + stable doc IDs
```

## The problem

Vector search alone is not enough. Exact identifiers, error codes, and names often need lexical retrieval; paraphrases and conceptual questions benefit from semantic retrieval. A RAG search layer should combine both predictably and make the ranking explainable.

## How it works

- `lex` queries route to BM25.
- `vec` and `hyde` queries route to dense retrieval.
- `intent` is context for planning/explanation, not a retrieval backend.
- Rankings are merged with reciprocal-rank fusion rather than comparing incompatible raw scores.
- Optional reranking only sees a bounded candidate pool.
- Results expose stable document IDs, collection context, and rank provenance.

The system is inspired by QMD's local retrieval and typed-query ideas, but this repository is **not a QMD clone** and does not attempt to reproduce its full feature set.

## Quickstart

CLI demo with the deterministic local demo embedding:

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

## Example

Exact identifier:

```bash
rag-search search "ORA-12516"
```

Vector-only CLI sugar (same core service, no special API route):

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

Explain mode shows BM25 rank, dense rank, RRF rank, reranker score when used, matched typed signals, and collection context.

## Results

The shipped **6-query** deterministic demo evaluation deliberately mixes exact-identifier and paraphrase cases. It is a product sanity check, **not a research benchmark**.

The serious P2 comparison is separately frozen to exactly three arms: existing
BM25, real `qwen3-embedding:0.6b` Dense, and BM25 + Dense RRF. Token-overlap
reranking, HyDE, generation, agents, and a vector database are not treatments.
See [`docs/p2-enterprise-rag-benchmark.md`](docs/p2-enterprise-rag-benchmark.md)
for the frozen protocol, canonical artifact, and final decision.

| Frozen EnterpriseRAG arm | Recall@1 | Recall@5 | Recall@10 | MRR@20 | Mean ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5000 | 0.5833 | 0.7500 | 0.5511 | 60.57 |
| Qwen Dense | 0.7500 | 0.8333 | 0.8333 | 0.7917 | 315.18 |
| BM25 + Dense RRF | 0.7500 | **0.9167** | **0.9167** | **0.7986** | 372.06 |

Frozen P2 verdict: **KEEP HYBRID**, limited to all 12 Confluence rows from the
EnterpriseRAG-Bench v1.0.0 metadata-extra question set; this is not the core leaderboard benchmark. RRF recovered two BM25 top-10 misses with no
BM25 top-10 loss; its latency and cache cost remain material.

| Mode | Recall@10 | MRR@20 |
|---|---:|---:|
| BM25 lexical | 83.3% | 70.0% |
| Demo dense | 100.0% | 70.8% |
| Hybrid RRF | **100.0%** | **77.4%** |

Run it yourself:

```bash
python evaluation/evaluate.py --mode lex
python evaluation/evaluate.py --mode vec
python evaluation/evaluate.py --mode hybrid
```

## Design decisions

- **Search works without an LLM.** Typed queries can be supplied directly; model-based query expansion is optional.
- **Persistent dense cache.** One atomic JSON cache records the full model
  identity, ordered document IDs, corpus identity, dimension, and normalized
  vectors; incompatible inputs rebuild and compatible corruption fails closed.
- **Context-aware documents.** Collection context such as incident/runbook/architecture participates in dense representation and is returned with results.
- **RRF for fusion.** Raw BM25 and cosine scores are never treated as directly comparable.
- **Demo-only bounded rerank.** Token overlap remains available for product
  demonstration, but it is excluded from the serious P2 architecture claim.
- **Thin interfaces.** CLI and FastAPI wrap the same `SearchService`.

See `docs/architecture.md`, `docs/qmd-inspiration.md`, and `docs/design-decisions.md`.

## Limitations

- The shipped deterministic embedding is a demo/test fixture, not evidence for
  semantic retrieval quality. The serious Dense path is the frozen local Qwen
  adapter.
- The six-query evaluation is too small for general retrieval claims.
- Auto LLM query expansion is not required for the MVP; explicit typed queries are the reliable baseline.
- Token-overlap reranking is demo-only and is not a recommended serious arm.

## Development / evaluation

```bash
pytest -q
python -m compileall src evaluation
```

Evaluation queries and gold document IDs are stored in physically separate files; runtime code never receives the gold IDs.
