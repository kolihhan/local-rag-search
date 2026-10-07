# Historical P2 EnterpriseRAG metadata-extra Confluence evaluation

> [!IMPORTANT]
> This document records the **historical 12-query metadata-extra development experiment**. It is not the current canonical evidence source. The current portfolio-facing validation is the 64-query Confluence core run in `runs/enterprise-rag-qwen-core-v1/report.json`, summarized in the root README and `evaluation/README.md`.

## Frozen question

Does real semantic or hybrid retrieval earn its complexity over the existing BM25 implementation?

Exactly three serious treatments were frozen before any retrieval result:

1. BM25, top 20.
2. Ollama `qwen3-embedding:0.6b` Dense, top 20.
3. BM25 top 60 + Dense top 60, fused by RRF with `k=60`, top 20.

The earlier SimpleEmbedding/TokenOverlap development report is preserved as historical demo evidence but is superseded for this decision. There is no reranker, HyDE, vector database, generation, or agent treatment in this run.

## Frozen data and contract

The data is Onyx `EnterpriseRAG-Bench` v1.0.0, revision `56ba6a6`:
5,189 unique Confluence documents and all 12 Confluence rows from the metadata-extra question set selected in source row order where `source_types == ["confluence"]` and every qrel resolves. All qrels are singleton. Manifest, gold, exclusions, license, and release-archive hashes are validated before model access.

This slice is legitimate but narrow: Confluence-only, metadata-only, 12/100 extra questions, not the current 64-query core validation, and singleton-gold. The verdict cannot be generalized to other source types or retrieval tasks.

Qwen queries use:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

Documents receive no instruction. The frozen Hugging Face revision is `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; the local Ollama digest is `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`.
Vectors are locally L2-normalized and compared by dot product. Document batch size is 16 and the per-request timeout is 600 seconds.

Each arm runs exactly once per case in deterministic rotated order. Metrics are Recall@1/5/10, MRR@20, and retrieval latency. Paired hit/miss transitions use the top-10 cutoff. Index construction is outside the latency boundary.

## Reproduce historical run

```powershell
uv run python evaluation/run_enterprise_benchmark.py `
  --dataset-root C:\Personal_Projects\ai-projects-data\datasets\EnterpriseRAG-Bench\v1.0.0-dev-confluence `
  --cache runs\enterprise-rag-qwen-dev-v1\dense-cache.json `
  --output runs\enterprise-rag-qwen-dev-v1\report.json
```

The command refuses to overwrite the report, validates the exact local model digest before cache reuse or inference, and writes the report atomically.

## Historical result

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR@20 | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5000 | 0.5833 | 0.7500 | 0.5511 | 60.57 | 59.31 | 82.84 |
| Qwen Dense | 0.7500 | 0.8333 | 0.8333 | 0.7917 | 315.18 | 309.27 | 415.05 |
| BM25 + Dense RRF | 0.7500 | **0.9167** | **0.9167** | **0.7986** | 372.06 | 362.79 | 437.90 |

Paired top-10 transitions against BM25:

- BM25 miss → Dense hit: 1 (`qst_0009`).
- BM25 hit → Dense miss: 0.
- BM25 miss → RRF hit: 2 (`qst_0009`, `qst_0057`).
- BM25 hit → RRF miss: 0.

This experiment produced the original **KEEP HYBRID** development decision. The later 64-query core run is the stronger current validation and supersedes this document for portfolio claims.

Historical artifacts:

- `runs/enterprise-rag-qwen-dev-v1/report.json`: SHA-256 `c9a5772e64e95f1c929ce4a79d9cdb4173cf908fe9654dfec5350320bdfc15ff`.
- `runs/enterprise-rag-qwen-dev-v1/dense-cache.json`: 5,189 × 1,024 vectors, SHA-256 `25b13583c60971d41f51e8906f6ac2b02463e5817ba04a890363e6a8ddef8e21`.

## Historical trade-off

The 12-case result was encouraging but small. RRF mean query latency was about 6.1× BM25 and the MRR@20 gain over Dense was only 0.0069. This was a development signal, not enough by itself for a broad retrieval claim. That limitation is one reason the later 64-query core validation is now the source of truth.
