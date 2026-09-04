# P2 EnterpriseRAG metadata-extra Confluence evaluation

## Frozen question

Does real semantic or hybrid retrieval earn its complexity over the existing
BM25 implementation?

Exactly three serious treatments were frozen before any retrieval result:

1. BM25, top 20.
2. Ollama `qwen3-embedding:0.6b` Dense, top 20.
3. BM25 top 60 + Dense top 60, fused by RRF with `k=60`, top 20.

The earlier SimpleEmbedding/TokenOverlap development report is preserved as
historical demo evidence but is superseded for this decision. There is no
reranker, HyDE, vector database, generation, or agent treatment in this run.

## Frozen data and contract

The data is official Onyx `EnterpriseRAG-Bench` v1.0.0, revision `56ba6a6`:
5,189 unique Confluence documents and all 12 Confluence rows from the official metadata-extra question set selected
in official row order where `source_types == ["confluence"]` and every qrel
resolves. All qrels are singleton. Manifest, gold, exclusions, license, and
release-archive hashes are validated before model access.

This slice is legitimate but narrow: Confluence-only, metadata-only, 12/100 extra questions, and it is not the core leaderboard benchmark
available questions, and singleton-gold. The verdict cannot be generalized to
other source types or retrieval tasks.

Qwen queries use:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

Documents receive no instruction. The frozen Hugging Face revision is
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; the local Ollama digest is
`ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`.
Vectors are locally L2-normalized and compared by dot product. Document batch
size is 16 and the per-request timeout is 600 seconds.

Each arm runs exactly once per case in deterministic rotated order. Metrics are
Recall@1/5/10, MRR@20, and retrieval latency. Paired hit/miss transitions use
the top-10 cutoff. Index construction is outside the latency boundary.

## Reproduce

```powershell
uv run python evaluation/run_enterprise_benchmark.py `
  --dataset-root C:\Personal_Projects\ai-projects-data\datasets\EnterpriseRAG-Bench\v1.0.0-dev-confluence `
  --cache runs\enterprise-rag-qwen-dev-v1\dense-cache.json `
  --output runs\enterprise-rag-qwen-dev-v1\report.json
```

The command refuses to overwrite the report, validates the exact local model
digest before cache reuse or inference, and writes the report atomically.

## Result

The single canonical run completed successfully. No real Dense retrieval result
was used to choose the cases, model contract, batching, cutoffs, or verdict
rule.

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

Dense and RRF both earned existence over BM25 under the frozen rule. RRF also
improved Recall@10 and MRR@20 over Dense without a top-10 regression, so the
frozen verdict is **KEEP HYBRID**.

Canonical artifacts:

- `runs/enterprise-rag-qwen-dev-v1/report.json`: 62,269 bytes, SHA-256
  `c9a5772e64e95f1c929ce4a79d9cdb4173cf908fe9654dfec5350320bdfc15ff`.
- `runs/enterprise-rag-qwen-dev-v1/dense-cache.json`: 112,712,005 bytes,
  5,189 × 1,024 vectors, SHA-256
  `25b13583c60971d41f51e8906f6ac2b02463e5817ba04a890363e6a8ddef8e21`.

Integrity checks passed: complete 12/12, exact three arms, unique top-20
rankings, exact paired-transition schema, cache count/dimension/hash, and all
recorded implementation hashes.

## Steel-man against the verdict

The hybrid gain is only one top-10 case over Dense and two over BM25 on a
12-case metadata-only slice. RRF mean query latency was about 6.1× BM25 and
1.18× Dense, while index construction took about twenty minutes and produced a
112.7 MB JSON cache. The MRR@20 gain over Dense was only 0.0069. A
latency-sensitive deployment or a broader lexical workload could still prefer
BM25, and this run provides no evidence for paraphrase, multi-gold, or
non-Confluence tasks. `KEEP HYBRID` therefore means retain the measured serious
path, not make a universal quality claim or remove the cheap BM25-only option.

## Frozen verdict rule

Recall@10 is primary and MRR@20 secondary. A more complex arm must have no
top-10 regression, be no worse on either metric, and be strictly better on at
least one. Exact ties favor simplicity; conflicting or incomplete evidence is
`INSUFFICIENT`. Valid verdicts are `KEEP BM25`, `KEEP DENSE`, `KEEP HYBRID`, and
`INSUFFICIENT`.
