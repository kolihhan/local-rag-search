# Evaluation

`inference_queries.json` contains only case IDs and queries. `gold_labels.json` contains relevant document IDs and is read only after retrieval. Runtime search code never receives evaluation labels.

The shipped six-query evaluation is intentionally tiny and demonstrates retrieval behavior. It is not a research benchmark claim.

## Current serious evidence

`run_enterprise_benchmark.py` is the serious frozen three-arm launcher. The current portfolio-facing source of truth is:

`runs/enterprise-rag-qwen-core-v1/report.json`

That run covers all **64 Confluence-only, qrel-compatible core rows** used by the current protocol over **5,189 documents**. It validates the EnterpriseRAG assets, corpus/qrels, frozen question selection, and local Qwen tag/digest before Dense index construction. The runner completes all three rankings before loading the separate gold labels for metrics and paired transitions.

Current frozen metrics:

| Arm | Recall@10 | MRR@20 | Hit@10 |
|---|---:|---:|---:|
| BM25 | 0.7341 | 0.6792 | 0.7969 |
| Qwen Dense | 0.6797 | 0.6919 | 0.7656 |
| BM25 + Dense RRF | **0.7630** | **0.7418** | **0.8438** |

RRF recovered five BM25 top-10 misses and lost two BM25 top-10 hits on this slice. This supports keeping the hybrid path here; it is not a universal retrieval claim and it is not the full EnterpriseRAG-Bench leaderboard benchmark.

## Historical development evidence

The earlier 12-query metadata-extra Confluence run remains useful historical development evidence:

`runs/enterprise-rag-qwen-dev-v1/report.json`

It was the first serious Qwen three-arm experiment and produced the original `KEEP HYBRID` decision, but it is **not the current canonical evidence source**. `docs/p2-enterprise-rag-benchmark.md` now labels that 12-query experiment as historical rather than mixing it with the 64-query core validation.

Neither runtime path receives gold IDs. Serious reports refuse overwrite and preserve model/corpus/protocol identities for repeatability.
