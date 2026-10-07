# Evaluation

`inference_queries.json` contains only case IDs and queries. `gold_labels.json` contains relevant document IDs and is read only after retrieval. Runtime search code never receives evaluation labels.

The shipped six-query evaluation is intentionally tiny and demonstrates retrieval behavior. It is not a research benchmark claim.

## Current canonical evidence

The resume-facing evidence is the frozen **64-query Confluence core-compatible run** over 5,189 documents:

- report: `runs/enterprise-rag-qwen-core-v1/report.json`
- BM25 Recall@10: `0.7341`
- Qwen Dense Recall@10: `0.6797`
- RRF Recall@10: `0.7630`
- RRF MRR@20: `0.7418`
- RRF Hit@10: `0.8438`
- paired top-10 transition vs BM25: 5 misses recovered, 2 hits lost

This is a frozen Confluence slice, not the full EnterpriseRAG-Bench leaderboard benchmark. The runner completes retrieval rankings before loading qrels for metrics and paired transitions.

## Historical development evidence

`run_enterprise_benchmark.py` originally launched the frozen **12-query metadata-extra Confluence development slice**. Those reports remain useful historical evidence for why hybrid retrieval was kept, but they are no longer the canonical portfolio result:

- `runs/enterprise-rag-qwen-dev-v1/report.json`
- `runs/enterprise-rag-qwen-dev-v2/report.json`

The v2 report only clarified provenance/scope; it did not change retrieval math or the frozen 12-case selection. See `docs/p2-enterprise-rag-benchmark.md` for that historical experiment.

Neither runtime path receives gold IDs. The benchmark runner validates dataset/model/corpus identities and refuses to overwrite an existing report.
