# Evaluation

`inference_queries.json` contains only case IDs and queries. `gold_labels.json` contains relevant document IDs and is read only after retrieval. Runtime search code never receives evaluation labels.

The shipped six-query evaluation is intentionally tiny and demonstrates retrieval behavior. It is not a research benchmark claim.

`run_enterprise_benchmark.py` is the serious, frozen three-arm launcher. It
validates the official EnterpriseRAG assets, exact 12-question metadata-extra Confluence selection (not the core leaderboard benchmark),
corpus qrels, and local Qwen tag/digest before Dense index construction. It
refuses to overwrite an existing report. See
`docs/p2-enterprise-rag-benchmark.md` for the canonical command and scope.

The completed canonical report is
`runs/enterprise-rag-qwen-dev-v1/report.json` (SHA-256
`c9a5772e64e95f1c929ce4a79d9cdb4173cf908fe9654dfec5350320bdfc15ff`).
Its frozen verdict is `KEEP HYBRID`.

Neither runtime path receives gold IDs. The serious runner finishes all three
rankings before computing metrics and paired transitions from the separately
loaded development labels.


Review-v2 does not change retrieval math or the frozen 12-case selection. It writes `runs/enterprise-rag-qwen-dev-v2/report.json` only to make the metadata-extra / non-leaderboard scope explicit in provenance; the validated v1 dense cache may be reused.
