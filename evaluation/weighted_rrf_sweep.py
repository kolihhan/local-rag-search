from __future__ import annotations

import json
from pathlib import Path

from local_rag.fusion import reciprocal_rank_fusion


def score_weighted_rrf_report(
    report: dict, *, bm25_weight: float, dense_weight: float, k: int = 60, cutoff: int = 10,
) -> dict[str, float | int]:
    question_ids = report["question_ids"]
    bm25_rows = report["arms"]["bm25"]["per_query"]
    dense_rows = report["arms"]["dense"]["per_query"]
    hits = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    rescues = 0
    regressions = 0

    for question_id in question_ids:
        bm25 = bm25_rows[question_id]
        dense = dense_rows[question_id]
        gold = set(bm25["gold_ranks"]) | set(dense["gold_ranks"])
        rankings = {
            "bm25": [(doc_id, 0.0) for doc_id in bm25["ranking"]],
            "dense": [(doc_id, 0.0) for doc_id in dense["ranking"]],
        }
        fused = reciprocal_rank_fusion(
            rankings,
            k=k,
            weights={"bm25": bm25_weight, "dense": dense_weight},
        )
        ranked = [row.doc_id for row in fused]
        top = ranked[:cutoff]
        hit = bool(gold & set(top))
        bm25_hit = bool(gold & set(bm25["ranking"][:cutoff]))
        hits += int(hit)
        recall_sum += len(gold & set(top)) / len(gold)
        first = min((rank for rank, doc_id in enumerate(ranked[:20], start=1) if doc_id in gold), default=None)
        reciprocal_rank_sum += 1.0 / first if first is not None else 0.0
        rescues += int(not bm25_hit and hit)
        regressions += int(bm25_hit and not hit)

    cases = len(question_ids)
    return {
        "cases": cases,
        f"hit_at_{cutoff}": hits / cases,
        f"recall_at_{cutoff}": recall_sum / cases,
        "mrr_at_20": reciprocal_rank_sum / cases,
        "bm25_miss_to_weighted_hit": rescues,
        "bm25_hit_to_weighted_miss": regressions,
    }


def sweep_report(path: str | Path) -> list[dict]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for bm25_weight in (1.0, 1.1, 1.25, 1.5, 2.0):
        scored = score_weighted_rrf_report(
            report, bm25_weight=bm25_weight, dense_weight=1.0, k=60, cutoff=10,
        )
        rows.append({"bm25_weight": bm25_weight, "dense_weight": 1.0, **scored})
    return rows
