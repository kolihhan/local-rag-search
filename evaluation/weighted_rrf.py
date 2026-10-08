from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def fuse_rankings(
    *,
    bm25: Sequence[str],
    dense: Sequence[str],
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (k + rank)
    for rank, doc_id in enumerate(dense, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _query_metrics(ranking: Sequence[str], gold: set[str]) -> tuple[float, float, bool]:
    top10 = set(ranking[:10])
    recall10 = len(top10 & gold) / len(gold)
    first = next((index for index, doc_id in enumerate(ranking[:20], 1) if doc_id in gold), None)
    mrr20 = 1.0 / first if first is not None else 0.0
    return recall10, mrr20, bool(top10 & gold)


def score_rankings(rows: Mapping[str, Mapping[str, object]]) -> dict[str, float | int]:
    recalls: list[float] = []
    mrrs: list[float] = []
    hits: list[bool] = []
    recoveries = 0
    regressions = 0
    for row in rows.values():
        gold = set(row["gold"])
        bm25 = list(row["bm25"])
        candidate = list(row["candidate"])
        recall, mrr, hit = _query_metrics(candidate, gold)
        recalls.append(recall)
        mrrs.append(mrr)
        hits.append(hit)
        _, _, bm25_hit = _query_metrics(bm25, gold)
        recoveries += int(not bm25_hit and hit)
        regressions += int(bm25_hit and not hit)
    n = len(recalls)
    return {
        "recall_at_10": sum(recalls) / n,
        "mrr_at_20": sum(mrrs) / n,
        "hit_at_10": sum(hits) / n,
        "recoveries": recoveries,
        "regressions": regressions,
    }


def rows_from_report(report: Mapping[str, object], *, bm25_weight: float, dense_weight: float = 1.0) -> dict[str, dict[str, object]]:
    arms = report["arms"]
    bm25_rows = arms["bm25"]["per_query"]
    dense_rows = arms["dense"]["per_query"]
    rows: dict[str, dict[str, object]] = {}
    for qid in report["question_ids"]:
        bm25 = list(bm25_rows[qid]["ranking"])
        dense = list(dense_rows[qid]["ranking"])
        gold = set(bm25_rows[qid]["gold_ranks"])
        rows[qid] = {
            "gold": gold,
            "bm25": bm25,
            "candidate": fuse_rankings(
                bm25=bm25,
                dense=dense,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )[:20],
        }
    return rows


def diagnose_from_reports(
    dev_path: str | Path,
    core_path: str | Path,
    *,
    weights: Iterable[float] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
) -> dict[str, object]:
    dev = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    core = json.loads(Path(core_path).read_text(encoding="utf-8"))
    candidates = []
    for weight in weights:
        metrics = score_rankings(rows_from_report(dev, bm25_weight=weight))
        candidates.append({"bm25_weight": weight, **metrics})
    selected = max(
        candidates,
        key=lambda row: (
            row["regressions"] == 0,
            row["recall_at_10"],
            row["mrr_at_20"],
            -row["regressions"],
            -row["bm25_weight"],
        ),
    )
    core_metrics = score_rankings(rows_from_report(core, bm25_weight=float(selected["bm25_weight"])))
    return {
        "selection_set": "enterprise-rag-qwen-dev-v2 (12-query historical development set)",
        "evaluation_set": "enterprise-rag-qwen-core-v1 (64-query frozen core)",
        "dev_candidates": candidates,
        "selected_bm25_weight": selected["bm25_weight"],
        "core": core_metrics,
    }
