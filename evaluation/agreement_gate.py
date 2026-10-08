from __future__ import annotations

import json
from pathlib import Path


def _metrics(report: dict, selected: dict[str, str]) -> dict[str, float | int]:
    question_ids = report["question_ids"]
    bm25_rows = report["arms"]["bm25"]["per_query"]
    rrf_rows = report["arms"]["rrf"]["per_query"]
    recalls: list[float] = []
    mrrs: list[float] = []
    hits: list[bool] = []
    recoveries = regressions = 0
    fallbacks = 0
    for qid in question_ids:
        arm = selected[qid]
        row = bm25_rows[qid] if arm == "bm25" else rrf_rows[qid]
        recalls.append(float(row["recall_at_10"]))
        mrrs.append(float(row["reciprocal_rank_at_20"]))
        hit = bool(row["hit_at_10"])
        hits.append(hit)
        bm25_hit = bool(bm25_rows[qid]["hit_at_10"])
        recoveries += int(not bm25_hit and hit)
        regressions += int(bm25_hit and not hit)
        fallbacks += int(arm == "bm25")
    n = len(question_ids)
    return {
        "recall_at_10": sum(recalls) / n,
        "mrr_at_20": sum(mrrs) / n,
        "hit_at_10": sum(hits) / n,
        "recoveries": recoveries,
        "regressions": regressions,
        "fallbacks": fallbacks,
    }


def evaluate_threshold(report: dict, *, min_overlap_to_fuse: int) -> dict[str, float | int]:
    bm25_rows = report["arms"]["bm25"]["per_query"]
    dense_rows = report["arms"]["dense"]["per_query"]
    selected: dict[str, str] = {}
    for qid in report["question_ids"]:
        bm25_top10 = set(bm25_rows[qid]["ranking"][:10])
        dense_top10 = set(dense_rows[qid]["ranking"][:10])
        overlap = len(bm25_top10 & dense_top10)
        selected[qid] = "rrf" if overlap >= min_overlap_to_fuse else "bm25"
    return _metrics(report, selected)


def diagnose(dev_path: str | Path, core_path: str | Path) -> dict[str, object]:
    dev = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    core = json.loads(Path(core_path).read_text(encoding="utf-8"))
    candidates = []
    for threshold in range(0, 7):
        candidates.append({"min_overlap_to_fuse": threshold, **evaluate_threshold(dev, min_overlap_to_fuse=threshold)})
    selected = max(
        candidates,
        key=lambda row: (
            row["regressions"] == 0,
            row["recall_at_10"],
            row["mrr_at_20"],
            -row["fallbacks"],
        ),
    )
    threshold = int(selected["min_overlap_to_fuse"])
    return {
        "rule": "use frozen RRF only when BM25/Dense top-10 overlap reaches threshold; otherwise use frozen BM25",
        "selection_set": "enterprise-rag-qwen-dev-v2",
        "selected_threshold": threshold,
        "dev_candidates": candidates,
        "core_exploratory": evaluate_threshold(core, min_overlap_to_fuse=threshold),
    }
