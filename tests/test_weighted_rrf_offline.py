import json
from pathlib import Path

from evaluation.weighted_rrf import diagnose_from_reports, fuse_rankings, score_rankings

ROOT = Path(__file__).parents[1]


def test_weighted_rrf_prefers_bm25_when_weight_is_higher():
    fused = fuse_rankings(
        bm25=["gold", "a", "b"],
        dense=["a", "b", "gold"],
        bm25_weight=2.0,
        dense_weight=1.0,
        k=60,
    )
    assert fused[0] == "gold"


def test_score_rankings_tracks_top10_regressions_and_recoveries():
    rows = {
        "q1": {"gold": {"gold"}, "bm25": ["gold"], "candidate": ["gold"]},
        "q2": {"gold": {"gold"}, "bm25": ["x"] * 10 + ["gold"], "candidate": ["gold"]},
        "q3": {"gold": {"gold"}, "bm25": ["gold"], "candidate": ["x"] * 10 + ["gold"]},
    }
    metrics = score_rankings(rows)
    assert metrics["recoveries"] == 1
    assert metrics["regressions"] == 1


def test_frozen_dev_selects_weight_before_core_evaluation():
    result = diagnose_from_reports(
        ROOT / "runs" / "enterprise-rag-qwen-dev-v2" / "report.json",
        ROOT / "runs" / "enterprise-rag-qwen-core-v1" / "report.json",
    )
    assert False, "WEIGHTED_RRF_DIAGNOSTIC=" + json.dumps(result, sort_keys=True)
