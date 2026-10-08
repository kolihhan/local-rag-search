from evaluation.weighted_rrf import fuse_rankings, score_rankings


def test_weighted_rrf_prefers_bm25_when_weight_is_higher():
    fused = fuse_rankings(
        bm25=["gold", "a", "b"],
        dense=["a", "b", "gold"],
        bm25_weight=1.5,
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
