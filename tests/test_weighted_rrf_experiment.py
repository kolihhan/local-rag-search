from evaluation.weighted_rrf_experiment import score_weighted_rankings


def test_weighted_scorer_tracks_rescues_and_regressions_from_full_candidates():
    dataset = {
        "q1": {"gold": {"gold1"}, "bm25": ["gold1", "x", "y"], "dense": ["x", "y", "gold1"]},
        "q2": {"gold": {"gold2"}, "bm25": ["a", "b", "c"], "dense": ["gold2", "x", "y"]},
    }

    scored = score_weighted_rankings(dataset, bm25_weight=1.0, dense_weight=1.0, k=60, cutoff=2)

    assert scored["cases"] == 2
    assert scored["bm25_miss_to_weighted_hit"] == 1
    assert scored["bm25_hit_to_weighted_miss"] == 0
    assert scored["hit_at_2"] == 1.0
