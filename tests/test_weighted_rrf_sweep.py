from pathlib import Path

from evaluation.weighted_rrf_sweep import score_weighted_rrf_report, sweep_report


def test_score_weighted_rrf_report_tracks_rescues_and_regressions():
    report = {
        "question_ids": ["q1", "q2"],
        "arms": {
            "bm25": {"per_query": {
                "q1": {"gold_ranks": {"gold1": 1}, "ranking": ["gold1", "x"]},
                "q2": {"gold_ranks": {"gold2": None}, "ranking": ["a"]},
            }},
            "dense": {"per_query": {
                "q1": {"gold_ranks": {"gold1": 2}, "ranking": ["x", "gold1"]},
                "q2": {"gold_ranks": {"gold2": 1}, "ranking": ["gold2"]},
            }},
        },
    }

    scored = score_weighted_rrf_report(report, bm25_weight=1.5, dense_weight=1.0, k=60, cutoff=2)

    assert scored["cases"] == 2
    assert scored["bm25_miss_to_weighted_hit"] == 1
    assert scored["bm25_hit_to_weighted_miss"] == 0
    assert scored["hit_at_2"] == 1.0


def test_diagnostic_surface_frozen_weight_sweep():
    rows = sweep_report(Path("runs/enterprise-rag-qwen-core-v1/report.json"))
    raise AssertionError(rows)
