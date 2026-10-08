import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WEIGHTS = (1.0, 1.25, 1.5, 2.0)


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _qrels(report: dict, qid: str) -> set[str]:
    return set(report["arms"]["bm25"]["per_query"][qid]["gold_ranks"])


def _fuse(report: dict, qid: str, bm25_weight: float, *, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for source, weight in (("bm25", bm25_weight), ("dense", 1.0)):
        ranking = report["arms"][source]["per_query"][qid]["ranking"]
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _metrics(report: dict, weight: float) -> dict[str, float | int]:
    qids = report["question_ids"]
    recalls, reciprocal_ranks, hits = [], [], []
    rescued = regressed = 0
    for qid in qids:
        relevant = _qrels(report, qid)
        ranking = _fuse(report, qid, weight)
        top10 = set(ranking[:10])
        recalls.append(len(relevant & top10) / len(relevant))
        first = next((rank for rank, doc_id in enumerate(ranking[:20], 1) if doc_id in relevant), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        hit = bool(relevant & top10)
        hits.append(hit)
        bm25_hit = bool(relevant & set(report["arms"]["bm25"]["per_query"][qid]["ranking"][:10]))
        rescued += int(not bm25_hit and hit)
        regressed += int(bm25_hit and not hit)
    n = len(qids)
    return {
        "recall10": sum(recalls) / n,
        "mrr20": sum(reciprocal_ranks) / n,
        "hit10": sum(hits) / n,
        "rescued": rescued,
        "regressed": regressed,
    }


def _select_on_dev(report: dict) -> tuple[float, dict[float, dict]]:
    rows = {weight: _metrics(report, weight) for weight in WEIGHTS}
    chosen = max(
        WEIGHTS,
        key=lambda weight: (
            rows[weight]["recall10"],
            rows[weight]["mrr20"],
            -rows[weight]["regressed"],
            -weight,
        ),
    )
    return chosen, rows


def test_diagnose_weighted_rrf_without_tuning_on_core():
    dev = _load("runs/enterprise-rag-qwen-dev-v2/report.json")
    core = _load("runs/enterprise-rag-qwen-core-v1/report.json")
    chosen, dev_rows = _select_on_dev(dev)
    core_equal = _metrics(core, 1.0)
    core_chosen = _metrics(core, chosen)
    pytest.fail(json.dumps({
        "chosen_bm25_weight": chosen,
        "dev": dev_rows,
        "core_equal": core_equal,
        "core_chosen": core_chosen,
    }, sort_keys=True))
