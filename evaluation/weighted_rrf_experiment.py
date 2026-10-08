from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from evaluation.evaluate import load_core_confluence_set, load_enterprise_rag_txt_corpus
from local_rag.bm25 import BM25Index
from local_rag.dense import DenseIndex, OllamaEmbeddingProvider


def _weighted_fuse(bm25: list[str], dense: list[str], *, bm25_weight: float, dense_weight: float, k: int) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (k + rank)
    for rank, doc_id in enumerate(dense, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))]


def score_weighted_rankings(
    dataset: dict[str, dict[str, object]], *, bm25_weight: float, dense_weight: float,
    k: int = 60, cutoff: int = 10,
) -> dict[str, object]:
    per_query: dict[str, dict[str, object]] = {}
    recalls = []
    reciprocal_ranks = []
    hits = []
    rescues: list[str] = []
    regressions: list[str] = []

    for question_id, row in dataset.items():
        gold = set(row["gold"])
        bm25 = list(row["bm25"])
        dense = list(row["dense"])
        ranked = _weighted_fuse(
            bm25, dense, bm25_weight=bm25_weight, dense_weight=dense_weight, k=k,
        )
        top = ranked[:cutoff]
        recall = len(gold & set(top)) / len(gold)
        positions = [index for index, doc_id in enumerate(ranked[:20], start=1) if doc_id in gold]
        reciprocal_rank = 1.0 / min(positions) if positions else 0.0
        hit = bool(gold & set(top))
        bm25_hit = bool(gold & set(bm25[:cutoff]))
        if not bm25_hit and hit:
            rescues.append(question_id)
        elif bm25_hit and not hit:
            regressions.append(question_id)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        hits.append(hit)
        per_query[question_id] = {
            "recall_at_10": recall,
            "reciprocal_rank_at_20": reciprocal_rank,
            "hit_at_10": hit,
            "first_relevant_rank_at_20": min(positions) if positions else None,
            "ranking": ranked[:20],
        }

    count = len(dataset)
    return {
        "cases": count,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "rrf_k": k,
        "recall_at_10": sum(recalls) / count,
        "mrr_at_20": sum(reciprocal_ranks) / count,
        "hit_at_10": sum(hits) / count,
        "bm25_miss_to_weighted_hit": len(rescues),
        "bm25_hit_to_weighted_miss": len(regressions),
        "rescued_case_ids": rescues,
        "regressed_case_ids": regressions,
        "per_query": per_query,
    }


def _assert_baseline_reproduces_canonical(candidate: dict[str, object], canonical: dict) -> None:
    expected = canonical["arms"]["rrf"]
    for key in ("recall_at_10", "mrr_at_20", "hit_at_10"):
        if abs(float(candidate[key]) - float(expected[key])) > 1e-12:
            raise ValueError(
                f"equal-weight baseline does not reproduce canonical {key}: "
                f"{candidate[key]} != {expected[key]}"
            )
    transition = canonical["transition_summary"]
    if int(candidate["bm25_miss_to_weighted_hit"]) != int(transition["bm25_miss_to_rrf_hit"]["count"]):
        raise ValueError("equal-weight baseline does not reproduce canonical rescue count")
    if int(candidate["bm25_hit_to_weighted_miss"]) != int(transition["bm25_hit_to_rrf_miss"]["count"]):
        raise ValueError("equal-weight baseline does not reproduce canonical regression count")


def run_experiment(
    *, corpus_roots: Iterable[Path], questions_path: Path, cache_path: Path,
    canonical_report_path: Path, output_path: Path, base_url: str,
    model: str = "qwen3-embedding:0.6b", batch_size: int = 16,
) -> dict:
    documents = list(load_enterprise_rag_txt_corpus(roots=list(corpus_roots)))
    if len(documents) != 5189:
        raise ValueError(f"expected 5189 frozen documents, got {len(documents)}")
    dataset = load_core_confluence_set(
        questions_path=questions_path,
        corpus_doc_ids={document.doc_id for document in documents},
    )
    provider = OllamaEmbeddingProvider(
        model=model, base_url=base_url, timeout_s=600.0, batch_size=batch_size,
    )
    bm25 = BM25Index(documents)
    dense = DenseIndex.build(documents, provider, cache_path=cache_path)

    rankings: dict[str, dict[str, object]] = {}
    for question_id in dataset.question_ids:
        query = dataset.questions[question_id]
        rankings[question_id] = {
            "gold": set(dataset.qrels[question_id]),
            "bm25": [hit.doc_id for hit in bm25.search(query, limit=60)],
            "dense": [hit.doc_id for hit in dense.search(query, limit=60)],
        }

    weights = (1.0, 1.05, 1.10, 1.15, 1.25, 1.50)
    rows = [
        score_weighted_rankings(
            rankings, bm25_weight=weight, dense_weight=1.0, k=60, cutoff=10,
        )
        for weight in weights
    ]
    canonical = json.loads(canonical_report_path.read_text(encoding="utf-8"))
    _assert_baseline_reproduces_canonical(rows[0], canonical)

    baseline = rows[0]
    eligible = [
        row for row in rows[1:]
        if row["bm25_hit_to_weighted_miss"] == 0
        and row["recall_at_10"] >= baseline["recall_at_10"]
        and row["mrr_at_20"] >= baseline["mrr_at_20"]
        and (
            row["recall_at_10"] > baseline["recall_at_10"]
            or row["mrr_at_20"] > baseline["mrr_at_20"]
        )
    ]
    winner = max(
        eligible,
        key=lambda row: (row["recall_at_10"], row["mrr_at_20"], -row["bm25_weight"]),
        default=None,
    )
    payload = {
        "experiment": "enterprise_core_weighted_rrf_top60_v1",
        "scope": "posthoc diagnostic on the same frozen 64-query core set",
        "model_identity": provider.cache_identity,
        "corpus_documents": len(documents),
        "candidate_depth": 60,
        "weights_tested": list(weights),
        "baseline_reproduced": True,
        "baseline": baseline,
        "candidates": rows[1:],
        "winner": winner,
        "promotion_status": "diagnostic_only_same_eval_set" if winner is not None else "no_candidate_passed_gate",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, action="append", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--canonical-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    payload = run_experiment(
        corpus_roots=args.corpus_root,
        questions_path=args.questions,
        cache_path=args.cache,
        canonical_report_path=args.canonical_report,
        output_path=args.output,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: value for key, value in payload.items() if key not in {"baseline", "candidates"}}, indent=2))
    if payload["winner"]:
        print(json.dumps(payload["winner"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
