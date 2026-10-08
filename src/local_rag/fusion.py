from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedDocument:
    doc_id: str
    score: float
    source_ranks: dict[str, int]


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedDocument]:
    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source, rows in rankings.items():
        weight = weights.get(source, 1.0) if weights is not None else 1.0
        for rank, (doc_id, _raw_score) in enumerate(rows, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            source_ranks.setdefault(doc_id, {})[source] = rank
    fused = [FusedDocument(doc_id, score, source_ranks[doc_id]) for doc_id, score in scores.items()]
    fused.sort(key=lambda row: (-row.score, row.doc_id))
    return fused
