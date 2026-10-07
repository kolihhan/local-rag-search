from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .dense import DenseIndex, EmbeddingProvider
from .documents import Document
from .fusion import reciprocal_rank_fusion
from .query import QueryPlan
from .rerank import Reranker, TokenOverlapReranker


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    path: str
    collection: str
    context: str
    snippet: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_rank: int | None = None
    reranker_score: float | None = None
    matched_signals: tuple[str, ...] = field(default_factory=tuple)


class SearchService:
    def __init__(
        self,
        documents: list[Document],
        bm25: BM25Index,
        dense: DenseIndex,
        *,
        reranker: Reranker | None = None,
        rerank_pool: int = 30,
    ) -> None:
        self.documents = {doc.doc_id: doc for doc in documents}
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker or TokenOverlapReranker()
        self.rerank_pool = max(1, rerank_pool)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        *,
        embedding_provider: EmbeddingProvider,
        cache_path: str | Path,
        reranker: Reranker | None = None,
        rerank_pool: int = 30,
    ) -> "SearchService":
        return cls(
            documents,
            BM25Index(documents),
            DenseIndex.build(documents, embedding_provider, cache_path=cache_path),
            reranker=reranker,
            rerank_pool=rerank_pool,
        )

    def get(self, doc_id: str) -> Document:
        return self.documents[doc_id]

    def get_many(self, doc_ids: list[str]) -> list[Document]:
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        limit: int = 10,
        rerank: bool = False,
    ) -> list[SearchResult]:
        if mode == "lex":
            plan = QueryPlan.lexical(query)
        elif mode == "vec":
            plan = QueryPlan.vector(query)
        elif mode == "hybrid":
            plan = QueryPlan.hybrid(query)
        else:
            raise ValueError(f"unknown search mode: {mode}")
        return self.query(plan, limit=limit, rerank=rerank)

    def query(
        self,
        plan: QueryPlan,
        *,
        limit: int = 10,
        rerank: bool = False,
    ) -> list[SearchResult]:
        candidate_limit = max(limit, self.rerank_pool if rerank else limit * 3)
        rankings: dict[str, list[tuple[str, float]]] = {}

        for index, query in enumerate(plan.lex):
            source = "lex" if len(plan.lex) == 1 else f"lex#{index + 1}"
            rankings[source] = [(hit.doc_id, hit.score) for hit in self.bm25.search(query, limit=candidate_limit)]
        for signal_name, queries in (("vec", plan.vec), ("hyde", plan.hyde)):
            for index, query in enumerate(queries):
                source = signal_name if len(queries) == 1 else f"{signal_name}#{index + 1}"
                rankings[source] = [(hit.doc_id, hit.score) for hit in self.dense.search(query, limit=candidate_limit)]

        if not rankings:
            rankings["lex"] = [(hit.doc_id, hit.score) for hit in self.bm25.search(plan.original, limit=candidate_limit)]
            rankings["vec"] = [(hit.doc_id, hit.score) for hit in self.dense.search(plan.original, limit=candidate_limit)]

        fused = reciprocal_rank_fusion(rankings)
        rrf_rank = {row.doc_id: index for index, row in enumerate(fused, start=1)}
        fused_by_id = {row.doc_id: row for row in fused}

        rerank_scores: dict[str, float] = {}
        ordered_ids = [row.doc_id for row in fused]
        if rerank and ordered_ids:
            pool_ids = ordered_ids[: self.rerank_pool]
            reranked = self.reranker.rerank(plan.original or (plan.intent or " ".join(plan.lex + plan.vec + plan.hyde)), self.get_many(pool_ids))
            rerank_scores = {doc_id: score for doc_id, score in reranked}
            ordered_ids = [doc_id for doc_id, _ in reranked] + [doc_id for doc_id in ordered_ids if doc_id not in rerank_scores]

        results: list[SearchResult] = []
        for doc_id in ordered_ids[:limit]:
            doc = self.documents[doc_id]
            row = fused_by_id[doc_id]
            lex_ranks = [rank for source, rank in row.source_ranks.items() if source.startswith("lex")]
            dense_ranks = [rank for source, rank in row.source_ranks.items() if source.startswith(("vec", "hyde"))]
            signals = sorted({source.split("#", 1)[0] for source in row.source_ranks})
            score = rerank_scores.get(doc_id, row.score)
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    path=doc.path,
                    collection=doc.collection,
                    context=doc.context,
                    snippet=" ".join(doc.text.split())[:280],
                    score=score,
                    bm25_rank=min(lex_ranks) if lex_ranks else None,
                    dense_rank=min(dense_ranks) if dense_ranks else None,
                    rrf_rank=rrf_rank[doc_id],
                    reranker_score=rerank_scores.get(doc_id),
                    matched_signals=tuple(signals),
                )
            )
        return results
