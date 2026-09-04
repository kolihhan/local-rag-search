from __future__ import annotations

from typing import Protocol

from .bm25 import tokenize
from .documents import Document


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[Document]) -> list[tuple[str, float]]: ...


class TokenOverlapReranker:
    """Deterministic lightweight reranker for the offline demo.

    A production deployment can replace this with a cross-encoder without
    changing SearchService.
    """

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[str, float]]:
        q = set(tokenize(query))
        rows: list[tuple[str, float]] = []
        for doc in documents:
            d = set(tokenize(f"{doc.title} {doc.text} {doc.context}"))
            score = len(q & d) / max(1, len(q | d))
            rows.append((doc.doc_id, score))
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows
