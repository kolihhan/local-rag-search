from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .documents import Document

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = {doc.doc_id: doc for doc in documents}
        self.k1 = k1
        self.b = b
        self._tokens = {doc.doc_id: tokenize(f"{doc.title}\n{doc.text}\n{doc.context}") for doc in documents}
        self._term_freqs = {doc_id: Counter(tokens) for doc_id, tokens in self._tokens.items()}
        self._doc_lens = {doc_id: len(tokens) for doc_id, tokens in self._tokens.items()}
        self._avg_len = (sum(self._doc_lens.values()) / len(self._doc_lens)) if self._doc_lens else 0.0
        df: Counter[str] = Counter()
        for tokens in self._tokens.values():
            df.update(set(tokens))
        self._df = df
        self._n = len(documents)

    def _idf(self, term: str) -> float:
        n_q = self._df.get(term, 0)
        return math.log(1 + (self._n - n_q + 0.5) / (n_q + 0.5)) if self._n else 0.0

    def search(self, query: str, *, limit: int = 10) -> list[RankedDocument]:
        terms = tokenize(query)
        scored: list[RankedDocument] = []
        for doc_id, freqs in self._term_freqs.items():
            score = 0.0
            doc_len = self._doc_lens[doc_id]
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_len if self._avg_len else 0.0))
                score += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(RankedDocument(doc_id, score))
        scored.sort(key=lambda item: (-item.score, item.doc_id))
        return scored[:limit]
