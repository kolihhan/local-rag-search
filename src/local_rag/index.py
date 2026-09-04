from __future__ import annotations

import json
from pathlib import Path

from .dense import OllamaEmbeddingProvider, SimpleEmbeddingProvider
from .documents import load_markdown_corpus
from .search import SearchService


def load_contexts(corpus_root: str | Path) -> dict[str, str]:
    path = Path(corpus_root) / "contexts.json"
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def build_search_service(
    corpus_root: str | Path,
    *,
    cache_dir: str | Path,
    embedding: str = "simple",
    embedding_model: str = "qwen3-embedding:0.6b",
    rerank_pool: int = 30,
) -> SearchService:
    corpus_root = Path(corpus_root)
    docs = load_markdown_corpus(corpus_root, load_contexts(corpus_root))
    provider = SimpleEmbeddingProvider() if embedding == "simple" else OllamaEmbeddingProvider(model=embedding_model)
    cache_path = Path(cache_dir) / f"dense-{embedding}.json"
    return SearchService.from_documents(docs, embedding_provider=provider, cache_path=cache_path, rerank_pool=rerank_pool)
