from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol
import urllib.request

from .documents import Document

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
_CACHE_SCHEMA = "local-rag-dense-cache/v3"
_OLLAMA_MODEL_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
_QWEN_HF_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_QWEN_CANONICAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_QWEN_DIMENSION = 1024
_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def embed_query(self, query: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _normalise_vectors(vectors: object, expected_count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise ValueError(f"embedding response count mismatch: expected {expected_count}")
    result: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("embedding vectors must be non-empty lists")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vectors must contain numbers") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding vectors must contain finite values")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise ValueError("embedding vectors must have a consistent dimension")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError("embedding vectors must have a nonzero norm")
        result.append([value / norm for value in values])
    return result


class SimpleEmbeddingProvider:
    """Deterministic semantic-ish fixture for the offline demo and tests."""

    _ALIASES = {
        "checkout": "payment", "payments": "payment", "authorization": "payment", "authorisation": "payment",
        "purchase": "payment", "purchasing": "payment", "buy": "payment", "card": "payment",
        "outage": "failure", "failed": "failure", "fails": "failure", "unavailable": "failure", "unable": "failure",
        "down": "failure", "rejected": "failure", "denial": "failure",
        "db": "database", "connections": "database", "connection": "database", "sessions": "database", "session": "database", "pool": "database",
        "slow": "latency", "sluggish": "latency", "delay": "latency", "delays": "latency", "timeouts": "timeout", "timed": "timeout",
        "release": "deployment", "deploy": "deployment",
        "restore": "recovery", "recover": "recovery", "playbook": "procedure", "runbook": "procedure",
        "credentials": "token", "tokens": "token", "check": "validate", "validated": "validate", "validation": "validate",
        "users": "user", "buyers": "user", "shoppers": "user", "customers": "user", "customer": "user",
    }

    def __init__(self, dimensions: int = 96) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    @property
    def identity(self) -> str:
        return f"simple-semantic-v2:{self.dimensions}"

    @property
    def cache_identity(self) -> dict[str, object]:
        return {"provider": "simple-demo", "identity": self.identity, "dimensions": self.dimensions}

    def _tokens(self, text: str) -> list[str]:
        raw = [token.casefold() for token in _TOKEN_RE.findall(text)]
        return [self._ALIASES.get(token, token) for token in raw]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        features = tokens + [f"{a}:{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[index] += sign
        return _normalise_vectors([vector], 1)[0]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    # Kept only for the shipped demo's old fixture callers.
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@dataclass
class OllamaEmbeddingProvider:
    model: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 600.0
    batch_size: int = 16

    @property
    def dimensions(self) -> int:
        return _QWEN_DIMENSION

    @property
    def identity(self) -> str:
        return f"ollama:{self.model}:{_OLLAMA_MODEL_DIGEST}"

    @property
    def cache_identity(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model": self.model,
            "model_digest": _OLLAMA_MODEL_DIGEST,
            "canonical_model": _QWEN_CANONICAL_MODEL,
            "hf_revision": _QWEN_HF_REVISION,
            "dimensions": self.dimensions,
            "base_url": self.base_url.rstrip("/"),
        }

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict) or "embeddings" not in body:
            raise ValueError("Ollama response did not contain embeddings")
        return _normalise_vectors(body["embeddings"], len(texts))

    def validate_model(self) -> None:
        if self.model != "qwen3-embedding:0.6b":
            raise ValueError("unsupported Ollama model tag")
        req = urllib.request.Request(self.base_url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            raise ValueError("Ollama tags response is invalid")
        match = next((row for row in models if isinstance(row, dict) and row.get("name") == self.model), None)
        if match is None or match.get("digest") != _OLLAMA_MODEL_DIGEST:
            raise ValueError("Ollama model tag or digest does not match frozen identity")

    def embed_query(self, query: str) -> list[float]:
        prompt = f"Instruct: {_QUERY_TASK}\nQuery:{query}"
        return self._request([prompt])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors: list[list[float]] = []
        dimension: int | None = None
        for start in range(0, len(texts), self.batch_size):
            batch = self._request(texts[start:start + self.batch_size])
            if dimension is None:
                dimension = len(batch[0]) if batch else None
            elif batch and len(batch[0]) != dimension:
                raise ValueError("embedding vectors must have a consistent dimension across batches")
            vectors.extend(batch)
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@dataclass(frozen=True)
class DenseHit:
    doc_id: str
    score: float


def _document_text(doc: Document) -> str:
    context = (doc.context + "\n") * 2 if doc.context else ""
    return f"{doc.title}\n{context}{doc.text}"


def corpus_identity(documents: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_document_text(doc).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _provider_identity(provider: EmbeddingProvider) -> object:
    return getattr(provider, "cache_identity", {"identity": provider.identity})


def _validate_cached_vectors(vectors: object, *, expected_count: int, expected_dimension: int | None = None) -> tuple[list[list[float]], int]:
    validated = _normalise_vectors(vectors, expected_count)
    dimension = len(validated[0]) if validated else 0
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError("cached embedding dimension mismatch")
    return validated, dimension


class DenseIndex:
    def __init__(self, documents: list[Document], vectors: dict[str, list[float]], provider: EmbeddingProvider) -> None:
        self.documents = {doc.doc_id: doc for doc in documents}
        self.vectors = vectors
        self.provider = provider
        self.dimension = len(next(iter(vectors.values()))) if vectors else 0

    @classmethod
    def build(cls, documents: list[Document], provider: EmbeddingProvider, *, cache_path: str | Path) -> "DenseIndex":
        docs = list(documents)
        if not docs:
            raise ValueError("dense corpus is empty")
        document_ids = [doc.doc_id for doc in docs]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique")
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_id = corpus_identity(docs)
        provider_id = _provider_identity(provider)
        vectors: list[list[float]] | None = None
        expected_cache_dimension = getattr(provider, "dimensions", None)
        validate_model = getattr(provider, "validate_model", None)
        if validate_model is not None:
            validate_model()
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("dense cache is corrupt") from exc
            if not isinstance(payload, dict):
                raise ValueError("dense cache is corrupt")
            compatible = (
                payload.get("schema_version") == _CACHE_SCHEMA and
                payload.get("provider") == provider_id and
                payload.get("corpus_identity") == corpus_id and
                payload.get("document_ids") == document_ids
            )
            if compatible:
                if payload.get("count") != len(docs):
                    raise ValueError("dense cache document count mismatch")
                declared_dimension = payload.get("dimension")
                if not isinstance(declared_dimension, int) or declared_dimension < 1:
                    raise ValueError("dense cache dimension is corrupt")
                vectors, dimension = _validate_cached_vectors(
                    payload.get("vectors"), expected_count=len(docs), expected_dimension=declared_dimension
                )
                if expected_cache_dimension is not None and dimension != expected_cache_dimension:
                    raise ValueError("cached embedding dimension mismatch")

        if vectors is None:
            vectors = _normalise_vectors(provider.embed_documents([_document_text(doc) for doc in docs]), len(docs))
            dimension = len(vectors[0])
            if expected_cache_dimension is not None and dimension != expected_cache_dimension:
                raise ValueError("embedding dimension mismatch")
            payload = {
                "schema_version": _CACHE_SCHEMA,
                "provider": provider_id,
                "corpus_identity": corpus_id,
                "document_ids": document_ids,
                "count": len(docs),
                "dimension": dimension,
                "vectors": vectors,
            }
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=cache_path.parent, delete=False) as handle:
                    temporary_name = handle.name
                    json.dump(payload, handle, separators=(",", ":"))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, cache_path)
            finally:
                if temporary_name:
                    Path(temporary_name).unlink(missing_ok=True)

        doc_vectors = {doc.doc_id: vector for doc, vector in zip(docs, vectors)}
        return cls(docs, doc_vectors, provider)

    def search(self, query: str, *, limit: int = 10) -> list[DenseHit]:
        query_vector = _normalise_vectors([self.provider.embed_query(query)], 1)[0]
        if len(query_vector) != self.dimension:
            raise ValueError("query embedding dimension mismatch")
        hits = [DenseHit(doc_id, sum(a * b for a, b in zip(query_vector, vector))) for doc_id, vector in self.vectors.items()]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:limit]
