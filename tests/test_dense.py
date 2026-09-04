import json
from pathlib import Path

from local_rag.dense import DenseIndex, OllamaEmbeddingProvider, SimpleEmbeddingProvider
from local_rag.documents import Document


class CountingProvider(SimpleEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def docs():
    return [
        Document("incident:payments", "Payment outage", "Payment authorization requests were rejected and checkout became unavailable.", "incidents/payments.md", "incidents"),
        Document("arch:auth", "Auth architecture", "Tokens are signed by the identity service.", "architecture/auth.md", "architecture"),
    ]


def test_dense_semantic_paraphrase_ranks_payment_outage_first(tmp_path):
    index = DenseIndex.build(docs(), SimpleEmbeddingProvider(), cache_path=tmp_path / "dense.json")
    results = index.search("why could users not complete checkout payments?", limit=2)
    assert results[0].doc_id == "incident:payments"


def test_embedding_provider_exposes_separate_query_and_document_contracts():
    provider = SimpleEmbeddingProvider(dimensions=8)

    query = provider.embed_query("find a payment incident")
    documents = provider.embed_documents(["payment incident", "authentication"])

    assert len(query) == 8
    assert len(documents) == 2
    assert all(len(vector) == 8 for vector in documents)


def test_ollama_provider_uses_frozen_query_instruction_and_bounded_document_batches(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"embeddings": [[3.0, 4.0] for _ in requests[-1]["input"]]}).encode()

    def fake_urlopen(request, *, timeout):
        requests.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaEmbeddingProvider(batch_size=2)

    provider.embed_query("find runbook")
    provider.embed_documents(["one", "two", "three"])

    assert requests[0] == {
        "model": "qwen3-embedding:0.6b",
        "input": ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:find runbook"],
    }
    assert [request["input"] for request in requests[1:]] == [["one", "two"], ["three"]]


def test_dense_cache_persists_identity_order_dimension_and_reuses_vectors(tmp_path):
    class Provider:
        identity = "fake:v1"
        dimensions = 2
        def __init__(self):
            self.document_calls = 0
        @property
        def cache_identity(self):
            return {"provider": "fake", "identity": self.identity}
        def embed_documents(self, texts):
            self.document_calls += 1
            return [[1.0, 0.0] for _ in texts]
        def embed_query(self, query):
            return [1.0, 0.0]

    cache = tmp_path / "dense.json"
    provider = Provider()
    DenseIndex.build(docs(), provider, cache_path=cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "local-rag-dense-cache/v3"
    assert payload["document_ids"] == [doc.doc_id for doc in docs()]
    assert payload["dimension"] == 2
    assert provider.document_calls == 1

    DenseIndex.build(docs(), provider, cache_path=cache)
    assert provider.document_calls == 1


def test_dense_cache_rejects_corrupt_compatible_vectors(tmp_path):
    class Provider:
        identity = "fake:v1"
        dimensions = 2
        cache_identity = {"provider": "fake", "identity": "fake:v1"}
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]
        def embed_query(self, query):
            return [1.0, 0.0]

    cache = tmp_path / "dense.json"
    DenseIndex.build(docs(), Provider(), cache_path=cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["vectors"][0] = [0.0, 0.0]
    cache.write_text(json.dumps(payload), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="nonzero norm"):
        DenseIndex.build(docs(), Provider(), cache_path=cache)


def test_dense_build_rejects_provider_output_dimension_mismatch(tmp_path):
    class Provider:
        identity = "fake:v1"
        dimensions = 2
        cache_identity = {"provider": "fake", "identity": "fake:v1"}

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, query):
            return [1.0, 0.0]

    import pytest
    with pytest.raises(ValueError, match="dimension mismatch"):
        DenseIndex.build(docs(), Provider(), cache_path=tmp_path / "dense.json")


def test_ollama_model_validation_requires_frozen_tag_and_digest(monkeypatch):
    requests = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"models": [{"name": "qwen3-embedding:0.6b", "digest": "wrong"}]}).encode()

    def fake_urlopen(request, *, timeout):
        requests.append(request.full_url)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    import pytest
    with pytest.raises(ValueError, match="digest"):
        OllamaEmbeddingProvider().validate_model()
    assert requests == ["http://localhost:11434/api/tags"]


def test_ollama_provider_declares_native_qwen_dimension():
    assert OllamaEmbeddingProvider().dimensions == 1024


def test_dense_cache_reuses_document_embeddings(tmp_path):
    provider = CountingProvider()
    cache = tmp_path / "dense.json"
    DenseIndex.build(docs(), provider, cache_path=cache)
    first_calls = provider.calls
    assert first_calls == 1
    DenseIndex.build(docs(), provider, cache_path=cache)
    assert provider.calls == first_calls
