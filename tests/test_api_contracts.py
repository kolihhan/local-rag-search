from pathlib import Path

from fastapi.testclient import TestClient

from local_rag.api import create_app


ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "demo_docs"


def test_index_rejects_unknown_embedding_name(tmp_path):
    client = TestClient(create_app(cache_dir=tmp_path))
    response = client.post("/index", json={"path": str(CORPUS), "embedding": "simpel"})
    assert response.status_code == 422


def test_search_rejects_unknown_mode_at_request_validation(tmp_path):
    client = TestClient(create_app(default_corpus=CORPUS, cache_dir=tmp_path))
    response = client.post("/search", json={"query": "payment", "mode": "hybird"})
    assert response.status_code == 422
