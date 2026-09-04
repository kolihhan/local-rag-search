import json
from pathlib import Path

from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.cli import main


ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "demo_docs"


def test_fastapi_search_query_and_document_primitives(tmp_path):
    client = TestClient(create_app(default_corpus=CORPUS, cache_dir=tmp_path))
    health = client.get("/health").json()
    assert health["indexed_documents"] == 15

    response = client.post("/search", json={"query": "ORA-12516", "mode": "lex", "limit": 3, "explain": True})
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["doc_id"] == "incidents:database-sessions"
    assert first["bm25_rank"] == 1

    typed = client.post("/query", json={
        "original": "why could shoppers not pay?",
        "intent": "payment incident",
        "lex": ["payment authorization"],
        "vec": ["users could not finish checkout"],
        "hyde": [],
        "limit": 3,
        "explain": True
    })
    assert typed.status_code == 200
    assert typed.json()["results"][0]["doc_id"] == "incidents:payment-outage"

    doc = client.get("/documents/incidents:payment-outage")
    assert doc.status_code == 200
    assert "checkout" in doc.json()["text"].lower()


def test_cli_vsearch_is_vec_only_sugar_and_query_explain(tmp_path, capsys):
    code = main(["--corpus", str(CORPUS), "--cache-dir", str(tmp_path), "vsearch", "users cannot finish checkout"])
    out = capsys.readouterr().out
    assert code == 0
    assert "incidents:payment-outage" in out
    assert "mode=vec" in out

    code = main(["--corpus", str(CORPUS), "--cache-dir", str(tmp_path), "query", "payment timeout", "--lex", "payment", "--vec", "checkout failure", "--explain"])
    out = capsys.readouterr().out
    assert code == 0
    assert "BM25" in out and "Dense" in out and "RRF" in out


def test_evaluation_files_physically_separate_queries_and_gold_ids():
    inference = json.loads((ROOT / "evaluation" / "inference_queries.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "evaluation" / "gold_labels.json").read_text(encoding="utf-8"))
    assert inference and labels
    assert all("gold" not in key and "expected" not in key for row in inference for key in row)
    assert all(set(row) == {"case_id", "query"} for row in inference)
    assert all(set(row) == {"case_id", "relevant_document_ids"} for row in labels)


def test_tiny_evaluation_requires_hybrid_to_cover_both_lexical_and_semantic_cases(tmp_path):
    from local_rag.index import build_search_service
    service = build_search_service(CORPUS, cache_dir=tmp_path)
    inference = json.loads((ROOT / "evaluation" / "inference_queries.json").read_text(encoding="utf-8"))
    labels = {row["case_id"]: set(row["relevant_document_ids"]) for row in json.loads((ROOT / "evaluation" / "gold_labels.json").read_text(encoding="utf-8"))}

    def recall(mode):
        hit = 0
        for row in inference:
            ids = {item.doc_id for item in service.search(row["query"], mode=mode, limit=10)}
            hit += bool(ids & labels[row["case_id"]])
        return hit / len(inference)

    lex = recall("lex")
    vec = recall("vec")
    hybrid = recall("hybrid")
    assert hybrid > lex
    assert hybrid >= vec


def test_module_demo_app_is_preindexed_for_one_command_portfolio_demo():
    from local_rag.api import demo_app
    client = TestClient(demo_app)
    assert client.get("/health").json()["indexed_documents"] == 15
