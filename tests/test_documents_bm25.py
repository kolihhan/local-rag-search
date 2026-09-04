from pathlib import Path

from local_rag.bm25 import BM25Index
from local_rag.documents import load_markdown_corpus


def test_loader_assigns_stable_ids_collection_and_context(tmp_path):
    (tmp_path / "incidents").mkdir()
    (tmp_path / "runbooks").mkdir()
    (tmp_path / "incidents" / "payment-outage.md").write_text("# Payment outage\nORA-12516 caused checkout failure", encoding="utf-8")
    (tmp_path / "runbooks" / "database-pool.md").write_text("# DB pool\nIncrease pool size carefully", encoding="utf-8")
    docs = load_markdown_corpus(tmp_path, {"incidents": "Production incident reports."})
    by_id = {doc.doc_id: doc for doc in docs}
    doc = by_id["incidents:payment-outage"]
    assert doc.collection == "incidents"
    assert doc.context == "Production incident reports."
    assert doc.path == "incidents/payment-outage.md"


def test_bm25_exact_identifier_ranks_relevant_doc_first(tmp_path):
    (tmp_path / "incidents").mkdir()
    (tmp_path / "runbooks").mkdir()
    (tmp_path / "incidents" / "oracle.md").write_text("Database refused new sessions with ORA-12516 during peak traffic.", encoding="utf-8")
    (tmp_path / "runbooks" / "payments.md").write_text("Payment recovery steps after gateway errors.", encoding="utf-8")
    docs = load_markdown_corpus(tmp_path)
    results = BM25Index(docs).search("ORA-12516", limit=2)
    assert results[0].doc_id == "incidents:oracle"
    assert results[0].score > 0
