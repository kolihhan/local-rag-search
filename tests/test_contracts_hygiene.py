from dataclasses import fields
from pathlib import Path

from local_rag.documents import Document
from local_rag.query import QueryPlan
from local_rag.search import SearchResult


def test_runtime_contracts_exclude_evaluation_labels():
    forbidden = {"gold_ids", "gold_doc_ids", "expected_doc_ids", "reference_answer", "answer_facts"}
    for cls in (Document, QueryPlan, SearchResult):
        assert {f.name for f in fields(cls)}.isdisjoint(forbidden)


def test_runtime_source_does_not_import_evaluation():
    src = Path(__file__).parents[1] / "src" / "local_rag"
    text = "\n".join(p.read_text(encoding="utf-8") for p in src.glob("*.py")).lower()
    assert "from evaluation" not in text
    assert "import evaluation" not in text
    for token in ("gold_doc_ids", "expected_doc_ids", "answer_facts"):
        assert token not in text
