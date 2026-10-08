from pathlib import Path

from local_rag.dense import SimpleEmbeddingProvider
from local_rag.documents import Document
from local_rag.fusion import reciprocal_rank_fusion
from local_rag.query import QueryPlan
from local_rag.rerank import Reranker
from local_rag.search import SearchService


def docs():
    return [
        Document("inc:oracle", "Oracle connection incident", "ORA-12516 blocked database sessions during peak traffic.", "incidents/oracle.md", "incidents", "Production incident reports."),
        Document("inc:payment", "Payment outage", "Payment authorization requests were rejected, so shoppers could not complete checkout.", "incidents/payment.md", "incidents", "Production incident reports."),
        Document("run:payment", "Payment recovery", "Restart the payment gateway worker and verify authorization health.", "runbooks/payment.md", "runbooks", "Operational procedures."),
    ]


def test_rrf_uses_ranks_not_incompatible_raw_scores():
    fused = reciprocal_rank_fusion({"lex": [("a", 999.0), ("b", 10.0)], "vec": [("b", 0.9), ("a", 0.1)]}, k=60)
    assert [row.doc_id for row in fused[:2]] == ["a", "b"]
    assert fused[0].source_ranks == {"lex": 1, "vec": 2}
    assert fused[1].source_ranks == {"lex": 2, "vec": 1}


def test_rrf_can_weight_a_more_reliable_source_without_using_raw_scores():
    rankings = {
        "bm25": [("a", 999.0), ("b", 1.0)],
        "dense": [("b", 0.99), ("a", 0.01)],
    }
    fused = reciprocal_rank_fusion(rankings, k=60, weights={"bm25": 1.5, "dense": 1.0})
    assert [row.doc_id for row in fused[:2]] == ["a", "b"]
    assert fused[0].source_ranks == {"bm25": 1, "dense": 2}


def test_typed_query_routes_lex_and_vec_and_exposes_explain_signals(tmp_path):
    service = SearchService.from_documents(docs(), embedding_provider=SimpleEmbeddingProvider(), cache_path=tmp_path / "dense.json")
    plan = QueryPlan(
        original="why could shoppers not pay?",
        intent="payment incident root cause",
        lex=("payment authorization",),
        vec=("users could not finish checkout",),
    )
    results = service.query(plan, limit=3, explain=True)
    assert results[0].doc_id == "inc:payment"
    assert "lex" in results[0].matched_signals
    assert "vec" in results[0].matched_signals
    assert results[0].bm25_rank is not None
    assert results[0].dense_rank is not None
    assert results[0].rrf_rank == 1
    assert results[0].context == "Production incident reports."


def test_hyde_is_routed_to_dense_not_lexical(tmp_path):
    service = SearchService.from_documents(docs(), embedding_provider=SimpleEmbeddingProvider(), cache_path=tmp_path / "dense.json")
    results = service.query(QueryPlan(original="x", hyde=("Payment checkout failed because authorization was rejected",)), limit=2, explain=True)
    assert results[0].doc_id == "inc:payment"
    assert "hyde" in results[0].matched_signals
    assert results[0].bm25_rank is None
    assert results[0].dense_rank is not None


def test_stable_document_get_and_get_many(tmp_path):
    service = SearchService.from_documents(docs(), embedding_provider=SimpleEmbeddingProvider(), cache_path=tmp_path / "dense.json")
    assert service.get("inc:oracle").title == "Oracle connection incident"
    assert [d.doc_id for d in service.get_many(["run:payment", "inc:payment"])] == ["run:payment", "inc:payment"]


class RecordingReranker:
    def __init__(self): self.seen = 0
    def rerank(self, query, documents):
        self.seen = len(documents)
        return [(doc.doc_id, float(len(doc.text))) for doc in documents]


def test_reranker_receives_only_bounded_candidate_pool(tmp_path):
    many = [Document(f"d{i}", f"Doc {i}", f"payment failure document {i}", f"d{i}.md", "x") for i in range(50)]
    reranker = RecordingReranker()
    service = SearchService.from_documents(many, embedding_provider=SimpleEmbeddingProvider(), cache_path=tmp_path / "dense.json", reranker=reranker, rerank_pool=7)
    service.search("payment failure", mode="hybrid", limit=5, rerank=True)
    assert reranker.seen == 7
