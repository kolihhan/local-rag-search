import json

import pytest

from evaluation.evaluate import (
    ArmMetrics,
    BenchmarkReport,
    PerQueryMetrics,
    RankTransition,
    build_enterprise_benchmark_arms,
    load_enterprise_rag_txt_corpus,
    persist_benchmark_report,
    run_enterprise_benchmark,
    validate_enterprise_provenance,
)
from local_rag.documents import Document
from local_rag.dense import OllamaEmbeddingProvider


def test_enterprise_arms_are_exactly_bm25_dense_rrf_with_frozen_depths(monkeypatch, tmp_path):
    docs = (Document("dsid_" + "a" * 32, "title", "text", "x.txt", "confluence"),)
    calls = {"bm25": [], "dense": [], "rrf": []}

    class FakeBM25:
        def __init__(self, documents):
            pass
        def search(self, query, *, limit):
            calls["bm25"].append(limit)
            return [type("Hit", (), {"doc_id": "bm", "score": 1.0})()]

    class FakeDense:
        def __init__(self, documents, vectors, provider):
            pass
        @classmethod
        def build(cls, documents, provider, *, cache_path):
            return cls(documents, {}, provider)
        def search(self, query, *, limit):
            calls["dense"].append(limit)
            return [type("Hit", (), {"doc_id": "de", "score": 1.0})()]

    monkeypatch.setattr("evaluation.evaluate.BM25Index", FakeBM25)
    monkeypatch.setattr("evaluation.evaluate.DenseIndex", FakeDense)
    arms = build_enterprise_benchmark_arms(documents=docs, embedding_provider=object(), dense_cache_path=tmp_path / "dense.json")

    assert tuple(arms) == ("bm25", "dense", "rrf")
    assert list(arms["bm25"]("query")) == ["bm"]
    assert list(arms["dense"]("query")) == ["de"]
    assert list(arms["rrf"]("query")) == ["bm", "de"]
    assert calls["bm25"] == [20, 60]
    assert calls["dense"] == [20, 60]


def test_enterprise_corpus_loader_is_sorted_and_fail_closed(tmp_path):
    root = tmp_path / "slice" / "confluence"
    root.mkdir(parents=True)
    (root / ("dsid_" + "b" * 32 + "__z.txt")).write_text("B title\nbody", encoding="utf-8")
    (root / ("dsid_" + "a" * 32 + "__a.txt")).write_text("A title\nbody", encoding="utf-8")
    docs = load_enterprise_rag_txt_corpus(roots=[tmp_path / "slice"])
    assert [doc.doc_id for doc in docs] == sorted(doc.doc_id for doc in docs)
    assert all(doc.collection == "confluence" for doc in docs)

    bad = tmp_path / "bad" / "confluence"
    bad.mkdir(parents=True)
    (bad / "bad.txt").write_text("title", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_enterprise_rag_txt_corpus(roots=[tmp_path / "bad"])

    duplicate = tmp_path / "duplicate" / "confluence"
    duplicate.mkdir(parents=True)
    duplicate_id = "dsid_" + "c" * 32
    (duplicate / f"{duplicate_id}__one.txt").write_text("one", encoding="utf-8")
    (duplicate / f"{duplicate_id}__two.txt").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_enterprise_rag_txt_corpus(roots=[tmp_path / "duplicate"])

    empty = tmp_path / "empty" / "confluence"
    empty.mkdir(parents=True)
    (empty / ("dsid_" + "d" * 32 + "__empty.txt")).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_enterprise_rag_txt_corpus(roots=[tmp_path / "empty"])

    invalid_utf8 = tmp_path / "invalid-utf8" / "confluence"
    invalid_utf8.mkdir(parents=True)
    (invalid_utf8 / ("dsid_" + "e" * 32 + "__bad.txt")).write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_enterprise_rag_txt_corpus(roots=[tmp_path / "invalid-utf8"])

    no_docs = tmp_path / "no-docs" / "confluence"
    no_docs.mkdir(parents=True)
    with pytest.raises(ValueError, match="empty"):
        load_enterprise_rag_txt_corpus(roots=[tmp_path / "no-docs"])


def test_report_persistence_refuses_overwrite_and_writes_three_arms_atomically(tmp_path):
    qid = "qst_0004"
    per_query = {qid: PerQueryMetrics(qid, {"doc": 1}, 1, 1.0, 1.0, 1.0, 1.0, True, None)}
    metric = ArmMetrics(1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, per_query)
    report = BenchmarkReport((qid,), {name: metric for name in ("bm25", "dense", "rrf")},
        {pair: {qid: RankTransition(1, 1, "both_hit")} for pair in ("bm25->dense", "bm25->rrf")})
    output = tmp_path / "report.json"
    persist_benchmark_report(report=report, output_path=output, provenance={"release": "v1.0.0"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert tuple(payload["arms"]) == ("bm25", "dense", "rrf")
    assert tuple(payload["transitions"]) == ("bm25->dense", "bm25->rrf")
    assert set(payload["transition_summary"]) == {
        "bm25_miss_to_dense_hit", "bm25_hit_to_dense_miss",
        "bm25_miss_to_rrf_hit", "bm25_hit_to_rrf_miss",
    }
    with pytest.raises(FileExistsError):
        persist_benchmark_report(report=report, output_path=output, provenance={})
    assert not list(tmp_path.glob("*.tmp"))


def test_provenance_validates_frozen_metadata_and_hashes(monkeypatch, tmp_path):
    manifest = {"benchmark": "EnterpriseRAG-Bench", "release": "v1.0.0", "revision": "56ba6a6",
                "official_release_url": "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0",
                "license_url": "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/v1.0.0/LICENSE"}
    files = {
        "frozen_manifest.json": json.dumps(manifest),
        "gold/development_questions.jsonl": "dev",
        "gold/extra_questions.jsonl": "extra",
        "gold/exclusions.json": "[]",
        "downloads/LICENSE": "license",
        "downloads/confluence_slice_0001.zip": "one",
        "downloads/confluence_slice_0002.zip": "two",
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    monkeypatch.setattr("evaluation.evaluate._HASHES", {
        name: __import__("hashlib").sha256(value.encode()).hexdigest()
        for name, value in {
            "manifest": files["frozen_manifest.json"], "development_questions": "dev", "extra_questions": "extra",
            "exclusions": "[]", "license": "license", "slice_0001": "one", "slice_0002": "two",
        }.items()
    })
    result = validate_enterprise_provenance(tmp_path)
    assert result["release"] == "v1.0.0"
    assert result["revision"] == "56ba6a6"
    assert result["benchmark_scope"] == "metadata-extra; all Confluence rows; 12 singleton-qrel cases; not core leaderboard"
    assert result["release_url"] == manifest["official_release_url"]
    assert result["license_url"] == manifest["license_url"]
    manifest["revision"] = "wrong"
    (tmp_path / "frozen_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_enterprise_provenance(tmp_path)


def test_ollama_frozen_runtime_defaults_and_distinct_paths(tmp_path):
    provider = OllamaEmbeddingProvider()
    assert provider.batch_size == 16
    assert provider.timeout_s == 600.0
    with pytest.raises(ValueError, match="paths must differ"):
        run_enterprise_benchmark(dataset_root=tmp_path, output_path=tmp_path / "same.json", cache_path=tmp_path / "same.json")


def test_core_confluence_loader_selects_official_rows_in_order(tmp_path):
    from evaluation.evaluate import load_core_confluence_set

    questions = tmp_path / "questions.jsonl"
    rows = [
        {
            "question_id": "qst_0001",
            "question": "first",
            "question_type": "basic",
            "source_types": ["confluence"],
            "expected_doc_ids": ["dsid_" + "a" * 32],
        },
        {
            "question_id": "qst_0002",
            "question": "ignore slack",
            "question_type": "basic",
            "source_types": ["slack"],
            "expected_doc_ids": ["dsid_" + "b" * 32],
        },
        {
            "question_id": "qst_0003",
            "question": "third",
            "question_type": "semantic",
            "source_types": ["confluence"],
            "expected_doc_ids": ["dsid_" + "c" * 32, "dsid_" + "d" * 32],
        },
    ]

    questions.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = load_core_confluence_set(
        questions_path=questions,
        corpus_doc_ids={
            "dsid_" + "a" * 32,
            "dsid_" + "c" * 32,
            "dsid_" + "d" * 32,
        },
        expected_count=2,
        expected_sha256=None,
    )

    assert dataset.question_ids == ("qst_0001", "qst_0003")
    assert dataset.questions["qst_0001"] == "first"
    assert dataset.qrels["qst_0003"] == frozenset({
        "dsid_" + "c" * 32,
        "dsid_" + "d" * 32,
    })


def test_core_confluence_loader_fails_closed_on_missing_qrel_or_wrong_count(tmp_path):
    from evaluation.evaluate import load_core_confluence_set

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "question_id": "qst_0001",
            "question": "question",
            "question_type": "basic",
            "source_types": ["confluence"],
            "expected_doc_ids": ["dsid_" + "a" * 32],
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from corpus"):
        load_core_confluence_set(
            questions_path=questions,
            corpus_doc_ids=set(),
            expected_count=1,
            expected_sha256=None,
        )

    with pytest.raises(ValueError, match="expected exactly 2"):
        load_core_confluence_set(
            questions_path=questions,
            corpus_doc_ids={"dsid_" + "a" * 32},
            expected_count=2,
            expected_sha256=None,
        )
