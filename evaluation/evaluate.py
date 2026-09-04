import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_rag.bm25 import BM25Index
from local_rag.dense import DenseIndex, EmbeddingProvider, OllamaEmbeddingProvider, corpus_identity
from local_rag.documents import Document
from local_rag.fusion import reciprocal_rank_fusion
from local_rag.index import build_search_service


APPROVED_DEV_IDS = (
    "qst_0004", "qst_0009", "qst_0016", "qst_0019", "qst_0023", "qst_0024",
    "qst_0036", "qst_0037", "qst_0039", "qst_0045", "qst_0057", "qst_0081",
)
_ENTERPRISE_DOC_RE = re.compile(r"^(dsid_[0-9a-f]{32})__(.+)\.txt$")
_HASHES = {
    "manifest": "d7d26db35fbda29f09d7e3ebf180d6c18e3c7685c2b05938e3303a2429a7eb93",
    "development_questions": "1d9ba38b9b6e2f937cfe009c75ba48f15b285c007a474128f5e23296ce783fd8",
    "extra_questions": "26e23e5ade467512433e0fc012b30ea14b079dde11ad6f93cf380eed1bd96807",
    "exclusions": "fbdd09e3854da3e8cd48e097299ced1aa0ada114ecf93a07bb7c19af49111e24",
    "license": "ec3c1cbcaee249b14bc11433949ddd6569837f1110eccc086503b8a5b80e66cf",
    "slice_0001": "a79320d9c11b58c904ddc444f4b5c374ea1d9e6e3075974bcaea294c2a7e2eb5",
    "slice_0002": "ecb4e395f710f0eef8332b1e27182c00223da847945a010c10078b3e0c693024",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_enterprise_rag_txt_corpus(*, roots: Sequence[Path]) -> tuple[Document, ...]:
    documents: list[Document] = []
    seen: set[str] = set()
    for root in roots:
        for path in sorted(Path(root).rglob("*.txt")):
            match = _ENTERPRISE_DOC_RE.match(path.name)
            if not match:
                raise ValueError(f"invalid EnterpriseRAG document filename: {path.name}")
            doc_id = match.group(1)
            if doc_id in seen:
                raise ValueError(f"duplicate EnterpriseRAG document ID: {doc_id}")
            seen.add(doc_id)
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 document: {path}") from exc
            if not text.strip():
                raise ValueError(f"empty EnterpriseRAG document: {path}")
            title = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if not title:
                raise ValueError(f"document has no non-empty title line: {path}")
            documents.append(Document(doc_id, title, text, str(path), "confluence", ""))
    if not documents:
        raise ValueError("EnterpriseRAG corpus is empty")
    return tuple(sorted(documents, key=lambda document: document.doc_id))


def validate_enterprise_provenance(dataset_root: Path) -> dict[str, object]:
    dataset_root = Path(dataset_root)
    files = {
        "manifest": dataset_root / "frozen_manifest.json",
        "development_questions": dataset_root / "gold" / "development_questions.jsonl",
        "extra_questions": dataset_root / "gold" / "extra_questions.jsonl",
        "exclusions": dataset_root / "gold" / "exclusions.json",
        "license": dataset_root / "downloads" / "LICENSE",
        "slice_0001": dataset_root / "downloads" / "confluence_slice_0001.zip",
        "slice_0002": dataset_root / "downloads" / "confluence_slice_0002.zip",
    }
    actual = {name: _sha256(path) for name, path in files.items()}
    if actual != _HASHES:
        raise ValueError("EnterpriseRAG provenance hash mismatch")
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if (manifest.get("benchmark") != "EnterpriseRAG-Bench" or manifest.get("release") != "v1.0.0" or
            manifest.get("revision") != "56ba6a6" or
            manifest.get("official_release_url") != "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0" or
            manifest.get("license_url") != "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/v1.0.0/LICENSE"):
        raise ValueError("EnterpriseRAG manifest metadata mismatch")
    return {
        "benchmark": manifest["benchmark"],
        "release": manifest["release"],
        "revision": manifest["revision"],
        "release_url": manifest["official_release_url"],
        "license_url": manifest["license_url"],
        "benchmark_scope": "metadata-extra; all Confluence rows; 12 singleton-qrel cases; not core leaderboard",
        "sha256": actual,
    }


@dataclass(frozen=True)
class FrozenDevSet:
    question_ids: tuple[str, ...]
    questions: Mapping[str, str]
    qrels: Mapping[str, frozenset[str]]


def load_frozen_dev_set(*, manifest_path: Path, questions_path: Path,
                        corpus_doc_ids: set[str] | None = None,
                        require_enterprise_metadata: bool = False) -> FrozenDevSet:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ids = manifest.get("selected_question_ids")
    if ids != list(APPROVED_DEV_IDS):
        raise ValueError("manifest does not contain the exact approved ordered development set")
    rows = []
    for line in Path(questions_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id, question, expected = row.get("question_id"), row.get("question"), row.get("expected_doc_ids")
        if require_enterprise_metadata and (row.get("question_type") != "metadata" or row.get("source_types") != ["confluence"]):
            raise ValueError("selected questions must be metadata Confluence rows")
        if (not isinstance(question_id, str) or not question_id or not isinstance(question, str) or not question.strip() or
                not isinstance(expected, list) or not expected or any(not isinstance(item, str) or not item for item in expected) or
                len(expected) != len(set(expected))):
            raise ValueError("question rows must have non-empty unique string fields")
        if corpus_doc_ids is not None and not set(expected).issubset(corpus_doc_ids):
            raise ValueError(f"qrel references a document absent from corpus: {question_id}")
        rows.append((question_id, question, frozenset(expected)))
    if len(rows) != len(ids) or [row[0] for row in rows] != list(ids):
        raise ValueError("questions do not exactly match the approved ordered development set")
    return FrozenDevSet(tuple(ids), {row[0]: row[1] for row in rows}, {row[0]: row[2] for row in rows})


def build_enterprise_benchmark_arms(*, documents: Sequence[Document], embedding_provider: EmbeddingProvider,
                                    dense_cache_path: Path) -> Mapping[str, Callable[[str], Sequence[str]]]:
    docs = list(documents)
    bm25 = BM25Index(docs)
    dense = DenseIndex.build(docs, embedding_provider, cache_path=dense_cache_path)

    def bm25_arm(query: str) -> Sequence[str]:
        return [hit.doc_id for hit in bm25.search(query, limit=20)]

    def dense_arm(query: str) -> Sequence[str]:
        return [hit.doc_id for hit in dense.search(query, limit=20)]

    def rrf_arm(query: str) -> Sequence[str]:
        lex = [(hit.doc_id, hit.score) for hit in bm25.search(query, limit=60)]
        vec = [(hit.doc_id, hit.score) for hit in dense.search(query, limit=60)]
        fused = reciprocal_rank_fusion({"bm25": lex, "dense": vec}, k=60)
        return [row.doc_id for row in fused[:20]]

    return {"bm25": bm25_arm, "dense": dense_arm, "rrf": rrf_arm}


@dataclass(frozen=True)
class PerQueryMetrics:
    question_id: str
    gold_ranks: Mapping[str, int | None]
    first_relevant_rank_at_20: int | None
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank_at_20: float
    hit_at_10: bool
    latency_ms: float | None
    ranking: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_20: float
    hit_at_10: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    per_query: Mapping[str, PerQueryMetrics]


@dataclass(frozen=True)
class RankTransition:
    source_rank_at_20: int | None
    target_rank_at_20: int | None
    category: str


@dataclass(frozen=True)
class BenchmarkReport:
    question_ids: tuple[str, ...]
    arms: Mapping[str, ArmMetrics]
    transitions: Mapping[str, Mapping[str, RankTransition]]
    complete: bool = True


def run_benchmark(*, dataset: FrozenDevSet, arms: Mapping[str, Callable[[str], Sequence[str]]], recall_k: int = 10,
                  mrr_k: int = 20, measure_latency: bool = False,
                  clock_ns: Callable[[], int] = time.perf_counter_ns) -> BenchmarkReport:
    canonical = ("bm25", "dense", "rrf")
    if tuple(arms) != canonical and set(arms) != set(canonical):
        raise ValueError("arms must be exactly bm25, dense, rrf")
    if recall_k < 1 or mrr_k < 1:
        raise ValueError("metric cutoffs must be positive")
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in canonical}
    latencies: dict[str, dict[str, float | None]] = {name: {} for name in canonical}

    def invoke(name: str, question_id: str) -> None:
        started = clock_ns() if measure_latency else None
        ranked = arms[name](dataset.questions[question_id])
        if started is not None:
            latencies[name][question_id] = (clock_ns() - started) / 1_000_000.0
        if isinstance(ranked, (str, bytes)):
            raise ValueError("arm results must be an ordered sequence of document IDs")
        ranked_ids = list(ranked)
        for doc_id in ranked_ids:
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("arm results must contain non-empty string IDs")
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("arm results must contain unique document IDs")
        rankings[name][question_id] = ranked_ids

    if measure_latency:
        for index, question_id in enumerate(dataset.question_ids):
            rotated = canonical[index % len(canonical):] + canonical[:index % len(canonical)]
            for name in rotated:
                invoke(name, question_id)
    else:
        for name in canonical:
            for question_id in dataset.question_ids:
                invoke(name, question_id)

    def query_metrics(name: str, question_id: str) -> PerQueryMetrics:
        ranked = rankings[name][question_id]
        relevant = dataset.qrels[question_id]
        positions = {doc_id: index for index, doc_id in enumerate(ranked, start=1)}
        gold_ranks = {doc_id: positions.get(doc_id) for doc_id in sorted(relevant)}
        first = min((rank for rank in gold_ranks.values() if rank is not None), default=None)
        first20 = first if first is not None and first <= mrr_k else None
        return PerQueryMetrics(question_id, gold_ranks, first20,
            len(relevant & set(ranked[:1])) / len(relevant),
            len(relevant & set(ranked[:5])) / len(relevant),
            len(relevant & set(ranked[:recall_k])) / len(relevant),
            1.0 / first20 if first20 is not None else 0.0,
            bool(relevant & set(ranked[:10])), latencies[name].get(question_id), tuple(ranked))

    metrics: dict[str, ArmMetrics] = {}
    for name in canonical:
        per_query = {qid: query_metrics(name, qid) for qid in dataset.question_ids}
        values = list(per_query.values())
        durations = sorted(value.latency_ms for value in values if value.latency_ms is not None)

        def percentile(p: float) -> float | None:
            if not durations:
                return None
            index = max(0, min(len(durations) - 1, int((p * len(durations) + 0.9999999999) - 1)))
            return durations[index]

        metrics[name] = ArmMetrics(
            sum(value.recall_at_1 for value in values) / len(values),
            sum(value.recall_at_5 for value in values) / len(values),
            sum(value.recall_at_10 for value in values) / len(values),
            sum(value.reciprocal_rank_at_20 for value in values) / len(values),
            sum(value.hit_at_10 for value in values) / len(values),
            sum(durations) / len(durations) if durations else None,
            percentile(0.50), percentile(0.95), per_query)

    transitions: dict[str, dict[str, RankTransition]] = {}
    for source, target in (("bm25", "dense"), ("bm25", "rrf")):
        pair = {}
        for qid in dataset.question_ids:
            source_rank = metrics[source].per_query[qid].first_relevant_rank_at_20
            target_rank = metrics[target].per_query[qid].first_relevant_rank_at_20
            source_hit = metrics[source].per_query[qid].hit_at_10
            target_hit = metrics[target].per_query[qid].hit_at_10
            category = ("both_hit" if source_hit and target_hit else
                        "source_hit_target_miss" if source_hit else
                        "source_miss_target_hit" if target_hit else "both_miss")
            pair[qid] = RankTransition(source_rank, target_rank, category)
        transitions[f"{source}->{target}"] = pair
    return BenchmarkReport(dataset.question_ids, metrics, transitions)


def decide_verdict(report: BenchmarkReport) -> str:
    if not report.complete:
        return "INSUFFICIENT"

    def earns(candidate: str, simpler: str) -> bool:
        target, source = report.arms[candidate], report.arms[simpler]
        if any(source.per_query[qid].hit_at_10 and not target.per_query[qid].hit_at_10 for qid in report.question_ids):
            return False
        return (target.recall_at_10 >= source.recall_at_10 and target.mrr_at_20 >= source.mrr_at_20 and
                (target.recall_at_10 > source.recall_at_10 or target.mrr_at_20 > source.mrr_at_20))

    def weakly_dominates(candidate: str, other: str) -> bool:
        target, source = report.arms[candidate], report.arms[other]
        return target.recall_at_10 >= source.recall_at_10 and target.mrr_at_20 >= source.mrr_at_20

    dense_over_bm25 = earns("dense", "bm25")
    rrf_over_bm25 = earns("rrf", "bm25")
    rrf_over_dense = earns("rrf", "dense")
    dense_over_rrf = earns("dense", "rrf")
    both_earn_without_winner = dense_over_bm25 and rrf_over_bm25 and not rrf_over_dense and not dense_over_rrf
    if both_earn_without_winner:
        return "INSUFFICIENT"
    if rrf_over_bm25 and (rrf_over_dense or not dense_over_bm25):
        return "KEEP HYBRID"
    if dense_over_bm25 and not rrf_over_dense:
        return "KEEP DENSE"
    bm25_recovery = any(value.category == "source_miss_target_hit" for value in report.transitions["bm25->dense"].values())
    bm25_rrf_recovery = any(value.category == "source_miss_target_hit" for value in report.transitions["bm25->rrf"].values())
    if (not dense_over_bm25 and not rrf_over_bm25 and not bm25_recovery and not bm25_rrf_recovery and
            weakly_dominates("bm25", "dense") and weakly_dominates("bm25", "rrf")):
        return "KEEP BM25"
    return "INSUFFICIENT"


def persist_benchmark_report(*, report: BenchmarkReport, output_path: Path, provenance: Mapping[str, object],
                             config: Mapping[str, object] | None = None) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for name in ("bm25", "dense", "rrf"):
        metric = report.arms[name]
        arms[name] = {"recall_at_1": metric.recall_at_1, "recall_at_5": metric.recall_at_5,
            "recall_at_10": metric.recall_at_10, "mrr_at_20": metric.mrr_at_20,
            "hit_at_10": metric.hit_at_10, "mean_latency_ms": metric.mean_latency_ms,
            "p50_latency_ms": metric.p50_latency_ms, "p95_latency_ms": metric.p95_latency_ms,
            "per_query": {qid: {"question_id": value.question_id, "gold_ranks": dict(value.gold_ranks),
                "first_relevant_rank_at_20": value.first_relevant_rank_at_20, "recall_at_1": value.recall_at_1,
                "recall_at_5": value.recall_at_5, "recall_at_10": value.recall_at_10,
                "reciprocal_rank_at_20": value.reciprocal_rank_at_20, "hit_at_10": value.hit_at_10,
                "latency_ms": value.latency_ms, "ranking": list(value.ranking)} for qid, value in metric.per_query.items()}}
    transitions = {pair: {qid: {"source_rank_at_20": value.source_rank_at_20,
        "target_rank_at_20": value.target_rank_at_20, "category": value.category} for qid, value in rows.items()}
        for pair, rows in report.transitions.items()}
    summary = {}
    for pair, miss_key, hit_key in (("bm25->dense", "bm25_miss_to_dense_hit", "bm25_hit_to_dense_miss"),
                                    ("bm25->rrf", "bm25_miss_to_rrf_hit", "bm25_hit_to_rrf_miss")):
        rows = report.transitions[pair]
        miss_hit = [qid for qid, value in rows.items() if value.category == "source_miss_target_hit"]
        hit_miss = [qid for qid, value in rows.items() if value.category == "source_hit_target_miss"]
        summary[miss_key] = {"case_ids": miss_hit, "count": len(miss_hit)}
        summary[hit_key] = {"case_ids": hit_miss, "count": len(hit_miss)}
    payload = {"schema_version": "local-rag-search-enterprise-benchmark/v2", "provenance": dict(provenance),
        "config": dict(config or {"final_depth": 20, "candidate_depth": 60, "rrf_k": 60}),
        "question_ids": list(report.question_ids), "arms": arms, "transitions": transitions,
        "transition_summary": summary,
        "verdict": decide_verdict(report), "complete": report.complete}
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output_path.parent, delete=False) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, output_path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return output_path


def run_enterprise_benchmark(*, dataset_root: Path, output_path: Path, cache_path: Path,
                             base_url: str = "http://localhost:11434", model: str = "qwen3-embedding:0.6b",
                             timeout_s: float = 600.0, batch_size: int = 16) -> Path:
    if Path(output_path).exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")
    if Path(output_path).resolve() == Path(cache_path).resolve():
        raise ValueError("benchmark output and dense cache paths must differ")
    provenance = validate_enterprise_provenance(dataset_root)
    corpus = load_enterprise_rag_txt_corpus(roots=[Path(dataset_root) / "corpus" / "slice_0001", Path(dataset_root) / "corpus" / "slice_0002"])
    dataset = load_frozen_dev_set(manifest_path=Path(dataset_root) / "frozen_manifest.json",
        questions_path=Path(dataset_root) / "gold" / "development_questions.jsonl",
        corpus_doc_ids={doc.doc_id for doc in corpus}, require_enterprise_metadata=True)
    provider = OllamaEmbeddingProvider(model=model, base_url=base_url, timeout_s=timeout_s, batch_size=batch_size)
    arms = build_enterprise_benchmark_arms(documents=corpus, embedding_provider=provider, dense_cache_path=cache_path)
    report = run_benchmark(dataset=dataset, arms=arms, measure_latency=True)
    implementation_files = [
        Path(__file__),
        Path(__file__).with_name("run_enterprise_benchmark.py"),
        ROOT / "src" / "local_rag" / "bm25.py",
        ROOT / "src" / "local_rag" / "dense.py",
        ROOT / "src" / "local_rag" / "fusion.py",
    ]
    config = {"final_depth": 20, "candidate_depth": 60, "rrf_k": 60, "model": model,
              "base_url": base_url, "timeout_s": timeout_s, "provider_identity": provider.cache_identity,
               "batch_size": batch_size,
               "transition_hit_cutoff": 10,
               "verdict_rule": "Recall@10 primary and MRR@20 secondary; no top-10 regression; one strict gain; ties favor simplicity; conflicts are insufficient",
               "query_contract": {"template": "Instruct: {task_description}\\nQuery:{query}",
                                  "task_description": "Given a web search query, retrieve relevant passages that answer the query",
                                  "document_instruction": "none"}}
    provenance = {**provenance, "corpus_identity": corpus_identity(list(corpus)), "corpus_document_count": len(corpus),
                  "cache_path": str(cache_path), "cache_identity": _sha256(cache_path),
                  "approved_question_ids": list(APPROVED_DEV_IDS),
                  "selection_rule": "official original row order; metadata; source_types exactly [confluence]; resolvable qrels",
                  "selection_scope": "EnterpriseRAG-Bench v1.0.0 metadata-extra; all Confluence rows; 12 singleton-qrel cases; not core leaderboard",
                  "scoring": "dot product over locally L2-normalized vectors; RRF k=60",
                  "normalization": "local L2 normalization; Ollama owns tokenization and pooling",
                  "latency_boundary": "one timed arm invocation per case, including retrieval only",
                  "implementation_file_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in implementation_files}}
    return persist_benchmark_report(report=report, output_path=output_path, provenance=provenance, config=config)



CORE_QUESTIONS_V1_SHA256 = "f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905"
CORE_CONFLUENCE_QUESTION_COUNT = 64


def load_core_confluence_set(
    *,
    questions_path: Path,
    corpus_doc_ids: set[str],
    expected_count: int = CORE_CONFLUENCE_QUESTION_COUNT,
    expected_sha256: str | None = CORE_QUESTIONS_V1_SHA256,
) -> FrozenDevSet:
    questions_path = Path(questions_path)

    if expected_count < 1:
        raise ValueError("expected core question count must be positive")

    if expected_sha256 is not None:
        actual = _sha256(questions_path)
        if actual != expected_sha256:
            raise ValueError(
                f"official core questions hash mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )

    selected: list[tuple[str, str, frozenset[str]]] = []
    seen_ids: set[str] = set()

    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("source_types") != ["confluence"]:
            continue

        expected = row.get("expected_doc_ids")
        if not expected:
            continue

        question_id = row.get("question_id")
        question = row.get("question")

        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in seen_ids
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(expected, list)
            or not expected
            or any(not isinstance(item, str) or not item for item in expected)
            or len(expected) != len(set(expected))
        ):
            raise ValueError("core question rows must have valid unique string fields")

        missing = set(expected) - corpus_doc_ids
        if missing:
            raise ValueError(
                f"qrel references a document absent from corpus: "
                f"{question_id}: {sorted(missing)}"
            )

        seen_ids.add(question_id)
        selected.append((question_id, question, frozenset(expected)))

    if len(selected) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} compatible core Confluence questions, "
            f"got {len(selected)}"
        )

    ids = tuple(row[0] for row in selected)

    return FrozenDevSet(
        ids,
        {row[0]: row[1] for row in selected},
        {row[0]: row[2] for row in selected},
    )


def run_enterprise_core_benchmark(
    *,
    dataset_root: Path,
    core_questions_path: Path,
    output_path: Path,
    cache_path: Path,
    base_url: str = "http://localhost:11434",
    model: str = "qwen3-embedding:0.6b",
    timeout_s: float = 600.0,
    batch_size: int = 16,
) -> Path:
    dataset_root = Path(dataset_root)
    core_questions_path = Path(core_questions_path)
    output_path = Path(output_path)
    cache_path = Path(cache_path)

    if output_path.exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")

    if output_path.resolve() == cache_path.resolve():
        raise ValueError("benchmark output and dense cache paths must differ")

    provenance = validate_enterprise_provenance(dataset_root)

    corpus = load_enterprise_rag_txt_corpus(
        roots=[
            dataset_root / "corpus" / "slice_0001",
            dataset_root / "corpus" / "slice_0002",
        ]
    )

    if len(corpus) != 5189:
        raise ValueError(
            f"expected frozen Confluence corpus to contain 5189 documents, "
            f"got {len(corpus)}"
        )

    corpus_ids = {doc.doc_id for doc in corpus}
    if len(corpus_ids) != 5189:
        raise ValueError("frozen Confluence corpus contains duplicate document IDs")

    dataset = load_core_confluence_set(
        questions_path=core_questions_path,
        corpus_doc_ids=corpus_ids,
    )

    provider = OllamaEmbeddingProvider(
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        batch_size=batch_size,
    )

    arms = build_enterprise_benchmark_arms(
        documents=corpus,
        embedding_provider=provider,
        dense_cache_path=cache_path,
    )

    report = run_benchmark(
        dataset=dataset,
        arms=arms,
        measure_latency=True,
    )

    implementation_files = [
        Path(__file__),
        Path(__file__).with_name("run_enterprise_benchmark.py"),
        ROOT / "src" / "local_rag" / "bm25.py",
        ROOT / "src" / "local_rag" / "dense.py",
        ROOT / "src" / "local_rag" / "fusion.py",
    ]

    config = {
        "final_depth": 20,
        "candidate_depth": 60,
        "rrf_k": 60,
        "model": model,
        "base_url": base_url,
        "timeout_s": timeout_s,
        "provider_identity": provider.cache_identity,
        "batch_size": batch_size,
        "transition_hit_cutoff": 10,
        "verdict_rule": (
            "Recall@10 primary and MRR@20 secondary; no top-10 regression; "
            "one strict gain; ties favor simplicity; conflicts are insufficient"
        ),
        "query_contract": {
            "template": "Instruct: {task_description}\\nQuery:{query}",
            "task_description": (
                "Given a web search query, retrieve relevant passages "
                "that answer the query"
            ),
            "document_instruction": "none",
        },
    }

    provenance = {
        **provenance,
        "benchmark_scope": (
            "EnterpriseRAG-Bench v1.0.0 core questions.jsonl; "
            "Confluence-only retrieval slice; 64 official compatible questions"
        ),
        "core_questions_path": str(core_questions_path),
        "core_questions_sha256": _sha256(core_questions_path),
        "corpus_identity": corpus_identity(list(corpus)),
        "corpus_document_count": len(corpus),
        "cache_path": str(cache_path),
        "cache_identity": _sha256(cache_path),
        "approved_question_ids": list(dataset.question_ids),
        "selection_rule": (
            "official original row order; source_types exactly [confluence]; "
            "expected_doc_ids non-empty; every qrel resolves against frozen corpus"
        ),
        "selection_scope": (
            "EnterpriseRAG-Bench v1.0.0 core questions; "
            "all 64 Confluence-only qrel-compatible rows"
        ),
        "scoring": "dot product over locally L2-normalized vectors; RRF k=60",
        "normalization": (
            "local L2 normalization; Ollama owns tokenization and pooling"
        ),
        "latency_boundary": (
            "one timed arm invocation per case, including retrieval only"
        ),
        "implementation_file_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in implementation_files
        },
    }

    return persist_benchmark_report(
        report=report,
        output_path=output_path,
        provenance=provenance,
        config=config,
    )


def evaluate(mode: str, *, corpus: Path, cache_dir: Path, depth: int = 20) -> dict[str, float]:
    queries = json.loads((ROOT / "evaluation" / "inference_queries.json").read_text(encoding="utf-8"))
    labels = {row["case_id"]: set(row["relevant_document_ids"]) for row in json.loads((ROOT / "evaluation" / "gold_labels.json").read_text(encoding="utf-8"))}
    service = build_search_service(corpus, cache_dir=cache_dir)
    recalls, reciprocal_ranks = [], []
    for row in queries:
        ids = [result.doc_id for result in service.search(row["query"], mode=mode, limit=depth)]
        gold = labels[row["case_id"]]
        recalls.append(len(gold & set(ids[:10])) / len(gold))
        ranks = [index for index, doc_id in enumerate(ids[:20], start=1) if doc_id in gold]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    return {"recall_at_10": sum(recalls) / len(recalls), "mrr_at_20": sum(reciprocal_ranks) / len(reciprocal_ranks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-search-evaluate-demo")
    parser.add_argument("--mode", choices=("lex", "vec", "hybrid"), default="hybrid")
    parser.add_argument("--corpus", type=Path, default=ROOT / "demo_docs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".rag-cache")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.mode, corpus=args.corpus, cache_dir=args.cache_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
