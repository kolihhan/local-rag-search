from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate import load_core_confluence_set, load_enterprise_rag_txt_corpus
from local_rag.bm25 import BM25Index
from local_rag.dense import DenseIndex, OllamaEmbeddingProvider
from local_rag.fusion import reciprocal_rank_fusion

_VARIANTS = ((1.0, 1.0), (1.25, 1.0), (1.5, 1.0))


def _metrics(rankings, qrels):
    recall10 = []
    rr20 = []
    hit10 = []
    for qid, ranked in rankings.items():
        relevant = qrels[qid]
        positions = {doc_id: i for i, doc_id in enumerate(ranked, 1)}
        recall10.append(len(relevant & set(ranked[:10])) / len(relevant))
        first = min((positions[d] for d in relevant if d in positions and positions[d] <= 20), default=None)
        rr20.append(0.0 if first is None else 1.0 / first)
        hit10.append(bool(relevant & set(ranked[:10])))
    n = len(rankings)
    return {
        "recall_at_10": sum(recall10) / n,
        "mrr_at_20": sum(rr20) / n,
        "hit_at_10": sum(hit10) / n,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-1", type=Path, required=True)
    p.add_argument("--corpus-2", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    docs = load_enterprise_rag_txt_corpus(roots=[args.corpus_1, args.corpus_2])
    if len(docs) != 5189:
        raise RuntimeError(f"expected 5189 documents, got {len(docs)}")
    dataset = load_core_confluence_set(questions_path=args.questions, corpus_doc_ids={d.doc_id for d in docs})
    provider = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b", batch_size=16)
    bm25 = BM25Index(list(docs))
    dense = DenseIndex.build(list(docs), provider, cache_path=args.cache)

    bm25_60, dense_60 = {}, {}
    for qid in dataset.question_ids:
        q = dataset.questions[qid]
        bm25_60[qid] = [(h.doc_id, h.score) for h in bm25.search(q, limit=60)]
        dense_60[qid] = [(h.doc_id, h.score) for h in dense.search(q, limit=60)]

    bm25_top20 = {qid: [d for d, _ in rows[:20]] for qid, rows in bm25_60.items()}
    variants = {}
    for wb, wd in _VARIANTS:
        name = f"bm25_{wb:g}_dense_{wd:g}"
        rankings = {}
        for qid in dataset.question_ids:
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_60[qid], "dense": dense_60[qid]},
                k=60,
                weights={"bm25": wb, "dense": wd},
            )
            rankings[qid] = [row.doc_id for row in fused[:20]]
        rescued = [qid for qid in dataset.question_ids if not (dataset.qrels[qid] & set(bm25_top20[qid][:10])) and (dataset.qrels[qid] & set(rankings[qid][:10]))]
        regressed = [qid for qid in dataset.question_ids if (dataset.qrels[qid] & set(bm25_top20[qid][:10])) and not (dataset.qrels[qid] & set(rankings[qid][:10]))]
        variants[name] = {
            **_metrics(rankings, dataset.qrels),
            "bm25_miss_to_hit": rescued,
            "bm25_hit_to_miss": regressed,
        }

    payload = {
        "schema_version": "weighted-rrf-sensitivity/v1",
        "scope": "post-hoc fixed sensitivity analysis; not canonical benchmark selection",
        "documents": len(docs),
        "questions": len(dataset.question_ids),
        "provider_identity": provider.cache_identity,
        "bm25": _metrics(bm25_top20, dataset.qrels),
        "variants": variants,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
