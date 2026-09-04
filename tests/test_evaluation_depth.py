from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]


def _load_eval_module():
    spec = spec_from_file_location("local_rag_eval", ROOT / "evaluation" / "evaluate.py")
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_uses_top10_for_recall_and_depth20_for_mrr(monkeypatch, tmp_path):
    module = _load_eval_module()
    labels = {
        row["case_id"]: row["relevant_document_ids"][0]
        for row in __import__("json").loads((ROOT / "evaluation" / "gold_labels.json").read_text(encoding="utf-8"))
    }

    class Service:
        def search(self, query, *, mode, limit):
            assert limit == 20
            case = next(
                row for row in __import__("json").loads((ROOT / "evaluation" / "inference_queries.json").read_text(encoding="utf-8"))
                if row["query"] == query
            )
            gold = labels[case["case_id"]]
            ids = [f"noise-{i}" for i in range(14)] + [gold] + [f"tail-{i}" for i in range(5)]
            return [SimpleNamespace(doc_id=doc_id) for doc_id in ids[:limit]]

    monkeypatch.setattr(module, "build_search_service", lambda corpus, cache_dir: Service())
    result = module.evaluate("hybrid", corpus=ROOT / "demo_docs", cache_dir=tmp_path)
    assert result["recall_at_10"] == 0.0
    assert result["mrr_at_20"] == 1 / 15
