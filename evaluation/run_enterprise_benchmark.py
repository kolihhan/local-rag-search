from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .evaluate import run_enterprise_benchmark, run_enterprise_core_benchmark
except ImportError:  # direct ``python evaluation/run_enterprise_benchmark.py``
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from evaluation.evaluate import run_enterprise_benchmark, run_enterprise_core_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="enterprise-rag-evaluate")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--question-set",
        choices=("metadata-extra", "core-confluence"),
        default="metadata-extra",
    )
    parser.add_argument(
        "--core-questions",
        type=Path,
        default=None,
        help="Official v1.0.0 questions.jsonl; required for core-confluence",
    )
    parser.add_argument("--output", "--output-path", dest="output", type=Path, required=True)
    parser.add_argument("--cache", "--cache-path", dest="cache", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", choices=("qwen3-embedding:0.6b",), default="qwen3-embedding:0.6b")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.question_set == "core-confluence":
        if args.core_questions is None:
            parser.error("--core-questions is required for --question-set core-confluence")
        run_enterprise_core_benchmark(
            dataset_root=args.dataset_root,
            core_questions_path=args.core_questions,
            output_path=args.output,
            cache_path=args.cache,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout,
            batch_size=args.batch_size,
        )
    else:
        run_enterprise_benchmark(
            dataset_root=args.dataset_root,
            output_path=args.output,
            cache_path=args.cache,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
